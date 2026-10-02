"""Matched 40k-update REIN residual control/CQE experiment (Phase 16)."""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

from tools.audit_rein_pilot import digest, require
from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
    validate_baseline_audit,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions
from tools.rein_schedule_smoke import DATA_SHA, restore_protocol_config, scheduled_lr


SEED = 20260931
UPDATES = 40000
ACCUMULATION = 4
MICROBATCHES = UPDATES * ACCUMULATION
CLASSES = 19
LR = 1e-4
POLY_POWER = 0.9
WEIGHT_DECAY = 0.01
LAMBDA_CQE = 1.0
SAVE_INTERVAL = 1000
BASE_SEGMENTOR = "fixed_accepted_rein_source40k_seed0"


def objective(loss_original, loss_photometric, loss_cqe, *, enabled):
    segmentation = 0.5 * (loss_original + loss_photometric)
    return segmentation + (LAMBDA_CQE * loss_cqe if enabled else 0.0)


def load_branch_module(repository):
    path = repository / "causalq/models/rein_query_residual.py"
    spec = importlib.util.spec_from_file_location("formal_rein_query_residual", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(torch, numpy, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed % (2 ** 32))


def tensor_fingerprint(tensors):
    value = hashlib.sha256()
    for tensor in tensors:
        value.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return value.hexdigest()


def module_state_fingerprint(module):
    """Hash every fixed parameter and buffer in deterministic name order."""
    value = hashlib.sha256()
    state = list(module.named_parameters()) + list(module.named_buffers())
    for name, tensor in sorted(state, key=lambda pair: pair[0]):
        cpu = tensor.detach().contiguous().cpu()
        value.update(name.encode("utf-8"))
        value.update(str(cpu.dtype).encode("ascii"))
        value.update(json.dumps(list(cpu.shape)).encode("ascii"))
        value.update(cpu.numpy().tobytes())
    return value.hexdigest()


def save_checkpoint(torch, path, state, metadata):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(model={key: value.detach().cpu()
                                   for key, value in state.items()},
                            metadata=metadata), stream)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def target_evaluation(torch, np, model, branch, dataset, metric):
    from mmengine.structures import PixelData
    from tools.check_rein_slide_eval import confusion

    require(len(dataset) == 500, "Expected all 500 Cityscapes validation images")
    metric.dataset_meta = dataset.metainfo
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    validation_ids = []
    model.eval()
    branch.eval()
    for index in range(500):
        item = dataset[index]
        original_gt = item["data_samples"].gt_sem_seg.data[0].cpu().numpy().copy()
        original_shape = tuple(item["data_samples"].metainfo["ori_shape"])
        require(original_shape == (1024, 2048) and original_gt.shape == original_shape,
                "Cityscapes GT must retain original 1024x2048 geometry")
        with torch.no_grad():
            results = model.test_step(dict(inputs=[item["inputs"]],
                                           data_samples=[item["data_samples"]]))
            output = results[0]
            base_logits = output.seg_logits.data
            if base_logits.ndim == 3:
                base_logits = base_logits.unsqueeze(0)
            refined = branch(base_logits.to(device="cuda", dtype=torch.float32))
            logits = refined.logits[0].detach()
            if tuple(logits.shape) != (CLASSES, *original_shape):
                raise ValueError("Final residual logits have wrong target geometry")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite target logits")
            prediction = logits.argmax(dim=0).to(dtype=torch.long)
        output.seg_logits = PixelData(data=logits.cpu())
        output.pred_sem_seg = PixelData(data=prediction.unsqueeze(0).cpu())
        if not np.array_equal(output.gt_sem_seg.data[0].cpu().numpy(), original_gt):
            raise ValueError("Inference changed the Cityscapes GT")
        current = confusion(prediction.cpu().numpy(), original_gt)
        metric.process({}, [output.to_dict()])
        official = metric.results[-1]
        expected = (current.diagonal(), current.sum(0) + current.sum(1) - current.diagonal(),
                    current.sum(0), current.sum(1))
        if any(not np.array_equal(actual.cpu().numpy(), wanted)
               for actual, wanted in zip(official, expected)):
            raise ValueError("Official and independent Cityscapes counts differ")
        matrix += current
        validation_ids.append(dict(index=index, img_path=output.metainfo["img_path"],
                                   valid_pixels=int(current.sum()),
                                   confusion_matrix=current.tolist()))
        if (index + 1) % 25 == 0:
            print(f"target_eval={index + 1}/500", file=sys.stderr)
        del results, output, base_logits, refined, logits, prediction

    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    valid = union > 0
    iou = np.divide(matrix.diagonal(), union, out=np.zeros(CLASSES), where=valid)
    official_metrics = metric.compute_metrics(metric.results)
    independent_miou = float(iou[valid].mean())
    official_miou = float(official_metrics["mIoU"]) / 100.0
    if abs(independent_miou - official_miou) > 5.1e-5:
        raise ValueError("Official/independent mIoU mismatch")
    return dict(sample_count=500, samples=validation_ids,
                miou=independent_miou,
                official_metrics={key: float(value) for key, value in official_metrics.items()},
                class_iou=[float(iou[i]) if valid[i] else None for i in range(CLASSES)],
                valid_classes=int(valid.sum()), confusion_matrix=matrix.tolist(),
                total_valid_pixels=int(matrix.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rein-root", "weights", "data-report", "baseline-audit",
                 "baseline-checkpoint", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    training_seed = args.seed
    repository = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    run_created = False
    report = dict(ok=False, phase=16, stage="preflight",
                  purpose="matched_formal_frozen_rein_query_residual_cqe_comparison",
                  git_sha=git_output(repository, "rev-parse", "HEAD"),
                  seed=training_seed, base_segmentor=BASE_SEGMENTOR,
                  base_final_miou_reference=BASELINE_MIOU,
                  native_mask2former_queries_relabelled=False,
                  target_labels_optimized=False,
                  formal_training_authorized=True,
                  requested_optimizer_updates_per_arm=UPDATES,
                  microbatches_per_arm=MICROBATCHES,
                  accumulation=ACCUMULATION, lambda_cqe=LAMBDA_CQE,
                  optimizer=dict(name="AdamW", lr=LR, weight_decay=WEIGHT_DECAY,
                                 scheduler="polynomial", poly_power=POLY_POWER,
                                 schedule_horizon=UPDATES),
                  query=dict(classes=CLASSES, queries_per_class=2,
                             hidden_channels=64, temperature=0.07,
                             alpha_init=0.0, interaction="logit_residual"),
                  cqe=dict(control_enabled=False, candidate_enabled=True,
                           lambda_cqe=LAMBDA_CQE,
                           objective="two_view_segmentation_mean_plus_normalized_cqe"),
                  views=["original", "photometric"], geometry_preserved=True,
                  augmentation=dict(brightness=[0.7, 1.3], contrast=[0.7, 1.3],
                                   saturation=[0.7, 1.3], gamma=[0.8, 1.2],
                                   temperature=[0.9, 1.1], grayscale_probability=0.1),
                  target_dataset="Cityscapes_val", target_metric="19-class mIoU",
                  target_sample_count_per_arm=500,
                  checkpoint_selection="fixed_final_no_target_selection")
    try:
        require(0 <= training_seed < 2 ** 32, "Training seed must fit uint32")
        require(not args.run_dir.exists(), "Preserve existing run directory")
        require(args.run_dir.parent.is_dir(), "Output parent directory is missing")
        require(shutil.disk_usage(args.run_dir.parent).free >= 8 * 2**30,
                "At least 8 GiB free disk is required before formal run")
        for path, expected, label in (
                (args.weights, WEIGHT_SHA, "DINOv2 weights"),
                (args.data_report, DATA_SHA, "REIN data protocol"),
                (args.baseline_audit, BASELINE_AUDIT_SHA, "REIN baseline audit"),
                (args.baseline_checkpoint, BASELINE_CHECKPOINT_SHA, "REIN baseline checkpoint")):
            require(digest(path) == expected, label + " SHA256 mismatch")
        validate_baseline_audit(json.loads(args.baseline_audit.read_text()))
        require(git_output(args.rein_root, "rev-parse", "HEAD") == REIN_SHA
                and not git_output(args.rein_root, "status", "--porcelain"),
                "Pinned clean REIN checkout required")
        require(platform.python_version_tuple()[:2] == ("3", "10"),
                "Pinned Python 3.10 environment required")

        import numpy as np
        import torch
        from torch.nn import functional as F
        from tools.check_rein_slide_eval import restore_compact
        from tools.rein_training_state import SourceSampler
        from causalq.interventions.style import StyleInterventionBank
        from causalq.losses.causal_query_effect import causal_query_effect_loss

        versions = {name: str(importlib.import_module(name).__version__)
                    for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        require(torch.version.cuda == "11.8" and torch.cuda.is_available(),
                "Pinned CUDA 11.8 runtime and GPU required")
        require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090 D",
                "Unexpected GPU")
        report.update(versions=versions, python=platform.python_version(),
                      gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                      baseline_audit_sha256=BASELINE_AUDIT_SHA,
                      baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
                      baseline_training_git_sha=BASELINE_PRODUCER, dtype="float32")

        sys.path.insert(0, str(args.rein_root.resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.registry import init_default_scope
            from mmseg.registry import DATASETS, METRICS, MODELS
            from tools.rein_protocol_adapter import register_data_transforms

            init_default_scope("mmseg")
            register_data_transforms()
            config = restore_protocol_config(json.loads(args.data_report.read_text()), args.rein_root)
            model = MODELS.build(config.model)
            model.decode_head.init_weights()
            pretrained = torch.load(args.weights, map_location="cpu", weights_only=True)
            base_keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
            loaded = model.backbone.load_state_dict(pretrained, strict=False)
            validate_load_keys(pretrained, base_keys, loaded.missing_keys, loaded.unexpected_keys)
            del pretrained
            model.train(True)
            baseline = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True)
            require(baseline["metadata"].get("git_sha") == BASELINE_PRODUCER
                    and baseline["metadata"].get("optimizer_updates") == 40000,
                    "Accepted source-only checkpoint metadata mismatch")
            restore_compact(model, baseline["model"])
            del baseline
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.cuda().eval()
            frozen_state_hash = module_state_fingerprint(model)
            report["fixed_base_state_sha256"] = frozen_state_hash

        train_dataset = DATASETS.build(config.train_dataloader.dataset)
        val_dataset = DATASETS.build(config.val_dataloader.dataset)
        require(len(train_dataset) == 24966 and len(val_dataset) == 500,
                "GTA5/Cityscapes dataset coverage mismatch")
        means = (model.data_preprocessor.mean.detach().float().flatten().cpu() / 255.0).tolist()
        stds = (model.data_preprocessor.std.detach().float().flatten().cpu() / 255.0).tolist()
        style = StyleInterventionBank(mean=tuple(means), std=tuple(stds)).cuda().eval()
        branch_module = load_branch_module(repository)
        sampler = SourceSampler(len(train_dataset), seed=training_seed)
        indices = sampler.take(MICROBATCHES)
        require(len(indices) == MICROBATCHES, "Source schedule length mismatch")
        report["source_schedule"] = dict(microbatches=MICROBATCHES,
                                         sampler_seed=training_seed,
                                         complete_epochs=MICROBATCHES // len(train_dataset),
                                         partial_epoch_samples=MICROBATCHES % len(train_dataset),
                                         indices_sha256=hashlib.sha256(
                                             json.dumps(indices).encode()).hexdigest())
        args.run_dir.mkdir(parents=True)
        run_created = True
        (args.run_dir / "metadata.json").write_text(json.dumps(report, indent=2))
        paired_input_hashes = {"no_cqe": hashlib.sha256(), "cqe": hashlib.sha256()}
        initial_fingerprint = None
        arm_results = {}
        for arm_name, use_cqe in (("no_cqe", False), ("cqe", True)):
            report["stage"] = "training_" + arm_name
            arm_dir = args.run_dir / arm_name
            arm_dir.mkdir()
            seed_everything(torch, np, (training_seed + 100) % (2 ** 32))
            branch = branch_module.ReinClassQueryResidual(
                num_classes=CLASSES, queries_per_class=2, hidden_channels=64,
                temperature=0.07, alpha_init=0.0).cuda().train()
            require(sum(parameter.numel() for parameter in branch.parameters()) == 3713,
                    "Unexpected residual branch parameter count")
            init_digest = hashlib.sha256()
            for name, value in sorted(branch.state_dict().items()):
                init_digest.update(name.encode("utf-8"))
                init_digest.update(value.detach().contiguous().cpu().numpy().tobytes())
            init_hash = init_digest.hexdigest()
            if initial_fingerprint is None:
                initial_fingerprint = init_hash
            else:
                require(init_hash == initial_fingerprint,
                        "Matched initial branch parameter hashes differ")
            optimizer = torch.optim.AdamW(branch.parameters(), lr=LR,
                                          weight_decay=WEIGHT_DECAY)
            trace_path = arm_dir / "train.jsonl"
            trace_hash = hashlib.sha256()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(0)
            first_losses, last_losses = [], []
            gradient_names = set()
            update_losses, update_seg_losses, update_cqe_losses = [], [], []
            with trace_path.open("x", encoding="utf-8") as trace:
                for microbatch, index in enumerate(indices, 1):
                    step_seed = (training_seed * 1000003 + microbatch) % (2 ** 32)
                    seed_everything(torch, np, step_seed)
                    item = train_dataset[index]
                    batch = model.data_preprocessor(
                        dict(inputs=[item["inputs"]],
                             data_samples=[item["data_samples"]]), training=True)
                    target = torch.stack([sample.gt_sem_seg.data[0]
                                          for sample in batch["data_samples"]]).long()
                    require(bool((target != 255).any().item()),
                            "Training crop contains no valid labels")
                    images = batch["inputs"]
                    photo = style.photometric_view(images)
                    fingerprint = tensor_fingerprint((images, target, photo))
                    paired_input_hashes[arm_name].update(fingerprint.encode("ascii"))
                    meta = [dict(ori_shape=(512, 512), img_shape=(512, 512),
                                 pad_shape=(512, 512))]
                    with torch.no_grad():
                        base_original = model.encode_decode(images, meta)
                        base_photo = model.encode_decode(photo, meta)
                    if (tuple(base_original.shape) != (1, CLASSES, 512, 512)
                            or not torch.isfinite(base_original).all().item()
                            or not torch.isfinite(base_photo).all().item()):
                        raise ValueError("Invalid fixed REIN logits")
                    out_original, out_photo = branch(base_original), branch(base_photo)
                    loss_original = F.cross_entropy(out_original.logits, target,
                                                    ignore_index=255)
                    loss_photo = F.cross_entropy(out_photo.logits, target,
                                                ignore_index=255)
                    loss_cqe = causal_query_effect_loss(
                        out_photo.alpha * out_photo.query_residual,
                        out_original.alpha * out_original.query_residual,
                        target, ignore_index=255)
                    loss_seg = 0.5 * (loss_original + loss_photo)
                    loss = objective(loss_original, loss_photo, loss_cqe,
                                     enabled=use_cqe)
                    scalars = torch.stack((loss_original, loss_photo, loss_cqe,
                                           loss_seg, loss))
                    if not torch.isfinite(scalars).all().item():
                        raise FloatingPointError("Nonfinite training objective")
                    (loss / ACCUMULATION).backward()
                    update = microbatch % ACCUMULATION == 0
                    grad_norm = None
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    if update:
                        grad = torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
                        if not torch.isfinite(grad).item():
                            raise FloatingPointError("Nonfinite branch gradient norm")
                        grad_norm = float(grad.item())
                        for name, parameter in branch.named_parameters():
                            if parameter.grad is not None and parameter.grad.abs().max().item() > 0:
                                gradient_names.add(name)
                        optimizer.step()
                        update_index = microbatch // ACCUMULATION
                        next_lr = scheduled_lr(update_index)
                        for group in optimizer.param_groups:
                            group["lr"] = next_lr
                        optimizer.zero_grad(set_to_none=True)
                        update_losses.append(float(loss.item()))
                        update_seg_losses.append(float(loss_seg.item()))
                        update_cqe_losses.append(float(loss_cqe.item()))
                        if len(update_losses) <= 100:
                            first_losses.append(float(loss.item()))
                        if update_index > UPDATES - 100:
                            last_losses.append(float(loss.item()))
                    record = dict(microbatch=microbatch, update=microbatch // ACCUMULATION,
                                  dataset_index=index, augmentation_seed=step_seed,
                                  loss=float(loss.item()), loss_seg=float(loss_seg.item()),
                                  loss_original=float(loss_original.item()),
                                  loss_photometric=float(loss_photo.item()),
                                  loss_cqe=float(loss_cqe.item()),
                                  optimizer_update=update, gradient_norm=grad_norm,
                                  lr=current_lr,
                                  alpha=float(branch.alpha.item()), input_sha256=fingerprint)
                    encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                    trace.write(encoded.decode("utf-8"))
                    trace_hash.update(encoded)
                    if update and microbatch % (SAVE_INTERVAL * ACCUMULATION) == 0:
                        save_checkpoint(torch, arm_dir / "latest.pth",
                                        branch.state_dict(),
                                        dict(arm=arm_name, optimizer_updates=update_index,
                                             git_sha=report["git_sha"], cqe_enabled=use_cqe))
                    if update and update_index % 100 == 0:
                        trace.flush()
                        print(json.dumps(dict(arm=arm_name, **record)), file=sys.stderr)
                    del out_original, out_photo, base_original, base_photo
                    del photo, images, target, loss, loss_cqe, loss_seg, scalars
            final_checkpoint = arm_dir / "final_branch.pth"
            save_checkpoint(torch, final_checkpoint, branch.state_dict(),
                            dict(arm=arm_name, optimizer_updates=UPDATES,
                                 git_sha=report["git_sha"], cqe_enabled=use_cqe,
                                 seed=training_seed, checkpoint_selection="fixed_final"))
            restored = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
            require(all(torch.equal(restored["model"][key], value.detach().cpu())
                        for key, value in branch.state_dict().items()),
                    "Final branch checkpoint roundtrip mismatch")
            require({"alpha", "query_bank", "pixel_projection.weight"}.issubset(gradient_names),
                    "Required branch gradients were not observed")
            require(branch.alpha.item() != 0, "Residual scale failed to open")
            require(module_state_fingerprint(model) == frozen_state_hash,
                    "Frozen REIN full state changed during training")

            val_metric = METRICS.build(config.val_evaluator)
            train_peak_allocated = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
            train_peak_reserved = round(torch.cuda.max_memory_reserved(0) / 2**30, 3)
            report["stage"] = "evaluation_" + arm_name
            torch.cuda.reset_peak_memory_stats(0)
            validation = target_evaluation(torch, np, model, branch, val_dataset, val_metric)
            eval_peak_allocated = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
            eval_peak_reserved = round(torch.cuda.max_memory_reserved(0) / 2**30, 3)
            require(train_peak_reserved >= train_peak_allocated
                    and eval_peak_reserved >= eval_peak_allocated,
                    "CUDA memory accounting mismatch")
            arm_results[arm_name] = dict(
                cqe_enabled=use_cqe, optimizer_updates=UPDATES,
                microbatch_count=MICROBATCHES, accumulation=ACCUMULATION,
                initial_branch_sha256=init_hash,
                final_alpha=float(branch.alpha.item()),
                first_100_update_loss_mean=sum(first_losses) / len(first_losses),
                last_100_update_loss_mean=sum(last_losses) / len(last_losses),
                first_100_update_seg_loss_mean=sum(update_seg_losses[:100]) / 100,
                last_100_update_seg_loss_mean=sum(update_seg_losses[-100:]) / 100,
                first_100_update_cqe_loss_mean=sum(update_cqe_losses[:100]) / 100,
                last_100_update_cqe_loss_mean=sum(update_cqe_losses[-100:]) / 100,
                nonzero_gradient_parameters=sorted(gradient_names),
                trace=str(trace_path), trace_sha256=trace_hash.hexdigest(),
                trace_record_count=MICROBATCHES,
                checkpoint=str(final_checkpoint), checkpoint_sha256=digest(final_checkpoint),
                checkpoint_bytes=final_checkpoint.stat().st_size,
                checkpoint_roundtrip_ok=True, validation=validation,
                train_peak_allocated_gib=train_peak_allocated,
                train_peak_reserved_gib=train_peak_reserved,
                eval_peak_allocated_gib=eval_peak_allocated,
                eval_peak_reserved_gib=eval_peak_reserved)
            torch.cuda.empty_cache()
            print(json.dumps(dict(stage="arm_complete", arm=arm_name,
                                  miou=validation["miou"],
                                  checkpoint_sha256=arm_results[arm_name]["checkpoint_sha256"])),
                  file=sys.stderr)
            del branch, optimizer, val_metric

        no_hash, cqe_hash = (paired_input_hashes["no_cqe"].hexdigest(),
                             paired_input_hashes["cqe"].hexdigest())
        require(no_hash == cqe_hash, "Paired per-step input hashes differ")
        require(arm_results["no_cqe"]["initial_branch_sha256"]
                == arm_results["cqe"]["initial_branch_sha256"],
                "Paired initial branch state differs")
        require(module_state_fingerprint(model) == frozen_state_hash,
                "Frozen REIN full state changed after both arms")
        require(not any(parameter.grad is not None for parameter in model.parameters()),
                "Frozen REIN base accumulated gradients")
        report.update(ok=True, stage="complete", arms=arm_results,
                      initial_branch_sha256=initial_fingerprint,
                      paired_input_hash_sha256=no_hash,
                      paired_inputs_identical=True, base_parameters_unchanged=True,
                      target_labels_optimized=False,
                      target_sample_count_per_arm=500,
                      final_miou_difference_cqe_minus_control=(
                          arm_results["cqe"]["validation"]["miou"]
                          - arm_results["no_cqe"]["validation"]["miou"]),
                      elapsed_seconds=round(time.monotonic() - started, 2))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
    if run_created:
        (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
