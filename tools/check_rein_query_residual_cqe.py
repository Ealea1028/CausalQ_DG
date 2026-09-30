"""Matched bounded REIN query-residual smoke with and without CQE."""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
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
    SMOKE_SEED,
    validate_baseline_audit,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions
from tools.rein_schedule_smoke import DATA_SHA, restore_protocol_config


MICROBATCHES = 80
ACCUMULATION = 4
UPDATES = MICROBATCHES // ACCUMULATION
LAMBDA_CQE = 1.0
PAIR_SEED = SMOKE_SEED + 1


def matched_objective(loss_original, loss_photometric, loss_cqe, *, enabled,
                      lambda_cqe=LAMBDA_CQE):
    """Average supervised views and add only the existing CQE term."""
    loss_seg = 0.5 * (loss_original + loss_photometric)
    return loss_seg + (lambda_cqe * loss_cqe if enabled else 0.0)


def _module(repository: Path):
    path = repository / "causalq/models/rein_query_residual.py"
    spec = importlib.util.spec_from_file_location("causalq_rein_query_residual_cqe", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _paired_fingerprint(images, target, style):
    import hashlib

    value = hashlib.sha256()
    for tensor in (images, target, style):
        value.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return value.hexdigest()


def _seed(torch, np, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rein-root", "weights", "data-report", "baseline-audit",
                 "baseline-checkpoint", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = dict(
        ok=False, phase=16, stage="preflight",
        purpose="matched_frozen_rein_query_residual_cqe_smoke_not_accuracy",
        git_sha=git_output(repository, "rev-parse", "HEAD"),
        seed=PAIR_SEED, base_segmentor="fixed_accepted_rein_source40k_seed0",
        base_final_miou_reference=BASELINE_MIOU,
        native_mask2former_queries_relabelled=False,
        query=dict(classes=19, queries_per_class=2, hidden_channels=64,
                   temperature=0.07, alpha_init=0.0, interaction="logit_residual"),
        views=["original", "photometric"], geometry_preserved=True,
        augmentation=dict(brightness=[0.7, 1.3], contrast=[0.7, 1.3],
                          saturation=[0.7, 1.3], gamma=[0.8, 1.2],
                          temperature=[0.9, 1.1], grayscale_probability=0.1),
        optimizer=dict(name="AdamW", learning_rate=1e-4,
                       weight_decay=0.01, schedule="constant_smoke_lr"),
        cqe=dict(control_enabled=False, candidate_enabled=True, lambda_cqe=LAMBDA_CQE,
                 objective="segmentation_mean_plus_existing_cqe"),
        target_labels_optimized=False, formal_training_authorized=False,
        requested_optimizer_updates=UPDATES, accumulation=ACCUMULATION,
        microbatches=MICROBATCHES,
    )
    created = False
    try:
        if args.run_dir.exists():
            raise FileExistsError("Preserve existing run")
        require(digest(args.weights) == WEIGHT_SHA, "Backbone checkpoint hash mismatch")
        require(digest(args.data_report) == DATA_SHA, "Data protocol hash mismatch")
        require(digest(args.baseline_audit) == BASELINE_AUDIT_SHA,
                "Baseline audit hash mismatch")
        require(digest(args.baseline_checkpoint) == BASELINE_CHECKPOINT_SHA,
                "Baseline checkpoint hash mismatch")
        validate_baseline_audit(json.loads(args.baseline_audit.read_text()))
        require(git_output(args.rein_root, "rev-parse", "HEAD") == REIN_SHA,
                "Pinned REIN source required")
        require(not git_output(args.rein_root, "status", "--porcelain"),
                "Clean REIN source required")
        require(platform.python_version_tuple()[:2] == ("3", "10"),
                "Isolated Python 3.10 required")
        require(shutil.disk_usage(args.run_dir.parent).free >= 2 * 2**30,
                "Need at least 2 GiB free")

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
                "Accepted CUDA 11.8 GPU required")
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
            from mmseg.registry import DATASETS, MODELS
            from tools.rein_protocol_adapter import register_data_transforms

            init_default_scope("mmseg")
            register_data_transforms()
            config = restore_protocol_config(
                json.loads(args.data_report.read_text()), args.rein_root)
            model = MODELS.build(config.model)
            model.decode_head.init_weights()
            # REIN enables the compactly saved adapters/head via its train hook.
            model.train(True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad != name.startswith(
                        ("backbone.reins.", "decode_head.")):
                    raise ValueError("Unexpected REIN trainability: " + name)
            pretrained = torch.load(args.weights, map_location="cpu", weights_only=True)
            keys = (set(dict(model.backbone.named_parameters()))
                    | set(dict(model.backbone.named_buffers())))
            loaded = model.backbone.load_state_dict(pretrained, strict=False)
            validate_load_keys(pretrained, keys, loaded.missing_keys,
                               loaded.unexpected_keys)
            del pretrained
            baseline = torch.load(args.baseline_checkpoint, map_location="cpu",
                                  weights_only=True)
            metadata = baseline.get("metadata", {})
            require(metadata.get("git_sha") == BASELINE_PRODUCER
                    and metadata.get("optimizer_updates") == 40000,
                    "Baseline checkpoint metadata mismatch")
            restore_compact(model, baseline["model"])
            del baseline
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.cuda().eval()
            frozen_probe = next(model.parameters()).detach().clone()

        dataset = DATASETS.build(config.train_dataloader.dataset)
        require(len(dataset) == 24966, "Source coverage changed")
        preprocessor = model.data_preprocessor
        means = (preprocessor.mean.detach().float().flatten().cpu() / 255.0).tolist()
        stds = (preprocessor.std.detach().float().flatten().cpu() / 255.0).tolist()
        style = StyleInterventionBank(mean=tuple(means), std=tuple(stds)).cuda().eval()
        branch_module = _module(repository)
        torch.cuda.reset_peak_memory_stats(0)
        args.run_dir.mkdir()
        created = True
        report["stage"] = "matched_training"
        (args.run_dir / "metadata.json").write_text(json.dumps(report, indent=2))

        arms = {}
        fingerprints = {}
        initial_branch_fingerprint = None
        for arm_name, use_cqe in (("no_cqe", False), ("cqe", True)):
            _seed(torch, np, PAIR_SEED + 100)
            branch = branch_module.ReinClassQueryResidual(
                num_classes=19, queries_per_class=2, hidden_channels=64,
                temperature=0.07, alpha_init=0.0).cuda().train()
            require(sum(parameter.numel() for parameter in branch.parameters()) == 3713,
                    "Unexpected branch parameter count")
            initial_state = {key: value.detach().clone()
                             for key, value in branch.state_dict().items()}
            state_digest = hashlib.sha256()
            for name, value in sorted(initial_state.items()):
                state_digest.update(name.encode("utf-8"))
                state_digest.update(value.cpu().contiguous().numpy().tobytes())
            current_initial_fingerprint = state_digest.hexdigest()
            if initial_branch_fingerprint is None:
                initial_branch_fingerprint = current_initial_fingerprint
            else:
                require(current_initial_fingerprint == initial_branch_fingerprint,
                        "Matched branch initializations differ")
            optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-4,
                                          weight_decay=0.01)
            sampler = SourceSampler(len(dataset), seed=PAIR_SEED)
            _seed(torch, np, PAIR_SEED)
            records = []
            indices = []
            grad_names = set()
            reconstruction_error = 0.0
            optimizer.zero_grad(set_to_none=True)
            with (args.run_dir / f"{arm_name}.jsonl").open("x") as trace:
                for iteration in range(1, MICROBATCHES + 1):
                    index = sampler.take(1)[0]
                    indices.append(index)
                    item = dataset[index]
                    batch = model.data_preprocessor(
                        dict(inputs=[item["inputs"]],
                             data_samples=[item["data_samples"]]), training=True)
                    target = torch.stack([
                        sample.gt_sem_seg.data[0] for sample in batch["data_samples"]
                    ]).long()
                    if not (target != 255).any().item():
                        raise ValueError("Source crop has no valid pixels")
                    images = batch["inputs"]
                    photo = style.photometric_view(images)
                    fingerprints.setdefault(iteration, {})[arm_name] = \
                        _paired_fingerprint(images, target, photo)
                    meta = [dict(ori_shape=(512, 512), img_shape=(512, 512),
                                 pad_shape=(512, 512))]
                    with torch.no_grad():
                        base_original = model.encode_decode(images, meta)
                        base_photo = model.encode_decode(photo, meta)
                    if (base_original.shape != (1, 19, 512, 512)
                            or not torch.isfinite(base_original).all().item()
                            or not torch.isfinite(base_photo).all().item()):
                        raise ValueError("Invalid frozen base logits")
                    out_original = branch(base_original)
                    out_photo = branch(base_photo)
                    loss_original = F.cross_entropy(
                        out_original.logits, target, ignore_index=255)
                    loss_photo = F.cross_entropy(
                        out_photo.logits, target, ignore_index=255)
                    loss_cqe = causal_query_effect_loss(
                        out_photo.alpha * out_photo.query_residual,
                        out_original.alpha * out_original.query_residual,
                        target, ignore_index=255)
                    loss_seg = 0.5 * (loss_original + loss_photo)
                    loss = matched_objective(loss_original, loss_photo, loss_cqe,
                                             enabled=use_cqe)
                    if not torch.isfinite(torch.stack((loss_original, loss_photo,
                                                       loss_cqe, loss_seg, loss))).all().item():
                        raise FloatingPointError("Nonfinite matched objective")
                    reconstructed = loss_seg + (LAMBDA_CQE * loss_cqe
                                                  if use_cqe else 0.0)
                    reconstruction_error = max(
                        reconstruction_error, abs(loss.item() - reconstructed.item()))
                    (loss / ACCUMULATION).backward()
                    squared_norm = 0.0
                    for name, parameter in branch.named_parameters():
                        if parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all().item():
                                raise FloatingPointError("Nonfinite branch gradient")
                            value = parameter.grad.float().norm().item()
                            squared_norm += value * value
                            if value > 0:
                                grad_names.add(name)
                    if any(parameter.grad is not None for parameter in model.parameters()):
                        raise ValueError("Frozen REIN base received gradients")
                    update = iteration % ACCUMULATION == 0
                    if update:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    record = dict(iteration=iteration, dataset_index=index,
                                  loss=loss.item(), loss_seg=loss_seg.item(),
                                  loss_original=loss_original.item(),
                                  loss_photometric=loss_photo.item(),
                                  loss_cqe=loss_cqe.item(),
                                  gradient_norm=squared_norm ** 0.5,
                                  optimizer_update=update, alpha=branch.alpha.item())
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    print(json.dumps(dict(arm=arm_name, **record)), file=sys.stderr)
                    records.append(record)
                    del out_original, out_photo, base_original, base_photo
                    del photo, images, target, loss, loss_cqe, loss_seg

            require(sampler.cursor == MICROBATCHES, "Source sampler cursor mismatch")
            require(len(set(indices)) == MICROBATCHES,
                    "Source prefix repeated an index")
            require(len(records) == MICROBATCHES
                    and sum(row["optimizer_update"] for row in records) == UPDATES,
                    "Microbatch/update count mismatch")
            require(grad_names >= {"alpha", "query_bank", "pixel_projection.weight"},
                    "Incomplete branch gradient flow")
            require(branch.alpha.item() != 0, "Residual scale did not open")
            require(all(record["loss_cqe"] >= 0 for record in records),
                    "Invalid CQE loss values")
            require(reconstruction_error <= 1e-7,
                    "Objective reconstruction mismatch")
            saved = {key: value.detach().cpu() for key, value in branch.state_dict().items()}
            checkpoint = args.run_dir / f"{arm_name}_branch.pth"
            with checkpoint.open("xb") as stream:
                torch.save(dict(model=saved, metadata=dict(arm=arm_name,
                           cqe_enabled=use_cqe, updates=UPDATES)), stream)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(all(torch.equal(restored["model"][key], value)
                        for key, value in saved.items()), "Branch roundtrip mismatch")
            arms[arm_name] = dict(
                cqe_enabled=use_cqe, optimizer_updates=UPDATES,
                record_count=len(records), unique_source_indices=len(set(indices)),
                objective_reconstruction_error=reconstruction_error,
                first_20_loss_mean=sum(
                    row["loss"] for row in records[:20]) / 20,
                last_20_loss_mean=sum(row["loss"] for row in records[-20:]) / 20,
                first_20_cqe_mean=sum(row["loss_cqe"] for row in records[:20]) / 20,
                last_20_cqe_mean=sum(row["loss_cqe"] for row in records[-20:]) / 20,
                final_alpha=branch.alpha.item(),
                nonzero_gradient_parameters=sorted(grad_names),
                checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                checkpoint_bytes=checkpoint.stat().st_size,
                branch_changed_from_init=any(not torch.equal(
                    initial_state[key], branch.state_dict()[key])
                    for key in initial_state),
            )
            del branch, optimizer
            torch.cuda.empty_cache()

        require(all(pair["no_cqe"] == pair["cqe"]
                    for pair in fingerprints.values()),
                "Matched data or style inputs differ between arms")
        for rows in fingerprints.values():
            require(set(rows) == {"no_cqe", "cqe"}, "Missing paired arm")
        require(torch.equal(frozen_probe, next(model.parameters()).detach()),
                "Frozen REIN base parameters changed")
        report.update(ok=True, stage="complete", arms=arms,
                      matched_input_count=len(fingerprints),
                      matched_inputs_identical=True,
                      initial_branch_fingerprint=initial_branch_fingerprint,
                      base_parameters_unchanged=True,
                      all_losses_finite=True, target_images_used=0,
                      elapsed_seconds=round(time.monotonic() - started, 2),
                      peak_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                      peak_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 2**30, 3))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
    if created:
        (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
