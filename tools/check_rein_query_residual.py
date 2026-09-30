"""Bounded real-data gate for a class-specific residual over frozen REIN logits."""

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions
from tools.rein_schedule_smoke import DATA_SHA, digest, restore_protocol_config


BASELINE_AUDIT_SHA = "b1a8e15345ca72dac5e9f36550dc5dadb5e4b07423e8bb56d8500fb5fde719d1"
BASELINE_CHECKPOINT_SHA = "84231e98dda68ac4b1fd4f59cc97881887a614de01b01e2697323c1eac80daf2"
BASELINE_PRODUCER = "d6fc52c5af9dcf0d6f57218b8b448c7df7a74ea4"
BASELINE_MIOU = 0.6560509975136082
SMOKE_SEED = 20260930


def smoke_budget() -> dict[str, int]:
    """Keep this gate bounded and distinct from formal training."""
    return {"optimizer_updates": 20, "accumulation": 4, "microbatches": 80}


def validate_baseline_audit(report: dict) -> None:
    expected = {
        "source40k_saved_audit_ok": True,
        "training_git_sha": BASELINE_PRODUCER,
        "optimizer_updates": 40000,
        "microbatches": 160000,
        "final_miou": BASELINE_MIOU,
        "final_checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "cqe_enabled": False,
        "target_labels_optimized": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Accepted source-only baseline audit required")


def load_query_module(repository: Path):
    """Load the bridge without importing optional DINOv3 dependencies."""
    name = "causalq_rein_query_residual_gate"
    path = repository / "causalq/models/rein_query_residual.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rein-root", "weights", "data-report", "baseline-audit",
                 "baseline-checkpoint", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    budget = smoke_budget()
    repository = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = dict(
        ok=False,
        phase=16,
        stage="preflight",
        purpose="frozen_rein_class_query_residual_smoke_not_accuracy_or_cqe",
        git_sha=git_output(repository, "rev-parse", "HEAD"),
        seed=SMOKE_SEED,
        base_segmentor="fixed_accepted_rein_source40k_seed0",
        base_final_miou_reference=BASELINE_MIOU,
        native_mask2former_queries_relabelled=False,
        query=dict(classes=19, queries_per_class=2, hidden_channels=64,
                   temperature=0.07, alpha_init=0.0, interaction="logit_residual"),
        cqe_enabled=False,
        target_labels_optimized=False,
        formal_training_authorized=False,
        requested_optimizer_updates=budget["optimizer_updates"],
        optimizer_updates=0,
        accumulation=budget["accumulation"],
        microbatches=budget["microbatches"],
    )
    created_run = False
    try:
        if args.run_dir.exists():
            raise FileExistsError("Preserve existing run")
        for path, expected in ((args.weights, WEIGHT_SHA),
                               (args.data_report, DATA_SHA),
                               (args.baseline_audit, BASELINE_AUDIT_SHA),
                               (args.baseline_checkpoint, BASELINE_CHECKPOINT_SHA)):
            if digest(path) != expected:
                raise ValueError("Input hash mismatch: " + str(path))
        validate_baseline_audit(json.loads(args.baseline_audit.read_text()))
        if (git_output(args.rein_root, "rev-parse", "HEAD") != REIN_SHA
                or git_output(args.rein_root, "status", "--porcelain")):
            raise ValueError("Pinned clean REIN source required")
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise ValueError("Isolated Python 3.10 required")
        if shutil.disk_usage(args.run_dir.parent).free < 2 * 2**30:
            raise OSError("Need at least 2 GiB free")

        import numpy as np
        import torch
        from torch.nn import functional as F
        from tools.check_rein_slide_eval import restore_compact
        from tools.rein_training_state import SourceSampler, cpu_tree

        versions = {name: str(importlib.import_module(name).__version__)
                    for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        if torch.version.cuda != "11.8" or not torch.cuda.is_available():
            raise RuntimeError("Accepted CUDA 11.8 GPU required")
        report.update(versions=versions, python=platform.python_version(),
                      gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                      baseline_audit_sha256=BASELINE_AUDIT_SHA,
                      baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
                      baseline_training_git_sha=BASELINE_PRODUCER, dtype="float32")

        torch.manual_seed(SMOKE_SEED)
        torch.cuda.manual_seed_all(SMOKE_SEED)
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
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
            pretrained = torch.load(args.weights, map_location="cpu", weights_only=True)
            keys = (set(dict(model.backbone.named_parameters()))
                    | set(dict(model.backbone.named_buffers())))
            loaded = model.backbone.load_state_dict(pretrained, strict=False)
            validate_load_keys(pretrained, keys, loaded.missing_keys,
                               loaded.unexpected_keys)
            del pretrained
            model.train(True)
            baseline = torch.load(args.baseline_checkpoint, map_location="cpu",
                                  weights_only=True)
            metadata = baseline.get("metadata", {})
            if (metadata.get("git_sha") != BASELINE_PRODUCER
                    or metadata.get("optimizer_updates") != 40000):
                raise ValueError("Baseline checkpoint metadata mismatch")
            restore_compact(model, baseline["model"])
            del baseline
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.cuda().eval()

        query_module = load_query_module(repository)
        torch.manual_seed(SMOKE_SEED)
        branch = query_module.ReinClassQueryResidual(
            num_classes=19, queries_per_class=2, hidden_channels=64,
            temperature=0.07, alpha_init=0.0).cuda().train()
        optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-4,
                                      weight_decay=0.01)
        report["branch_trainable_parameters"] = sum(
            parameter.numel() for parameter in branch.parameters())
        if report["branch_trainable_parameters"] != 3713:
            raise ValueError("Unexpected branch parameter count")

        dataset = DATASETS.build(config.train_dataloader.dataset)
        if len(dataset) != 24966:
            raise ValueError("Source coverage changed")
        sampler = SourceSampler(len(dataset), seed=SMOKE_SEED)
        args.run_dir.mkdir()
        created_run = True
        (args.run_dir / "metadata.json").write_text(json.dumps(report, indent=2))
        report["stage"] = "branch_optimization"
        frozen_probe = model.backbone.patch_embed.proj.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(0)
        losses = []
        gradient_names = set()
        last_output = last_target = None
        with (args.run_dir / "train.jsonl").open("x") as trace:
            for iteration in range(1, budget["microbatches"] + 1):
                index = sampler.take(1)[0]
                item = dataset[index]
                batch = model.data_preprocessor(
                    dict(inputs=[item["inputs"]],
                         data_samples=[item["data_samples"]]), training=True)
                target = torch.stack([
                    sample.gt_sem_seg.data[0] for sample in batch["data_samples"]
                ]).long()
                if not (target != 255).any().item():
                    raise ValueError("Source crop has no valid pixels")
                meta = [dict(ori_shape=(512, 512), img_shape=(512, 512),
                             pad_shape=(512, 512))]
                with torch.no_grad():
                    base_logits = model.encode_decode(batch["inputs"], meta)
                if (tuple(base_logits.shape) != (1, 19, 512, 512)
                        or not torch.isfinite(base_logits).all().item()):
                    raise ValueError("Invalid frozen base logits")
                output = branch(base_logits)
                loss = F.cross_entropy(output.logits, target, ignore_index=255)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite branch loss")
                (loss / budget["accumulation"]).backward()
                squared_norm = 0.0
                for name, parameter in branch.named_parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all().item():
                            raise FloatingPointError("Nonfinite branch gradient: " + name)
                        value = parameter.grad.float().norm().item()
                        squared_norm += value * value
                        if value > 0:
                            gradient_names.add(name)
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise ValueError("Frozen base received a gradient")
                update = iteration % budget["accumulation"] == 0
                if update:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    report["optimizer_updates"] = iteration // budget["accumulation"]
                record = dict(iteration=iteration, dataset_index=index,
                              loss=loss.item(), gradient_norm=math.sqrt(squared_norm),
                              optimizer_update=update, alpha=branch.alpha.item())
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                print(json.dumps(record), file=sys.stderr)
                losses.append(record["loss"])
                last_output, last_target = output, target

        if (report["optimizer_updates"] != budget["optimizer_updates"]
                or not torch.equal(frozen_probe,
                                   model.backbone.patch_embed.proj.weight)):
            raise ValueError("Update count or frozen base changed")
        required_gradients = {"alpha", "query_bank", "pixel_projection.weight"}
        if not required_gradients.issubset(gradient_names):
            raise ValueError("Incomplete branch gradient flow")
        if branch.alpha.item() == 0:
            raise ValueError("Residual scale did not open")

        present = torch.unique(last_target[last_target != 255]).tolist()
        if not present:
            raise ValueError("No class available for intervention gate")
        class_id = int(present[0])
        factual = last_output.logits
        counterfactual = last_output.counterfactual_logits(class_id)
        difference = factual - counterfactual
        outside = torch.cat((difference[:, :class_id], difference[:, class_id + 1:]), dim=1)
        outside_error = outside.abs().max().item() if outside.numel() else 0.0
        selected_error = (difference[:, class_id]
                          - last_output.class_effect(class_id)).abs().max().item()
        if outside_error != 0 or selected_error > 1e-5:
            raise ValueError("Class intervention isolation failed")

        checkpoint = args.run_dir / "query_residual_20updates.pth"
        saved = cpu_tree(branch.state_dict())
        with checkpoint.open("xb") as stream:
            torch.save(dict(model=saved, metadata=dict(report)), stream)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (set(restored.get("model", {})) != set(saved)
                or any(not torch.equal(restored["model"][key], value)
                       for key, value in saved.items())):
            raise ValueError("Branch checkpoint roundtrip mismatch")
        report.update(
            ok=True,
            stage="complete",
            training_record_count=len(losses),
            all_losses_finite=True,
            first_20_loss_mean=sum(losses[:20]) / 20,
            last_20_loss_mean=sum(losses[-20:]) / 20,
            final_alpha=branch.alpha.item(),
            nonzero_gradient_parameters=sorted(gradient_names),
            intervention_class_id=class_id,
            outside_selected_class_max_error=outside_error,
            selected_class_effect_max_error=selected_error,
            base_parameters_unchanged=True,
            branch_checkpoint=str(checkpoint),
            branch_checkpoint_sha256=digest(checkpoint),
            branch_checkpoint_bytes=checkpoint.stat().st_size,
            elapsed_seconds=round(time.monotonic() - started, 2),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
    if created_run:
        (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
