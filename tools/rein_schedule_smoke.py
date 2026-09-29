"""Bounded source-only REIN accumulation/schedule/checkpoint gate (not accuracy)."""

import argparse
import contextlib
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import traceback

from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions
from tools.check_rein_segmentor import validate_losses, validate_prediction

DATA_SHA = "b893b991c4bc85eb6105e8d8ae1f39df670b256ce12a10aaca8c1a7c89d71c69"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def accepted_protocol(report):
    if (report.get("ok") is not True or report.get("stage") != "complete"
            or report.get("exact_project_pairing") is not True
            or report.get("source_pairs") != 24966 or report.get("target_pairs") != 500
            or len(report.get("samples", [])) != 10
            or report.get("target_labels_optimized") is not False):
        raise ValueError("Incomplete accepted data protocol")
    return report["adapted_protocol"]


def scheduled_lr(updates):
    if not 0 <= updates <= 40000:
        raise ValueError("Invalid optimizer update count")
    return 1e-4 * (1 - updates / 40000) ** 0.9


def rebuild_protocol(report, upstream):
    """Reapply the audited adapter to typed source; JSON is evidence, not code."""
    from tools.rein_protocol_adapter import adapt_protocol

    saved = accepted_protocol(report)
    dataset = saved["train_dataloader"]["dataset"]
    rebuilt = adapt_protocol(upstream, Path(dataset["data_root"]),
                             Path(dataset["data_prefix"]["img_path"]),
                             Path(dataset["data_prefix"]["seg_map_path"]))
    if json.loads(json.dumps(rebuilt, default=str)) != saved:
        raise ValueError("Reconstructed protocol differs from accepted JSON values")
    return rebuilt


def restore_protocol_config(report, rein_root):
    """Restore both recursive ConfigDict and original tuple/list distinctions."""
    from mmengine.config import Config
    from tools.inspect_rein_protocol import CONFIG

    upstream = Config.fromfile(Path(rein_root) / CONFIG).to_dict()
    config = Config(rebuild_protocol(report, upstream))
    if config.model.decode_head.transformer_decoder.layer_cfg.cross_attn_cfg.num_heads <= 0:
        raise ValueError("Invalid transformer decoder head count")
    return config


def compact_state(model):
    # Do not depend on REIN's customized state_dict filtering.
    buffers = dict(model.named_buffers())
    return {name: tensor.detach().cpu().clone()
            for name, tensor in list(model.named_parameters()) + list(buffers.items())
            if name in buffers or tensor.requires_grad}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = dict(ok=False, purpose="scheduled_source_smoke_not_accuracy_or_exact_resume",
                  git_sha=git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
                  seed=0, microbatches=80, accumulation=4, optimizer_updates=0,
                  schedule_horizon_optimizer_updates=40000, target_labels_optimized=False,
                  formal_training_authorized=False, stage="preflight")
    try:
        if args.run_dir.exists():
            raise FileExistsError("Run directory already exists; preserve it")
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise ValueError("Use isolated Python 3.10")
        if digest(args.data_report) != DATA_SHA or digest(args.weights) != WEIGHT_SHA:
            raise ValueError("Accepted report/weights hash mismatch")
        data_report = json.loads(args.data_report.read_text())
        accepted_protocol(data_report)
        if git_output(args.rein_root, "rev-parse", "HEAD") != REIN_SHA or git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("REIN source must be pinned and clean")
        if shutil.disk_usage(args.run_dir.parent).free < 2 * 2**30:
            raise OSError("Need at least 2 GiB free for bounded checkpoint gate")
        import torch
        import numpy as np
        versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        if torch.version.cuda != "11.8" or not torch.cuda.is_available():
            raise RuntimeError("CUDA 11.8 GPU required")
        report.update(versions=versions, python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                      pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA, dtype="float32")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)
        sys.path.insert(0, str(args.rein_root.resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.registry import init_default_scope
            from mmengine.optim import build_optim_wrapper, PolyLR
            from mmseg.registry import DATASETS, MODELS
            from tools.rein_protocol_adapter import register_data_transforms
            init_default_scope("mmseg")
            register_data_transforms()
            report["stage"] = "config_restore"
            config = restore_protocol_config(data_report, args.rein_root)
            report["config_container"] = "typed_pinned_upstream_plus_audited_adapter"
            report["stage"] = "model_build"
            model = MODELS.build(config["model"])
            model.decode_head.init_weights()
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
            loaded = model.backbone.load_state_dict(state, strict=False)
            validate_load_keys(state, keys, loaded.missing_keys, loaded.unexpected_keys)
            del state
            model.train(True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad != name.startswith(("backbone.reins.", "decode_head.")):
                    raise ValueError(f"Unexpected trainability: {name}")
            model.to("cuda:0")
            report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = model.backbone.patch_embed.proj.weight.detach().clone()
            wrapper = build_optim_wrapper(model, config["optim_wrapper"])
            if wrapper._accumulative_counts != 4:
                raise ValueError("Expected accumulation 4")
            wrapper.initialize_count_status(model, 0, 80)
            scheduler = PolyLR(wrapper, eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
            report["stage"] = "dataset_build"
            dataset = DATASETS.build(config["train_dataloader"]["dataset"])
            if len(dataset) != 24966:
                raise ValueError("Source coverage changed")
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:80].tolist()
            args.run_dir.mkdir(parents=False)
            (args.run_dir / "metadata.json").write_text(json.dumps(report, indent=2))
            torch.cuda.reset_peak_memory_stats(0)
            report["stage"] = "source_optimization"
            with (args.run_dir / "train.jsonl").open("x") as trace:
                for iteration, index in enumerate(indices, 1):
                    item = dataset[index]
                    batch = model.data_preprocessor(dict(inputs=[item["inputs"]], data_samples=[item["data_samples"]]), training=True)
                    if not (batch["data_samples"][0].gt_sem_seg.data != 255).any().item():
                        raise ValueError("Source crop has no valid pixels")
                    with wrapper.optim_context(model):
                        losses = model.loss(batch["inputs"], batch["data_samples"])
                        total = validate_losses(losses)
                    wrapper.backward(wrapper.scale_loss(total))
                    norms = []
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if not parameter.requires_grad or not torch.isfinite(parameter.grad).all().item():
                                raise FloatingPointError(f"Invalid gradient: {name}")
                            norms.append(parameter.grad.float().norm().item())
                    norm = math.sqrt(sum(value * value for value in norms))
                    if not math.isfinite(norm) or norm <= 0:
                        raise FloatingPointError("Invalid accumulated gradient norm")
                    update = wrapper.should_update()
                    if update != (iteration % 4 == 0):
                        raise ValueError("Unexpected accumulation boundary")
                    if update:
                        wrapper.step()
                        wrapper.zero_grad(set_to_none=True)
                        scheduler.step()
                        report["optimizer_updates"] += 1
                    rates = [group["lr"] for group in wrapper.optimizer.param_groups]
                    if any(abs(rate - scheduled_lr(report["optimizer_updates"])) > 1e-10 for rate in rates):
                        raise ValueError("Optimizer-update PolyLR mismatch")
                    record = dict(iteration=iteration, dataset_index=index, loss=total.item(),
                                  gradient_norm=norm, optimizer_update=update, lr=rates[0])
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    print(json.dumps(record), file=sys.stderr)
                    del total, losses
            if report["optimizer_updates"] != 20 or not torch.equal(frozen, model.backbone.patch_embed.proj.weight):
                raise ValueError("Update count/frozen backbone gate failed")
            if any(not torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
                raise FloatingPointError("Non-finite trained parameter")
            model.eval()
            inputs = batch["inputs"]
            meta = [dict(ori_shape=(512, 512), img_shape=(512, 512), pad_shape=(512, 512))]
            with torch.no_grad():
                before = model.encode_decode(inputs, meta)
            validate_prediction(before)
            report["stage"] = "checkpoint_roundtrip"
            checkpoint = args.run_dir / "checkpoint_20updates.pth"
            saved = compact_state(model)
            with checkpoint.open("xb") as stream:
                torch.save(dict(model=saved, optimizer=wrapper.state_dict(), scheduler=scheduler.state_dict(),
                                metadata=dict(report)), stream)
            if checkpoint.stat().st_size > 2**30:
                raise ValueError("Checkpoint exceeded 1 GiB budget; preserved for inspection")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if set(payload["model"]) != set(saved):
                raise ValueError("Compact checkpoint key mismatch")
            named = dict(model.named_parameters()) | dict(model.named_buffers())
            with torch.no_grad():
                # Force an actual restoration, not a no-op load.
                first = next(iter(saved))
                named[first].add_(1)
                for name, tensor in payload["model"].items():
                    named[name].copy_(tensor.to(named[name].device))
            wrapper.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            with torch.no_grad():
                after = model.encode_decode(inputs, meta)
            validate_prediction(after)
            error = (before - after).abs().max().item()
            if error > 1e-5 or any(not torch.equal(named[name].detach().cpu(), tensor) for name, tensor in saved.items()):
                raise ValueError("Checkpoint restoration mismatch")
            if any(abs(group["lr"] - scheduled_lr(20)) > 1e-10 for group in wrapper.optimizer.param_groups):
                raise ValueError("Restored optimizer LR mismatch")
            report.update(ok=True, stage="complete", checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                          checkpoint_bytes=checkpoint.stat().st_size, prediction_roundtrip_max_error=error,
                          peak_allocated_gib=round(torch.cuda.max_memory_allocated(0)/2**30, 3),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved(0)/2**30, 3))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    if args.run_dir.is_dir() and report.get("stage") != "preflight":
        (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
