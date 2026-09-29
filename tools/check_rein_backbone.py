"""GPU-only synthetic REIN backbone gate; no datasets or training checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import sys

from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions

REIN_SHA = "dc063429c4dadc0da9c6252b3db22fc55a9882ab"
WEIGHT_SHA = "91730ebf59fb634f5572cf5071fef8665473dcffcbef7ba4f4fa497533a8c837"


def validate_load_keys(source_keys, model_keys, missing_keys, unexpected_keys):
    """Require complete pretrained coverage, allowing only new REIN parameters."""
    source, model = set(source_keys), set(model_keys)
    expected = {key for key in model if not key.startswith("reins.")}
    adapters = model - expected
    if len(source) != 343 or source != expected or not adapters:
        raise ValueError("Pretrained key coverage is not exact (expected 343 backbone tensors)")
    if set(missing_keys) != adapters or unexpected_keys:
        raise ValueError("Only newly initialized reins.* keys may be missing")


def validate_outputs(features, queries):
    import torch

    shapes = [(1, 1024, size, size) for size in (128, 64, 32, 16)]
    if len(features) != 4 or [tuple(x.shape) for x in features] != shapes:
        raise ValueError("Unexpected upstream REIN feature pyramid")
    if tuple(queries.shape) != (100, 256):
        raise ValueError("Unexpected linked-query shape")
    if not all(torch.isfinite(x).all().item() for x in [*features, queries]):
        raise FloatingPointError("Non-finite backbone output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "purpose": "synthetic_backbone_gate_not_segmentation_validation",
              "git_sha": git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
              "python": platform.python_version(), "seed": 20260929,
              "weights": str(args.weights), "rein_root": str(args.rein_root)}
    try:
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise ValueError("Use the isolated Python 3.10 REIN environment")
        report["rein_sha"] = git_output(args.rein_root, "rev-parse", "HEAD")
        if report["rein_sha"] != REIN_SHA or git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("Upstream REIN must be pinned and clean")
        digest = hashlib.sha256()
        with args.weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["checkpoint_sha256"] = digest.hexdigest()
        if report["checkpoint_sha256"] != WEIGHT_SHA:
            raise ValueError("Converted checkpoint hash mismatch")
        import importlib
        import torch

        versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        report["versions"] = versions
        if torch.version.cuda != "11.8" or not torch.cuda.is_available():
            raise RuntimeError("Validated CUDA 11.8 GPU runtime is required")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda"] = torch.version.cuda
        torch.manual_seed(report["seed"])
        torch.cuda.manual_seed_all(report["seed"])
        sys.path.insert(0, str(args.rein_root.resolve()))
        # Upstream emits optional ConvNeXt messages to stdout; keep JSON clean.
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.config import Config
            from mmseg.registry import MODELS
            from mmengine.registry import init_default_scope

            init_default_scope("mmseg")
            config = Config.fromfile(args.rein_root / "configs/_base_/models/rein_dinov2_mask2former.py")
            backbone_config = config.model.backbone.copy()
            backbone_config["init_cfg"] = None  # Explicit safe, audited load below.
            model = MODELS.build(backbone_config)
        payload = torch.load(args.weights, map_location="cpu", weights_only=True)
        model_keys = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
        result = model.load_state_dict(payload, strict=False)
        validate_load_keys(payload, model_keys, result.missing_keys, result.unexpected_keys)
        report["loaded_tensor_count"] = len(payload)
        del payload
        # Upstream train(True) freezes the pretrained path and returns None.
        model.train(True)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(not name.startswith("reins.") for name in trainable):
            raise ValueError("Only REIN adapters must be trainable")
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.to("cuda:0")
        torch.cuda.reset_peak_memory_stats(0)
        inputs = torch.randn(1, 3, 512, 512, device="cuda:0")
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            features, queries = model(inputs)
        validate_outputs(features, queries)
        report["feature_shapes"] = [list(x.shape) for x in features]
        report["query_shape"] = list(queries.shape)
        del features, queries
        model.train(True)
        with torch.autocast("cuda", dtype=torch.float16):
            features, queries = model(inputs)
        validate_outputs(features, queries)
        # Diagnostic objective, NOT a segmentation loss or an optimization step.
        loss = sum(x.float().square().mean() for x in features) + queries.float().square().mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite diagnostic objective")
        loss.backward()
        nonzero = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad and parameter.grad is not None:
                raise ValueError("Frozen backbone received a gradient")
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"Non-finite gradient: {name}")
                if parameter.grad.abs().max().item() > 0:
                    nonzero.append(name)
        if not nonzero:
            raise ValueError("No nonzero adapter gradients")
        torch.cuda.synchronize()
        report.update(synthetic_loss=loss.item(), nonzero_adapter_gradient_count=len(nonzero),
                      frozen_backbone_gradients_absent=True,
                      peak_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                      peak_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 2**30, 3), ok=True)
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
