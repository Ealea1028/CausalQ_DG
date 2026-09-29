"""Synthetic full REIN/Mask2Former GPU gate, without optimization or datasets."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import traceback

from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions


def synthetic_label():
    """All 19 classes and an ignore band, independent of real target data."""
    import torch

    label = (torch.arange(512) * 19 // 512).view(1, 1, 512).expand(1, 512, 512).clone()
    label[:, :16, :] = 255
    return label.long()


def validate_losses(losses):
    import torch

    stages = ["decode"] + [f"decode.d{i}" for i in range(9)]
    expected = {f"{stage}.loss_{kind}" for stage in stages for kind in ("cls", "mask", "dice")}
    if set(losses) != expected:
        raise ValueError("Expected all 30 upstream final and auxiliary Mask2Former losses")
    if any(not isinstance(value, torch.Tensor) or value.numel() != 1
           or not torch.isfinite(value).all().item() for value in losses.values()):
        raise FloatingPointError("Non-finite or non-scalar segmentation loss")
    total = sum(losses.values())
    if not torch.isfinite(total).item() or total.item() <= 0:
        raise FloatingPointError("Invalid total segmentation objective")
    return total


def validate_prediction(scores):
    import torch

    if tuple(scores.shape) != (1, 19, 512, 512):
        raise ValueError("Unexpected full-resolution semantic score shape")
    if not torch.isfinite(scores).all().item():
        raise FloatingPointError("Non-finite semantic scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="Opt into the bounded 20-step real-data smoke after the synthetic gate")
    args = parser.parse_args()
    report = dict(ok=False, purpose="synthetic_full_segmentor_not_accuracy_evaluation",
                  git_sha=git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
                  python=platform.python_version(), seed=20260929, dtype="float32",
                  rein_root=str(args.rein_root), weights=str(args.weights), stage="preflight")
    if args.data_root is not None:
        report["purpose"] = "real_data_20step_optimization_smoke_not_accuracy_evaluation"
        report["data_root"] = str(args.data_root)
    try:
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise ValueError("Use the isolated Python 3.10 environment")
        report["rein_sha"] = git_output(args.rein_root, "rev-parse", "HEAD")
        if report["rein_sha"] != REIN_SHA or git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("Upstream REIN source must be pinned and clean")
        digest = hashlib.sha256()
        with args.weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["checkpoint_sha256"] = digest.hexdigest()
        if report["checkpoint_sha256"] != WEIGHT_SHA:
            raise ValueError("Converted checkpoint SHA mismatch")
        import torch

        report["versions"] = {name: str(importlib.import_module(name).__version__)
                              for name in EXPECTED_VERSIONS}
        validate_versions(report["versions"])
        if torch.version.cuda != "11.8" or not torch.cuda.is_available():
            raise RuntimeError("Validated CUDA 11.8 GPU environment is required")
        report.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        torch.manual_seed(report["seed"])
        torch.cuda.manual_seed_all(report["seed"])
        sys.path.insert(0, str(args.rein_root.resolve()))
        # Redirect upstream prints AND newly created logging handlers away from JSON.
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.config import Config
            from mmengine.registry import init_default_scope
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample

            init_default_scope("mmseg")
            config_path = args.rein_root / "configs/_base_/models/rein_dinov2_mask2former.py"
            report["upstream_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            config = Config.fromfile(config_path)
            model_config = copy.deepcopy(config.model)
            model_config.backbone.init_cfg = None
            report["stage"] = "model_build"
            model = MODELS.build(model_config)
            # Initialize the random head BEFORE safely loading pretrained backbone.
            model.decode_head.init_weights()
            report["stage"] = "weight_load"
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
            loaded = model.backbone.load_state_dict(state, strict=False)
            validate_load_keys(state, keys, loaded.missing_keys, loaded.unexpected_keys)
            report["loaded_tensor_count"] = len(state)
            del state
            model.train(True)
            for name, parameter in model.named_parameters():
                expected_trainable = name.startswith(("backbone.reins.", "decode_head."))
                if parameter.requires_grad != expected_trainable:
                    raise ValueError(f"Unexpected trainability: {name}")
            report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.to("cuda:0")
            torch.cuda.reset_peak_memory_stats(0)
            sample = SegDataSample(metainfo=dict(ori_shape=(512, 512), img_shape=(512, 512),
                                                pad_shape=(512, 512), scale_factor=(1.0, 1.0)))
            sample.gt_sem_seg = PixelData(data=synthetic_label())
            image = torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8)
            report["stage"] = "preprocessing"
            batch = model.data_preprocessor(dict(inputs=[image], data_samples=[sample]), training=True)
            inputs, samples = batch["inputs"], batch["data_samples"]
            report["input_shape"] = list(inputs.shape)
            report["synthetic_label_ids"] = samples[0].gt_sem_seg.data.unique().cpu().tolist()
            report["stage"] = "full_prediction"
            model.eval()
            with torch.no_grad():
                scores = model.encode_decode(inputs, [item.metainfo for item in samples])
            validate_prediction(scores)
            report["semantic_score_shape"] = list(scores.shape)
            del scores
            report["stage"] = "segmentation_losses"
            model.train(True)
            losses = model.loss(inputs, samples)
            total = validate_losses(losses)
            report["losses"] = {key: value.item() for key, value in losses.items()}
            report["total_loss"] = total.item()
            report["stage"] = "backward"
            total.backward()
            gradient_counts = dict(adapter=0, head=0, pixel_decoder=0)
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                if not parameter.requires_grad and gradient is not None:
                    raise ValueError(f"Frozen parameter received gradient: {name}")
                if gradient is not None:
                    if not torch.isfinite(gradient).all().item():
                        raise FloatingPointError(f"Non-finite gradient: {name}")
                    if gradient.abs().max().item() > 0:
                        if name.startswith("backbone.reins."):
                            gradient_counts["adapter"] += 1
                        if name.startswith("decode_head."):
                            gradient_counts["head"] += 1
                        if name.startswith("decode_head.pixel_decoder."):
                            gradient_counts["pixel_decoder"] += 1
            if not all(gradient_counts.values()):
                raise ValueError(f"Missing nonzero adapter/head/pixel-decoder gradients: {gradient_counts}")
            torch.cuda.synchronize()
            report.update(ok=True, stage="complete", nonzero_gradient_counts=gradient_counts,
                          frozen_backbone_gradients_absent=True,
                          peak_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 2**30, 3))
            if args.data_root is not None:
                from tools.rein_real_data_smoke import run_real_data_smoke

                report["ok"] = False
                model.zero_grad(set_to_none=True)
                del losses, total, gradient
                report.update(run_real_data_smoke(model, args.data_root, report))
                torch.cuda.synchronize()
                report.update(ok=True, stage="complete",
                              peak_allocated_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                              peak_reserved_gib=round(torch.cuda.max_memory_reserved(0) / 2**30, 3))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
