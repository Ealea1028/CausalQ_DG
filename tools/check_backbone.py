"""Load DINOv3 on a GPU and report dense-feature shape and memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.models.dinov3_wrapper import DINOV3_MODEL_IDS, DINOv3Backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DINOV3_MODEL_IDS), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--intermediate-indices", type=int, nargs="*", default=())
    return parser.parse_args()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_hashes(weights: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(weights.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "ok": False,
        "git_sha": git_sha(),
        "model": args.model,
        "official_model_id": DINOV3_MODEL_IDS[args.model],
    }
    if not torch.cuda.is_available():
        report["error"] = "CUDA is not available"
        print(json.dumps(report, indent=2))
        return 1

    dtype = getattr(torch, args.dtype)
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    weights = args.weights or pretrained_root / args.model
    device = torch.device("cuda:0")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        backbone = DINOv3Backbone.from_pretrained(
            args.model,
            weights=weights,
            freeze=True,
            intermediate_indices=args.intermediate_indices,
            dtype=dtype,
            local_files_only=True,
        ).to(device)
        images = torch.randn(
            args.batch_size,
            3,
            *args.image_size,
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            features = backbone(images)
        hashes = checkpoint_hashes(weights)
        tensors = (features.patch_map, *features.intermediate_maps)
        nan_count = sum(torch.isnan(tensor).sum().item() for tensor in tensors)
        inf_count = sum(torch.isinf(tensor).sum().item() for tensor in tensors)
        total_parameters = sum(parameter.numel() for parameter in backbone.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        )
        report.update(
            {
                "weights": str(weights),
                "weights_loaded": True,
                "checkpoint_sha256": hashes,
                "device": torch.cuda.get_device_name(device),
                "dtype": str(features.patch_map.dtype),
                "input_shape": list(images.shape),
                "patch_size": list(backbone.patch_size),
                "num_prefix_tokens": backbone.num_prefix_tokens,
                "hidden_size": backbone.hidden_size,
                "patch_token_shape": list(features.patch_tokens.shape),
                "patch_map_shape": list(features.patch_map.shape),
                "intermediate_map_shapes": [
                    list(feature.shape) for feature in features.intermediate_maps
                ],
                "nan_count": nan_count,
                "inf_count": inf_count,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / 1024**3, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device) / 1024**3, 3
                ),
            }
        )
        expected_grid = (
            args.image_size[0] // backbone.patch_size[0],
            args.image_size[1] // backbone.patch_size[1],
        )
        report["ok"] = (
            tuple(features.patch_map.shape[-2:]) == expected_grid
            and nan_count == 0
            and inf_count == 0
            and trainable_parameters == 0
            and bool(hashes)
        )
    except Exception as exc:
        report["weights"] = str(weights)
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
