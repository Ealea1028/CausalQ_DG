"""Convert the audited DINOv2-L/14 weight to REIN's 512px patch-16 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.check_dinov2_checkpoint import git_sha, inspect_state_dict, sha256


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the official REIN DINOv2 conversion, preserving all other tensors."""
    inspect_state_dict(state)
    patch = state["patch_embed.proj.weight"]
    position = state["pos_embed"]
    if tuple(position.shape) != (1, 1370, 1024):
        raise ValueError("Expected the original 37x37 DINOv2-L position grid")

    converted = dict(state)
    converted["patch_embed.proj.weight"] = F.interpolate(
        patch.float(), size=(16, 16), mode="bicubic", align_corners=False
    )
    patch_positions = position[:, 1:, :].reshape(1, 37, 37, 1024)
    patch_positions = F.interpolate(
        patch_positions.permute(0, 3, 1, 2).float(),
        size=(32, 32),
        mode="bicubic",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, 1024, 1024)
    converted["pos_embed"] = torch.cat((position[:, :1, :], patch_positions), dim=1)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "ok": False,
        "git_sha": git_sha(),
        "source": str(args.weights),
        "expected_source_sha256": args.expected_sha256.lower(),
        "output": str(args.output),
        "pytorch": torch.__version__,
        "upstream_conversion": "w1oves/Rein tools/convert_models/convert_dinov2.py",
        "source_load_mode": "weights_only_true_cpu",
    }
    try:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        stage = args.output.with_name(args.output.name + ".part")
        if stage.exists() or stage.is_symlink():
            raise FileExistsError(stage)
        actual_hash = sha256(args.weights)
        report["source_sha256"] = actual_hash
        if actual_hash != args.expected_sha256.lower():
            raise ValueError("Source SHA256 mismatch; load refused")
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        inspect_state_dict(state)
        converted = convert_state_dict(state)
        report["source_patch_shape"] = list(state["patch_embed.proj.weight"].shape)
        report["converted_patch_shape"] = list(converted["patch_embed.proj.weight"].shape)
        report["source_pos_shape"] = list(state["pos_embed"].shape)
        report["converted_pos_shape"] = list(converted["pos_embed"].shape)
        report["tensor_count"] = len(converted)
        if report["converted_patch_shape"] != [1024, 3, 16, 16]:
            raise ValueError("Converted patch embedding has unexpected shape")
        if report["converted_pos_shape"] != [1, 1025, 1024]:
            raise ValueError("Converted position embedding has unexpected shape")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(converted, stage)
        report["converted_sha256"] = sha256(stage)
        report["converted_size_bytes"] = stage.stat().st_size
        stage.rename(args.output)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
