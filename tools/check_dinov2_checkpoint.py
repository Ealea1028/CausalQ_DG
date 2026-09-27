"""Safely inspect a staged official DINOv2 ViT-L/14 backbone checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import subprocess

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_state_dict(payload: object) -> dict[str, object]:
    """Check the defining tensor shapes without constructing or running a model."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a tensor mapping")
    if "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("State dict must be a nonempty mapping")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in payload.items()):
        raise ValueError("State dict must contain only string-to-tensor entries")

    required_shapes = {
        "patch_embed.proj.weight": (1024, 3, 14, 14),
        "cls_token": (1, 1, 1024),
        "blocks.0.attn.qkv.weight": (3072, 1024),
        "blocks.23.attn.qkv.weight": (3072, 1024),
        "norm.weight": (1024,),
    }
    shapes: dict[str, list[int]] = {}
    for key, expected in required_shapes.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            actual = None if value is None else list(value.shape)
            raise ValueError(f"Unexpected {key} shape: {actual}; expected {expected}")
        shapes[key] = list(value.shape)

    pos_embed = payload.get("pos_embed")
    if (not isinstance(pos_embed, torch.Tensor)
            or pos_embed.ndim != 3
            or pos_embed.shape[0] != 1
            or pos_embed.shape[1] < 2
            or pos_embed.shape[2] != 1024):
        raise ValueError("Unexpected pos_embed shape")
    if any(key.endswith("register_tokens") for key in payload):
        raise ValueError("Expected the non-register DINOv2 ViT-L/14 variant")

    block_indices = {
        int(parts[1])
        for key in payload
        if (parts := key.split("."))[0] == "blocks"
        and len(parts) > 1
        and parts[1].isdigit()
    }
    if block_indices != set(range(24)):
        raise ValueError("Expected exactly 24 transformer blocks")

    shapes["pos_embed"] = list(pos_embed.shape)
    return {
        "architecture": "dinov2_vitl14_no_registers",
        "hidden_size": 1024,
        "patch_size": 14,
        "block_count": 24,
        "tensor_count": len(payload),
        "parameter_count": sum(value.numel() for value in payload.values()),
        "representative_shapes": shapes,
    }


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "ok": False,
        "git_sha": git_sha(),
        "weights": str(args.weights),
        "expected_sha256": args.expected_sha256.lower(),
        "pytorch": torch.__version__,
        "load_mode": "weights_only_true_cpu",
    }
    try:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        actual_hash = sha256(args.weights)
        report["actual_sha256"] = actual_hash
        report["size_bytes"] = args.weights.stat().st_size
        if actual_hash != args.expected_sha256.lower():
            raise ValueError("Checkpoint SHA256 mismatch; tensor load refused")
        payload = torch.load(args.weights, map_location="cpu", weights_only=True)
        report.update(inspect_state_dict(payload))
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
