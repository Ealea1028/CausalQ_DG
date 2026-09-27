"""Independently audit an on-disk REIN DINOv2-L conversion against its source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.check_dinov2_checkpoint import git_sha, inspect_state_dict, sha256
from tools.convert_dinov2_for_rein import convert_state_dict


def compare_converted_tensors(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> int:
    """Require exact keys, shapes, dtypes, and tensor values after serialization."""
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"Tensor key mismatch: missing={missing}, extra={extra}")
    for key, reference in expected.items():
        candidate = actual[key]
        if not isinstance(candidate, torch.Tensor):
            raise ValueError(f"Non-tensor value at {key}")
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError(f"Tensor shape or dtype mismatch at {key}")
        if not torch.equal(reference, candidate):
            raise ValueError(f"Tensor value mismatch at {key}")
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--expected-converted-sha256", required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "ok": False,
        "git_sha": git_sha(),
        "source": str(args.source),
        "converted": str(args.converted),
        "pytorch": torch.__version__,
        "load_mode": "weights_only_true_cpu",
        "comparison": "exact_keys_shapes_dtypes_and_tensor_values",
    }
    try:
        for name, path, expected_hash in (
            ("source", args.source, args.expected_source_sha256),
            ("converted", args.converted, args.expected_converted_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = sha256(path)
            report[f"{name}_sha256"] = actual_hash
            report[f"{name}_size_bytes"] = path.stat().st_size
            if actual_hash != expected_hash.lower():
                raise ValueError(f"{name} SHA256 mismatch; tensor load refused")

        source = torch.load(args.source, map_location="cpu", weights_only=True)
        converted = torch.load(args.converted, map_location="cpu", weights_only=True)
        inspect_state_dict(source)
        if not isinstance(converted, dict):
            raise ValueError("Converted checkpoint must contain a tensor mapping")
        expected = convert_state_dict(source)
        report["matching_tensor_count"] = compare_converted_tensors(expected, converted)
        report["source_patch_shape"] = list(source["patch_embed.proj.weight"].shape)
        report["converted_patch_shape"] = list(converted["patch_embed.proj.weight"].shape)
        report["source_pos_shape"] = list(source["pos_embed"].shape)
        report["converted_pos_shape"] = list(converted["pos_embed"].shape)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
