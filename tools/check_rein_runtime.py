"""Probe an isolated, pinned REIN/OpenMMLab runtime before model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess


EXPECTED_VERSIONS = {
    "torch": "2.0.1+cu118",
    "torchvision": "0.15.2+cu118",
    "numpy": "1.26.4",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmseg": "1.2.2",
    "mmdet": "3.3.0",
}


def validate_versions(versions: dict[str, str]) -> None:
    """Refuse silently mixed OpenMMLab or PyTorch environments."""
    for name, expected in EXPECTED_VERSIONS.items():
        if versions.get(name) != expected:
            raise ValueError(
                f"{name} version {versions.get(name)!r}; expected {expected!r}"
            )


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--expected-rein-sha", required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "ok": False,
        "python": platform.python_version(),
        "rein_root": str(args.rein_root),
        "expected_rein_sha": args.expected_rein_sha,
    }
    try:
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise ValueError("Isolated REIN environment must use Python 3.10")
        if not args.rein_root.is_dir():
            raise FileNotFoundError(args.rein_root)
        actual_sha = git_output(args.rein_root, "rev-parse", "HEAD")
        report["rein_sha"] = actual_sha
        if actual_sha != args.expected_rein_sha:
            raise ValueError("REIN source SHA mismatch")
        if git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("REIN source worktree is not clean")

        import mmcv
        import mmengine
        import mmdet
        import mmseg
        import numpy
        import torch
        import torchvision

        versions = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "mmcv": mmcv.__version__,
            "mmengine": mmengine.__version__,
            "mmseg": mmseg.__version__,
            "mmdet": mmdet.__version__,
        }
        report["versions"] = versions
        validate_versions(versions)
        report["torch_cuda_runtime"] = torch.version.cuda
        if torch.version.cuda != "11.8" or not torch.cuda.is_available():
            raise RuntimeError("CUDA 11.8 runtime or GPU is unavailable")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))

        from mmcv.ops import MultiScaleDeformableAttention, nms

        if MultiScaleDeformableAttention is None:
            raise RuntimeError("MultiScaleDeformableAttention is unavailable")
        boxes = torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 3.0, 3.0]], device="cuda:0"
        )
        scores = torch.tensor([0.9, 0.8], device="cuda:0")
        _, keep = nms(boxes, scores, 0.5)
        report["cuda_nms_kept_indices"] = keep.cpu().tolist()
        if report["cuda_nms_kept_indices"] != [0]:
            raise RuntimeError("MMCV CUDA NMS returned an unexpected result")

        import sys

        sys.path.insert(0, str(args.rein_root.resolve()))
        import rein  # noqa: F401
        from mmengine.config import Config
        from mmseg.registry import MODELS

        config_path = (
            args.rein_root
            / "configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py"
        )
        config = Config.fromfile(config_path)
        report["config_model_backbone"] = config.model.backbone.type
        report["config_model_head"] = config.model.decode_head.type
        report["registered_backbone"] = MODELS.get("ReinsDinoVisionTransformer") is not None
        report["registered_head"] = MODELS.get("ReinMask2FormerHead") is not None
        if (report["config_model_backbone"] != "ReinsDinoVisionTransformer"
                or report["config_model_head"] != "ReinMask2FormerHead"
                or not report["registered_backbone"]
                or not report["registered_head"]):
            raise RuntimeError("Pinned REIN model configuration is unavailable")
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
