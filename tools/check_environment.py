"""Report the runtime needed before any AutoDL training is attempted."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path


PACKAGES = {
    "torchvision": "torchvision",
    "transformers": "transformers",
    "timm": "timm",
    "lightning": "lightning",
    "torchmetrics": "torchmetrics",
}

DEFAULT_PATHS = {
    "data": "/root/autodl-tmp/datasets",
    "pretrained": "/root/autodl-tmp/pretrained",
    "outputs": "/root/autodl-tmp/outputs/CausalQ_DG",
}


def package_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    report: dict[str, object] = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {},
        "paths": {
            "data": os.getenv("CAUSALQ_DATA_ROOT", DEFAULT_PATHS["data"]),
            "pretrained": os.getenv(
                "CAUSALQ_PRETRAINED_ROOT", DEFAULT_PATHS["pretrained"]
            ),
            "outputs": os.getenv("CAUSALQ_OUTPUT_ROOT", DEFAULT_PATHS["outputs"]),
        },
        "path_source": {
            "data": "environment" if os.getenv("CAUSALQ_DATA_ROOT") else "default",
            "pretrained": (
                "environment" if os.getenv("CAUSALQ_PRETRAINED_ROOT") else "default"
            ),
            "outputs": "environment" if os.getenv("CAUSALQ_OUTPUT_ROOT") else "default",
        },
    }

    package_report: dict[str, str] = {}
    for label, module_name in PACKAGES.items():
        try:
            package_report[label] = package_version(module_name)
        except Exception as exc:  # dependency failures should appear in one report
            package_report[label] = f"ERROR: {exc}"
            report["ok"] = False
    report["packages"] = package_report

    gpu_report: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_report.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 2**30, 2),
                    "capability": f"{properties.major}.{properties.minor}",
                }
            )
        probe = torch.tensor([1.0, 2.0, 3.0], device="cuda").square().sum()
        report["cuda_probe"] = float(probe.cpu())
    else:
        report["ok"] = False
    report["gpus"] = gpu_report

    for value in report["paths"].values():
        if value is None or not Path(value).is_absolute():
            report["ok"] = False

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
