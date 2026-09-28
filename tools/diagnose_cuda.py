"""Report CUDA visibility and initialization errors without changing the environment.

Run separately with each AutoDL interpreter. No model, dataset, installation,
or GPU visibility override is involved; the compute probe allocates one scalar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def probe_torch(torch) -> dict[str, object]:
    report: dict[str, object] = {
        "ok": False,
        "torch": str(torch.__version__),
        "torch_file": str(torch.__file__),
        "torch_cuda_runtime": torch.version.cuda,
    }
    try:
        report["cuda_available"] = torch.cuda.is_available()
        report["device_count"] = torch.cuda.device_count()
        # init() retains the real driver/runtime error even if availability is False.
        torch.cuda.init()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))
        value = (torch.ones(1, device="cuda:0") + 1).item()
        torch.cuda.synchronize()
        report["scalar_probe"] = value
        report["ok"] = value == 2.0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    report: dict[str, object] = {
        "ok": False,
        "python": platform.python_version(),
        "executable": sys.executable,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "CONDA_PREFIX",
            "PYTORCH_NVML_BASED_CUDA_CHECK",
        )},
        "nvidia_device_nodes": sorted(str(p) for p in Path("/dev").glob("nvidia*")),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        report["nvidia_smi"] = {
            "exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch

        report["probe"] = probe_torch(torch)
        report["ok"] = report["probe"]["ok"]
    except Exception as exc:
        report["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
