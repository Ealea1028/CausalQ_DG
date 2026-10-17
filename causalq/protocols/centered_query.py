"""M1 changes only explicit centered residual parameterization from strict M0."""

from copy import deepcopy
from pathlib import Path

import yaml

from causalq.protocols.strict_dg import validate_strict_config

ROOT = Path(__file__).resolve().parents[2]


def validate_centered_config(config):
    baseline = yaml.safe_load((ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml").read_text(encoding="utf-8"))
    validate_strict_config(baseline)
    expected = deepcopy(baseline)
    expected["experiment"] = {"id": "M1_CENTERED", "phase": 18, "status": "awaiting_centered_gpu_gate"}
    expected["query"]["centered_residuals"] = True
    if config != expected:
        raise ValueError("M1 may change only centered residual parameterization; formal training blocked")
