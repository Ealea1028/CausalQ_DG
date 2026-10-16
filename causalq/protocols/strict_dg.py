"""Explicit M0 protocol guard; historical configurations keep their behavior."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "name": "strict_dg", "target_access": "forbidden_during_training",
    "source_validation": "gta5_holdout", "target_evaluation": "final_only",
    "checkpoint_selection": "fixed_final", "training_authorized": False,
}


def validate_strict_config(config: dict) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Strict-DG protocol contract mismatch (training is not authorized)")
    if config.get("experiment") != {"id": "M0_STRICT_DG", "phase": 17,
                                     "status": "awaiting_source_data_gate"}:
        raise ValueError("Only the isolated M0 source-data gate is prepared")
    reference = yaml.safe_load((ROOT / "configs/query_interaction/gta_dinov3l_static.yaml")
                               .read_text(encoding="utf-8"))
    expected = deepcopy(reference)
    expected["experiment"] = config["experiment"]
    expected["protocol"] = PROTOCOL
    expected["data"].update(targets=[], validation="gta5_holdout")
    expected["validation"].update(long_edge=1024, patch_size=16,
                                   geometry="aspect_resize_pad", scope="full_source_holdout")
    if config != expected:
        raise ValueError("M0 data gate changes only source split/validation protocol from Static R=2")
