"""CPU guards for the Phase-15 matched backbone comparison."""

from copy import deepcopy
from pathlib import Path

import pytest

from tools.compare_backbone_scaling import verify_protocol, verify_run
from tools.train import load_config


ROOT = Path(__file__).resolve().parents[1]
B = load_config(ROOT / "configs/scaling/gta_dinov3b_static.yaml")
L = load_config(ROOT / "configs/query_interaction/gta_dinov3l_static.yaml")


def test_matched_scaling_configs_pass() -> None:
    verify_protocol(B, L)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("query", "queries_per_class", 3),
        ("style", "views", ["original", "fourier"]),
        ("train", "seed", 1),
        ("data", "random_scale", [0.75, 1.25]),
        ("model", "decoder_channels", 128),
    ],
)
def test_protocol_drift_rejected(section: str, key: str, value: object) -> None:
    small = deepcopy(B)
    small[section][key] = value
    with pytest.raises(ValueError):
        verify_protocol(small, L)


def test_checkpoint_and_summary_provenance() -> None:
    payload = {
        "iteration": 40000,
        "metadata": {
            "git_sha": "expected",
            "backbone": B["model"]["backbone"],
            "pretrained_checkpoint_sha256": B["model"]["weights_sha256"],
            "seed": 0,
            "max_iterations": 40000,
            "source": "gta5",
            "validation": "cityscapes_val",
            "query": B["query"],
            "style": B["style"],
        },
    }
    summary = {
        "ok": True,
        "finite_losses": True,
        "git_sha": "expected",
        "validation_results": [
            {"iteration": iteration, "sample_count": 500, "miou": 0.5}
            for iteration in range(500, 40001, 500)
        ],
    }
    assert verify_run(B, payload, summary, expected_sha="expected") == 0.5
    bad = deepcopy(summary)
    bad["validation_results"][-1]["sample_count"] = 50
    with pytest.raises(ValueError, match="500"):
        verify_run(B, payload, bad, expected_sha="expected")
    bad_payload = deepcopy(payload)
    bad_payload["metadata"]["backbone"] = "dinov3_vitl16"
    with pytest.raises(ValueError, match="backbone"):
        verify_run(B, bad_payload, summary, expected_sha="expected")
