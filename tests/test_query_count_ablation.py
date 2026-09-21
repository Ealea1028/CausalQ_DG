from pathlib import Path

import pytest

from tools.train import load_config, resolve_query_count, style_view_weights


ROOT = Path(__file__).resolve().parents[1]


def test_query_count_config_fixes_the_accepted_style_control() -> None:
    config = load_config(ROOT / "configs/query_count/gta_dinov3l.yaml")

    assert config["experiment"]["phase"] == 12
    assert config["train"]["seed"] == 0
    assert style_view_weights(config["style"]) == {
        "original": 1.0,
        "photometric": 1.0,
    }
    assert "prediction_consistency" not in config
    assert "causal_query_effect" not in config
    assert "query_diversity" not in config


@pytest.mark.parametrize("query_count", (1, 2, 4))
def test_phase_12_accepts_planned_query_counts(query_count: int) -> None:
    config = load_config(ROOT / "configs/query_count/gta_dinov3l.yaml")

    assert resolve_query_count(config, query_count) == query_count


@pytest.mark.parametrize("query_count", (0, 3, 5))
def test_phase_12_rejects_reference_or_out_of_scope_counts(query_count: int) -> None:
    config = load_config(ROOT / "configs/query_count/gta_dinov3l.yaml")

    with pytest.raises(ValueError, match=r"R in \{1, 2, 4\}"):
        resolve_query_count(config, query_count)
