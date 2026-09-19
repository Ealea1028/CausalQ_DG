from pathlib import Path

import pytest

from tools.train import load_config, resolve_seed, style_view_weights


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "expected_views"),
    (
        ("gta_dinov3l_photo.yaml", ["original", "photometric"]),
        ("gta_dinov3l_fourier.yaml", ["original", "fourier"]),
    ),
)
def test_style_ablation_configs_isolate_one_view(
    filename: str,
    expected_views: list[str],
) -> None:
    config = load_config(ROOT / "configs/style_ablation" / filename)

    assert config["experiment"]["phase"] == 11
    assert config["style"]["views"] == expected_views
    assert style_view_weights(config["style"]) == {
        "original": 1.0,
        expected_views[1]: 1.0,
    }
    assert "prediction_consistency" not in config
    assert "causal_query_effect" not in config
    assert "query_diversity" not in config


def test_combined_style_keeps_total_counterfactual_weight_fixed() -> None:
    config = load_config(ROOT / "configs/style/gta_dinov3l_style.yaml")

    assert style_view_weights(config["style"]) == {
        "original": 1.0,
        "photometric": 0.5,
        "fourier": 0.5,
    }


def test_style_weights_reject_duplicate_or_missing_original_views() -> None:
    with pytest.raises(ValueError, match="original"):
        style_view_weights({"views": ["photometric"], "lambda_cf": 1.0})
    with pytest.raises(ValueError, match="unique"):
        style_view_weights(
            {"views": ["original", "photometric", "photometric"], "lambda_cf": 1.0}
        )


def test_training_seed_can_be_overridden_for_close_ablation_repeats() -> None:
    config = load_config(ROOT / "configs/style/gta_dinov3l_style.yaml")

    assert resolve_seed(config, None) == 0
    assert resolve_seed(config, 1) == 1
    assert resolve_seed(config, 2) == 2
    with pytest.raises(ValueError, match="non-negative"):
        resolve_seed(config, -1)
