"""Phase-15 DINOv3-B scaling changes only the frozen backbone and scope."""

from copy import deepcopy
from pathlib import Path

import pytest

from tools.train import load_config


ROOT = Path(__file__).resolve().parents[1]
B_CONFIG = ROOT / "configs/scaling/gta_dinov3b_static.yaml"
L_CONFIG = ROOT / "configs/query_interaction/gta_dinov3l_static.yaml"


def test_vitb_scaling_preserves_selected_static_protocol() -> None:
    small = load_config(B_CONFIG)
    large = load_config(L_CONFIG)
    assert small["experiment"]["phase"] == 15
    assert small["data"]["targets"] == ["cityscapes"]
    for section in ("query", "style", "train", "optimizer", "validation"):
        assert small[section] == large[section]
    for key in small["data"]:
        if key != "targets":
            assert small["data"][key] == large["data"][key]
    for key in small["model"]:
        if key not in {"backbone", "weights_dir", "weights_sha256", "intermediate_indices"}:
            assert small["model"][key] == large["model"][key]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("query", "interaction", "one_way", "Static R=2"),
        ("query", "queries_per_class", 3, "Static R=2"),
        ("model", "weights_sha256", "wrong", "verified DINOv3-B"),
        ("data", "targets", ["bdd100k"], "GTA5-to-Cityscapes"),
        ("style", "views", ["original", "fourier"], "GTA5-to-Cityscapes"),
    ],
)
def test_scaling_rejects_protocol_drift(
    section: str, key: str, value: object, message: str, tmp_path: Path
) -> None:
    config = load_config(B_CONFIG)
    config = deepcopy(config)
    config[section][key] = value
    # Validate the modified object through the same public config entry point.
    import yaml

    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_scaling_disables_abandoned_auxiliary_objectives(tmp_path: Path) -> None:
    import yaml

    for mechanism in ("prediction_consistency", "causal_query_effect", "query_diversity", "learned_null"):
        config = load_config(B_CONFIG)
        config[mechanism] = {"enabled": True}
        path = tmp_path / f"{mechanism}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)
