from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_causal_effect_contract_is_recorded() -> None:
    method = (ROOT / "docs/METHOD.md").read_text(encoding="utf-8")
    config = (ROOT / "configs/causalq/gta_dinov3l_causalq.yaml").read_text(encoding="utf-8")
    assert "do(Q_c = 0)" in method
    assert "effect_space: logits" in config
    assert "present_classes_only: true" in config

