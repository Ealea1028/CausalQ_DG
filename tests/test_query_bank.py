from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_query_configuration_contract_is_recorded() -> None:
    config = (ROOT / "configs/query/gta_dinov3l_query.yaml").read_text(encoding="utf-8")
    assert "queries_per_class: 3" in config
    assert "alpha_init: 0.0" in config
    assert "cross_attention_layers: 1" in config

