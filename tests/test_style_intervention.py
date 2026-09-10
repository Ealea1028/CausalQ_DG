from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_style_configuration_preserves_geometry() -> None:
    config = (ROOT / "configs/style/gta_dinov3l_style.yaml").read_text(encoding="utf-8")
    assert "views: [original, photometric, fourier]" in config
    assert "preserve_geometry: true" in config
    assert "sequential_forward: true" in config

