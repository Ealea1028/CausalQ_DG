from pathlib import Path

import pytest
import torch

from causalq.interventions import StyleInterventionBank
from tools.train import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_style_configuration_preserves_geometry() -> None:
    config = (ROOT / "configs/style/gta_dinov3l_style.yaml").read_text(encoding="utf-8")
    assert "views: [original, photometric, fourier]" in config
    assert "preserve_geometry: true" in config
    assert "sequential_forward: true" in config
    assert "lambda_cf: 1.0" in config

    parsed = load_config(ROOT / "configs/style/gta_dinov3l_style.yaml")
    assert parsed["experiment"]["phase"] == 7
    assert parsed["train"]["style"] is True


def test_style_views_preserve_shape_and_original() -> None:
    torch.manual_seed(7)
    images = torch.randn(2, 3, 16, 24)
    bank = StyleInterventionBank(grayscale_probability=0.0)
    views = bank(images)

    assert views.original.data_ptr() == images.data_ptr()
    for _, view in views.items():
        assert view.shape == images.shape
        assert torch.isfinite(view).all()
    assert not torch.equal(views.photometric, images)
    assert not torch.equal(views.fourier, images)


def test_fourier_intervention_preserves_unclipped_phase() -> None:
    torch.manual_seed(11)
    bank = StyleInterventionBank(fourier_mix_strength=(0.2, 0.2))
    images = torch.rand(1, 3, 12, 10)
    normalized = (images - bank.mean) / bank.std

    original_spectrum = torch.fft.fft2(images, dim=(-2, -1))
    original_phase = original_spectrum / original_spectrum.abs().clamp_min(1e-8)
    amplitude = original_spectrum.abs()
    donor = amplitude.roll(shifts=1, dims=1)
    reconstructed = torch.fft.ifft2(
        amplitude.lerp(donor, 0.2) * original_phase, dim=(-2, -1)
    ).real
    expected = bank._normalize(reconstructed, normalized.dtype)

    torch.manual_seed(11)
    actual = bank.fourier_view(normalized)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_style_bank_rejects_invalid_input() -> None:
    bank = StyleInterventionBank()
    with pytest.raises(ValueError, match="Bx3xHxW"):
        bank(torch.randn(3, 16, 16))
