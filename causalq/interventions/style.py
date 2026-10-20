"""Geometry-preserving appearance interventions for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StyleViews:
    """Aligned normalized views of one geometrically transformed batch."""

    original: Tensor
    photometric: Tensor
    fourier: Tensor

    def items(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            ("original", self.original),
            ("photometric", self.photometric),
            ("fourier", self.fourier),
        )


class StyleInterventionBank(nn.Module):
    """Create appearance-only views while preserving spatial correspondence."""

    def __init__(
        self,
        *,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        brightness: tuple[float, float] = (0.7, 1.3),
        contrast: tuple[float, float] = (0.7, 1.3),
        saturation: tuple[float, float] = (0.7, 1.3),
        gamma: tuple[float, float] = (0.8, 1.2),
        temperature: tuple[float, float] = (0.9, 1.1),
        grayscale_probability: float = 0.1,
        fourier_mix_strength: tuple[float, float] = (0.1, 0.35),
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.brightness = self._range(brightness, "brightness")
        self.contrast = self._range(contrast, "contrast")
        self.saturation = self._range(saturation, "saturation")
        self.gamma = self._range(gamma, "gamma")
        self.temperature = self._range(temperature, "temperature")
        self.fourier_mix_strength = self._range(
            fourier_mix_strength, "fourier_mix_strength"
        )
        if not 0.0 <= grayscale_probability <= 1.0:
            raise ValueError("grayscale_probability must be between zero and one")
        self.grayscale_probability = float(grayscale_probability)

    @staticmethod
    def _range(values: tuple[float, float], name: str) -> tuple[float, float]:
        low, high = map(float, values)
        if low <= 0 or high < low:
            raise ValueError(f"Invalid {name} range: {values}")
        return low, high

    def _unit(self, images: Tensor) -> Tensor:
        return (images.float() * self.std + self.mean).clamp(0.0, 1.0)

    def _normalize(self, images: Tensor, dtype: torch.dtype) -> Tensor:
        return ((images.clamp(0.0, 1.0) - self.mean) / self.std).to(dtype)

    @staticmethod
    def _sample(
        value_range: tuple[float, float], images: Tensor
    ) -> Tensor:
        low, high = value_range
        return torch.empty(
            images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype
        ).uniform_(low, high)

    def photometric_view(self, images: Tensor) -> Tensor:
        return self._photometric(images)

    def photometric_view_with_trace(self, images: Tensor) -> tuple[Tensor, dict]:
        """Identical draws/order to photometric_view; trace does not draw RNG."""
        trace = {}
        return self._photometric(images, trace), trace

    def _photometric(self, images: Tensor, trace: dict | None = None) -> Tensor:
        dtype = images.dtype
        unit = self._unit(images)

        def sample(name):
            value = self._sample(getattr(self, name), unit)
            if trace is not None:
                trace[name] = value.detach().cpu().flatten().tolist()
            return value

        unit = unit * sample('brightness')
        channel_mean = unit.mean(dim=(2, 3), keepdim=True)
        unit = (unit - channel_mean) * sample('contrast') + channel_mean
        gray = unit.mean(dim=1, keepdim=True)
        unit = (unit - gray) * sample('saturation') + gray
        unit = unit.clamp(1e-6, 1.0).pow(sample('gamma'))

        warmth = sample('temperature')
        temperature_scale = torch.cat(
            (warmth, torch.ones_like(warmth), warmth.reciprocal()), dim=1
        )
        unit = unit * temperature_scale
        if self.grayscale_probability:
            grayscale = (
                torch.rand(unit.shape[0], 1, 1, 1, device=unit.device)
                < self.grayscale_probability
            )
            if trace is not None:
                trace['grayscale'] = grayscale.detach().cpu().flatten().tolist()
            unit = torch.where(grayscale, unit.mean(dim=1, keepdim=True), unit)
        elif trace is not None:
            trace['grayscale'] = [False] * images.shape[0]
        return self._normalize(unit, dtype)

    def fourier_view(self, images: Tensor) -> Tensor:
        """Mix RGB amplitudes while retaining each channel's Fourier phase."""
        dtype = images.dtype
        unit = self._unit(images)
        spectrum = torch.fft.fft2(unit, dim=(-2, -1))
        amplitude = spectrum.abs()
        phase = spectrum / amplitude.clamp_min(1e-8)

        if unit.shape[0] > 1:
            donor = amplitude.roll(shifts=1, dims=0)
        else:
            donor = amplitude.roll(shifts=1, dims=1)
        strength = self._sample(self.fourier_mix_strength, unit)
        mixed_amplitude = amplitude.lerp(donor, strength)
        mixed = torch.fft.ifft2(mixed_amplitude * phase, dim=(-2, -1)).real
        return self._normalize(mixed, dtype)

    def forward(self, images: Tensor) -> StyleViews:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape Bx3xHxW")
        return StyleViews(
            original=images,
            photometric=self.photometric_view(images),
            fourier=self.fourier_view(images),
        )
