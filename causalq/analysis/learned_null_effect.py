"""Paired localization diagnostics for zero and learned-null effects."""

from __future__ import annotations

from torch import Tensor

from .zero_effect import zero_effect_localization_values


def paired_effect_localization_values(
    zero_effects: Tensor,
    learned_null_effects: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
) -> dict[str, Tensor]:
    """Return aligned localization values for two intervention baselines."""
    if zero_effects.shape != learned_null_effects.shape:
        raise ValueError("zero and learned-null effects must have identical shapes")
    zero = zero_effect_localization_values(
        zero_effects,
        labels,
        ignore_index=ignore_index,
    )
    learned_null = zero_effect_localization_values(
        learned_null_effects,
        labels,
        ignore_index=ignore_index,
    )
    if zero["absolute_ratio"].shape != learned_null["absolute_ratio"].shape:
        raise RuntimeError("intervention baselines produced unaligned class maps")

    result = {f"zero_{key}": value for key, value in zero.items()}
    result.update(
        {f"learned_null_{key}": value for key, value in learned_null.items()}
    )
    result["learned_null_has_higher_absolute_ratio"] = (
        learned_null["absolute_ratio"] > zero["absolute_ratio"]
    )
    result["learned_null_has_lower_outside_absolute_mean"] = (
        learned_null["outside_absolute_mean"] < zero["outside_absolute_mean"]
    )
    return result
