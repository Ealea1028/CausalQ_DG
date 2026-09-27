"""Focused CPU tests for independently comparing a converted state dict."""

import pytest
import torch

from tools.check_dinov2_rein_conversion import compare_converted_tensors


def test_comparison_accepts_exact_tensor_mapping() -> None:
    reference = {"patch": torch.tensor([[1.0, 2.0]]), "other": torch.tensor([3])}
    candidate = {key: value.clone() for key, value in reference.items()}

    assert compare_converted_tensors(reference, candidate) == 2


@pytest.mark.parametrize(
    "candidate,match",
    [
        ({"patch": torch.tensor([[1.0, 2.0]])}, "key mismatch"),
        ({"patch": torch.tensor([[1.0, 2.0]]), "other": torch.tensor([4])},
         "value mismatch"),
        ({"patch": torch.tensor([[1.0, 2.0]]), "other": torch.tensor([3.0])},
         "shape or dtype mismatch"),
    ],
)
def test_comparison_rejects_nonidentical_state(
    candidate: dict[str, torch.Tensor], match: str
) -> None:
    reference = {"patch": torch.tensor([[1.0, 2.0]]), "other": torch.tensor([3])}

    with pytest.raises(ValueError, match=match):
        compare_converted_tensors(reference, candidate)
