import pytest
import torch

from causalq.analysis import (
    active_query_behavior_values,
    present_class_mask,
    within_class_query_similarity_values,
)


def test_within_class_similarity_uses_unordered_pairs() -> None:
    states = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]]
    )

    values = within_class_query_similarity_values(states)

    assert values.shape == (3,)
    assert torch.allclose(values, torch.tensor([0.0, 1.0, 0.0]))


def test_within_class_similarity_returns_empty_for_one_query() -> None:
    values = within_class_query_similarity_values(torch.randn(2, 3, 1, 4))

    assert values.numel() == 0


def test_present_class_mask_excludes_ignore_pixels_and_absent_classes() -> None:
    labels = torch.tensor([[[0, 255], [2, 2]], [[1, 1], [255, 1]]])

    present = present_class_mask(labels, num_classes=3)

    assert torch.equal(
        present,
        torch.tensor([[True, False, True], [False, True, False]]),
    )


def test_active_query_behavior_for_balanced_queries() -> None:
    scores = torch.zeros(1, 2, 2, 2, 2)
    labels = torch.tensor([[[0, 0], [1, 255]]])

    values = active_query_behavior_values(scores, labels)

    assert values["effective_query_count"].shape == (2,)
    assert torch.allclose(values["effective_query_count"], torch.full((2,), 2.0))
    assert torch.allclose(values["effective_query_fraction"], torch.ones(2))
    assert torch.allclose(values["dominant_query_share"], torch.full((2,), 0.5))
    assert torch.allclose(
        values["pixel_top1_responsibility"], torch.full((2,), 0.5)
    )


def test_active_query_behavior_for_single_query() -> None:
    scores = torch.randn(1, 3, 1, 2, 2)
    labels = torch.tensor([[[0, 1], [2, 255]]])

    values = active_query_behavior_values(scores, labels)

    for metric in values.values():
        assert torch.allclose(metric, torch.ones(3))


def test_active_query_behavior_rejects_invalid_labels() -> None:
    scores = torch.zeros(1, 2, 2, 1, 1)

    with pytest.raises(ValueError, match="outside"):
        active_query_behavior_values(scores, torch.tensor([[[2]]]))
