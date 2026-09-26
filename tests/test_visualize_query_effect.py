"""CPU-only tests for the label-selected qualitative audit."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from causalq.datasets.segmentation import SegmentationPair
from tools.visualize_query_effect import select_samples, signed_effect_rgb


def test_select_samples_uses_labels_only_and_distinct_examples(tmp_path: Path) -> None:
    pairs = []
    for index, class_id in enumerate((0, 13, 11, 8)):
        label = tmp_path / f"{index}.png"
        Image.fromarray(np.full((4, 4), class_id, dtype=np.uint8)).save(label)
        pairs.append(SegmentationPair(tmp_path / "missing-image.png", label, str(index)))
    result = select_samples(SimpleNamespace(pairs=pairs), min_class_pixels=8)
    assert result == {"road": 0, "car": 1, "person": 2, "vegetation": 3}
    with pytest.raises(RuntimeError, match="road"):
        select_samples(SimpleNamespace(pairs=pairs), min_class_pixels=17)


def test_signed_effect_uses_shared_direction_and_scale() -> None:
    output = signed_effect_rgb(np.array([[-2.0, 0.0, 2.0]]), 2.0)
    assert output.tolist() == [[[0, 0, 255], [255, 255, 255], [255, 0, 0]]]
    with pytest.raises(ValueError, match="positive"):
        signed_effect_rgb(np.zeros((1, 1)), 0.0)
