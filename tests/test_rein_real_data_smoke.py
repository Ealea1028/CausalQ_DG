import pytest
import torch

from tools.rein_real_data_smoke import normalized_rgb_to_bgr255, center_target_crop
from causalq.datasets.segmentation import IMAGENET_MEAN, IMAGENET_STD


def test_normalization_bridge_preserves_rgb_channels_without_quantization():
    rgb = torch.rand(3, 8, 8) * 255
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    normalized = (rgb / 255 - mean) / std
    bgr = normalized_rgb_to_bgr255(normalized)
    assert torch.allclose(bgr, rgb[[2, 1, 0]], atol=4e-5)
    recovered = (bgr[[2, 1, 0]] - mean * 255) / (std * 255)
    assert (recovered - normalized).abs().max() < 1e-5


def test_bridge_rejects_nonfinite_and_out_of_range():
    with pytest.raises(ValueError):
        normalized_rgb_to_bgr255(torch.full((3, 8, 8), float("nan")))
    with pytest.raises(ValueError):
        normalized_rgb_to_bgr255(torch.full((3, 8, 8), 100.))


def test_target_crop_preserves_alignment_and_label_ids():
    label = torch.zeros(32, 64, dtype=torch.long)
    label[:, 16:48] = 13
    image = label.float().expand(3, -1, -1)
    item = center_target_crop(dict(image=image, label=label, id="test"))
    assert item["image"].shape == (3, 512, 512)
    assert item["label"].shape == (512, 512)
    assert item["label"].unique().tolist() == [13]
    assert item["id"] == "test"
