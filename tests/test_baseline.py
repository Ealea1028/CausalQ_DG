from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from causalq.datasets.segmentation import (
    PairedSegmentationDataset,
    SegmentationPair,
    TrainTransform,
    align_scale_equivalent_label,
)
from causalq.losses import segmentation_cross_entropy
from causalq.metrics import MeanIoU
from causalq.models import BaselineSegmentor, DINOv3Backbone
from causalq.utils.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    trainable_state_dict,
)
from tools.train import load_config


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=2,
            hidden_size=8,
            num_register_tokens=2,
        )
        self.projection = nn.Conv2d(3, 8, kernel_size=2, stride=2)

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        del return_dict
        patch_map = self.projection(pixel_values)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        prefix = patch_tokens.new_zeros(patch_tokens.shape[0], 3, 8)
        hidden = torch.cat((prefix, patch_tokens), dim=1)
        states = (hidden, hidden * 0.5) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=states)


def save_pair(root: Path) -> SegmentationPair:
    image_path = root / "image.png"
    label_path = root / "label.png"
    Image.new("RGB", (8, 6), color=(124, 116, 104)).save(image_path)
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(label_path)
    return SegmentationPair(image_path, label_path, "sample")


def test_scale_equivalent_label_is_aligned() -> None:
    image = Image.new("RGB", (8, 6))
    label = Image.fromarray(np.zeros((3, 4), dtype=np.uint8))

    assert align_scale_equivalent_label(image, label).size == image.size

    incompatible = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="geometry mismatch"):
        align_scale_equivalent_label(image, incompatible)


def test_training_dataset_returns_shared_fixed_geometry(tmp_path: Path) -> None:
    dataset = PairedSegmentationDataset(
        [save_pair(tmp_path)],
        transform=TrainTransform(
            (4, 4), scale_range=(1.0, 1.0), horizontal_flip_probability=0.0
        ),
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 4, 4)
    assert sample["label"].shape == (4, 4)
    assert sample["label"].dtype == torch.int64


def test_training_dataset_reports_corrupt_image_path(tmp_path: Path) -> None:
    pair = save_pair(tmp_path)
    payload = pair.image.read_bytes()
    pair.image.write_bytes(payload[:16])
    dataset = PairedSegmentationDataset([pair])

    with pytest.raises(OSError, match=r"Failed to decode image for sample sample") as error:
        dataset[0]

    assert str(pair.image) in str(error.value)


def test_training_crop_prefers_valid_pixels() -> None:
    torch.manual_seed(0)
    image = Image.new("RGB", (8, 8), color=(124, 116, 104))
    values = np.full((8, 8), 255, dtype=np.uint8)
    values[:, :4] = 0
    label = Image.fromarray(values)
    transform = TrainTransform(
        (4, 4),
        scale_range=(1.0, 1.0),
        horizontal_flip_probability=0.0,
        min_valid_fraction=0.5,
        crop_attempts=20,
    )

    _, cropped_label = transform(image, label)

    assert (cropped_label != 255).float().mean() >= 0.5


def test_training_crop_falls_back_to_a_real_valid_pixel(monkeypatch) -> None:
    image = Image.new("RGB", (8, 8), color=(124, 116, 104))
    values = np.full((8, 8), 255, dtype=np.uint8)
    values[7, 7] = 0
    label = Image.fromarray(values)
    transform = TrainTransform(
        (4, 4),
        scale_range=(1.0, 1.0),
        horizontal_flip_probability=0.0,
        min_valid_fraction=0.5,
        crop_attempts=1,
    )

    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size: torch.zeros(size, dtype=torch.int64),
    )

    _, cropped_label = transform(image, label)

    assert torch.any(cropped_label != 255)
    assert cropped_label[-1, -1] == 0


def test_baseline_outputs_full_resolution_and_only_decoder_trains() -> None:
    backbone = DINOv3Backbone(
        FakeBackbone(), freeze=True, intermediate_indices=(0, 1)
    )
    model = BaselineSegmentor(backbone, decoder_channels=32, dropout=0.0)

    logits = model(torch.randn(2, 3, 8, 8))
    loss = segmentation_cross_entropy(
        logits, torch.zeros(2, 8, 8, dtype=torch.long)
    )
    loss.backward()

    assert logits.shape == (2, 19, 8, 8)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert set(trainable_state_dict(model)) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_checkpoint_excludes_frozen_backbone(tmp_path: Path) -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    model = BaselineSegmentor(backbone, decoder_channels=32, dropout=0.0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    path = tmp_path / "checkpoint.pth"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        iteration=3,
        metadata={"git_sha": "example", "pytorch": torch.__version__},
    )
    payload = load_training_checkpoint(path)

    assert payload["iteration"] == 3
    assert payload["metadata"]["git_sha"] == "example"
    assert str(payload["metadata"]["pytorch"]) == str(torch.__version__)
    assert payload["trainable_model"]
    assert all(name.startswith("decoder.") for name in payload["trainable_model"])


def test_mean_iou_ignores_255_and_averages_present_classes() -> None:
    predictions = torch.tensor([[[0, 1], [1, 0]]])
    labels = torch.tensor([[[0, 1], [0, 255]]])
    metric = MeanIoU(num_classes=2)

    metric.update(predictions, labels)
    result = metric.compute()

    assert result["class_iou"] == pytest.approx([0.5, 0.5])
    assert result["miou"] == pytest.approx(0.5)


def test_segmentation_loss_rejects_all_ignore_batch() -> None:
    with pytest.raises(ValueError, match="no valid pixels"):
        segmentation_cross_entropy(
            torch.randn(1, 19, 2, 2),
            torch.full((1, 2, 2), 255, dtype=torch.long),
        )


def test_phase5_config_disables_future_mechanisms() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/baseline/gta_dinov3l.yaml")

    assert config["experiment"]["phase"] == 5
    assert config["model"]["freeze_backbone"] is True
    assert config["model"]["query"] is False
    assert config["train"]["style"] is False
    assert config["model"]["weights_sha256"].startswith("dcb2e451")
