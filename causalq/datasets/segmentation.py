"""Paired semantic-segmentation datasets and geometry-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .cityscapes import IGNORE_INDEX
from .gta5 import pair_by_relative_path, resolve_flat_payload_root


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SegmentationPair:
    image: Path
    label: Path
    sample_id: str


def _tensorize(image: Image.Image, label: Image.Image) -> tuple[Tensor, Tensor]:
    image_array = np.asarray(image, dtype=np.float32).copy()
    label_array = np.asarray(label, dtype=np.uint8).copy()
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).div_(255.0)
    mean = image_tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image_tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return image_tensor.sub_(mean).div_(std), torch.from_numpy(label_array).long()


def _relative_aspect_error(first: tuple[int, int], second: tuple[int, int]) -> float:
    first_ratio = first[0] / first[1]
    second_ratio = second[0] / second[1]
    return abs(first_ratio - second_ratio) / max(first_ratio, second_ratio)


def align_scale_equivalent_label(image: Image.Image, label: Image.Image) -> Image.Image:
    """Resize a scale-only label mismatch and reject incompatible geometry."""
    if image.size == label.size:
        return label
    if _relative_aspect_error(image.size, label.size) > 0.002:
        raise ValueError(
            f"Image/label geometry mismatch: image={image.size}, label={label.size}"
        )
    return label.resize(image.size, Image.Resampling.NEAREST)


class TrainTransform:
    """Apply shared scale/crop/flip transforms without appearance intervention."""

    def __init__(
        self,
        crop_size: Sequence[int],
        *,
        scale_range: Sequence[float] = (0.5, 2.0),
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if len(crop_size) != 2 or len(scale_range) != 2:
            raise ValueError("crop_size and scale_range must each contain two values")
        self.crop_size = int(crop_size[0]), int(crop_size[1])
        self.scale_range = float(scale_range[0]), float(scale_range[1])
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __call__(self, image: Image.Image, label: Image.Image) -> tuple[Tensor, Tensor]:
        scale = torch.empty(()).uniform_(*self.scale_range).item()
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        label = label.resize((width, height), Image.Resampling.NEAREST)

        crop_height, crop_width = self.crop_size
        pad_width = max(0, crop_width - width)
        pad_height = max(0, crop_height - height)
        if pad_width or pad_height:
            image = ImageOps.expand(
                image,
                border=(0, 0, pad_width, pad_height),
                fill=(124, 116, 104),
            )
            label = ImageOps.expand(
                label,
                border=(0, 0, pad_width, pad_height),
                fill=IGNORE_INDEX,
            )

        max_left = image.width - crop_width
        max_top = image.height - crop_height
        left = int(torch.randint(max_left + 1, ()).item()) if max_left else 0
        top = int(torch.randint(max_top + 1, ()).item()) if max_top else 0
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        label = label.crop(box)

        if torch.rand(()).item() < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return _tensorize(image, label)


class PairedSegmentationDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        pairs: Sequence[SegmentationPair],
        *,
        transform: TrainTransform | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("Segmentation dataset contains no paired samples")
        self.pairs = tuple(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        pair = self.pairs[index]
        with Image.open(pair.image) as image_file:
            image = image_file.convert("RGB")
        with Image.open(pair.label) as label_file:
            label = label_file.convert("L")
        label = align_scale_equivalent_label(image, label)
        if self.transform is None:
            image_tensor, label_tensor = _tensorize(image, label)
        else:
            image_tensor, label_tensor = self.transform(image, label)
        return {"image": image_tensor, "label": label_tensor, "id": pair.sample_id}


def gta5_dataset(root: Path, *, transform: TrainTransform | None) -> PairedSegmentationDataset:
    image_root = resolve_flat_payload_root(root / "images")
    label_root = resolve_flat_payload_root(root / "labels_trainIds")
    paired, missing_labels, missing_images = pair_by_relative_path(image_root, label_root)
    if missing_labels or missing_images:
        raise ValueError(
            f"GTA5 pairing is incomplete: {len(missing_labels)} missing labels, "
            f"{len(missing_images)} missing images"
        )
    pairs = [
        SegmentationPair(image, label, image.relative_to(image_root).with_suffix("").as_posix())
        for image, label in paired
    ]
    return PairedSegmentationDataset(pairs, transform=transform)


def cityscapes_dataset(root: Path, *, split: str) -> PairedSegmentationDataset:
    image_root = root / "leftImg8bit" / split
    label_root = root / "gtFine" / split
    pairs: list[SegmentationPair] = []
    for image in sorted(image_root.rglob("*_leftImg8bit.png")):
        relative = image.relative_to(image_root)
        label_name = image.name.replace(
            "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
        )
        label = label_root / relative.parent / label_name
        if not label.is_file():
            raise FileNotFoundError(f"Cityscapes label is missing for {image}: {label}")
        pairs.append(
            SegmentationPair(
                image,
                label,
                relative.as_posix().removesuffix("_leftImg8bit.png"),
            )
        )
    return PairedSegmentationDataset(pairs)
