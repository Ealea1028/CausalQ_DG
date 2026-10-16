"""Strict-DG GTA5 membership and deterministic source holdout preprocessing."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageOps

from causalq.protocols.source_split import manifest_bytes, source_split, verify_membership
from .gta5 import discover_files
from .segmentation import (PairedSegmentationDataset, SegmentationPair,
                           TrainTransform, _tensorize)


MANIFEST_HASHES = {
    "gta5_source_train.txt": "c33c1316eb1e547199aa0be79f749d6af9a7c8a39d6ab40313e410b1fe021ac9",
    "gta5_source_val.txt": "57dec87d8081f50162e29738a7052cd014404d23bf6cd2268c77b23bc0f011cb",
}
MANIFEST_DIR = Path(__file__).resolve().parents[2] / "manifests"


def load_membership(directory: Path = MANIFEST_DIR) -> tuple[list[str], list[str]]:
    memberships = []
    for name, expected_hash in MANIFEST_HASHES.items():
        payload = (directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Frozen membership hash mismatch: {name}")
        ids = payload.decode("ascii").splitlines()
        if manifest_bytes(ids) != payload:
            raise ValueError(f"Noncanonical membership: {name}")
        memberships.append(ids)
    train, val = memberships
    verify_membership(train + val, train, val)
    if (train, val) != source_split(train + val):
        raise ValueError("Frozen membership does not reproduce the fixed split rule")
    return train, val


class SourceEvalTransform:
    """Resize the full source frame preserving aspect, then pad to patch size.

    No crop, flip, photometric change or RNG; labels use nearest interpolation
    and ignore padding. Metrics refer to the resized GT, not original-resolution GT.
    """

    def __init__(self, long_edge: int = 1024, patch_size: int = 16) -> None:
        if long_edge <= 0 or patch_size <= 0:
            raise ValueError("Source evaluation dimensions must be positive")
        self.long_edge, self.patch_size = long_edge, patch_size

    def __call__(self, image, label):
        scale = self.long_edge / max(image.size)
        width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        label = label.resize((width, height), Image.Resampling.NEAREST)
        pad_w = math.ceil(width / self.patch_size) * self.patch_size - width
        pad_h = math.ceil(height / self.patch_size) * self.patch_size - height
        border = (0, 0, pad_w, pad_h)
        image = ImageOps.expand(image, border=border, fill=(124, 116, 104))
        label = ImageOps.expand(label, border=border, fill=255)
        return _tensorize(image, label)


def _source_index(root: Path, suffixes: set[str]) -> dict[str, Path]:
    result = {}
    for path in discover_files(root, suffixes):
        if path.stem in result:
            raise ValueError(f"Duplicate source ID: {path.stem}")
        # Explicit payload roots are pinned by the accepted inventory. Do not
        # traverse symlinked ancestors or silently unwrap scene directories.
        current = path
        while current != root.parent:
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                raise ValueError(f"Source path is a link/junction: {current}")
            current = current.parent
        if path.parent != root:
            raise ValueError(f"Frozen inventory expects flat source payload: {path}")
        result[path.stem] = path
    return result


def build_source_datasets(gta5_root: Path, train_transform: TrainTransform,
                          eval_transform: SourceEvalTransform,
                          manifest_dir: Path = MANIFEST_DIR):
    train_ids, val_ids = load_membership(manifest_dir)
    # Scan exactly the accepted GTA5 image/train-ID-label payloads. No common
    # data-root traversal, raw-label fallback or target dataset construction.
    image_root = gta5_root / "images" / "images"
    label_root = gta5_root / "labels_trainIds"
    for path in (gta5_root, gta5_root / "images", image_root, label_root):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"Source root is a link/junction: {path}")
    images = _source_index(image_root, {".png", ".jpg", ".jpeg"})
    labels = _source_index(label_root, {".png"})
    expected = set(train_ids + val_ids)
    if set(images) != expected or set(labels) != expected:
        raise ValueError("Actual GTA5 payload is not an exact match to frozen membership")

    def dataset(ids, transform):
        return PairedSegmentationDataset(
            [SegmentationPair(images[key], labels[key], key) for key in ids], transform=transform)

    return dataset(train_ids, train_transform), dataset(val_ids, eval_transform)


def source_manifest_metadata() -> dict:
    return {"manifest_sha256": MANIFEST_HASHES, "train_count": 22469, "val_count": 2497,
            "source_validation": "gta5_holdout", "target_data_accessed": False}
