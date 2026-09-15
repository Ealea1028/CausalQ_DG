"""GTA5 discovery and non-destructive label conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .cityscapes import label_ids_to_train_ids, read_index_mask


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def discover_files(root: Path, suffixes: set[str] = IMAGE_SUFFIXES) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )


def resolve_flat_payload_root(root: Path) -> Path:
    """Unwrap one redundant archive directory when GTA5 PNGs are nested."""
    if not root.is_dir():
        return root
    direct_files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if direct_files:
        return root
    populated_children = [
        child
        for child in root.iterdir()
        if child.is_dir() and discover_files(child)
    ]
    return populated_children[0] if len(populated_children) == 1 else root


def pair_by_relative_path(image_root: Path, label_root: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Pair images and labels while preserving nested relative paths."""
    images = discover_files(image_root)
    labels = discover_files(label_root, {".png"})
    image_map = {path.relative_to(image_root).with_suffix("").as_posix(): path for path in images}
    label_map = {path.relative_to(label_root).with_suffix("").as_posix(): path for path in labels}
    shared = sorted(image_map.keys() & label_map.keys())
    missing_labels = sorted(image_map.keys() - label_map.keys())
    missing_images = sorted(label_map.keys() - image_map.keys())
    return (
        [(image_map[key], label_map[key]) for key in shared],
        missing_labels,
        missing_images,
    )


def convert_labels(
    raw_label_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Convert GTA5 raw label IDs, preserving the source directory unchanged."""
    raw_labels = discover_files(raw_label_root, {".png"})
    if limit is not None:
        raw_labels = raw_labels[:limit]
    converted_count = 0
    skipped_count = 0
    for source in raw_labels:
        destination = output_root / source.relative_to(raw_label_root)
        if destination.exists() and not overwrite:
            skipped_count += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            raw = read_index_mask(image)
        converted = label_ids_to_train_ids(raw)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        Image.fromarray(converted).save(temporary, format="PNG")
        temporary.replace(destination)
        converted_count += 1
    return {
        "source_count": len(raw_labels),
        "converted_count": converted_count,
        "skipped_existing_count": skipped_count,
    }
