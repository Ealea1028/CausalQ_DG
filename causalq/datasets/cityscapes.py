"""Backward-compatible Cityscapes label helpers and offline conversion."""

from pathlib import Path
from PIL import Image
from .label_space import (IGNORE_INDEX, NUM_CLASSES, LABEL_ID_TO_TRAIN_ID,
                          TRAIN_ID_PALETTE, read_index_mask, label_ids_to_train_ids,
                          invalid_train_ids, colorize_train_ids)


def convert_label_files(
    root: Path, *, overwrite: bool = False, limit: int | None = None
) -> dict[str, int]:
    """Create train-ID masks beside official ``labelIds`` masks."""
    raw_files = sorted(root.rglob("*_gtFine_labelIds.png")) if root.is_dir() else []
    if limit is not None:
        raw_files = raw_files[:limit]
    converted_count = 0
    skipped_count = 0
    for source in raw_files:
        destination = source.with_name(
            source.name.replace("_gtFine_labelIds.png", "_gtFine_labelTrainIds.png")
        )
        if destination.exists() and not overwrite:
            skipped_count += 1
            continue
        with Image.open(source) as image_file:
            raw = read_index_mask(image_file)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        Image.fromarray(label_ids_to_train_ids(raw)).save(temporary, format="PNG")
        temporary.replace(destination)
        converted_count += 1
    return {
        "source_count": len(raw_files),
        "converted_count": converted_count,
        "skipped_existing_count": skipped_count,
    }
