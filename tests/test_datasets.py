from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from causalq.datasets.cityscapes import (
    IGNORE_INDEX,
    invalid_train_ids,
    label_ids_to_train_ids,
)
from causalq.datasets.gta5 import convert_labels, pair_by_relative_path
from tools.check_datasets import inspect_cityscapes, inspect_gta5


def save_rgb(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(12, 34, 56)).save(path)


def save_label(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def arguments(**overrides) -> Namespace:
    values = {
        "convert_gta5": False,
        "overwrite_converted": False,
        "conversion_limit": None,
        "label_scan_limit": 0,
        "samples": 1,
        "seed": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cityscapes_label_mapping() -> None:
    raw = np.asarray([[7, 8, 11, 0, 33]], dtype=np.uint8)
    converted = label_ids_to_train_ids(raw)
    assert converted.tolist() == [[0, 1, 2, IGNORE_INDEX, 18]]
    assert invalid_train_ids(converted) == set()


def test_gta5_conversion_preserves_raw_labels(tmp_path: Path) -> None:
    raw_root = tmp_path / "labels"
    output_root = tmp_path / "labels_trainIds"
    raw = np.asarray([[7, 8], [26, 0]], dtype=np.uint8)
    source = raw_root / "00001.png"
    save_label(source, raw)

    result = convert_labels(raw_root, output_root)

    assert result["converted_count"] == 1
    assert np.array_equal(np.asarray(Image.open(source)), raw)
    converted = np.asarray(Image.open(output_root / "00001.png"))
    assert converted.tolist() == [[0, 1], [13, IGNORE_INDEX]]


def test_gta5_inspection_pairs_and_visualizes(tmp_path: Path) -> None:
    root = tmp_path / "gta5"
    save_rgb(root / "images/00001.png", (2, 2))
    save_label(root / "labels/00001.png", np.asarray([[7, 8], [26, 0]]))
    config = {
        "images": "images",
        "labels_original": "labels",
        "labels_train_ids": "labels_trainIds",
    }

    result = inspect_gta5(
        root, config, arguments(convert_gta5=True), tmp_path / "visualizations"
    )

    assert result["ok"] is True
    assert result["paired_count"] == 1
    assert result["unique_label_ids"] == [0, 1, 13, 255]
    assert len(result["visualizations"]) == 1


def test_cityscapes_inspection_detects_valid_pair(tmp_path: Path) -> None:
    root = tmp_path / "cityscapes"
    image = root / "leftImg8bit/train/demo/demo_000001_000001_leftImg8bit.png"
    label = root / "gtFine/train/demo/demo_000001_000001_gtFine_labelTrainIds.png"
    save_rgb(image, (2, 2))
    save_label(label, np.asarray([[0, 1], [18, 255]]))
    # Official test images do not have public semantic ground truth and must
    # not be treated as missing-label failures.
    save_rgb(
        root / "leftImg8bit/test/demo/demo_000002_000002_leftImg8bit.png",
        (2, 2),
    )

    result = inspect_cityscapes(
        root,
        {"images": "leftImg8bit", "labels_train_ids": "gtFine"},
        arguments(),
        tmp_path / "visualizations",
    )

    assert result["ok"] is True
    assert result["paired_count"] == 1
    assert result["invalid_train_ids"] == []
    assert result["checked_splits"] == ["train", "val"]
    assert result["image_split_counts"]["train"] == 1
    assert result["image_split_counts"]["test"] == 1
    assert result["label_split_counts"]["train"] == 1


def test_relative_pairing_reports_missing_files(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    save_rgb(image_root / "a.png")
    save_rgb(image_root / "b.png")
    save_label(label_root / "a.png", np.zeros((6, 8), dtype=np.uint8))

    pairs, missing_labels, missing_images = pair_by_relative_path(
        image_root, label_root
    )

    assert len(pairs) == 1
    assert missing_labels == ["b"]
    assert missing_images == []
