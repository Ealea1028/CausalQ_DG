"""Inspect DGSS datasets and optionally convert GTA5 labels safely."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import yaml
from PIL import Image

# Preserve the documented `python tools/check_datasets.py` entry point even
# before the repository has been installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.datasets.cityscapes import colorize_train_ids, invalid_train_ids
from causalq.datasets.gta5 import convert_labels, discover_files, pair_by_relative_path


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or "datasets" not in manifest:
        raise ValueError(f"Invalid data manifest: {path}")
    return manifest


def canonical_city_image(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative.removesuffix("_leftImg8bit.png")


def canonical_city_label(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative.removesuffix("_gtFine_labelTrainIds.png")


def pair_with_keys(
    image_paths: list[Path],
    label_paths: list[Path],
    image_key: Callable[[Path], str],
    label_key: Callable[[Path], str],
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    images = {image_key(path): path for path in image_paths}
    labels = {label_key(path): path for path in label_paths}
    shared = sorted(images.keys() & labels.keys())
    return (
        [(images[key], labels[key]) for key in shared],
        sorted(images.keys() - labels.keys()),
        sorted(labels.keys() - images.keys()),
    )


def save_visualizations(
    pairs: list[tuple[Path, Path]], output_dir: Path, count: int, seed: int
) -> list[str]:
    if count <= 0 or not pairs:
        return []
    rng = random.Random(seed)
    selected = rng.sample(pairs, min(count, len(pairs)))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for index, (image_path, label_path) in enumerate(selected):
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(label_path) as label_file:
            label = np.asarray(label_file.convert("L"))
        colored = Image.fromarray(colorize_train_ids(label), mode="RGB")
        if colored.size != image.size:
            continue
        canvas = Image.new("RGB", (image.width * 2, image.height))
        canvas.paste(image, (0, 0))
        canvas.paste(colored, (image.width, 0))
        destination = output_dir / f"{index:02d}_{image_path.stem}.png"
        canvas.save(destination)
        outputs.append(str(destination))
    return outputs


def inspect_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    train_ids: bool,
    scan_limit: int,
    visualization_dir: Path,
    visualization_count: int,
    seed: int,
) -> dict:
    if scan_limit == 0 or scan_limit >= len(pairs):
        scan_pairs = pairs
        scan_scope = "all"
    else:
        scan_pairs = random.Random(seed).sample(pairs, scan_limit)
        scan_scope = f"random_{scan_limit}"
    unique_ids: set[int] = set()
    invalid_ids: set[int] = set()
    shape_mismatches: list[dict[str, object]] = []
    unreadable: list[dict[str, str]] = []
    for image_path, label_path in scan_pairs:
        try:
            with Image.open(image_path) as image_file:
                image_size = image_file.size
            with Image.open(label_path) as label_file:
                label = np.asarray(label_file.convert("L"))
                label_size = label_file.size
            unique_ids.update(int(value) for value in np.unique(label))
            if train_ids:
                invalid_ids.update(invalid_train_ids(label))
            if image_size != label_size:
                shape_mismatches.append(
                    {
                        "image": str(image_path),
                        "label": str(label_path),
                        "image_size": image_size,
                        "label_size": label_size,
                    }
                )
        except Exception as exc:
            unreadable.append({"path": str(label_path), "error": str(exc)})

    visualizations: list[str] = []
    if train_ids and not invalid_ids:
        visualizations = save_visualizations(
            pairs, visualization_dir, visualization_count, seed
        )
    return {
        "scanned_label_count": len(scan_pairs),
        "scan_scope": scan_scope,
        "unique_label_ids": sorted(unique_ids),
        "invalid_train_ids": sorted(invalid_ids),
        "shape_mismatches": shape_mismatches[:20],
        "shape_mismatch_count": len(shape_mismatches),
        "unreadable": unreadable[:20],
        "unreadable_count": len(unreadable),
        "visualizations": visualizations,
    }


def inspect_gta5(
    dataset_root: Path,
    config: dict,
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    image_root = dataset_root / config.get("images", "images")
    raw_label_root = dataset_root / config.get("labels_original", "labels")
    train_label_root = dataset_root / config.get("labels_train_ids", "labels_trainIds")

    conversion = None
    if args.convert_gta5:
        if not raw_label_root.is_dir():
            raise FileNotFoundError(f"GTA5 raw label directory not found: {raw_label_root}")
        conversion = convert_labels(
            raw_label_root,
            train_label_root,
            overwrite=args.overwrite_converted,
            limit=args.conversion_limit,
        )

    train_label_files = discover_files(train_label_root, {".png"})
    using_train_ids = bool(train_label_files)
    selected_label_root = train_label_root if using_train_ids else raw_label_root
    pairs, missing_labels, missing_images = pair_by_relative_path(
        image_root, selected_label_root
    )
    inspection = inspect_pairs(
        pairs,
        train_ids=using_train_ids,
        scan_limit=args.label_scan_limit,
        visualization_dir=output_root / "gta5",
        visualization_count=args.samples,
        seed=args.seed,
    )
    ok = bool(pairs) and not missing_labels and not missing_images
    ok = ok and inspection["shape_mismatch_count"] == 0
    ok = ok and inspection["unreadable_count"] == 0
    ok = ok and using_train_ids and not inspection["invalid_train_ids"]
    return {
        "ok": ok,
        "root": str(dataset_root),
        "image_count": len(discover_files(image_root)),
        "label_count": len(discover_files(selected_label_root, {".png"})),
        "paired_count": len(pairs),
        "label_space": "cityscapes_train_ids" if using_train_ids else "raw_label_ids",
        "needs_conversion": not using_train_ids,
        "missing_label_count": len(missing_labels),
        "missing_image_count": len(missing_images),
        "missing_label_examples": missing_labels[:20],
        "missing_image_examples": missing_images[:20],
        "conversion": conversion,
        **inspection,
    }


def inspect_cityscapes(
    dataset_root: Path,
    config: dict,
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    image_root = dataset_root / config.get("images", "leftImg8bit")
    label_root = dataset_root / config.get("labels_train_ids", "gtFine")
    all_image_paths = (
        sorted(image_root.rglob("*_leftImg8bit.png")) if image_root.is_dir() else []
    )
    all_label_paths = (
        sorted(label_root.rglob("*_gtFine_labelTrainIds.png"))
        if label_root.is_dir()
        else []
    )
    checked_splits = ("train", "val")
    image_paths = [
        path
        for path in all_image_paths
        if path.relative_to(image_root).parts[0] in checked_splits
    ]
    label_paths = [
        path
        for path in all_label_paths
        if path.relative_to(label_root).parts[0] in checked_splits
    ]
    image_split_counts = {
        split: sum(
            1
            for path in all_image_paths
            if path.relative_to(image_root).parts
            and path.relative_to(image_root).parts[0] == split
        )
        for split in ("train", "val", "test")
    }
    label_split_counts = {
        split: sum(
            1
            for path in all_label_paths
            if path.relative_to(label_root).parts
            and path.relative_to(label_root).parts[0] == split
        )
        for split in ("train", "val", "test")
    }
    pairs, missing_labels, missing_images = pair_with_keys(
        image_paths,
        label_paths,
        lambda path: canonical_city_image(path, image_root),
        lambda path: canonical_city_label(path, label_root),
    )
    inspection = inspect_pairs(
        pairs,
        train_ids=True,
        scan_limit=args.label_scan_limit,
        visualization_dir=output_root / "cityscapes",
        visualization_count=args.samples,
        seed=args.seed,
    )
    ok = bool(pairs) and not missing_labels and not missing_images
    ok = ok and inspection["shape_mismatch_count"] == 0
    ok = ok and inspection["unreadable_count"] == 0
    ok = ok and not inspection["invalid_train_ids"]
    return {
        "ok": ok,
        "root": str(dataset_root),
        "checked_splits": list(checked_splits),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "image_split_counts": image_split_counts,
        "label_split_counts": label_split_counts,
        "paired_count": len(pairs),
        "label_space": "cityscapes_train_ids",
        "missing_label_count": len(missing_labels),
        "missing_image_count": len(missing_images),
        "missing_label_examples": missing_labels[:20],
        "missing_image_examples": missing_images[:20],
        **inspection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data_manifest.yaml"
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["gta5", "cityscapes"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--label-scan-limit",
        type=int,
        default=100,
        help="Number of labels to scan per dataset; use 0 to scan all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--convert-gta5", action="store_true")
    parser.add_argument("--overwrite-converted", action="store_true")
    parser.add_argument("--conversion-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    default_root = manifest.get("default_root", "/root/autodl-tmp/datasets")
    data_root = args.data_root or Path(os.getenv("CAUSALQ_DATA_ROOT", default_root))
    output_default = Path(
        os.getenv("CAUSALQ_OUTPUT_ROOT", "/root/autodl-tmp/outputs/CausalQ_DG")
    ) / "data_check"
    output_root = args.output_dir or output_default

    report: dict[str, object] = {
        "ok": True,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "datasets": {},
    }
    dataset_configs = manifest["datasets"]
    for name in args.datasets:
        if name not in dataset_configs:
            report["datasets"][name] = {"ok": False, "error": "not in manifest"}
            report["ok"] = False
            continue
        config = dataset_configs[name]
        dataset_root = data_root / config["relative_root"]
        try:
            if name == "gta5":
                result = inspect_gta5(dataset_root, config, args, output_root)
            elif name == "cityscapes":
                result = inspect_cityscapes(dataset_root, config, args, output_root)
            else:
                result = {
                    "ok": dataset_root.is_dir(),
                    "root": str(dataset_root),
                    "note": "Full adapter validation is not implemented in Phase 3.",
                }
        except Exception as exc:
            result = {"ok": False, "root": str(dataset_root), "error": str(exc)}
        report["datasets"][name] = result
        if not result["ok"]:
            report["ok"] = False

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
