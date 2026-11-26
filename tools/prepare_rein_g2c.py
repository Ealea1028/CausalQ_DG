"""Prepare the official Rein GTA5-to-Cityscapes configuration on AutoDL."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from causalq.datasets.gta5 import pair_by_relative_path, resolve_flat_payload_root
from causalq.datasets.label_space import invalid_train_ids, label_ids_to_train_ids, read_index_mask
from causalq.datasets.segmentation import align_scale_equivalent_label


UPSTREAM_CONFIG = "configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py"
CITY_IMAGE_SUFFIX = "_leftImg8bit.png"
CITY_TRAIN_SUFFIX = "_gtFine_labelTrainIds.png"
CITY_RAW_SUFFIX = "_gtFine_labelIds.png"


def prepare_mask(image_path: Path, label_path: Path, output: Path, *, raw_city=False, gta=False):
    with Image.open(image_path) as image, Image.open(label_path) as label:
        if not gta and label.size != image.size:
            raise ValueError(f"Cityscapes image/label sizes differ: {image_path}, {label_path}")
        if raw_city:
            result = Image.fromarray(label_ids_to_train_ids(read_index_mask(label)))
            operation = "labelIds_to_trainIds"
        else:
            colors = label.getcolors(256) if label.mode in {"L", "P"} else None
            if colors is None:
                bad = invalid_train_ids(read_index_mask(label))
            else:
                bad = {value for _, value in colors if value > 18 and value != 255}
            if bad:
                raise ValueError(f"Invalid train IDs {sorted(bad)}: {label_path}")
            if label.size == image.size and label.mode == "L":
                result, operation = None, "linked"
            else:
                result = Image.fromarray(read_index_mask(label))
                operation = "index_png"
                if result.size != image.size:
                    try:
                        result = align_scale_equivalent_label(image, result)
                    except ValueError as exc:
                        raise ValueError(f"{image_path}: {exc}") from exc
                    operation = "nearest_alignment"
        output.parent.mkdir(parents=True, exist_ok=True)
        if result is None:
            output.symlink_to(label_path.resolve())
        else:
            result.save(output)
        return operation


def prepare_data(data: Path, city_manifest: Path, out: Path):
    gta_root = data / "gta5"
    images = resolve_flat_payload_root(gta_root / "images")
    pairs, missing_labels, missing_images = pair_by_relative_path(
        images, resolve_flat_payload_root(gta_root / "labels_trainIds")
    )
    if missing_labels or missing_images or len(pairs) != 24966 or any(
        image.suffix != ".png" or label.suffix != ".png" for image, label in pairs
    ):
        raise ValueError(f"Expected 24,966 GTA5 PNG pairs; paired={len(pairs)}, "
                         f"missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}")

    city_root = data / "cityscapes"
    city_images = city_root / "leftImg8bit" / "val"
    city_labels = city_root / "gtFine" / "val"
    ids = city_manifest.read_text(encoding="utf-8").splitlines()
    found = {path.relative_to(city_images).as_posix().removesuffix(CITY_IMAGE_SUFFIX)
             for path in city_images.rglob(f"*{CITY_IMAGE_SUFFIX}")}
    if len(ids) != 500 or len(set(ids)) != 500 or found != set(ids):
        raise ValueError(f"Cityscapes val membership mismatch: found={len(found)}, expected=500")

    report = {"gta5_pairs": len(pairs), "cityscapes_pairs": len(ids),
              "gta5_operations": {}, "cityscapes_operations": {}, "gta5_resized_ids": []}
    stage = out / "labels"
    stage.mkdir()
    with (out / "pairs.tsv").open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("domain\tsample_id\timage\toriginal_label\teffective_label\toperation\n")
        for i, (image, label) in enumerate(pairs, 1):
            rel = image.relative_to(images)
            effective = stage / "gta5" / rel
            operation = prepare_mask(image, label, effective, gta=True)
            sample_id = rel.with_suffix("").as_posix()
            manifest.write(f"gta5\t{sample_id}\t{image}\t{label}\t{effective}\t{operation}\n")
            counts = report["gta5_operations"]
            counts[operation] = counts.get(operation, 0) + 1
            if operation == "nearest_alignment":
                report["gta5_resized_ids"].append(sample_id)
            if i % 2000 == 0:
                print(f"GTA5 aligned: {i}/{len(pairs)}", flush=True)
        for sample_id in ids:
            image = city_images / (sample_id + CITY_IMAGE_SUFFIX)
            label = city_labels / (sample_id + CITY_TRAIN_SUFFIX)
            raw = not label.is_file()
            if raw:
                label = city_labels / (sample_id + CITY_RAW_SUFFIX)
            effective = stage / "cityscapes" / (sample_id + CITY_TRAIN_SUFFIX)
            operation = prepare_mask(image, label, effective, raw_city=raw)
            manifest.write(f"cityscapes\t{sample_id}\t{image}\t{label}\t{effective}\t{operation}\n")
            counts = report["cityscapes_operations"]
            counts[operation] = counts.get(operation, 0) + 1
    return images, city_images, stage, report


def configure(config: dict, images: Path, city_images: Path, labels: Path, backbone: Path):
    cfg = deepcopy(config)
    source = cfg["train_dataloader"]["dataset"]
    source.update(data_root="", img_suffix=".png", seg_map_suffix=".png",
                  data_prefix=dict(img_path=str(images), seg_map_path=str(labels / "gta5")))
    template = deepcopy(cfg["val_dataloader"])
    city = next(dataset for dataset in template["dataset"]["datasets"]
                if dataset["data_prefix"]["img_path"].rstrip("/") == "leftImg8bit/val")
    city.update(data_root="", img_suffix=CITY_IMAGE_SUFFIX, seg_map_suffix=CITY_TRAIN_SUFFIX,
                data_prefix=dict(img_path=str(city_images), seg_map_path=str(labels / "cityscapes")))
    for dataset in (source, city):
        dataset["reduce_zero_label"] = False
        for step in dataset["pipeline"]:
            if step["type"] == "LoadAnnotations":
                step["reduce_zero_label"] = False
    template["dataset"] = city
    for split in ("val", "test"):
        cfg[f"{split}_dataloader"] = deepcopy(template)
        cfg[f"{split}_evaluator"] = dict(type="IoUMetric", iou_metrics=["mIoU"], ignore_index=255)
    cfg["model"]["backbone"]["init_cfg"]["checkpoint"] = str(backbone)
    cfg.update(randomness=dict(seed=0), load_from=None, resume=False)
    if (cfg["train_dataloader"]["batch_size"], cfg["train_cfg"]["max_iters"],
        cfg["optim_wrapper"].get("accumulative_counts", 1)) != (4, 40000, 1):
        raise ValueError("Expected batch 4 and 40,000 optimizer updates")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("rein", "data", "city-manifest", "backbone", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()

    import rein  # noqa: F401  # Imported from the pinned upstream checkout on AutoDL.
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import DATASETS

    base = Config.fromfile(args.rein / UPSTREAM_CONFIG).to_dict()
    images, city, stage, report = prepare_data(args.data, args.city_manifest, args.out)
    config = configure(base, images, city, stage, args.backbone)
    init_default_scope("mmseg")
    for key, expected in (("train", 24966), ("val", 500), ("test", 500)):
        dataset = DATASETS.build(config[f"{key}_dataloader"]["dataset"])
        if len(dataset) != expected:
            raise ValueError(f"Rein {key}: expected {expected}, got {len(dataset)}")
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]["data_samples"]
            if key != "train" and tuple(sample.gt_sem_seg.data.shape[-2:]) != tuple(sample.ori_shape):
                raise ValueError(f"Rein {key}: target mask no longer matches original geometry")
    Config(config).dump(str(args.out / "gta_cityscapes.py"))
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("REIN_DATA_OK: 24966 GTA5 and 500 Cityscapes val pairs", flush=True)


if __name__ == "__main__":
    main()
