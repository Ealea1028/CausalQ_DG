"""Phase-17 CPU-only real GTA5 gate for frozen strict-DG data routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

from causalq.datasets.strict_source import source_manifest_metadata
from causalq.interventions import StyleInterventionBank
from tools.build_gta5_source_split import PAIR_SHA256
from tools.check_datasets import inspect_pairs
from tools.inspect_gta5_source_structure import _git
from tools.train import PROJECT_ROOT, build_training_datasets, load_config


def inspect_split(dataset, name: str, output: Path) -> dict:
    pairs = [(pair.image, pair.label) for pair in dataset.pairs]
    print(f"inspect_start split={name} samples={len(pairs)}", flush=True)
    result = inspect_pairs(pairs, train_ids=True, scan_limit=0,
                           visualization_dir=output / name, visualization_count=0, seed=0)
    result["ok"] = (result["scanned_label_count"] == len(pairs)
                    and not result["invalid_train_ids"]
                    and result["geometry_mismatch_count"] == 0
                    and result["unreadable_count"] == 0
                    and result["all_ignore_label_count"] == 0)
    print(f"inspect_end split={name} ok={result['ok']}", flush=True)
    return result


def check_preprocessing(train, val) -> dict:
    # Every holdout frame must support its fixed geometry. No target data and
    # no mIoU. Saving/restoring RNG isolates this gate from future training.
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        min_valid = 1.0
        shapes = set()
        for index in range(len(val)):
            before = torch.get_rng_state().clone()
            sample = val[index]
            if not torch.equal(before, torch.get_rng_state()):
                raise ValueError("Holdout preprocessing consumed torch RNG")
            image, label = sample["image"], sample["label"]
            if image.shape[-2:] != label.shape or any(size % 16 for size in label.shape):
                raise ValueError("Invalid holdout input/label/patch geometry")
            if not torch.isfinite(image).all():
                raise ValueError("Non-finite source holdout image")
            min_valid = min(min_valid, float((label != 255).float().mean()))
            shapes.add(tuple(label.shape))
            if (index + 1) % 250 == 0:
                print(f"holdout_transform_checked={index + 1}", flush=True)
        # Fixed source-only samples check train crop and photo paths without
        # loading a backbone or executing an optimizer update.
        random.seed(20261016)
        np.random.seed(20261016)
        torch.manual_seed(20261016)
        bank = StyleInterventionBank()
        photo_change = []
        for index in range(min(32, len(train))):
            sample = train[index]
            image, label = sample["image"], sample["label"]
            if tuple(image.shape) != (3, 512, 512) or tuple(label.shape) != (512, 512):
                raise ValueError("Invalid source training crop geometry")
            photo = bank.photometric_view(image.unsqueeze(0))
            if photo.shape != image.unsqueeze(0).shape or not torch.isfinite(photo).all():
                raise ValueError("Invalid source photometric view")
            photo_change.append(float((photo - image.unsqueeze(0)).abs().mean()))
        if not all(change > 0 for change in photo_change):
            raise ValueError("Photometric source intervention did not change appearance")
        return {"ok": True, "holdout_checked": len(val),
                "holdout_shapes": [list(shape) for shape in sorted(shapes)],
                "holdout_valid_fraction_min": min_valid,
                "train_crop_photo_checked": len(photo_change),
                "photo_mean_absolute_change": sum(photo_change) / len(photo_change),
                "holdout_torch_rng_unchanged": True}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def run_gate(config: dict, data_root: Path, output: Path) -> dict:
    train, val = build_training_datasets(config, data_root)
    pairs = sorted(train.pairs + val.pairs, key=lambda pair: pair.sample_id)
    gta5_root = data_root / "gta5"
    records = [{"id": pair.sample_id, "image": pair.image.relative_to(gta5_root).as_posix(),
                "label": pair.label.relative_to(gta5_root).as_posix(),
                "image_bytes": pair.image.stat().st_size,
                "label_bytes": pair.label.stat().st_size} for pair in pairs]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != PAIR_SHA256:
        raise ValueError("Real GTA5 file IDs/paths/sizes changed since the accepted inventory")
    train_scan, val_scan = inspect_split(train, "train", output), inspect_split(val, "val", output)
    if not (train_scan["ok"] and val_scan["ok"]):
        return {"ok": False, "train_scan": train_scan, "val_scan": val_scan,
                "preprocessing": {"ok": False, "skipped": "source pixel inspection failed"}}
    preprocessing = check_preprocessing(train, val)
    return {"ok": True, "pair_manifest_sha256": digest, "train_scan": train_scan,
            "val_scan": val_scan, "preprocessing": preprocessing}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.is_relative_to(args.data_root.resolve()) or output.is_relative_to(PROJECT_ROOT):
        parser.error("Evidence output must be outside the datasets and source checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 17, "scope": "gta5_strict_source_data_only",
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "executable": sys.executable, "torch": str(torch.__version__),
                              "numpy": np.__version__},
              "training_authorized": False, "optimizer_updates": 0,
              "target_data_accessed": False, **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Source data gate requires clean Git checkout")
        config = load_config(PROJECT_ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml")
        report["config"] = config
        report.update(run_gate(config, args.data_root, output))
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (output / "source_data_report.json").write_bytes(payload)
    (output / "SHA256SUMS").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  source_data_report.json\n", encoding="utf-8")
    print(payload.decode("utf-8"), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
