"""Verify the pinned Phase-17 inventory and export candidate source manifests.

No dataset, image/label, weight, GPU or target access. Standard library only.
Candidates must return to local Git before any strict-DG training route uses them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import zipfile

from causalq.protocols.source_split import SPLIT_SALT, manifest_bytes, source_split, validate_ids
from tools.inspect_gta5_source_structure import _git


INVENTORY_SHA256 = "7224adc8aaafbcaac412aa5ee3904c11ba10d3c81aac7dffea231259e21e0d25"
PRODUCER_SHA = "f41dbeea92d67bffc571e2c3ef103de7b3693e00"
PAIR_SHA256 = "92a67a0e32a297d2f41618c6919a0573cfde6ec23890b5dd9a6ae7353842662f"
PAIR_COUNT = 24966
ROOT = "/root/autodl-tmp/datasets/gta5"


def verify_inventory(payload: bytes) -> dict:
    if hashlib.sha256(payload).hexdigest() != INVENTORY_SHA256:
        raise ValueError("Saved inventory SHA256 does not match operator evidence")
    report = json.loads(payload)
    expected = {
        "schema_version": 1, "phase": 17, "git_sha": PRODUCER_SHA, "git_status": "",
        "ok": True, "scope": "gta5_source_filesystem_only", "root": ROOT,
        "resolved_image_root": ROOT + "/images/images",
        "resolved_label_root": ROOT + "/labels_trainIds",
        "image_count": PAIR_COUNT, "label_count": PAIR_COUNT, "paired_count": PAIR_COUNT,
        "missing_labels": [], "missing_images": [], "duplicate_image_ids": {},
        "duplicate_label_ids": {}, "parent_directory_counts": {".": PAIR_COUNT},
        "filename_pattern_counts": {"{digits}": PAIR_COUNT}, "metadata_candidates": [],
        "target_data_accessed": False, "images_decoded": False,
        "label_values_validated": False, "training_authorized": False,
        "split_status": "pending_metadata_review", "pair_manifest_sha256": PAIR_SHA256,
    }
    for key, value in expected.items():
        if key not in report or type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"Inventory field disagrees with reviewed evidence: {key}")
    pairs = report["pairs"]
    if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
        raise ValueError("Incomplete inventory pair records")
    digest = hashlib.sha256(json.dumps(pairs, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if digest.hexdigest() != PAIR_SHA256:
        raise ValueError("Inventory pair digest mismatch")
    ids = validate_ids(pair["id"] for pair in pairs)
    if ids != [pair["id"] for pair in pairs]:
        raise ValueError("Inventory IDs must be in canonical lexical order")
    for pair in pairs:
        for key, parent, suffixes in (("image", "images/images", {".png", ".jpg", ".jpeg"}),
                                      ("label", "labels_trainIds", {".png"})):
            path = PurePosixPath(pair[key])
            if (str(path) != pair[key] or path.parent.as_posix() != parent
                    or path.stem != pair["id"] or path.suffix.lower() not in suffixes):
                raise ValueError(f"Unsafe or unexpected {key} path")
            size = pair[key + "_bytes"]
            if type(size) is not int or size <= 0:
                raise ValueError(f"Invalid {key} size")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-val-fraction", default="0.1")
    args = parser.parse_args(argv)
    # Never overwrite input evidence or earlier success/failure directories.
    output = args.output_dir.resolve()
    if args.inventory.resolve().is_relative_to(output):
        parser.error("output-dir must not contain the immutable input inventory")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "phase": 17, "ok": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": INVENTORY_SHA256,
        "inventory_producer_git_sha": PRODUCER_SHA,
        "pair_manifest_sha256": PAIR_SHA256,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform()},
        "target_data_accessed": False, "dataset_files_opened": False,
        "frozen_in_git": False, "training_authorized": False,
    }
    files = {}
    try:
        report["git_sha"] = _git("rev-parse", "HEAD")
        report["git_status"] = _git("status", "--porcelain")
        if report["git_status"]:
            raise ValueError("Export requires a clean Git checkout")
        inventory = verify_inventory(args.inventory.read_bytes())
        ids = [pair["id"] for pair in inventory["pairs"]]
        train, val = source_split(ids, args.source_val_fraction)
        files = {"gta5_source_train.txt": manifest_bytes(train),
                 "gta5_source_val.txt": manifest_bytes(val)}
        for name, content in files.items():
            (output / name).write_bytes(content)
        report.update(
            ok=True, split_status="candidate_pending_local_git_freeze",
            method="sha256_ranked_exact_filename_fallback_v1", salt=SPLIT_SALT,
            source_val_fraction=str(Decimal(args.source_val_fraction)),
            validation_count_rule="ceil(N * source_val_fraction)",
            total_count=len(ids), train_count=len(train), val_count=len(val),
            overlap_count=0, exact_partition=True,
            manifest_sha256={name: hashlib.sha256(content).hexdigest()
                             for name, content in files.items()},
            limitation="No sequence/scene metadata: correlated frames/scenes may cross this filename split.",
        )
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    report_payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    files["split_report.json"] = report_payload
    (output / "split_report.json").write_bytes(report_payload)
    checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                        for name, data in sorted(files.items()))
    (output / "SHA256SUMS").write_bytes(checksums.encode("ascii"))
    # Small metadata-only return bundle; raw inventory and datasets are excluded.
    if report["ok"]:
        with zipfile.ZipFile(output / "source_split_export.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(files.items()):
                archive.writestr(name, data)
            archive.writestr("SHA256SUMS", checksums)
    print(report_payload.decode("utf-8"), end="")
    print(f"output_dir={output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
