"""Verify the returned Phase-17 metadata bundle before creating Git manifests.

Accept the exact ZIP or its extracted files with pinned manifest hashes.
No datasets or GPU needed; extracted input does not claim ZIP-byte verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from causalq.protocols.source_split import SPLIT_SALT, manifest_bytes, source_split, verify_membership
from tools.build_gta5_source_split import INVENTORY_SHA256, PAIR_SHA256, PRODUCER_SHA


EXPORT_SHA = "57a724428b60d0f6636db056071bc620d45f45e0"
BUNDLE_SHA256 = "a555c740546a992ec72199f314a750699acf3782ceda1c98e9cf8908206b96db"
MANIFEST_SHA256 = {
    "gta5_source_train.txt": "c33c1316eb1e547199aa0be79f749d6af9a7c8a39d6ab40313e410b1fe021ac9",
    "gta5_source_val.txt": "57dec87d8081f50162e29738a7052cd014404d23bf6cd2268c77b23bc0f011cb",
}
COUNTS = (24966, 22469, 2497)


def verify_bundle(path: Path) -> tuple[dict[str, bytes], dict]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != BUNDLE_SHA256:
        raise ValueError("ZIP SHA256 disagrees with the operator-returned evidence")
    names = {*MANIFEST_SHA256, "split_report.json", "SHA256SUMS"}
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != len(names) or set(archive.namelist()) != names:
            raise ValueError("Unexpected or duplicate ZIP members")
        if any(item.file_size > 1024 * 1024 for item in archive.infolist()):
            raise ValueError("Unexpectedly large manifest metadata")
        contents = {name: archive.read(name) for name in names}
    return verify_contents(contents)


def verify_directory(path: Path) -> tuple[dict[str, bytes], dict]:
    names = {*MANIFEST_SHA256, "split_report.json", "SHA256SUMS"}
    if path.is_symlink() or {p.name for p in path.iterdir()} != names:
        raise ValueError("Unexpected extracted directory members")
    contents = {}
    for name in names:
        member = path / name
        if member.is_symlink() or not member.is_file() or member.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid extracted metadata file")
        contents[name] = member.read_bytes()
    return verify_contents(contents)


def verify_contents(contents: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    names = {*MANIFEST_SHA256, "split_report.json", "SHA256SUMS"}
    checksums = {}
    for line in contents["SHA256SUMS"].decode("ascii").splitlines():
        digest, name = line.split()
        if name in checksums:
            raise ValueError("Duplicate checksum entry")
        checksums[name] = digest
    if set(checksums) != names - {"SHA256SUMS"}:
        raise ValueError("Incomplete checksum manifest")
    for name, digest in checksums.items():
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    for name, digest in MANIFEST_SHA256.items():
        if checksums[name] != digest:
            raise ValueError(f"Operator-pinned manifest hash mismatch: {name}")
    report = json.loads(contents["split_report.json"])
    expected = {
        "schema_version": 1, "phase": 17, "ok": True,
        "git_sha": EXPORT_SHA, "git_status": "", "inventory_sha256": INVENTORY_SHA256,
        "inventory_producer_git_sha": PRODUCER_SHA, "pair_manifest_sha256": PAIR_SHA256,
        "method": "sha256_ranked_exact_filename_fallback_v1", "salt": SPLIT_SALT,
        "source_val_fraction": "0.1", "validation_count_rule": "ceil(N * source_val_fraction)",
        "total_count": COUNTS[0], "train_count": COUNTS[1], "val_count": COUNTS[2],
        "overlap_count": 0, "exact_partition": True,
        "split_status": "candidate_pending_local_git_freeze", "frozen_in_git": False,
        "training_authorized": False, "dataset_files_opened": False,
        "target_data_accessed": False, "manifest_sha256": MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if key not in report or type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"Report field disagrees with reviewed evidence: {key}")
    train_name, val_name = "gta5_source_train.txt", "gta5_source_val.txt"
    train, val = (contents[name].decode("ascii").splitlines() for name in (train_name, val_name))
    if (len(train) + len(val), len(train), len(val)) != COUNTS:
        raise ValueError("Manifest counts disagree with report")
    verify_membership(train + val, train, val)
    for name, ids in ((train_name, train), (val_name, val)):
        if manifest_bytes(ids) != contents[name]:
            raise ValueError(f"Noncanonical manifest bytes: {name}")
    expected_train, expected_val = source_split(train + val, "0.1")
    if (train, val) != (expected_train, expected_val):
        raise ValueError("Membership disagrees with the versioned deterministic rule")
    return {name: contents[name] for name in MANIFEST_SHA256}, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "manifests")
    args = parser.parse_args(argv)
    try:
        manifests, source_report = (verify_bundle(args.bundle) if args.bundle
                                    else verify_directory(args.input_dir))
        # Validate everything before making any destination; never overwrite a
        # previous candidate/frozen directory. Git commit is a later local step.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in manifests.items():
            (args.output_dir / name).write_bytes(payload)
        provenance = {
            "schema_version": 1, "export_zip_sha256": BUNDLE_SHA256,
            "input_format": "zip" if args.bundle else "extracted_directory",
            "zip_bytes_verified": args.bundle is not None,
            "manifest_sha256": MANIFEST_SHA256,
            "membership_policy": "fixed_for_all_training_seeds",
            "next_action": "commit_verified_manifests_to_git_before_training",
            "training_authorized": False, "source_report": source_report,
        }
        (args.output_dir / "gta5_source_split.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"ok": True, "train_count": COUNTS[1], "val_count": COUNTS[2],
                      "manifest_sha256": MANIFEST_SHA256, "output_dir": str(args.output_dir),
                      "training_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
