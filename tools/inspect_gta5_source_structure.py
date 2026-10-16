"""Priority-0 source-only filesystem inventory; never selects a training split.

Standard-library only: no image decoding, torch, model or target-data access.
Run with ``python -m tools.inspect_gta5_source_structure --help``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
METADATA_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".yaml", ".yml",
                     ".xml", ".mat", ".h5", ".hdf5", ".npy", ".npz"}
TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml"}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def _raise_walk_error(error: OSError) -> None:
    raise error


def _files(root: Path) -> list[Path]:
    """Refuse links rather than following a source path into another domain."""
    if not root.is_dir():
        raise FileNotFoundError(f"GTA5 directory not found: {root}")
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"Source inventory refuses links/junctions: {path}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Source path escapes GTA5: {path}")
        files.extend(Path(directory) / name for name in sorted(names))
    return sorted(files)


def _payload(root: Path, files: list[Path], suffixes: set[str]) -> tuple[Path, list[Path]]:
    """Match the historical adapter's single redundant-directory unwrapping."""
    selected = [p for p in files if p.is_relative_to(root) and p.suffix.lower() in suffixes]
    if selected and not any(p.parent == root for p in selected):
        children = {p.relative_to(root).parts[0] for p in selected}
        if len(children) == 1:
            root = root / next(iter(children))
    return root, selected


def _index(root: Path, files: list[Path]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[path.relative_to(root).with_suffix("").as_posix()].append(path)
    duplicates = {key: [p.relative_to(root).as_posix() for p in paths]
                  for key, paths in groups.items() if len(paths) > 1}
    return {key: paths[0] for key, paths in groups.items()}, duplicates


def inspect_source(gta5_root: Path) -> dict:
    # The CLI takes the GTA5 subtree itself, never a multi-domain data root.
    if gta5_root.is_symlink() or getattr(gta5_root, "is_junction", lambda: False)():
        raise ValueError("GTA5 root must not be a link/junction")
    root = gta5_root.resolve()
    files = _files(root)
    image_root, images = _payload(root / "images", files, IMAGE_SUFFIXES)
    label_root, labels = _payload(root / "labels_trainIds", files, {".png"})
    image_map, duplicate_images = _index(image_root, images)
    label_map, duplicate_labels = _index(label_root, labels)
    shared = sorted(image_map.keys() & label_map.keys())
    missing_labels = sorted(image_map.keys() - label_map.keys())
    missing_images = sorted(label_map.keys() - image_map.keys())
    pairs = [{"id": key, "image": image_map[key].relative_to(root).as_posix(),
              "label": label_map[key].relative_to(root).as_posix(),
              "image_bytes": image_map[key].stat().st_size,
              "label_bytes": label_map[key].stat().st_size} for key in shared]
    parents = Counter(Path(key).parent.as_posix() for key in shared)
    patterns = Counter(re.sub(r"\d+", "{digits}", Path(key).name) for key in shared)
    metadata = []
    # Include sidecars anywhere INSIDE GTA5, also within image/label folders.
    # Binary containers are listed only; their semantics require later review.
    for path in files:
        if path.suffix.lower() not in METADATA_SUFFIXES:
            continue
        record = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        if path.suffix.lower() in TEXT_SUFFIXES:
            with path.open("rb") as stream:
                preview = stream.read(4096)
            record["preview_utf8"] = preview.decode("utf-8", errors="replace")
            record["preview_truncated"] = record["bytes"] > 4096
        metadata.append(record)
    # Identify candidate evidence, never infer that numeric IDs are video frames.
    return {
        "ok": bool(shared) and not (missing_labels or missing_images
                                    or duplicate_images or duplicate_labels),
        "scope": "gta5_source_filesystem_only",
        "root": str(root), "resolved_image_root": str(image_root),
        "resolved_label_root": str(label_root),
        "image_count": len(images), "label_count": len(labels),
        "paired_count": len(shared), "pairs": pairs,
        "missing_labels": missing_labels, "missing_images": missing_images,
        "duplicate_image_ids": duplicate_images, "duplicate_label_ids": duplicate_labels,
        "parent_directory_counts": dict(sorted(parents.items())),
        "filename_pattern_counts": dict(sorted(patterns.items())),
        "metadata_candidates": metadata,
        "directory_counts": dict(sorted(Counter(
            p.parent.relative_to(root).as_posix() for p in files).items())),
        "file_suffix_counts": dict(sorted(Counter(p.suffix.lower() for p in files).items())),
        "pair_manifest_sha256": hashlib.sha256(
            json.dumps(pairs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "split_status": "pending_metadata_review",
        "split_priority": ["sequence", "scene", "validated_filename_group",
                           "deterministic_filename_fallback_only_if_no_grouping"],
        "warning": "Directory names and numeric filename patterns do not prove scene/sequence independence.",
        "target_data_accessed": False, "images_decoded": False,
        "label_values_validated": False, "training_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gta5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.is_relative_to(args.gta5_root.resolve()):
        parser.error("output-dir must be outside the read-only GTA5 subtree")
    # Exclusive directory creation preserves earlier successful AND failed evidence.
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "phase": 17,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform()},
    }
    try:
        report["git_sha"] = _git("rev-parse", "HEAD")
        report["git_status"] = _git("status", "--porcelain")
        report.update(inspect_source(args.gta5_root))
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                      training_authorized=False)
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    destination = output / "inventory.json"
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (output / "SHA256SUMS").write_text(f"{digest}  inventory.json\n", encoding="utf-8")
    summary = {key: value for key, value in report.items()
               if key not in {"pairs", "metadata_candidates", "directory_counts"}}
    summary.update(inventory=str(destination), inventory_sha256=digest,
                   metadata_candidate_count=len(report.get("metadata_candidates", [])))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
