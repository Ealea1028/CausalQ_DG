"""Synthetic files suffice: the source gate must not decode pixels or access target."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.inspect_gta5_source_structure import inspect_source, main


def populate(root: Path, *ids: str) -> None:
    for sample_id in ids:
        for folder in ("images", "labels_trainIds"):
            path = root / folder / f"{sample_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"deliberately not an image: no pixel access allowed")


def test_flat_inventory_does_not_claim_sequences_or_create_split(tmp_path):
    root = tmp_path / "gta5"
    populate(root, "00001", "00002")
    result = inspect_source(root)
    assert result["ok"] and result["paired_count"] == 2
    assert result["parent_directory_counts"] == {".": 2}
    assert result["filename_pattern_counts"] == {"{digits}": 2}
    assert result["split_status"] == "pending_metadata_review"
    assert not result["target_data_accessed"]
    assert not result["images_decoded"]
    assert not result["training_authorized"]
    assert not list(root.rglob("*source_train*"))


def test_nested_pairs_and_metadata_are_preserved(tmp_path):
    root = tmp_path / "gta5"
    populate(root, "seq_a/0001", "seq_b/0002")
    metadata = root / "scene_groups.csv"
    metadata.write_text("id,scene\nseq_a/0001,A\nseq_b/0002,B\n", encoding="utf-8")
    result = inspect_source(root)
    assert result["parent_directory_counts"] == {"seq_a": 1, "seq_b": 1}
    assert [p["id"] for p in result["pairs"]] == ["seq_a/0001", "seq_b/0002"]
    assert result["metadata_candidates"][0]["preview_utf8"].startswith("id,scene")
    assert result == inspect_source(root)


def test_single_archive_wrapper_matches_historical_sample_ids(tmp_path):
    root = tmp_path / "gta5"
    for folder, wrapper in (("images", "images"), ("labels_trainIds", "labels")):
        path = root / folder / wrapper / "00001.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
    result = inspect_source(root)
    assert result["ok"]
    assert result["pairs"][0]["id"] == "00001"
    assert result["pairs"][0]["label"] == "labels_trainIds/labels/00001.png"


def test_missing_and_duplicate_pairs_fail_closed(tmp_path):
    root = tmp_path / "gta5"
    populate(root, "00001", "00002")
    (root / "images/00001.jpg").write_bytes(b"ambiguous duplicate")
    (root / "labels_trainIds/00002.png").unlink()
    result = inspect_source(root)
    assert not result["ok"]
    assert result["missing_labels"] == ["00002"]
    assert "00001" in result["duplicate_image_ids"]


def test_cli_hash_provenance_exclusive_output_and_target_guard(tmp_path, monkeypatch):
    root = tmp_path / "gta5"
    populate(root, "00001")
    target = tmp_path / "cityscapes"
    target.mkdir()
    original_walk = __import__("os").walk

    def guarded_walk(path, **kwargs):
        assert Path(path).resolve() == root.resolve()
        return original_walk(path, **kwargs)

    monkeypatch.setattr("tools.inspect_gta5_source_structure.os.walk", guarded_walk)
    output = tmp_path / "report"
    args = ["--gta5-root", str(root), "--output-dir", str(output)]
    assert main(args) == 0
    payload = (output / "inventory.json").read_bytes()
    result = json.loads(payload)
    assert len(result["git_sha"]) == 40
    assert result["environment"]["python"]
    assert (output / "SHA256SUMS").read_text().split()[0] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main(["--gta5-root", str(root), "--output-dir", str(root / "bad")])


def test_missing_dataset_produces_failed_evidence(tmp_path):
    output = tmp_path / "report"
    assert main(["--gta5-root", str(tmp_path / "absent"), "--output-dir", str(output)]) == 1
    report = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert not report["ok"]
    assert "FileNotFoundError" in report["error"]


def test_external_link_is_rejected_before_traversal(tmp_path):
    root = tmp_path / "gta5"
    populate(root, "00001")
    outside = tmp_path / "cityscapes"
    outside.mkdir()
    link = root / "external"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit unprivileged symlinks")
    with pytest.raises(ValueError, match="links/junctions"):
        inspect_source(root)


def test_cli_is_independent_of_site_packages():
    # -S hides all third-party packages, including torch, numpy, Pillow and yaml.
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tools.inspect_gta5_source_structure", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--gta5-root" in result.stdout


def test_metadata_preview_is_bounded_and_binary_metadata_is_not_parsed(tmp_path):
    root = tmp_path / "gta5"
    populate(root, "00001")
    (root / "sequence.txt").write_bytes(b"a" * 5000)
    (root / "scene.mat").write_bytes(b"not a MAT file; list only")
    candidates = {p["path"]: p for p in inspect_source(root)["metadata_candidates"]}
    assert len(candidates["sequence.txt"]["preview_utf8"]) == 4096
    assert candidates["sequence.txt"]["preview_truncated"]
    assert "preview_utf8" not in candidates["scene.mat"]


def test_link_guard_without_host_symlink_privilege(tmp_path, monkeypatch):
    root = tmp_path / "gta5"
    populate(root, "00001")
    original = Path.is_symlink

    def simulated_link(path):
        return path == root / "images" or original(path)

    monkeypatch.setattr(Path, "is_symlink", simulated_link)
    with pytest.raises(ValueError, match="links/junctions"):
        inspect_source(root)
