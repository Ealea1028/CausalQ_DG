"""Source fallback must be exact, deterministic and independent of training RNG."""

import hashlib
from decimal import localcontext
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import pytest

from causalq.protocols.source_split import manifest_bytes, source_split, verify_membership
from tools import build_gta5_source_split as exporter
from tools.inspect_gta5_source_structure import inspect_source


def test_membership_ignores_input_order_and_training_seed():
    ids = [f"{i:05d}" for i in range(1, 101)]
    expected = source_split(ids)
    for seed in (0, 1, 2):
        shuffled = ids.copy()
        random.Random(seed).shuffle(shuffled)
        assert source_split(shuffled) == expected
    assert len(expected[0]) == 90 and len(expected[1]) == 10


def test_versioned_rule_has_fixed_golden_membership():
    _, val = source_split([f"{i:05d}" for i in range(1, 11)], "0.3")
    assert val == ["00001", "00005", "00010"]


def test_actual_inventory_count_rounding_without_inventing_real_ids():
    # Synthetic IDs test the count arithmetic only, never become real manifests.
    train, val = source_split([str(i) for i in range(24966)])
    assert (len(train), len(val)) == (22469, 2497)
    verify_membership(train + val, train, val)


@pytest.mark.parametrize("fraction", ["0", "1", "-0.1", "1.1", "NaN", "Infinity", "bad"])
def test_bad_fractions_rejected(fraction):
    with pytest.raises(ValueError):
        source_split(["01", "02"], fraction)


@pytest.mark.parametrize("ids", [[], ["01"], ["01", "01"], ["scene/01", "02"],
                                    ["../01", "02"], ["０１", "02"], ["01\n02", "03"]])
def test_invalid_or_unusable_membership_rejected(ids):
    with pytest.raises(ValueError):
        source_split(ids)


def test_overlap_and_incomplete_partition_rejected():
    with pytest.raises(ValueError, match="overlap"):
        verify_membership(["01", "02"], ["01", "02"], ["02"])
    with pytest.raises(ValueError, match="partition"):
        verify_membership(["01", "02", "03"], ["01"], ["02"])


def test_manifest_preserves_spelling_and_uses_lf():
    assert manifest_bytes(["02", "001", "01"]) == b"001\n01\n02\n"


def test_fraction_arithmetic_is_independent_of_decimal_context():
    with localcontext() as context:
        context.prec = 1
        _, val = source_split([str(i) for i in range(100)], "0.101")
    assert len(val) == 11


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    root = tmp_path / "gta5"
    for folder in ("images/images", "labels_trainIds"):
        (root / folder).mkdir(parents=True)
        for sample_id in ("00001", "00003", "00008", "00010"):
            (root / folder / f"{sample_id}.png").write_bytes(b"synthetic")
    report = inspect_source(root)
    report.update(schema_version=1, phase=17, git_sha=exporter.PRODUCER_SHA,
                  git_status="", root=exporter.ROOT,
                  resolved_image_root=exporter.ROOT + "/images/images",
                  resolved_label_root=exporter.ROOT + "/labels_trainIds")
    monkeypatch.setattr(exporter, "PAIR_COUNT", 4)
    monkeypatch.setattr(exporter, "PAIR_SHA256", report["pair_manifest_sha256"])
    path = tmp_path / "inventory.json"
    repin_fixture(report, path, monkeypatch)
    monkeypatch.setattr(exporter, "_git", lambda *args: "" if args[0] == "status" else "a" * 40)
    return report, path


def repin_fixture(report, path, monkeypatch):
    payload = json.dumps(report).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(exporter, "INVENTORY_SHA256", hashlib.sha256(payload).hexdigest())


def test_verifies_saved_bytes_and_pair_digest(inventory, monkeypatch):
    report, path = inventory
    assert exporter.verify_inventory(path.read_bytes())["paired_count"] == 4
    with pytest.raises(ValueError, match="SHA256"):
        exporter.verify_inventory(path.read_bytes() + b" ")
    report["pairs"][0]["image_bytes"] += 1
    repin_fixture(report, path, monkeypatch)
    with pytest.raises(ValueError, match="pair digest"):
        exporter.verify_inventory(path.read_bytes())


@pytest.mark.parametrize("key,value", [("metadata_candidates", [{"path": "scene.mat"}]),
                                         ("target_data_accessed", True),
                                         ("training_authorized", 0),
                                         ("paired_count", 5)])
def test_rejects_unreviewed_or_inconsistent_inventory(inventory, monkeypatch, key, value):
    report, path = inventory
    report[key] = value
    repin_fixture(report, path, monkeypatch)
    with pytest.raises(ValueError, match="field disagrees"):
        exporter.verify_inventory(path.read_bytes())


def test_export_bundle_roundtrip_and_immutable_input(inventory, tmp_path):
    _, path = inventory
    before = path.read_bytes()
    output = tmp_path / "candidate"
    args = ["--inventory", str(path), "--output-dir", str(output)]
    assert exporter.main(args) == 0
    report = json.loads((output / "split_report.json").read_bytes())
    assert (report["train_count"], report["val_count"]) == (3, 1)
    assert not report["frozen_in_git"] and not report["training_authorized"]
    assert not report["dataset_files_opened"] and not report["target_data_accessed"]
    assert path.read_bytes() == before
    with zipfile.ZipFile(output / "source_split_export.zip") as archive:
        assert set(archive.namelist()) == {"gta5_source_train.txt", "gta5_source_val.txt",
                                          "split_report.json", "SHA256SUMS"}
        for line in archive.read("SHA256SUMS").decode("ascii").splitlines():
            digest, name = line.split()
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        exporter.main(args)


def test_bad_inventory_preserves_failure_without_export(inventory, tmp_path):
    _, path = inventory
    path.write_bytes(b"bad inventory")
    output = tmp_path / "failed"
    assert exporter.main(["--inventory", str(path), "--output-dir", str(output)]) == 1
    assert not (output / "source_split_export.zip").exists()
    assert not (output / "gta5_source_train.txt").exists()
    assert not json.loads((output / "split_report.json").read_bytes())["ok"]


def test_export_cli_needs_no_site_packages():
    result = subprocess.run([sys.executable, "-S", "-m", "tools.build_gta5_source_split", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--source-val-fraction" in result.stdout
