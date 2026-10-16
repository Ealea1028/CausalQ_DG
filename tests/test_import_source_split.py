"""Local import must verify real bytes before creating permanently versioned IDs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from causalq.protocols.source_split import manifest_bytes
from tools import build_gta5_source_split as exporter
from tools import import_gta5_source_split as importer


def write_zip(path, contents, monkeypatch):
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "x") as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
    monkeypatch.setattr(importer, "BUNDLE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())


def repin_contents(contents, monkeypatch):
    hashes = {name: hashlib.sha256(contents[name]).hexdigest() for name in importer.MANIFEST_SHA256}
    monkeypatch.setattr(importer, "MANIFEST_SHA256", hashes)
    report = json.loads(contents["split_report.json"])
    report["manifest_sha256"] = hashes
    contents["split_report.json"] = json.dumps(report).encode("utf-8")
    contents["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(contents.items()) if name != "SHA256SUMS").encode("ascii")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    inventory.write_bytes(b"synthetic inventory")
    monkeypatch.setattr(exporter, "_git", lambda *args: "" if args[0] == "status" else importer.EXPORT_SHA)
    monkeypatch.setattr(exporter, "verify_inventory", lambda _: {
        "pairs": [{"id": value} for value in ("00001", "00003", "00007", "00009")]})
    output = tmp_path / "export"
    assert exporter.main(["--inventory", str(inventory), "--output-dir", str(output)]) == 0
    path = output / "source_split_export.zip"
    monkeypatch.setattr(importer, "BUNDLE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    report = json.loads((output / "split_report.json").read_bytes())
    monkeypatch.setattr(importer, "MANIFEST_SHA256", report["manifest_sha256"])
    monkeypatch.setattr(importer, "COUNTS", (4, 3, 1))
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    return path, contents


def test_valid_bundle_freezes_exact_bytes_without_overwrite(bundle, tmp_path):
    path, contents = bundle
    output = tmp_path / "manifests"
    args = ["--bundle", str(path), "--output-dir", str(output)]
    assert importer.main(args) == 0
    for name in importer.MANIFEST_SHA256:
        assert (output / name).read_bytes() == contents[name]
    provenance = json.loads((output / "gta5_source_split.json").read_bytes())
    assert provenance["export_zip_sha256"] == importer.BUNDLE_SHA256
    assert not provenance["training_authorized"]
    assert importer.main(args) == 1


def test_wrong_bundle_fails_before_destination_creation(bundle, tmp_path):
    path, _ = bundle
    path.write_bytes(path.read_bytes() + b"tampered")
    output = tmp_path / "not_created"
    assert importer.main(["--bundle", str(path), "--output-dir", str(output)]) == 1
    assert not output.exists()


def test_checksum_mismatch_rejected(bundle, monkeypatch):
    path, contents = bundle
    contents["gta5_source_train.txt"] += b"00999\n"
    write_zip(path, contents, monkeypatch)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        importer.verify_bundle(path)


def test_archive_path_not_extracted(bundle, tmp_path, monkeypatch):
    path, contents = bundle
    contents["../escaped.txt"] = b"must not extract"
    write_zip(path, contents, monkeypatch)
    with pytest.raises(ValueError, match="ZIP members"):
        importer.verify_bundle(path)
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize("field,value", [("git_sha", "b" * 40), ("target_data_accessed", True),
                                           ("training_authorized", 0), ("salt", "other")])
def test_report_contract_mismatch_rejected(bundle, monkeypatch, field, value):
    path, contents = bundle
    report = json.loads(contents["split_report.json"])
    report[field] = value
    contents["split_report.json"] = json.dumps(report).encode("utf-8")
    repin_contents(contents, monkeypatch)
    write_zip(path, contents, monkeypatch)
    with pytest.raises(ValueError, match="Report field"):
        importer.verify_bundle(path)


def test_different_valid_partition_fails_rule_recomputation(bundle, monkeypatch):
    path, contents = bundle
    train = contents["gta5_source_train.txt"].decode().splitlines()
    val = contents["gta5_source_val.txt"].decode().splitlines()
    train[0], val[0] = val[0], train[0]
    contents["gta5_source_train.txt"] = manifest_bytes(train)
    contents["gta5_source_val.txt"] = manifest_bytes(val)
    repin_contents(contents, monkeypatch)
    write_zip(path, contents, monkeypatch)
    with pytest.raises(ValueError, match="deterministic rule"):
        importer.verify_bundle(path)


def test_cli_imports_without_site_packages():
    result = subprocess.run([sys.executable, "-S", "-m", "tools.import_gta5_source_split", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--bundle" in result.stdout
