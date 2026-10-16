"""Strict source routing uses frozen membership and never constructs target data."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from causalq.datasets import TrainTransform
from causalq.datasets import strict_source as source
from causalq.protocols.source_split import manifest_bytes, source_split
from causalq.protocols.strict_dg import validate_strict_config
from tools import check_strict_source_data as gate
from tools import train as trainer
from tools.inspect_gta5_source_structure import inspect_source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml"


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    data_root = tmp_path / "datasets"
    gta = data_root / "gta5"
    train, val = source_split(["00001", "00003", "00005", "00009"])
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    hashes = {}
    for name, ids in zip(source.MANIFEST_HASHES, (train, val)):
        payload = manifest_bytes(ids)
        (manifest_dir / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(source, "MANIFEST_HASHES", hashes)
    for key in train + val:
        for folder, array in (("images/images", np.full((12, 20, 3), 120, dtype=np.uint8)),
                               ("labels_trainIds", np.full((12, 20), 4, dtype=np.uint8))):
            path = gta / folder / f"{key}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)
    return data_root, manifest_dir, train, val


def test_real_frozen_manifest_hashes_and_membership():
    train, val = source.load_membership()
    assert (len(train), len(val)) == (22469, 2497)
    assert not set(train) & set(val)
    assert (train, val) == source_split(train + val)


def test_source_datasets_partition_without_target_access(source_fixture, monkeypatch):
    data, manifests, train_ids, val_ids = source_fixture
    monkeypatch.setattr("causalq.datasets.cityscapes_dataset",
                        lambda *a, **k: pytest.fail("target dataset constructed"))
    train, val = source.build_source_datasets(data / "gta5", TrainTransform([512, 512]),
                                               source.SourceEvalTransform(32), manifests)
    assert [pair.sample_id for pair in train.pairs] == train_ids
    assert [pair.sample_id for pair in val.pairs] == val_ids
    assert not (data / "cityscapes").exists()
    before = torch.get_rng_state().clone()
    sample = val[0]
    assert torch.equal(before, torch.get_rng_state())
    assert sample["image"].shape[-2:] == sample["label"].shape
    assert all(size % 16 == 0 for size in sample["label"].shape)


def test_eval_full_frame_aspect_and_ignore_padding():
    transform = source.SourceEvalTransform(32)
    image = Image.fromarray(np.zeros((9, 20, 3), dtype=np.uint8))
    label = Image.fromarray(np.full((9, 20), 7, dtype=np.uint8))
    rng = torch.get_rng_state().clone()
    x, y = transform(image, label)
    assert tuple(x.shape) == (3, 16, 32)
    assert torch.all(y[:14] == 7) and torch.all(y[14:] == 255)
    assert torch.equal(rng, torch.get_rng_state())


def test_modified_manifest_and_extra_payload_rejected(source_fixture):
    data, manifests, _, _ = source_fixture
    payload = manifests / "gta5_source_train.txt"
    saved = payload.read_bytes()
    payload.write_bytes(saved + b"99999\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        source.load_membership(manifests)
    payload.write_bytes(saved)
    (data / "gta5/images/images/extra.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exact match"):
        source.build_source_datasets(data / "gta5", TrainTransform([512, 512]),
                                     source.SourceEvalTransform(32), manifests)


@pytest.mark.parametrize("section,key,value", [
    ("protocol", "target_access", "allowed"), ("protocol", "training_authorized", True),
    ("data", "validation", "cityscapes_val"), ("data", "targets", ["cityscapes"]),
    ("query", "queries_per_class", 3), ("style", "views", ["original", "fourier"]),
])
def test_strict_gate_rejects_protocol_or_mechanism_drift(tmp_path, section, key, value):
    config = trainer.load_config(CONFIG)
    config[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.load_config(path)


def test_training_is_blocked_before_cuda_or_data(monkeypatch):
    monkeypatch.setattr(trainer, "parse_args", lambda: type("Args", (), {"config": CONFIG})())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA check before gate"))
    monkeypatch.setattr(trainer, "build_training_datasets", lambda *a: pytest.fail("data opened"))
    with pytest.raises(RuntimeError, match="source data gate"):
        trainer.main()


def test_strict_and_legacy_routing_are_distinct(source_fixture, monkeypatch):
    data, manifests, _, _ = source_fixture
    original = source.build_source_datasets
    monkeypatch.setattr(source, "build_source_datasets",
                        lambda root, train, val: original(root, train, val, manifests))
    monkeypatch.setattr("causalq.datasets.cityscapes_dataset", lambda *a, **k: "legacy_target")
    strict = trainer.build_training_datasets(trainer.load_config(CONFIG), data)
    assert all(isinstance(dataset, source.PairedSegmentationDataset) for dataset in strict)
    monkeypatch.setattr(trainer, "gta5_dataset", lambda *a, **k: "legacy_source")
    config = trainer.load_config(ROOT / "configs/query_interaction/gta_dinov3l_static.yaml")
    assert trainer.build_training_datasets(config, data) == ("legacy_source", "legacy_target")


def test_pixel_inspection_rejects_raw_or_invalid_train_ids(source_fixture, tmp_path):
    data, manifests, _, _ = source_fixture
    train, _ = source.build_source_datasets(data / "gta5", TrainTransform([512, 512]),
                                            source.SourceEvalTransform(32), manifests)
    Image.fromarray(np.full((12, 20), 99, dtype=np.uint8)).save(train.pairs[0].label)
    result = gate.inspect_split(train, "train", tmp_path)
    assert not result["ok"] and result["invalid_train_ids"] == [99]


def test_preprocessing_gate_restores_rng(source_fixture):
    data, manifests, _, _ = source_fixture
    train, val = source.build_source_datasets(data / "gta5", TrainTransform([512, 512]),
                                              source.SourceEvalTransform(32), manifests)
    before = torch.get_rng_state().clone()
    result = gate.check_preprocessing(train, val)
    assert result["ok"] and result["holdout_checked"] == 1
    assert torch.equal(before, torch.get_rng_state())


def test_full_data_gate_checks_both_splits_and_inventory(source_fixture, tmp_path, monkeypatch):
    data, manifests, _, _ = source_fixture
    datasets = source.build_source_datasets(data / "gta5", TrainTransform([512, 512]),
                                             source.SourceEvalTransform(32), manifests)
    monkeypatch.setattr(gate, "build_training_datasets", lambda *a: datasets)
    monkeypatch.setattr(gate, "PAIR_SHA256", inspect_source(data / "gta5")["pair_manifest_sha256"])
    result = gate.run_gate({}, data, tmp_path)
    assert result["ok"]
    assert result["train_scan"]["scanned_label_count"] == 3
    assert result["val_scan"]["scanned_label_count"] == 1
    monkeypatch.setattr(gate, "PAIR_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="changed since"):
        gate.run_gate({}, data, tmp_path)
