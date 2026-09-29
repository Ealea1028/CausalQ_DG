from pathlib import Path
import copy
import numpy as np
import pytest

from tools.rein_protocol_adapter import adapt_protocol, valid_crop_box, validate_train_ids


def fixture_config():
    target = dict(type="CityscapesDataset", pipeline=[dict(type="LoadAnnotations")])
    return dict(train_dataloader=dict(dataset=dict(pipeline=[dict(type="LoadAnnotations"),
                    dict(type="RandomCrop", crop_size=(512, 512)), dict(type="PhotoMetricDistortion")])),
                val_dataloader=dict(dataset=dict(datasets=[target, dict(type="OtherTarget")])),
                optim_wrapper=dict(constructor="PEFTOptimWrapperConstructor",
                    paramwise_cfg=dict(norm_decay_mult=0)),
                model=dict(backbone=dict(init_cfg="old")),
                default_hooks=dict(checkpoint=dict(interval=4000)))


def test_adapter_isolated_targets_and_preserved_upstream_rules():
    upstream = fixture_config()
    original = copy.deepcopy(upstream)
    config = adapt_protocol(upstream, Path("/data"), Path("/images"), Path("/labels"))
    assert upstream == original
    assert config["train_dataloader"]["batch_size"] == 1
    assert config["optim_wrapper"]["accumulative_counts"] == 4
    assert config["optim_wrapper"]["paramwise_cfg"] == upstream["optim_wrapper"]["paramwise_cfg"]
    assert config["val_dataloader"]["dataset"]["type"] == "CityscapesDataset"
    assert config["val_evaluator"]["type"] == "IoUMetric"
    assert config["train_dataloader"]["dataset"]["seg_map_suffix"] == ".png"
    assert not config["train_dataloader"]["dataset"]["reduce_zero_label"]
    assert config["default_hooks"]["checkpoint"]["max_keep_ckpts"] == 1


def test_crop_fallback_retains_single_valid_pixel():
    label = np.full((9, 13), 255, dtype=np.uint8)
    label[8, 12] = 18
    top, bottom, left, right = valid_crop_box(label, (5, 5))
    assert np.any(label[top:bottom, left:right] == 18)
    with pytest.raises(ValueError):
        valid_crop_box(np.full((3, 3), 255), (5, 5))


def test_train_ids_not_remapped_and_invalid_ids_rejected():
    assert validate_train_ids(np.array([[0, 18, 255]], dtype=np.uint8)) == [0, 18, 255]
    with pytest.raises(ValueError):
        validate_train_ids(np.array([[33]]))
