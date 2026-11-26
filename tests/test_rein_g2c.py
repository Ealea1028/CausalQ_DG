from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.prepare_rein_g2c import configure, prepare_mask


def test_city_raw_annotations_convert_to_train_ids(tmp_path):
    image, raw, output = (tmp_path / name for name in ("image.png", "raw.png", "staged/mask.png"))
    Image.new("RGB", (4, 1)).save(image)
    Image.fromarray(np.array([[7, 26, 0, 33]], dtype=np.uint8)).save(raw)
    assert prepare_mask(image, raw, output, raw_city=True) == "labelIds_to_trainIds"
    with Image.open(output) as mask:
        assert list(mask.getdata()) == [0, 13, 255, 18]
    with Image.open(raw) as source:
        assert list(source.getdata()) == [7, 26, 0, 33]


def test_source_geometry_repair_is_nearest_and_rejects_a_different_aspect(tmp_path):
    image, label, output = (tmp_path / name for name in ("image.png", "raw.png", "staged/mask.png"))
    Image.new("RGB", (4, 4)).save(image)
    Image.fromarray(np.array([[0, 13], [255, 18]], dtype=np.uint8)).save(label)
    assert prepare_mask(image, label, output, gta=True) == "nearest_alignment"
    with Image.open(output) as mask:
        assert np.asarray(mask).tolist() == [
            [0, 0, 13, 13], [0, 0, 13, 13], [255, 255, 18, 18], [255, 255, 18, 18]
        ]
    Image.new("RGB", (8, 4)).save(image)
    with pytest.raises(ValueError, match="geometry mismatch"):
        prepare_mask(image, label, output, gta=True)


def test_invalid_train_id_and_city_geometry_are_rejected(tmp_path):
    image, label, output = (tmp_path / name for name in ("image.png", "mask.png", "staged.png"))
    Image.new("RGB", (2, 2)).save(image)
    Image.new("L", (2, 2), 26).save(label)
    with pytest.raises(ValueError, match="Invalid train IDs"):
        prepare_mask(image, label, output, gta=True)
    Image.new("L", (1, 1), 7).save(label)
    with pytest.raises(ValueError, match="sizes differ"):
        prepare_mask(image, label, output, raw_city=True)
    assert not output.exists()


def test_config_scopes_evaluator_to_city_and_preserves_upstream_training():
    source_pipeline = [{"type": "LoadAnnotations"}, {"type": "RandomCrop"}]
    bdd = dict(data_prefix=dict(img_path="images/10k/val"))
    city = dict(data_prefix=dict(img_path="leftImg8bit/val"), pipeline=[{"type": "LoadAnnotations"}])
    config = dict(
        train_dataloader=dict(batch_size=4, dataset=dict(pipeline=source_pipeline)),
        val_dataloader=dict(batch_size=1, dataset=dict(datasets=[bdd, city])),
        train_cfg=dict(max_iters=40000), optim_wrapper=dict(optimizer=dict(type="AdamW")),
        model=dict(backbone=dict(init_cfg=dict(checkpoint="original")), test_cfg=dict(mode="slide")),
    )
    cfg = configure(config, Path("images"), Path("city"), Path("labels"), Path("backbone.pth"))
    assert cfg["train_dataloader"]["dataset"]["pipeline"][1]["type"] == "RandomCrop"
    assert cfg["train_cfg"] == config["train_cfg"]
    assert cfg["optim_wrapper"] == config["optim_wrapper"]
    assert cfg["model"]["test_cfg"] == config["model"]["test_cfg"]
    assert cfg["test_dataloader"]["dataset"]["data_prefix"]["img_path"] == "city"
    assert cfg["test_evaluator"] == {"type": "IoUMetric", "iou_metrics": ["mIoU"], "ignore_index": 255}
    assert city["data_prefix"]["img_path"] == "leftImg8bit/val"
    assert source_pipeline[0] == {"type": "LoadAnnotations"}
