import json

import numpy as np
from PIL import Image
import torch

from causalq.datasets.segmentation import SegmentationPair
from causalq.models.scs import SCSConfig
from causalq.runtime.scs_source import ReinPairedSource, format_crop
from tools.train_scs_p4 import PROJECT_ROOT, save_state


def test_rein_crop_pad_and_mask_are_geometry_only():
    bgr = np.full((18, 32, 3), [103, 116, 124], dtype=np.uint8)
    labels = np.full((18, 32), 255, dtype=np.uint8)
    pair = format_crop(bgr, bgr, labels)
    assert pair["views"].shape == (2, 3, 512, 512)
    assert pair["patch_valid"].sum() == 4
    assert (pair["label"] == 255).all()
    assert torch.count_nonzero(pair["views"][:, :, 18:]) == 0
    torch.testing.assert_close(pair["views"][0], pair["views"][1])
    assert pair["views"][:, :, :18, :32].abs().max() < 0.02


def test_rein_stream_indices_and_random_views_survive_resume(tmp_path):
    image = tmp_path / "img.png"
    label = tmp_path / "label.png"
    Image.new("RGB", (32, 18), color=(124, 116, 103)).save(image)
    Image.new("L", (32, 18), color=2).save(label)
    pairs = (SegmentationPair(image, label, "example"),)

    def create(start):
        data = ReinPairedSource(pairs, None, seed=0, start=start, stop=12)
        data.geometry = lambda payload: payload
        data.photo = lambda payload: {"img": payload["img"] - np.random.randint(1, 32)}
        return data

    np.random.seed(71)
    expected_external_draw = np.random.randint(100000)
    np.random.seed(71)
    whole = create(0)[7]
    assert np.random.randint(100000) == expected_external_draw
    resumed = create(6)[1]
    for key in whole:
        torch.testing.assert_close(whole[key], resumed[key])
    assert not torch.equal(whole["views"][0], whole["views"][1])


def test_release_has_paired_seed_zero_and_recoverable_state(tmp_path):
    config = json.loads((PROJECT_ROOT / "configs/scs/p4_seed0.json").read_text(encoding="utf-8-sig"))
    assert config["released_arms"] == ["A0", "A4"]
    assert config["source_batch"] == 4 and config["total_updates"] == 40_000
    assert config["arms"]["A0"] == {"layers": [], "aux_weight": 0.0}
    assert tuple(config["arms"]["A4"]["layers"]) == SCSConfig().selected_layers
    path = tmp_path / "latest.pt"
    for step in (1, 1000):
        save_state(path, {"step": step, "rng_cpu": torch.get_rng_state()})
    assert torch.load(path, weights_only=True)["step"] == 1000
    assert not path.with_suffix(".partial").exists()
