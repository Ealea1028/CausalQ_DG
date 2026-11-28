"""Deterministic REIN paired crops indexed by the absolute source-image cursor."""

from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from causalq.datasets.label_space import read_index_mask
from causalq.datasets.segmentation import IMAGENET_MEAN, IMAGENET_STD, align_scale_equivalent_label
from .input_stream import epoch_order, input_seed


REIN_TRAIN_CONFIG = "configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py"


class ReinPairedSource(Dataset):
    def __init__(self, pairs, pipeline, *, seed: int, start: int, stop: int):
        self.pairs = pairs
        self.pipeline_config = pipeline
        self.seed, self.start, self.stop = seed, start, stop
        self.geometry = self.photo = None

    def __len__(self):
        return self.stop - self.start

    def _build_transforms(self):
        import cv2
        from mmcv.transforms import Compose
        from mmengine.registry import init_default_scope
        from mmseg.registry import TRANSFORMS

        cv2.setNumThreads(1)
        init_default_scope("mmseg")
        self.geometry = Compose([TRANSFORMS.build(deepcopy(cfg)) for cfg in self.pipeline_config[:3]])
        self.photo = TRANSFORMS.build(deepcopy(self.pipeline_config[3]))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.geometry is None:
            self._build_transforms()
        cursor = self.start + index
        epoch, offset = divmod(cursor, len(self.pairs))
        pair = self.pairs[epoch_order(len(self.pairs), self.seed, epoch)[offset]]
        with Image.open(pair.image) as image_file, Image.open(pair.label) as label_file:
            image = image_file.convert("RGB")
            label = align_scale_equivalent_label(image, Image.fromarray(read_index_mask(label_file)))
        mask = np.array(label)
        if np.any((mask != 255) & (mask >= 19)):
            raise ValueError(f"Invalid GTA5 train ID in {pair.sample_id}")
        result = {"img": np.asarray(image)[:, :, ::-1].copy(), "gt_seg_map": mask, "seg_fields": ["gt_seg_map"]}
        rng = np.random.get_state()
        try:
            np.random.seed(input_seed(self.seed, cursor, pair.sample_id, "rein_geometry") % 2**32)
            result = self.geometry(result)
            np.random.seed(input_seed(self.seed, cursor, pair.sample_id, "rein_photometric") % 2**32)
            photo = self.photo({"img": result["img"].copy()})["img"]
        finally:
            np.random.set_state(rng)
        return format_crop(result["img"], photo, result["gt_seg_map"])


def format_crop(original, photo, labels):
    h, w = labels.shape
    if not (0 < h <= 512 and 0 < w <= 512) or original.shape != photo.shape or original.shape[:2] != (h, w):
        raise ValueError("Unaligned or oversized REIN crop")
    images = torch.from_numpy(np.stack((original[:, :, ::-1], photo[:, :, ::-1]))).permute(0, 3, 1, 2).float()
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    views = torch.zeros(2, 3, 512, 512)
    views[:, :, :h, :w] = (images / 255 - mean) / std
    mask = torch.full((512, 512), 255, dtype=torch.long)
    mask[:h, :w] = torch.from_numpy(labels.copy())
    valid = (torch.arange(0, 512, 16)[:, None] < h) & (torch.arange(0, 512, 16)[None, :] < w)
    return {"views": views, "label": mask, "patch_valid": valid.flatten()}
