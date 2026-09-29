"""Versioned GTA5-to-Cityscapes adaptation, not an exact upstream reproduction."""

import copy


def adapt_protocol(upstream, data_root, image_root, label_root):
    config = copy.deepcopy(upstream)
    source = config["train_dataloader"]
    source.update(batch_size=1, num_workers=0, persistent_workers=False)
    dataset = source["dataset"]
    dataset.update(type="CityscapesDataset", data_root=str(data_root),
                   data_prefix=dict(img_path=str(image_root), seg_map_path=str(label_root)),
                   img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False)
    pipeline = dataset["pipeline"]
    for index, step in enumerate(pipeline):
        if step["type"] == "LoadAnnotations":
            step["reduce_zero_label"] = False
            pipeline.insert(index + 1, dict(type="ReinAlignTrainLabel"))
            break
    else:
        raise ValueError("Missing source annotation loader")
    for step in pipeline:
        if step["type"] == "RandomCrop":
            step["type"] = "ReinValidRandomCrop"
    target = copy.deepcopy(config["val_dataloader"]["dataset"]["datasets"][0])
    if target["type"] != "CityscapesDataset":
        raise ValueError("Expected first upstream target to be Cityscapes")
    target.update(data_root=str(data_root / "cityscapes"), reduce_zero_label=False,
                  img_suffix="_leftImg8bit.png", seg_map_suffix="_gtFine_labelTrainIds.png")
    for step in target["pipeline"]:
        if step["type"] == "LoadAnnotations":
            step["reduce_zero_label"] = False
    for key in ("val_dataloader", "test_dataloader"):
        config[key] = dict(batch_size=1, num_workers=0, persistent_workers=False,
                           sampler=dict(type="DefaultSampler", shuffle=False), dataset=copy.deepcopy(target))
    for key in ("val_evaluator", "test_evaluator"):
        config[key] = dict(type="IoUMetric", iou_metrics=["mIoU"])
    config["optim_wrapper"].update(type="OptimWrapper", accumulative_counts=4,
                                    clip_grad=dict(max_norm=1.0, norm_type=2))
    config["model"]["backbone"]["init_cfg"] = None  # Future runner must explicitly safe-load audited weights.
    config["randomness"] = dict(seed=0)
    config["default_hooks"]["checkpoint"].update(max_keep_ckpts=1, save_best=None)
    return config


def validate_train_ids(label):
    import numpy as np

    values = np.unique(label)
    if not np.all(((values >= 0) & (values <= 18)) | (values == 255)):
        raise ValueError(f"Not Cityscapes train IDs: {values.tolist()}")
    return values.tolist()


def valid_crop_box(label, crop_size):
    """Fallback only for an all-ignore random crop. Never silently drop a sample."""
    import numpy as np

    height, width = label.shape
    crop_h, crop_w = min(crop_size[0], height), min(crop_size[1], width)
    positions = np.flatnonzero(label != 255)
    if not len(positions):
        raise ValueError("Full source label contains no valid training pixels")
    y, x = divmod(int(np.random.choice(positions)), width)
    top = int(np.random.randint(max(0, y - crop_h + 1), min(y, height - crop_h) + 1))
    left = int(np.random.randint(max(0, x - crop_w + 1), min(x, width - crop_w) + 1))
    return top, top + crop_h, left, left + crop_w


def register_data_transforms():
    """Called only in the isolated OpenMMLab runtime, never by local CPU tests."""
    import numpy as np
    from PIL import Image
    from mmcv.transforms import BaseTransform
    from mmseg.datasets.transforms import RandomCrop
    from mmseg.registry import TRANSFORMS

    @TRANSFORMS.register_module()
    class ReinAlignTrainLabel(BaseTransform):
        def transform(self, results):
            from causalq.datasets.segmentation import align_scale_equivalent_label

            label = results["gt_seg_map"]
            validate_train_ids(label)
            height, width = results["img"].shape[:2]
            if label.shape != (height, width):
                # Reuse the project's 0.2% aspect-error gate and nearest resampling.
                label_image = align_scale_equivalent_label(
                    Image.new("RGB", (width, height)), Image.fromarray(label))
                results["gt_seg_map"] = np.asarray(label_image).copy()
            return results

    @TRANSFORMS.register_module()
    class ReinValidRandomCrop(RandomCrop):
        def transform(self, results):
            original = copy.deepcopy(results)
            output = super().transform(results)
            if np.any(output["gt_seg_map"] != 255):
                return output
            top, bottom, left, right = valid_crop_box(original["gt_seg_map"], self.crop_size)
            original["img"] = original["img"][top:bottom, left:right]
            original["img_shape"] = original["img"].shape[:2]
            for key in original.get("seg_fields", []):
                original[key] = original[key][top:bottom, left:right]
            return original
