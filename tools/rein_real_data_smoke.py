"""Real-data bridge and bounded optimization diagnostic; not a formal REIN run."""

from pathlib import Path
import hashlib
import json


def normalized_rgb_to_bgr255(image):
    """Undo project normalization before the upstream BGR preprocessor, once."""
    import torch
    from causalq.datasets.segmentation import IMAGENET_MEAN, IMAGENET_STD

    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError("Expected finite normalized CHW RGB")
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (image * std + mean) * 255
    if rgb.min().item() < -0.01 or rgb.max().item() > 255.01:
        raise ValueError("Project image is outside normalized RGB range")
    return rgb[[2, 1, 0]].contiguous()  # Float32, no uint8 quantization.


def prepare_sample(model, item, *, training):
    import torch
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    image, label = item["image"], item["label"]
    if image.shape != (3, 512, 512) or label.shape != (512, 512):
        raise ValueError("Real smoke requires matched 512x512 image/label")
    valid = label != 255
    if torch.any(valid & ((label < 0) | (label > 18))) or (training and not valid.any()):
        raise ValueError(f"Invalid train IDs or empty training crop: {item['id']}")
    sample = SegDataSample(metainfo=dict(ori_shape=(512, 512), img_shape=(512, 512),
                                        pad_shape=(512, 512), scale_factor=(1., 1.)))
    sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
    batch = model.data_preprocessor(dict(inputs=[normalized_rgb_to_bgr255(image)],
                                         data_samples=[sample]), training=training)
    error = (batch["inputs"][0] - image.to(batch["inputs"].device)).abs().max().item()
    if error > 1e-5:
        raise ValueError(f"Normalization round-trip error: {error}")
    return batch, error


def center_target_crop(item):
    """Aspect-preserving resize then center crop, inference diagnostic only."""
    import torch.nn.functional as F

    image, label = item["image"], item["label"]
    height, width = label.shape
    scale = 512 / min(height, width)
    size = (round(height * scale), round(width * scale))
    image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    label = F.interpolate(label[None, None].float(), size=size, mode="nearest")[0, 0].long()
    top, left = (size[0] - 512) // 2, (size[1] - 512) // 2
    return dict(image=image[:, top:top+512, left:left+512],
                label=label[top:top+512, left:left+512], id=item["id"])


def pair_inventory_digest(dataset, root):
    rows = [(pair.sample_id, pair.image.relative_to(root).as_posix(),
             pair.label.relative_to(root).as_posix()) for pair in dataset.pairs]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def run_real_data_smoke(model, data_root: Path, report):
    import torch
    from causalq.datasets.segmentation import TrainTransform, gta5_dataset, cityscapes_dataset
    from tools.check_rein_segmentor import validate_losses, validate_prediction

    report["stage"] = "real_data_inventory"
    source = gta5_dataset(data_root / "gta5", transform=TrainTransform((512, 512)))
    target = cityscapes_dataset(data_root / "cityscapes", split="val")
    if len(source) != 24966 or len(target) != 500:
        raise ValueError(f"Unexpected paired counts: GTA5={len(source)}, Cityscapes={len(target)}")
    report["data_inventory"] = dict(source_pairs=len(source), target_pairs=len(target),
        source_pair_paths_sha256=pair_inventory_digest(source, data_root),
        target_pair_paths_sha256=pair_inventory_digest(target, data_root),
        scope="paired_paths_not_full_file_integrity_scan")
    report["smoke_protocol"] = dict(steps=20, target_inference_samples=5, data_seed=0,
        optimizer="AdamW", lr=1e-4, weight_decay=0.05, betas=[0.9, 0.999],
        clip_norm=1.0, batch_size=1, workers=0, dtype="float32", crop_size=[512, 512],
        source_transform="existing_project_scale_crop_flip_with_valid_pixel_fallback",
        target_transform="aspect_preserving_resize_center_crop_diagnostic_only",
        normalization="project_RGB_inverse_to_BGR255_then_upstream_BGR_to_RGB_normalize",
        accuracy_evaluation=False, checkpoint_saved=False, formal_baseline=False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    order = torch.randperm(len(source), generator=torch.Generator().manual_seed(0)).tolist()[:20]
    target_order = torch.randperm(len(target), generator=torch.Generator().manual_seed(0)).tolist()[:5]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.05)
    tracked = model.backbone.reins.scale
    initial = tracked.detach().clone()
    frozen_reference = model.backbone.patch_embed.proj.weight.detach().clone()
    records = []
    report["real_training_records"] = records  # Retain partial evidence on failure.
    report["stage"] = "real_source_optimization"
    model.train(True)
    max_error = 0.
    for iteration, index in enumerate(order, 1):
        item = source[index]
        batch, error = prepare_sample(model, item, training=True)
        max_error = max(max_error, error)
        optimizer.zero_grad(set_to_none=True)
        values = model.loss(batch["inputs"], batch["data_samples"])
        total = validate_losses(values)
        total.backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad and parameter.grad is not None:
                raise ValueError(f"Frozen backbone gradient: {name}")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in parameters):
            raise FloatingPointError("Non-finite parameter after optimizer update")
        record = dict(iteration=iteration, sample_id=item["id"], loss=total.item(),
                      gradient_norm=norm.item(), valid_pixel_count=(item["label"] != 255).sum().item(),
                      losses={key: value.item() for key, value in values.items()})
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "losses"}), flush=True)
        del values, total, batch
    if torch.equal(initial, tracked.detach()):
        raise ValueError("REIN adapter scale did not change during optimization")
    if not torch.equal(frozen_reference, model.backbone.patch_embed.proj.weight.detach()):
        raise ValueError("Frozen pretrained patch weight changed")
    optimizer.zero_grad(set_to_none=True)
    report["stage"] = "real_target_inference"
    model.eval()
    targets = []
    report["target_inference_records"] = targets
    with torch.no_grad():
        for index in target_order:
            item = center_target_crop(target[index])
            batch, error = prepare_sample(model, item, training=False)
            max_error = max(max_error, error)
            scores = model.encode_decode(batch["inputs"], [x.metainfo for x in batch["data_samples"]])
            validate_prediction(scores)
            targets.append(dict(sample_id=item["id"], semantic_score_shape=list(scores.shape)))
            del scores, batch
    return dict(real_data_smoke_ok=True, phase=16, optimizer_steps=20, adapter_parameter_changed=True,
                frozen_patch_weight_unchanged=True, max_normalization_roundtrip_error=max_error,
                target_inference_count=5)
