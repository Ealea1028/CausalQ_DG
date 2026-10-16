"""Streaming source-only metric gate, independent of the target evaluator."""

from pathlib import Path
import json
import math

import numpy as np
import torch

from causalq.losses import segmentation_cross_entropy
from causalq.metrics import MeanIoU
from causalq.protocols.source_split import manifest_bytes
from tools.strict_source_smoke import precision
import hashlib


def evaluate_holdout(model, dataset, expected_ids, output: Path, *, compatibility=None,
                     num_classes=19, device="cuda:0", amp=True):
    if len(dataset) != len(expected_ids) or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Full source holdout membership/coverage required")
    references = {row["source_id"]: row for row in (compatibility or [])}
    if not set(references) <= set(expected_ids):
        raise ValueError("Compatibility reference is outside source holdout")
    metric = MeanIoU(num_classes)
    independent = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_total, pixel_loss_sum, image_loss_sum = 0, 0.0, 0.0
    replay_error, seen = 0.0, []
    model.eval()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []
    with (output / "source_eval.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
        for index, expected_id in enumerate(expected_ids):
            sample = dataset[index]
            if sample["id"] != expected_id:
                raise ValueError("Unexpected source holdout ID/order")
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            if image.shape[-2:] != label.shape[-2:] or any(size % 16 for size in label.shape[-2:]):
                raise ValueError("Source evaluation input/label/patch shape mismatch")
            with precision(device, amp):
                logits = model(image)
                if logits.shape != (1, num_classes, *label.shape[-2:]) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid source evaluation logits")
                loss = segmentation_cross_entropy(logits, label)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite source evaluation loss")
            predictions = logits.argmax(1).cpu().numpy().astype(np.int64).reshape(-1)
            target = label.cpu().numpy().astype(np.int64).reshape(-1)
            valid = target != 255
            if not valid.any() or np.any((target[valid] < 0) | (target[valid] >= num_classes)):
                raise ValueError("Invalid/all-ignore source evaluation labels")
            counts = np.bincount(target[valid] * num_classes + predictions[valid],
                                 minlength=num_classes**2).reshape(num_classes, num_classes)
            independent += counts
            metric.update(logits, label)
            valid_count = int(valid.sum())
            valid_total += valid_count
            loss_value = float(loss)
            image_loss_sum += loss_value
            pixel_loss_sum += loss_value * valid_count
            if expected_id in references:
                reference = references[expected_id]
                if list(label.shape) != reference["shape"]:
                    raise ValueError("Restored checkpoint compatibility shape mismatch")
                replay_error = max(replay_error, abs(loss_value - reference["ce"]))
                if replay_error > 1e-5:
                    raise ValueError("Fresh-process source compatibility CE differs by more than 1e-5")
            seen.append(expected_id)
            stream.write(json.dumps({"source_id": expected_id, "shape": list(label.shape),
                                      "valid_pixels": valid_count, "ce": loss_value}) + "\n")
            if (index + 1) % 250 == 0:
                stream.flush()
                print(f"source_holdout_evaluated={index + 1}/{len(expected_ids)}", flush=True)
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise ValueError("Source evaluation changed CPU RNG")
    if cuda_rng and not all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise ValueError("Source evaluation changed CUDA RNG")
    if not np.array_equal(metric.confusion.numpy(), independent) or int(independent.sum()) != valid_total:
        raise ValueError("Framework/independent confusion or valid-pixel counts differ")
    matrix = independent.astype(np.float64)
    intersection = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    present = union > 0
    class_iou = [float(intersection[i] / union[i]) if present[i] else None for i in range(num_classes)]
    independent_miou = float(np.mean([value for value in class_iou if value is not None]))
    computed = metric.compute()
    if not math.isfinite(computed["miou"]) or abs(computed["miou"] - independent_miou) > 1e-12:
        raise ValueError("Framework/independent source mIoU differ")
    for first, second in zip(computed["class_iou"], class_iou):
        if ((first is None) != (second is None)
                or (first is not None and abs(first - second) > 1e-12)):
            raise ValueError("Framework/independent per-class IoU differ")
    return {"sample_count": len(seen), "seen_ids_sha256": hashlib.sha256(manifest_bytes(seen)).hexdigest(),
            "valid_pixel_count": valid_total, "confusion_matrix": independent.tolist(),
            "source_miou": computed["miou"], "class_iou": computed["class_iou"],
            "valid_classes": computed["valid_classes"], "independent_miou": independent_miou,
            "confusion_counts_exact": True, "evaluation_rng_unchanged": True,
            "compatibility_reference_count": len(references), "compatibility_ce_max_error": replay_error,
            "image_mean_ce": image_loss_sum / len(seen), "pixel_weighted_ce": pixel_loss_sum / valid_total,
            "trace_sha256": hashlib.sha256((output / "source_eval.jsonl").read_bytes()).hexdigest()}
