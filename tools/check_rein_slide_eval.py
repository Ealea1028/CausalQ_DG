"""Five-image original-resolution Cityscapes REIN evaluation plumbing gate."""

import argparse
import contextlib
import importlib
import json
from pathlib import Path
import sys
import traceback

from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions
from tools.rein_schedule_smoke import DATA_SHA, digest, restore_protocol_config

SCHEDULE_SHA = "ea528c623bd071a7aeb8214e2b1434073fe4ef00055478fc073fd482c3abbfad"
COMPACT_SHA = "d0cf1dd59fd7eb2ee9e2aa280b47e03819b8d26d814f12ab9f6162f7bcd410af"
PRODUCER = "3ca67315bc4f1dc91f1891de3f305e3162a0a53a"


def restore_compact(model, state):
    import torch

    named = dict(model.named_parameters()) | dict(model.named_buffers())
    expected = {name for name, p in model.named_parameters() if p.requires_grad} | set(dict(model.named_buffers()))
    if set(state) != expected:
        raise ValueError("Compact checkpoint coverage mismatch")
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != named[name].shape or tensor.dtype != named[name].dtype or not torch.isfinite(tensor).all().item():
                raise ValueError(f"Invalid compact tensor: {name}")
            named[name].copy_(tensor.to(named[name].device))


def confusion(prediction, label):
    import numpy as np

    prediction, label = np.asarray(prediction), np.asarray(label)
    if prediction.shape != label.shape or label.ndim != 2:
        raise ValueError("Prediction/GT must share original two-dimensional geometry")
    if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(label.dtype, np.integer):
        raise ValueError("Integer train IDs required")
    if np.any((prediction < 0) | (prediction > 18)) or np.any(((label < 0) | (label > 18)) & (label != 255)):
        raise ValueError("Invalid prediction/GT train IDs")
    valid = label != 255
    return np.bincount(19 * label[valid].astype(np.int64) + prediction[valid], minlength=361).reshape(19, 19)


def evaluate_target(model, dataset, metric, count):
    """Shared bounded metric path; never optimizes target labels."""
    import torch
    import numpy as np

    if count not in (5, 50) or len(dataset) != 500:
        raise ValueError("Expected bounded sample count and full Cityscapes dataset")
    metric.dataset_meta = dataset.metainfo
    matrix = np.zeros((19, 19), dtype=np.int64)
    samples = []
    model.eval()
    for index in range(count):
        item = dataset[index]
        gt = item['data_samples'].gt_sem_seg.data[0].cpu().numpy().copy()
        shape = tuple(item['data_samples'].metainfo['ori_shape'])
        if gt.shape != shape or shape != (1024, 2048):
            raise ValueError("Original Cityscapes GT resolution required")
        with torch.no_grad():
            outputs = model.test_step(dict(inputs=[item['inputs']], data_samples=[item['data_samples']]))
        output = outputs[0]
        scores = output.seg_logits.data
        pred = output.pred_sem_seg.data[0].cpu().numpy()
        if tuple(scores.shape) != (19, *shape) or not torch.isfinite(scores).all().item():
            raise ValueError("Invalid original-resolution logits")
        if not np.array_equal(output.gt_sem_seg.data[0].cpu().numpy(), gt):
            raise ValueError("Inference changed target GT")
        current = confusion(pred, gt)
        metric.process({}, [output.to_dict()])
        expected = (current.diagonal(), current.sum(0) + current.sum(1) - current.diagonal(), current.sum(0), current.sum(1))
        if any(not np.array_equal(a.cpu().numpy(), b) for a, b in zip(metric.results[-1], expected)):
            raise ValueError("Official/independent metric count mismatch")
        matrix += current
        samples.append(dict(index=index, img_path=output.metainfo['img_path'],
                            prediction_shape=list(pred.shape), gt_shape=list(gt.shape), valid_pixel_count=int(current.sum())))
        print(f"slide_sample_complete={index + 1}/{count}", file=sys.stderr)
        del outputs, output, scores
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    valid = union > 0
    iou = np.divide(matrix.diagonal(), union, out=np.zeros(19, dtype=float), where=valid)
    official = metric.compute_metrics(metric.results)
    if abs(float(official['mIoU']) / 100 - float(iou[valid].mean())) > 5.1e-5:
        raise ValueError("Rounded official/independent mIoU mismatch")
    return dict(sample_count=count, samples=samples, diagnostic_miou=float(iou[valid].mean()),
                official_diagnostic_metrics={key: float(value) for key, value in official.items()},
                class_iou=[float(iou[i]) if valid[i] else None for i in range(19)],
                valid_classes=int(valid.sum()), confusion_matrix=matrix.tolist(), total_valid_pixels=int(matrix.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rein-root", "weights", "data-report", "schedule-report", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = dict(ok=False, stage="preflight", purpose="five_image_slide_eval_plumbing_not_accuracy_acceptance",
                  git_sha=git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
                  accuracy_claim=False, formal_training_authorized=False, optimizer_updates=0,
                  target_labels_optimized=False, sample_count=0)
    try:
        if digest(args.schedule_report) != SCHEDULE_SHA or digest(args.checkpoint) != COMPACT_SHA:
            raise ValueError("Accepted schedule report/checkpoint SHA mismatch")
        if digest(args.weights) != WEIGHT_SHA or digest(args.data_report) != DATA_SHA:
            raise ValueError("Frozen weights/data protocol SHA mismatch")
        accepted = json.loads(args.schedule_report.read_text())
        if (accepted.get("ok") is not True or accepted.get("git_sha") != PRODUCER
                or accepted.get("optimizer_updates") != 20 or accepted.get("stage") != "complete"
                or accepted.get("prediction_roundtrip_max_error") != 0):
            raise ValueError("Incomplete scheduled gate evidence")
        if git_output(args.rein_root, "rev-parse", "HEAD") != REIN_SHA or git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("REIN must remain pinned and clean")
        import torch
        import numpy as np
        versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        if not torch.cuda.is_available() or torch.version.cuda != "11.8":
            raise RuntimeError("Accepted CUDA 11.8 GPU required")
        report.update(versions=versions, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      checkpoint_sha256=COMPACT_SHA, training_git_sha=PRODUCER, pretrained_sha256=WEIGHT_SHA)
        torch.manual_seed(0)
        sys.path.insert(0, str(args.rein_root.resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.registry import init_default_scope
            from mmseg.registry import DATASETS, MODELS, METRICS
            from tools.rein_protocol_adapter import register_data_transforms
            init_default_scope("mmseg")
            register_data_transforms()
            config = restore_protocol_config(json.loads(args.data_report.read_text()), args.rein_root)
            if (config.model.test_cfg.mode != "slide" or tuple(config.model.test_cfg.crop_size) != (512, 512)
                    or tuple(config.model.test_cfg.stride) != (341, 341)):
                raise ValueError("Unexpected sliding inference policy")
            report["stage"] = "checkpoint_load"
            model = MODELS.build(config.model)
            model.decode_head.init_weights()
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            loaded = model.backbone.load_state_dict(state, strict=False)
            keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
            validate_load_keys(state, keys, loaded.missing_keys, loaded.unexpected_keys)
            del state
            model.train(True)
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if payload["metadata"]["git_sha"] != PRODUCER or payload["metadata"]["optimizer_updates"] != 20:
                raise ValueError("Compact checkpoint producer mismatch")
            restore_compact(model, payload["model"])
            del payload
            model.cuda().eval()
            dataset = DATASETS.build(config.val_dataloader.dataset)
            if len(dataset) != 500:
                raise ValueError("Expected complete Cityscapes validation dataset")
            metric = METRICS.build(config.val_evaluator)
            metric.dataset_meta = dataset.metainfo
            matrix = np.zeros((19, 19), dtype=np.int64)
            report.update(stage="slide_predictions", samples=[], policy=dict(crop_size=[512, 512], stride=[341, 341]))
            torch.cuda.reset_peak_memory_stats(0)
            for index in range(5):
                item = dataset[index]
                original_gt = item["data_samples"].gt_sem_seg.data[0].cpu().numpy().copy()
                original_shape = tuple(item["data_samples"].metainfo["ori_shape"])
                if original_gt.shape != original_shape or original_shape != (1024, 2048):
                    raise ValueError("Cityscapes GT must retain original 1024x2048 resolution")
                with torch.no_grad():
                    outputs = model.test_step(dict(inputs=[item["inputs"]], data_samples=[item["data_samples"]]))
                output = outputs[0]
                scores = output.seg_logits.data
                prediction = output.pred_sem_seg.data[0].cpu().numpy()
                if tuple(scores.shape) != (19, *original_shape) or not torch.isfinite(scores).all().item():
                    raise ValueError("Invalid original-resolution semantic scores")
                if not np.array_equal(output.gt_sem_seg.data[0].cpu().numpy(), original_gt):
                    raise ValueError("Inference altered target GT")
                current = confusion(prediction, original_gt)
                if current.sum() != np.count_nonzero(original_gt != 255):
                    raise ValueError("Metric valid-pixel accounting mismatch")
                metric.process({}, [output.to_dict()])
                official = metric.results[-1]
                expected_areas = (current.diagonal(), current.sum(0) + current.sum(1) - current.diagonal(),
                                  current.sum(0), current.sum(1))
                if any(not np.array_equal(actual.cpu().numpy(), expected)
                       for actual, expected in zip(official, expected_areas)):
                    raise ValueError("Official IoUMetric and independent confusion counts disagree")
                matrix += current
                report["samples"].append(dict(index=index, img_path=output.metainfo["img_path"],
                    prediction_shape=list(prediction.shape), gt_shape=list(original_gt.shape),
                    valid_pixel_count=int(current.sum())))
                report["sample_count"] += 1
                print(f"slide_sample_complete={index + 1}/5", file=sys.stderr)
                del outputs, output, scores
            union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
            valid = union > 0
            iou = np.divide(matrix.diagonal(), union, out=np.zeros(19, dtype=float), where=valid)
            official_metrics = metric.compute_metrics(metric.results)
            if abs(float(official_metrics["mIoU"]) / 100 - float(iou[valid].mean())) > 5.1e-5:
                raise ValueError("Official rounded mIoU differs from independent calculation")
            report.update(ok=True, stage="complete", diagnostic_miou=float(iou[valid].mean()),
                          official_diagnostic_metrics={key: float(value) for key, value in official_metrics.items()},
                          class_iou=[float(iou[i]) if valid[i] else None for i in range(19)],
                          valid_classes=int(valid.sum()), confusion_matrix=matrix.tolist(),
                          total_valid_pixels=int(matrix.sum()), peak_allocated_gib=round(torch.cuda.max_memory_allocated(0)/2**30, 3),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved(0)/2**30, 3))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
