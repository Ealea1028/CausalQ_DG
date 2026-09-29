"""Verify adapted REIN dataset/pipelines without model, optimizer or GPU use."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

from tools.audit_rein_real_data import audit_report
from tools.check_rein_backbone import REIN_SHA
from tools.check_rein_runtime import git_output
from tools.inspect_rein_protocol import CONFIG
from tools.rein_protocol_adapter import adapt_protocol, register_data_transforms, validate_train_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    report = dict(ok=False, stage="preflight", purpose="data_pipeline_gate_not_training",
        git_sha=git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        formal_training_authorized=False)
    try:
        if hashlib.sha256(args.inventory.read_bytes()).hexdigest() != \
                "2cdf617c9370735ac275c4875af52f3a017898dd690ff3cc793298322298e64a":
            raise ValueError("Accepted protocol inventory hash mismatch")
        inventory = json.loads(args.inventory.read_text())
        if inventory.get("ok") is not True or inventory.get("stage") != "complete":
            raise ValueError("Upstream protocol inventory did not complete")
        if git_output(args.rein_root, "rev-parse", "HEAD") != REIN_SHA or \
                git_output(args.rein_root, "status", "--porcelain"):
            raise ValueError("Upstream must remain pinned and clean")
        # Audit the accepted smoke before adapting its data protocol.
        previous = args.inventory.parent / "rein_phase16_real_data_17883de_v1.json"
        audit_report(json.loads(previous.read_text()))
        sys.path.insert(0, str(args.rein_root.resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            import numpy as np
            from mmengine.config import Config
            from mmengine.registry import init_default_scope
            from mmseg.registry import DATASETS
            from causalq.datasets.gta5 import resolve_flat_payload_root
            from causalq.datasets.segmentation import gta5_dataset, cityscapes_dataset

            init_default_scope("mmseg")
            np.random.seed(0)
            register_data_transforms()
            config = adapt_protocol(Config.fromfile(args.rein_root / CONFIG).to_dict(), args.data_root,
                resolve_flat_payload_root(args.data_root / "gta5/images"),
                resolve_flat_payload_root(args.data_root / "gta5/labels_trainIds"))
            report["adapted_protocol"] = config
            report["stage"] = "pairing"
            source = DATASETS.build(config["train_dataloader"]["dataset"])
            target = DATASETS.build(config["val_dataloader"]["dataset"])
            expected_source = gta5_dataset(args.data_root / "gta5", transform=None)
            expected_target = cityscapes_dataset(args.data_root / "cityscapes", split="val")
            def compare_pairs(actual, expected, count):
                pairs = {(str(Path(row["img_path"]).resolve()), str(Path(row["seg_map_path"]).resolve()))
                         for row in (actual.get_data_info(i) for i in range(len(actual)))}
                known = {(str(p.image.resolve()), str(p.label.resolve())) for p in expected.pairs}
                if len(actual) != count or pairs != known:
                    raise ValueError(f"MMSeg/project pairing mismatch, expected {count}")
            compare_pairs(source, expected_source, 24966)
            compare_pairs(target, expected_target, 500)
            report.update(source_pairs=len(source), target_pairs=len(target), exact_project_pairing=True)
            report["stage"] = "pipeline_samples"
            report["samples"] = []
            for name, dataset in (("gta5", source), ("cityscapes_val", target)):
                for index in range(5):
                    item = dataset[index]
                    inputs, sample = item["inputs"], item["data_samples"]
                    label = sample.gt_sem_seg.data[0].numpy()
                    ids = validate_train_ids(label)
                    if not np.any(label != 255):
                        raise ValueError("Empty pipeline label")
                    if name == "gta5":
                        if tuple(inputs.shape[1:]) != label.shape or max(label.shape) > 512:
                            raise ValueError("Mismatched source crop")
                    elif tuple(label.shape) != tuple(sample.metainfo["ori_shape"]):
                        raise ValueError("Target ground truth must retain original resolution")
                    report["samples"].append(dict(dataset=name, index=index,
                        input_shape=list(inputs.shape), label_shape=list(label.shape), train_ids=ids))
        report.update(ok=True, stage="complete", model_built=False, optimizer_steps=0,
                      checkpoint_saved=False, target_labels_optimized=False)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
