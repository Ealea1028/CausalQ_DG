"""Inspect pinned upstream REIN configuration without building a model/dataset."""

import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys

from tools.check_rein_backbone import REIN_SHA
from tools.check_rein_runtime import git_output


CONFIG = "configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py"


def protocol_fields(config):
    """Keep effective inherited loader/optimizer/evaluation fields, not just leaf text."""
    required = ("train_dataloader", "val_dataloader", "test_dataloader",
                "optim_wrapper", "param_scheduler", "train_cfg", "val_evaluator")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Incomplete upstream configuration: {missing}")
    keys = required + ("test_evaluator", "default_hooks", "env_cfg", "randomness")
    fields = {key: config[key] for key in keys if key in config}
    model = config["model"]
    fields["model_contract"] = {key: model.get(key) for key in
                               ("type", "data_preprocessor", "train_cfg", "test_cfg")}
    fields["backbone_contract"] = model.get("backbone")
    fields["head_num_classes"] = model.get("decode_head", {}).get("num_classes")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rein-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.rein_root.resolve()
    report = dict(ok=False, purpose="read_only_protocol_inventory_not_training",
                  git_sha=git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
                  rein_root=str(root), formal_training_authorized=False,
                  source_scope="GTA5_to_Cityscapes_only", stage="preflight")
    try:
        report["rein_sha"] = git_output(root, "rev-parse", "HEAD")
        if report["rein_sha"] != REIN_SHA or git_output(root, "status", "--porcelain"):
            raise ValueError("Upstream source must be pinned and clean")
        sys.path.insert(0, str(root))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.config import Config

            config = Config.fromfile(root / CONFIG)
            report["effective_protocol"] = protocol_fields(config.to_dict())
            # Dataset implementations reveal train-ID remapping conventions. Read
            # pinned source only: never instantiate a dataset or execute a pipeline.
            dataset_package = importlib.import_module("rein.datasets")
            dataset_dir = Path(dataset_package.__file__).resolve().parent
            if not dataset_dir.is_relative_to(root):
                raise ValueError("Dataset package is outside pinned upstream checkout")
            report["dataset_implementations"] = []
            for path in sorted(dataset_dir.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                if "gta" in source.lower() or "cityscapes" in source.lower():
                    report["dataset_implementations"].append(dict(
                        path=path.relative_to(root).as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=source))
        # Hash every config source, including inherited dataset/optimizer bases.
        report["config_inventory"] = [dict(path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted((root / "configs").rglob("*.py"))]
        report["upstream_entry_config"] = CONFIG
        report["upstream_entry_config_sha256"] = hashlib.sha256((root / CONFIG).read_bytes()).hexdigest()
        report["review_required"] = ["train-ID handling and nested local paths",
            "source transforms versus project baseline", "effective batch and optimizer groups",
            "full Cityscapes-only validation with no target optimization",
            "checkpoint retention and disk budget"]
        report.update(ok=True, stage="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
