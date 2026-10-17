"""Fixed 500-update source-only runner integration pilot; formal training blocked."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

import torch
import yaml

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import (SourceEvalTransform, build_source_datasets,
                                            load_membership, source_manifest_metadata)
from causalq.protocols.strict_dg import validate_strict_config
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.interventions import StyleInterventionBank
from causalq.utils.seed import seed_everything
from causalq.utils.checkpoint import load_training_checkpoint
from tools.audit_strict_source_resume import audit_resume
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.deterministic_source_ce import BACKEND, cuda_preflight
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import tree_digest
from tools.strict_source_runner import train_segment, evaluate_without_training_side_effects, sampler_state

ROOT = Path(__file__).resolve().parents[1]
PILOT_STEPS = 500
BOUNDARY = 250


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pilot(root, git_sha):
    reports, rows = {}, []
    for stage, bounds in (("prefix", (0, BOUNDARY)), ("finish", (BOUNDARY, PILOT_STEPS))):
        directory = root / stage
        report = json.loads((directory / "worker_report.json").read_bytes())
        if (report["ok"] is not True or report["git_sha"] != git_sha or report["git_status"]
                or report["target_data_accessed"] is not False
                or report["formal_training_authorized"] is not False
                or (report["initial_iteration"], report["final_iteration"]) != bounds):
            raise ValueError("Source runner worker/budget mismatch")
        for name, digest in report["files_sha256"].items():
            if file_sha(directory / name) != digest:
                raise ValueError("Source runner worker artifact hash mismatch")
        rows.extend(json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines())
        reports[stage] = report
    if reports["prefix"]["process_id"] == reports["finish"]["process_id"]:
        raise ValueError("Pilot must resume across independent processes")
    if reports["prefix"]["metadata"] != reports["finish"]["metadata"]:
        raise ValueError("Source runner metadata changed across processes")
    if len(rows) != PILOT_STEPS or [r["update"] for r in rows] != list(range(1, PILOT_STEPS + 1)):
        raise ValueError("Source runner trace coverage mismatch")
    train_ids, _ = load_membership()
    order = sampler_state(len(train_ids), 0)["order"]
    if [row["source_id"] for row in rows] != [train_ids[i] for i in order[:PILOT_STEPS]]:
        raise ValueError("Pilot does not follow the frozen source permutation")
    for row in rows:
        if not all(math.isfinite(row[key]) for key in ("objective", "loss_original", "loss_photometric", "alpha", "gradient_norm_pre_clip")):
            raise ValueError("Nonfinite source runner trace")
        if row["objective"] != row["loss_original"] + row["loss_photometric"]:
            raise ValueError("Source runner objective mismatch")
        for key, step in (("lr_used", row["update"] - 1), ("next_lr", row["update"])):
            if not math.isclose(row[key], 1e-4 * (1 - step / 40000) ** 0.9, rel_tol=1e-12):
                raise ValueError("Source runner LR horizon changed")
    final = reports["finish"]
    evaluation = final["evaluation"]
    if (evaluation["sample_count"] != 2497 or evaluation["training_state_unchanged"] is not True
            or evaluation["confusion_counts_exact"] is not True
            or evaluation["seen_ids_sha256"] != final["manifest_sha256"]["gta5_source_val.txt"]):
        raise ValueError("Incomplete source validation integration")
    payload = load_training_checkpoint(root / "finish/state.pth")
    if (payload["iteration"] != PILOT_STEPS or payload["metadata"] != final["metadata"]
            or tree_digest(payload["trainable_model"]) != final["head_sha256"]):
        raise ValueError("Selected fixed-final checkpoint differs from evaluated model")
    return {"optimizer_updates": PILOT_STEPS, "resumed_at": BOUNDARY,
            "selected_checkpoint": "finish/state.pth", "checkpoint_selection": "fixed_final",
            "source_miou": evaluation["source_miou"], "evaluation": evaluation,
            "workers": reports, "exact_resume_previously_accepted": True,
            "pilot_is_formal_m0": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prefix", "finish", "summary"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if any(root.is_relative_to(path.resolve()) for path in (ROOT, args.data_root, args.pretrained_root)):
        parser.error("Pilot output must be outside source, data and weights trees")
    output = root if args.stage == "summary" else root / args.stage
    if args.stage != "summary":
        output.mkdir(parents=True, exist_ok=False)
    path = output / ("pilot_report.json" if args.stage == "summary" else "worker_report.json")
    if path.exists():
        parser.error("Refusing to overwrite pilot evidence")
    report = {"schema_version": 1, "phase": 17, "scope": "source_runner_pilot500",
              "created_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
              "process_id": os.getpid(), "formal_training_authorized": False,
              "target_data_accessed": False, "target_miou": None, **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Source pilot requires clean exact commit")
        if "causalq.datasets.cityscapes" in sys.modules or "tools.train" in sys.modules:
            raise ValueError("Legacy/target module imported into source-only runner")
        report["resume_audit"] = audit_resume(ROOT / "experiments/STRICT_DG_PHASE17/resume_report.json")
        if args.stage == "summary":
            report.update(verify_pilot(root, report["git_sha"]))
        else:
            config_path = ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_strict_config(config)
            if not torch.cuda.is_available():
                raise RuntimeError("Real source runner pilot requires AutoDL CUDA")
            if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1":
                raise ValueError("Use pinned torch 2.7.0+cu126 / transformers 4.56.1")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
            report["environment"] = {"python": sys.version, "executable": sys.executable,
                                     "platform": platform.platform(), "torch": str(torch.__version__),
                                     "cuda": torch.version.cuda, "transformers": version("transformers"),
                                     "gpu": torch.cuda.get_device_name(0)}
            seed_everything(0)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            report["loss_preflight"] = cuda_preflight()
            torch.cuda.reset_peak_memory_stats()
            report["source_validation_geometry"] = {**config["validation"], "gt_resolution": "resized_then_ignore_padded"}
            weights = args.pretrained_root / config["model"]["weights_dir"]
            if file_sha(weights / "model.safetensors") != config["model"]["weights_sha256"]:
                raise ValueError("Backbone weight hash mismatch")
            transform = TrainTransform(config["train"]["crop_size"], scale_range=config["data"]["random_scale"],
                                       horizontal_flip_probability=config["data"]["horizontal_flip_probability"],
                                       min_valid_fraction=config["data"]["min_valid_fraction"], crop_attempts=config["data"]["crop_attempts"])
            train, holdout = build_source_datasets(args.data_root / "gta5", transform, SourceEvalTransform())
            _, val_ids = load_membership()
            backbone = DINOv3Backbone.from_pretrained(config["model"]["backbone"], weights=weights,
                freeze=True, intermediate_indices=config["model"]["intermediate_indices"],
                dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
            if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
                raise ValueError("Backbone state identity mismatch")
            model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=0.1,
                queries_per_class=2, interaction="static", cross_attention_layers=0, num_heads=8,
                temperature=0.07, alpha_init=0.0).to("cuda:0")
            style = StyleInterventionBank(**{key: tuple(value) if isinstance(value, list) else value
                for key, value in config["style"]["photometric"].items()}).to("cuda:0")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
            metadata = {"git_sha": report["git_sha"], "seed": 0, "config_sha256": file_sha(config_path),
                        "config_tree_sha256": tree_digest(config), "manifest_sha256": report["manifest_sha256"],
                        "backbone_weights_sha256": config["model"]["weights_sha256"], "backbone_sha256": BACKBONE_STATE_SHA,
                        "environment": report["environment"], "loss_backend": BACKEND, "sample_workers": 0,
                        "schedule_horizon": 40000, "pilot_updates": PILOT_STEPS, "amp": "bfloat16",
                        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
                        "checkpoint_selection": "fixed_final_pilot500", "target_data_accessed": False}
            report["metadata"] = metadata
            if args.stage == "prefix":
                (root / "config.yaml").write_bytes(config_path.read_bytes())
                (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                (root / "git_commit.txt").write_text(report["git_sha"] + "\n", encoding="utf-8")
            else:
                previous = json.loads((root / "prefix/worker_report.json").read_bytes())
                if previous["ok"] is not True or any(file_sha(root / "prefix" / name) != digest
                        for name, digest in previous["files_sha256"].items()):
                    raise ValueError("Prefix checkpoint audit/hash mismatch")
                if json.loads((root / "metadata.json").read_bytes()) != metadata or file_sha(root / "config.yaml") != metadata["config_sha256"]:
                    raise ValueError("Frozen run config/metadata mismatch")
            start, stop = (0, BOUNDARY) if args.stage == "prefix" else (BOUNDARY, PILOT_STEPS)
            report.update(train_segment(model, optimizer, train, style, config, output, start=start,
                stop=stop, metadata=metadata, resume=root / "prefix/state.pth" if start else None))
            if args.stage == "finish":
                report["evaluation"] = evaluate_without_training_side_effects(model, optimizer, holdout, val_ids, output)
            report["files_sha256"] = {name: file_sha(output / name) for name in
                ("state.pth", "train.jsonl", "source_eval.jsonl") if (output / name).exists()}
            report.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        report["ok"] = True
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
