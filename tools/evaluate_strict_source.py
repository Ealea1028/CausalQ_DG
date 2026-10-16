"""Read-only full GTA5 holdout evaluation of the immutable 20-update checkpoint."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import torch

from causalq.datasets.strict_source import load_membership, source_manifest_metadata
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.audit_strict_source_smoke import (audit_smoke, CHECKPOINT_SHA, BACKBONE_STATE_SHA,
                                            SMOKE_PRODUCER_SHA)
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import tree_digest
from tools.strict_source_evaluation import evaluate_holdout
from tools.train import PROJECT_ROOT, build_training_datasets, load_config, sha256


def restore_for_evaluation(model, checkpoint_path, smoke_report):
    if sha256(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Original smoke checkpoint SHA256 mismatch")
    payload = load_training_checkpoint(checkpoint_path)
    if (payload["iteration"] != 20 or payload["metadata"]["git_sha"] != SMOKE_PRODUCER_SHA
            or payload["metadata"]["seed"] != 0
            or payload["metadata"]["target_data_accessed"] is not False
            or payload["metadata"]["manifest_sha256"] != smoke_report["manifest_sha256"]
            or payload["metadata"]["pretrained_checkpoint_sha256"] != smoke_report["pretrained_checkpoint_sha256"]
            or payload["metadata"]["protocol"] != smoke_report["config"]["protocol"]
            or payload["metadata"]["checkpoint_selection"] != "fixed_final_after_20_updates"):
        raise ValueError("Original checkpoint provenance mismatch")
    expected = set(trainable_state_dict(model))
    if set(payload["trainable_model"]) != expected:
        raise ValueError("Original checkpoint parameter coverage mismatch")
    result = model.load_state_dict(payload["trainable_model"], strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError("Unexpected source checkpoint restoration coverage")
    if tree_digest(trainable_state_dict(model)) != smoke_report["final_trainable_state_sha256"]:
        raise ValueError("Restored head/query state differs from original smoke")
    return payload["iteration"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in
           (args.data_root, args.pretrained_root, args.smoke_dir, PROJECT_ROOT)):
        parser.error("New evidence must be outside source/data/weights/original smoke trees")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 17, "scope": "full_source_holdout_20update_diagnostic",
              "created_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates_this_run": 0,
              "checkpoint_training_updates": 20, "formal_training_authorized": False,
              "diagnostic_only": True, "exact_resume_verified": False,
              "target_data_accessed": False, "target_miou": None,
              "checkpoint_selection": "immutable_preselected_smoke_final", **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Source evaluation requires clean exact checkout")
        # Integrity and complete trace audit happen BEFORE model/dataset access.
        report["smoke_audit"] = audit_smoke(args.smoke_dir)
        smoke = json.loads((args.smoke_dir / "smoke_report.json").read_bytes())
        if sha256(args.smoke_dir / "final.pth") != CHECKPOINT_SHA:
            raise ValueError("Original smoke checkpoint SHA256 mismatch before model access")
        config = load_config(PROJECT_ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml")
        if smoke["config"] != config:
            raise ValueError("Source protocol changed after the original smoke")
        report["source_validation_geometry"] = {**config["validation"], "gt_resolution": "resized_then_ignore_padded"}
        if not torch.cuda.is_available():
            raise RuntimeError("Full DINOv3 source evaluation requires AutoDL CUDA")
        if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1":
            raise ValueError("Use pinned DINOv3 torch2.7.0+cu126 / transformers4.56.1 runtime")
        report["environment"] = {"python": sys.version, "executable": sys.executable,
                                 "platform": platform.platform(), "torch": str(torch.__version__),
                                 "cuda": torch.version.cuda, "transformers": version("transformers"),
                                 "gpu": torch.cuda.get_device_name(0)}
        seed_everything(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        weights = args.pretrained_root / config["model"]["weights_dir"]
        weight_sha = sha256(weights / "model.safetensors")
        if weight_sha != config["model"]["weights_sha256"]:
            raise ValueError("Pinned source backbone weights hash mismatch")
        report["pretrained_checkpoint_sha256"] = weight_sha
        _, holdout = build_training_datasets(config, args.data_root)
        _, val_ids = load_membership()
        torch.cuda.reset_peak_memory_stats()
        backbone = DINOv3Backbone.from_pretrained(
            config["model"]["backbone"], weights=weights, freeze=True,
            intermediate_indices=config["model"]["intermediate_indices"], dtype=torch.bfloat16,
            local_files_only=True).to("cuda:0")
        model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=0.1,
                               queries_per_class=2, interaction="static", cross_attention_layers=0,
                               num_heads=8, temperature=0.07, alpha_init=0.0).to("cuda:0").eval()
        before_backbone = tree_digest(backbone.state_dict())
        if before_backbone != BACKBONE_STATE_SHA:
            raise ValueError("Reloaded backbone state differs from original smoke")
        report["restored_iteration"] = restore_for_evaluation(model, args.smoke_dir / "final.pth", smoke)
        report["smoke_audit"]["checkpoint_bytes_independently_audited"] = True
        head_before = tree_digest(trainable_state_dict(model))
        report.update(evaluate_holdout(model, holdout, val_ids, output,
                                        compatibility=smoke["holdout_compatibility"]))
        if report["seen_ids_sha256"] != report["manifest_sha256"]["gta5_source_val.txt"]:
            raise ValueError("Evaluated source IDs do not match the frozen holdout manifest hash")
        if (head_before != tree_digest(trainable_state_dict(model))
                or before_backbone != tree_digest(backbone.state_dict())):
            raise ValueError("Read-only source evaluation modified model state")
        audit_smoke(args.smoke_dir)
        if sha256(args.smoke_dir / "final.pth") != CHECKPOINT_SHA:
            raise ValueError("Original checkpoint changed during source evaluation")
        report.update(ok=True, checkpoint_sha256=CHECKPOINT_SHA,
                      checkpoint_bytes_independently_audited=True, model_unchanged=True,
                      original_smoke_files_unchanged=True,
                      backbone_state_sha256=before_backbone, restored_trainable_state_sha256=head_before,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    (output / "source_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{sha256(output / name)}  {name}\n" for name in
        ("source_evaluation.json", "source_eval.jsonl") if (output / name).exists()), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
