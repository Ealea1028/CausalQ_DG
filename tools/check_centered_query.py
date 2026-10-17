"""Phase18 bounded centered-query engineering gate; no formal training or target."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import torch
import yaml

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, source_manifest_metadata
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.centered_query import validate_centered_config
from causalq.utils.seed import seed_everything
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from tools.audit_strict_source_pilot import audit_pilot
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.deterministic_source_ce import BACKEND, cuda_preflight, deterministic_source_ce
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import precision, restore_compact, tree_digest
from tools.strict_source_runner import train_segment
from tools.train_strict_source import file_sha

ROOT = Path(__file__).resolve().parents[1]
UPDATES, HOLDOUT = 20, 8


def centered_diagnostics(model):
    bank = model.query_bank
    if not bank.centered_residuals:
        raise ValueError("Centered gate requires explicit centered parameterization")
    residual = bank.get_centered_residuals()
    result = {"residual_mean_max_error": float(residual.mean(1).abs().max().detach()),
              "query_mean_anchor_max_error": float((bank(1)[0].mean(1) - bank.get_anchors()).abs().max().detach())}
    if any(not torch.isfinite(torch.tensor(value)) or value > 1e-6 for value in result.values()):
        raise ValueError("Centered query algebra exceeds 1e-6")
    return result


def restore_centered(model, optimizer, path, metadata):
    payload = load_training_checkpoint(path)
    if (model.query_bank.centered_residuals is not True or metadata.get("centered_residuals") is not True
            or payload["metadata"] != metadata or payload["iteration"] != UPDATES):
        raise ValueError("Centered checkpoint semantic mode/provenance mismatch")
    return restore_compact(model, optimizer, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (ROOT, args.data_root, args.pretrained_root)):
        parser.error("Centered evidence must be outside source/data/weights")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 18, "scope": "m1_centered_query_engineering20",
              "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 0,
              "authorized_optimizer_updates": UPDATES, "authorized_holdout_samples": HOLDOUT,
              "formal_training_authorized": False, "target_data_accessed": False,
              "source_miou": None, "target_miou": None, "exact_resume_verified": False,
              "roundtrip_scope": "new_head_same_frozen_backbone", **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Centered gate requires clean exact commit")
        if "tools.train" in sys.modules or "causalq.datasets.cityscapes" in sys.modules:
            raise ValueError("Legacy/target module imported into centered source gate")
        report["source_pilot_audit"] = audit_pilot(ROOT / "experiments/STRICT_DG_PHASE17")
        config_path = ROOT / "configs/strict_dg/gta_dinov3l_m1_centered.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_centered_config(config)
        report["config"] = config
        if not torch.cuda.is_available():
            raise RuntimeError("Real centered DINOv3 gate requires AutoDL CUDA")
        if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1":
            raise ValueError("Use pinned torch2.7.0+cu126 / transformers4.56.1")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
        report["environment"] = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "transformers": version("transformers"), "gpu": torch.cuda.get_device_name(0)}
        seed_everything(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        report["loss_preflight"] = cuda_preflight()
        torch.cuda.reset_peak_memory_stats()
        weights = args.pretrained_root / config["model"]["weights_dir"]
        weight_sha = file_sha(weights / "model.safetensors")
        if weight_sha != config["model"]["weights_sha256"]:
            raise ValueError("Pinned backbone weight SHA256 mismatch")
        train, holdout = build_source_datasets(args.data_root / "gta5",
            TrainTransform(config["train"]["crop_size"], scale_range=config["data"]["random_scale"],
                horizontal_flip_probability=config["data"]["horizontal_flip_probability"],
                min_valid_fraction=config["data"]["min_valid_fraction"], crop_attempts=config["data"]["crop_attempts"]),
            SourceEvalTransform())
        backbone = DINOv3Backbone.from_pretrained(config["model"]["backbone"], weights=weights,
            freeze=True, intermediate_indices=config["model"]["intermediate_indices"], dtype=torch.bfloat16,
            local_files_only=True).to("cuda:0")
        if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
            raise ValueError("Backbone state identity mismatch")
        def make_model():
            return QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1,
                queries_per_class=2, interaction="static", cross_attention_layers=0, num_heads=8,
                temperature=.07, alpha_init=0., centered_residuals=True).to("cuda:0")
        model = make_model()
        report["initial_centering"] = centered_diagnostics(model)
        sample = holdout[0]
        model.eval()
        with torch.inference_mode(), precision("cuda:0", True):
            components = model.forward_components(sample["image"].unsqueeze(0).to("cuda:0"))
            zero_error = float((components.logits - components.base_logits).abs().max())
        del components
        if zero_error != 0:
            raise ValueError("Centered alpha-zero logits differ from base path")
        report["alpha_zero_error"] = zero_error
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
        metadata = {"git_sha": report["git_sha"], "seed": 0, "centered_residuals": True,
            "config_sha256": file_sha(config_path), "config_tree_sha256": tree_digest(config),
            "manifest_sha256": report["manifest_sha256"], "backbone_weights_sha256": weight_sha,
            "sample_workers": 0, "schedule_horizon": 40000, "loss_backend": BACKEND,
            "amp": "bfloat16", "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
            "authorized_optimizer_updates": UPDATES, "variant": "M1_CENTERED",
            "target_data_accessed": False, "environment": report["environment"]}
        report["metadata"] = metadata
        report.update(train_segment(model, optimizer, train, StyleInterventionBank(**{
            key: tuple(value) if isinstance(value, list) else value for key, value in config["style"]["photometric"].items()
        }).to("cuda:0"), config, output, start=0, stop=UPDATES, metadata=metadata))
        report["final_centering"] = centered_diagnostics(model)
        rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        maxima = {}
        for row in rows:
            for name, value in row["gradients"].items():
                maxima[name] = max(maxima.get(name, 0), value)
        for prefix in ("decoder.", "query_bank.class_anchors", "query_bank.query_residuals",
                       "query_head.feature_projection.", "query_head.query_projection.", "query_head.alpha"):
            if not any(value > 0 for name, value in maxima.items() if name.startswith(prefix)):
                raise ValueError(f"Missing centered training gradient group: {prefix}")
        report["gradient_max"] = maxima
        gradient = model.query_bank.query_residuals.grad
        if gradient is None or not torch.isfinite(gradient).all() or float(gradient.norm()) == 0:
            raise ValueError("Centered residual receives no finite nonzero gradient")
        report["residual_gradient_sum_max_error"] = float(gradient.sum(1).abs().max())
        if report["residual_gradient_sum_max_error"] > 1e-6:
            raise ValueError("Raw residual gradient contains an unexpected shared direction")
        original_head, original_opt = tree_digest(trainable_state_dict(model)), tree_digest(optimizer.state_dict())
        model.eval()
        restored = make_model().eval()
        restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
        restore_centered(restored, restored_optimizer, output / "state.pth", metadata)
        if original_head != tree_digest(trainable_state_dict(restored)) or original_opt != tree_digest(restored_optimizer.state_dict()):
            raise ValueError("Centered model/optimizer checkpoint roundtrip differs")
        references, error = [], 0.
        with torch.inference_mode():
            for i in range(HOLDOUT):
                sample = holdout[i]
                image, label = sample["image"].unsqueeze(0).to("cuda:0"), sample["label"].unsqueeze(0).to("cuda:0")
                with precision("cuda:0", True):
                    first, second = model(image), restored(image)
                    ce = deterministic_source_ce(second, label)
                if not torch.isfinite(first).all() or not torch.isfinite(second).all() or not torch.isfinite(ce):
                    raise ValueError("Nonfinite centered source compatibility output")
                error = max(error, float((first - second).abs().max()))
                references.append({"source_id": sample["id"], "ce": float(ce), "shape": list(label.shape)})
        if error > 1e-5 or tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
            raise ValueError("Centered checkpoint prediction/backbone compatibility mismatch")
        report.update(model_roundtrip_exact=True, optimizer_roundtrip_exact=True,
            prediction_roundtrip_max_error=error, holdout_compatibility=references,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        report["ok"] = True
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    (output / "centered_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{file_sha(output / name)}  {name}\n" for name in
        ("centered_report.json", "train.jsonl", "state.pth") if (output / name).exists()), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
