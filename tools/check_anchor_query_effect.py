"""Read-only anchor/full/residual computation gate on the accepted M1 checkpoint."""

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

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, source_manifest_metadata
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.interventions import StyleInterventionBank
from causalq.protocols.centered_query import validate_centered_config
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.audit_centered_query import audit_centered, REPORT_SHA, SUMS_SHA, TRACE_SHA, CHECKPOINT_SHA
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.check_centered_query import centered_diagnostics
from tools.deterministic_source_ce import deterministic_source_ce
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import tree_digest, precision
from tools.strict_source_resume import capture_rng
from tools.train_strict_source import file_sha

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 8


def audit_original(directory):
    expected = {"centered_report.json": REPORT_SHA, "SHA256SUMS": SUMS_SHA,
                "train.jsonl": TRACE_SHA, "state.pth": CHECKPOINT_SHA}
    for name, digest in expected.items():
        if file_sha(directory / name) != digest:
            raise ValueError(f"Original centered artifact SHA256 mismatch: {name}")
    return expected


def restore_anchor_source(model, path, report):
    if file_sha(path) != CHECKPOINT_SHA:
        raise ValueError("Original centered checkpoint SHA256 mismatch")
    payload = load_training_checkpoint(path)
    if (payload["iteration"] != 20 or payload["metadata"] != report["metadata"]
            or payload["metadata"]["centered_residuals"] is not True
            or model.query_bank.centered_residuals is not True or model.interaction != "static"):
        raise ValueError("Anchor source checkpoint semantic/provenance mismatch")
    expected = set(trainable_state_dict(model))
    if set(payload["trainable_model"]) != expected:
        raise ValueError("Anchor checkpoint parameter coverage mismatch")
    result = model.load_state_dict(payload["trainable_model"], strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError("Unexpected anchor checkpoint restoration coverage")
    if tree_digest(trainable_state_dict(model)) != report["head_sha256"]:
        raise ValueError("Anchor source head identity mismatch")


def component_checks(model, output):
    full = model.get_query_effect(output=output).float()
    anchor = model.get_anchor_query_effect(output=output)
    residual = model.get_residual_contribution(output=output)
    if any(not torch.isfinite(value).all() for value in (output.logits, output.anchor_logits, full, anchor, residual)):
        raise ValueError("Nonfinite anchor/full/residual component")
    error = float((full - (anchor + residual)).abs().max())
    if error > 1e-6:
        raise ValueError("FP32 effect decomposition identity exceeds 1e-6")
    for c in range(output.logits.shape[1]):
        anchored = output.anchor_counterfactual_logits(c)
        zeroed = output.counterfactual_logits(c)
        if not torch.equal(anchored[:, c], output.anchor_logits[:, c]) or not torch.equal(zeroed[:, c], output.base_logits[:, c]):
            raise ValueError("Anchor/zero counterfactual class routing differs")
        for part in (slice(None, c), slice(c + 1, None)):
            if not torch.equal(anchored[:, part], output.logits[:, part]) or not torch.equal(zeroed[:, part], output.logits[:, part]):
                raise ValueError("Counterfactual modified another class")
    # Quantized base-logit addition can lose low bits. It is diagnostic only;
    # canonical effects retain the established pre-addition alpha*delta contract.
    roundoff = float(((output.anchor_logits.float() - output.base_logits.float()) - anchor).abs().max())
    return {"decomposition_max_error": error, "counterfactual_routing_exact": True,
            "full_abs_max": float(full.abs().max()), "anchor_abs_max": float(anchor.abs().max()),
            "residual_abs_max": float(residual.abs().max()), "base_addition_roundoff_max": roundoff}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--centered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (ROOT, args.data_root, args.pretrained_root, args.centered_dir)):
        parser.error("New anchor evidence must be outside source/data/weights/original centered run")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 19, "scope": "anchor_component_readonly8x2",
              "created_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates_this_run": 0,
              "checkpoint_training_updates": 20, "source_miou": None, "target_miou": None,
              "formal_training_authorized": False, "target_data_accessed": False,
              "new_loss_enabled": False, **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Anchor gate requires clean exact checkout")
        if "tools.train" in sys.modules or "causalq.datasets.cityscapes" in sys.modules:
            raise ValueError("Legacy/target module imported into anchor gate")
        report["centered_audit"] = audit_centered(ROOT / "experiments/STRICT_DG_PHASE18")
        report["original_artifacts"] = audit_original(args.centered_dir)
        original = json.loads((args.centered_dir / "centered_report.json").read_bytes())
        config = original["config"]
        validate_centered_config(config)
        if not torch.cuda.is_available():
            raise RuntimeError("Real anchor-component DINOv3 inference requires AutoDL CUDA")
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
        torch.cuda.reset_peak_memory_stats()
        weights = args.pretrained_root / config["model"]["weights_dir"]
        if file_sha(weights / "model.safetensors") != config["model"]["weights_sha256"]:
            raise ValueError("Pinned backbone weights differ")
        backbone = DINOv3Backbone.from_pretrained(config["model"]["backbone"], weights=weights,
            freeze=True, intermediate_indices=config["model"]["intermediate_indices"], dtype=torch.bfloat16,
            local_files_only=True).to("cuda:0")
        if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
            raise ValueError("Original backbone state identity mismatch")
        model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1,
            queries_per_class=2, interaction="static", cross_attention_layers=0, num_heads=8,
            temperature=.07, alpha_init=0., centered_residuals=True).to("cuda:0").eval()
        restore_anchor_source(model, args.centered_dir / "state.pth", original)
        report["centering"] = centered_diagnostics(model)
        _, holdout = build_source_datasets(args.data_root / "gta5", TrainTransform((512, 512)), SourceEvalTransform())
        style = StyleInterventionBank(**{key: tuple(value) if isinstance(value, list) else value
            for key, value in config["style"]["photometric"].items()}).to("cuda:0")
        before = tree_digest((model.state_dict(), capture_rng()))
        calls, rows, ce_error = [0], [], 0.
        def count_call(*unused):
            calls[0] += 1
        handle = backbone.register_forward_hook(count_call)
        try:
            with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                for index in range(SAMPLES):
                    sample = holdout[index]
                    reference = original["holdout_compatibility"][index]
                    if sample["id"] != reference["source_id"]:
                        raise ValueError("Anchor source reference membership mismatch")
                    image = sample["image"].unsqueeze(0).to("cuda:0")
                    label = sample["label"].unsqueeze(0).to("cuda:0")
                    photo = style.photometric_view(image)
                    for name, value in (("original", image), ("photometric", photo)):
                        initial_calls = calls[0]
                        with precision("cuda:0", True):
                            components = model.forward_components(value, include_anchor=True)
                            ce = deterministic_source_ce(components.logits, label)
                        if not torch.isfinite(ce) or calls[0] - initial_calls != 1:
                            raise ValueError("Anchor components require finite CE and one backbone pass")
                        if name == "original":
                            ce_error = max(ce_error, abs(float(ce) - reference["ce"]))
                        rows.append({"source_id": sample["id"], "view": name, "ce": float(ce),
                            "input_sha256": tree_digest((value, label)), **component_checks(model, components)})
        finally:
            handle.remove()
        if len(rows) != 16 or calls[0] != 16 or ce_error > 1e-5:
            raise ValueError("Anchor coverage/original checkpoint CE replay mismatch")
        if before != tree_digest((model.state_dict(), capture_rng())):
            raise ValueError("Anchor read-only inference changed model or RNG")
        if audit_original(args.centered_dir) != report["original_artifacts"]:
            raise ValueError("Original centered artifacts changed")
        (output / "components.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report.update(ok=True, sample_count=8, view_count=16, backbone_forward_calls=16,
            rows=rows, original_ce_max_error=ce_error, model_rng_unchanged=True,
            original_files_unchanged=True, checkpoint_sha256=CHECKPOINT_SHA,
            head_sha256=tree_digest(trainable_state_dict(model)),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    (output / "anchor_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{file_sha(output / name)}  {name}\n" for name in
        ("anchor_report.json", "components.jsonl") if (output / name).exists()), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
