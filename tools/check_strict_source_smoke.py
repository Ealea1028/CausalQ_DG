"""AutoDL-only 20-update M0 source smoke; no long training or target data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys

import torch

from causalq.datasets.strict_source import source_manifest_metadata
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.seed import seed_everything
from tools.audit_strict_source_data import audit_report
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import HOLDOUT_SAMPLES, UPDATES, run_bounded
from tools.train import PROJECT_ROOT, build_training_datasets, load_config, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in
           (args.data_root, args.pretrained_root, PROJECT_ROOT)):
        parser.error("Output must be outside datasets, weights and source checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 17, "scope": "m0_strict_source_smoke_20_updates",
              "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 0,
              "formal_training_authorized": False, "target_data_accessed": False,
              "authorized_optimizer_updates": UPDATES, "authorized_holdout_samples": HOLDOUT_SAMPLES,
              "source_miou": None, "target_miou": None, **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("GPU smoke requires clean exact Git checkout")
        report["source_data_audit"] = audit_report(
            PROJECT_ROOT / "experiments/STRICT_DG_PHASE17/source_data_report.json")
        config = load_config(PROJECT_ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml")
        report["config"] = config
        if not torch.cuda.is_available():
            raise RuntimeError("Real DINOv3 weight loading and smoke require AutoDL CUDA")
        if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1":
            raise ValueError("Use the pinned DINOv3 runtime: torch2.7.0+cu126 / transformers4.56.1")
        report["environment"] = {"python": sys.version, "executable": sys.executable,
                                 "platform": platform.platform(), "torch": str(torch.__version__),
                                 "cuda": torch.version.cuda, "transformers": version("transformers"),
                                 "gpu": torch.cuda.get_device_name(0)}
        if shutil.disk_usage(output).free < 1024**3:
            raise ValueError("Need at least 1GiB free before bounded checkpoint creation")
        weights = args.pretrained_root / config["model"]["weights_dir"]
        weight_hash = sha256(weights / "model.safetensors")
        if weight_hash != config["model"]["weights_sha256"]:
            raise ValueError("Pinned DINOv3 weights SHA256 mismatch")
        report["pretrained_checkpoint_sha256"] = weight_hash
        seed_everything(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        train, holdout = build_training_datasets(config, args.data_root)
        torch.cuda.reset_peak_memory_stats()
        backbone = DINOv3Backbone.from_pretrained(
            config["model"]["backbone"], weights=weights, freeze=True,
            intermediate_indices=config["model"]["intermediate_indices"],
            dtype=torch.bfloat16, local_files_only=True).to("cuda:0")

        def make_model():
            return QuerySegmentor(
                backbone, decoder_channels=config["model"]["decoder_channels"],
                num_classes=config["data"]["num_classes"], dropout=config["model"]["dropout"],
                queries_per_class=2, interaction="static", cross_attention_layers=0,
                num_heads=config["query"]["num_heads"], temperature=config["query"]["temperature"],
                alpha_init=0.0, learned_null=False)

        model = make_model().to("cuda:0")
        photo = config["style"]["photometric"]
        style_bank = StyleInterventionBank(
            brightness=tuple(photo["brightness"]), contrast=tuple(photo["contrast"]),
            saturation=tuple(photo["saturation"]), gamma=tuple(photo["gamma"]),
            temperature=tuple(photo["temperature"]),
            grayscale_probability=photo["grayscale_probability"]).to("cuda:0")
        metadata = {"git_sha": report["git_sha"], "seed": 0, "protocol": config["protocol"],
                    "manifest_sha256": report["manifest_sha256"],
                    "pretrained_checkpoint_sha256": weight_hash,
                    "checkpoint_selection": "fixed_final_after_20_updates", "target_data_accessed": False}
        report.update(run_bounded(model, make_model, train, holdout, style_bank, config, output,
                                  device="cuda:0", amp=True, metadata=metadata))
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (output / "smoke_report.json").write_bytes(payload)
    names = [name for name in ("smoke_report.json", "train.jsonl", "final.pth") if (output / name).is_file()]
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256(output / name)}  {name}\n" for name in names), encoding="utf-8")
    print(payload.decode("utf-8"), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
