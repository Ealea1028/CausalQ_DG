"""AutoDL fresh-process 8 versus 4+4 source-only continuation diagnostic."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import torch

from causalq.datasets.strict_source import source_manifest_metadata
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.seed import seed_everything
from tools.audit_strict_source_evaluation import audit_evaluation
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_resume import run_segment, compare_states
from tools.strict_source_smoke import tree_digest
from tools.train import PROJECT_ROOT, build_training_datasets, load_config, sha256


def verify_workers(root, git_sha):
    process_ids = []
    workers = {}
    for mode in ("continuous", "prefix", "resume"):
        folder = root / mode
        worker = json.loads((folder / "worker_report.json").read_bytes())
        if (worker.get("ok") is not True or worker["git_sha"] != git_sha
                or worker["git_status"] or worker["mode"] != mode
                or worker["target_data_accessed"] is not False):
            raise ValueError(f"Failed or incompatible worker: {mode}")
        process_ids.append(worker["process_id"])
        workers[mode] = worker
        for name in ("train.jsonl", "state.pth"):
            if sha256(folder / name) != worker["files_sha256"][name]:
                raise ValueError(f"Worker artifact changed: {mode}/{name}")
    if len(set(process_ids)) != 3:
        raise ValueError("Continuation gate requires three independent worker processes")
    return {**compare_states(root / "continuous", root / "prefix", root / "resume"),
            "workers": workers, "independent_worker_processes": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("continuous", "prefix", "resume", "compare"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if any(root.is_relative_to(path.resolve()) for path in
           (args.data_root, args.pretrained_root, PROJECT_ROOT)):
        parser.error("Diagnostic outputs must be outside source/data/weights trees")
    output = root if args.mode == "compare" else root / args.mode
    if args.mode != "compare":
        output.mkdir(parents=True, exist_ok=False)
    report_path = output / ("resume_report.json" if args.mode == "compare" else "worker_report.json")
    if report_path.exists():
        parser.error("Refusing to overwrite evidence")
    report = {"schema_version": 1, "phase": 17, "scope": "source_resume_8_vs_4_plus_4",
              "created_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "process_id": os.getpid(),
              "seed": 0, "formal_training_authorized": False, "target_data_accessed": False,
              "exact_resume_verified": False, "source_miou": None, "target_miou": None,
              **source_manifest_metadata()}
    try:
        report.update(git_sha=_git("rev-parse", "HEAD"), git_status=_git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Continuation diagnostic requires clean exact checkout")
        report["source_evaluation_audit"] = audit_evaluation(
            PROJECT_ROOT / "experiments/STRICT_DG_PHASE17/source_evaluation.json")
        if args.mode == "compare":
            report.update(verify_workers(root, report["git_sha"]))
        else:
            config = load_config(PROJECT_ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml")
            if not torch.cuda.is_available():
                raise RuntimeError("Real source continuation requires AutoDL CUDA")
            if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1":
                raise ValueError("Use pinned torch2.7.0+cu126 / transformers4.56.1")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise ValueError("Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8")
            report["environment"] = {"python": sys.version, "executable": sys.executable,
                                     "platform": platform.platform(), "torch": str(torch.__version__),
                                     "cuda": torch.version.cuda, "transformers": version("transformers"),
                                     "gpu": torch.cuda.get_device_name(0)}
            seed_everything(0)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            weights = args.pretrained_root / config["model"]["weights_dir"]
            if sha256(weights / "model.safetensors") != config["model"]["weights_sha256"]:
                raise ValueError("Pinned backbone weight SHA256 mismatch")
            train, _ = build_training_datasets(config, args.data_root)
            torch.cuda.reset_peak_memory_stats()
            backbone = DINOv3Backbone.from_pretrained(
                config["model"]["backbone"], weights=weights, freeze=True,
                intermediate_indices=config["model"]["intermediate_indices"],
                dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
            if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
                raise ValueError("Backbone state differs from accepted source smoke")
            model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=0.1,
                                   queries_per_class=2, interaction="static", cross_attention_layers=0,
                                   num_heads=8, temperature=0.07, alpha_init=0.0).to("cuda:0")
            photo = config["style"]["photometric"]
            style = StyleInterventionBank(**{
                key: tuple(value) if isinstance(value, list) else value for key, value in photo.items()
            }).to("cuda:0")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=1e-4, weight_decay=0.01)
            metadata = {"git_sha": report["git_sha"], "seed": 0, "config_sha256": tree_digest(config),
                        "backbone_weights_sha256": config["model"]["weights_sha256"],
                        "manifest_sha256": report["manifest_sha256"], "target_data_accessed": False,
                        "environment": report["environment"], "schedule_horizon": 40000,
                        "sample_workers": 0, "amp": "bfloat16", "deterministic_algorithms": True,
                        "cublas_workspace_config": ":4096:8"}
            start, stop = {"continuous": (0, 8), "prefix": (0, 4), "resume": (4, 8)}[args.mode]
            resume = root / "prefix/state.pth" if start else None
            if resume is not None:
                prefix_report = json.loads((root / "prefix/worker_report.json").read_bytes())
                if prefix_report["ok"] is not True or sha256(resume) != prefix_report["files_sha256"]["state.pth"]:
                    raise ValueError("Prefix checkpoint hash/audit mismatch")
            records = run_segment(model, optimizer, train, style, config, output,
                                  metadata=metadata, start=start, stop=stop, resume=resume)
            (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            report.update(initial_iteration=start, final_iteration=stop,
                          optimizer_updates_this_process=stop - start, metadata=metadata,
                          files_sha256={name: sha256(output / name) for name in ("train.jsonl", "state.pth")},
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        report["ok"] = True
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.mode == "compare":
        files = [path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        (root / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in sorted(files)), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
