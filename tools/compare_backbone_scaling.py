"""Compare frozen ViT-B and ViT-L Static R=2 runs on matched style views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.datasets import cityscapes_dataset
from tools.analyze_query_count import evaluate, load_model, make_style_bank
from tools.train import git_sha, load_config, sha256


def verify_protocol(small: dict[str, Any], large: dict[str, Any]) -> None:
    """Reject changes other than the backbone and declared target scope."""
    if small["experiment"]["phase"] != 15 or large["experiment"]["phase"] != 13:
        raise ValueError("Expected Phase-15 ViT-B and Phase-13 ViT-L configs")
    if small["model"]["backbone"] != "dinov3_vitb16" or large["model"]["backbone"] != "dinov3_vitl16":
        raise ValueError("Expected DINOv3-B and DINOv3-L backbones")
    if small["data"]["targets"] != ["cityscapes"]:
        raise ValueError("ViT-B target scope must be Cityscapes only")
    for section in ("query", "style", "train", "optimizer", "validation"):
        if small[section] != large[section]:
            raise ValueError(f"Protocol drift in {section}")
    for section, exceptions in (
        ("data", {"targets"}),
        ("model", {"backbone", "weights_dir", "weights_sha256", "intermediate_indices"}),
    ):
        small_shared = {key: value for key, value in small[section].items() if key not in exceptions}
        large_shared = {key: value for key, value in large[section].items() if key not in exceptions}
        if small_shared != large_shared:
            raise ValueError(f"Protocol drift in {section}")
    query = small["query"]
    if query["interaction"] != "static" or query["queries_per_class"] != 2:
        raise ValueError("Only Static R=2 is admissible")
    if small["style"]["views"] != ["original", "photometric"]:
        raise ValueError("Only matched original/photometric views are admissible")
    forbidden = {"prediction_consistency", "causal_query_effect", "query_diversity", "learned_null"}
    if forbidden & (small.keys() | large.keys()):
        raise ValueError("Auxiliary mechanisms are outside this comparison")


def verify_run(
    config: dict[str, Any],
    payload: dict[str, Any],
    summary: dict[str, Any],
    *,
    expected_sha: str,
) -> float:
    metadata = payload["metadata"]
    if metadata["git_sha"] != expected_sha:
        raise ValueError("Checkpoint training SHA does not match the accepted run")
    if metadata["backbone"] != config["model"]["backbone"]:
        raise ValueError("Checkpoint backbone mismatch")
    if metadata["pretrained_checkpoint_sha256"] != config["model"]["weights_sha256"]:
        raise ValueError("Checkpoint pretrained-weight SHA mismatch")
    if metadata["seed"] != 0 or metadata["max_iterations"] != 40000:
        raise ValueError("Expected seed-0 40k checkpoint")
    if metadata["source"] != "gta5" or metadata["validation"] != "cityscapes_val":
        raise ValueError("Checkpoint dataset protocol mismatch")
    if metadata["query"] != config["query"] or metadata["style"] != config["style"]:
        raise ValueError("Checkpoint query/style protocol mismatch")
    if int(payload["iteration"]) != 40000:
        raise ValueError("Expected final iteration-40000 checkpoint")
    if summary.get("ok") is not True or summary.get("finite_losses") is not True:
        raise ValueError("Training summary did not pass")
    if summary["git_sha"] != expected_sha:
        raise ValueError("Summary training SHA mismatch")
    validations = summary["validation_results"]
    if len(validations) != 80 or validations[-1]["iteration"] != 40000:
        raise ValueError("Expected 80 validations ending at iteration 40000")
    if any(item["sample_count"] != 500 for item in validations):
        raise ValueError("Validation did not cover all 500 images")
    return float(validations[-1]["miou"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitb-checkpoint", type=Path, required=True)
    parser.add_argument("--vitb-summary", type=Path, required=True)
    parser.add_argument("--vitl-checkpoint", type=Path, required=True)
    parser.add_argument("--vitl-summary", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Backbone-scaling comparison requires CUDA")
    if args.max_samples != 500:
        raise ValueError("This controlled comparison requires all 500 validation images")

    configs = {
        "vitb16": load_config(PROJECT_ROOT / "configs/scaling/gta_dinov3b_static.yaml"),
        "vitl16": load_config(PROJECT_ROOT / "configs/query_interaction/gta_dinov3l_static.yaml"),
    }
    verify_protocol(configs["vitb16"], configs["vitl16"])
    pretrained_root = Path(os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained"))
    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    dataset = cityscapes_dataset(data_root / "cityscapes", split="val")
    if len(dataset) != 500:
        raise ValueError(f"Expected 500 Cityscapes validation images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    specs = (
        ("vitb16", args.vitb_checkpoint, args.vitb_summary, "10993b605b53bcf1d7f67f93602ed62ced9b9a97"),
        ("vitl16", args.vitl_checkpoint, args.vitl_summary, "367694ee8bc2e670be0a896f40e089760fe2177a"),
    )
    results: dict[str, dict[str, Any]] = {}
    for name, checkpoint, summary_path, training_sha in specs:
        config = configs[name]
        weights = pretrained_root / config["model"]["weights_dir"] / "model.safetensors"
        if sha256(weights) != config["model"]["weights_sha256"]:
            raise RuntimeError(f"{name} pretrained-weight SHA mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model, payload = load_model(
            config,
            query_count=2,
            checkpoint_path=checkpoint,
            pretrained_root=pretrained_root,
        )
        final_miou = verify_run(config, payload, summary, expected_sha=training_sha)
        style_bank = make_style_bank(config)
        result = evaluate(model, loader, style_bank, max_samples=500, seed=args.seed)
        result.update(
            {
                "final_cityscapes_miou": final_miou,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_git_sha": training_sha,
                "pretrained_checkpoint_sha256": config["model"]["weights_sha256"],
                "iteration": 40000,
            }
        )
        results[name] = result
        del model, style_bank
        torch.cuda.empty_cache()

    small, large = results["vitb16"], results["vitl16"]
    if small["sample_count"] != large["sample_count"] or small["effect_class_map_count"] != large["effect_class_map_count"]:
        raise RuntimeError("Backbones did not evaluate the same class-map cohort")
    b_variance = float(small["cross_style_effect_variance"])
    l_variance = float(large["cross_style_effect_variance"])
    if l_variance <= 0:
        raise RuntimeError("ViT-L reference effect variance is not positive")
    report = {
        "ok": True,
        "scope": "GTA5_to_Cityscapes_val_only",
        "purpose": "matched_backbone_scaling_diagnostic_not_causal_proof",
        "evaluation_git_sha": git_sha(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "seed": args.seed,
        "sample_count": 500,
        "views": ["original", "photometric"],
        "metric": "population variance across normalized valid-pixel class-logit query-effect maps, mean over GT-present classes",
        "models": results,
        "vitb_minus_vitl_miou": float(small["final_cityscapes_miou"]) - float(large["final_cityscapes_miou"]),
        "vitb_minus_vitl_effect_variance": b_variance - l_variance,
        "vitb_to_vitl_effect_variance_ratio": b_variance / l_variance,
        "vitl_has_lower_effect_variance": l_variance < b_variance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
