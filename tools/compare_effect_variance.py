"""Compare cross-style causal-query-effect variance for two checkpoints."""

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

from causalq.analysis import cross_style_effect_variance_values
from causalq.datasets import cityscapes_dataset
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from tools.train import git_sha, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-name", default="A3_PRED_CONS")
    parser.add_argument("--candidate-name", default="A4_CQE")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_style_bank(config: dict[str, Any]) -> StyleInterventionBank:
    style = config["style"]
    photo = style["photometric"]
    return StyleInterventionBank(
        brightness=tuple(photo["brightness"]),
        contrast=tuple(photo["contrast"]),
        saturation=tuple(photo["saturation"]),
        gamma=tuple(photo["gamma"]),
        temperature=tuple(photo["temperature"]),
        grayscale_probability=float(photo["grayscale_probability"]),
        fourier_mix_strength=tuple(style["fourier"]["mix_strength"]),
    ).to("cuda:0")


def load_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    pretrained_root: Path,
) -> tuple[QuerySegmentor, dict[str, Any]]:
    weights = pretrained_root / config["model"]["weights_dir"]
    backbone = DINOv3Backbone.from_pretrained(
        config["model"]["backbone"],
        weights=weights,
        freeze=True,
        intermediate_indices=config["model"]["intermediate_indices"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    query = config["query"]
    model = QuerySegmentor(
        backbone,
        decoder_channels=int(config["model"]["decoder_channels"]),
        num_classes=int(config["data"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
        queries_per_class=int(query["queries_per_class"]),
        num_heads=int(query["num_heads"]),
        cross_attention_layers=int(query["cross_attention_layers"]),
        temperature=float(query["temperature"]),
        alpha_init=float(query["alpha_init"]),
    ).to("cuda:0")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(payload["trainable_model"], strict=False)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(set(missing) & trainable_names)
    if missing_trainable or unexpected:
        raise RuntimeError(
            "Checkpoint state mismatch: "
            f"missing trainable={missing_trainable}, unexpected={unexpected}"
        )
    if int(payload["iteration"]) != 40000:
        raise ValueError(f"Expected iteration 40000, got {payload['iteration']}")
    model.eval()
    return model, payload


def evaluate(
    model: QuerySegmentor,
    loader: DataLoader,
    style_bank: StyleInterventionBank,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, float | int]:
    total = 0.0
    class_map_count = 0
    sample_count = 0
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if sample_count >= max_samples:
                break
            torch.manual_seed(seed + index)
            torch.cuda.manual_seed_all(seed + index)
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"].to("cuda:0", non_blocking=True)
            views = style_bank(images)
            effects = []
            for _, view in views.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_components(view)
                    effects.append(model.get_query_effect(output=output))
            values = cross_style_effect_variance_values(
                torch.stack(effects, dim=1), labels
            )
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    f"Non-finite effect variance at sample index {index}"
                )
            total += float(values.sum().cpu())
            class_map_count += int(values.numel())
            sample_count += images.shape[0]

    if not class_map_count:
        raise RuntimeError("No present class maps were evaluated")
    return {
        "effect_variance": total / class_map_count,
        "sample_count": sample_count,
        "present_class_map_count": class_map_count,
        "alpha": float(model.query_head.alpha.detach().float().cpu()),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Effect-variance evaluation requires CUDA")
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")

    config = load_config(args.config)
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    weights_file = pretrained_root / config["model"]["weights_dir"] / "model.safetensors"
    actual_weights_sha = sha256(weights_file)
    if actual_weights_sha != config["model"]["weights_sha256"]:
        raise RuntimeError("DINOv3 checkpoint SHA-256 mismatch")

    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = cityscapes_dataset(data_root / "cityscapes", split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )
    style_bank = make_style_bank(config)

    model_results: dict[str, dict[str, Any]] = {}
    checkpoints = (
        (args.reference_name, args.reference_checkpoint),
        (args.candidate_name, args.candidate_checkpoint),
    )
    for name, checkpoint in checkpoints:
        model, payload = load_model(config, checkpoint, pretrained_root)
        result = evaluate(
            model,
            loader,
            style_bank,
            max_samples=min(args.max_samples, len(dataset)),
            seed=args.seed,
        )
        result.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_git_sha": payload["metadata"]["git_sha"],
                "iteration": int(payload["iteration"]),
            }
        )
        model_results[name] = result
        del model
        torch.cuda.empty_cache()

    reference = float(model_results[args.reference_name]["effect_variance"])
    candidate = float(model_results[args.candidate_name]["effect_variance"])
    report = {
        "ok": True,
        "evaluation_git_sha": git_sha(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "metric": {
            "name": "cross_style_normalized_query_effect_variance",
            "views": ["original", "photometric", "fourier"],
            "normalization": "l2_per_class_map_over_valid_pixels",
            "variance": "population_across_views_then_spatial_sum",
            "aggregation": "mean_over_gt_present_class_maps",
        },
        "dataset": "cityscapes_val",
        "seed": args.seed,
        "requested_max_samples": args.max_samples,
        "pretrained_checkpoint_sha256": actual_weights_sha,
        "models": model_results,
        "candidate_minus_reference": candidate - reference,
        "candidate_to_reference_ratio": candidate / reference if reference else None,
        "passes_lower_variance_target": candidate < reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
