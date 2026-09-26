"""Audit whether factual-minus-zero query effects localize to GT regions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.analysis import zero_effect_localization_values
from causalq.datasets import cityscapes_dataset
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import load_training_checkpoint
from tools.train import git_sha, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for every checkpoint in the audit.",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("Each checkpoint must use NAME=PATH")
        if name in names:
            raise ValueError(f"Duplicate checkpoint name: {name}")
        names.add(name)
        parsed.append((name, Path(path)))
    return parsed


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
        interaction=query.get("interaction", "one_way"),
    ).to("cuda:0")

    payload = load_training_checkpoint(checkpoint_path)
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
    *,
    max_samples: int,
) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    localized_count = 0
    class_map_count = 0
    sample_count = 0
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    with torch.inference_mode():
        for batch in loader:
            if sample_count >= max_samples:
                break
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"].to("cuda:0", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_components(images)
                effects = model.get_query_effect(output=output)
            values = zero_effect_localization_values(effects, labels)
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise FloatingPointError(
                    f"Non-finite localization value at sample {sample_count}"
                )
            count = int(values["absolute_ratio"].numel())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.sum().cpu())
            localized_count += int(
                (values["inside_absolute_mean"] > values["outside_absolute_mean"])
                .sum()
                .cpu()
            )
            class_map_count += count
            sample_count += images.shape[0]

    if not class_map_count:
        raise RuntimeError("No comparable present class maps were evaluated")
    means = {f"mean_{key}": total / class_map_count for key, total in totals.items()}
    fraction = localized_count / class_map_count
    return {
        **means,
        "localized_class_map_fraction": fraction,
        "sample_count": sample_count,
        "present_class_map_count": class_map_count,
        "alpha": float(model.query_head.alpha.detach().float().cpu()),
        "passes_gt_region_magnitude_target": (
            means["mean_inside_absolute_mean"]
            > means["mean_outside_absolute_mean"]
        ),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Zero-effect localization analysis requires CUDA")
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")

    checkpoints = parse_checkpoints(args.checkpoint)
    config = load_config(args.config)
    if config["query"].get("interaction") != "static":
        raise ValueError("The selected zero-effect audit requires Static Query")
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    weights_file = pretrained_root / config["model"]["weights_dir"] / "model.safetensors"
    actual_weights_sha = sha256(weights_file)
    if actual_weights_sha != config["model"]["weights_sha256"]:
        raise RuntimeError("DINOv3 checkpoint SHA-256 mismatch")

    dataset = cityscapes_dataset(data_root / "cityscapes", split="val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )

    model_results: dict[str, dict[str, Any]] = {}
    for name, checkpoint in checkpoints:
        model, payload = load_model(config, checkpoint, pretrained_root)
        result = evaluate(
            model,
            loader,
            max_samples=min(args.max_samples, len(dataset)),
        )
        result.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_git_sha": payload["metadata"]["git_sha"],
                "training_seed": int(payload["metadata"]["seed"]),
                "iteration": int(payload["iteration"]),
            }
        )
        model_results[name] = result
        del model
        torch.cuda.empty_cache()

    ratios = [float(result["mean_absolute_ratio"]) for result in model_results.values()]
    fractions = [
        float(result["localized_class_map_fraction"])
        for result in model_results.values()
    ]
    report = {
        "ok": True,
        "evaluation_git_sha": git_sha(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "metric": {
            "name": "zero_ablation_effect_localization",
            "effect": "absolute_factual_minus_zero_class_logit",
            "inside": "mean_on_matching_ground_truth_class_pixels",
            "outside": "mean_on_other_valid_pixels",
            "aggregation": "mean_over_ground_truth_present_class_maps",
        },
        "dataset": "cityscapes_val",
        "requested_max_samples": args.max_samples,
        "pretrained_checkpoint_sha256": actual_weights_sha,
        "models": model_results,
        "across_seed_mean_absolute_ratio": statistics.fmean(ratios),
        "across_seed_sample_std_absolute_ratio": (
            statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        ),
        "across_seed_mean_localized_class_map_fraction": statistics.fmean(fractions),
        "all_seeds_pass_gt_region_magnitude_target": all(
            bool(result["passes_gt_region_magnitude_target"])
            for result in model_results.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
