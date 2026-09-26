"""Compare learned-null and zero-ablation effect localization."""

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

from causalq.analysis import paired_effect_localization_values
from causalq.datasets import cityscapes_dataset
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import load_training_checkpoint
from tools.train import git_sha, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
        interaction=query["interaction"],
        learned_null=bool(config["learned_null"]["enabled"]),
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
) -> dict[str, Any]:
    totals: dict[str, float] = {}
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
                zero_effects = model.get_query_effect(output=output)
                null_effects = model.get_null_query_effect(output=output)
            values = paired_effect_localization_values(
                zero_effects,
                null_effects,
                labels,
            )
            if not all(torch.isfinite(value.float()).all() for value in values.values()):
                raise FloatingPointError(
                    f"Non-finite localization value at sample {sample_count}"
                )
            count = int(values["zero_absolute_ratio"].numel())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.sum().cpu())
            class_map_count += count
            sample_count += images.shape[0]

    if not class_map_count:
        raise RuntimeError("No comparable present class maps were evaluated")

    means = {key: total / class_map_count for key, total in totals.items()}
    zero_ratio = means["zero_absolute_ratio"]
    null_ratio = means["learned_null_absolute_ratio"]
    zero_contrast = means["zero_normalized_absolute_contrast"]
    null_contrast = means["learned_null_normalized_absolute_contrast"]
    return {
        "zero_ablation": {
            "mean_inside_absolute_mean": means["zero_inside_absolute_mean"],
            "mean_outside_absolute_mean": means["zero_outside_absolute_mean"],
            "mean_absolute_ratio": zero_ratio,
            "mean_normalized_absolute_contrast": zero_contrast,
            "mean_inside_signed_mean": means["zero_inside_signed_mean"],
            "mean_outside_signed_mean": means["zero_outside_signed_mean"],
        },
        "learned_null": {
            "mean_inside_absolute_mean": means[
                "learned_null_inside_absolute_mean"
            ],
            "mean_outside_absolute_mean": means[
                "learned_null_outside_absolute_mean"
            ],
            "mean_absolute_ratio": null_ratio,
            "mean_normalized_absolute_contrast": null_contrast,
            "mean_inside_signed_mean": means["learned_null_inside_signed_mean"],
            "mean_outside_signed_mean": means[
                "learned_null_outside_signed_mean"
            ],
        },
        "paired": {
            "learned_null_minus_zero_absolute_ratio": null_ratio - zero_ratio,
            "learned_null_minus_zero_normalized_absolute_contrast": (
                null_contrast - zero_contrast
            ),
            "fraction_learned_null_higher_absolute_ratio": means[
                "learned_null_has_higher_absolute_ratio"
            ],
            "fraction_learned_null_lower_outside_absolute_mean": means[
                "learned_null_has_lower_outside_absolute_mean"
            ],
        },
        "sample_count": sample_count,
        "present_class_map_count": class_map_count,
        "alpha": float(model.query_head.alpha.detach().float().cpu()),
        "learned_null_passes_gt_region_magnitude_target": (
            means["learned_null_inside_absolute_mean"]
            > means["learned_null_outside_absolute_mean"]
        ),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Learned-null effect analysis requires CUDA")
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")

    config = load_config(args.config)
    if config["query"].get("interaction") != "static":
        raise ValueError("Learned-null effect analysis requires Static Query")
    if not config.get("learned_null", {}).get("enabled"):
        raise ValueError("learned_null.enabled must be true")

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
    model, payload = load_model(config, args.checkpoint, pretrained_root)
    result = evaluate(
        model,
        loader,
        max_samples=min(args.max_samples, len(dataset)),
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "training_git_sha": payload["metadata"]["git_sha"],
            "training_seed": int(payload["metadata"]["seed"]),
            "iteration": int(payload["iteration"]),
        }
    )
    report = {
        "ok": True,
        "evaluation_git_sha": git_sha(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "metric": {
            "name": "paired_zero_and_learned_null_effect_localization",
            "effect_space": "class_logits",
            "inside": "matching_ground_truth_class_pixels",
            "outside": "other_valid_pixels",
            "aggregation": "mean_over_ground_truth_present_class_maps",
        },
        "dataset": "cityscapes_val",
        "requested_max_samples": args.max_samples,
        "pretrained_checkpoint_sha256": actual_weights_sha,
        "model": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
