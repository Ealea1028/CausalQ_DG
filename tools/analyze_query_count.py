"""Compare query behavior and effect variance across query-count checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.analysis import (
    active_query_behavior_values,
    cross_style_effect_variance_values,
    present_class_mask,
    within_class_query_similarity_values,
)
from causalq.datasets import cityscapes_dataset
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import load_training_checkpoint
from tools.train import git_sha, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "QUERY_COUNT", "CHECKPOINT", "MIOU"),
        required=True,
        help="Repeat for each model: name, R, final checkpoint, final mIoU",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924)
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
    *,
    query_count: int,
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
        queries_per_class=query_count,
        num_heads=int(query["num_heads"]),
        cross_attention_layers=int(query["cross_attention_layers"]),
        temperature=float(query["temperature"]),
        alpha_init=float(query["alpha_init"]),
    ).to("cuda:0")

    payload = load_training_checkpoint(checkpoint_path)
    metadata = payload["metadata"]
    recorded_count = int(metadata["query"]["queries_per_class"])
    if recorded_count != query_count:
        raise ValueError(
            f"Checkpoint records R={recorded_count}, requested R={query_count}"
        )
    if int(payload["iteration"]) != 40000:
        raise ValueError(f"Expected iteration 40000, got {payload['iteration']}")

    missing, unexpected = model.load_state_dict(
        payload["trainable_model"], strict=False
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(set(missing) & trainable_names)
    if missing_trainable or unexpected:
        raise RuntimeError(
            "Checkpoint state mismatch: "
            f"missing trainable={missing_trainable}, unexpected={unexpected}"
        )
    model.eval()
    return model, payload


def similarity_summary(values: Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().cpu()
    if not values.numel():
        return {
            "pair_count": 0,
            "mean_cosine": None,
            "mean_absolute_cosine": None,
            "mean_squared_cosine": None,
        }
    return {
        "pair_count": int(values.numel()),
        "mean_cosine": float(values.mean()),
        "mean_absolute_cosine": float(values.abs().mean()),
        "mean_squared_cosine": float(values.square().mean()),
    }


def behavior_summary(
    values: dict[str, list[Tensor]],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, chunks in values.items():
        combined = torch.cat(chunks).float()
        result[name] = float(combined.mean())
        result[f"{name}_std_population"] = float(combined.std(unbiased=False))
        result["class_map_count"] = int(combined.numel())
    return result


def evaluate(
    model: QuerySegmentor,
    loader: DataLoader,
    style_bank: StyleInterventionBank,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    view_names = ("original", "photometric")
    contextual_values: dict[str, list[Tensor]] = {name: [] for name in view_names}
    behavior_values: dict[str, dict[str, list[Tensor]]] = {
        name: {
            "effective_query_count": [],
            "effective_query_fraction": [],
            "dominant_query_share": [],
            "pixel_top1_responsibility": [],
        }
        for name in view_names
    }
    effect_values: list[Tensor] = []
    sample_count = 0

    residual_similarity = within_class_query_similarity_values(
        model.get_query_residuals().detach().unsqueeze(0)
    )
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if sample_count >= max_samples:
                break
            torch.manual_seed(seed + index)
            torch.cuda.manual_seed_all(seed + index)
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"].to("cuda:0", non_blocking=True)
            views = {
                "original": images,
                "photometric": style_bank.photometric_view(images),
            }
            present = present_class_mask(labels, num_classes=model.num_classes)
            sample_effects = []

            for view_name, view_images in views.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_components(view_images)
                if not all(
                    torch.isfinite(value).all()
                    for value in (
                        output.logits,
                        output.query_states,
                        output.query_score_maps,
                        output.scaled_delta_logits,
                    )
                ):
                    raise FloatingPointError(
                        f"Non-finite query analysis tensor at sample {index}"
                    )

                contextual_values[view_name].append(
                    within_class_query_similarity_values(
                        output.query_states,
                        present_classes=present,
                    ).cpu()
                )
                score_size = output.query_score_maps.shape[-2:]
                patch_labels = F.interpolate(
                    labels.unsqueeze(1).float(),
                    size=score_size,
                    mode="nearest",
                ).squeeze(1).long()
                behavior = active_query_behavior_values(
                    output.query_score_maps,
                    patch_labels,
                )
                for metric_name, metric_values in behavior.items():
                    behavior_values[view_name][metric_name].append(
                        metric_values.cpu()
                    )
                sample_effects.append(output.scaled_delta_logits)

            effect_values.append(
                cross_style_effect_variance_values(
                    torch.stack(sample_effects, dim=1),
                    labels,
                ).cpu()
            )
            sample_count += images.shape[0]

    if not sample_count:
        raise RuntimeError("No validation samples were evaluated")
    combined_effects = torch.cat(effect_values).float()
    return {
        "sample_count": sample_count,
        "alpha": float(model.query_head.alpha.detach().float().cpu()),
        "residual_query_similarity": similarity_summary(residual_similarity),
        "contextual_query_similarity": {
            name: similarity_summary(torch.cat(values) if values else torch.empty(0))
            for name, values in contextual_values.items()
        },
        "active_query_behavior": {
            name: behavior_summary(values)
            for name, values in behavior_values.items()
        },
        "cross_style_effect_variance": float(combined_effects.mean()),
        "effect_class_map_count": int(combined_effects.numel()),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def parse_model_specs(
    raw_specs: list[list[str]],
) -> list[tuple[str, int, Path, float]]:
    specs: list[tuple[str, int, Path, float]] = []
    names: set[str] = set()
    counts: set[int] = set()
    for name, raw_count, raw_checkpoint, raw_miou in raw_specs:
        query_count = int(raw_count)
        if query_count < 1:
            raise ValueError("Query counts must be positive")
        if name in names or query_count in counts:
            raise ValueError("Model names and query counts must be unique")
        names.add(name)
        counts.add(query_count)
        specs.append((name, query_count, Path(raw_checkpoint), float(raw_miou)))
    return specs


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Query-count analysis requires CUDA")
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")

    specs = parse_model_specs(args.model)
    config = load_config(args.config)
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    weights_file = (
        pretrained_root / config["model"]["weights_dir"] / "model.safetensors"
    )
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
    models: dict[str, dict[str, Any]] = {}

    for name, query_count, checkpoint, miou in specs:
        model, payload = load_model(
            config,
            query_count=query_count,
            checkpoint_path=checkpoint,
            pretrained_root=pretrained_root,
        )
        result = evaluate(
            model,
            loader,
            style_bank,
            max_samples=min(args.max_samples, len(dataset)),
            seed=args.seed,
        )
        result.update(
            {
                "queries_per_class": query_count,
                "final_cityscapes_miou": miou,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_git_sha": payload["metadata"]["git_sha"],
                "iteration": int(payload["iteration"]),
            }
        )
        models[name] = result
        del model
        torch.cuda.empty_cache()

    ranking = sorted(
        models,
        key=lambda name: float(models[name]["final_cityscapes_miou"]),
        reverse=True,
    )
    report = {
        "ok": True,
        "evaluation_git_sha": git_sha(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "dataset": "cityscapes_val",
        "seed": args.seed,
        "requested_max_samples": args.max_samples,
        "views": ["original", "photometric"],
        "pretrained_checkpoint_sha256": actual_weights_sha,
        "metric_definitions": {
            "query_similarity": (
                "unordered within-class cosine similarities; contextual values "
                "use GT-present classes"
            ),
            "active_query_behavior": (
                "softmax responsibilities over R on downsampled GT-class pixels; "
                "effective count is exp(entropy(mean responsibility))"
            ),
            "effect_variance": (
                "population variance across original/photometric normalized "
                "valid-pixel effect maps, averaged over GT-present classes"
            ),
        },
        "models": models,
        "ranking_by_final_cityscapes_miou": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
