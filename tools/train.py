"""Train the Phase-5--13 frozen-DINOv3 source-only models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from time import time
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.datasets import TrainTransform, cityscapes_dataset, gta5_dataset
from causalq.interventions import StyleInterventionBank
from causalq.losses import (
    causal_query_effect_loss,
    prediction_consistency_kl,
    query_diversity_loss,
    segmentation_cross_entropy,
)
from causalq.metrics import MeanIoU
from causalq.models import BaselineSegmentor, DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import save_training_checkpoint
from causalq.utils.seed import seed_everything


SUPPORTED_PHASES = frozenset(range(5, 14))
QUERY_PHASES = frozenset(range(6, 14))
STYLE_PHASES = frozenset(range(7, 14))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--validation-max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--queries-per-class", type=int)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_seed(config: dict[str, Any], override: int | None) -> int:
    seed = int(config["train"]["seed"] if override is None else override)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def resolve_query_count(config: dict[str, Any], override: int | None) -> int:
    configured = int(config["query"]["queries_per_class"])
    query_count = configured if override is None else int(override)
    phase = int(config["experiment"]["phase"])
    if phase == 12:
        if query_count not in (1, 2, 4):
            raise ValueError("Phase 12 query-count runs require R in {1, 2, 4}")
    elif override is not None and query_count != configured:
        raise ValueError("Query-count overrides are reserved for Phase 12")
    return query_count


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    phase = int(config["experiment"]["phase"])
    if phase not in SUPPORTED_PHASES:
        raise ValueError("tools/train.py accepts only Phase 5--13 configs")
    style_enabled = bool(config["train"].get("style", False))
    if phase in (5, 6) and style_enabled:
        raise ValueError("Phase 5/6 cannot enable style mechanisms")
    query_enabled = bool(config["model"].get("query", False))
    if phase == 5 and query_enabled:
        raise ValueError("Phase 5 baseline cannot enable queries")
    if phase in QUERY_PHASES and not query_enabled:
        raise ValueError("Phase 6--13 requires the query branch")
    if phase in QUERY_PHASES and "query" not in config:
        raise ValueError("Phase 6--13 requires query configuration")
    if phase in QUERY_PHASES and config["query"].get("aggregation") != "logsumexp":
        raise ValueError("Phase 6--13 currently requires logsumexp query aggregation")
    if phase in STYLE_PHASES:
        style = config.get("style", {})
        if not style_enabled:
            raise ValueError("Phase 7--13 requires style training")
        views = style.get("views")
        if phase in (11, 12, 13):
            allowed = (
                ["original", "photometric"],
                ["original", "fourier"],
            )
            if views not in allowed:
                raise ValueError("Phase 11--13 requires exactly one counterfactual view")
        elif views != ["original", "photometric", "fourier"]:
            raise ValueError("Phase 7--10 requires original, photometric, and fourier views")
        if not style.get("preserve_geometry", False):
            raise ValueError("Style phases require geometry-preserving style views")
        if not style.get("sequential_forward", False):
            raise ValueError("Style phases require sequential style forwards")
        if float(style.get("lambda_cf", 0.0)) < 0:
            raise ValueError("style.lambda_cf must be non-negative")
    if phase == 12 and int(config["query"]["queries_per_class"]) not in (1, 2, 4):
        raise ValueError("Phase 12 query-count configs require R in {1, 2, 4}")
    interaction = config.get("query", {}).get("interaction", "one_way")
    if interaction not in {"one_way", "static"}:
        raise ValueError("query.interaction must be one_way or static")
    if phase < 13 and interaction != "one_way":
        raise ValueError("Static query interaction is reserved for Phase 13")
    if phase == 13:
        if interaction != "static":
            raise ValueError("Phase 13 static-query control requires static interaction")
        if int(config["query"]["queries_per_class"]) != 2:
            raise ValueError("Phase 13 fixes queries_per_class=2")
    prediction = config.get("prediction_consistency", {})
    prediction_enabled = bool(prediction.get("enabled", False))
    if phase < 8 and prediction_enabled:
        raise ValueError("Prediction consistency is disabled before Phase 8")
    if prediction_enabled:
        if float(prediction.get("lambda_pred", -1.0)) < 0:
            raise ValueError("prediction_consistency.lambda_pred must be non-negative")
        if float(prediction.get("temperature", 0.0)) <= 0:
            raise ValueError("prediction_consistency.temperature must be positive")
        if prediction.get("divergence") != "kl_reference_to_view":
            raise ValueError("Phase 8 requires KL(reference || view)")
        if not prediction.get("reference_stop_gradient", False):
            raise ValueError("Phase 8 requires a stop-gradient reference")
        if not prediction.get("valid_pixels_only", False):
            raise ValueError("Phase 8 requires valid-pixel masking")
    if phase == 8:
        if not prediction_enabled:
            raise ValueError("Phase 8 requires prediction consistency")
        if "causal_query_effect" in config:
            raise ValueError("Phase 8 cannot enable causal-query-effect losses")
    if phase == 9 and prediction_enabled:
        raise ValueError("Phase 9 CQE control cannot enable prediction consistency")
    cqe = config.get("causal_query_effect", {})
    cqe_enabled = bool(cqe.get("enabled", False))
    if phase < 9 and cqe_enabled:
        raise ValueError("CQE is disabled before Phase 9")
    if cqe_enabled:
        required = {
            "effect_space": "logits",
            "reference_stop_gradient": True,
            "normalization": "l2_per_class_map",
            "present_classes_only": True,
            "valid_pixels_only": True,
            "loss": "smooth_l1",
        }
        for key, expected in required.items():
            if cqe.get(key) != expected:
                raise ValueError(f"Phase 9 requires causal_query_effect.{key}={expected}")
        if float(cqe.get("lambda_cqe", -1.0)) < 0:
            raise ValueError("causal_query_effect.lambda_cqe must be non-negative")
        if float(cqe.get("smooth_l1_beta", 0.0)) <= 0:
            raise ValueError("causal_query_effect.smooth_l1_beta must be positive")
    if phase == 9 and not cqe_enabled:
        raise ValueError("Phase 9 requires causal-query-effect distillation")

    diversity = config.get("query_diversity", {})
    diversity_enabled = bool(diversity.get("enabled", False))
    if phase < 10 and diversity_enabled:
        raise ValueError("Query diversity is disabled before Phase 10")
    if phase == 10:
        if not prediction_enabled or not cqe_enabled or not diversity_enabled:
            raise ValueError(
                "Phase 10 requires prediction consistency, CQE, and query diversity"
            )
        required_diversity = {
            "target": "query_residuals",
            "loss": "off_diagonal_cosine_squared",
        }
        for key, expected in required_diversity.items():
            if diversity.get(key) != expected:
                raise ValueError(f"Phase 10 requires query_diversity.{key}={expected}")
        if float(diversity.get("lambda_div", -1.0)) != 0.01:
            raise ValueError("Phase 10 fixes query_diversity.lambda_div=0.01")
    if phase in (11, 12, 13) and (
        prediction_enabled or cqe_enabled or diversity_enabled
    ):
        raise ValueError("Phase 11--13 ablations disable all consistency losses")
    return config


def style_view_weights(style: dict[str, Any]) -> dict[str, float]:
    """Keep total counterfactual supervision weight fixed across view ablations."""
    views = list(style["views"])
    if len(views) < 2 or views[0] != "original" or len(set(views)) != len(views):
        raise ValueError("Style views require original followed by unique interventions")
    counterfactual_weight = float(style["lambda_cf"]) / (len(views) - 1)
    return {"original": 1.0, **{name: counterfactual_weight for name in views[1:]}}


def make_loader(dataset, config: dict[str, Any], *, training: bool) -> DataLoader:
    workers = int(config["data"].get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]) if training else 1,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        persistent_workers=workers > 0,
    )


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    amp: bool,
    max_samples: int | None,
) -> dict[str, object]:
    metric = MeanIoU()
    model.eval()
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images)
            metric.update(logits, labels)
            sample_count += images.shape[0]
            if max_samples is not None and sample_count >= max_samples:
                break
    result = metric.compute()
    result["sample_count"] = sample_count
    model.train()
    return result


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5--13 training requires CUDA")

    seed = resolve_seed(config, args.seed)
    query_count = (
        resolve_query_count(config, args.queries_per_class)
        if int(config["experiment"]["phase"]) != 5
        else None
    )
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    output_root = Path(
        os.getenv("CAUSALQ_OUTPUT_ROOT", "/root/autodl-tmp/outputs/CausalQ_DG")
    )
    run_id = args.run_id or config["experiment"]["id"]
    run_dir = output_root / run_id

    transform = TrainTransform(
        config["train"]["crop_size"],
        scale_range=config["data"]["random_scale"],
        horizontal_flip_probability=config["data"]["horizontal_flip_probability"],
        min_valid_fraction=config["data"]["min_valid_fraction"],
        crop_attempts=config["data"]["crop_attempts"],
    )
    if config["data"]["source"] != "gta5":
        raise NotImplementedError("The Phase 5--13 trainer supports GTA5 source only")
    train_dataset = gta5_dataset(data_root / "gta5", transform=transform)
    if config["data"]["validation"] != "cityscapes_val":
        raise NotImplementedError("Phase 5--13 validates on Cityscapes val")
    val_dataset = cityscapes_dataset(data_root / "cityscapes", split="val")
    train_loader = make_loader(train_dataset, config, training=True)
    val_loader = make_loader(val_dataset, config, training=False)

    weights = pretrained_root / config["model"]["weights_dir"]
    checkpoint_file = weights / "model.safetensors"
    actual_backbone_sha = sha256(checkpoint_file)
    expected_backbone_sha = config["model"]["weights_sha256"]
    if actual_backbone_sha != expected_backbone_sha:
        raise RuntimeError(
            "DINOv3 checkpoint SHA-256 mismatch: "
            f"expected {expected_backbone_sha}, got {actual_backbone_sha}"
        )

    backbone = DINOv3Backbone.from_pretrained(
        config["model"]["backbone"],
        weights=weights,
        freeze=True,
        intermediate_indices=config["model"]["intermediate_indices"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    common_model_options = {
        "decoder_channels": int(config["model"]["decoder_channels"]),
        "num_classes": int(config["data"]["num_classes"]),
        "dropout": float(config["model"]["dropout"]),
    }
    if int(config["experiment"]["phase"]) == 5:
        model = BaselineSegmentor(backbone, **common_model_options)
    else:
        query_config = config["query"]
        model = QuerySegmentor(
            backbone,
            **common_model_options,
            queries_per_class=int(query_count),
            num_heads=int(query_config["num_heads"]),
            cross_attention_layers=int(query_config["cross_attention_layers"]),
            temperature=float(query_config["temperature"]),
            alpha_init=float(query_config["alpha_init"]),
            interaction=query_config.get("interaction", "one_way"),
        )
    model = model.to("cuda:0")
    model.train()

    style_bank = None
    phase = int(config["experiment"]["phase"])
    if phase in STYLE_PHASES:
        style_config = config["style"]
        photo_config = style_config["photometric"]
        fourier_config = style_config["fourier"]
        style_bank = StyleInterventionBank(
            brightness=tuple(photo_config["brightness"]),
            contrast=tuple(photo_config["contrast"]),
            saturation=tuple(photo_config["saturation"]),
            gamma=tuple(photo_config["gamma"]),
            temperature=tuple(photo_config["temperature"]),
            grayscale_probability=float(photo_config["grayscale_probability"]),
            fourier_mix_strength=tuple(fourier_config["mix_strength"]),
        ).to("cuda:0")

    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, run_dir / "config.yaml")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    initial_lr = float(config["optimizer"]["lr"])
    max_iterations = args.max_iterations or int(config["train"]["max_iterations"])
    accumulation = int(config["train"].get("gradient_accumulation", 1))
    validation_interval = min(
        int(config["validation"]["interval"]), max_iterations
    )
    validation_max_samples = args.validation_max_samples
    amp = bool(config["train"]["amp"])

    metadata = {
        "experiment_id": run_id,
        "phase": int(config["experiment"]["phase"]),
        "git_sha": git_sha(),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "backbone": config["model"]["backbone"],
        "pretrained_checkpoint_sha256": actual_backbone_sha,
        "source": config["data"]["source"],
        "validation": config["data"]["validation"],
        "seed": seed,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "max_iterations": max_iterations,
    }
    if isinstance(model, QuerySegmentor):
        metadata["query"] = {
            "queries_per_class": model.queries_per_class,
            "interaction": model.interaction,
            "cross_attention_layers": (
                len(model.query_attention.layers)
                if model.query_attention is not None
                else 0
            ),
            "num_heads": int(config["query"]["num_heads"]),
            "temperature": model.query_head.temperature,
            "aggregation": config["query"]["aggregation"],
            "alpha_init": float(model.query_head.alpha.detach().cpu()),
        }
    if style_bank is not None:
        metadata["style"] = {
            "views": config["style"]["views"],
            "preserve_geometry": True,
            "sequential_forward": True,
            "lambda_cf": float(config["style"]["lambda_cf"]),
            "photometric": config["style"]["photometric"],
            "fourier": config["style"]["fourier"],
        }
    prediction_enabled = bool(
        config.get("prediction_consistency", {}).get("enabled", False)
    )
    if prediction_enabled:
        metadata["prediction_consistency"] = config["prediction_consistency"]
    cqe_enabled = bool(config.get("causal_query_effect", {}).get("enabled", False))
    if cqe_enabled:
        metadata["causal_query_effect"] = config["causal_query_effect"]
    diversity_enabled = bool(config.get("query_diversity", {}).get("enabled", False))
    if diversity_enabled:
        metadata["query_diversity"] = config["query_diversity"]
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    losses: list[float] = []
    validation_results: list[dict[str, object]] = []
    started = time()

    for iteration in range(1, max_iterations + 1):
        micro_loss = 0.0
        component_losses: dict[str, float] = {}
        for _ in range(accumulation):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"].to("cuda:0", non_blocking=True)
            valid_pixel_count = int((labels != 255).sum().item())
            if style_bank is None:
                named_views = (("original", images),)
                view_weights = {"original": 1.0}
            else:
                configured_views = config["style"]["views"]
                generated_views = {"original": images}
                if "photometric" in configured_views:
                    generated_views["photometric"] = style_bank.photometric_view(images)
                if "fourier" in configured_views:
                    generated_views["fourier"] = style_bank.fourier_view(images)
                named_views = tuple(
                    (name, generated_views[name]) for name in configured_views
                )
                view_weights = style_view_weights(config["style"])

            reference_logits = None
            reference_effect = None
            for view_name, view_images in named_views:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    if cqe_enabled:
                        model_output = model.forward_components(view_images)
                        logits = model_output.logits
                        query_effect = model.get_query_effect(output=model_output)
                    else:
                        logits = model(view_images)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            f"Non-finite {view_name} logits at iteration {iteration}"
                        )
                    view_loss = segmentation_cross_entropy(logits, labels)
                if not torch.isfinite(view_loss):
                    raise FloatingPointError(
                        f"Non-finite {view_name} segmentation loss at iteration "
                        f"{iteration}: {view_loss.item()}"
                    )
                objective = view_loss * view_weights[view_name]
                if prediction_enabled:
                    if view_name == "original":
                        reference_logits = logits.detach()
                    else:
                        if reference_logits is None:
                            raise RuntimeError("Original view must precede style views")
                        prediction_loss = prediction_consistency_kl(
                            logits,
                            reference_logits,
                            labels=labels,
                            temperature=float(
                                config["prediction_consistency"]["temperature"]
                            ),
                        )
                        if not torch.isfinite(prediction_loss):
                            raise FloatingPointError(
                                f"Non-finite {view_name} prediction consistency "
                                f"at iteration {iteration}: {prediction_loss.item()}"
                            )
                        prediction_weight = (
                            float(config["prediction_consistency"]["lambda_pred"])
                            / 2.0
                        )
                        objective = objective + prediction_weight * prediction_loss
                        key = f"prediction_{view_name}"
                        component_losses[key] = component_losses.get(key, 0.0) + (
                            prediction_loss.detach().item() / accumulation
                        )
                if cqe_enabled:
                    if view_name == "original":
                        reference_effect = query_effect.detach()
                    else:
                        if reference_effect is None:
                            raise RuntimeError("Original effect must precede style effects")
                        cqe_loss = causal_query_effect_loss(
                            query_effect,
                            reference_effect,
                            labels,
                            beta=float(
                                config["causal_query_effect"]["smooth_l1_beta"]
                            ),
                        )
                        if not torch.isfinite(cqe_loss):
                            raise FloatingPointError(
                                f"Non-finite {view_name} CQE loss at iteration "
                                f"{iteration}: {cqe_loss.item()}"
                            )
                        cqe_weight = (
                            float(config["causal_query_effect"]["lambda_cqe"])
                            / 2.0
                        )
                        objective = objective + cqe_weight * cqe_loss
                        key = f"cqe_{view_name}"
                        component_losses[key] = component_losses.get(key, 0.0) + (
                            cqe_loss.detach().item() / accumulation
                        )
                (objective / accumulation).backward()
                micro_loss += objective.detach().item() / accumulation
                component_losses[view_name] = component_losses.get(view_name, 0.0) + (
                    view_loss.detach().item() / accumulation
                )

            if diversity_enabled:
                if not isinstance(model, QuerySegmentor):
                    raise TypeError("Query diversity requires QuerySegmentor")
                diversity_loss = query_diversity_loss(model.get_query_residuals())
                if not torch.isfinite(diversity_loss):
                    raise FloatingPointError(
                        f"Non-finite query diversity loss at iteration {iteration}: "
                        f"{diversity_loss.item()}"
                    )
                diversity_weight = float(config["query_diversity"]["lambda_div"])
                weighted_diversity = diversity_weight * diversity_loss
                (weighted_diversity / accumulation).backward()
                micro_loss += weighted_diversity.detach().item() / accumulation
                component_losses["diversity"] = component_losses.get(
                    "diversity", 0.0
                ) + diversity_loss.detach().item() / accumulation

        gradient_norm = float(clip_grad_norm_(trainable, max_norm=1.0))
        if not torch.isfinite(torch.tensor(gradient_norm)):
            raise FloatingPointError(
                f"Non-finite gradient norm at iteration {iteration}: {gradient_norm}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scale = (1.0 - iteration / max_iterations) ** float(
            config["optimizer"]["poly_power"]
        )
        for group in optimizer.param_groups:
            group["lr"] = initial_lr * lr_scale
        losses.append(micro_loss)

        record = {
            "iteration": iteration,
            "loss": micro_loss,
            "gradient_norm": gradient_norm,
            "valid_pixel_count": valid_pixel_count,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if style_bank is not None:
            record.update(
                {
                    f"loss_{name}": component_losses[name]
                    for name in config["style"]["views"]
                }
            )
        if prediction_enabled:
            prediction_mean = 0.5 * (
                component_losses["prediction_photometric"]
                + component_losses["prediction_fourier"]
            )
            record.update(
                {
                    "loss_prediction": prediction_mean,
                    "loss_prediction_photometric": component_losses[
                        "prediction_photometric"
                    ],
                    "loss_prediction_fourier": component_losses[
                        "prediction_fourier"
                    ],
                }
            )
        if cqe_enabled:
            cqe_mean = 0.5 * (
                component_losses["cqe_photometric"]
                + component_losses["cqe_fourier"]
            )
            record.update(
                {
                    "loss_cqe": cqe_mean,
                    "loss_cqe_photometric": component_losses[
                        "cqe_photometric"
                    ],
                    "loss_cqe_fourier": component_losses["cqe_fourier"],
                }
            )
        if diversity_enabled:
            diversity_raw = component_losses["diversity"]
            diversity_weight = float(config["query_diversity"]["lambda_div"])
            record.update(
                {
                    "loss_diversity": diversity_raw,
                    "loss_diversity_weighted": diversity_weight * diversity_raw,
                }
            )
        if isinstance(model, QuerySegmentor):
            record["alpha"] = float(model.query_head.alpha.detach().cpu())
        with (run_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if iteration == 1 or iteration % int(config["train"]["log_interval"]) == 0:
            print(json.dumps(record), flush=True)

        if iteration % validation_interval == 0 or iteration == max_iterations:
            result = validate(
                model,
                val_loader,
                amp=amp,
                max_samples=validation_max_samples,
            )
            result["iteration"] = iteration
            validation_results.append(result)
            print(json.dumps({"validation": result}), flush=True)
            save_training_checkpoint(
                run_dir / "checkpoints" / f"iter_{iteration:06d}.pth",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                metadata=metadata,
            )

    summary = {
        "ok": True,
        **metadata,
        "elapsed_seconds": round(time() - started, 2),
        "first_20_loss_mean": sum(losses[:20]) / min(20, len(losses)),
        "last_20_loss_mean": sum(losses[-20:]) / min(20, len(losses)),
        "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "last_gradient_norm": gradient_norm,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "validation_results": validation_results,
    }
    if isinstance(model, QuerySegmentor):
        summary["final_alpha"] = float(model.query_head.alpha.detach().cpu())
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
