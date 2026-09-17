"""Train the Phase-5/6 frozen-DINOv3 source-only models."""

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
from causalq.losses import segmentation_cross_entropy
from causalq.metrics import MeanIoU
from causalq.models import BaselineSegmentor, DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import save_training_checkpoint
from causalq.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--validation-max-samples", type=int)
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    phase = int(config["experiment"]["phase"])
    if phase not in (5, 6):
        raise ValueError("tools/train.py accepts only Phase 5 or Phase 6 configs")
    if config["train"].get("style", False):
        raise ValueError("Phase 5/6 cannot enable style mechanisms")
    query_enabled = bool(config["model"].get("query", False))
    if phase == 5 and query_enabled:
        raise ValueError("Phase 5 baseline cannot enable queries")
    if phase == 6 and not query_enabled:
        raise ValueError("Phase 6 requires the query branch")
    if phase == 6 and "query" not in config:
        raise ValueError("Phase 6 requires query configuration")
    if phase == 6 and config["query"].get("aggregation") != "logsumexp":
        raise ValueError("Phase 6 currently requires logsumexp query aggregation")
    return config


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
        raise RuntimeError("Phase 5/6 training requires CUDA")

    seed = int(config["train"]["seed"])
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
        raise NotImplementedError("The Phase 5/6 trainer supports GTA5 source only")
    train_dataset = gta5_dataset(data_root / "gta5", transform=transform)
    if config["data"]["validation"] != "cityscapes_val":
        raise NotImplementedError("Phase 5/6 validates on Cityscapes val")
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
            queries_per_class=int(query_config["queries_per_class"]),
            num_heads=int(query_config["num_heads"]),
            cross_attention_layers=int(query_config["cross_attention_layers"]),
            temperature=float(query_config["temperature"]),
            alpha_init=float(query_config["alpha_init"]),
        )
    model = model.to("cuda:0")
    model.train()

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
            "cross_attention_layers": len(model.query_attention.layers),
            "num_heads": int(config["query"]["num_heads"]),
            "temperature": model.query_head.temperature,
            "aggregation": config["query"]["aggregation"],
            "alpha_init": float(model.query_head.alpha.detach().cpu()),
        }
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
        for _ in range(accumulation):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            images = batch["image"].to("cuda:0", non_blocking=True)
            labels = batch["label"].to("cuda:0", non_blocking=True)
            valid_pixel_count = int((labels != 255).sum().item())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite model logits at iteration {iteration}"
                    )
                loss = segmentation_cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite segmentation loss at iteration {iteration}: {loss.item()}"
                )
            (loss / accumulation).backward()
            micro_loss += loss.detach().item() / accumulation

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
