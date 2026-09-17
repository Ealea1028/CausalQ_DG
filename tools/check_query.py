"""Validate the Phase-6 query branch with real DINOv3 weights on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.utils.seed import seed_everything
from tools.train import git_sha, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/query/gta_dinov3l_query.yaml",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if int(config["experiment"]["phase"]) != 6:
        raise ValueError("Query checker requires a Phase 6 config")
    if not torch.cuda.is_available():
        raise RuntimeError("Query checker requires CUDA")

    seed_everything(int(config["train"]["seed"]))
    pretrained_root = Path(
        os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained")
    )
    weights = pretrained_root / config["model"]["weights_dir"]
    checkpoint = weights / "model.safetensors"
    actual_sha = sha256(checkpoint)
    expected_sha = config["model"]["weights_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"DINOv3 checkpoint SHA-256 mismatch: {actual_sha} != {expected_sha}"
        )

    backbone = DINOv3Backbone.from_pretrained(
        config["model"]["backbone"],
        weights=weights,
        freeze=True,
        intermediate_indices=config["model"]["intermediate_indices"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    query_config = config["query"]
    model = QuerySegmentor(
        backbone,
        decoder_channels=int(config["model"]["decoder_channels"]),
        num_classes=int(config["data"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
        queries_per_class=int(query_config["queries_per_class"]),
        num_heads=int(query_config["num_heads"]),
        cross_attention_layers=int(query_config["cross_attention_layers"]),
        temperature=float(query_config["temperature"]),
        alpha_init=float(query_config["alpha_init"]),
    ).to("cuda:0")
    model.eval()

    height, width = (int(value) for value in args.image_size)
    images = torch.randn(
        args.batch_size,
        3,
        height,
        width,
        device="cuda:0",
    )
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_components(images)

    max_baseline_difference = float(
        (output.logits - output.base_logits).abs().max().cpu()
    )
    expected_query_shape = (
        args.batch_size,
        int(config["data"]["num_classes"]),
        int(query_config["queries_per_class"]),
        backbone.hidden_size,
    )
    ok = (
        tuple(output.logits.shape)
        == (args.batch_size, int(config["data"]["num_classes"]), height, width)
        and tuple(output.query_states.shape) == expected_query_shape
        and torch.isfinite(output.logits).all().item()
        and torch.isfinite(output.delta_logits).all().item()
        and max_baseline_difference < 1e-6
        and float(model.query_head.alpha.detach().cpu()) == 0.0
    )
    report = {
        "ok": bool(ok),
        "git_sha": git_sha(),
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_sha256": actual_sha,
        "input_shape": list(images.shape),
        "logit_shape": list(output.logits.shape),
        "delta_logit_shape": list(output.delta_logits.shape),
        "query_state_shape": list(output.query_states.shape),
        "queries_per_class": model.queries_per_class,
        "cross_attention_layers": len(model.query_attention.layers),
        "temperature": model.query_head.temperature,
        "alpha": float(model.query_head.alpha.detach().cpu()),
        "max_abs_full_vs_baseline": max_baseline_difference,
        "finite_logits": bool(torch.isfinite(output.logits).all().item()),
        "finite_delta_logits": bool(
            torch.isfinite(output.delta_logits).all().item()
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(0) / 1024**3, 3
        ),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(0) / 1024**3, 3
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
