"""Create label-selected GTA5-to-Cityscapes query-effect diagnostic panels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.datasets import cityscapes_dataset
from causalq.datasets.cityscapes import colorize_train_ids, read_index_mask
from causalq.interventions import StyleInterventionBank
from causalq.models import BaselineSegmentor, DINOv3Backbone, QuerySegmentor
from causalq.utils.checkpoint import load_training_checkpoint
from causalq.utils.seed import seed_everything
from tools.train import git_sha, load_config, sha256


CLASSES = {"road": 0, "car": 13, "person": 11, "vegetation": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--min-class-pixels", type=int, default=1000)
    return parser.parse_args()


def select_samples(dataset, *, min_class_pixels: int) -> dict[str, int]:
    """Select distinct examples from GT alone, never from model outcomes."""
    if min_class_pixels < 1:
        raise ValueError("min-class-pixels must be positive")
    selected: dict[str, int] = {}
    used: set[int] = set()
    for class_name, class_id in CLASSES.items():
        for index, pair in enumerate(dataset.pairs):
            if index in used:
                continue
            with Image.open(pair.label) as mask_file:
                mask = read_index_mask(mask_file)
            if int(np.count_nonzero(mask == class_id)) >= min_class_pixels:
                selected[class_name] = index
                used.add(index)
                break
        if class_name not in selected:
            raise RuntimeError(f"No Cityscapes val sample contains enough {class_name} pixels")
    return selected


def signed_effect_rgb(effect: np.ndarray, scale: float) -> np.ndarray:
    """Blue = negative, white = zero, red = positive, with a shared scale."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Effect scale must be finite and positive")
    values = np.clip(effect.astype(np.float32) / scale, -1, 1)
    rgb = np.full((*effect.shape, 3), 255, dtype=np.float32)
    positive = values >= 0
    rgb[..., 1][positive] = 255 * (1 - values[positive])
    rgb[..., 2][positive] = 255 * (1 - values[positive])
    rgb[..., 0][~positive] = 255 * (1 + values[~positive])
    rgb[..., 1][~positive] = 255 * (1 + values[~positive])
    return np.rint(rgb).astype(np.uint8)


def image_rgb(image: torch.Tensor, bank: StyleInterventionBank) -> np.ndarray:
    unit = (image.float().unsqueeze(0) * bank.std + bank.mean).clamp(0, 1)
    return unit[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()


def load_model(config: dict, checkpoint: Path, pretrained_root: Path, *, baseline: bool):
    backbone = DINOv3Backbone.from_pretrained(
        config["model"]["backbone"],
        weights=pretrained_root / config["model"]["weights_dir"],
        freeze=True,
        intermediate_indices=config["model"]["intermediate_indices"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    common = {
        "decoder_channels": int(config["model"]["decoder_channels"]),
        "num_classes": int(config["data"]["num_classes"]),
        "dropout": float(config["model"]["dropout"]),
    }
    if baseline:
        model = BaselineSegmentor(backbone, **common)
    else:
        query = config["query"]
        if query.get("interaction") != "static" or int(query["queries_per_class"]) != 2:
            raise ValueError("This figure requires the selected Static R=2 configuration")
        model = QuerySegmentor(
            backbone,
            **common,
            queries_per_class=2,
            num_heads=int(query["num_heads"]),
            cross_attention_layers=int(query["cross_attention_layers"]),
            temperature=float(query["temperature"]),
            alpha_init=float(query["alpha_init"]),
            interaction="static",
        )
    model = model.to("cuda:0").eval()
    payload = load_training_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(payload["trainable_model"], strict=False)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(missing) & trainable_names or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if int(payload["iteration"]) != 40000:
        raise ValueError("Both figures require final 40k checkpoints")
    if payload["metadata"].get("source") != "gta5":
        raise ValueError("Checkpoint must have GTA5 as source")
    return model, payload


def make_montage(panels: list[tuple[str, np.ndarray]], output: Path) -> None:
    width, height = panels[0][1].shape[1], panels[0][1].shape[0]
    display_width = 768
    display_height = round(height * display_width / width)
    margin = 28
    canvas = Image.new("RGB", (display_width * 2, (display_height + margin) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, pixels) in enumerate(panels):
        panel = Image.fromarray(pixels).resize(
            (display_width, display_height), Image.Resampling.BILINEAR
            if title in {"Original", "Photometric", "Query effect original", "Query effect style"}
            else Image.Resampling.NEAREST,
        )
        x = (index % 2) * display_width
        y = (index // 2) * (display_height + margin)
        draw.text((x + 8, y + 7), title, fill="black")
        canvas.paste(panel, (x, y + margin))
    canvas.save(output)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference is required on AutoDL")
    seed_everything(args.seed)
    static_config = load_config(args.static_config)
    baseline_config = load_config(args.baseline_config)
    pretrained_root = Path(os.getenv("CAUSALQ_PRETRAINED_ROOT", "/root/autodl-tmp/pretrained"))
    data_root = Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets"))
    weights = pretrained_root / static_config["model"]["weights_dir"] / "model.safetensors"
    weights_sha = sha256(weights)
    if weights_sha != static_config["model"]["weights_sha256"] or weights_sha != baseline_config["model"]["weights_sha256"]:
        raise RuntimeError("Pretrained-weight SHA-256 mismatch")
    if baseline_config["data"]["source"] != "gta5" or static_config["data"]["source"] != "gta5":
        raise ValueError("Both models must be GTA5-trained")

    dataset = cityscapes_dataset(data_root / "cityscapes", split="val")
    selected = select_samples(dataset, min_class_pixels=args.min_class_pixels)
    baseline, baseline_payload = load_model(baseline_config, args.baseline_checkpoint, pretrained_root, baseline=True)
    static, static_payload = load_model(static_config, args.static_checkpoint, pretrained_root, baseline=False)
    photo = static_config["style"]["photometric"]
    bank = StyleInterventionBank(
        brightness=tuple(photo["brightness"]), contrast=tuple(photo["contrast"]),
        saturation=tuple(photo["saturation"]), gamma=tuple(photo["gamma"]),
        temperature=tuple(photo["temperature"]),
        grayscale_probability=float(photo["grayscale_probability"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for class_name, index in selected.items():
            class_id = CLASSES[class_name]
            sample = dataset[index]
            image = sample["image"].unsqueeze(0)
            torch.manual_seed(args.seed + index)
            style = bank.photometric_view(image)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base_logits = baseline(image.to("cuda:0"))
                original = static.forward_components(image.to("cuda:0"))
                styled = static.forward_components(style.to("cuda:0"))
            effect_original = original.scaled_delta_logits[0, class_id].float().cpu().numpy()
            effect_style = styled.scaled_delta_logits[0, class_id].float().cpu().numpy()
            if not np.isfinite(effect_original).all() or not np.isfinite(effect_style).all():
                raise FloatingPointError(f"Non-finite effect for {sample['id']}")
            scale = max(float(np.max(np.abs(effect_original))), float(np.max(np.abs(effect_style))), 1e-8)
            label = sample["label"].byte().numpy()
            path = args.output_dir / f"{class_name}.png"
            make_montage([
                ("Original", image_rgb(image[0], bank)),
                ("GT", colorize_train_ids(label)),
                ("A0 baseline prediction", colorize_train_ids(base_logits.argmax(1)[0].byte().cpu().numpy())),
                ("Photometric", image_rgb(style[0], bank)),
                ("Query effect original", signed_effect_rgb(effect_original, scale)),
                ("Query effect style", signed_effect_rgb(effect_style, scale)),
                ("Static R=2 prediction", colorize_train_ids(original.logits.argmax(1)[0].byte().cpu().numpy())),
            ], path)
            rows.append({"class": class_name, "class_id": class_id, "sample_id": sample["id"],
                         "dataset_index": index, "gt_class_pixels": int(np.count_nonzero(label == class_id)),
                         "effect_color_scale": scale, "figure": str(path)})
    report = {
        "ok": True, "scope": "GTA5_to_Cityscapes_val", "purpose": "qualitative_diagnostic_not_causal_proof",
        "evaluation_git_sha": git_sha(), "seed": args.seed, "sample_selection": "first_distinct_GT_eligible_per_class",
        "min_class_pixels": args.min_class_pixels, "pretrained_sha256": weights_sha,
        "baseline": {"checkpoint": str(args.baseline_checkpoint), "sha256": sha256(args.baseline_checkpoint),
                     "training_git_sha": baseline_payload["metadata"]["git_sha"]},
        "static": {"checkpoint": str(args.static_checkpoint), "sha256": sha256(args.static_checkpoint),
                   "training_git_sha": static_payload["metadata"]["git_sha"]},
        "effect": "signed_alpha_times_delta_logits; red_positive_blue_negative; shared_scale_per_sample",
        "samples": rows,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
