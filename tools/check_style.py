"""Generate and validate aligned Phase-7 style-intervention examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causalq.datasets import TrainTransform, gta5_dataset
from causalq.datasets.cityscapes import colorize_train_ids
from causalq.interventions import StyleInterventionBank
from causalq.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/style/gta_dinov3l_style.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("CAUSALQ_DATA_ROOT", "/root/autodl-tmp/datasets")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.getenv("CAUSALQ_OUTPUT_ROOT", "/root/autodl-tmp/outputs/CausalQ_DG")
        )
        / "analysis/style_examples",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def denormalize(image: torch.Tensor, bank: StyleInterventionBank) -> np.ndarray:
    unit = (image.float().unsqueeze(0) * bank.std + bank.mean).clamp(0.0, 1.0)
    return (
        unit.squeeze(0).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    transform = TrainTransform(
        config["train"]["crop_size"],
        scale_range=config["data"]["random_scale"],
        horizontal_flip_probability=config["data"]["horizontal_flip_probability"],
        min_valid_fraction=config["data"]["min_valid_fraction"],
        crop_attempts=config["data"]["crop_attempts"],
    )
    dataset = gta5_dataset(args.data_root / "gta5", transform=transform)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    sample = dataset[args.sample_index]
    image = sample["image"].unsqueeze(0)
    label = sample["label"]

    photo = config["style"]["photometric"]
    fourier = config["style"]["fourier"]
    bank = StyleInterventionBank(
        brightness=tuple(photo["brightness"]),
        contrast=tuple(photo["contrast"]),
        saturation=tuple(photo["saturation"]),
        gamma=tuple(photo["gamma"]),
        temperature=tuple(photo["temperature"]),
        grayscale_probability=float(photo["grayscale_probability"]),
        fourier_mix_strength=tuple(fourier["mix_strength"]),
    )
    views = bank(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    output_files: dict[str, str] = {}
    for name, tensor in views.items():
        panel = Image.fromarray(denormalize(tensor[0], bank))
        path = args.output_dir / f"{name}.png"
        panel.save(path)
        panels.append(panel)
        output_files[name] = str(path)
    gt = Image.fromarray(colorize_train_ids(label.byte().cpu().numpy()))
    gt_path = args.output_dir / "gt.png"
    gt.save(gt_path)
    panels.append(gt)
    output_files["gt"] = str(gt_path)

    montage = Image.new("RGB", (panels[0].width * len(panels), panels[0].height))
    for index, panel in enumerate(panels):
        montage.paste(panel, (index * panel.width, 0))
    montage_path = args.output_dir / "style_montage.png"
    montage.save(montage_path)
    output_files["montage"] = str(montage_path)

    report = {
        "ok": all(torch.isfinite(view).all().item() for _, view in views.items()),
        "sample_id": sample["id"],
        "seed": args.seed,
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "shared_geometry": all(view.shape == image.shape for _, view in views.items()),
        "photometric_mean_absolute_difference": float(
            (views.photometric - views.original).abs().mean()
        ),
        "fourier_mean_absolute_difference": float(
            (views.fourier - views.original).abs().mean()
        ),
        "outputs": output_files,
    }
    report["ok"] = bool(
        report["ok"]
        and report["shared_geometry"]
        and report["photometric_mean_absolute_difference"] > 0
        and report["fourier_mean_absolute_difference"] > 0
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
