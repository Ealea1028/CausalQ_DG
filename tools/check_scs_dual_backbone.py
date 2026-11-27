"""P3 AutoDL check of both pretrained SCS backbones and the shared REIN head."""

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback

import torch

from causalq.datasets.segmentation import PairedSegmentationDataset, SegmentationPair, TrainTransform
from causalq.datasets.strict_source import load_membership, source_manifest_metadata
from causalq.interventions.style import StyleInterventionBank
from causalq.models.backbone_factory import build_backbone
from causalq.models.scs import SCSConfig
from causalq.models.scs_segmentor import REIN_TAPS, SCSBackbone, build_scs_segmentor, rein_pyramid
from causalq.runtime.scs_training import (
    build_scs_optimizer, load_scs_trainable_state, paired_scs_backward, scs_trainable_state,
)
from tools.check_rein_backbone import WEIGHT_SHA
from tools.probe_scs_rein_runtime import PROJECT_ROOT, REIN_SHA, git


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_source_samples(root: Path):
    train, _ = load_membership()
    pairs = [SegmentationPair(root / "images/images" / f"{key}.png",
                              root / "labels_trainIds" / f"{key}.png", key)
             for key in train[:2]]
    transform = TrainTransform((512, 512), scale_range=(1.0, 1.0), horizontal_flip_probability=0)
    torch.manual_seed(0)
    dataset = PairedSegmentationDataset(pairs, transform=transform)
    return [dataset[i] for i in range(len(dataset))]


@torch.no_grad()
def check_native_features(backbone: SCSBackbone) -> list[dict]:
    native = backbone.backbone
    results = []
    for height, width in ((512, 512), (512, 640)):
        images = torch.zeros(1, 3, height, width, device="cuda")
        captures = []
        with ExitStack() as cleanup:
            if not backbone.is_dino2:
                # HF publishes a normalized last hidden state. Reference the raw taps from the blocks.
                for layer in REIN_TAPS:
                    hook = native.model.layer[layer - 1].register_forward_hook(
                        lambda _module, _inputs, output: captures.append(output)
                    )
                    cleanup.callback(hook.remove)
            expected = native(images)
        features, aux = backbone(images, return_aux=True)
        if captures:
            grid = (height // 16, width // 16)
            taps = [native._to_patch_map(tokens, grid)[0] for tokens in captures]
        else:
            taps = expected.intermediate_maps
        for actual, reference in [(features.patch_tokens, expected.patch_tokens),
                                  (features.patch_map, expected.patch_map),
                                  (features.cls_token, expected.cls_token),
                                  (features.register_tokens, expected.register_tokens),
                                  *zip(features.intermediate_maps, taps, strict=True)]:
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
            if not torch.isfinite(actual).all().item():
                raise RuntimeError("DINO returned non-finite features")
        pyramid = rein_pyramid(features)
        if not torch.isfinite(aux).all().item():
            raise RuntimeError("SCS returned non-finite auxiliary logits")
        results.append({"image": [height, width], "patch": list(features.patch_tokens.shape),
                        "cls": list(features.cls_token.shape), "register": list(features.register_tokens.shape),
                        "dtype": str(features.patch_tokens.dtype), "device": str(features.patch_tokens.device),
                        "decoder": [list(x.shape) for x in pyramid], "aux": list(aux.shape)})
    return results


def required_grad(parameter) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all().item():
        raise RuntimeError("SCS gradient is absent or non-finite")
    norm = gradient.float().norm().item()
    if norm <= 0:
        raise RuntimeError("SCS gradient is zero")
    return norm


def check_model(name, weights, rein_root, samples, result):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    native = build_backbone(name, weights=weights).cuda().eval()
    if name.startswith("dinov2"):
        native.intermediate_indices = tuple(layer - 1 for layer in REIN_TAPS)
    backbone = SCSBackbone(native, SCSConfig()).cuda()
    result["native_features"] = check_native_features(backbone)
    model = build_scs_segmentor(backbone, rein_root).cuda().train()
    optimizer, scheduler = build_scs_optimizer(model)
    injector = backbone.injector
    result["scs_parameters"] = sum(p.numel() for p in injector.parameters())
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    versions = {name: p._version for name, p in native.named_parameters()}
    style = StyleInterventionBank().cuda()
    style_generator = torch.Generator(device="cuda").manual_seed(0)
    result["updates"] = []
    for step, sample in enumerate(samples, start=1):
        original = sample["image"].unsqueeze(0).cuda()
        labels = sample["label"].unsqueeze(0).cuda()
        styled, style_trace = style.photometric_view_with_trace(original, generator=style_generator)
        optimizer.zero_grad(set_to_none=True)
        loss = paired_scs_backward(model, original, styled, labels, auxiliary_weight=0.1 if step == 1 else 0,
                                   amp=True)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite objective")
        gradients = {f"up_{layer}": required_grad(bridge.up.weight) for layer, bridge in injector.bridges.items()}
        gradients["query"] = required_grad(injector.query_bank)
        if step == 2:
            gradients["early_down"] = required_grad(injector.bridges["12"].down.weight)
            gradients["read"] = required_grad(injector.read.in_proj_weight)
            gradients["write"] = required_grad(injector.write.in_proj_weight)
        optimizer.step()
        scheduler.step()
        result["updates"].append({"step": step, "source_id": sample["id"], "loss": loss.item(),
                                  "gradients": gradients, "style": style_trace, "lr": scheduler.get_last_lr()[0]})
    if any(p.grad is not None or p.requires_grad or versions[name] != p._version
           for name, p in native.named_parameters()):
        raise RuntimeError("Frozen DINO weights changed or acquired gradients")

    # A local serialization roundtrip of the compact state. Do not create a training donor.
    state_file = io.BytesIO()
    torch.save({"model": scs_trainable_state(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "source_cursor": len(samples)}, state_file)
    state_file.seek(0)
    state = torch.load(state_file, weights_only=True, map_location="cpu")
    expected_up = injector.bridges["12"].up.weight.detach().clone()
    with torch.no_grad():
        injector.bridges["12"].up.weight.zero_()
    load_scs_trainable_state(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    torch.testing.assert_close(injector.bridges["12"].up.weight, expected_up, rtol=0, atol=0)
    result.update(checkpoint_roundtrip=True, frozen_dino=True,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--gta-root", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "formal_optimizer_updates": 0, "python": sys.version,
              "python_path": sys.executable, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "target_data_accessed": False, "models": []}
    try:
        report["git_sha"] = git(PROJECT_ROOT, "rev-parse", "HEAD")
        report["rein_sha"] = git(args.rein_root, "rev-parse", "HEAD")
        for root, actual, expected in ((PROJECT_ROOT, report["git_sha"], args.expected_sha),
                                       (args.rein_root, report["rein_sha"], REIN_SHA)):
            if actual != expected or git(root, "status", "--porcelain", "--untracked-files=no"):
                raise RuntimeError(f"Git identity mismatch at {root}")
        import mmcv
        import mmdet
        import mmengine
        import mmseg
        import transformers

        gpu = torch.cuda.get_device_properties(0)
        report.update(transformers=transformers.__version__, mmcv=mmcv.__version__,
                      mmdet=mmdet.__version__, mmengine=mmengine.__version__, mmseg=mmseg.__version__,
                      gpu=gpu.name, gpu_total_bytes=gpu.total_memory, source=source_manifest_metadata())
        p0 = json.loads((PROJECT_ROOT / "docs/causalq_scs/p0_acceptance.json").read_text(encoding="utf-8-sig"))
        for key in ("torch", "cuda", "transformers", "mmcv", "mmdet", "mmengine", "mmseg"):
            if report[key] != p0[key]:
                raise RuntimeError(f"SCS runtime differs from P0: {key}")
        samples = read_source_samples(args.gta_root)
        historical = json.loads((PROJECT_ROOT / "experiments/LORA_CAUSALQ_1114/phase39_report.json")
                                .read_text(encoding="utf-8"))
        d3_hashes = next(model["weight_files"] for model in historical["models"]
                         if model["model"] == "dinov3_vitl16")
        for name, path, files in (
            ("dinov2_vitl14_hpi_p16", args.d2, {args.d2: WEIGHT_SHA}),
            ("dinov3_vitl16", args.d3, {args.d3 / file: digest for file, digest in d3_hashes.items()}),
        ):
            print(f"Checking {name}", flush=True)
            hashes = {str(file): file_sha(file) for file in files}
            if hashes != {str(file): digest for file, digest in files.items()}:
                raise RuntimeError(f"Pretrained weight identity mismatch: {name}")
            result = {"backbone": name, "weight_files": hashes}
            report["models"].append(result)
            check_model(name, path, args.rein_root, samples, result)
            torch.cuda.empty_cache()
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
