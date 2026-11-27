"""Check whether the existing DINOv3 runtime can also train the pinned REIN head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from causalq.rein_head import load_rein_mask2former_head


REIN_SHA = "dc063429c4dadc0da9c6252b3db22fc55a9882ab"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def probe(rein_root: Path, report: dict) -> None:
    import torch
    import transformers

    report.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                  transformers=transformers.__version__)
    properties = torch.cuda.get_device_properties(0)
    report.update(gpu=properties.name, gpu_total_bytes=properties.total_memory)
    from transformers import DINOv3ViTModel

    report["dinov3_model"] = DINOv3ViTModel.__name__
    report["stage"] = "rein_import"
    import mmcv
    import mmdet
    import mmengine
    import mmseg
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    report.update(mmcv=mmcv.__version__, mmdet=mmdet.__version__,
                  mmengine=mmengine.__version__, mmseg=mmseg.__version__)
    head_class = load_rein_mask2former_head(rein_root)
    report["rein_head_source"] = str(rein_root / "rein/models/heads/rein_mask2former.py")
    report["stage"] = "rein_head_build"
    init_default_scope("mmseg")
    config = Config.fromfile(str(rein_root / "configs/_base_/models/rein_dinov2_mask2former.py"))
    head_config = config.model.decode_head.copy()
    if head_config.pop("type") != head_class.__name__:
        raise RuntimeError("Unexpected head type in the pinned REIN config")
    torch.manual_seed(0)
    head = head_class(**head_config).cuda().train()
    head.init_weights()
    features = [torch.randn(1, 1024, size, size, device="cuda", requires_grad=True)
                for size in (16, 8, 4, 2)]
    queries = torch.randn(100, 256, device="cuda", requires_grad=True)
    sample = SegDataSample(metainfo={"img_shape": (64, 64), "pad_shape": (64, 64)})
    label = (torch.arange(64, device="cuda") * 19 // 64).view(1, 1, 64).repeat(1, 64, 1)
    label[:, :2, :] = 255
    sample.gt_sem_seg = PixelData(data=label)
    report["stage"] = "rein_head_forward_backward"
    torch.cuda.reset_peak_memory_stats()
    classes, masks = head((features, queries), [sample])
    if classes[-1].shape != (1, 100, 20) or masks[-1].shape != (1, 100, 16, 16):
        raise RuntimeError("Unexpected Mask2Former output shape")
    instances, metas = head._seg_data_to_instance_data([sample])
    losses = head.loss_by_feat(classes, masks, instances, metas)
    expected = {f"loss_{name}" for name in ("cls", "mask", "dice")}
    if not expected.issubset(losses) or any(not torch.isfinite(item).item() for item in losses.values()):
        raise RuntimeError("Missing or non-finite Mask2Former losses")
    loss = sum(losses.values())
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite head outputs")
    for tensor in [*features, queries]:
        if (tensor.grad is None or not torch.isfinite(tensor.grad).all().item()
                or not torch.count_nonzero(tensor.grad).item()):
            raise RuntimeError("Missing, zero, or non-finite gradient into a head input")
    report.update(class_shape=list(classes[-1].shape), mask_shape=list(masks[-1].shape),
                  loss_terms=len(losses), loss=float(loss.detach()),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rein_root = args.rein_root.resolve()
    report = {"ok": False, "stage": "git", "python": sys.version,
              "python_path": sys.executable, "rein_root": str(rein_root),
              "formal_optimizer_updates": 0}
    try:
        report["git_sha"] = git(PROJECT_ROOT, "rev-parse", "HEAD")
        report["rein_sha"] = git(rein_root, "rev-parse", "HEAD")
        for root, actual, expected in ((PROJECT_ROOT, report["git_sha"], args.expected_sha),
                                       (rein_root, report["rein_sha"], REIN_SHA)):
            if actual != expected or git(root, "status", "--porcelain", "--untracked-files=no"):
                raise RuntimeError(f"Expected a clean checkout at {expected}: {root}")
        report["stage"] = "dinov3_import"
        probe(rein_root, report)
        report.update(ok=True, stage="complete")
    except Exception:
        report["error"] = traceback.format_exc()
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
