"""Bounded M0 optimizer/checkpoint engine, CPU-testable with synthetic features."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from causalq.losses import segmentation_cross_entropy
from causalq.utils.checkpoint import (load_training_checkpoint, save_training_checkpoint,
                                      trainable_state_dict)


UPDATES = 20
HOLDOUT_SAMPLES = 8
HORIZON = 40000


def tree_digest(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def precision(device, amp):
    return torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()


def optimizer_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                     loss_fn=segmentation_cross_entropy):
    if not 1 <= step <= UPDATES:
        raise ValueError("Only 20 bounded optimizer updates are authorized")
    return source_optimizer_update(model, optimizer, image, photo, label, step, config,
                                   device=device, amp=amp, loss_fn=loss_fn)


def source_optimizer_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                            loss_fn=segmentation_cross_entropy):
    """Same two-view update at an absolute step in the pinned 40k schedule."""
    if not 1 <= step <= HORIZON:
        raise ValueError("Source optimizer step must be within the 40000-step horizon")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    lr_used = optimizer.param_groups[0]["lr"]
    losses = {}
    for name, view in (("original", image), ("photometric", photo)):
        with precision(device, amp):
            logits = model(view)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite {name} logits")
            loss = loss_fn(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        weight = 1.0 if name == "original" else float(config["style"]["lambda_cf"])
        (weight * loss).backward()
        losses[name] = float(loss.detach())
    gradients = {name: float(p.grad.detach().float().norm()) for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError("Non-finite parameter after update")
    next_lr = float(config["optimizer"]["lr"]) * (1.0 - step / HORIZON) ** float(
        config["optimizer"]["poly_power"])
    for group in optimizer.param_groups:
        group["lr"] = next_lr
    return {"update": step, "loss_original": losses["original"],
            "loss_photometric": losses["photometric"],
            "objective": losses["original"] + float(config["style"]["lambda_cf"]) * losses["photometric"],
            "gradient_norm_pre_clip": norm, "lr_used": lr_used, "next_lr": next_lr,
            "alpha": float(model.query_head.alpha.detach()), "gradients": gradients}


def restore_compact(model, optimizer, path):
    checkpoint = load_training_checkpoint(path)
    expected_keys = set(trainable_state_dict(model))
    if set(checkpoint["trainable_model"]) != expected_keys:
        raise ValueError("Compact checkpoint trainable coverage mismatch")
    incompatible = model.load_state_dict(checkpoint["trainable_model"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(model.state_dict()) - expected_keys:
        raise ValueError("Unexpected compact checkpoint restoration coverage")
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def run_bounded(model, make_model, train, holdout, style_bank, config, output: Path,
                *, device, amp, metadata):
    if len(train) < UPDATES or len(holdout) < HOLDOUT_SAMPLES:
        raise ValueError("Insufficient source samples for the fixed smoke")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Foundation backbone must be frozen")
    frozen_before = tree_digest(model.backbone.state_dict())
    initial_trainable = tree_digest(trainable_state_dict(model))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["optimizer"]["lr"]),
                                 weight_decay=float(config["optimizer"]["weight_decay"]))
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0)).tolist()[:UPDATES]
    model.train()
    trace = []
    gradient_max = {}
    zero_error = None
    with (output / "train.jsonl").open("x", encoding="utf-8") as stream:
        for step, index in enumerate(order, 1):
            sample = train[index]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            photo = style_bank.photometric_view(image)
            if step == 1:
                if float(model.query_head.alpha.detach()) != 0:
                    raise ValueError("M0 requires alpha initialized to zero")
                model.eval()
                with torch.inference_mode(), precision(device, amp):
                    components = model.forward_components(image)
                    zero_error = float((components.logits - components.base_logits).abs().max())
                del components
                model.train()
                if zero_error != 0:
                    raise ValueError("Alpha-zero factual logits disagree with the base path")
            record = optimizer_update(model, optimizer, image, photo, label, step, config,
                                      device=device, amp=amp)
            record.update(source_id=sample["id"], input_sha256=tree_digest((image, photo, label)))
            for name, value in record["gradients"].items():
                gradient_max[name] = max(gradient_max.get(name, 0), value)
            trace.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps({key: value for key, value in record.items() if key != "gradients"}), flush=True)
    if len({row["source_id"] for row in trace}) != UPDATES:
        raise ValueError("Repeated source IDs in bounded smoke")
    for prefix in ("decoder.", "query_bank.class_anchors", "query_bank.query_residuals",
                   "query_head.feature_projection.", "query_head.query_projection.", "query_head.alpha"):
        if not any(value > 0 for name, value in gradient_max.items() if name.startswith(prefix)):
            raise ValueError(f"Missing nonzero training gradients: {prefix}")
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise ValueError("Frozen backbone acquired gradients")
    if initial_trainable == tree_digest(trainable_state_dict(model)):
        raise ValueError("No trainable parameter update occurred")
    checkpoint_path = output / "final.pth"
    save_training_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                             iteration=UPDATES, metadata=metadata)
    model.eval()
    compatibility = []
    with torch.inference_mode():
        for index in range(HOLDOUT_SAMPLES):
            sample = holdout[index]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            with precision(device, amp):
                logits = model(image)
                loss = segmentation_cross_entropy(logits, label)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source holdout prediction")
            compatibility.append({"source_id": sample["id"], "shape": list(label.shape),
                                  "ce": float(loss)})
    if set(row["source_id"] for row in compatibility) & set(row["source_id"] for row in trace):
        raise ValueError("Train/holdout overlap in smoke")
    # Same frozen backbone, newly initialized decoder/query parameters. This is
    # a serialization check, not optimizer/RNG/sampler continuation evidence.
    restored = make_model().to(device).eval()
    restored_optimizer = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad],
        lr=float(config["optimizer"]["lr"]), weight_decay=float(config["optimizer"]["weight_decay"]))
    checkpoint = restore_compact(restored, restored_optimizer, checkpoint_path)
    if checkpoint["iteration"] != UPDATES or checkpoint["metadata"] != metadata:
        raise ValueError("Compact checkpoint provenance mismatch")
    model_match = tree_digest(trainable_state_dict(model)) == tree_digest(trainable_state_dict(restored))
    optimizer_match = tree_digest(optimizer.state_dict()) == tree_digest(restored_optimizer.state_dict())
    if not model_match or not optimizer_match:
        raise ValueError("Compact model/optimizer state roundtrip mismatch")
    with torch.inference_mode(), precision(device, amp):
        reference = model(image)
        replay = restored(image)
        roundtrip_error = float((reference - replay).abs().max())
    if not torch.isfinite(replay).all() or roundtrip_error > 1e-5:
        raise ValueError("Compact prediction roundtrip exceeds 1e-5")
    frozen_after = tree_digest(model.backbone.state_dict())
    if frozen_after != frozen_before:
        raise ValueError("Frozen backbone state changed")
    return {"ok": True, "optimizer_updates": UPDATES, "source_ids": [row["source_id"] for row in trace],
            "holdout_compatibility": compatibility, "source_miou": None, "target_miou": None,
            "alpha_zero_error": zero_error, "gradient_max": gradient_max,
            "model_roundtrip_exact": model_match, "optimizer_roundtrip_exact": optimizer_match,
            "prediction_roundtrip_max_error": roundtrip_error,
            "backbone_unchanged": True, "backbone_state_sha256": frozen_before,
            "initial_trainable_state_sha256": initial_trainable,
            "final_trainable_state_sha256": tree_digest(trainable_state_dict(model)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "trace_sha256": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
            "schedule_horizon": HORIZON, "sample_workers": 0, "amp": amp,
            "checkpoint_policy": "fixed_final_only_no_selection", "exact_resume_verified": False}
