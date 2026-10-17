"""Bounded checkpoint continuation engine; no formal trainer or target access."""

import os
from pathlib import Path
import random

import numpy as np
import torch

from causalq.utils.checkpoint import trainable_state_dict
from tools.strict_source_smoke import optimizer_update, restore_compact, tree_digest

TOTAL = 8
BOUNDARY = 4


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Checkpoint CUDA RNG device coverage differs")
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def sample_order(count):
    if count < TOTAL:
        raise ValueError("Insufficient source samples for resume diagnostic")
    return torch.randperm(count, generator=torch.Generator().manual_seed(0)).tolist()


def save_state(path, model, optimizer, cursor, count, metadata):
    if path.exists() or not 0 < cursor <= TOTAL:
        raise ValueError("Checkpoint already exists or cursor outside bounded gate")
    payload = {"schema_version": 1, "iteration": cursor,
               "trainable_model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
               "metadata": metadata, "rng": capture_rng(),
               "sampler": {"count": count, "order": sample_order(count), "cursor": cursor},
               "backbone_sha256": tree_digest(model.backbone.state_dict())}
    temporary = path.with_suffix(".partial")
    stream = temporary.open("xb")
    try:
        with stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_segment(model, optimizer, train, style, config, output: Path, *, metadata,
                start, stop, resume=None, device="cuda:0", amp=True):
    if (start, stop) not in ((0, BOUNDARY), (0, TOTAL), (BOUNDARY, TOTAL)):
        raise ValueError("Only fixed 8 or 4+4 update diagnostics are authorized")
    if (resume is not None) != (start == BOUNDARY):
        raise ValueError("Resume path required exactly at update four")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Backbone must remain frozen")
    if set(model.state_dict()) != set(trainable_state_dict(model)) | {
            "backbone." + name for name in model.backbone.state_dict()}:
        raise ValueError("Unsupported mutable/frozen head state in compact continuation")
    order = sample_order(len(train))
    frozen = tree_digest(model.backbone.state_dict())
    model.train()
    if resume is not None:
        state = restore_compact(model, optimizer, resume)
        if (state.get("schema_version") != 1 or state["metadata"] != metadata
                or state["iteration"] != start or state["backbone_sha256"] != frozen
                or state["sampler"] != {"count": len(train), "order": order, "cursor": start}):
            raise ValueError("Resume checkpoint protocol/sampler/backbone mismatch")
        expected_lr = float(config["optimizer"]["lr"]) * (1 - start / 40000) ** float(config["optimizer"]["poly_power"])
        if any(group["lr"] != expected_lr for group in optimizer.param_groups):
            raise ValueError("Restored optimizer LR disagrees with iteration")
        restore_rng(state["rng"])
    records = []
    for cursor in range(start, stop):
        sample = train[order[cursor]]
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].unsqueeze(0).to(device)
        photo = style.photometric_view(image)
        fingerprint = tree_digest((image, photo, label))
        row = optimizer_update(model, optimizer, image, photo, label, cursor + 1,
                               config, device=device, amp=amp)
        row.update(source_id=sample["id"], input_sha256=fingerprint)
        records.append(row)
        print(f"source_resume_update={cursor + 1}/{stop}", flush=True)
    if frozen != tree_digest(model.backbone.state_dict()) or any(
            p.grad is not None for p in model.backbone.parameters()):
        raise ValueError("Frozen backbone changed during continuation")
    save_state(output / "state.pth", model, optimizer, stop, len(train), metadata)
    return records


def compare_states(continuous, prefix, resumed):
    def load(directory):
        from causalq.utils.checkpoint import load_training_checkpoint
        import json
        return (load_training_checkpoint(directory / "state.pth"),
                [json.loads(row) for row in (directory / "train.jsonl").read_text().splitlines()])
    first, trace = load(continuous)
    middle, prefix_trace = load(prefix)
    last, resumed_trace = load(resumed)
    if (first["iteration"], middle["iteration"], last["iteration"]) != (TOTAL, BOUNDARY, TOTAL):
        raise ValueError("Continuation iteration coverage mismatch")
    if len(prefix_trace) != BOUNDARY or len(resumed_trace) != TOTAL - BOUNDARY:
        raise ValueError("Continuation segment coverage mismatch")
    for key in ("metadata", "backbone_sha256"):
        if tree_digest(first[key]) != tree_digest(middle[key]):
            raise ValueError(f"Prefix provenance mismatch: {key}")
    if len(trace) != TOTAL or trace != prefix_trace + resumed_trace:
        raise ValueError("Continuous versus resumed input/loss/LR/gradient traces differ")
    if [row["update"] for row in trace] != list(range(1, TOTAL + 1)):
        raise ValueError("Noncontiguous continuation trace")
    for key in ("metadata", "trainable_model", "optimizer", "rng", "sampler", "backbone_sha256"):
        if tree_digest(first[key]) != tree_digest(last[key]):
            raise ValueError(f"Continuous versus resumed state mismatch: {key}")
    return {"exact_resume_verified": True, "trace_exact": True, "model_exact": True,
            "optimizer_exact": True, "rng_exact": True, "sampler_exact": True,
            "final_iteration": TOTAL, "split_iteration": BOUNDARY,
            "total_optimizer_updates_across_processes": 16, "target_data_accessed": False,
            "formal_training_authorized": False, "source_miou": None, "target_miou": None}
