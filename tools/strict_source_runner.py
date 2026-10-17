"""Source-only runner state, fixed-final policy and stochastic continuation."""

import json
import os
from pathlib import Path

import torch

from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from tools.strict_source_smoke import source_optimizer_update, restore_compact, tree_digest
from tools.strict_source_resume import capture_rng, restore_rng
from tools.deterministic_source_ce import deterministic_source_ce
from tools.strict_source_evaluation import evaluate_holdout


def sampler_state(count, iteration, seed=0):
    if count <= 0 or not 0 <= iteration <= 40000:
        raise ValueError("Invalid source sampler count/iteration")
    epoch, cursor = divmod(iteration, count)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return {"count": count, "epoch": epoch, "cursor": cursor, "seed": seed, "order": order}


def save_runner_state(path, model, optimizer, iteration, count, metadata):
    if path.exists():
        raise ValueError("Refusing to overwrite source checkpoint")
    payload = {"schema_version": 2, "iteration": iteration, "metadata": metadata,
               "trainable_model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
               "backbone_sha256": tree_digest(model.backbone.state_dict()), "rng": capture_rng(),
               "sampler": sampler_state(count, iteration, metadata["seed"])}
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


def train_segment(model, optimizer, dataset, style, config, output, *, start, stop,
                  metadata, resume=None, device="cuda:0", amp=True):
    if not 0 <= start < stop <= 40000 or (resume is not None) != (start > 0):
        raise ValueError("Invalid source continuation bounds")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Source backbone must remain frozen")
    if set(model.state_dict()) != set(trainable_state_dict(model)) | {
            "backbone." + key for key in model.backbone.state_dict()}:
        raise ValueError("Unsupported mutable/frozen head state")
    frozen = tree_digest(model.backbone.state_dict())
    if resume is not None:
        payload = load_training_checkpoint(resume)
        if (payload.get("schema_version") != 2 or payload["metadata"] != metadata
                or payload["iteration"] != start or payload["backbone_sha256"] != frozen
                or payload["sampler"] != sampler_state(len(dataset), start, metadata["seed"])):
            raise ValueError("Runner checkpoint protocol/sampler identity mismatch")
        restore_compact(model, optimizer, resume)
        lr = float(config["optimizer"]["lr"]) * (1 - start / 40000) ** float(config["optimizer"]["poly_power"])
        if any(group["lr"] != lr for group in optimizer.param_groups):
            raise ValueError("Runner restored LR does not match absolute iteration")
        restore_rng(payload["rng"])
    model.train()
    sampler = sampler_state(len(dataset), start, metadata["seed"])
    with (output / "train.jsonl").open("x", encoding="utf-8") as stream:
        for iteration in range(start, stop):
            if sampler["cursor"] == len(dataset):
                sampler = sampler_state(len(dataset), iteration, metadata["seed"])
            sample = dataset[sampler["order"][sampler["cursor"]]]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            photo = style.photometric_view(image)
            fingerprint = tree_digest((image, photo, label))
            record = source_optimizer_update(model, optimizer, image, photo, label, iteration + 1,
                                             config, device=device, amp=amp,
                                             loss_fn=deterministic_source_ce)
            record.update(source_id=sample["id"], input_sha256=fingerprint, epoch=sampler["epoch"])
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            sampler["cursor"] += 1
            if (iteration + 1) % 10 == 0:
                print(f"source_runner_update={iteration + 1}/{stop} objective={record['objective']}", flush=True)
    if frozen != tree_digest(model.backbone.state_dict()) or any(p.grad is not None for p in model.backbone.parameters()):
        raise ValueError("Source backbone state or gradient changed")
    save_runner_state(output / "state.pth", model, optimizer, stop, len(dataset), metadata)
    return {"initial_iteration": start, "final_iteration": stop, "optimizer_updates": stop - start,
            "backbone_unchanged": True, "backbone_sha256": frozen,
            "head_sha256": tree_digest(trainable_state_dict(model)), "checkpoint_selection": "fixed_final"}


def evaluate_without_training_side_effects(model, optimizer, dataset, ids, output, *, device="cuda:0", amp=True):
    before = tree_digest((model.state_dict(), optimizer.state_dict(), capture_rng()))
    was_training = model.training
    try:
        result = evaluate_holdout(model, dataset, ids, output, device=device, amp=amp,
                                  loss_fn=deterministic_source_ce)
    finally:
        model.train(was_training)
    if before != tree_digest((model.state_dict(), optimizer.state_dict(), capture_rng())):
        raise ValueError("Source validation modified model/optimizer/RNG state")
    return {**result, "training_state_unchanged": True, "model_mode_restored": True}
