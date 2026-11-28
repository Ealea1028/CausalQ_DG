"""Train a released SCS seed-0 source-only arm with resumable REIN updates."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

from causalq.datasets.strict_source import build_source_datasets, source_manifest_metadata
from causalq.models.backbone_factory import build_backbone
from causalq.models.scs import SCSConfig
from causalq.models.scs_segmentor import SCSBackbone, build_scs_segmentor
from causalq.runtime.identity import file_sha, tree_digest, write_json
from causalq.runtime.input_stream import worker_init
from causalq.runtime.scs_source import REIN_TRAIN_CONFIG, ReinPairedSource
from causalq.runtime.scs_training import (
    TOTAL_UPDATES, build_scs_optimizer, load_scs_trainable_state, paired_scs_backward, scs_trainable_state,
)
from tools.probe_scs_rein_runtime import PROJECT_ROOT, REIN_SHA, git


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def loader_workers(batch: int) -> int:
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text().split()
        if quota != "max":
            cpu = min(cpu, max(1, int(quota) // int(period)))
    memory = Path("/sys/fs/cgroup/memory.max")
    memory_in_use = Path("/sys/fs/cgroup/memory.current")
    host_workers = 8
    if memory.is_file() and memory_in_use.is_file():
        limit = memory.read_text().strip()
        if limit != "max":
            spare = int(limit) - int(memory_in_use.read_text())
            host_workers = max(0, (spare - (2 << 30)) // (512 << 20))
    shm = Path("/dev/shm")
    free = shutil.disk_usage(shm).free if shm.is_dir() else 0
    # Prefetch one batch per worker. Account for queued and in-flight copies.
    return max(0, min(host_workers, cpu - 2, int(free * 0.75) // (2 * batch * (9 << 20))))


def save_state(path, payload):
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def candidate(out, step):
    return out / f"source_candidate_{step:05d}.pt"


def publish_candidate(out, step, identity, model):
    path = candidate(out, step)
    if not path.is_file():
        save_state(path, {"step": step, "identity": identity, "model": scs_trainable_state(model)})


def run(args):
    from mmengine.config import Config
    import mmcv
    import mmdet
    import mmengine
    import mmseg
    import transformers

    p3 = read_json(PROJECT_ROOT / "docs/causalq_scs/p3_acceptance.json")
    if not p3["ok"]:
        raise RuntimeError("The P3 dual-backbone gate must pass before formal training")
    config_path = PROJECT_ROOT / "configs/scs/p4_seed0.json"
    config = read_json(config_path)
    if args.arm not in config["released_arms"] or config["source_batch"] != 4 or config["total_updates"] != TOTAL_UPDATES:
        raise ValueError("Unreleased experiment arm or invalid formal budget")
    recipe = config["arms"][args.arm]
    record = next(row for row in p3["models"] if row["backbone"] == args.backbone)
    if not record["ok"]:
        raise RuntimeError("Backbone failed P3")
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                   transformers=transformers.__version__, mmcv=mmcv.__version__,
                   mmdet=mmdet.__version__, mmengine=mmengine.__version__, mmseg=mmseg.__version__)
    if any(p3[name] != value for name, value in runtime.items()):
        raise RuntimeError("Runtime package versions differ from P3")
    weights = record["weight_files"]
    if any(file_sha(path) != digest for path, digest in weights.items()):
        raise RuntimeError("Pretrained weight identity differs from P3")
    first = Path(next(iter(weights)))
    weights_path = first if args.backbone.startswith("dinov2") else first.parent
    identity = dict(git_sha=args.expected_sha, rein_sha=REIN_SHA, arm=args.arm, backbone=args.backbone,
                    weight_files=weights, runtime=runtime, seed=config["seed"], recipe=recipe,
                    config_sha256=file_sha(config_path), source=source_manifest_metadata(),
                    source_batch=4, schedule_updates=TOTAL_UPDATES, aux=recipe["aux_weight"],
                    precision="bf16_backbone_fp32_head", source_selection=config["source_selection"])
    out = args.output
    summary_path = out / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary["identity"] != identity or summary["completed_updates"] != TOTAL_UPDATES or any(
            file_sha(out / name) != digest for name, digest in summary["candidate_sha256"].items()
        ):
            raise RuntimeError("Existing completed run mismatches the requested experiment")
        print(f"Complete: {out}", flush=True)
        return

    seed = config["seed"]
    torch.manual_seed(seed)
    native = build_backbone(args.backbone, weights=weights_path)
    scs = SCSConfig(selected_layers=tuple(recipe["layers"])) if recipe["layers"] else None
    backbone = SCSBackbone(native, scs)
    # Same decoder initialization in all arms and both backbones.
    torch.manual_seed(seed)
    model = build_scs_segmentor(backbone, args.rein_root).cuda().train()
    optimizer, scheduler = build_scs_optimizer(model)
    torch.manual_seed(seed)
    decoder_digest = tree_digest({"head": model.head.state_dict(), "query": model.query_embed.state_dict()})
    latest = out / "latest.pt"
    step = 0
    if latest.is_file():
        state = torch.load(latest, weights_only=True, map_location="cpu")
        if state["identity"] != identity or state["step"] != state["scheduler"]["last_epoch"]:
            raise RuntimeError("Resume state does not match the experiment")
        step = state["step"]
        if not 0 <= step <= TOTAL_UPDATES:
            raise RuntimeError("Invalid resume cursor")
        load_scs_trainable_state(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        del state
    steps_for_selection = config["source_selection_steps"]
    if step in steps_for_selection:
        publish_candidate(out, step, identity, model)

    if step < TOTAL_UPDATES:
        pairs = build_source_datasets(args.gta_root, None, None)[0].pairs
        rein_train = Config.fromfile(str(args.rein_root / REIN_TRAIN_CONFIG))
        pipeline = rein_train.train_pipeline[2:6]
        if [item["type"] for item in pipeline] != [
            "RandomChoiceResize", "RandomCrop", "RandomFlip", "PhotoMetricDistortion",
        ]:
            raise ValueError("Unexpected pinned REIN transform sequence")
        stream = ReinPairedSource(pairs, pipeline, seed=seed, start=4 * step, stop=4 * TOTAL_UPDATES)
        workers = loader_workers(4)
        loader_options = dict(num_workers=workers, batch_size=4, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed), drop_last=True)
        if workers:
            loader_options.update(prefetch_factor=1, persistent_workers=True, worker_init_fn=worker_init,
                                  multiprocessing_context="spawn")
        stream_loader = DataLoader(stream, **loader_options)
        write_json(out / "run.json", dict(identity=identity, decoder_initial_sha256=decoder_digest,
                                        workers=workers, physical_batch=4, resumed_from=step,
                                        gpu=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                                        python=sys.version, executable=sys.executable, rein_pipeline=pipeline))
        version = {key: value._version for key, value in native.named_parameters()}
        start_step = step
        window_time = available_at = time.monotonic()
        data_wait = 0.0
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(stream_loader, start=step + 1):
            data_wait += time.monotonic() - available_at
            views = batch["views"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            mask = batch["patch_valid"].cuda(non_blocking=True) if not batch["patch_valid"].all() else None
            optimizer.zero_grad(set_to_none=True)
            loss = paired_scs_backward(model, views[:, 0], views[:, 1], labels,
                                       auxiliary_weight=recipe["aux_weight"], amp=True, patch_valid=mask)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite task loss at {step}")
            if step == start_step + 1:
                finite = [p.grad.isfinite().all() for p in model.parameters() if p.grad is not None]
                if not finite or not torch.stack(finite).all().item():
                    raise RuntimeError("First optimizer update has absent or nonfinite gradients")
            optimizer.step()
            scheduler.step()
            if step % 50 == 0 or step == start_step + 1:
                elapsed = time.monotonic() - window_time
                updates = 1 if step == start_step + 1 else step - window_step
                print(json.dumps(dict(step=step, loss=value, lr=scheduler.get_last_lr()[0],
                                      seconds=elapsed, updates_per_s=updates / elapsed,
                                      source_images_per_s=4 * updates / elapsed, workers=workers,
                                      data_wait_fraction=data_wait / elapsed,
                                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved())), flush=True)
                window_step, window_time = step, time.monotonic()
                data_wait = 0.0
            if step == start_step + 1 or step % config["resume_interval"] == 0 or step in steps_for_selection:
                if any(p.grad is not None or p._version != version[key] for key, p in native.named_parameters()):
                    raise RuntimeError("Frozen DINO was modified")
                save_state(latest, dict(step=step, identity=identity, model=scs_trainable_state(model),
                                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()))
                if step in steps_for_selection:
                    publish_candidate(out, step, identity, model)
            available_at = time.monotonic()
    if step != TOTAL_UPDATES:
        raise RuntimeError("Training did not reach the locked horizon")
    candidates = {candidate(out, step).name: file_sha(candidate(out, step)) for step in steps_for_selection}
    write_json(summary_path, dict(identity=identity, completed_updates=step, candidate_sha256=candidates,
                                  decoder_initial_sha256=decoder_digest, target_data_accessed=False))
    print(f"Complete: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--rein-root", type=Path, required=True)
    parser.add_argument("--gta-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=("dinov2_vitl14_hpi_p16", "dinov3_vitl16"), required=True)
    parser.add_argument("--arm", choices=("A0", "A4"), required=True)
    args = parser.parse_args()
    for root, expected in ((PROJECT_ROOT, args.expected_sha), (args.rein_root, REIN_SHA)):
        if git(root, "rev-parse", "HEAD") != expected or git(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Git identity mismatch at {root}")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
