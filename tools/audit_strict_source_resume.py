"""Pinned acceptance of the returned deterministic CUDA continuation report."""

import hashlib
import json
import math
from pathlib import Path

from causalq.datasets.strict_source import MANIFEST_HASHES
from tools.deterministic_source_ce import BACKEND
from tools.strict_source_smoke import tree_digest
from tools.audit_strict_source_evaluation import REPORT_SHA as EVALUATION_SHA

REPORT_SHA = "8bd864462f8119447619bbafaa947596409ac76003900183b4dfdcfed065fde9"
PRODUCER_SHA = "5dc6caaa58668eb84cd79410fc69a5d719f9134d"


def validate_resume(report):
    expected = {"schema_version": 1, "phase": 17, "scope": "source_resume_8_vs_4_plus_4",
                "mode": "compare", "seed": 0, "git_sha": PRODUCER_SHA, "git_status": "",
                "ok": True, "formal_training_authorized": False, "target_data_accessed": False,
                "source_miou": None, "target_miou": None, "exact_resume_verified": True,
                "trace_exact": True, "model_exact": True, "optimizer_exact": True,
                "rng_exact": True, "sampler_exact": True, "final_iteration": 8,
                "split_iteration": 4, "total_optimizer_updates_across_processes": 16,
                "independent_worker_processes": 3, "manifest_sha256": MANIFEST_HASHES}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Resume acceptance mismatch: {key}")
    if report["source_evaluation_audit"]["report_sha256"] != EVALUATION_SHA:
        raise ValueError("Source evaluation provenance mismatch")
    workers = report["workers"]
    if set(workers) != {"continuous", "prefix", "resume"}:
        raise ValueError("Missing independent worker")
    if len({worker["process_id"] for worker in workers.values()}) != 3:
        raise ValueError("Worker process identities overlap")
    for mode, worker in workers.items():
        start, stop = {"continuous": (0, 8), "prefix": (0, 4), "resume": (4, 8)}[mode]
        if (worker["ok"] is not True or worker["mode"] != mode or worker["git_sha"] != PRODUCER_SHA
                or worker["git_status"] or worker["target_data_accessed"] is not False
                or worker["formal_training_authorized"] is not False
                or (worker["initial_iteration"], worker["final_iteration"]) != (start, stop)
                or worker["optimizer_updates_this_process"] != stop - start):
            raise ValueError(f"Worker protocol mismatch: {mode}")
        metadata, preflight = worker["metadata"], worker["loss_preflight"]
        if (metadata["loss_backend"] != BACKEND or metadata["deterministic_algorithms"] is not True
                or metadata["sample_workers"] != 0 or metadata["schedule_horizon"] != 40000
                or preflight["gpu_replay_exact"] is not True
                or preflight["strict_determinism_enabled"] is not True):
            raise ValueError("Deterministic runtime/CE preflight mismatch")
        for key, limit in (("cpu_reference_loss_max_error", 1e-5),
                           ("cpu_reference_gradient_max_error", 1e-6)):
            if not math.isfinite(preflight[key]) or not 0 <= preflight[key] <= limit:
                raise ValueError("CUDA CE reference error exceeds contract")
        if tree_digest(metadata) != tree_digest(workers["continuous"]["metadata"]):
            raise ValueError("Worker configuration/runtime identity mismatch")
    if workers["continuous"]["files_sha256"]["state.pth"] != workers["resume"]["files_sha256"]["state.pth"]:
        raise ValueError("Returned final checkpoint hashes differ")
    return {"source_resume_accepted": True, "report_sha256": REPORT_SHA,
            "exact_resume_verified": True, "target_data_accessed": False,
            "formal_training_authorized": False}


def audit_resume(path: Path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA:
        raise ValueError("Resume report SHA256 mismatch")
    return validate_resume(json.loads(payload))
