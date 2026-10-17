"""Saved evidence audit and stochastic continuation, including fresh processes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalq.interventions import StyleInterventionBank
from causalq.utils.seed import seed_everything
from tools.audit_strict_source_evaluation import audit_evaluation, validate_evaluation
from tools.strict_source_resume import capture_rng, restore_rng, run_segment, compare_states
from tools.strict_source_smoke import tree_digest
from tools.train import sha256
from test_strict_source_smoke import fixture


def test_accepted_full_source_report_and_recomputed_metrics(tmp_path):
    source = ROOT / "experiments/STRICT_DG_PHASE17/source_evaluation.json"
    result = audit_evaluation(source)
    assert result["source_evaluation_accepted"]
    assert result["source_miou"] == pytest.approx(0.12971660656076558)
    altered = tmp_path / "changed.json"
    altered.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_evaluation(altered)


@pytest.mark.parametrize("key,value", [
    ("target_data_accessed", True), ("sample_count", 2496), ("source_miou", float("nan")),
    ("valid_pixel_count", 0), ("compatibility_ce_max_error", 0.1),
    ("class_iou", [0.] * 19), ("exact_resume_verified", True),
])
def test_returned_evaluation_rejects_changed_contract(key, value):
    report = json.loads((ROOT / "experiments/STRICT_DG_PHASE17/source_evaluation.json").read_bytes())
    report[key] = value
    with pytest.raises(ValueError):
        validate_evaluation(report)


def test_rng_roundtrip_covers_python_numpy_and_torch():
    seed_everything(31)
    np.random.normal()  # Include cached Gaussian state.
    saved = capture_rng()
    def draws():
        return (random.random(), np.random.normal(), torch.rand(5))
    before = draws()
    restore_rng(saved)
    assert tree_digest(before) == tree_digest(draws())


class StochasticDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = deepcopy(self.items[index])
        item["image"] = item["image"] * (0.9 + 0.1 * random.random())
        item["image"] += float(np.random.uniform(-0.01, 0.01)) + torch.rand_like(item["image"]) * 0.01
        return item


def worker(mode, root):
    seed_everything(0)
    make, config, items, _ = fixture()
    model = make()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-4, weight_decay=0.01)
    dataset = StochasticDataset(items)
    output = root / mode
    output.mkdir()
    start, stop = {"continuous": (0, 8), "prefix": (0, 4), "resume": (4, 8)}[mode]
    records = run_segment(model, optimizer, dataset, StyleInterventionBank(), config, output,
                          metadata={"synthetic": True, "target_data_accessed": False},
                          start=start, stop=stop, resume=root / "prefix/state.pth" if start else None,
                          device="cpu", amp=False)
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "worker_report.json").write_text(json.dumps({
        "ok": True, "git_sha": "a" * 40, "git_status": "", "mode": mode,
        "target_data_accessed": False, "process_id": os.getpid(),
        "files_sha256": {name: sha256(output / name) for name in ("train.jsonl", "state.pth")},
    }))


def test_fresh_process_stochastic_8_equals_4_plus_4(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    for mode in ("continuous", "prefix", "resume"):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", mode, str(tmp_path)],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
    result = compare_states(tmp_path / "continuous", tmp_path / "prefix", tmp_path / "resume")
    assert result["exact_resume_verified"] and result["total_optimizer_updates_across_processes"] == 16
    from tools.check_strict_source_resume import verify_workers
    assert verify_workers(tmp_path, "a" * 40)["independent_worker_processes"] == 3
    report_path = tmp_path / "resume/worker_report.json"
    report_text = report_path.read_text()
    worker_report = json.loads(report_text)
    worker_report["process_id"] = json.loads((tmp_path / "prefix/worker_report.json").read_text())["process_id"]
    report_path.write_text(json.dumps(worker_report))
    with pytest.raises(ValueError, match="independent worker"):
        verify_workers(tmp_path, "a" * 40)
    report_path.write_text(report_text)
    trace = tmp_path / "resume/train.jsonl"
    original_trace = trace.read_text()
    rows = [json.loads(line) for line in original_trace.splitlines()]
    rows[0]["input_sha256"] = "0" * 64
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="artifact changed"):
        verify_workers(tmp_path, "a" * 40)
    with pytest.raises(ValueError, match="traces differ"):
        compare_states(tmp_path / "continuous", tmp_path / "prefix", tmp_path / "resume")
    trace.write_text(original_trace)
    state_path = tmp_path / "resume/state.pth"
    original = torch.load(state_path, weights_only=True)
    for key in ("optimizer", "rng", "sampler", "trainable_model"):
        modified = deepcopy(original)
        modified[key] = {"wrong": 1}
        torch.save(modified, state_path)
        with pytest.raises(ValueError, match=f"mismatch: {key}"):
            compare_states(tmp_path / "continuous", tmp_path / "prefix", tmp_path / "resume")


@pytest.mark.parametrize("field", ["metadata", "sampler", "backbone_sha256", "iteration"])
def test_resume_rejects_wrong_provenance_before_updates(tmp_path, field):
    worker("prefix", tmp_path)
    path = tmp_path / "prefix/state.pth"
    state = torch.load(path, weights_only=True)
    state[field] = {"wrong": True} if field in ("metadata", "sampler") else "wrong"
    torch.save(state, path)
    with pytest.raises(ValueError, match="protocol/sampler/backbone"):
        worker("resume", tmp_path)
    assert not (tmp_path / "resume/state.pth").exists()


def test_resume_refuses_formal_training_budget(tmp_path):
    make, config, items, _ = fixture()
    model = make()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    with pytest.raises(ValueError, match="fixed 8 or 4"):
        run_segment(model, optimizer, items, StyleInterventionBank(), config, tmp_path,
                    metadata={}, start=0, stop=40000, device="cpu", amp=False)


def test_real_launcher_stops_before_weights_on_cpu(tmp_path, monkeypatch):
    from tools import check_strict_source_resume as launcher
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path / "checkout")
    for name in ("experiments/STRICT_DG_PHASE17/source_evaluation.json",
                 "configs/strict_dg/gta_dinov3l_m0.yaml"):
        destination = launcher.PROJECT_ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(launcher, "_git", lambda *a: "" if a[0] == "status" else "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(launcher.DINOv3Backbone, "from_pretrained",
                        lambda *a, **k: pytest.fail("Local real weight loading attempted"))
    output = tmp_path / "output"
    assert launcher.main(["--mode", "continuous", "--data-root", str(tmp_path / "data"),
                          "--pretrained-root", str(tmp_path / "weights"), "--output-root", str(output)]) == 1
    report = json.loads((output / "continuous/worker_report.json").read_bytes())
    assert not report["ok"] and "AutoDL CUDA" in report["error"]
    assert report["source_evaluation_audit"]["source_evaluation_accepted"]


def test_atomic_checkpoint_failure_leaves_no_final_or_partial_file(tmp_path, monkeypatch):
    from tools.strict_source_resume import save_state
    make, _, items, _ = fixture()
    model = make()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    def fail(*args, **kwargs):
        raise OSError("simulated disk error")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        save_state(tmp_path / "state.pth", model, optimizer, 4, len(items), {})
    assert not (tmp_path / "state.pth").exists()
    assert not (tmp_path / "state.partial").exists()


if __name__ == "__main__":
    assert sys.argv[1] == "worker"
    worker(sys.argv[2], Path(sys.argv[3]))
