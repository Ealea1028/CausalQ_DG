"""Source runner integration, epoch boundaries, validation isolation and gates."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from causalq.interventions import StyleInterventionBank
from causalq.utils.seed import seed_everything
from causalq.utils.checkpoint import load_training_checkpoint
from tools.audit_strict_source_resume import audit_resume, validate_resume
from tools.strict_source_runner import train_segment, sampler_state, evaluate_without_training_side_effects
from tools.strict_source_smoke import tree_digest, source_optimizer_update
from tools import train_strict_source as launcher
from test_strict_source_smoke import fixture
from test_strict_source_resume import StochasticDataset

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_gpu_resume_report_passes_and_rejects_altered_bytes(tmp_path):
    source = ROOT / "experiments/STRICT_DG_PHASE17/resume_report.json"
    assert audit_resume(source)["source_resume_accepted"]
    path = tmp_path / "altered.json"
    path.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_resume(path)


@pytest.mark.parametrize("key,value", [("exact_resume_verified", False),
    ("total_optimizer_updates_across_processes", 17), ("target_data_accessed", True),
    ("independent_worker_processes", 2)])
def test_resume_acceptance_rejects_changed_contract(key, value):
    report = json.loads((ROOT / "experiments/STRICT_DG_PHASE17/resume_report.json").read_bytes())
    report[key] = value
    with pytest.raises(ValueError):
        validate_resume(report)


def make_state():
    seed_everything(0)
    make, config, items, _ = fixture()
    model = make()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    return model, optimizer, StochasticDataset(items[:5]), config


def test_runner_resume_across_epochs_beyond_smoke_budget(tmp_path):
    metadata = {"seed": 0, "target_data_accessed": False}
    outputs = [tmp_path / name for name in ("full", "prefix", "finish")]
    for path in outputs:
        path.mkdir()
    model, optimizer, dataset, config = make_state()
    train_segment(model, optimizer, dataset, StyleInterventionBank(), config, outputs[0],
                  start=0, stop=25, metadata=metadata, device="cpu", amp=False)
    model, optimizer, dataset, config = make_state()
    train_segment(model, optimizer, dataset, StyleInterventionBank(), config, outputs[1],
                  start=0, stop=10, metadata=metadata, device="cpu", amp=False)
    model, optimizer, dataset, config = make_state()
    train_segment(model, optimizer, dataset, StyleInterventionBank(), config, outputs[2],
                  start=10, stop=25, metadata=metadata, resume=outputs[1] / "state.pth", device="cpu", amp=False)
    full, resumed = [load_training_checkpoint(path / "state.pth") for path in (outputs[0], outputs[2])]
    for key in ("trainable_model", "optimizer", "rng", "sampler"):
        assert tree_digest(full[key]) == tree_digest(resumed[key])
    assert full["sampler"] == sampler_state(5, 25)
    assert full["sampler"]["epoch"] == 5 and full["sampler"]["cursor"] == 0
    assert (outputs[0] / "train.jsonl").read_text() == (outputs[1] / "train.jsonl").read_text() + (outputs[2] / "train.jsonl").read_text()
    bad = deepcopy(full)
    bad["sampler"]["cursor"] = 1
    torch.save(bad, outputs[0] / "bad.pth")
    with pytest.raises(ValueError, match="identity mismatch"):
        train_segment(model, optimizer, dataset, StyleInterventionBank(), config, tmp_path,
                      start=25, stop=26, metadata=metadata, resume=outputs[0] / "bad.pth", device="cpu", amp=False)


def test_generic_step_keeps_absolute_40k_schedule_after_update20():
    model, optimizer, dataset, config = make_state()
    sample = dataset[0]
    optimizer.param_groups[0]["lr"] = 1e-4 * (1 - 20 / 40000) ** 0.9
    result = source_optimizer_update(model, optimizer, sample["image"].unsqueeze(0),
        sample["image"].unsqueeze(0), sample["label"].unsqueeze(0), 21, config, device="cpu", amp=False)
    assert result["next_lr"] == pytest.approx(1e-4 * (1 - 21 / 40000) ** 0.9)
    with pytest.raises(ValueError, match="40000"):
        source_optimizer_update(model, optimizer, None, None, None, 40001, config, device="cpu", amp=False)


class TinyEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 19, 1)

    def forward(self, images):
        return self.conv(images)


def test_validation_restores_training_mode_and_all_training_state(tmp_path):
    model = TinyEvalModel().train()
    optimizer = torch.optim.AdamW(model.parameters())
    dataset = [{"id": "00001", "image": torch.ones(3, 16, 16), "label": torch.zeros(16, 16, dtype=torch.long)}]
    result = evaluate_without_training_side_effects(model, optimizer, dataset, ["00001"], tmp_path, device="cpu", amp=False)
    assert result["training_state_unchanged"] and result["sample_count"] == 1
    assert model.training and (tmp_path / "source_eval.jsonl").exists()


def test_source_runner_imports_no_legacy_or_target_module_in_fresh_process():
    script = """
import importlib.abc, sys
class RejectTarget(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in ('tools.train', 'causalq.datasets.cityscapes'):
            raise AssertionError('target or legacy import: ' + fullname)
sys.meta_path.insert(0, RejectTarget())
import tools.train_strict_source
assert 'causalq.datasets.cityscapes' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_label_exports_are_preserved():
    from causalq.datasets import cityscapes, label_space
    for name in ("IGNORE_INDEX", "NUM_CLASSES", "LABEL_ID_TO_TRAIN_ID", "TRAIN_ID_PALETTE",
                 "read_index_mask", "label_ids_to_train_ids", "invalid_train_ids", "colorize_train_ids"):
        assert getattr(cityscapes, name) is getattr(label_space, name)


def test_pilot_cli_audits_evidence_and_stops_before_real_weights_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path / "checkout")
    for name in ("experiments/STRICT_DG_PHASE17/resume_report.json", "configs/strict_dg/gta_dinov3l_m0.yaml"):
        destination = launcher.ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(launcher, "_git", lambda *args: "" if args[0] == "status" else "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(launcher.DINOv3Backbone, "from_pretrained", lambda *a, **k: pytest.fail("Local real weights attempted"))
    # Other tests import the historical label compatibility module. Remove it
    # temporarily; the fresh-process poison test above proves runner isolation.
    monkeypatch.delitem(sys.modules, "causalq.datasets.cityscapes", raising=False)
    monkeypatch.delitem(sys.modules, "tools.train", raising=False)
    output = tmp_path / "pilot"
    assert launcher.main(["--stage", "prefix", "--data-root", str(tmp_path / "data"),
                          "--pretrained-root", str(tmp_path / "weights"), "--output-root", str(output)]) == 1
    report = json.loads((output / "prefix/worker_report.json").read_bytes())
    assert report["resume_audit"]["source_resume_accepted"]
    assert not report["ok"] and "AutoDL CUDA" in report["error"]


def test_pilot_cli_has_no_budget_override(tmp_path):
    with pytest.raises(SystemExit):
        launcher.main(["--stage", "prefix", "--data-root", str(tmp_path), "--pretrained-root", str(tmp_path),
                       "--output-root", str(tmp_path / "out"), "--max-iterations", "40000"])


def test_pilot_summary_checks_trace_and_selected_checkpoint(tmp_path, monkeypatch):
    from causalq.datasets.strict_source import MANIFEST_HASHES
    monkeypatch.setattr(launcher, "PILOT_STEPS", 4)
    monkeypatch.setattr(launcher, "BOUNDARY", 2)
    ids = [f"{i:05d}" for i in range(5)]
    monkeypatch.setattr(launcher, "load_membership", lambda: (ids, ["99999"]))
    metadata = {"seed": 0, "target_data_accessed": False}
    model, optimizer, dataset, config = make_state()
    for item, sample_id in zip(dataset.items, ids):
        item["id"] = sample_id
    for stage, start, stop in (("prefix", 0, 2), ("finish", 2, 4)):
        folder = tmp_path / stage
        folder.mkdir()
        report = train_segment(model, optimizer, dataset, StyleInterventionBank(), config, folder,
                               start=start, stop=stop, metadata=metadata,
                               resume=tmp_path / "prefix/state.pth" if start else None,
                               device="cpu", amp=False)
        report.update(ok=True, git_sha="a" * 40, git_status="", process_id=start + 100,
                      metadata=metadata, target_data_accessed=False, formal_training_authorized=False,
                      manifest_sha256=MANIFEST_HASHES,
                      files_sha256={name: launcher.file_sha(folder / name) for name in ("state.pth", "train.jsonl")})
        if stage == "finish":
            report["evaluation"] = {"sample_count": 2497, "training_state_unchanged": True,
                                    "confusion_counts_exact": True, "source_miou": 0.1,
                                    "seen_ids_sha256": MANIFEST_HASHES["gta5_source_val.txt"]}
        (folder / "worker_report.json").write_text(json.dumps(report))
    result = launcher.verify_pilot(tmp_path, "a" * 40)
    assert result["optimizer_updates"] == 4 and result["selected_checkpoint"] == "finish/state.pth"
    path = tmp_path / "finish/state.pth"
    payload = load_training_checkpoint(path)
    payload["trainable_model"]["query_head.alpha"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="artifact hash"):
        launcher.verify_pilot(tmp_path, "a" * 40)
    report["files_sha256"]["state.pth"] = launcher.file_sha(path)
    (tmp_path / "finish/worker_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evaluated model"):
        launcher.verify_pilot(tmp_path, "a" * 40)
