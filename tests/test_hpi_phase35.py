import json
from types import SimpleNamespace
import pytest
import torch

from causalq.analysis.selective_stability import effect_and_correlation_rows, structural_effect_and_correlation_rows
from causalq.datasets.segmentation import PairedSegmentationDataset, TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform
from causalq.interventions.style import StyleInterventionBank
from causalq.models.hpi_anchor_only import verify_frozen_residuals
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.runtime.cohort_runner import CohortRunner, IndependentHead, capture_rng, restore_head, head_state
from causalq.runtime.hpi_double_buffer import DoubleBuffer
from causalq.runtime.hpi_packets import PacketDataset
from causalq.runtime.hpi_phase35 import ablation_backward, build_ablation_head
from causalq.runtime.hpi_source_loader import SourceDiagnosticDataset, source_loader, source_loader_policy
from causalq.runtime.hpi_training import (
    export_coverage, require_final_source_heads, source_diagnostics, train_windows, verify_freeze,
)
from causalq.runtime.identity import tree_digest
from test_hpi_backbone import tiny_backbone
from test_hpi_runtime import samples, stream


def test_phase35_profile_and_joint_training_checkpoint_diagnostics(samples,tmp_path):
    freeze=verify_freeze(35)
    assert freeze['total_optimizer_updates']==120000
    profile={**load_profile(phase=35),'backbone_microbatch_views':8,'loader_packet_samples':2}
    backbone=tiny_backbone()

    def create(name):
        model=build_ablation_head(backbone,name,None,decoder_channels=32,num_classes=3)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        return IndependentHead(name,model,optimizer,capture_rng())

    heads=[create(name) for name in freeze['variants']]
    assert tree_digest(head_state(heads[1].model))==tree_digest(head_state(heads[2].model))
    identity={'git_sha':'original-training'}
    runner=CohortRunner(heads,horizon=7)
    windows=DoubleBuffer(backbone,profile,device='cpu',amp=False)(PacketDataset(stream(samples),2))
    train_windows(runner,windows,tmp_path,identity,amp=False,interval=3,update_fn=ablation_backward,batched_metrics=True)
    assert export_coverage(tmp_path,freeze['variants'],7)['total_optimizer_updates']==21
    for head in heads:
        restored=create(head.name)
        assert restore_head(restored,tmp_path/'checkpoints',identity)==7
        verify_frozen_residuals(restored.model)
        assert tree_digest(head_state(head.model))==tree_digest(head_state(restored.model))

    val=PairedSegmentationDataset(samples,transform=TrainTransform((32,32),scale_range=(1,1),horizontal_flip_probability=0))
    source=tmp_path/'source'
    source_diagnostics(heads,backbone,val,source,identity,device='cpu',amp=False,profile=profile,intervention_ids=['00000'])
    for name in freeze['variants']:
        metrics=json.loads((source/name/'source_metrics.json').read_text())
        assert metrics['residual_status']=='structurally_absent' and metrics['sample_count']==4
        summary=json.loads((source/name/'source_effect_summary.json').read_text())
        residual=[r for r in summary['image_macro_statistics'] if r.get('component')=='residual']
        assert residual and all(r['metrics']['raw_variance']['mean']==0 for r in residual)
    for name in ('B2','B3'):
        path=json.loads((source/name/'main_path128.json').read_text())
        assert set(path['by_image'])=={'00000'} and len(path['by_image']['00000'][0]['tau'])==5
        assert json.loads((source/name/'routing_summary.json').read_text())['image_count']==4

    # A prior release did not store a separate diagnostic producer.
    record = source/'records/00000.json'
    previous = json.loads(record.read_text())
    del previous['diagnostic_git_sha']
    record.write_text(json.dumps(previous))
    original_bytes = record.read_bytes()
    for path in (source/'records').glob('*.json'):
        if path != record:
            path.unlink()
    source_diagnostics(heads,backbone,val,source,identity,device='cpu',amp=False,profile=profile,
                       intervention_ids=['00000'],diagnostic_git_sha='diagnostic-patch')
    assert record.read_bytes() == original_bytes
    assert json.loads((source/'provenance.json').read_text()) == {
        'training_git_sha': 'original-training',
        'records_by_diagnostic_git_sha': {'original-training': 1, 'diagnostic-patch': 3},
    }


def test_source_loader_spawn_preserves_order_labels_and_view_traces(samples):
    profile = load_profile(phase=35)
    val = PairedSegmentationDataset(samples, transform=SourceEvalTransform(long_edge=64))
    dataset = SourceDiagnosticDataset(val, [0, 2, 3], StyleInterventionBank())
    expected = [dataset[i] for i in range(len(dataset))]
    loader, policy = source_loader(dataset, profile, workers=2, maximum_edge=64, shm_bytes=2<<30)
    assert policy['mode'] == 'spawn'
    assert policy['batch_size'] == 32
    actual = [sample for packet in loader for sample in packet]
    assert [item['id'] for item in actual] == ['00000', '00002', '00003']
    assert [tree_digest(item) for item in actual] == [tree_digest(item) for item in expected]
    policy = source_loader_policy(profile, 12, shm_bytes=55<<30)
    assert policy['mode'] == 'spawn' and policy['workers'] == 12
    assert 1 <= policy['batch_size'] < 32 and policy['host_bound_bytes'] <= 24<<30
    assert source_loader_policy(profile, 12, shm_bytes=0)['mode'] == 'threads'


def test_structural_reduction_matches_legacy_metrics():
    rng = torch.Generator().manual_seed(7)
    maps = torch.randn(5, 3, 12, 14, generator=rng)
    maps[2, 0] = 0
    maps[:, 1] = 0.25
    maps[:, 2] *= 1e-7
    labels = torch.randint(0, 2, (1, 12, 14), generator=rng)
    labels[:, :, -1] = 255
    views = [{'full': v, 'anchor': v, 'residual': torch.zeros_like(v)} for v in maps.split(1)]
    expected = effect_and_correlation_rows(views, labels)
    actual = structural_effect_and_correlation_rows(maps, labels)
    for actual_rows, expected_rows in zip(actual, expected, strict=True):
        for row, reference in zip(actual_rows, expected_rows, strict=True):
            assert row.keys() == reference.keys()
            for key, value in reference.items():
                if isinstance(value, (float, list)):
                    assert row[key] == pytest.approx(value, rel=1e-10, abs=1e-14)
                else:
                    assert row[key] == value


def test_phase35_source_resume_targets_original_run_and_requires_final_heads(tmp_path, monkeypatch):
    import sys
    from tools import run_hpi_dinov2_20261031 as coordinator

    diagnostic_sha = 'd' * 40
    monkeypatch.setenv('CAUSALQ_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['coordinator', diagnostic_sha, 'resume-phase35-source'])
    monkeypatch.setattr(coordinator, 'verify_checkout', lambda sha: None)
    launches = []
    monkeypatch.setattr(coordinator, 'launch', lambda *args: launches.append(args))
    assert coordinator.main() == 0
    assert launches == [(tmp_path/'phase35/B1_B3_c4a7b6f', diagnostic_sha, 'resume-phase35-source')]
    heads = [SimpleNamespace(cursor=40000) for _ in range(3)]
    require_final_source_heads(heads)
    heads[1].cursor = 39000
    with pytest.raises(ValueError, match='final 40000-update heads'):
        require_final_source_heads(heads)
