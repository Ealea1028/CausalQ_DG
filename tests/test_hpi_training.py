import copy
import json
import pytest
import torch
from causalq.runtime.hpi_training import verify_freeze, diagnostic_seeds, source_diagnostics, train_windows, export_coverage
from causalq.protocols.diagnostic_20261020 import style_seeds
from causalq.runtime.cohort_runner import CohortRunner, restore_head, head_state
from causalq.runtime.feature_pipeline import FeaturePipeline
from causalq.runtime.identity import tree_digest
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.datasets.segmentation import PairedSegmentationDataset, TrainTransform
from test_hpi_runtime import samples, stream, make_heads
from test_hpi_backbone import tiny_backbone


def test_phase34_freeze_and_legacy_seed_compatibility():
    freeze=verify_freeze()
    assert freeze['variants']==['U0','UA'] and freeze['total_optimizer_updates']==80000
    assert diagnostic_seeds('18734')==style_seeds('18734')


def test_formal_window_checkpoints_resume_each_cursor(samples,tmp_path):
    backbone=tiny_backbone();heads=make_heads(backbone)
    runner=CohortRunner(heads,horizon=7)
    profile={**load_profile(),'backbone_microbatch_views':4}
    train_windows(runner,FeaturePipeline(backbone,profile,device='cpu',amp=False)(stream(samples)),tmp_path,{'synthetic':True},amp=False,interval=3)
    assert [h.cursor for h in heads]==[7,7]
    fresh=make_heads(backbone)
    assert [restore_head(h,tmp_path/'checkpoints',{'synthetic':True}) for h in fresh]==[7,7]
    for a,b in zip(heads,fresh):assert tree_digest(head_state(a.model))==tree_digest(head_state(b.model))
    status=json.loads((tmp_path/'training_status.json').read_text())
    assert status['total_optimizer_updates']==14
    assert export_coverage(tmp_path,['U0','UA'],7)['total_optimizer_updates']==14


def test_coverage_merges_partial_attempts_and_rejects_unpaired_inputs(tmp_path):
    old=tmp_path/'attempts/old';old.mkdir(parents=True)
    def row(tick,variant):
        return {'cohort_tick':tick,'input':{'step_zero_based':tick-1,'id':str(tick)},'heads':{variant:{}}}
    (old/'training.jsonl').write_text(json.dumps(row(1,'U0'))+'\n'+json.dumps(row(1,'UA'))+'\n'+ '{partial')
    current=[row(1,'UA'),row(2,'U0'),row(2,'UA')]
    (tmp_path/'training.jsonl').write_text('\n'.join(map(json.dumps,current)))
    assert export_coverage(tmp_path,['U0','UA'],2)['paired_inputs']
    current[0]['input']['id']='changed'
    (tmp_path/'training.jsonl').write_text('\n'.join(map(json.dumps,current)))
    with pytest.raises(ValueError,match='stream changed'):
        export_coverage(tmp_path,['U0','UA'],2)


def test_joint_source_diagnostics_exact_coverage_and_readonly_resume(samples,tmp_path,monkeypatch):
    backbone=tiny_backbone();heads=make_heads(backbone)
    dataset=PairedSegmentationDataset(samples,transform=TrainTransform((32,32),scale_range=(1,1),horizontal_flip_probability=0))
    before=[tree_digest(head_state(h.model)) for h in heads]
    source_diagnostics(heads,backbone,dataset,tmp_path,{'synthetic':True},device='cpu',amp=False)
    for h,digest in zip(heads,before):
        assert tree_digest(head_state(h.model))==digest
        metrics=json.loads((tmp_path/h.name/'source_metrics.json').read_text())
        assert metrics['sample_count']==4 and len(set(metrics['source_ids']))==4
        assert sum(map(sum,metrics['confusion_matrix']))==4*32*32
        assert metrics['target_miou'] is None
        assert json.loads((tmp_path/h.name/'class_responsibility.json').read_text())['sample_count']==4
    monkeypatch.setattr(backbone,'forward',lambda *a:pytest.fail('Completed source image replayed'))
    source_diagnostics(heads,backbone,dataset,tmp_path,{'synthetic':True},device='cpu',amp=False)
