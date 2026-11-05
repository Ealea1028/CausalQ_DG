import json
import torch

from causalq.datasets.segmentation import PairedSegmentationDataset, TrainTransform
from causalq.models.hpi_anchor_only import verify_frozen_residuals
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.runtime.cohort_runner import CohortRunner, IndependentHead, capture_rng, restore_head, head_state
from causalq.runtime.hpi_double_buffer import DoubleBuffer
from causalq.runtime.hpi_packets import PacketDataset
from causalq.runtime.hpi_phase35 import ablation_backward, build_ablation_head
from causalq.runtime.hpi_training import export_coverage, source_diagnostics, train_windows, verify_freeze
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
    identity={'synthetic_phase35':True}
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
