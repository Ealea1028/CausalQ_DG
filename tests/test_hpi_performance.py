import io
import json
import pytest
import torch
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.runtime.cohort_runner import CohortRunner, head_state
from causalq.runtime.feature_pipeline import FeaturePipeline
from causalq.runtime.hpi_double_buffer import DoubleBuffer
from causalq.runtime.hpi_packets import PacketDataset, packet_loader, packet_policy
from causalq.runtime.hpi_performance import activity, choose, flush_metrics, settings, update_budget
from causalq.runtime.hpi_source_batch import source_batches, batched_source_readout
from causalq.runtime.identity import tree_digest
from test_hpi_runtime import samples, stream, make_heads
from test_hpi_backbone import tiny_backbone


def profile():
    return {**load_profile(),'loader_packet_samples':2,'backbone_microbatch_views':8,'data_workers':2,'pin_memory':False}


def test_packet_spawn_and_thread_fallback_keep_steps_and_tail(samples):
    canonical=stream(samples,start=1,stop=8)
    expected=[s['metadata'] for s in canonical]
    for shm,mode in ((0,'threads'),(1<<40,'spawn')):
        loader,policy=packet_loader(canonical,profile(),shm_bytes=shm)
        packets=list(loader)
        assert policy['mode']==mode
        assert [len(p['metadata']) for p in packets]==[2,2,2,1]
        assert [m for p in packets for m in p['metadata']]==expected
        assert torch.equal(torch.cat([p['views'] for p in packets]),torch.stack([s['views'] for s in canonical]))
    with pytest.raises(ValueError,match='budget'):
        packet_policy({**profile(),'pin_memory':True,'pinned_memory_budget_gib':.001})


def test_double_buffer_preserves_head_updates_and_releases_slots(samples):
    torch.manual_seed(19);backbone=tiny_backbone();canonical=stream(samples)
    old=make_heads(backbone);new=make_heads(backbone)
    paths=[FeaturePipeline(backbone,profile(),device='cpu',amp=False)(canonical),
        DoubleBuffer(backbone,profile(),device='cpu',amp=False)(PacketDataset(canonical,2))]
    for heads,windows in zip((old,new),paths):
        runner=CohortRunner(heads)
        for window in windows:
            for sample,original,photo in window.logical():runner.update(sample,original,photo)
            del sample,original,photo
    assert [h.cursor for h in new]==[7,7]
    assert [tree_digest(head_state(h.model)) for h in old]==[tree_digest(head_state(h.model)) for h in new]
    assert [tree_digest(h.optimizer.state_dict()) for h in old]==[tree_digest(h.optimizer.state_dict()) for h in new]
    assert [tree_digest(h.rng) for h in old]==[tree_digest(h.rng) for h in new]
    buffer=DoubleBuffer(backbone,profile(),device='cpu',amp=False)
    windows=buffer(PacketDataset(stream(samples,stop=13),2));first=next(windows)
    assert buffer.peak_live_slots==2
    second=next(windows);assert first.features is None
    windows.close();assert second.features is None


def test_bounded_budget_and_batch_metric_flush(tmp_path):
    options=settings()
    assert update_budget([{'backbone_microbatch_views':n} for n in (32,32,64)],options)==384
    with pytest.raises(ValueError):update_budget([{'backbone_microbatch_views':512}],options)
    a={'workers':12,'memory_bytes':200,'rate':100.};b={'workers':8,'memory_bytes':100,'rate':96.}
    assert choose([a,b],'rate',.05)==b
    item={'losses':torch.tensor([1.,2.,3.]),'gradient_norm':torch.tensor(4.),'optimizer_updates':1,'timing_events':()}
    log=io.StringIO();record=[{'heads':{'U0':item}}]
    assert flush_metrics(record,batched=True,stream=log)==flush_metrics(record,batched=False)
    assert json.loads(log.getvalue())['losses_norm']==[1.,2.,3.,4.]
    csv=tmp_path/'gpu.csv';csv.write_text('elapsed_seconds,utilization_percent,status\n0,99,ok\n5,40,ok\n10,80,ok\n')
    summary=activity(csv,4,12,120)
    assert summary['median_percent']==60 and not summary['steady_eligible']


def test_source_geometry_buckets_and_one_original_confusion():
    backbone=tiny_backbone();heads=make_heads(backbone)
    for h in heads:h.model.eval()
    cases=[(str(i),torch.randn(5,3,32,w),torch.zeros(1,32,w,dtype=torch.long)) for i,w in enumerate((32,48,32))]
    batches=list(source_batches(cases,max_images=2,max_buckets=1))
    assert {c[0] for batch in batches for c in batch}=={'0','1','2'}
    assert all(len({c[1].shape for c in batch})==1 for batch in batches)
    result=batched_source_readout(heads,backbone,[cases[0],cases[2]],device='cpu',amp=False)
    assert set(result)=={'0','2'}
    for item in result.values():
        for head in item.values():assert sum(map(sum,head['confusion']))==32*32
