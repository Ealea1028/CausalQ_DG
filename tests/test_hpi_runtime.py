import copy
import random
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import pytest
import torch
from causalq.datasets.segmentation import SegmentationPair, TrainTransform
from causalq.protocols.hpi_dinov2_20261031 import build_head, load_config, load_profile
from causalq.runtime.input_stream import CanonicalInputStream, make_loader, input_seed, loader_policy
from causalq.runtime.feature_pipeline import FeaturePipeline, feature_tensors, exact_shape_batches
from causalq.runtime.cohort_runner import (CohortRunner, IndependentHead, capture_rng, head_state,
                                          commit_head, restore_head)
from causalq.runtime.identity import tree_digest
from test_hpi_backbone import tiny_backbone


@pytest.fixture
def samples(tmp_path):
    pairs=[]
    for i in range(4):
        image=tmp_path/f'image{i}.png';label=tmp_path/f'label{i}.png'
        Image.fromarray(np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3)).save(image)
        Image.fromarray(np.full((48,64),i%3,dtype=np.uint8)).save(label)
        pairs.append(SegmentationPair(image,label,f'{i:05d}'))
    return pairs


def stream(pairs,start=0,stop=7):
    return CanonicalInputStream(pairs,start=start,stop=stop,transform=TrainTransform((32,32)))


def test_explicit_generators_replay_workers_and_recovery(samples):
    torch.manual_seed(12)
    before=tree_digest(capture_rng())
    baseline=list(stream(samples))
    assert before==tree_digest(capture_rng())
    profile={**load_profile(),'pin_memory':False,'data_workers':2}
    for shm in (0,10**12):
        actual=list(make_loader(stream(samples),profile,shm_bytes=shm))
        assert [x['metadata']['id_parameters_sha256'] for x in actual]==[x['metadata']['id_parameters_sha256'] for x in baseline]
        assert all(torch.equal(x['views'],y['views']) for x,y in zip(actual,baseline))
    resumed=list(stream(samples,start=3))
    assert all(tree_digest(a)==tree_digest(b) for a,b in zip(resumed,baseline[3:]))
    assert input_seed(0,3,'001','photo')!=input_seed(0,4,'001','photo')
    assert before==tree_digest(capture_rng())


def make_heads(backbone,names=('U0','UA')):
    result=[]
    for name in names:
        model=build_head(backbone,decoder_channels=32,num_classes=3,dropout=.1)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        result.append(IndependentHead(name,model,optimizer,capture_rng(),anchor_cqe=name=='UA'))
    return result


def execute(runner,backbone,pairs,start=0,stop=7):
    profile={**load_profile(),'backbone_microbatch_views':4,'feature_queue_windows':2}
    windows=FeaturePipeline(backbone,profile,device='cpu',amp=False)(stream(pairs,start,stop))
    count=0
    for window in windows:
        before=tree_digest(feature_tensors(window.features))
        for sample,original,photo in window.logical():
            runner.update(sample,original,photo)
            count+=1
        assert before==tree_digest(feature_tensors(window.features))
    return count


def test_shared_vs_solo_updates_optimizer_rng_and_isolation(samples):
    torch.manual_seed(0)
    backbone=tiny_backbone()
    grouped=make_heads(backbone)
    solo=copy.deepcopy(grouped)
    assert execute(CohortRunner(grouped),backbone,samples)==7
    for head in solo:
        execute(CohortRunner([head]),backbone,samples)
    for a,b in zip(grouped,solo):
        assert tree_digest(head_state(a.model))==tree_digest(head_state(b.model))
        assert tree_digest(a.optimizer.state_dict())==tree_digest(b.optimizer.state_dict())
        assert tree_digest(a.rng)==tree_digest(b.rng)
    assert tree_digest(head_state(grouped[0].model))!=tree_digest(head_state(grouped[1].model))
    assert all(p.grad is None for p in backbone.parameters())
    with pytest.raises(ValueError,match='share trainable'):
        CohortRunner([grouped[0],grouped[0]])


def test_midwindow_different_head_cursors_and_partial_commit(samples,tmp_path):
    torch.manual_seed(5)
    backbone=tiny_backbone()
    initial=make_heads(backbone)
    expected=copy.deepcopy(initial)
    execute(CohortRunner(expected),backbone,samples)
    running=copy.deepcopy(initial)
    execute(CohortRunner([running[0]]),backbone,samples,stop=3)
    execute(CohortRunner([running[1]]),backbone,samples,stop=2)
    identity={'study':'test','backbone':'synthetic'}
    for head in running:
        commit_head(head,tmp_path,identity)
    partial=tmp_path/'U0'/'99999_partial';partial.mkdir();(partial/'state.pth').write_bytes(b'bad')
    resumed=copy.deepcopy(initial)
    assert [restore_head(h,tmp_path,identity) for h in resumed]==[3,2]
    # Full aligned window starts0; skip each committed cursor independently.
    execute(CohortRunner(resumed),backbone,samples)
    for a,b in zip(resumed,expected):
        assert a.cursor==b.cursor==7
        assert tree_digest(head_state(a.model))==tree_digest(head_state(b.model))
        assert tree_digest(a.optimizer.state_dict())==tree_digest(b.optimizer.state_dict())
        assert tree_digest(a.rng)==tree_digest(b.rng)
    with pytest.raises(ValueError,match='identity'):
        restore_head(resumed[0],tmp_path,{'wrong':True})


def test_exact_shape_diagnostics_preserve_ids_and_tail():
    items=[(str(i),torch.zeros(3,32,48 if i%2 else 32)) for i in range(11)]
    batches=list(exact_shape_batches(items,4))
    assert sorted(k for batch in batches for k,_ in batch)==sorted(k for k,_ in items)
    assert all(len({tuple(x.shape) for _,x in batch})==1 for batch in batches)


def test_ce_matches_historical_and_recipe_mapping():
    from causalq.losses.source_ce import deterministic_source_ce
    from tools.deterministic_source_ce import deterministic_source_ce as old
    values=torch.randn(1,19,4,4,requires_grad=True)
    label=torch.arange(16).reshape(1,4,4);label[0,0,0]=255
    assert torch.equal(deterministic_source_ce(values,label),old(values,label))
    u0,ua=load_config('U0'),load_config('UA')
    assert u0['model']==ua['model'] and u0['query']==ua['query']
    assert 'num_workers' not in u0['data']
    assert ua['causal_query_effect']['component']=='anchor'
    with pytest.raises(ValueError):load_config('B1')
    with pytest.raises(ValueError):load_profile('/tmp/wrong')
    with pytest.raises(ValueError):loader_policy({**load_profile(),'pinned_memory_budget_gib':.001})


def test_launcher_rejects_unreleased_actions_and_has_valid_bash():
    import shutil
    root=Path(__file__).resolve().parents[1]
    bash='D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').exists() else shutil.which('bash')
    if bash is None:pytest.skip('Bash unavailable')
    script=(root/'scripts/run_hpi_dinov2_20261031.sh').as_posix()
    assert subprocess.run([bash,'-n',script],capture_output=True).returncode==0
    result=subprocess.run([bash,'-c','source "$1" invalid prepare; echo parent_alive:$?', 'test',script],capture_output=True,text=True)
    assert 'parent_alive:2' in result.stdout
    result=subprocess.run([bash,script,'a'*40,'train-b4-b5'],capture_output=True)
    assert result.returncode==2
