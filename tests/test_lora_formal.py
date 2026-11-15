import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch

from test_hpi_backbone import tiny_backbone
from test_lora_phase40 import packet
from tools.check_lora_resume import compare
from causalq.runtime.lora_training import FrozenFeatureBatch,make_optimizer,compact_state,update,save_checkpoint
from causalq.runtime.lora_formal import core_head,FormalCohort,VARIANTS
from causalq.models.lora import inject_dino_lora


def cohort(horizon=2):
    b=tiny_backbone()
    return FormalCohort(b,{name:core_head(b,name,decoder_channels=32,num_classes=3) for name in VARIANTS},horizon=horizon)


def test_shared_frozen_features_preserve_independent_head_updates(monkeypatch):
    c=cohort(); individual=copy.deepcopy(c.heads)
    opts={n:make_optimizer(m) for n,m in individual.items()}
    calls=[]; original=c.backbone.forward
    def count(x): calls.append(len(x)); return original(x)
    monkeypatch.setattr(c.backbone,'forward',count)
    samples=packet()
    c.tick(samples)
    assert calls==[4,4,4,4]
    for name,m in individual.items():
        update(m,opts[name],samples,0,microbatch=4,variant=name,prevalidated_source=True)
        assert compare(compact_state(m),compact_state(c.heads[name]))==0
        assert compare(opts[name].state_dict(),c.optimizers[name].state_dict())==0
    assert all(not p.requires_grad for p in c.backbone.parameters())
    assert c.heads['CA'].query_bank.centered_residuals and not c.heads['UA'].query_bank.centered_residuals


def test_skewed_head_recovery_skips_committed_updates(tmp_path):
    a=cohort(); initial=copy.deepcopy(a)
    ids={n:{'variant':n,'lora_enabled':False} for n in VARIANTS}
    a.tick(packet()); a.tick(packet(8))
    # Interrupted checkpoint cohort: only U0 committed at step1; others remain0.
    initial.tick(packet())
    save_checkpoint(tmp_path/'U0/00000001',initial.heads['U0'],initial.optimizers['U0'],1,ids['U0'])
    b=cohort()
    # Reconstruct the exact same synthetic frozen base rather than new random weights.
    b.backbone.load_state_dict(a.backbone.state_dict())
    assert b.restore(tmp_path,ids)==0
    b.tick(packet());b.tick(packet(8))
    assert b.cursors=={n:2 for n in VARIANTS}
    for name in VARIANTS:
        assert compare(compact_state(a.heads[name]),compact_state(b.heads[name]))==0
        assert compare(a.optimizers[name].state_dict(),b.optimizers[name].state_dict())==0
    records=b.commit(tmp_path,ids)
    assert all(r['step']==2 for r in records.values())
    with pytest.raises(ValueError): b.tick(packet(16))


def test_frozen_cache_rejects_mutation_and_adapter_use():
    c=cohort();samples=packet()
    cache=FrozenFeatureBatch.extract(c.backbone,samples)
    cache.verify(c.heads['U0'],samples,4)
    with torch.no_grad(): next(iter(cache.entries.values())).patch_tokens.add_(1)
    with pytest.raises(ValueError,match='mutated'):cache.verify(c.heads['U0'],samples,4)
    inject_dino_lora(c.backbone,blocks=(2,3))
    with pytest.raises(ValueError,match='unadapted'): FrozenFeatureBatch.extract(c.backbone,samples)


def test_emergency_commit_excludes_partially_updated_head(tmp_path,monkeypatch):
    import causalq.runtime.lora_formal as formal
    c=cohort(); real=formal.update
    def fail(model,*args,**kwargs):
        if kwargs['variant']=='UA':
            with torch.no_grad(): next(p for p in model.parameters() if p.requires_grad).add_(1)
            raise RuntimeError('Simulated interrupted optimizer')
        return real(model,*args,**kwargs)
    monkeypatch.setattr(formal,'update',fail)
    with pytest.raises(RuntimeError):c.tick(packet())
    ids={n:{'variant':n} for n in VARIANTS}
    records={}
    c.commit(tmp_path,ids,records)
    assert c.tainted_head=='UA' and c.cursors==dict(U0=1,UA=0,C0=0,CA=0)
    assert set(records)=={'U0'} and records['U0']['step']==1
    assert not (tmp_path/'UA').exists()


def test_core_cohort_resume_preserves_complete_rng(tmp_path):
    from causalq.runtime.lora_training import seed_training_rng,rng_digest
    from causalq.runtime.cohort_runner import capture_rng
    ids={n:{'variant':n} for n in VARIANTS}
    seed_training_rng(0); a=cohort(); a.tick(packet()); a.tick(packet(8))
    expected=rng_digest(capture_rng())
    seed_training_rng(0); b=cohort(); b.tick(packet()); b.commit(tmp_path,ids)
    seed_training_rng(0); c=cohort(); assert c.restore(tmp_path,ids)==1
    c.tick(packet(8))
    assert rng_digest(capture_rng())==expected
    assert all(compare(compact_state(c.heads[n]),compact_state(a.heads[n]))==0 for n in VARIANTS)


def test_freeze_accepts_checkout_line_endings_but_rejects_changed_contract(tmp_path,monkeypatch):
    import causalq.runtime.lora_formal as formal
    d=tmp_path/'experiments';d.mkdir()
    config=tmp_path/'configs/lora_causalq_1114/phase41_core_l0.json';config.parent.mkdir(parents=True)
    payload={'variants':list(VARIANTS),'formal_steps':40000,'effective_source_batch':8,'lora_enabled':False,'target_access_authorized':False}
    raw=(json.dumps(payload)+'\n').encode();config.write_bytes(raw.replace(b'\n',b'\r\n'))
    evidence=b'{"accepted":true}\n';(d/'phase40_resume_acceptance.json').write_bytes(evidence)
    freeze={'file_hash_scheme':'sha256_lf_text_v1','files':{str(config.relative_to(tmp_path)):hashlib.sha256(raw).hexdigest()},
        'resume_acceptance_git_blob_sha256':hashlib.sha256(evidence).hexdigest()}
    (d/'phase41_freeze.json').write_text(json.dumps(freeze))
    monkeypatch.setattr(formal,'ROOT',tmp_path);monkeypatch.setattr(formal,'DIRECTORY',d)
    assert formal.load_contract()[0]==payload
    config.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='changed'):formal.load_contract()
