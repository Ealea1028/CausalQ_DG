import json

import numpy as np
import pytest
import torch

from causalq.analysis import lora_joint_pca as pca
from causalq.runtime.lora_source_diagnostics import inventory, readout, summarize, write_record
from causalq.runtime.lora_phase44_b2 import b2_head
from test_hpi_backbone import tiny_backbone
from tools.run_lora_phase45_source import capture_pca, verified_cache


def test_phase45_inventory_excludes_retired_checkpoints():
    models = inventory()
    assert len(models) == 15
    assert [k for k,v in models.items() if not v['lora']] == ['D3_U0_L0']
    assert sum(v['variant'] == 'B3' for v in models.values()) == 2
    assert all(p['step'] == 40000 for p in models.values())


def test_same_feature_pass_serves_diagnostics_router_delete_and_pca(tmp_path):
    model = b2_head(tiny_backbone(), decoder_channels=32, num_classes=19, blocks=(2,3)).eval()
    views = torch.randn(5,3,32,32)
    label = torch.ones(1,32,32,dtype=torch.long)
    label[:,:,16:] = 255
    calls, cache = [], {}
    model.backbone.register_forward_hook(lambda *_: calls.append(True))
    def observe(features, prediction, batch):
        cache['pca_sha256'] = capture_pca(tmp_path,'00001',features,0,views,label,True,prediction.logits)
    cases = [('00001',views,label)]
    outputs = readout(model,cases,device='cpu',amp=False,intervention_ids={'00001'},observer=observe)
    assert len(calls) == 1
    r = outputs['00001']
    assert len(r['main_path']) == 1 and len(r['main_path'][0]['tau']) == 5
    assert r['main_path'][0]['class_id'] == 1
    residual = next(x for x in r['effects'] if x['component']=='residual' and x['class_id']==1)
    assert residual['raw_variance'] == 0 and residual['rms_normalized_variance'] is None
    slots = [x for x in r['responsibility'] if x['class_id']==1 and x['scope']=='gt_class']
    assert all(x['mean_responsibility']==.5 and x['top1_share']==.5 and x['five_view_population_variance']==0 for x in slots)
    write_record(tmp_path/'records/00001.json.gz', {'record':r})
    metrics = summarize(tmp_path,['00001'],{'00001'},True)
    assert metrics['valid_pixels'] == 512 and metrics['sample_count'] == 1
    assert json.loads((tmp_path/'main_path128.json').read_bytes())['images'] == 1
    verified_cache(tmp_path,'00001',cache,{'00001'})
    with np.load(tmp_path/'features/00001.npz') as data:
        assert data['patch'].shape[0] == data['cls'].shape[0] == 5
        assert data['dense'].shape[0] == 2 and data['prediction'].shape == (2,32,32)
        vectors = data['patch'].reshape(-1,data['patch'].shape[-1])
        mean,basis,_,ranges,_ = pca.fit_joint([vectors,vectors+0.01])
        pca.draw_panel(tmp_path/'panel.png','00001','D2 B2',data,data,data['images'],data['label'],
                       {'adapted':data['prediction']},mean,basis,ranges)
        assert (tmp_path/'panel.png').is_file()
    with pytest.raises(ValueError): verified_cache(tmp_path,'00001',{'pca_sha256':'wrong'},{'00001'})


def test_joint_basis_shared_projection_sign_and_coverage(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20,5)).astype('float32')
    b = a + np.array([0,0,2,0,0],dtype='float32')
    mean, basis, projections, ranges, ratio = pca.fit_joint([a,b])
    np.testing.assert_allclose(mean,np.concatenate([a,b]).mean(0),atol=1e-7)
    np.testing.assert_allclose(basis.T @ basis,np.eye(3),atol=1e-12)
    assert (basis[np.abs(basis).argmax(0),np.arange(3)] >= 0).all()
    np.testing.assert_allclose(projections[1], pca.project(b,mean,basis))
    assert (ratio >= 0).all() and np.sum(ratio) <= 1
    np.testing.assert_array_equal(pca.rgb(projections[0],ranges),pca.rgb(pca.project(a,mean,basis),ranges))
    with pytest.raises(ValueError): pca.fit_joint([a,b[:-1]])
    pca.draw_scatter(tmp_path/'scatter.png','shared axes',projections,['Original','LoRA'],np.arange(20)%5,ranges)
    assert (tmp_path/'scatter.png').is_file()


def test_export_keeps_basis_and_summaries_but_excludes_dense_caches(tmp_path, monkeypatch):
    import tarfile
    import tools.run_lora_phase45_source as runner
    from causalq.runtime.identity import file_sha
    monkeypatch.setattr(runner,'BASE',tmp_path)
    out=tmp_path/'run'; out.mkdir()
    (out/'report.json').write_text('{}')
    pca.atomic_npz(out/'pca/basis.npz',mean=np.zeros(3))
    pca.atomic_npz(out/'models/D2_U0_L1/features/dense.npz',dense=np.ones((2,3)))
    for name in ('first','second'):
        archive=tmp_path/f'{name}_return.tar.gz'
        runner.export(out,archive)
        with tarfile.open(archive) as bundle:
            assert set(bundle.getnames())=={'report.json','pca/basis.npz','export_manifest.json'}
            manifest=json.load(bundle.extractfile('export_manifest.json'))
            assert manifest['pca/basis.npz']['sha256']==file_sha(out/'pca/basis.npz')
