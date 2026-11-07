import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from causalq.runtime.hpi_target import NativeTargetDataset, evaluate_batches, load_final_head, target_contract
from causalq.runtime.identity import file_sha, tree_digest
from causalq.runtime.cohort_runner import head_state
from causalq.protocols.hpi_dinov2_20261031 import build_head
from test_hpi_backbone import tiny_backbone


def test_contract_has_seven_finals_and_config_hashes():
    contract = target_contract()
    assert contract['formal_optimizer_updates'] == 0
    assert contract['batch_size'] == 2 and contract['geometry'] == [1024,2048]
    assert len(contract['models']) == 7


def test_native_reader_converts_label_ids_and_rejects_geometry(tmp_path):
    image_root = tmp_path/'leftImg8bit/val/city'
    label_root = tmp_path/'gtFine/val/city'
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    Image.fromarray(np.zeros((4,8,3),dtype=np.uint8)).save(image_root/'sample_leftImg8bit.png')
    Image.fromarray(np.full((4,8),7,dtype=np.uint8)).save(label_root/'sample_gtFine_labelIds.png')
    dataset = NativeTargetDataset(tmp_path,['city/sample'],geometry=(4,8))
    sample = dataset[0]
    assert sample['image'].shape == (3,4,8)
    assert torch.count_nonzero(sample['label']) == 0
    with pytest.raises(ValueError, match='native'):
        NativeTargetDataset(tmp_path,['city/sample'])[0]
    with pytest.raises(ValueError, match='membership'):
        NativeTargetDataset(tmp_path,[])


def test_batched_heads_tail_and_readonly_resume(tmp_path):
    class Backbone(torch.nn.Module):
        def forward(self, images):
            return images

    class Head(torch.nn.Module):
        def __init__(self, category):
            super().__init__()
            self.category = category

        def forward_features(self, features, size):
            logits = torch.zeros(len(features),19,*size)
            logits[:,self.category] = 1
            return SimpleNamespace(logits=logits)

    samples = [{'id':str(i),'image':torch.zeros(3,4,8),'label':torch.zeros(4,8,dtype=torch.long),
                'image_sha256':str(i),'label_sha256':'label'} for i in range(3)]
    loader = DataLoader(samples,batch_size=2)
    backbone = Backbone()
    heads = {'A':Head(0),'B':Head(1)}
    metrics, ids = evaluate_batches(heads,backbone,loader,tmp_path,{'sha':'fixed'},device='cpu',amp=False)
    assert ids == ['0','1','2']
    assert metrics['A'].confusion[0,0] == 96
    assert metrics['B'].confusion[0,1] == 96
    original = {p:p.read_bytes() for p in (tmp_path/'records').glob('*.json')}
    backbone.forward = lambda _:pytest.fail('Replayed completed target')
    evaluate_batches(heads,backbone,loader,tmp_path,{'sha':'fixed'},device='cpu',amp=False)
    assert all(p.read_bytes()==content for p,content in original.items())
    with pytest.raises(ValueError, match='identity'):
        evaluate_batches(heads,backbone,loader,tmp_path,{'sha':'other'},device='cpu',amp=False)


def test_final_head_identity_and_complete_tensor_coverage(tmp_path):
    model = build_head(tiny_backbone(),decoder_channels=32)
    identity = {'git_sha':'training'}
    path = tmp_path/'state.pth'
    torch.save({'identity':identity,'cursor':40000,'model':head_state(model)},path)
    record = {'checkpoint_sha256':file_sha(path),'identity':identity,'head_sha256':tree_digest(head_state(model))}
    load_final_head(model,path,record)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    record['identity'] = {'git_sha':'other'}
    with pytest.raises(ValueError, match='identity'):
        load_final_head(model,path,record)
