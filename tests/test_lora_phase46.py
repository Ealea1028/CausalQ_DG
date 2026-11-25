import json
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from causalq.runtime.hpi_target import NativeTargetDataset, evaluate_batches
from causalq.runtime.identity import file_sha
from causalq.runtime.lora_phase46 import BATCH_SIZE, load_contract, record_coverage
from tools.accept_lora_phase45_source import confusion_miou
from tools import run_lora_phase46_target as runner


def test_fixed_target_contract_has_all_accepted_finals():
    source, pins, ids = load_contract()
    assert source['accepted'] and len(pins) == 15 and len(ids) == 500
    assert sum(pin['lora'] for pin in pins.values()) == 14
    assert pins['D3_U0_L0']['lora'] is False
    assert all(p['checkpoint_sha256'] == pins[n]['sha256'] for n,p in source['models'].items())


def test_native_readout_resumes_without_recounting_and_checks_payload_identity(tmp_path):
    city = tmp_path/'cityscapes'
    size = (16,32)
    sample = 'city/city_000001_000002'
    image = city/'leftImg8bit/val'/f'{sample}_leftImg8bit.png'
    label = city/'gtFine/val'/f'{sample}_gtFine_labelIds.png'
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((*size,3),np.uint8)).save(image)
    raw = np.full(size,7,np.uint8)  # Official road label -> train ID 0.
    raw[0] = 0                   # Void -> ignore.
    Image.fromarray(raw).save(label)
    dataset = NativeTargetDataset(city,[sample],size)
    calls = []

    def backbone(batch):
        calls.append(tuple(batch.shape))
        return batch

    def forward(features, geometry):
        assert tuple(geometry) == size
        return SimpleNamespace(logits=features.new_zeros(1,19,*size))  # argmax -> road.

    out = tmp_path/'out'
    identity = {'checkpoint_sha256':'a'*64}
    for _ in range(2):
        result, covered = evaluate_batches({'model':SimpleNamespace(forward_features=forward)},backbone,
            DataLoader(dataset,batch_size=BATCH_SIZE),out,identity,device='cpu',amp=False)
        assert covered == [sample] and int(result['model'].confusion[0,0]) == 480
        score, classes, pixels = confusion_miou(result['model'].confusion.tolist())
        assert score == 1 and classes == [1]+[None]*18 and pixels == 480
    assert calls == [(1,3,16,32)]
    seen = record_coverage(out,'model',identity,[sample])
    assert seen[sample] == (file_sha(image),file_sha(label),480)
    record = out/'records'/f'{sample}.json'
    data = json.loads(record.read_bytes())
    data['key']['identity']['checkpoint_sha256'] = 'b'*64
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='identity'):
        record_coverage(out,'model',identity,[sample])


def test_target_packaging_keeps_paired_confusion_records_only(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'BASE',tmp_path)
    out = tmp_path/'run'
    record = out/'models/U0/records/city/id.json'
    record.parent.mkdir(parents=True)
    record.write_text('{}')
    (out/'prediction.png').write_bytes(b'not exported')
    for i in range(2):
        archive = tmp_path/f'target_{i}_return.tar.gz'
        runner.export(out,archive)
        with tarfile.open(archive) as f:
            assert set(f.getnames()) == {'export_manifest.json','models/U0/records/city/id.json'}
            manifest = json.load(f.extractfile('export_manifest.json'))
            assert manifest['models/U0/records/city/id.json']['sha256'] == file_sha(record)
