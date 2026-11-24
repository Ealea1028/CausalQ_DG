import json

import numpy as np
import pytest
import torch

from causalq.analysis.lora_effect_moments import effect_rows, validate_reference
from causalq.analysis.lora_joint_pca import atomic_npz
from causalq.runtime.identity import file_sha
from causalq.runtime.lora_phase45_resume import check_record, compatible_key, migrate_sample
from causalq.runtime.lora_source_diagnostics import read_record, write_record
from tools.run_lora_phase45_source import verified_cache


@pytest.mark.parametrize('structural', [False, True])
def test_fp64_statistics_match_v1_at_tile_edge_and_zero_energy(structural):
    g = torch.Generator().manual_seed(7)
    full = torch.randn(5, 3, 19, 231, generator=g)
    full[:, 1] = 0
    full[:, 2] = 0.2  # Spatially constant but positive energy.
    full[1, 2] += 0.01
    anchor = None if structural else full * torch.linspace(0.25, 1.25, 5)[:, None, None, None]
    label = torch.zeros(1, 19, 231, dtype=torch.long)
    label[:, :, :8] = 255
    label[:, :, 50:89] = 2
    rows = effect_rows(full, anchor, label)
    assert validate_reference(full, anchor, label, rows) < 1e-10
    row = next(x for x in rows[0] if x['class_id'] == 1 and x['component'] == 'full')
    assert row['rms_normalized_variance'] is None
    assert row['raw_variance'] == 0


def test_old_record_import_keeps_bytes_and_rejects_changed_pca(tmp_path):
    source, dest = tmp_path/'old', tmp_path/'new'
    old = {'git_sha':'old','freeze_sha256':'a','source_fingerprint':'data','checkpoint_sha256':'weight'}
    new = dict(old, git_sha='new', freeze_sha256='b')
    assert compatible_key(old,new)
    assert not compatible_key(old,dict(new,checkpoint_sha256='different'))
    cache = source/'features/00001.npz'
    atomic_npz(cache,patch=np.arange(4))
    record = source/'records/00001.json.gz'
    write_record(record, {'id':'00001','key':old,'record':{'test':True},'pca_sha256':file_sha(cache)})
    assert migrate_sample(source,dest,'00001',old,new,{'00001'},verified_cache)
    new_record = dest/'records/00001.json.gz'
    assert file_sha(new_record) == file_sha(record)
    check_record(read_record(new_record),'00001',new,old)
    assert not migrate_sample(source,dest,'00002',old,new,{'00001'},verified_cache)
    cache.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='PCA cache hash mismatch'):
        migrate_sample(source,tmp_path/'invalid','00001',old,new,{'00001'},verified_cache)
    assert not (tmp_path/'invalid/records/00001.json.gz').exists()
