import hashlib
import json
from pathlib import Path
import zipfile
import pytest
import torch
from causalq.models.dinov3_wrapper import DINOv3Features
from causalq.protocols.hpi_dinov2_20261031 import load_profile
from causalq.runtime.hpi_prepare import feature_error, telemetry_summary
from tools.run_hpi_dinov2_20261031 import package


def test_feature_numerical_gate_checks_all_raw_maps_and_final_tokens():
    x=torch.ones(1,4,2,2)
    def features(value):
        return DINOv3Features(value,value.flatten(2).transpose(1,2),(value,value*2))
    assert all(r['max_abs']==0 for r in feature_error(features(x),features(x.clone()),load_profile()))
    with pytest.raises(ValueError,match='equivalence'):
        feature_error(features(x),features(x+1),load_profile())
    with pytest.raises(ValueError,match='shape or values'):
        feature_error(features(x),features(x*float('nan')),load_profile())


def test_package_hashes_and_telemetry_are_read_only_and_exclude_weights(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'complete'}))
    (tmp_path/'prepare_report.json').write_text(json.dumps({'optimizer_steps':0}))
    (tmp_path/'state.pth').write_bytes(b'not exportable')
    (tmp_path/'gpu_telemetry.csv').write_text('elapsed_seconds,utilization_percent,memory_used_mib,power_w,status\n0,50,100,70,ok\n2,80,120,90,ok\n')
    summary=telemetry_summary(tmp_path/'gpu_telemetry.csv')
    assert summary['utilization_percent']['mean']==65
    package(tmp_path)
    with zipfile.ZipFile(tmp_path/'export.zip') as archive:
        assert 'state.pth' not in archive.namelist()
        for line in archive.read('SHA256SUMS').decode().splitlines():
            digest,name=line.split('  ')
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    (tmp_path/'status.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(RuntimeError,match='finished or failed'):
        package(tmp_path)
