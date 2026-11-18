import json
import pytest
from tools.check_lora_vgpu48_resume import check_runtime, verify_donor
from causalq.runtime.identity import file_sha


def test_new_hardware_is_explicitly_allowed_but_critical_software_stays_fixed():
    source={'python':'3.12','torch':'2.7','cuda':'12.8','transformers':'4.56','gpu':'old84','visible_memory_bytes':84,
            'pip_freeze':['torch==2.7']}
    actual=dict(source,gpu='audited48',visible_memory_bytes=48,platform='new-host')
    accepted={'target_gpu':{'name':'audited48','memory_bytes':48}}
    check_runtime(actual,source,accepted)
    for key,value in [('gpu','other'),('visible_memory_bytes',24),('torch','2.8')]:
        with pytest.raises(ValueError): check_runtime(dict(actual,**{key:value}),source,accepted)


def test_cloned_payload_and_marker_are_read_only_and_both_checked(tmp_path):
    folder=tmp_path/'CA/checkpoints/00000751';folder.mkdir(parents=True)
    payload=folder/'state.pt';payload.write_bytes(b'synthetic')
    identity={'source':'synthetic'}
    marker={'step':751,'identity':identity,'sha256':file_sha(payload)}
    path=folder/'complete.json';path.write_text(json.dumps(marker))
    row={'step':751,'identity':identity,'expected_sha256':marker['sha256'],'bytes':payload.stat().st_size}
    before={p:p.read_bytes() for p in (path,payload)}
    assert verify_donor(tmp_path,'CA',row)==folder
    assert before=={p:p.read_bytes() for p in before}
    payload.write_bytes(b'corrupted')
    with pytest.raises(ValueError): verify_donor(tmp_path,'CA',row)


def test_changed_marker_identity_is_rejected_even_with_valid_payload(tmp_path):
    folder=tmp_path/'UA/checkpoints/00000753';folder.mkdir(parents=True)
    payload=folder/'state.pt';payload.write_bytes(b'synthetic')
    row={'step':753,'identity':{'variant':'UA'},'expected_sha256':file_sha(payload),'bytes':9}
    (folder/'complete.json').write_text(json.dumps({'step':753,'identity':{'variant':'CA'},'sha256':file_sha(payload)}))
    with pytest.raises(ValueError): verify_donor(tmp_path,'UA',row)
