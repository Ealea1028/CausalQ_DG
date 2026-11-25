import hashlib

import pytest

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_source_diagnostics import inventory
from tools.accept_lora_phase46_target import audit_target_model, contrasts
from tools.verify_lora_checkpoint_preview import PREVIEW, PREVIEW_SHA, load_preview


def test_target_audit_recomputes_dataset_miou_and_rejects_changed_hash(tmp_path):
    name = 'D3_U0_L1'
    ids = ['city/a', 'city/b']
    identity = {'checkpoint_sha256':'c'*64, 'geometry':[1024, 2048]}
    index = {}
    folder = tmp_path/'models'/name
    for i, sample in enumerate(ids):
        matrix = [[0]*19 for _ in range(19)]
        matrix[i][i] = 2
        index[sample] = ['a'*64, 'b'*64, 2]
        write_json(folder/'records'/f'{sample}.json', {
            'key':{'identity':identity, 'id':sample, 'image_sha256':'a'*64, 'label_sha256':'b'*64},
            'confusion':{name:matrix},
        })
    matrix = [[0]*19 for _ in range(19)]
    matrix[0][0] = matrix[1][1] = 2
    write_json(folder/'target_metrics.json', {
        'ids':ids, 'geometry':identity['geometry'], 'identity':identity, 'sample_count':2,
        'confusion_matrix':matrix, 'valid_pixels':4, 'class_iou':[1., 1.]+[None]*17,
        'valid_classes':2, 'state_unchanged':True, 'rng_unchanged':True, 'source_miou':0.7,
        'records_reused':0, 'new_native_image_forwards':2, 'miou':1., 'miou_percent':100.,
        'seconds_this_invocation':1., 'data_workers':1, 'parameter_count':200000,
        'parameters_in_lora':98304, 'peak_allocated_bytes':10, 'peak_reserved_bytes':20,
    })
    (folder/'telemetry_test.csv').write_text('utilization_percent,status\n80,ok\n')
    receipt = {'sample_count':2, 'metrics_sha256':file_sha(folder/'target_metrics.json'),
               'checkpoint_sha256':'c'*64, 'miou':1.}
    table = {'checkpoint_sha256':'c'*64, 'source_miou':'0.7', 'target_miou':'1.0',
             'target_miou_percent':'100.0', 'target_images':'2'}
    assert audit_target_model(tmp_path, name, ids, identity, index, receipt, .7, table)['target_miou'] == 1
    index['city/b'][0] = 'e'*64
    with pytest.raises(ValueError, match='Target input differs'):
        audit_target_model(tmp_path, name, ids, identity, index, receipt, .7, table)
    (folder/'records/city/b.json').unlink()
    with pytest.raises(ValueError, match='Target record coverage differs'):
        audit_target_model(tmp_path, name, ids, identity, index, receipt, .7, table)


def test_missing_control_and_structural_residual_remain_undefined():
    model = {'effect_scope':'readout_effect', 'source_miou_percent':77, 'residual_status':'defined',
             'residual_raw_variance':0, 'residual_rms_normalized_variance':None}
    results = {r['contrast']:r for r in contrasts({
        'D3_U0_L1':model,
        'D3_U0_L0':dict(model, source_miou_percent=76, residual_raw_variance=1.),
    })}
    paired = results['D3 U0 L1−L0']
    assert paired['source_miou_percent'] == 1
    assert paired['residual_raw_variance'] == -1
    assert paired['residual_rms_normalized_variance'] is None
    assert results['D3 UA L1−L0']['missing_models'] == 'D3_UA_L1,D3_UA_L0'


def test_disk_preview_keeps_final_and_migration_steps(tmp_path):
    pins = inventory()
    rows = load_preview(PREVIEW, PREVIEW_SHA, pins)
    assert len(rows) == 568
    assert sum(r['allocated_bytes'] for r in rows.values() if r['decision'] == 'CANDIDATE') == 26897846272
    data = PREVIEW.read_bytes().replace(b'KEEP\tD3_UA_L1\t40000', b'CANDIDATE\tD3_UA_L1\t40000')
    data = data.replace(b'CANDIDATE\tD3_UA_L1\t38000', b'KEEP\tD3_UA_L1\t38000')
    changed = tmp_path/'changed.tsv'
    changed.write_bytes(data)
    with pytest.raises(ValueError, match='Reviewed preview SHA differs'):
        load_preview(changed, PREVIEW_SHA, pins)
    with pytest.raises(ValueError, match='Protected checkpoint set differs'):
        load_preview(changed, hashlib.sha256(data).hexdigest(), pins)
