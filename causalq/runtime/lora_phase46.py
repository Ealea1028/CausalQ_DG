"""Frozen target contract and audit helpers for the fifteen LoRA-study finals."""
import hashlib
import json
from pathlib import Path

from causalq.runtime.identity import file_sha
from causalq.runtime.lora_formal import DIRECTORY, ROOT
from causalq.runtime.lora_source_diagnostics import inventory

FREEZE = DIRECTORY/'phase46_freeze.json'
SOURCE = DIRECTORY/'phase45_completion_acceptance.json'
TARGET_IDS = ROOT/'manifests/cityscapes_val500.txt'
TARGET_SHA = '3960bb34afdfbbc6a5a5be887626d7ab03196691de54d0c00eb047153e03ab4c'
GEOMETRY = (1024, 2048)
BATCH_SIZE = 1


def load_contract():
    contract = json.loads(FREEZE.read_bytes())
    for name,digest in contract['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError(f'Phase46 dependency differs: {name}')
    source = json.loads(SOURCE.read_bytes())
    models = inventory()
    if (source['phase'] != 45 or source['accepted'] is not True or source['target_data_accessed'] is not False
            or source['optimizer_updates'] != 0 or source['images_per_model'] != 2497
            or source['pca_png_count'] != 172 or set(source['models']) != set(models)
            or source['model_count'] != 15 or file_sha(SOURCE) != contract['source_acceptance_sha256']
            or any(source['models'][n]['checkpoint_sha256'] != pin['sha256'] for n,pin in models.items())):
        raise ValueError('Phase45 fixed-model acceptance differs')
    ids = TARGET_IDS.read_text(encoding='ascii').splitlines()
    if (file_sha(TARGET_IDS) != TARGET_SHA or len(ids) != 500 or ids != sorted(set(ids))
            or contract['target_manifest_sha256'] != TARGET_SHA or contract['geometry'] != list(GEOMETRY)
            or contract['batch_size'] != BATCH_SIZE or contract['model_count'] != 15
            or contract['target_image_count'] != len(ids) or contract['target_pca'] is not False
            or contract['formal_optimizer_updates'] != 0
            or contract['checkpoint_sha256'] != {n:p['sha256'] for n,p in models.items()}):
        raise ValueError('Phase46 target protocol differs')
    return source,models,ids


def validate_record(model, sample_id, record, identity):
    key = record['key']
    if key['identity'] != identity or key['id'] != sample_id or set(record['confusion']) != {model}:
        raise ValueError(f'Record identity or model differs: {model}/{sample_id}')
    payloads = tuple(key[k] for k in ('image_sha256','label_sha256'))
    if any(len(x) != 64 or any(ch not in '0123456789abcdef' for ch in x) for x in payloads):
        raise ValueError(f'Image/label identity missing: {sample_id}')
    matrix = record['confusion'][model]
    if len(matrix) != 19 or any(len(row) != 19 or any(type(x) is not int or x < 0 for x in row) for row in matrix):
        raise ValueError(f'Invalid target confusion counts: {model}/{sample_id}')
    pixels = sum(map(sum,matrix))
    if not 0 <= pixels <= GEOMETRY[0]*GEOMETRY[1]:
        raise ValueError(f'Invalid target valid-pixel count: {model}/{sample_id}')
    return (*payloads,pixels)


def record_coverage(folder, model, identity, ids):
    records = folder/'records'
    expected = {records/f'{sample_id}.json' for sample_id in ids}
    if set(records.rglob('*.json')) != expected:
        raise ValueError(f'Incomplete or extra target records: {model}')
    index = {}
    for sample_id in ids:
        relative = Path(f'{sample_id}.json')
        row = json.loads((records/relative).read_bytes())
        index[sample_id] = validate_record(model,sample_id,row,identity)
    return index
