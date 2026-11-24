"""Import committed v1 Phase45 records with their original producer identity."""
import json
import os
from pathlib import Path
import shutil

from causalq.runtime.lora_source_diagnostics import CONTRACT, read_record

INTERRUPTION = CONTRACT.parent/'phase45_interrupted_1d28c61.json'


def old_source(base):
    evidence = json.loads(INTERRUPTION.read_bytes())
    path = base/f"phase45_source_{evidence['git_sha'][:7]}"
    return path, evidence['source_protocol']


def compatible_key(previous, current):
    # The new commit changes only the dense effect reduction. Keep the original
    # execution geometry, checkpoint/weight hashes, decoded input and runtime.
    ignore = {'git_sha','freeze_sha256'}
    return {k:v for k,v in previous.items() if k not in ignore} == {k:v for k,v in current.items() if k not in ignore}


def check_record(row, sample_id, current, previous=None):
    if row['id'] != sample_id or (row['key'] != current and row['key'] != previous):
        raise ValueError(f'Phase45 record key differs: {sample_id}')


def copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_suffix('.partial')
    shutil.copyfile(source, stage)
    os.replace(stage, destination)


def migrate_sample(source, destination, sample_id, previous, current, selected, cache_check):
    """Commit the cache first, then the original record as the commit marker."""
    from tools.retire_lora_l0 import safe
    src_record = safe(source,source/'records'/f'{sample_id}.json.gz')
    if not src_record.exists(): return False
    row = read_record(src_record)
    check_record(row,sample_id,previous)
    cache_check(source,sample_id,row,selected)
    if sample_id in selected:
        feature = safe(source,source/'features'/f'{sample_id}.npz')
        copy_atomic(feature,destination/'features'/feature.name)
        cache_check(destination,sample_id,row,selected)
    copy_atomic(src_record,destination/'records'/src_record.name)
    check_record(read_record(destination/'records'/src_record.name),sample_id,current,previous)
    return True
