"""CPU-only validation of returned Phase39 bytes and engineering evidence."""
import argparse
import json
import math
from pathlib import Path
import tarfile

from causalq.runtime.identity import file_sha, write_json

PRODUCER = '68992d869509cbcc5ab4b824fcc3fe67f35f43f4'
WEIGHTS = {'dinov3_vitl16': ('model.safetensors', 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'),
           'dinov2_vitl14_hpi_p16': ('dinov2_vitl14_hpi_p16.pth', '054d08a7c672f07ee290f5202290d9807b3ae026e201263edbab284aad96d6af')}


def validate(report):
    if not (report['ok'] is True and report['exit_code'] == 0 and report['git_sha'] == PRODUCER
            and report['study'] == 'lora_causalq_1114' and report['phase'] == 39
            and report['action'] == 'backbone-gate' and report['formal_optimizer_updates'] == 0
            and report['dataset_accessed'] is False and report['target_data_accessed'] is False):
        raise ValueError('Phase39 scope/status/producer mismatch')
    models = report['models']
    if len(models) != 2 or {m['model'] for m in models} != set(WEIGHTS):
        raise ValueError('Expected exactly both unique backbones')
    for m in models:
        key, digest = WEIGHTS[m['model']]
        if m['weight_files'][key] != digest:
            raise ValueError('Historical weight identity mismatch')
        lora = m['lora']
        expected = [{'path':f'model.layer.{i}.attention.{p}_proj', 'components':[p]}
                    for i in range(18,24) for p in ('q','v')] if m['model']=='dinov3_vitl16' else [
                        {'path':f'model.blocks.{i}.attn.qkv','components':['q','v']} for i in range(18,24)]
        if (lora['blocks_zero_based'] != list(range(18,24)) or lora['rank'] != 4 or lora['alpha_lora'] != 4
                or lora['trainable_parameters'] != 98304 or lora['targets'] != expected):
            raise ValueError('LoRA target/layout mismatch')
        gradients = m['final_adapter_gradient_norms']
        keys = {target['path']+f'.{component}.{matrix}' for target in expected
                for component in (['q','v'] if len(target['components'])==2 else ['q']) for matrix in ('A','B')}
        if set(gradients) != keys or not all(math.isfinite(v) and v>0 for v in gradients.values()):
            raise ValueError('Missing or invalid adapter gradients')
        if not (m['zero_delta_exact'] is True and m['engineering_updates']==3 and len(m['losses'])==3
                and m['native_patch_shape']==[1,8192,1024] and m['native_cls_shape']==[1,1024]
                and len(m['frozen_digest'])==64 and all(math.isfinite(v) for row in m['losses'] for v in row.values())):
            raise ValueError('Invalid feature/gradient engineering evidence')
    return report


def accept(directory, destination):
    directory, destination = Path(directory), Path(destination)
    reports = list(directory.rglob('report.json'))
    if len(reports) != 1:
        raise ValueError('Expected one unambiguous returned report')
    path = reports[0]
    report = validate(json.loads(path.read_text(encoding='utf-8')))
    log = path.with_name('launcher.log')
    if json.loads(log.read_text(encoding='utf-8')) != report:
        raise ValueError('Launcher/report mismatch')
    archive = directory/'return.tar.gz'
    with tarfile.open(archive) as tar:
        if sorted(tar.getnames()) != ['launcher.log','report.json']:
            raise ValueError('Unexpected archive members')
        for p in (path,log):
            if tar.extractfile(p.name).read() != p.read_bytes():
                raise ValueError('Extracted/archive bytes differ')
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'phase39_report.json',report)
    write_json(destination/'phase39_acceptance.json', {
        'accepted':True,'producer_git_sha':PRODUCER,'formal_optimizer_updates':0,
        'engineering_updates':6,'source_data_accessed':False,'target_data_accessed':False,
        'received_files':{p.name:file_sha(p) for p in (path,log,archive)},
        'independently_checked':'archive bytes, report schema, target layout, all gradients, historical weight hashes',
        'remote_only_checks':'weight immutability and numerical forward/backward were reported by AutoDL; not replayed locally',
        'next_phase':40,'performance_and_fresh_process_resume_accepted':False})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--destination',type=Path,default=Path('experiments/LORA_CAUSALQ_1114'))
    a=p.parse_args(); accept(a.directory,a.destination)
