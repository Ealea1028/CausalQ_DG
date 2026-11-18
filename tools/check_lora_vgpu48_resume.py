"""Bounded new-GPU recovery gate; original formal checkpoints are read-only."""
import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets, SourceEvalTransform
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.hpi_packets import packet_loader
from causalq.runtime.hpi_prepare import configure
from causalq.runtime.identity import file_sha, tree_digest, write_json
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.runtime.lora_formal import build_backbone
from causalq.runtime.lora_l1_formal import DIRECTORY, NAME, VARIANTS, L1Trainer, core_l1_head
from causalq.runtime.lora_parallel import MPS
from causalq.runtime.lora_training import compact_state, load_state_file, rng_digest, seed_training_rng
from tools.check_lora_backbones import frozen_digest
from tools.check_lora_resume import compare, compare_rng
from tools.retire_lora_l0 import safe
from tools.train_lora_core_d3_l1 import BASE, runtime, feature_record, package

ACCEPTANCE = DIRECTORY/'phase42_vgpu48_audit_acceptance.json'


def check_runtime(actual, source, accepted):
    for key in ('python', 'torch', 'cuda', 'transformers', 'pip_freeze'):
        if actual[key] != source[key]: raise ValueError('Critical runtime changed: '+key)
    if (actual['gpu'] != accepted['target_gpu']['name']
            or actual['visible_memory_bytes'] != accepted['target_gpu']['memory_bytes']):
        raise ValueError('Not the audited48GB GPU')


def verify_donor(root, variant, row):
    folder = safe(root, Path(root)/variant/'checkpoints'/f"{row['step']:08d}")
    marker = json.loads((folder/'complete.json').read_text())
    payload = safe(root, folder/'state.pt')
    if (marker != {'step':row['step'], 'identity':row['identity'], 'sha256':row['expected_sha256']}
            or payload.stat().st_size != row['bytes'] or file_sha(payload) != row['expected_sha256']):
        raise ValueError('Cloned donor changed: '+variant)
    return folder


def restore_original(accepted, variant, pretrained):
    row = accepted['donors'][variant]; root = Path(accepted['source_root'])
    folder = verify_donor(root, variant, row)
    model = core_l1_head(build_backbone(NAME, pretrained), variant)
    trainer = L1Trainer(model, variant)
    if (trainer.initial_head_sha256 != row['identity']['initial_head_sha256']
            or trainer.initial_adapter_sha256 != row['identity']['initial_adapter_sha256']):
        raise ValueError('Common initialization changed')
    if trainer.restore(root/variant/'checkpoints', row['identity']) != row['step']:
        raise ValueError('Unexpected newer donor state; return new audit')
    original = load_state_file(folder/'state.pt')
    result = {'step':trainer.step, 'model_equal':tree_digest(compact_state(model))==tree_digest(original['model']),
              'optimizer_equal':tree_digest(trainer.optimizer.state_dict())==tree_digest(original['optimizer']),
              'rng_equal':rng_digest(capture_rng())==rng_digest(original['rng']), 'sha256':row['expected_sha256']}
    if not all(result[k] for k in ('model_equal','optimizer_equal','rng_equal')):
        raise ValueError('State transfer changed model/AdamW/RNG')
    return trainer, result


def worker(args):
    accepted = json.loads(ACCEPTANCE.read_text()); configure(); torch.set_num_threads(1); seed_training_rng(0)
    actual = runtime(); source = accepted['donors']['CA']['identity']['runtime']
    check_runtime(actual, source, accepted)
    props = torch.cuda.get_device_properties(0)
    if [props.major,props.minor] != accepted['target_gpu']['capability'] or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA capability/BF16 changed')
    report = {'ok':False,'mode':args.mode,'runtime':actual,'formal_optimizer_updates':0,'engineering_updates':0}
    if args.mode == 'inventory':
        report['models'] = {}
        for variant in VARIANTS:
            trainer, record = restore_original(accepted, variant, args.pretrained_root)
            trainer.check_base(); record['base_digest'] = frozen_digest(trainer.model.backbone)
            report['models'][variant] = record
            del trainer; gc.collect(); torch.cuda.empty_cache()
        report['ok'] = True; write_json(args.output/'inventory.json',report); return
    train,_ = build_source_datasets(args.data_root/'gta5',TrainTransform((512,512)),SourceEvalTransform())
    trainer, record = restore_original(accepted,'CA',args.pretrained_root)
    report['donor_transfer'] = record
    identity = {'action':'vgpu48-recovery-gate','formal':False,'variant':'CA','producer':args.expected_sha,
                'donor_sha256':record['sha256'],'source_producer':accepted['source_producer'],'runtime':actual}
    if args.mode == 'resume':
        trainer.restore(args.output/'prefix',identity)
    start = trainer.step
    stop = accepted['donors']['CA']['step']+(1 if args.mode=='prefix' else 2)
    profile = json.loads((DIRECTORY/'execution_profile.json').read_text())
    profile.update(data_workers=2,host_pipeline_budget_gib=16,pinned_memory_budget_gib=4)
    loader,policy = packet_loader(CanonicalInputStream(train.pairs,start=start*8,stop=stop*8),profile)
    before = frozen_digest(trainer.model.backbone); rows=[]; features={}
    hook = trainer.model.backbone.register_forward_hook(lambda m,i,o: features.update(feature_record(o)) if not features else None)
    try:
        for packet in loader:
            samples=[{'views':packet['views'][j],'label':packet['labels'][j],'metadata':meta}
                     for j,meta in enumerate(packet['metadata'])]
            fingerprint=tree_digest((packet['views'],packet['labels']))
            values=trainer.advance(samples,amp=True)
            rows.append({'step':trainer.step,'input_sha256':fingerprint,
                         'parameter_hashes':[m['id_parameters_sha256'] for m in packet['metadata']],
                         'loss_gradient_norm':values.float().cpu().tolist()})
    finally: hook.remove()
    if trainer.step != stop or frozen_digest(trainer.model.backbone) != before:
        raise ValueError('Coverage/original DINO invariant failed')
    checkpoint=trainer.commit(args.output/args.mode,identity)
    report.update(ok=True,checkpoint=checkpoint,rows=rows,first_features=features,
                  engineering_updates=len(rows),base_unchanged=True,loader_policy=policy,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output/f'{args.mode}.json',report)


def run(args):
    accepted=json.loads(ACCEPTANCE.read_text()); source=Path(accepted['source_root'])
    result={'action':'vgpu48-recovery-gate','git_sha':args.expected_sha,'ok':False,
            'formal_optimizer_updates':0,'engineering_updates':0,'source_committed_total':3238,
            'old_to_new_gpu_bitwise_equivalence_claimed':False,'engineering_state_eligible_for_formal':False}
    try:
        with MPS(args.output) as service:
            result['mps']=service.record
            for mode in ('inventory','continuous','prefix','resume'):
                command=[sys.executable,'-u','-m','tools.check_lora_vgpu48_resume','--worker','--mode',mode,
                         '--output',str(args.output),'--expected-sha',args.expected_sha,
                         '--data-root',str(args.data_root),'--pretrained-root',str(args.pretrained_root)]
                with (args.output/f'{mode}.log').open('w') as log:
                    completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=service.env,timeout=600,
                                             pass_fds=getattr(args,'lock_fds',()))
                if completed.returncode: raise RuntimeError(f'{mode} exited{completed.returncode}; return log')
                r=json.loads((args.output/f'{mode}.json').read_text())
                if not r['ok']: raise ValueError('Failed child report')
                if mode=='inventory': result['runtime']=r['runtime']
                elif r['runtime']!=result['runtime']: raise ValueError('Runtime changed between recovery processes')
                result['engineering_updates']+=r['engineering_updates']
            a=load_state_file(args.output/'continuous'/f"{accepted['donors']['CA']['step']+2:08d}"/'state.pt')
            b=load_state_file(args.output/'resume'/f"{accepted['donors']['CA']['step']+2:08d}"/'state.pt')
            result['model_max_abs']=compare(a['model'],b['model'])
            result['optimizer_max_abs']=compare(a['optimizer'],b['optimizer'])
            result['rng']=compare_rng(a['rng'],b['rng'])
            if not all(v['equal'] for v in result['rng'].values()): raise ValueError('RNG recovery mismatch')
            left=json.loads((args.output/'continuous.json').read_text())['rows']
            right=json.loads((args.output/'prefix.json').read_text())['rows']+json.loads((args.output/'resume.json').read_text())['rows']
            fields=('step','input_sha256','parameter_hashes')
            result['input_rows_equal']=[{k:r[k] for k in fields} for r in left]==[{k:r[k] for k in fields} for r in right]
            if not result['input_rows_equal']: raise ValueError('Input recovery rows differ')
            torch.testing.assert_close(torch.tensor([r['loss_gradient_norm'] for r in left],dtype=torch.float64),
                                       torch.tensor([r['loss_gradient_norm'] for r in right],dtype=torch.float64),atol=1e-6,rtol=1e-5)
            for variant,row in accepted['donors'].items(): verify_donor(source,variant,row)
            result['original_donors_unchanged']=True; result['ok']=True
    except BaseException: result['error']=traceback.format_exc()
    write_json(args.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['ok'] else 2


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--expected-sha',required=True)
    p.add_argument('--worker',action='store_true'); p.add_argument('--mode',choices=('inventory','continuous','prefix','resume'))
    p.add_argument('--package',action='store_true'); p.add_argument('--archive',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    a=p.parse_args()
    if sys.platform!='linux' or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.expected_sha:
        p.error('AutoDL exact code SHA required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): p.error('Tracked code modified')
    source=Path(json.loads(ACCEPTANCE.read_text())['source_root'])
    safe(BASE,a.output)
    if a.output==source or source in a.output.parents: p.error('Disposable output must be outside original run')
    if a.worker:
        if a.mode is None:p.error('mode required')
        worker(a); return 0
    import fcntl
    with ExitStack() as stack:
        handles=[]
        for path in (BASE/'gpu0.lock',source/'run.lock'):
            handle=stack.enter_context(path.open('r+')); handles.append(handle)
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.lock_fds=tuple(h.fileno() for h in handles)
        if a.package:
            if a.archive is None:p.error('archive required')
            package(a.output,a.archive);return 0
        a.output.mkdir(parents=True,exist_ok=False)
        return run(a)


if __name__=='__main__':raise SystemExit(main())
