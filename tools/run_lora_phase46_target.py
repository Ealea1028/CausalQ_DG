"""AutoDL Phase46: fixed-final native Cityscapes validation for all 15 models."""
import argparse
from contextlib import ExitStack
import csv
import gc
import hashlib
import json
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from causalq.runtime.hpi_prepare import Telemetry, configure
from causalq.runtime.hpi_target import NativeTargetDataset, evaluate_batches
from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.input_stream import worker_init
from causalq.runtime.lora_parallel import hardware
from causalq.runtime.lora_phase46 import (BATCH_SIZE, FREEZE, GEOMETRY, SOURCE, TARGET_SHA,
                                         load_contract, record_coverage)
from causalq.runtime.lora_source_diagnostics import ROOT, load_final, verify_final
from tools.retire_lora_l0 import safe
from tools.run_lora_phase45_source import state_stamp
from tools.train_lora_core_d3_l1 import BASE, git_sha, runtime


def digest_index(index):
    return hashlib.sha256(json.dumps(index,separators=(',',':'),sort_keys=True).encode()).hexdigest()


def loader_for(dataset, machine):
    # Single native frame per forward. Prefetch on host while the GPU runs.
    workers = min(6, max(0, machine['cpu_budget']-4))
    if machine['dev_shm_free_bytes'] < 2*(1<<30) or machine['available_host_bytes'] < 8*(1<<30): workers = 0
    parallel = {'worker_init_fn':worker_init,'multiprocessing_context':'spawn',
                'prefetch_factor':2,'timeout':180} if workers else {}
    return DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=workers,pin_memory=True,
                      generator=torch.Generator().manual_seed(0),**parallel)


def run(args, source, pins, ids, report):
    configure()
    actual = runtime()
    if actual != source['runtime']:
        keys = sorted(k for k in actual.keys() | source['runtime'].keys() if actual.get(k) != source['runtime'].get(k))
        raise ValueError(f'Frozen Phase46 runtime differs from Phase45: {keys}')
    report['runtime'] = actual
    report['hardware'] = machine = hardware()
    for pin in pins.values(): verify_final(pin,BASE)

    report['target_data_accessed'] = True
    write_json(args.output/'report.json',report)
    dataset = NativeTargetDataset(args.target_root,ids,GEOMETRY)
    index = None
    started = time.monotonic()
    for name,pin in pins.items():
        folder = args.output/'models'/name
        folder.mkdir(parents=True,exist_ok=True)
        identity = {'model':name,'evaluation_git_sha':args.expected_sha,'freeze_sha256':file_sha(FREEZE),
                    'checkpoint_sha256':pin['sha256'],'geometry':list(GEOMETRY),'native_batch_size':BATCH_SIZE,
                    'target_manifest_sha256':TARGET_SHA,'precision':'bf16_autocast','target_root':str(args.target_root)}
        print('Evaluating',name,flush=True)
        model = load_final(pin,BASE,args.pretrained_root)
        before = state_stamp(model)
        loader = loader_for(dataset,machine)
        clock = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        num_existing = len(list((folder/'records').rglob('*.json')))
        telemetry = folder/f'telemetry_{uuid.uuid4().hex[:12]}.csv'
        with Telemetry(telemetry,1):
            scores,covered = evaluate_batches({name:model},model.backbone,loader,folder,identity)
        if covered != ids or before != state_stamp(model):
            raise ValueError(f'Incomplete or changed target evaluation: {name}')
        seen = record_coverage(folder,name,identity,ids)
        if index is None:
            index = seen
            write_json(args.output/'target_payload_identity.json',{
                'fields':['image_sha256','label_sha256','valid_pixels'],'records':index,'sha256':digest_index(index)})
        elif seen != index:
            raise ValueError(f'Target images, label files or valid-pixel counts changed: {name}')
        metric = scores[name]
        score = metric.compute()
        pixels = int(metric.confusion.sum())
        if pixels != sum(x[2] for x in index.values()): raise ValueError(f'Incorrect cumulative confusion: {name}')
        result = {**score,'sample_count':len(ids),'ids':ids,'geometry':list(GEOMETRY),'identity':identity,
            'miou_percent':100*score['miou'],'source_miou':source['models'][name]['source_miou'],
            'confusion_matrix':metric.confusion.tolist(),'valid_pixels':pixels,'state_unchanged':True,'rng_unchanged':True,
            'seconds_this_invocation':time.monotonic()-clock,'records_reused':num_existing,
            'new_native_image_forwards':len(ids)-num_existing,'data_workers':loader.num_workers,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'parameters_in_lora':getattr(model.backbone,'lora_manifest',{}).get('trainable_parameters',0),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        write_json(folder/'target_metrics.json',result)
        report['models'][name] = {'miou':score['miou'],'sample_count':len(ids),'checkpoint_sha256':pin['sha256'],
                                  'metrics_sha256':file_sha(folder/'target_metrics.json')}
        write_json(args.output/'report.json',report)
        del model,loader
        gc.collect()
        torch.cuda.empty_cache()

    with (args.output/'target_miou.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','target_images','source_miou','target_miou','target_miou_percent','checkpoint_sha256'])
        for name, result in report['models'].items():
            writer.writerow([name,len(ids),source['models'][name]['source_miou'],result['miou'],
                             100*result['miou'],pins[name]['sha256']])
    for pin in pins.values(): verify_final(pin,BASE)
    report.update(ok=True,status='complete',model_count=len(pins),images_per_model=len(ids),
                  logical_image_forwards=len(ids)*len(pins),payload_index_sha256=digest_index(index),
                  checkpoints_unchanged=True,seconds=time.monotonic()-started)
    write_json(args.output/'report.json',report)
    print(f'Phase46 complete: {len(pins)} models × {len(ids)} target images',flush=True)


def export(out, archive):
    archive = safe(BASE,archive)
    if archive.parent != BASE or not archive.name.endswith('_return.tar.gz') or archive.exists():
        raise ValueError('Use a new return archive in the experiment output directory')
    manifest = {}
    for p in sorted(out.rglob('*')):
        if p == out/'export_manifest.json' or not p.is_file() or p.suffix not in ('.json','.csv','.log','.txt'):
            continue
        p = safe(out,p)
        manifest[p.relative_to(out).as_posix()] = {'sha256':file_sha(p),'bytes':p.stat().st_size}
    write_json(out/'export_manifest.json',manifest)
    with tarfile.open(archive,'x:gz') as bundle:
        for rel in sorted({*manifest,'export_manifest.json'}):
            bundle.add(out/rel,arcname=rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-root',type=Path,default=Path('/root/autodl-tmp/datasets/cityscapes'))
    parser.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    parser.add_argument('--archive',type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path.cwd().resolve() != ROOT.resolve() or git_sha() != args.expected_sha:
        parser.error('Use the pinned AutoDL checkout')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        parser.error('Tracked sources changed')
    if args.output != BASE/f'phase46_target_{args.expected_sha[:7]}' or args.target_root.name != 'cityscapes':
        parser.error('Use fixed Phase46 output and Cityscapes root')
    safe(BASE,args.output)
    source,pins,ids = load_contract()
    args.output.mkdir(exist_ok=True)
    import fcntl
    with ExitStack() as stack:
        locks = [args.output/'run.lock']
        if args.archive is None:
            roots = {Path(pin['path']).relative_to(BASE).parts[0] for pin in pins.values()}
            locks = [BASE/'gpu0.lock',*locks,*[BASE/r/'run.lock' for r in sorted(roots)]]
        for path in locks:
            handle = stack.enter_context(safe(BASE,path).open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.archive is not None:
            export(args.output,args.archive)
            return
        def stop(signum,frame): raise KeyboardInterrupt('Phase46 stop requested')
        signal.signal(signal.SIGTERM,stop)
        report = {'phase':46,'git_sha':args.expected_sha,'status':'running','ok':False,'models':{},
                  'formal_optimizer_updates':0,'target_data_accessed':False,'target_pca':False,
                  'source_acceptance_sha256':file_sha(SOURCE),'target_manifest_sha256':TARGET_SHA,
                  'target_geometry':list(GEOMETRY),'native_batch_size':BATCH_SIZE,'precision':'bf16_autocast'}
        write_json(args.output/'report.json',report)
        try:
            run(args,source,pins,ids,report)
        except BaseException as error:
            report.update(ok=False,status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                          error=traceback.format_exc())
            write_json(args.output/'report.json',report)
            raise


if __name__ == '__main__':
    main()
