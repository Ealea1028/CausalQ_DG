"""AutoDL Phase45: source-only diagnostics and shared PCA from pinned finals."""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import uuid

import numpy as np
import torch

from causalq.analysis import lora_joint_pca as pca
from causalq.analysis.lora_effect_moments import validate_reference
from causalq.datasets.segmentation import IMAGENET_MEAN, IMAGENET_STD
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets
from causalq.interventions.style import StyleInterventionBank
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.hpi_prepare import Telemetry, configure
from causalq.runtime.hpi_source_batch import source_batches
from causalq.runtime.hpi_source_loader import SourceDiagnosticDataset, source_loader
from causalq.runtime.identity import file_sha, tree_digest, write_json
from causalq.runtime.lora_formal import ROOT, build_backbone
from causalq.runtime.lora_parallel import hardware
from causalq.runtime.lora_source_diagnostics import (
    VARIANTS, BACKBONES, SAMPLE, CONTRACT, inventory, verify_contract, verify_final,
    load_final, read_record, write_record, summarize, readout,
)
from causalq.runtime.lora_training import rng_digest
from causalq.runtime.lora_phase45_resume import check_record, compatible_key, migrate_sample, old_source
from tools.retire_lora_l0 import safe
from tools.train_lora_core_d3_l1 import BASE, git_sha, runtime


def model_dir(out, name):
    return out/'models'/name


def samples(dataset):
    ids = [pair.sample_id for pair in dataset.pairs]
    selected = SAMPLE.read_text(encoding='ascii').splitlines()
    if len(ids) != len(set(ids)) or len(ids) != 2497 or len(selected) != 128 or len(set(selected)) != 128 or not set(selected) <= set(ids):
        raise ValueError('Frozen source/PCA coverage differs')
    dense = sorted(np.random.default_rng(pca.SEED).choice(selected, 12, replace=False).tolist())
    return ids, selected, dense


def source_inputs(dataset):
    """One digest of the source file inventory, shared by the fifteen resume keys."""
    index = {}
    for pair in dataset.pairs:
        index[pair.sample_id] = []
        for path in (pair.image, pair.label):
            st = path.stat()
            index[pair.sample_id].append([str(path), st.st_size, st.st_mtime_ns])
    return hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()


def prepared(dataset, indices, machine, threads):
    profile = {'host_pipeline_budget_gib': max(6, min(24, machine['available_host_bytes']/(1<<30)*.45)),
               'prefetch_factor': 2}
    workers = max(0, min(8, machine['cpu_budget']-threads-1))
    if machine['available_host_bytes'] < 12*(1<<30): workers = 0
    loader, policy = source_loader(SourceDiagnosticDataset(dataset, indices, StyleInterventionBank()),
                                   profile, workers=workers)
    print('source_loader', json.dumps(policy), flush=True)
    for packet in loader:
        for item in packet:
            yield item['id'], item['views'], item['label']


def capture_pca(folder, sample_id, features, offset, views, label, dense, logits=None):
    data = pca.extract(features, offset, label, sample_id, dense=dense)
    if dense:
        mean = views.new_tensor(IMAGENET_MEAN)[None,:,None,None]
        std = views.new_tensor(IMAGENET_STD)[None,:,None,None]
        data['images'] = ((views[:2]*std+mean).clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
        data['label'] = label[0].byte().numpy()
        if logits is not None:
            data['prediction'] = logits[offset:offset+2].argmax(1).byte().cpu().numpy()
    path = folder/'features'/f'{sample_id}.npz'
    pca.atomic_npz(path, **data)
    return file_sha(path)


def verified_cache(folder, sample_id, record, selected):
    if sample_id in selected:
        p = folder/'features'/f'{sample_id}.npz'
        if not p.is_file() or file_sha(p) != record.get('pca_sha256'):
            raise ValueError(f'PCA cache hash mismatch for {sample_id}')


def state_stamp(model):
    # Frozen and trainable tensors plus buffers, checked without optimizer restore.
    return tree_digest(model.state_dict()), rng_digest(capture_rng())


def run_model(args, name, pin, dataset, ids, selected, dense, base_key, machine, threads, legacy):
    folder = model_dir(args.output, name)
    key = dict(base_key, model=name, checkpoint_sha256=pin['sha256'])
    legacy_root, legacy_key = legacy
    previous = dict(legacy_key, model=name, checkpoint_sha256=pin['sha256']) if name == 'D3_UA_L1' else None
    todo = []
    observed_producers = {}
    for j, sample_id in enumerate(ids):
        path = folder/'records'/f'{sample_id}.json.gz'
        if not path.is_file():
            if previous is None or not migrate_sample(model_dir(legacy_root,name),folder,sample_id,
                                                       previous,key,selected,verified_cache):
                todo.append(j)
                continue
        record = read_record(path)
        check_record(record,sample_id,key,previous)
        verified_cache(folder, sample_id, record, selected)
        producer = record['key']['git_sha']
        observed_producers[producer] = observed_producers.get(producer,0)+1
    print(name, 'records reused', len(ids)-len(todo), 'pending images', len(todo), flush=True)
    model = None
    started = time.monotonic()
    routed = pin['variant'] in ('B2', 'B3')
    interventions = selected if routed else ()
    pca_hashes = {}
    reference = folder/'effect_reference_check.json'
    if reference.exists() and json.loads(reference.read_bytes())['key'] != key:
        raise ValueError(f'FP64 equivalence receipt differs: {name}')

    def reference_check(sample_id, full, anchor, label, rows):
        difference = validate_reference(full,anchor,label,rows)
        write_json(reference, {'key':key,'same_forward_sample_id':sample_id,'max_abs_difference':difference,
                              'rtol':1e-7,'atol':1e-10,'nulls_match':True})

    def observer(features, prediction, batch):
        for j, (sample_id, views, label) in enumerate(batch):
            if sample_id in selected:
                pca_hashes[sample_id] = capture_pca(folder, sample_id, features, j*5,
                    views, label, sample_id in dense, prediction.logits)

    if todo:
        model = load_final(pin, BASE, args.pretrained_root)
        before = state_stamp(model)
        folder.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        samples_iter = prepared(dataset, todo, machine, threads)
        with Telemetry(folder/f'telemetry_{uuid.uuid4().hex[:12]}.csv', 1):
            for i, batch in enumerate(source_batches(samples_iter, max_images=args.image_batch)):
                output = readout(model,batch,intervention_ids=interventions,observer=observer,
                                 reference_check=None if reference.exists() else reference_check)
                for sample_id, record in output.items():
                    write_record(folder/'records'/f'{sample_id}.json.gz',
                                 {'key':key, 'id':sample_id, 'record':record,
                                  'pca_sha256':pca_hashes.pop(sample_id, None)})
                if i % 25 == 0:
                    print(name, 'new batches', i+1, 'pending images at start', len(todo), flush=True)
        if before != state_stamp(model):
            raise ValueError(f'Read-only diagnostic changed model or RNG: {name}')
        write_json(folder/'inference_receipt.json', {'key':key,'state_unchanged':True,'rng_unchanged':True,
                'logical_view_count':len(ids)*5,'new_backbone_image_views':len(todo)*5,
                'seconds':time.monotonic()-started,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})
    receipt = folder/'inference_receipt.json'
    if receipt.exists() and json.loads(receipt.read_bytes())['key'] != key:
        raise ValueError(f'Read-only inference receipt differs: {name}')
    if not receipt.exists():
        # A process can stop just after writing its last sample. Verify a new
        # read-only model load instead of dropping already completed records.
        model = load_final(pin, BASE, args.pretrained_root)
        before = state_stamp(model)
        if before != state_stamp(model): raise ValueError('Read-only checkpoint changed')
        write_json(receipt, {'key':key,'state_unchanged':True,'rng_unchanged':True,'logical_view_count':len(ids)*5,
                            'new_backbone_image_views':0,'recovered_after_final_record':True})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not reference.exists(): raise ValueError(f'Missing phase45 effect reference receipt: {name}')
    if todo:
        observed_producers[args.expected_sha]=observed_producers.get(args.expected_sha,0)+len(todo)
    write_json(folder/'record_provenance.json',{'records_by_git_sha':observed_producers,'sample_count':len(ids)})
    return summarize(folder, ids, interventions, pin['variant'].startswith('B')), observed_producers


@torch.inference_mode()
def d2_original(args, dataset, ids, selected, dense, key, machine, threads):
    folder = model_dir(args.output, 'D2_original')
    marker = folder/'extraction.json'
    if marker.exists():
        row = json.loads(marker.read_bytes())
        if row['key'] != key or set(row['features']) != set(selected):
            raise ValueError('Original D2 PCA receipt changed')
        for sample_id, digest in row['features'].items():
            verified_cache(folder, sample_id, {'pca_sha256':digest}, selected)
        return
    model = build_backbone(BACKBONES['D2'], args.pretrained_root).eval()
    before = state_stamp(model)
    hashes = {}
    indices = []
    for i, sample_id in enumerate(ids):
        if sample_id not in selected: continue
        path = folder/'receipts'/f'{sample_id}.json'
        if not path.exists():
            indices.append(i)
            continue
        row = json.loads(path.read_bytes())
        if row['key'] != key: raise ValueError('Original D2 extraction resume identity changed')
        verified_cache(folder, sample_id, row, selected)
        hashes[sample_id] = row['pca_sha256']
    for batch in source_batches(prepared(dataset, indices, machine, threads), max_images=args.image_batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = model(torch.cat([x[1] for x in batch]).cuda())
        for i, (sample_id, views, label) in enumerate(batch):
            hashes[sample_id] = capture_pca(folder, sample_id, features, i*5, views, label, sample_id in dense)
            write_json(folder/'receipts'/f'{sample_id}.json', {'key':key,'pca_sha256':hashes[sample_id]})
    if before != state_stamp(model) or set(hashes) != set(selected):
        raise ValueError('Original D2 PCA coverage/state mismatch')
    write_json(marker, {'key':key,'features':hashes,'state_unchanged':True,'rng_unchanged':True,'backbone_image_views':len(hashes)*5})


def pca_outputs(out, selected, dense, pins):
    destination = out/'pca'
    destination.mkdir(exist_ok=True)
    manifest = {'seed':pca.SEED, 'views':['original','photo1','photo2','photo3','photo4'],
                'sample_ids':selected,'dense_ids':dense,'dense_views':['original','photo1'],
                'fit':'equal groups, centered FP64 covariance, unwhitened', 'backbones':{}}
    for short in BACKBONES:
        orig = 'D3_U0_L0' if short == 'D3' else 'D2_original'
        names = [orig] + [f'{short}_{v}_L1' for v in VARIANTS]
        paths = {name:[model_dir(out,name)/'features'/f'{i}.npz' for i in selected] for name in names}
        row = {'groups':['Original', *VARIANTS], 'original_group':orig, 'checkpoint_sha256':{
               name:pins[name]['sha256'] for name in names if name in pins},'layers':{},'coordinates':{}}
        for i, path in zip(selected,paths[orig]):
            with np.load(path) as data:
                row['coordinates'][i] = {'grid':data['grid'].tolist(),'flat_indices':data['indices'].tolist()}
        folder = destination/short
        folder.mkdir(exist_ok=True)
        for layer in pca.LAYERS:
            groups, labels, per_group_style = [], [], []
            for name in names:
                arrays, view_ids, style = [], [], []
                for i, path in zip(selected, paths[name]):
                    with np.load(path) as data:
                        coord = row['coordinates'][i]
                        if data['indices'].tolist() != coord['flat_indices'] or data['grid'].tolist() != coord['grid']:
                            raise ValueError('Joint PCA spatial coordinates changed between models')
                        x = data[layer]
                    arrays.append(x.reshape(-1,x.shape[-1]))
                    view_ids.extend(np.repeat(np.arange(5), len(arrays[-1])//5))
                    if x.size: style.append(float(x.var(axis=0, dtype=np.float64).mean()))
                groups.append(np.concatenate(arrays))
                labels.append(view_ids)
                per_group_style.append(fmean_or_none(style))
            if any(v != labels[0] for v in labels[1:]): raise ValueError('Unaligned PCA view labels')
            mu, basis, projections, ranges, explained = pca.fit_joint(groups)
            path = folder/f'{layer}_basis.npz'
            pca.atomic_npz(path, mean=mu, components=basis, rgb_ranges=ranges, explained_variance_ratio=explained)
            row['layers'][layer] = {'basis':str(path.relative_to(out)),'sha256':file_sha(path),
                'rows_per_group':len(groups[0]),'rgb_ranges':ranges.tolist(),'explained_variance_ratio':explained.tolist(),
                'cumulative_explained_variance_ratio':explained.cumsum().tolist(),
                'style_variance_by_group':dict(zip(row['groups'],per_group_style)),
                'matched_drift':{v:pca.drift(groups[0],x) for v,x in zip(VARIANTS,groups[1:])}}
            if layer in ('patch','cls'):
                pca.draw_scatter(folder/f'{layer}_scatter.png', f'{short} | Joint {layer} PCA | shared 1-99% axes',
                                 projections,row['groups'],labels[0],ranges)
            if layer != 'patch': continue
            for sample_id in dense:
                with np.load(model_dir(out,orig)/'features'/f'{sample_id}.npz') as raw:
                    for name in names[1:]:
                        with np.load(model_dir(out,name)/'features'/f'{sample_id}.npz') as adapted:
                            predictions = {'adapted':adapted['prediction']}
                            if short == 'D3' and name == 'D3_U0_L1': predictions['reference'] = raw['prediction']
                            pca.draw_panel(folder/f'{name}_{sample_id}.png', sample_id, name,
                                           raw, adapted, raw['images'],raw['label'],predictions,mu,basis,ranges)
        manifest['backbones'][short] = row
    write_json(destination/'pca_manifest.json', manifest)


def fmean_or_none(values):
    return sum(values)/len(values) if values else None


def run(args):
    configure()
    machine = hardware()
    threads = max(1,min(4,machine['cpu_budget']//2))
    torch.set_num_threads(threads)
    actual = runtime()
    pins = inventory()
    dataset = build_source_datasets(args.source_root, None, SourceEvalTransform())[1]
    ids, selected, dense = samples(dataset)
    accepted_weights = json.loads((CONTRACT.parent/'phase39_report.json').read_bytes())
    base_key = {'git_sha':args.expected_sha,'freeze_sha256':file_sha(CONTRACT),'source_fingerprint':source_inputs(dataset),
               'source_root':str(args.source_root),'pretrained_root':str(args.pretrained_root),
               'sample_sha256':file_sha(SAMPLE),'long_edge':1024,'pad_multiple':16,'views':5,'image_batch':args.image_batch,
               'weight_manifest_sha256':file_sha(CONTRACT.parent/'phase39_report.json'), 'runtime':actual}
    legacy = old_source(BASE)
    if not compatible_key(legacy[1],base_key):
        raise ValueError('Phase45 resume inputs or runtime differ from the interrupted run')
    started = time.monotonic()
    report = {'phase':45,'git_sha':args.expected_sha,'status':'running','ok':False,'runtime':actual,'hardware':machine,
              'optimizer_updates':0,'target_data_accessed':False,'models':{},'source_protocol':base_key}
    write_json(args.output/'report.json', report)
    for pin in pins.values(): verify_final(pin, BASE)
    for name, pin in pins.items():
        print('Diagnosing', name, flush=True)
        summary, producers = run_model(args,name,pin,dataset,ids,set(selected),set(dense),base_key,machine,threads,legacy)
        report['models'][name] = {'samples':summary['sample_count'],'source_miou':summary['miou'],
                                 'checkpoint_sha256':pin['sha256'],'target_miou':None,'records_by_git_sha':producers}
        write_json(args.output/'report.json',report)
    with (args.output/'source_miou.csv').open('w',newline='',encoding='utf-8') as stream:
        import csv
        writer=csv.writer(stream)
        writer.writerow(['model','backbone','variant','lora','source_samples','source_miou','source_miou_percent','checkpoint_sha256'])
        for name,row in report['models'].items():
            pin=pins[name]
            writer.writerow([name,pin['backbone'],pin['variant'],int(pin['lora']),row['samples'],row['source_miou'],
                             100*row['source_miou'],row['checkpoint_sha256']])
    d2_original(args,dataset,ids,set(selected),set(dense),base_key,machine,threads)
    gc.collect(); torch.cuda.empty_cache()
    pca_outputs(args.output,selected,dense,pins)
    manifest_path = args.output/'pca/pca_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['pretrained_weight_files'] = {x['model']:x['weight_files'] for x in accepted_weights['models']}
    manifest['sample_manifest_sha256'] = file_sha(SAMPLE)
    manifest['dense_manifest_sha256'] = hashlib.sha256(('\n'.join(dense)+'\n').encode()).hexdigest()
    manifest['tool_git_sha'] = args.expected_sha
    manifest['feature_semantics'] = {'patch':'normalized final DINO patch tokens before Query/router',
                                    'layer6':'native backbone intermediate map 1',
                                    'layer12':'native backbone intermediate map 2',
                                    'layer18':'native backbone intermediate map 3',
                                    'layer24':'native backbone intermediate map 4',
                                    'cls':'normalized final DINO CLS; independent basis'}
    write_json(manifest_path,manifest)
    for pin in pins.values(): verify_final(pin, BASE)
    report.update(ok=True,status='complete',seconds=time.monotonic()-started,
                  model_count=15,logical_model_image_views=15*2497*5,
                  expected_backbone_image_views=15*2497*5+128*5,
                  reference_checkpoints_unchanged=True,pca_manifest_sha256=file_sha(manifest_path))
    write_json(args.output/'report.json',report)
    print('Phase45 complete',len(report['models']),'models',flush=True)


def export(out, archive):
    archive = safe(BASE, archive)
    if archive.parent != BASE or not archive.name.endswith('_return.tar.gz'):
        raise ValueError('Use a new return archive beside the output directory')
    selected = [p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png','.npz','.log','.txt')
                and p != out/'export_manifest.json'
                and not {'features','records','receipts'}.intersection(p.relative_to(out).parts)]
    if archive.exists(): raise FileExistsError(archive)
    manifest = {p.relative_to(out).as_posix():{'sha256':file_sha(safe(out,p)),'bytes':p.stat().st_size} for p in selected}
    write_json(out/'export_manifest.json',manifest)
    with tarfile.open(archive,'x:gz') as bundle:
        for path in sorted(set(selected) | {out/'export_manifest.json'}):
            bundle.add(path,arcname=path.relative_to(out).as_posix())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-sha',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/datasets/gta5'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    p.add_argument('--image-batch',type=int,choices=(1,2),default=2)
    p.add_argument('--archive',type=Path)
    args = p.parse_args()
    if sys.platform != 'linux' or git_sha() != args.expected_sha or Path.cwd().resolve() != ROOT.resolve():
        p.error('AutoDL exact checkout required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        p.error('Tracked source changed')
    if args.output != BASE/f'phase45_source_{args.expected_sha[:7]}' or args.source_root.name != 'gta5':
        p.error('Phase45 requires its exact-SHA output and a GTA5 root')
    safe(BASE,args.output)
    verify_contract()
    args.output.mkdir(exist_ok=True)
    import fcntl
    with ExitStack() as stack:
        paths = [args.output/'run.lock']
        if args.archive is None:
            paths = [BASE/'gpu0.lock', *paths, *sorted({BASE/Path(pin['path']).relative_to(BASE).parts[0]/'run.lock'
                                                      for pin in inventory().values()})]
            previous, _ = old_source(BASE)
            if previous.is_dir(): paths.append(previous/'run.lock')
        for path in paths:
            handle = stack.enter_context(safe(BASE,path).open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.archive is not None:
            export(args.output,args.archive)
        else:
            def stop_on_term(signum, frame):
                raise KeyboardInterrupt('Phase45 stop requested')
            signal.signal(signal.SIGTERM, stop_on_term)
            try:
                run(args)
            except BaseException as error:
                path = args.output/'report.json'
                report = json.loads(path.read_bytes()) if path.exists() else {'phase':45,'git_sha':args.expected_sha}
                report.update(ok=False, status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                              error=traceback.format_exc(),optimizer_updates=0,target_data_accessed=False)
                write_json(path,report)
                raise


if __name__ == '__main__':
    main()
