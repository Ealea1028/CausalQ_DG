"""Phase40 source-only 4 versus 2+2 fresh-process recovery; no formal training."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets, SourceEvalTransform, MANIFEST_HASHES
from causalq.models.backbone_factory import build_backbone
from causalq.models.lora import inject_dino_lora
from causalq.models.query_segmentor import QuerySegmentor
from causalq.runtime.hpi_prepare import configure
from causalq.runtime.hpi_packets import packet_loader
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.runtime.identity import write_json,file_sha,tree_digest
from causalq.runtime.lora_training import (make_optimizer,update,save_checkpoint,restore_checkpoint,load_state_file,rng_digest,seed_training_rng)
from causalq.runtime.cohort_runner import capture_rng
from tools.check_lora_backbones import frozen_digest

ROOT=Path(__file__).resolve().parents[1]
ACCEPTED=ROOT/'experiments/LORA_CAUSALQ_1114/phase39_report.json'
PROFILE={'data_workers':12,'prefetch_factor':2,'loader_packet_samples':8,'backbone_microbatch_views':16,
         'pin_memory':True,'pinned_memory_budget_gib':8,'host_pipeline_budget_gib':32}


def build(name,root,*,head_type=QuerySegmentor):
    kwargs={} if name.startswith('dinov2') else {'intermediate_indices':(6,12,18,24)}
    weights=root/'dinov2_vitl14/dinov2_vitl14_hpi_p16.pth' if name.startswith('dinov2') else root/'dinov3_vitl16'
    accepted=next(m for m in json.loads(ACCEPTED.read_text())['models'] if m['model']==name)
    for key,digest in accepted['weight_files'].items():
        path=weights/key if weights.is_dir() else weights
        if file_sha(path)!=digest: raise ValueError(f'Accepted weight changed: {path}')
    b=build_backbone(name,weights=weights,dtype=torch.bfloat16,**kwargs).cuda()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model=head_type(b,queries_per_class=2,interaction='static',cross_attention_layers=0,
                             decoder_channels=256,dropout=.1,num_classes=19).cuda()
    inject_dino_lora(b)
    return model


def worker(args):
    configure()
    seed_training_rng(0)
    import transformers
    expected=json.loads(ACCEPTED.read_text())['environment']
    actual={'python':sys.version,'torch':str(torch.__version__),'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    if any(actual[k]!=expected[k] for k in actual): raise ValueError(f'Runtime changed: {actual}')
    train,_=build_source_datasets(args.data_root/'gta5',TrainTransform((512,512)),SourceEvalTransform())
    model=build(args.model,args.pretrained_root)
    optimizer=make_optimizer(model)
    identity={'study':'lora_causalq_1114','phase':40,'model':args.model,'variant':'UA','lora':model.backbone.lora_manifest,
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'phase39_report_sha256':file_sha(ACCEPTED),'source_manifests':MANIFEST_HASHES,
              'runtime':actual,'effective_source_batch':8,'microbatch':1,'seed':0,
              'optimizer_recipe':'hpi_40k_poly09_warm1500_ratio1e-6_wd1e-5_norm0_v1',
              'source_stream':'hpi_step_sha256_v1','source_root':str(args.data_root.resolve()),
              'rng_initialization':'python_numpy_torch_seed0_v1'}
    start=restore_checkpoint(args.output/'split2',model,optimizer,identity) if args.mode=='resume' else 0
    if start!=(2 if args.mode=='resume' else 0): raise ValueError('Unexpected resume cursor')
    rng_at_start={key:rng_digest(value) for key,value in capture_rng().items()}
    stop=2 if args.mode=='prefix' else 4
    stream=CanonicalInputStream(train.pairs,start=start*8,stop=stop*8)
    loader,policy=packet_loader(stream,PROFILE)
    before=frozen_digest(model)
    versions=tuple(p._version for p in model.backbone.parameters() if not p.requires_grad)
    rows=[]
    for step,packet in enumerate(loader,start=start):
        samples=[{'views':packet['views'][j],'label':packet['labels'][j],'metadata':meta}
                 for j,meta in enumerate(packet['metadata'])]
        values=update(model,optimizer,samples,step,microbatch=1,amp=True)
        if versions!=tuple(p._version for p in model.backbone.parameters() if not p.requires_grad):
            raise ValueError('Frozen parameter version changed')
        rows.append({'step':step,'inputs':[s['metadata']['id_parameters_sha256'] for s in samples],
                     'input_tensor_sha256':tree_digest((packet['views'],packet['labels'])),**values})
    if len(rows)!=stop-start or frozen_digest(model)!=before: raise ValueError('Coverage/base invariance failed')
    target={'continuous':'continuous4','prefix':'split2','resume':'resumed4'}[args.mode]
    save_checkpoint(args.output/target,model,optimizer,stop,identity)
    write_json(args.output/f'{args.mode}.json',{'ok':True,'identity':identity,'rows':rows,'loader_policy':policy,
        'engineering_updates':len(rows),'formal_optimizer_updates':0,'frozen_digest':before,
        'rng_at_start_by_component':rng_at_start,
        'rng_at_end_by_component':{key:rng_digest(value) for key,value in capture_rng().items()},
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    del loader,model,optimizer
    gc.collect()


def compare(a,b):
    """Compare full nested optimizer/model tensors, including zero states and group order."""
    errors=[]
    def visit(x,y):
        if isinstance(x,torch.Tensor):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
            errors.append(float((x.double()-y.double()).abs().max()) if x.numel() else 0.)
        elif isinstance(x,dict):
            if set(x)!=set(y): raise ValueError('State keys differ')
            for k in x: visit(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            if type(x)!=type(y) or len(x)!=len(y): raise ValueError('State sequence differs')
            for left,right in zip(x,y): visit(left,right)
        elif x!=y: raise ValueError('State scalar differs')
    visit(a,b)
    return max(errors,default=0.)


def compare_rng(continuous,resumed):
    if set(continuous)!=set(resumed): raise ValueError('RNG provider coverage differs')
    return {key:{'equal':rng_digest(continuous[key])==rng_digest(resumed[key]),
                 'continuous_digest':rng_digest(continuous[key]),'resumed_digest':rng_digest(resumed[key])}
            for key in continuous}


def diagnose_previous(directory):
    """Read-only audit of old own hash-verified snapshots, never acceptance."""
    directory=Path(directory)
    result={'previous_accepted':False,'directory':str(directory),'models':[]}
    for name in ('dinov3_vitl16','dinov2_vitl14_hpi_p16'):
        values=[]
        for tag in ('continuous4','resumed4'):
            root=directory/name/tag
            if not (root/'complete.json').exists(): break
            marker=json.loads((root/'complete.json').read_text())
            if marker['identity']['git_sha']!='2cfbb3b89d2761b576b688085ca4ba18adab5f8c':
                raise ValueError('Unexpected previous producer')
            if marker['sha256']!=file_sha(root/'state.pt'): raise ValueError('Previous checkpoint corrupted')
            values.append(load_state_file(root/'state.pt')['rng'])
        if len(values)==2:
            result['models'].append({'model':name,'rng_by_component':compare_rng(*values)})
    return result


def run(args):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Tracked source must be clean')
    report={'study':'lora_causalq_1114','phase':40,'action':'resume-check','ok':False,
            'formal_optimizer_updates':0,'target_data_accessed':False,'source_data_accessed':True,
            'performance_calibration_complete':False,'models':[],
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    report['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    report['weight_identity_source']=file_sha(ACCEPTED)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists(): raise ValueError('Existing report; return evidence before retry')
    try:
        if args.previous_output is not None:
            write_json(args.output/'previous_rng_audit.json',diagnose_previous(args.previous_output))
        if not json.loads((ACCEPTED.parent/'phase39_acceptance.json').read_text())['accepted']:
            raise ValueError('Phase39 not accepted')
        for name in ('dinov3_vitl16','dinov2_vitl14_hpi_p16'):
            directory=args.output/name; directory.mkdir(exist_ok=True)
            for mode in ('continuous','prefix','resume'):
                path=directory/f'{mode}.json'
                if not path.exists():
                    cmd=[sys.executable,'-u','-m','tools.check_lora_resume','--worker','--model',name,'--mode',mode,
                         '--output',str(directory),'--data-root',str(args.data_root),'--pretrained-root',str(args.pretrained_root)]
                    with (directory/f'{mode}.log').open('a') as log:
                        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                print(name,mode,'complete',flush=True)
            for tag in ('continuous4','split2','resumed4'):
                marker=json.loads((directory/tag/'complete.json').read_text())
                if marker['sha256']!=file_sha(directory/tag/'state.pt'): raise ValueError('Checkpoint bytes corrupted')
            continuous=load_state_file(directory/'continuous4/state.pt')
            resumed=load_state_file(directory/'resumed4/state.pt')
            if continuous['identity']!=resumed['identity']: raise ValueError('Run identities differ')
            model_error=compare(continuous['model'],resumed['model'])
            optimizer_error=compare(continuous['optimizer'],resumed['optimizer'])
            rng_comparison=compare_rng(continuous['rng'],resumed['rng'])
            write_json(directory/'comparison.json',{'model_max_abs':model_error,'optimizer_max_abs':optimizer_error,
                       'rng_by_component':rng_comparison})
            if not all(row['equal'] for row in rng_comparison.values()): raise ValueError('Final RNG differs; see comparison.json')
            c,p,r=[json.loads((directory/f'{m}.json').read_text()) for m in ('continuous','prefix','resume')]
            if continuous['step']!=4 or resumed['step']!=4 or continuous['source_cursor']!=32 or resumed['source_cursor']!=32:
                raise ValueError('Final recovery cursor mismatch')
            if ([row['inputs'] for row in c['rows']] != [row['inputs'] for row in p['rows']+r['rows']]
                or len(c['rows'])!=4 or len(p['rows'])!=2 or len(r['rows'])!=2): raise ValueError('Input coverage mismatch')
            if [row['input_tensor_sha256'] for row in c['rows']] != [row['input_tensor_sha256'] for row in p['rows']+r['rows']]:
                raise ValueError('Actual source tensors differ across processes')
            if not c['frozen_digest']==p['frozen_digest']==r['frozen_digest']: raise ValueError('Base identity differs')
            report['models'].append({'model':name,'engineering_updates':8,'continuous_vs_resumed_model_max_abs':model_error,
                'optimizer_max_abs':optimizer_error,'rng_equal':True,'inputs_equal':True,'base_unchanged':True,
                'checkpoint_sha256':{tag:file_sha(directory/tag/'state.pt') for tag in ('continuous4','split2','resumed4')},
                'runs':[c,p,r]})
        report['ok']=True
    except Exception:
        report['error']=traceback.format_exc()
    report['exit_code']=0 if report['ok'] else 1
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report['exit_code']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--model',choices=('dinov3_vitl16','dinov2_vitl14_hpi_p16'))
    p.add_argument('--mode',choices=('continuous','prefix','resume'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    p.add_argument('--previous-output',type=Path)
    a=p.parse_args()
    if a.worker:
        if not a.model or not a.mode: p.error('Worker requires model and mode')
        worker(a); return 0
    return run(a)


if __name__=='__main__': raise SystemExit(main())
