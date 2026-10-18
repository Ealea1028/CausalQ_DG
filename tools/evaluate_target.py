"""Phase26 frozen four-model final Cityscapes500 evaluation; no trainer/optimizer."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import uuid

import numpy as np
from PIL import Image
import torch

from causalq.datasets.label_space import read_index_mask
from causalq.datasets.segmentation import _tensorize
from causalq.metrics import MeanIoU
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.formal_source import formal_config, validate_formal_config
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.audit_trained_anchor_stop import audit_stop
from tools.audit_formal_source_pair import audit_pair
from tools.audit_prediction_source_result import audit_prediction
from tools.audit_full_cqe_source_result import audit_full_cqe
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest, precision
from tools.train_strict_source import file_sha

ROOT=Path(__file__).resolve().parents[1]
MANIFEST_SHA='3960bb34afdfbbc6a5a5be887626d7ab03196691de54d0c00eb047153e03ab4c'
FREEZE_PATH=ROOT/'experiments/STRICT_DG_PHASE26/final_evaluation_freeze.json'


def write_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    temporary.write_bytes((json.dumps(value,indent=2)+'\n').encode());os.replace(temporary,path)


def target_ids(path=ROOT/'manifests/cityscapes_val500.txt'):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('Frozen target manifest SHA256 differs')
    ids=raw.decode('ascii').splitlines()
    if len(ids)!=500 or len(set(ids))!=500 or ids!=sorted(ids) or any(not re.fullmatch(r'[a-z]+/[a-z]+_\d{6}_\d{6}',i) for i in ids):
        raise ValueError('Target500 membership differs')
    return ids


def source_report(variant):
    if variant in ('M0','M1'):
        return json.loads((ROOT/f'experiments/STRICT_DG_PHASE22/{variant.lower()}_formal_report.json').read_bytes())
    phase={'P1':23,'M2':24}[variant]
    return json.loads((ROOT/f'experiments/STRICT_DG_PHASE{phase}/formal_report.json').read_bytes())


def frozen_contract():
    audit_stop();audit_pair();audit_prediction();audit_full_cqe()
    freeze=json.loads(FREEZE_PATH.read_bytes())
    if (set(freeze['variants'])!={'M0','M1','P1','M2'} or freeze['target_manifest_sha256']!=MANIFEST_SHA
            or freeze['target_sample_count']!=500 or freeze['optimizer_updates_this_phase']!=0
            or freeze['target_results_may_change_recipe'] is not False or freeze['m3_status']!='not_run_hypothesis_unsupported'):
        raise ValueError('Final target contract differs')
    target_ids()
    for variant,record in freeze['variants'].items():
        source=source_report(variant)
        config=ROOT/record['config_path']
        if (record['checkpoint_sha256']!=source['checkpoint_sha256'] or record['head_sha256']!=source['final_head_sha256']
                or record['checkpoint_iteration']!=40000 or record['producer_git_sha']!=source['metadata']['git_sha']
                or record['config_sha256']!=file_sha(config) or record['config_sha256']!=source['metadata']['config_sha256']):
            raise ValueError('Frozen final model identity differs')
    return freeze


class FinalCityscapes:
    """Read exactly frozen val pairs at native geometry; never resize target GT."""
    def __init__(self,root,ids):
        self.root,self.ids=root,ids
        actual=sorted(p.relative_to(root/'leftImg8bit/val').as_posix().removesuffix('_leftImg8bit.png')
                      for p in (root/'leftImg8bit/val').rglob('*_leftImg8bit.png'))
        if actual!=ids:raise ValueError('Remote Cityscapes val does not exactly match the frozen500 IDs')
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        identity=self.ids[index];city,stem=identity.split('/')
        image_path=self.root/f'leftImg8bit/val/{city}/{stem}_leftImg8bit.png'
        label_path=self.root/f'gtFine/val/{city}/{stem}_gtFine_labelTrainIds.png'
        with Image.open(image_path) as f:image=f.convert('RGB')
        with Image.open(label_path) as f:labels=read_index_mask(f)
        if image.size!=(2048,1024) or labels.shape!=(1024,2048):raise ValueError('Native1024x2048 target geometry required')
        if not np.all(((labels>=0)&(labels<=18))|(labels==255)) or not np.any(labels!=255):raise ValueError('Invalid target trainIds')
        image_tensor,label_tensor=_tensorize(image,Image.fromarray(labels.astype(np.uint8)))
        return {'id':identity,'image':image_tensor,'label':label_tensor}


def evaluate(model,dataset,ids,output,*,device='cuda:0',amp=True):
    if len(dataset)!=len(ids) or not ids or len(set(ids))!=len(ids):raise ValueError('Final evaluation coverage differs')
    before=tree_digest((model.state_dict(),capture_rng()))
    model.eval();metric=MeanIoU(19);independent=np.zeros((19,19),dtype=np.int64)
    trace=output/'target_eval.jsonl'
    with trace.open('x',encoding='utf-8') as stream,torch.inference_mode():
        for index,identity in enumerate(ids):
            sample=dataset[index]
            if sample['id']!=identity or sample['label'].shape!=(1024,2048):raise ValueError('Target ID/order/geometry differs')
            image=sample['image'].unsqueeze(0).to(device);label=sample['label'].unsqueeze(0)
            with precision(device,amp):logits=model(image)
            if logits.shape!=(1,19,1024,2048) or not torch.isfinite(logits).all():raise ValueError('Invalid native target logits')
            prediction=logits.argmax(1).cpu();gt=label.numpy().reshape(-1);pred=prediction.numpy().reshape(-1);valid=gt!=255
            if not valid.any() or np.any((gt[valid]<0)|(gt[valid]>18)):raise ValueError('Invalid target labels')
            counts=np.bincount(gt[valid]*19+pred[valid],minlength=361).reshape(19,19)
            independent+=counts;metric.update(prediction,label)
            stream.write(json.dumps({'id':identity,'prediction_shape':list(prediction.shape),'valid_pixels':int(counts.sum())})+'\n');stream.flush()
            if (index+1)%25==0:print(f'final_target_evaluated={index+1}/{len(ids)}',flush=True)
    if before!=tree_digest((model.state_dict(),capture_rng())):raise ValueError('Read-only target evaluation changed model/RNG')
    if not np.array_equal(metric.confusion.numpy(),independent):raise ValueError('Independent target confusion differs')
    union=independent.sum(0)+independent.sum(1)-independent.diagonal();present=union>0
    ious=np.divide(independent.diagonal(),union,out=np.zeros(19,dtype=float),where=present)
    result=metric.compute()
    if abs(float(ious[present].mean())-result['miou'])>1e-12:raise ValueError('Independent target mIoU differs')
    return {'sample_count':len(ids),'target_miou':result['miou'],'class_iou':result['class_iou'],
            'valid_classes':result['valid_classes'],'confusion_matrix':independent.tolist(),'valid_pixel_count':int(independent.sum()),
            'confusion_counts_exact':True,'model_rng_unchanged':True,'target_labels_optimized':False,
            'trace_sha256':file_sha(trace)}


def validate_result(directory,identity):
    report=json.loads((directory/'metrics.json').read_bytes())
    if (report['ok'] is not True or report['identity']!=identity or report['sample_count']!=500
            or report['target_labels_optimized'] is not False or report['model_rng_unchanged'] is not True
            or report['trace_sha256']!=file_sha(directory/'target_eval.jsonl')):
        raise ValueError('Existing final target result differs')
    counts=np.asarray(report['confusion_matrix'])
    if counts.shape!=(19,19) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or int(counts.sum())!=report['valid_pixel_count']:
        raise ValueError('Existing target confusion counts differ')
    union=counts.sum(0)+counts.sum(1)-counts.diagonal();present=union>0
    iou=np.divide(counts.diagonal(),union,out=np.zeros(19,dtype=float),where=present)
    if abs(float(iou[present].mean())-report['target_miou'])>1e-12:
        raise ValueError('Existing target mIoU differs from confusion')
    return report


def run_variant(args,freeze):
    variant=args.variant;record=freeze['variants'][variant];source=source_report(variant)
    config=formal_config(variant);validate_formal_config(config,variant)
    identity={'git_sha':_git('rev-parse','HEAD'),'variant':variant,'freeze_sha256':file_sha(FREEZE_PATH),
              'checkpoint_sha256':record['checkpoint_sha256'],'target_manifest_sha256':MANIFEST_SHA,
              'geometry':freeze['target_geometry'],'checkpoint_selection':'fixed_final_iteration40000'}
    root=args.output_root/variant;root.mkdir(parents=True,exist_ok=True)
    # Completed results are verified and reused, never reselected after target feedback.
    for done in sorted(root.glob('*/metrics.json')):
        result=validate_result(done.parent,identity);print(json.dumps(result,indent=2),flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Final target evaluation requires AutoDL CUDA')
    if str(torch.__version__)!='2.7.0+cu126' or version('transformers')!='4.56.1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Use frozen runtime/CUBLAS configuration')
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':str(torch.__version__),
                 'cuda':torch.version.cuda,'transformers':version('transformers'),'gpu':torch.cuda.get_device_name(0)}
    if environment!=source['metadata']['environment']:raise ValueError('Final evaluator runtime differs from trained baseline')
    checkpoint=args.training_output_root/record['checkpoint_relative_to_output_root']
    if file_sha(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Frozen final checkpoint differs')
    weights=args.pretrained_root/config['model']['weights_dir']
    if file_sha(weights/'model.safetensors')!=config['model']['weights_sha256']:raise ValueError('Frozen backbone weights differ')
    seed_everything(0);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
    backbone=DINOv3Backbone.from_pretrained(config['model']['backbone'],weights=weights,freeze=True,
        intermediate_indices=config['model']['intermediate_indices'],dtype=torch.bfloat16,local_files_only=True).to('cuda:0')
    if tree_digest(backbone.state_dict())!=BACKBONE_STATE_SHA:raise ValueError('Frozen backbone differs')
    model=QuerySegmentor(backbone,decoder_channels=256,num_classes=19,dropout=.1,queries_per_class=2,
        interaction='static',cross_attention_layers=0,num_heads=8,temperature=.07,alpha_init=0.,centered_residuals=variant!='M0').to('cuda:0').eval()
    payload=load_training_checkpoint(checkpoint)
    if payload.get('schema_version')!=2 or payload['iteration']!=40000 or payload['metadata']!=source['metadata'] or set(payload['trainable_model'])!=set(trainable_state_dict(model)):
        raise ValueError('Frozen checkpoint provenance/coverage differs')
    restored=model.load_state_dict(payload['trainable_model'],strict=False)
    if restored.unexpected_keys or set(restored.missing_keys)!={'backbone.'+k for k in backbone.state_dict()}:raise ValueError('Frozen model restoration differs')
    del payload
    if tree_digest(trainable_state_dict(model))!=record['head_sha256']:raise ValueError('Frozen final head differs')
    ids=target_ids();dataset=FinalCityscapes(args.data_root/'cityscapes',ids)
    attempt=root/uuid.uuid4().hex;attempt.mkdir()
    torch.cuda.reset_peak_memory_stats()
    result=evaluate(model,dataset,ids,attempt)
    if file_sha(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Original final checkpoint changed')
    report={'schema_version':1,'phase':26,'ok':True,'identity':identity,'variant':variant,
            'created_utc':datetime.now(timezone.utc).isoformat(),'environment':environment,'source_miou':record['source_miou'],
            'optimizer_updates_this_run':0,'target_data_accessed':True,'selection_or_tuning_on_target':False,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,**result}
    write_json(attempt/'metrics.json',report);print(json.dumps(report,indent=2),flush=True)


def summarize(args,freeze):
    reports={}
    for variant in freeze['variants']:
        files=list((args.output_root/variant).glob('*/metrics.json'))
        if len(files)!=1:raise ValueError('Need exactly one committed final result per frozen variant')
        r=json.loads(files[0].read_bytes());identity=r['identity']
        if identity['git_sha']!=_git('rev-parse','HEAD') or identity['freeze_sha256']!=file_sha(FREEZE_PATH) or identity['checkpoint_sha256']!=freeze['variants'][variant]['checkpoint_sha256']:
            raise ValueError('Final result does not match frozen model/commit')
        reports[variant]=validate_result(files[0].parent,identity)
    result={'schema_version':1,'phase':26,'ok':True,'git_sha':_git('rev-parse','HEAD'),'freeze_sha256':file_sha(FREEZE_PATH),
            'target_manifest_sha256':MANIFEST_SHA,'target_data_accessed':True,'optimizer_updates_this_run':0,
            'selection_or_tuning_on_target':False,'m3_status':'not_run_hypothesis_unsupported',
            'reports':reports,'target_miou':{v:r['target_miou'] for v,r in reports.items()}}
    write_json(args.output_root/'final_comparison.json',result);print(json.dumps(result,indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('M0','M1','P1','M2','summarize'),required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--training-output-root',type=Path,default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    args=p.parse_args(argv);args.output_root=args.output_root.resolve()
    if any(args.output_root.is_relative_to(x.resolve()) for x in (ROOT,args.data_root,args.pretrained_root)):
        p.error('Final results must be outside code/data/weights')
    if _git('status','--porcelain'):raise ValueError('Final evaluator requires clean exact checkout')
    if 'tools.train' in sys.modules:raise ValueError('Final evaluator must not import legacy trainer')
    freeze=frozen_contract()
    if args.variant=='summarize':summarize(args,freeze)
    else:run_variant(args,freeze)
    return 0


if __name__=='__main__':raise SystemExit(main())
