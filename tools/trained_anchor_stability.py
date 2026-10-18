"""Plan section46 prerequisite on the formal M1 final model; no optimizer."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys

import torch

from causalq.analysis.query_components import intervention_seed
from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, load_membership
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.formal_source import formal_config
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.audit_formal_source_pair import ROOT, PAIR_SHA, audit_pair
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest, precision
from tools.train_strict_source import file_sha


def decision(rows):
    if not rows or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Stability requires distinct source rows')
    values = {}
    for component in ('anchor', 'residual'):
        arrays = [r[component + '_variance_by_class'] for r in rows]
        if any(len(a) != 19 or any(not math.isfinite(x) or x < 0 for x in a) for a in arrays):
            raise ValueError('Stability needs finite nonnegative19-class raw variances')
        values[component] = math.fsum(math.fsum(a) / 19 for a in arrays) / len(rows)
    a, r = values['anchor'], values['residual']
    return {'raw_anchor_variance': a, 'raw_residual_variance': r,
            'raw_anchor_residual_ratio': a / r if r > 0 else None,
            'images_anchor_lower': sum(math.fsum(x['anchor_variance_by_class']) < math.fsum(x['residual_variance_by_class']) for x in rows),
            'm3_training_authorized': 0 < a < r,
            'rule': 'strict 0 < raw_anchor_variance < raw_residual_variance; equal image/class macro; no normalized substitution'}


@torch.no_grad()
def measure_pair(model, image, label, style, seed):
    valid = label[0] != 255
    if not valid.any() or model.training or image.shape[0] != 1:
        raise ValueError('Stability requires eval mode,one image and valid source labels')
    devices = [image.device.index or 0] if image.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        photo = style.photometric_view(image)
        maps = []
        for view in (image, photo):
            with precision(str(image.device), image.is_cuda):
                output = model.forward_components(view, include_anchor=True)
            maps.append({'anchor': model.get_anchor_query_effect(output=output),
                         'residual': model.get_residual_contribution(output=output)})
        result = {'valid_pixels': int(valid.sum())}
        for component in ('anchor', 'residual'):
            x, y = [m[component][0, :, valid].float() for m in maps]
            variance = ((x - y).square() / 4).mean(-1)
            if not torch.isfinite(variance).all(): raise ValueError('Nonfinite raw variance')
            result[component + '_variance_by_class'] = variance.cpu().tolist()
            result[component + '_original_rms'] = x.square().mean(-1).sqrt().cpu().tolist()
            result[component + '_photo_rms'] = y.square().mean(-1).sqrt().cpu().tolist()
        return result


def accepted_m1():
    audit_pair()
    return json.loads((ROOT / 'experiments/STRICT_DG_PHASE22/m1_formal_report.json').read_bytes())


def validate_stability(path, git_sha):
    report = json.loads(path.read_bytes())
    original = accepted_m1()
    if (report.get('ok') is not True or report['git_sha'] != git_sha or report['phase'] != 25
            or report['checkpoint_sha256'] != original['checkpoint_sha256'] or report['checkpoint_training_updates'] != 40000
            or report['baseline_pair_report_sha256'] != PAIR_SHA or report['sample_count'] != 2497
            or report['target_data_accessed'] is not False or report['optimizer_updates_this_run'] != 0
            or report['read_only_state_unchanged'] is not True or report['backbone_forward_calls'] != 4994
            or file_sha(path.parent / 'sample_index.jsonl') != report['sample_index_sha256']):
        raise ValueError('Trained M1 stability identity/coverage differs')
    rows = [json.loads(line) for line in (path.parent / 'sample_index.jsonl').read_text().splitlines()]
    _, ids = load_membership()
    if [r['source_id'] for r in rows] != ids or decision(rows) != report['decision']:
        raise ValueError('Stability sample coverage/raw decision differs')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG/phase22/M0_M1_FORMAL_SEED0_2537653/M1'))
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(x.resolve()) for x in (ROOT, args.baseline_root, args.data_root, args.pretrained_root)):
        p.error('Stability output must be outside original runs/code/data/weights')
    git_sha = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain'): raise ValueError('Stability needs clean exact checkout')
    report_path = output / 'stability_report.json'
    if report_path.exists():
        validate_stability(report_path, git_sha)
        print('Existing complete trained M1 stability result retained', flush=True)
        return 0
    original = accepted_m1()
    config = formal_config('M1')
    if 'tools.train' in sys.modules or 'causalq.datasets.cityscapes' in sys.modules:
        raise ValueError('Legacy/target module imported')
    if not torch.cuda.is_available(): raise RuntimeError('Formal M1 stability requires AutoDL CUDA')
    if str(torch.__version__) != '2.7.0+cu126' or version('transformers') != '4.56.1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Use pinned CUDA/runtime/CUBLAS configuration')
    environment = {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'transformers':version('transformers'),
        'gpu':torch.cuda.get_device_name(0)}
    if environment != original['metadata']['environment']: raise ValueError('Use accepted matched baseline runtime')
    checkpoint = args.baseline_root / original['checkpoint']
    if file_sha(checkpoint) != original['checkpoint_sha256']: raise ValueError('Formal M1 checkpoint differs')
    seed_everything(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    weights = args.pretrained_root / config['model']['weights_dir']
    if file_sha(weights / 'model.safetensors') != config['model']['weights_sha256']: raise ValueError('Frozen weights differ')
    backbone = DINOv3Backbone.from_pretrained(config['model']['backbone'], weights=weights, freeze=True,
        intermediate_indices=config['model']['intermediate_indices'], dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
    if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA: raise ValueError('Frozen backbone differs')
    model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1, queries_per_class=2,
        interaction='static', cross_attention_layers=0, num_heads=8, temperature=.07, alpha_init=0., centered_residuals=True).to('cuda:0').eval()
    payload = load_training_checkpoint(checkpoint)
    if (payload.get('schema_version') != 2 or payload['iteration'] != 40000 or payload['metadata'] != original['metadata']
            or set(payload['trainable_model']) != set(trainable_state_dict(model))):
        raise ValueError('M1 checkpoint protocol/parameter coverage differs')
    incompatible = model.load_state_dict(payload['trainable_model'], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {'backbone.' + k for k in backbone.state_dict()}:
        raise ValueError('M1 checkpoint restore coverage differs')
    del payload
    if tree_digest(trainable_state_dict(model)) != original['final_head_sha256']: raise ValueError('M1 final head differs')
    _, holdout = build_source_datasets(args.data_root / 'gta5', TrainTransform((512,512)), SourceEvalTransform())
    _, ids = load_membership()
    style = StyleInterventionBank(**{k: tuple(v) if isinstance(v,list) else v for k,v in config['style']['photometric'].items()}).to('cuda:0')
    before = tree_digest((model.state_dict(), style.state_dict(), capture_rng()))
    output.mkdir(parents=True, exist_ok=True)
    calls = [0]
    hook = backbone.register_forward_hook(lambda *unused: calls.__setitem__(0, calls[0]+1))
    rows = []
    try:
        # A failed partial census is safely recomputed; the frozen model is never updated.
        with (output / 'sample_index.jsonl').open('w', encoding='utf-8') as stream:
            for index, source_id in enumerate(ids):
                sample = holdout[index]
                if sample['id'] != source_id: raise ValueError('Source holdout order differs')
                image,label = [sample[k].unsqueeze(0).to('cuda:0') for k in ('image','label')]
                row = {'source_id': source_id, 'intervention_seed': intervention_seed(source_id),
                       **measure_pair(model,image,label,style,intervention_seed(source_id))}
                rows.append(row); stream.write(json.dumps(row)+'\n')
                if (index+1)%250 == 0: print(f'trained_m1_stability={index+1}/2497',flush=True)
    finally:
        hook.remove()
    if before != tree_digest((model.state_dict(), style.state_dict(), capture_rng())) or file_sha(checkpoint) != original['checkpoint_sha256']:
        raise ValueError('Read-only M1 census changed state/checkpoint')
    report = {'schema_version':1,'phase':25,'ok':True,'git_sha':git_sha,
        'created_utc':datetime.now(timezone.utc).isoformat(),'environment':environment,
        'baseline_pair_report_sha256':PAIR_SHA,'checkpoint_sha256':original['checkpoint_sha256'],
        'checkpoint_training_updates':40000,'optimizer_updates_this_run':0,'sample_count':len(rows),
        'backbone_forward_calls':calls[0],'read_only_state_unchanged':True,'target_data_accessed':False,
        'sample_index_sha256':file_sha(output/'sample_index.jsonl'),'decision':decision(rows),
        'inference_limit':'Descriptive single-seed,one-paired-style source census; no independence,DG utility or semantic proof'}
    from tools.train_formal_source import write_json
    write_json(report_path,report)
    validate_stability(report_path,git_sha)
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
