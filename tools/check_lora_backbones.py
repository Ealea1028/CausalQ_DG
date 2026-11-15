"""Phase39 AutoDL-only synthetic real-weight integration gate; no dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from causalq.models.backbone_factory import build_backbone
from causalq.models.lora import inject_dino_lora
from causalq.models.query_segmentor import QuerySegmentor
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.lora_causalq_1114 import STUDY, require_action


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            d.update(chunk)
    return d.hexdigest()


def check(model_name, weights):
    indices = (5,11,17,23) if model_name.startswith('dinov2') else (6,12,18,24)
    kwargs = {} if model_name.startswith('dinov2') else {'intermediate_indices': indices}
    backbone = build_backbone(model_name, weights=weights, dtype=torch.bfloat16, **kwargs).cuda()
    torch.manual_seed(20261114)
    image = torch.randn(1, 3, 512, 512, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = backbone(image)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    manifest = inject_dino_lora(backbone)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise RuntimeError('Injection changed caller RNG')
    if manifest['trainable_parameters'] != 98304:
        raise RuntimeError('Unexpected LoRA parameter count')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        after = backbone(image)
    for a,b in [(before.patch_tokens, after.patch_tokens), (before.cls_token, after.cls_token)]:
        torch.testing.assert_close(a,b, atol=0, rtol=0)
    if after.cls_token.shape != (1,1024) or after.patch_tokens.shape != (1,1024,1024) or len(after.intermediate_maps)!=4:
        raise RuntimeError('Unexpected Patch/CLS or intermediate shapes')
    del before, after
    model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1,
                          queries_per_class=2, interaction='static', cross_attention_layers=0).cuda()
    frozen_before = frozen_digest(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    label = torch.randint(0,19,(1,512,512),device='cuda')
    losses = []
    gradients = {}
    for step in range(3):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            original = model.forward_components(image, include_anchor=True, component_protocol=STUDY)
            reference = original.anchor_scaled_delta_logits.detach().float()
            ce = torch.nn.functional.cross_entropy(original.logits.float(), label)
        ce.backward()
        del original
        with torch.autocast('cuda', dtype=torch.bfloat16):
            photo = model.forward_components(image*.8, include_anchor=True, component_protocol=STUDY)
            loss = torch.nn.functional.cross_entropy(photo.logits.float(), label)
            cqe = causal_query_effect_loss(photo.anchor_scaled_delta_logits.float(), reference, label)
        (loss+cqe).backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        gradients = {n: float(p.grad.float().norm()) for n,p in backbone.named_parameters() if p.requires_grad and p.grad is not None}
        if not gradients or not any(v>0 for n,v in gradients.items() if n.endswith('.B')):
            raise RuntimeError('LoRA B gradient is disconnected')
        if any(p.grad is not None for p in backbone.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen backbone received gradients')
        opt.step()
        losses.append({'ce_original':float(ce.detach()),'ce_photo':float(loss.detach()),'cqe':float(cqe.detach()),'grad_norm':float(norm)})
        del photo, loss, cqe, ce, reference
    if not any(v>0 for n,v in gradients.items() if n.endswith('.A')):
        raise RuntimeError('LoRA A gradient is disconnected after engineering updates')
    if frozen_digest(model) != frozen_before:
        raise RuntimeError('Base weights changed')
    # Native target-shaped random tensor, never a target image or target GT.
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        native = backbone(torch.zeros(1,3,1024,2048,device='cuda',dtype=torch.bfloat16))
    if native.patch_tokens.shape != (1,8192,1024) or native.cls_token.shape != (1,1024):
        raise RuntimeError('Native non-square feature shape mismatch')
    identity_files = sorted(weights.glob('*.safetensors')) + sorted(weights.glob('*.json')) if weights.is_dir() else [weights]
    if not identity_files:
        raise RuntimeError('No weight identity files')
    return {'model':model_name, 'weights':str(weights), 'weight_files':{p.name:file_digest(p) for p in identity_files},
            'lora':manifest,'zero_delta_exact':True,'frozen_digest':frozen_before,
            'engineering_updates':3,'losses':losses,'final_adapter_gradient_norms':gradients,
            'native_patch_shape':list(native.patch_tokens.shape),'native_cls_shape':list(native.cls_token.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', default='backbone-gate')
    parser.add_argument('--pretrained-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require_action(args.action)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'ok':False, 'study':STUDY,'phase':39,'action':args.action,'formal_optimizer_updates':0,
              'dataset_accessed':False,'target_data_accessed':False,'models':[]}
    started = time.monotonic()
    try:
        report['git_sha'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('Tracked source modifications are forbidden')
        if not torch.cuda.is_available():
            raise RuntimeError('This gate must run on AutoDL CUDA')
        report['environment'] = {'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),
                                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                                 'visible_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
        import transformers
        report['environment']['transformers'] = transformers.__version__
        report['pip_freeze'] = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
        torch.cuda.reset_peak_memory_stats()
        for name,path in [('dinov3_vitl16',args.pretrained_root/'dinov3_vitl16'),
                          ('dinov2_vitl14_hpi_p16',args.pretrained_root/'dinov2_vitl14'/'dinov2_vitl14_hpi_p16.pth')]:
            report['models'].append(check(name,path))
            (args.output/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            torch.cuda.empty_cache()
        report['ok'] = True
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        report['seconds'] = time.monotonic()-started
        report['exit_code'] = 0 if report['ok'] else 1
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
