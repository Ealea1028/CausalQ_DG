"""Phase44 second module: anchor-routed B2 on both accepted DINO backbones."""
import torch
from causalq.models.lora_routing import LoRARoutedQuerySegmentor
from causalq.models.hpi_query_route import QueryFeatureRouter
import hashlib
import json
from pathlib import Path
from causalq.runtime.lora_l1_formal import DIRECTORY, ROOT, L1Trainer, core_l1_head
from causalq.models.hpi_anchor_only import freeze_query_residuals, verify_frozen_residuals
from causalq.protocols.lora_causalq_1114 import BACKBONES
from causalq.runtime.identity import file_sha
CONFIG = 'configs/lora_causalq_1114/phase44_b2.json'
FREEZE = DIRECTORY/'phase44_b2_freeze.json'
POLICY = DIRECTORY/'phase44_b2_policy.json'
SPECS = {'D3_B2': {'backbone':'dinov3_vitl16','variant':'B2'},
         'D2_B2': {'backbone':'dinov2_vitl14_hpi_p16','variant':'B2'}}

def b2_head(backbone, **kwargs):
    model = freeze_query_residuals(core_l1_head(backbone, 'U0', head_type=LoRARoutedQuerySegmentor, **kwargs))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261031)
        model.query_router = QueryFeatureRouter(backbone.hidden_size).to(next(backbone.parameters()).device)
    return model

class B2Trainer(L1Trainer):
    def __init__(self, model, *, horizon=40000):
        verify_frozen_residuals(model)
        super().__init__(model, 'U0', horizon=horizon)
    def check_base(self):
        super().check_base()
        verify_frozen_residuals(self.model)

def load_contract():
    freeze=json.loads(FREEZE.read_text())
    for n,h in freeze['files'].items():
        if hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise ValueError('Phase44 B2 source changed: '+n)
    cfg=json.loads((ROOT/CONFIG).read_text())
    b1=json.loads((DIRECTORY/'phase44_b1_completion_acceptance.json').read_text())
    if not b1['accepted'] or b1['formal_optimizer_updates_total']!=80000:
        raise ValueError('B1 not accepted')
    for phase in (42,43):
        a=json.loads((DIRECTORY/f'phase{phase}_completion_acceptance.json').read_text())
        if not a['accepted'] or a['formal_optimizer_updates_total']!=160000:
            raise ValueError('Core cohort not accepted')
    integrated=json.loads((DIRECTORY/'phase39_report.json').read_text())['models']
    if (cfg['phase']!=44 or cfg['variants']!=['B2'] or set(cfg['backbones'])!=set(BACKBONES)
        or cfg['formal_steps']!=40000 or not cfg['lora_enabled'] or cfg['target_access_authorized']
        or cfg['effective_source_batch']!=8 or cfg['physical_source_batch']!=4
        or cfg.get('router_seed')!=20261031 or cfg.get('localization_weight')!=0 or not cfg['frozen_zero_residual'] or cfg['query']['anchor_cqe_variants']!=[]):
        raise ValueError('B2 contract changed')
    if any(sum(m['model']==n for m in integrated)!=1 for n in BACKBONES):
        raise ValueError('Backbone missing from Phase39 integration')
    return cfg,freeze

def references():
    for phase in (42,43):
        a=json.loads((DIRECTORY/f'phase{phase}_completion_acceptance.json').read_text())
        for v,pin in a['models'].items(): yield phase,a['producer'],v,pin
    a=json.loads((DIRECTORY/'phase44_b1_completion_acceptance.json').read_text())
    for v,pin in a['models'].items(): yield 44,a['producer'],v,pin

def verify_references():
    from tools.retire_lora_l0 import safe
    for phase,producer,v,pin in references():
        path=Path(pin['path']);root=path.parents[3];safe(root,path)
        marker=path.parent/'complete.json';m=json.loads(marker.read_text());i=m['identity']
        digest=hashlib.sha256(json.dumps(i,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (file_sha(path)!=pin['sha256'] or file_sha(marker)!=pin['marker_sha256']
            or digest!=pin['identity_sha256'] or m['step']!=40000 or m['sha256']!=pin['sha256']
            or i['phase']!=phase or i['variant']!=pin.get('variant',v) or i['training_producer_git_sha']!=producer):
            raise ValueError('Protected core final changed')
