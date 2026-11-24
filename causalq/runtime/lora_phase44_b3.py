"""Phase44 B3: unchanged B2 router plus mean-view localization weight0.1."""
import hashlib,json
from pathlib import Path
from causalq.runtime.lora_phase44_b2 import b2_head, B2Trainer, references as b2_references
from causalq.runtime.lora_l1_formal import ROOT,DIRECTORY
from causalq.protocols.lora_causalq_1114 import BACKBONES
from causalq.runtime.identity import file_sha
CONFIG='configs/lora_causalq_1114/phase44_b3.json'
FREEZE=DIRECTORY/'phase44_b3_freeze.json'
POLICY=DIRECTORY/'phase44_b3_policy.json'
SPECS={'D3_B3':{'backbone':'dinov3_vitl16','variant':'B3'},
       'D2_B3':{'backbone':'dinov2_vitl14_hpi_p16','variant':'B3'}}
b3_head=b2_head

class B3Trainer(B2Trainer):
    def __init__(self,model,*,horizon=40000):
        super().__init__(model,horizon=horizon)
        # Existing B3 primitive adds0.05 per view. Summed views /8 source images
        # equals0.1 times the mean localization loss over16 image views.
        self.variant='B3'

def load_contract():
    f=json.loads(FREEZE.read_text())
    for p,h in f['files'].items():
        if hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise ValueError('B3 frozen source changed: '+p)
    c=json.loads((ROOT/CONFIG).read_text())
    reference=json.loads((ROOT/'configs/lora_causalq_1114/phase44_b2.json').read_text())
    allowed={'variants','initialization','execution','localization_weight','localization_reduction','localization_per_view_coefficient'}
    if (c['phase']!=44 or c['variants']!=['B3'] or c['localization_weight']!=.1
        or c['localization_per_view_coefficient']!=.05 or c['target_access_authorized']
        or set(c['backbones'])!=set(BACKBONES)
        or any(c.get(k)!=v for k,v in reference.items() if k not in allowed)):
        raise ValueError('B3 recipe changed beyond prescribed localization')
    a=json.loads((DIRECTORY/'phase44_b2_completion_acceptance.json').read_text())
    if not a['accepted'] or a['formal_optimizer_updates_total']!=80000:raise ValueError('B2 not accepted')
    return c,f

def references():
    yield from b2_references()
    a=json.loads((DIRECTORY/'phase44_b2_completion_acceptance.json').read_text())
    for job,pin in a['models'].items():yield 44,a['producer'],job,pin

def verify_references():
    from tools.retire_lora_l0 import safe
    for phase,producer,key,pin in references():
        p=Path(pin['path']);safe(p.parents[3],p);marker=p.parent/'complete.json';m=json.loads(marker.read_text());i=m['identity']
        h=hashlib.sha256(json.dumps(i,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (file_sha(p)!=pin['sha256'] or file_sha(marker)!=pin['marker_sha256'] or h!=pin['identity_sha256']
            or m['step']!=40000 or m['sha256']!=pin['sha256'] or i['phase']!=phase
            or i['variant']!=pin.get('variant',key) or i['training_producer_git_sha']!=producer):
            raise ValueError('Protected previous final changed')
