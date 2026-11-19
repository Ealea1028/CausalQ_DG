"""Phase43 contract: fresh DINOv2 core LoRA models after accepted D3 finals."""
import hashlib
import json
from pathlib import Path

from causalq.runtime.identity import file_sha
from causalq.runtime.lora_l1_formal import DIRECTORY, ROOT, VARIANTS

PHASE = 43
NAME = 'dinov2_vitl14'
CONFIG = 'configs/lora_causalq_1114/phase43_core_d2_l1.json'
FREEZE = DIRECTORY/'phase43_freeze.json'
POLICY = DIRECTORY/'phase43_parallel_policy.json'
ACCEPTANCE = DIRECTORY/'phase42_completion_acceptance.json'


def completion():
    result = json.loads(ACCEPTANCE.read_text())
    if (result['accepted'] is not True or result['phase'] != 42
            or result['formal_optimizer_updates_total'] != 160000
            or set(result['models']) != set(VARIANTS)
            or any(m['step'] != 40000 or m['source_cursor'] != 320000 for m in result['models'].values())
            or result['target_data_accessed'] is not False):
        raise ValueError('Phase42 completion not accepted')
    return result


def load_contract():
    accepted = completion()
    freeze = json.loads(FREEZE.read_text())
    for name, digest in freeze['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Phase43 frozen source changed: '+name)
    config = json.loads((ROOT/CONFIG).read_text())
    reference = json.loads((ROOT/'configs/lora_causalq_1114/phase42_vgpu48.json').read_text())
    allowed_changes = {'phase', 'backbones', 'initialization', 'execution', 'performance_policy'}
    if (config['phase'] != PHASE or config['backbones'] != [NAME]
            or config['variants'] != list(VARIANTS) or config['lora_enabled'] is not True
            or config['formal_steps'] != 40000 or config['target_access_authorized'] is not False
            or set(config) != set(reference)
            or any(config[k] != reference[k] for k in reference if k not in allowed_changes)
            or accepted['runtime']['visible_memory_bytes'] != json.loads(POLICY.read_text())['visible_memory_bytes']):
        raise ValueError('Phase43 scope/scientific recipe changed')
    return config, freeze


def check_runtime(actual):
    expected = completion()['runtime']
    if actual != expected:
        raise ValueError('Accepted Phase43 runtime changed: '+repr(
            sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))))


def reference_root():
    roots = {Path(row['path']).parents[3] for row in completion()['models'].values()}
    if len(roots) != 1: raise ValueError('Ambiguous Phase42 reference root')
    return roots.pop()


def verify_references(root=None, *, accepted=None):
    """Hash only; never load the trained D3 heads into D2 or modify a reference."""
    from tools.retire_lora_l0 import safe
    accepted = completion() if accepted is None else accepted
    root = reference_root() if root is None else Path(root)
    for variant, pin in accepted['models'].items():
        folder = safe(root, root/variant/'checkpoints/00040000')
        marker_path = safe(root, folder/'complete.json')
        payload = safe(root, folder/'state.pt')
        marker = json.loads(marker_path.read_text())
        identity = marker['identity']
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (file_sha(marker_path) != pin['marker_sha256'] or file_sha(payload) != pin['sha256']
                or marker['step'] != 40000 or marker['sha256'] != pin['sha256']
                or identity_sha != pin['identity_sha256'] or identity['variant'] != variant
                or identity['phase'] != 42 or identity['backbone'] != 'dinov3_vitl16'
                or identity['training_producer_git_sha'] != accepted['producer']):
            raise ValueError('Protected Phase42 final changed: '+variant)
