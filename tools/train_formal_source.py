"""Formal source-only M0/M1/P1/M2 with immutable1000-update recovery segments."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import torch
import yaml

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, source_manifest_metadata, load_membership
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.formal_source import ROOT, validate_formal_config
from causalq.utils.checkpoint import trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.deterministic_source_ce import BACKEND
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_runner import train_segment, evaluate_without_training_side_effects
from tools.strict_source_smoke import tree_digest, restore_compact
from tools.train_strict_source import file_sha

HORIZON = 40000
INTERVAL = 1000
POLICY = {'m1_resume_readiness': 'user_assumed_passed', 'm1_cuda_exact_resume_measured_this_release': False,
          'authority': 'user requested formal training and waived repeated readiness gates',
          'new_loss_enabled': False, 'target_data_accessed': False}


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_bytes((json.dumps(value, indent=2) + '\n').encode())
    os.replace(temporary, path)


def completed_segments(root, metadata, *, horizon=HORIZON, interval=INTERVAL):
    """Only a hash-verified committed segment advances the recovery cursor."""
    segments = []
    for marker in sorted(root.glob('segments/*/complete.json')):
        record = json.loads(marker.read_bytes())
        if record['metadata'] != metadata:
            raise ValueError('Recovery metadata differs')
        if set(record['files_sha256']) != {'train.jsonl', 'state.pth'}:
            raise ValueError('Recovery artifact coverage differs')
        for name, digest in record['files_sha256'].items():
            if file_sha(marker.parent / name) != digest:
                raise ValueError('Recovery artifact hash differs')
        segments.append((record, marker.parent))
    segments.sort(key=lambda item: item[0]['initial_iteration'])
    cursor = 0
    for record, _ in segments:
        stop = min(cursor + interval, horizon)
        if (record['initial_iteration'] != cursor or record['final_iteration'] != stop
                or record['optimizer_updates'] != stop - cursor or record['backbone_unchanged'] is not True):
            raise ValueError('Recovery segments overlap, have gaps or wrong budgets')
        cursor = stop
    return segments


def run_segments(model, optimizer, dataset, style, config, root, metadata, *, horizon=HORIZON,
                 interval=INTERVAL, device='cuda:0', amp=True, update_fn=None):
    segments = completed_segments(root, metadata, horizon=horizon, interval=interval)
    start = segments[-1][0]['final_iteration'] if segments else 0
    resume = segments[-1][1] / 'state.pth' if segments else None
    while start < horizon:
        stop = min(start + interval, horizon)
        output = root / 'segments' / f'{start:05d}_{stop:05d}_{uuid.uuid4().hex}'
        output.mkdir(parents=True)
        result = train_segment(model, optimizer, dataset, style, config, output,
            start=start, stop=stop, metadata=metadata, resume=resume, device=device, amp=amp, update_fn=update_fn)
        record = {**result, 'metadata': metadata,
                  'files_sha256': {name: file_sha(output / name) for name in ('train.jsonl', 'state.pth')}}
        write_json(output / 'complete.json', record)
        segments.append((record, output))
        start, resume = stop, output / 'state.pth'
        print(f'formal_checkpoint_committed={start}/{horizon} path={resume}', flush=True)
    # Also restores the final trained head when resuming directly into evaluation.
    restore_compact(model, optimizer, resume)
    return segments


def run_variant(args):
    variant = args.variant
    config_path = ROOT / f'configs/strict_dg/formal/{variant.lower()}_seed0.yaml'
    config = yaml.safe_load(config_path.read_bytes())
    validate_formal_config(config, variant)
    policy = POLICY
    baseline_audit = None
    update_fn = None
    if variant == 'P1':
        from tools.audit_formal_source_pair import audit_pair
        from tools.prediction_source_update import prediction_source_update
        baseline_audit = audit_pair()
        policy = {**POLICY, 'new_loss_enabled': True, 'cqe_enabled': False,
                  'prediction_consistency': config['prediction_consistency']}
        update_fn = prediction_source_update
    if variant == 'M2':
        from tools.audit_formal_source_pair import audit_pair
        from tools.full_cqe_source_update import full_cqe_source_update
        baseline_audit = audit_pair()
        policy = {**POLICY, 'new_loss_enabled': True, 'cqe_enabled': True,
                  'prediction_consistency_enabled': False, 'effect_definition': 'alpha_times_full_query_delta_pre_addition',
                  'causal_query_effect': config['causal_query_effect']}
        update_fn = full_cqe_source_update
    if variant == 'M3':
        from tools.audit_formal_source_pair import audit_pair
        from tools.full_cqe_source_update import anchor_cqe_source_update
        from tools.trained_anchor_stability import validate_stability
        if args.stability_report is None:
            raise ValueError('M3 requires trained M1 raw-stability evidence under plan section46')
        evidence = validate_stability(args.stability_report, _git('rev-parse', 'HEAD'))
        if evidence['decision']['m3_training_authorized'] is not True:
            raise ValueError('M3 forbidden: trained M1 raw anchor-stability hypothesis unsupported')
        baseline_audit = audit_pair()
        policy = {**POLICY, 'new_loss_enabled': True, 'cqe_enabled': True, 'anchor_cqe_enabled': True,
                  'prediction_consistency_enabled': False, 'effect_definition': 'alpha_times_anchor_only_delta_pre_addition',
                  'causal_query_effect': config['causal_query_effect'],
                  'stability_report_sha256': file_sha(args.stability_report), 'stability_decision': evidence['decision']}
        update_fn = anchor_cqe_source_update
    git_sha = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain'):
        raise ValueError('Formal training requires clean exact checkout')
    if 'tools.train' in sys.modules or 'causalq.datasets.cityscapes' in sys.modules:
        raise ValueError('Legacy/target module imported')
    if not torch.cuda.is_available():
        raise RuntimeError('Formal training requires AutoDL CUDA')
    if (str(torch.__version__) != '2.7.0+cu126' or version('transformers') != '4.56.1'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('Use pinned runtime and CUBLAS_WORKSPACE_CONFIG=:4096:8')
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'transformers': version('transformers'),
        'gpu': torch.cuda.get_device_name(0)}
    if baseline_audit is not None and environment != baseline_audit['environment']:
        raise ValueError('Formal control environment differs from accepted matched baseline runtime')
    metadata = {'git_sha': git_sha, 'variant': variant, 'seed': 0, 'config_sha256': file_sha(config_path),
        'config_tree_sha256': tree_digest(config), 'environment': environment,
        **source_manifest_metadata(), **policy, 'formal_training_authorized': True,
        'schedule_horizon': HORIZON, 'checkpoint_interval': INTERVAL, 'checkpoint_selection': 'fixed_final',
        'centered_residuals': variant != 'M0', 'loss_backend': BACKEND, 'sample_workers': 0,
        'amp': 'bfloat16', 'deterministic_algorithms': True, 'matmul_allow_tf32': True,
        'cudnn_benchmark': False, 'cublas_workspace_config': ':4096:8',
        'backbone_weights_sha256': config['model']['weights_sha256'], 'backbone_sha256': BACKBONE_STATE_SHA}
    if baseline_audit is not None:
        metadata['baseline_audit'] = baseline_audit
    root = args.output_root / variant
    root.mkdir(parents=True, exist_ok=True)
    identity = root / 'metadata.json'
    if identity.exists():
        if json.loads(identity.read_bytes()) != metadata or file_sha(root / 'config.yaml') != metadata['config_sha256']:
            raise ValueError('Formal run identity/config differs; refusing recovery')
    else:
        (root / 'config.yaml').write_bytes(config_path.read_bytes())
        write_json(identity, metadata)
    if (root / 'formal_report.json').exists():
        validate_final_report(root, metadata)
        print(f'{variant}: formal run already complete; preserving result', flush=True)
        return
    seed_everything(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    weights = args.pretrained_root / config['model']['weights_dir']
    if file_sha(weights / 'model.safetensors') != config['model']['weights_sha256']:
        raise ValueError('Backbone weight SHA differs')
    transform = TrainTransform(config['train']['crop_size'], scale_range=config['data']['random_scale'],
        horizontal_flip_probability=config['data']['horizontal_flip_probability'],
        min_valid_fraction=config['data']['min_valid_fraction'], crop_attempts=config['data']['crop_attempts'])
    train, validation = build_source_datasets(args.data_root / 'gta5', transform, SourceEvalTransform())
    backbone = DINOv3Backbone.from_pretrained(config['model']['backbone'], weights=weights, freeze=True,
        intermediate_indices=config['model']['intermediate_indices'], dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
    if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA:
        raise ValueError('Frozen backbone differs')
    model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1, queries_per_class=2,
        interaction='static', cross_attention_layers=0, num_heads=8, temperature=.07, alpha_init=0.,
        centered_residuals=variant != 'M0').to('cuda:0')
    initial_head = tree_digest(trainable_state_dict(model))
    if baseline_audit is not None and initial_head != baseline_audit['initial_head_sha256']:
        raise ValueError('Formal control fresh initialization differs from accepted M0/M1')
    initial = root / 'initialization.json'
    if initial.exists() and json.loads(initial.read_bytes())['head_sha256'] != initial_head:
        raise ValueError('Fresh seed0 initialization differs')
    if not initial.exists():
        write_json(initial, {'head_sha256': initial_head, 'fresh_seed': 0, 'promoted_pilot_checkpoint': False})
    if variant == 'M1':
        baseline = args.output_root / 'M0'
        if ((baseline / 'initialization.json').exists()
                and json.loads((baseline / 'initialization.json').read_bytes())['head_sha256'] != initial_head):
            raise ValueError('M1 fresh initialization differs from paired M0')
        if ((baseline / 'metadata.json').exists()
                and json.loads((baseline / 'metadata.json').read_bytes())['environment'] != environment):
            raise ValueError('M1 environment differs from paired M0')
    style = StyleInterventionBank(**{k: tuple(v) if isinstance(v, list) else v
        for k, v in config['style']['photometric'].items()}).to('cuda:0')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    segments = run_segments(model, optimizer, train, style, config, root, metadata, update_fn=update_fn)
    output = root / 'evaluations' / uuid.uuid4().hex
    output.mkdir(parents=True)
    _, ids = load_membership()
    evaluation = evaluate_without_training_side_effects(model, optimizer, validation, ids, output)
    final = segments[-1][1] / 'state.pth'
    report = {'schema_version': 1, 'phase': config['experiment']['phase'], 'ok': True, 'variant': variant, 'metadata': metadata,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'formal_training_authorized': True,
        'formal_optimizer_updates': HORIZON, 'checkpoint_selection': 'fixed_final',
        'initial_head_sha256': initial_head, 'final_head_sha256': tree_digest(trainable_state_dict(model)),
        'checkpoint': str(final.relative_to(root)), 'checkpoint_sha256': file_sha(final),
        'source_evaluation': evaluation, 'source_miou': evaluation['source_miou'],
        'evaluation_trace': str((output / 'source_eval.jsonl').relative_to(root)),
        'peak_reserved_gib_this_process': torch.cuda.max_memory_reserved() / 1024**3,
        'target_miou': None, **policy}
    write_json(root / 'formal_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def validate_final_report(root, metadata):
    report = json.loads((root / 'formal_report.json').read_bytes())
    segments = completed_segments(root, metadata)
    if (report['ok'] is not True or report['metadata'] != metadata or report['formal_optimizer_updates'] != HORIZON
            or not segments or segments[-1][0]['final_iteration'] != HORIZON
            or report['checkpoint'] != str((segments[-1][1] / 'state.pth').relative_to(root))
            or report['checkpoint_sha256'] != file_sha(root / report['checkpoint'])
            or report['source_evaluation']['sample_count'] != 2497
            or report['target_data_accessed'] is not False or report['target_miou'] is not None
            or file_sha(root / report['evaluation_trace']) != report['source_evaluation']['trace_sha256']):
        raise ValueError('Final formal result/checkpoint/evaluation differs')
    return report


def summarize(root):
    reports = {}
    for variant in ('M0', 'M1'):
        directory = root / variant
        metadata = json.loads((directory / 'metadata.json').read_bytes())
        config = yaml.safe_load((directory / 'config.yaml').read_bytes())
        validate_formal_config(config, variant)
        if (metadata['git_sha'] != _git('rev-parse', 'HEAD')
                or metadata['config_sha256'] != file_sha(directory / 'config.yaml')):
            raise ValueError('Pair identity differs from exact checkout')
        reports[variant] = validate_final_report(directory, metadata)
    if reports['M0']['initial_head_sha256'] != reports['M1']['initial_head_sha256']:
        raise ValueError('Matched formal pair did not use identical fresh head initialization')
    if reports['M0']['metadata']['environment'] != reports['M1']['metadata']['environment']:
        raise ValueError('Matched formal pair environment differs')
    result = {'schema_version': 1, 'phase': 22, 'ok': True, 'git_sha': _git('rev-parse', 'HEAD'),
        'seed': 0, 'optimizer_updates_per_variant': HORIZON, 'formal_training_authorized': True,
        'source_miou': {v: r['source_miou'] for v, r in reports.items()},
        'source_miou_delta_M1_minus_M0': reports['M1']['source_miou'] - reports['M0']['source_miou'],
        'target_miou': None, 'reports': reports, **POLICY}
    write_json(root / 'pair_report.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('M0', 'M1', 'P1', 'M2', 'M3', 'summarize'), required=True)
    parser.add_argument('--stability-report', type=Path, help='Required positive formal M1 raw-stability evidence for M3')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if any(args.output_root.is_relative_to(p.resolve()) for p in (ROOT, args.data_root, args.pretrained_root)):
        parser.error('Formal output must be outside source/data/weights')
    if args.variant == 'summarize':
        summarize(args.output_root)
    else:
        run_variant(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
