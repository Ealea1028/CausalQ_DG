"""Phase21 fixed M1 schema2 runner continuation gate; no formal training."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

import torch
import yaml

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, source_manifest_metadata
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.centered_query import validate_centered_config
from causalq.utils.checkpoint import load_training_checkpoint
from causalq.utils.seed import seed_everything
from tools.audit_query_census_completion import audit_completed_diagnostics
from tools.audit_strict_source_pilot import audit_pilot
from tools.audit_strict_source_smoke import BACKBONE_STATE_SHA
from tools.check_centered_query import centered_diagnostics
from tools.deterministic_source_ce import BACKEND, cuda_preflight
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_resume import compare_states
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest
from tools.train_strict_source import file_sha

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = {'continuous': (0, 8), 'prefix': (0, 4), 'resume': (4, 8)}


def require_centered_model(model, metadata):
    if (not model.query_bank.centered_residuals or model.interaction != 'static' or model.learned_null
            or metadata.get('centered_residuals') is not True or metadata.get('schedule_horizon') != 40000):
        raise ValueError('M1 continuation requires explicit centered Static semantics and the40k schedule')


def verify_workers(root, git_sha):
    workers = {}
    for mode, bounds in BOUNDS.items():
        directory = root / mode
        worker = json.loads((directory / 'worker_report.json').read_bytes())
        if (worker.get('ok') is not True or worker['git_sha'] != git_sha or worker['git_status'] or worker['mode'] != mode
                or worker['formal_training_authorized'] is not False or worker['target_data_accessed'] is not False
                or worker['checkpoint_schema_version'] != 2 or worker['centered_residuals'] is not True
                or (worker['initial_iteration'], worker['final_iteration']) != bounds
                or worker['optimizer_updates'] != bounds[1] - bounds[0] or worker['backbone_unchanged'] is not True):
            raise ValueError(f'M1 continuation worker/budget mismatch: {mode}')
        if set(worker['files_sha256']) != {'train.jsonl', 'state.pth'}:
            raise ValueError('M1 worker artifact coverage differs')
        for name, digest in worker['files_sha256'].items():
            if file_sha(directory / name) != digest:
                raise ValueError(f'M1 worker artifact hash differs: {mode}/{name}')
        payload = load_training_checkpoint(directory / 'state.pth')
        if (payload.get('schema_version') != 2 or payload['metadata'] != worker['metadata']
                or payload['iteration'] != bounds[1] or payload['metadata'].get('centered_residuals') is not True):
            raise ValueError('M1 checkpoint schema/semantic identity differs')
        if payload['metadata'].get('schedule_horizon') != 40000 or payload['metadata'].get('seed') != 0:
            raise ValueError('M1 checkpoint schedule/seed differs')
        workers[mode] = worker
    if len({w['process_id'] for w in workers.values()}) != 3:
        raise ValueError('M1 continuation requires three independent worker processes')
    if any(w['metadata'] != workers['continuous']['metadata'] for w in workers.values()):
        raise ValueError('M1 worker metadata differs')
    comparison = compare_states(root / 'continuous', root / 'prefix', root / 'resume')
    rows = [json.loads(line) for line in (root / 'continuous/train.jsonl').read_text().splitlines()]
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('objective', 'loss_original', 'loss_photometric', 'alpha', 'gradient_norm_pre_clip')):
            raise ValueError('Nonfinite M1 continuation trace')
        if row['objective'] != row['loss_original'] + row['loss_photometric']:
            raise ValueError('M1 objective is not original+photometric CE')
        for key, step in (('lr_used', row['update'] - 1), ('next_lr', row['update'])):
            if not math.isclose(row[key], 1e-4 * (1 - step / 40000) ** .9, rel_tol=1e-12):
                raise ValueError('M1 continuation LR horizon differs')
    return {**comparison, 'independent_worker_processes': 3, 'checkpoint_schema_version': 2,
            'centered_residuals': True, 'workers': workers, 'formal_m1_result': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=(*BOUNDS, 'compare'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if any(root.is_relative_to(p.resolve()) for p in (ROOT, args.data_root, args.pretrained_root)):
        parser.error('Continuation evidence must be outside source/data/weights')
    output = root if args.mode == 'compare' else root / args.mode
    if args.mode != 'compare': output.mkdir(parents=True, exist_ok=False)
    path = output / ('resume_report.json' if args.mode == 'compare' else 'worker_report.json')
    if path.exists(): parser.error('Refusing to overwrite M1 continuation evidence')
    report = {'schema_version': 1, 'phase': 21, 'scope': 'm1_schema2_source_resume8_vs4plus4',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'process_id': os.getpid(), 'mode': args.mode,
        'seed': 0, 'formal_training_authorized': False, 'target_data_accessed': False, 'new_loss_enabled': False,
        'exact_resume_verified': False, 'source_miou': None, 'target_miou': None, **source_manifest_metadata()}
    try:
        report.update(git_sha=_git('rev-parse', 'HEAD'), git_status=_git('status', '--porcelain'))
        if report['git_status']: raise ValueError('M1 continuation requires clean exact checkout')
        if 'tools.train' in sys.modules or 'causalq.datasets.cityscapes' in sys.modules:
            raise ValueError('Legacy/target module imported')
        report['diagnostics_audit'] = audit_completed_diagnostics()
        report['m0_runner_audit'] = audit_pilot(ROOT / 'experiments/STRICT_DG_PHASE17')
        if args.mode == 'compare':
            report.update(verify_workers(root, report['git_sha']))
        else:
            config_path = ROOT / 'configs/strict_dg/gta_dinov3l_m1_centered.yaml'
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            validate_centered_config(config)
            report['config'] = config
            if not torch.cuda.is_available(): raise RuntimeError('Real M1 source continuation requires AutoDL CUDA')
            if str(torch.__version__) != '2.7.0+cu126' or version('transformers') != '4.56.1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
                raise ValueError('Use pinned CUDA/HF runtime and CUBLAS_WORKSPACE_CONFIG=:4096:8')
            report['environment'] = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'transformers': version('transformers'), 'gpu': torch.cuda.get_device_name(0)}
            seed_everything(0)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            report['loss_preflight'] = cuda_preflight()
            torch.cuda.reset_peak_memory_stats()
            weights = args.pretrained_root / config['model']['weights_dir']
            if file_sha(weights / 'model.safetensors') != config['model']['weights_sha256']:
                raise ValueError('M1 backbone weight SHA mismatch')
            transform = TrainTransform(config['train']['crop_size'], scale_range=config['data']['random_scale'],
                horizontal_flip_probability=config['data']['horizontal_flip_probability'],
                min_valid_fraction=config['data']['min_valid_fraction'], crop_attempts=config['data']['crop_attempts'])
            train, _ = build_source_datasets(args.data_root / 'gta5', transform, SourceEvalTransform())
            backbone = DINOv3Backbone.from_pretrained(config['model']['backbone'], weights=weights, freeze=True,
                intermediate_indices=config['model']['intermediate_indices'], dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
            if tree_digest(backbone.state_dict()) != BACKBONE_STATE_SHA: raise ValueError('M1 backbone state differs')
            model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1, queries_per_class=2,
                interaction='static', cross_attention_layers=0, num_heads=8, temperature=.07, alpha_init=0., centered_residuals=True).to('cuda:0')
            style = StyleInterventionBank(**{k: tuple(v) if isinstance(v, list) else v for k, v in config['style']['photometric'].items()}).to('cuda:0')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
            metadata = {'git_sha': report['git_sha'], 'seed': 0, 'config_sha256': file_sha(config_path),
                'config_tree_sha256': tree_digest(config), 'manifest_sha256': report['manifest_sha256'],
                'backbone_weights_sha256': config['model']['weights_sha256'], 'backbone_sha256': BACKBONE_STATE_SHA,
                'environment': report['environment'], 'centered_residuals': True, 'variant': 'M1_CENTERED',
                'loss_backend': BACKEND, 'sample_workers': 0, 'schedule_horizon': 40000, 'amp': 'bfloat16',
                'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8', 'matmul_allow_tf32': True,
                'cudnn_benchmark': False, 'authorized_final_iteration': 8, 'formal_training_authorized': False,
                'new_loss_enabled': False, 'target_data_accessed': False}
            require_centered_model(model, metadata)
            if args.mode == 'continuous':
                (root / 'config.yaml').write_bytes(config_path.read_bytes())
                (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            elif json.loads((root / 'metadata.json').read_bytes()) != metadata or file_sha(root / 'config.yaml') != metadata['config_sha256']:
                raise ValueError('M1 frozen config/metadata differs across processes')
            if args.mode == 'resume':
                prefix = json.loads((root / 'prefix/worker_report.json').read_bytes())
                if prefix['ok'] is not True or any(file_sha(root / 'prefix' / name) != value for name, value in prefix['files_sha256'].items()):
                    raise ValueError('M1 prefix evidence changed before restore')
            start, stop = BOUNDS[args.mode]
            report['initial_centering'] = centered_diagnostics(model)
            report.update(train_segment(model, optimizer, train, style, config, output, start=start, stop=stop,
                metadata=metadata, resume=root / 'prefix/state.pth' if start else None))
            report.update(metadata=metadata, centered_residuals=True, checkpoint_schema_version=2,
                final_centering=centered_diagnostics(model),
                files_sha256={name: file_sha(output / name) for name in ('train.jsonl', 'state.pth')},
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        report['ok'] = True
    except Exception as exc:
        report.update(ok=False, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__': raise SystemExit(main())
