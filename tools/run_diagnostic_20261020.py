"""Phase28: four frozen formal source models; no optimizer or target reads."""

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import uuid
import zipfile

import torch
from PIL import Image

from causalq.analysis.diagnostic_summary import DiagnosticSummary
from causalq.analysis.intervention_examples import export_examples
from causalq.analysis.query_gradients import query_gradient_rows, parameter_rows
from causalq.analysis.selective_stability import effect_and_correlation_rows, multiview_responsibility_rows
from causalq.datasets.segmentation import PairedSegmentationDataset, TrainTransform
from causalq.datasets.strict_source import build_source_datasets, SourceEvalTransform, load_membership
from causalq.interventions import StyleInterventionBank
from causalq.models import DINOv3Backbone, QuerySegmentor
from causalq.protocols.diagnostic_20261020 import (ROOT, STUDY, BASELINES, baseline_records, baseline_source_report,
    stable_seed, style_seeds, preview_ids, contract_sha)
from causalq.protocols.formal_source import formal_config
from causalq.utils.checkpoint import load_training_checkpoint, trainable_state_dict
from causalq.utils.seed import seed_everything
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest, precision
from tools.train_strict_source import file_sha


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    os.replace(temporary, path)


@contextmanager
def isolated_rng(seed, device):
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        if device.type == 'cuda':torch.manual_seed(seed)
        else:torch.random.default_generator.manual_seed(seed)
        yield


def make_views(image, style, identity):
    views, traces = [image], []
    for seed in style_seeds(identity):
        with isolated_rng(seed, image.device):
            view, trace = style.photometric_view_with_trace(image)
        views.append(view);traces.append(trace)
    return views, traces


@torch.no_grad()
def multiview_measurement(model, views, label, *, amp=False):
    maps, scores = [], []
    for view in views:
        with precision(str(view.device), amp):
            output = model.forward_components(view, include_anchor=True, component_protocol=STUDY)
        if not torch.isfinite(output.logits).all():raise ValueError('Nonfinite source logits')
        full = model.get_query_effect(output=output).float()
        anchor = model.get_anchor_query_effect(output=output, component_protocol=STUDY)
        maps.append({'full': full.cpu(), 'anchor': anchor.cpu(), 'residual': (full - anchor).cpu()})
        scores.append(output.query_score_maps.detach().float().cpu())
        del output, full, anchor
    label_cpu = label.cpu()
    effects, correlations = effect_and_correlation_rows(maps, label_cpu)
    responsibility = multiview_responsibility_rows(scores, label_cpu)
    return effects, correlations, responsibility


def state_versions(model):
    return [(name, id(tensor), tensor._version) for name, tensor in (*model.named_parameters(), *model.named_buffers())]


def fingerprint(pair):
    return {kind: {'name': getattr(pair, kind).name, 'size': getattr(pair, kind).stat().st_size,
                   'mtime_ns': getattr(pair, kind).stat().st_mtime_ns} for kind in ('image', 'label')}


def load_model(args, variant, *, frozen=None, source=None, config=None):
    frozen = baseline_records()[variant] if frozen is None else frozen
    source = baseline_source_report(variant) if source is None else source
    config = formal_config(BASELINES[variant]) if config is None else config
    checkpoint = args.training_output_root / frozen['checkpoint_relative_to_output_root']
    if file_sha(checkpoint) != frozen['checkpoint_sha256']:raise ValueError('Original checkpoint SHA differs')
    weights = args.pretrained_root / config['model']['weights_dir']
    if file_sha(weights / 'model.safetensors') != config['model']['weights_sha256']:raise ValueError('DINO weights differ')
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'transformers': version('transformers'),
        'gpu': torch.cuda.get_device_name(0)}
    if environment != source['metadata']['environment']:raise ValueError('Use original matched formal runtime')
    seed_everything(0)
    backbone = DINOv3Backbone.from_pretrained(config['model']['backbone'], weights=weights, freeze=True,
        intermediate_indices=config['model']['intermediate_indices'], dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
    model = QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1, queries_per_class=2,
        interaction='static', cross_attention_layers=0, num_heads=8, temperature=.07, alpha_init=0.,
        centered_residuals=config['query']['centered_residuals']).to('cuda:0').eval()
    payload = load_training_checkpoint(checkpoint)
    if (payload.get('schema_version') != 2 or payload['iteration'] != 40000 or payload['metadata'] != source['metadata']
            or set(payload['trainable_model']) != set(trainable_state_dict(model))):
        raise ValueError('Original checkpoint provenance/coverage differs')
    restored = model.load_state_dict(payload['trainable_model'], strict=False)
    if restored.unexpected_keys or set(restored.missing_keys) != {'backbone.' + k for k in backbone.state_dict()}:
        raise ValueError('Original checkpoint restoration differs')
    del payload
    if tree_digest(trainable_state_dict(model)) != frozen['head_sha256']:raise ValueError('Restored head differs')
    style = StyleInterventionBank(**{k: tuple(v) if isinstance(v, list) else v
        for k, v in config['style']['photometric'].items()}).to('cuda:0')
    return model, style, environment


def validate_cached(path, identity, source_id, pair):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:record = json.load(stream)
    if record['identity'] != identity or record['source_id'] != source_id or record['source_files'] != fingerprint(pair):
        raise ValueError('Cached source diagnostic contract/ID/data differs')
    return record


def diagnose_one(model, style, sample, gradient_sample, identity, variant, *, amp=False):
    device = next(model.parameters()).device
    image, label = [sample[k].unsqueeze(0).to(device) for k in ('image', 'label')]
    classes = model.num_classes
    if ((label != 255) & ((label < 0) | (label >= classes))).any():raise ValueError('Invalid source trainIds')
    views, traces = make_views(image, style, identity)
    effects, correlations, responsibility = multiview_measurement(model, views, label, amp=amp)
    gi, gl = [gradient_sample[k].unsqueeze(0).to(device) for k in ('image', 'label')]
    with isolated_rng(style_seeds(identity)[0], device):gp, gradient_trace = style.photometric_view_with_trace(gi)
    gradients = query_gradient_rows(model, gi, gp, gl, variant, amp=amp)
    return {'effects': effects, 'correlations': correlations, 'responsibility': responsibility, 'gradients': gradients,
            'photometric_seeds': style_seeds(identity), 'photometric_parameters': traces,
            'gradient_photometric_parameters': gradient_trace, 'gradient_crop_seed': stable_seed(identity, 'gradient-crop'),
            'evaluation_input_sha256': tree_digest((sample['image'], sample['label'])),
            'gradient_input_sha256': tree_digest((gradient_sample['image'], gradient_sample['label']))}, views, traces


def run_baseline(args):
    variant = args.variant;frozen = baseline_records()[variant]
    return run_diagnostic(args, frozen, baseline_source_report(variant), formal_config(BASELINES[variant]),
                          phase=28, contract_digest=contract_sha(),
                          config_path=ROOT / f'configs/diagnostic_20261020/{variant.lower()}_seed0.yaml',
                          export_previews=variant == 'U0')


def run_diagnostic(args, frozen, source, config, *, phase, contract_digest, config_path, export_previews=False):
    """Shared measurement loop; stage-specific callers supply frozen model identity."""
    variant = args.variant
    root = args.output_root / variant;root.mkdir(parents=True, exist_ok=True)
    identity = {'study': STUDY, 'phase': phase, 'variant': variant, 'git_sha': _git('rev-parse', 'HEAD'),
                'contract_sha256': contract_digest, 'checkpoint_sha256': frozen['checkpoint_sha256'],
                'original_config_sha256': frozen['config_sha256'],
                'diagnostic_config_sha256': file_sha(config_path)}
    done = root / 'summary.json'
    if done.exists():
        old = json.loads(done.read_bytes())
        if old['identity'] != identity or old['sample_count'] != 2497 or old['ok'] is not True or not (root / 'class_summary.csv').is_file():
            raise ValueError('Completed diagnostic summary differs')
        print(f'{variant}: complete; reused without GPU work', flush=True);return
    if not torch.cuda.is_available():raise RuntimeError(f'Phase{phase} requires AutoDL CUDA/source data')
    if str(torch.__version__) != '2.7.0+cu126' or version('transformers') != '4.56.1' or os.getenv('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Use pinned runtime and CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark = False;torch.backends.cuda.matmul.allow_tf32 = True
    model, style, environment = load_model(args, variant, frozen=frozen, source=source, config=config)
    _, dataset = build_source_datasets(args.data_root / 'gta5', TrainTransform((512, 512)), SourceEvalTransform())
    gradient_data = PairedSegmentationDataset(dataset.pairs, transform=TrainTransform((512, 512)))
    _, ids = load_membership()
    if len(dataset) != 2497:raise ValueError('Full source diagnostic coverage required')
    before_versions = state_versions(model);before_rng = tree_digest(capture_rng())
    before_style = tree_digest(style.state_dict())
    before_head = tree_digest(trainable_state_dict(model));torch.cuda.reset_peak_memory_stats()
    rows_dir = root / 'samples';rows_dir.mkdir(exist_ok=True)
    summary = DiagnosticSummary();summary.add_rows('parameters', parameter_rows(model))
    preview = set(preview_ids())
    for index, source_id in enumerate(ids):
        path = rows_dir / f'{source_id}.json.gz'
        if path.exists():
            record = validate_cached(path, identity, source_id, dataset.pairs[index])
        else:
            sample = dataset[index]
            with isolated_rng(stable_seed(source_id, 'gradient-crop'), torch.device('cpu')):
                gradient_sample = gradient_data[index]
            record, views, traces = diagnose_one(model, style, sample, gradient_sample, source_id, variant, amp=True)
            record.update(identity=identity, source_id=source_id, source_files=fingerprint(dataset.pairs[index]))
            if export_previews and source_id in preview:
                with Image.open(dataset.pairs[index].image) as image:raw = image.convert('RGB')
                export_examples(args.output_root / 'intervention_examples' / source_id, raw, views, traces,
                                style_seeds(source_id), style, source_id, identity=identity)
            del views, traces
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
            with gzip.open(temporary, 'wt', encoding='utf-8') as stream:json.dump(record, stream, allow_nan=False)
            os.replace(temporary, path)
        summary.add_image(record)
        if (index + 1) % 50 == 0:print(f'{variant} source_diagnostics={index+1}/2497', flush=True)
    unchanged = (before_versions == state_versions(model) and before_rng == tree_digest(capture_rng())
                 and before_head == tree_digest(trainable_state_dict(model)) and before_style == tree_digest(style.state_dict())
                 and all(p.grad is None for p in model.parameters()))
    if not unchanged:raise ValueError('Read-only diagnostic changed model/RNG/parameter gradients')
    result = {'schema_version': 1, 'ok': True, 'identity': identity, 'environment': environment,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_training_updates': 40000,
        'training_producer_git_sha': frozen['producer_git_sha'], 'optimizer_updates_this_run': 0,
        'target_data_accessed': False, 'model_rng_unchanged': unchanged,
        'effect_geometry': 'longedge1024_aspect_pad16', 'gradient_geometry': 'deterministic_512_crop',
        'views': 5, 'new_training_gate': False, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
        'limits': 'single trained seed; source-only descriptive diagnostics; uncentered components not identified',
        **summary.result()}
    csv_path = root / 'class_summary.csv'
    csv_tmp = root / 'class_summary.csv.partial'
    with csv_tmp.open('w', encoding='utf-8', newline='') as stream:
        fields = ['kind', 'population', 'class_id', 'component', 'view_index', 'scope', 'term', 'query_r', 'observation_unit',
                  'metric', 'count', 'null_count', 'mean', 'p10', 'median', 'p90', 'fisher_mean', 'fisher_clipped_count']
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for row in result['image_macro_statistics']:
            for metric, values in row['metrics'].items():
                writer.writerow({**{k: v for k, v in row.items() if k != 'metrics'}, 'metric': metric, **values})
    os.replace(csv_tmp, csv_path)
    write_json(done, result)
    print(f'{variant} summary={done}', flush=True)


def package(args):
    reports = {}
    for variant in BASELINES:
        report = json.loads((args.output_root / variant / 'summary.json').read_bytes())
        if (report['ok'] is not True or report['sample_count'] != 2497 or report['identity']['git_sha'] != _git('rev-parse', 'HEAD')
                or report['identity']['contract_sha256'] != contract_sha() or report['optimizer_updates_this_run'] != 0
                or report['model_rng_unchanged'] is not True or report['target_data_accessed'] is not False):
            raise ValueError('Need all four complete same-commit source summaries')
        reports[variant] = {k: v for k, v in report.items() if k not in ('image_macro_statistics', 'pixel_weighted_responsibility')}
    report = {'schema_version': 1, 'phase': 28, 'ok': True, 'git_sha': _git('rev-parse', 'HEAD'),
        'study': STUDY, 'reports': reports, 'preview_ids': preview_ids(), 'target_data_accessed': False,
        'optimizer_updates_this_run': 0, 'next_phase': '29: UP/UF/UA/UR formal40000; no raw-stability gate'}
    write_json(args.output_root / 'job_report.json', report)
    files = [args.output_root / 'job_report.json']
    for v in BASELINES:files += [args.output_root / v / 'summary.json', args.output_root / v / 'class_summary.csv']
    for identity in preview_ids():
        directory = args.output_root / 'intervention_examples' / identity
        if not (directory / 'interventions.json').is_file():raise ValueError('Missing frozen intervention example')
        files += sorted(directory.glob('*'))
    sums = args.output_root / 'SHA256SUMS'
    sums.write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(args.output_root).as_posix()}\n' for p in files).encode())
    with zipfile.ZipFile(args.output_root / 'phase28_export.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, sums]:archive.write(path, path.relative_to(args.output_root))
    print(json.dumps({'ok': True, 'export': str(args.output_root / 'phase28_export.zip'),
                      'sha256': file_sha(args.output_root / 'phase28_export.zip')}, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=(*BASELINES, 'package'), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--training-output-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args(argv)
    protected = [ROOT, args.data_root, args.pretrained_root]
    protected += [(args.training_output_root / r['checkpoint_relative_to_output_root']).parents[2] for r in baseline_records().values()]
    if any(args.output_root.resolve().is_relative_to(x.resolve()) for x in protected):
        p.error('Output must be outside code/data/weights/original training runs')
    if _git('status', '--porcelain'):raise ValueError('Use clean exact checkout')
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.variant == 'package':package(args)
    else:run_baseline(args)
    return 0


if __name__ == '__main__':raise SystemExit(main())
