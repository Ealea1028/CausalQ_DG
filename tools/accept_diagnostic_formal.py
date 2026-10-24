"""Accept reports-only Phase29/30 exports locally; no weights or datasets."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from causalq.datasets.strict_source import MANIFEST_HASHES
from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, baseline_source_report
from tools.package_diagnostic_formal import stage

PRODUCERS = {29: '0a0350d025b063cca2668ab0606bd80179f13ea3',
             30: '61e5d64bf68e26dc22663ae5a8a9999e93e73911'}


def accept(root, phase, archive_sha=None):
    root = Path(root).resolve()
    variants, baseline, contract, validate_config, verify = stage(phase)
    checked = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        rel = PurePosixPath(name)
        path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or name in checked:
            raise ValueError('Unsafe/duplicate checksum entry')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Returned SHA256 differs: ' + name)
        checked[name] = digest
    expected = {'run_manifest.json', f'phase{phase}_report.json'}
    expected.update(f'{v}/{n}' for v in variants for n in
                    ('formal_report.json', 'metadata.json', 'config.yaml', 'initialization.json'))
    if set(checked) != expected:
        raise ValueError('Missing or unexpected checked reports')
    job = json.loads((root / f'phase{phase}_report.json').read_bytes())
    manifest = json.loads((root / 'run_manifest.json').read_bytes())
    contract_key = f'phase{phase}_contract_sha256'
    if (job['ok'] is not True or job['study'] != STUDY or job['phase'] != phase
            or job['git_sha'] != PRODUCERS[phase] or job[contract_key] != verify()
            or job['target_data_accessed'] is not False or job['target_miou'] is not None
            or job['total_optimizer_updates'] != 40000 * len(variants)
            or job['optimizer_updates_per_variant'] != 40000 or set(job['reports']) != set(variants)
            or manifest['contract'] != contract() or manifest['git_sha'] != PRODUCERS[phase]
            or manifest[contract_key] != verify() or manifest['phase'] != phase):
        raise ValueError('Formal producer/budget/freeze differs')
    baseline_value = baseline_source_report('C0' if phase == 30 else 'U0')['source_miou']
    previous = json.loads((ROOT / 'experiments/DIAGNOSTIC_20261020/phase29_acceptance.json').read_bytes())
    gt = previous['source_gt_class_totals']
    independent = {}
    for v in variants:
        report = json.loads((root / v / 'formal_report.json').read_bytes())
        meta = json.loads((root / v / 'metadata.json').read_bytes())
        initial = json.loads((root / v / 'initialization.json').read_bytes())
        config = yaml.safe_load((root / v / 'config.yaml').read_bytes())
        validate_config(config, v)
        if (report != job['reports'][v] or report['metadata'] != meta or report['variant'] != v or report['phase'] != phase
                or meta['git_sha'] != PRODUCERS[phase] or meta[contract_key] != verify()
                or meta['variant'] != v or meta['centered_residuals'] is not (phase == 30)
                or meta['config_sha256'] != checked[f'{v}/config.yaml']
                or meta['manifest_sha256'] != MANIFEST_HASHES
                or meta['environment'] != baseline()['environment']
                or report['initial_head_sha256'] != baseline()['initial_head_sha256']
                or initial['head_sha256'] != report['initial_head_sha256']
                or report['ok'] is not True or report['formal_optimizer_updates'] != 40000
                or report['checkpoint_selection'] != 'fixed_final'
                or report['target_data_accessed'] is not False or report['target_miou'] is not None):
            raise ValueError('Formal recipe/provenance differs: ' + v)
        evaluation = report['source_evaluation']
        counts = evaluation['confusion_matrix']
        if (len(counts) != 19 or any(len(row) != 19 for row in counts)
                or any(type(n) is not int or n < 0 for row in counts for n in row)):
            raise ValueError('Invalid integer confusion matrix')
        matrix = np.array(counts, dtype=np.int64)
        union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
        iou = matrix.diagonal() / union
        value = float(iou.mean())
        if (evaluation['sample_count'] != 2497 or evaluation['training_state_unchanged'] is not True
                or evaluation['evaluation_rng_unchanged'] is not True
                or evaluation['seen_ids_sha256'] != MANIFEST_HASHES['gta5_source_val.txt']
                or matrix.sum(1).tolist() != gt or evaluation['valid_pixel_count'] != int(matrix.sum())
                or not np.allclose(iou, evaluation['class_iou'], rtol=0, atol=1e-12)
                or any(abs(value - x) > 1e-12 for x in
                       (report['source_miou'], evaluation['source_miou'], job['source_miou'][v]))):
            raise ValueError('Source coverage/confusion reduction differs: ' + v)
        independent[v] = value
    return {'schema_version': 1, 'phase': phase, 'accepted_date': date.today().isoformat(),
            'accepted': True, 'producer_git_sha': PRODUCERS[phase], 'file_sha256': checked,
            'provided_archive_sha256': archive_sha, 'archive_bytes_available_locally': False,
            'checkpoints_and_full_traces_available_locally': False,
            'source_confusion_reduced_independently': True, 'source_gt_class_totals': gt,
            'independent_source_miou': independent, 'baseline_source_miou': baseline_value,
            'source_delta_vs_baseline_percentage_points': {v: 100*(x-baseline_value) for v, x in independent.items()},
            'target_data_accessed': False, 'formal_report': job}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--phase', type=int, choices=(29, 30), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive-sha')
    args = parser.parse_args(argv)
    result = accept(args.input, args.phase, args.archive_sha)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({k: result[k] for k in ('accepted', 'independent_source_miou',
                                           'source_delta_vs_baseline_percentage_points')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
