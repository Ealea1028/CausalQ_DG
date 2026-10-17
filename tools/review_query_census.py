"""Independent CPU review of returned source-analysis artifacts, without DINO."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from PIL import Image

from causalq.analysis.query_components import intervention_seed
from causalq.datasets.strict_source import load_membership
from tools.audit_query_component_census import ROOT, audit_census, census_artifact_hashes, sha_stream

EFFECT_KEYS = ('original_mean_abs', 'photometric_mean_abs', 'cross_view_population_variance', 'rms_normalized_pair_mse')
RESP_KEYS = ('mean_responsibility', 'mean_pixel_entropy_nats', 'usage_entropy_nats', 'above_uniform_activation_rate',
             'top1_share_ties_split', 'cross_view_population_variance', 'slot_logsumexp_deletion_drop')
COMPONENTS = ('full', 'anchor', 'residual')
VIEWS = ('original', 'photometric')


def close(actual, expected, *, relative=1e-9, absolute=1e-18):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
        raise ValueError(f'Independent numeric reconstruction differs: {actual} != {expected}')


def optional_number(value):
    if value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite CSV number')
    return result


def summary(values):
    return {'mean': math.fsum(values) / len(values) if values else None, 'defined_count': len(values)}


def compare_summary(values, expected):
    actual = summary(values)
    if actual['defined_count'] != expected['defined_count'] or (actual['mean'] is None) != (expected['mean'] is None):
        raise ValueError('CSV defined-value denominator differs')
    if actual['mean'] is not None:
        close(actual['mean'], expected['mean'])


def verify_directory(directory, report):
    expected = census_artifact_hashes(report)
    del expected['analysis_export.zip']
    files = list(directory.rglob('*'))
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in files):
        raise ValueError('Returned analysis contains a link/junction')
    actual = {p.relative_to(directory).as_posix() for p in files if p.is_file()}
    if actual != set(expected):
        raise ValueError(f'Returned file coverage differs; missing={set(expected)-actual}; extra={actual-set(expected)}')
    for name, digest in expected.items():
        with (directory / name).open('rb') as handle:
            if sha_stream(handle) != digest:
                raise ValueError(f'Returned artifact SHA256 differs: {name}')
    return expected


def review(directory):
    audit_census(ROOT / 'experiments/STRICT_DG_PHASE20')
    report = json.loads((ROOT / 'experiments/STRICT_DG_PHASE20/census_report.json').read_bytes())
    hashes = verify_directory(directory, report)
    _, ids = load_membership()
    samples = [json.loads(line) for line in (directory / 'sample_index.jsonl').read_text().splitlines()]
    if [s['source_id'] for s in samples] != ids:
        raise ValueError('Returned sample index membership/order differs')
    for s in samples:
        if s['intervention_seed'] != intervention_seed(s['source_id']) or not s['appearance_replay_exact'] or not s['counterfactual_routing_exact']:
            raise ValueError('Sample index seed/replay/routing differs')
        if not 0 <= s['decomposition_max_error'] <= 1e-6:
            raise ValueError('Sample decomposition failed')
    original_samples = {s['source_id']: s for s in report['samples']}
    for sample in samples:
        if sample['source_id'] in original_samples and sample != original_samples[sample['source_id']]:
            raise ValueError('Compact/full sample-index record differs')
    effect_values, per_image, per_class, counts = defaultdict(list), defaultdict(list), defaultdict(list), {}
    with (directory / 'effect_statistics.csv').open(newline='', encoding='utf-8') as handle:
        effect_count = 0
        for i, raw in enumerate(csv.DictReader(handle)):
            expected_key = (ids[i // 57], (i % 57) // 3, COMPONENTS[i % 3])
            key = (raw['source_id'], int(raw['class_id']), raw['component'])
            if key != expected_key or raw['batch_index'] != '0':
                raise ValueError('Effect row coverage/order differs')
            source, c, component = key
            valid, inside, outside = (int(raw[k]) for k in ('valid_pixels', 'inside_pixels', 'outside_pixels'))
            if min(valid, inside, outside) < 0 or inside + outside != valid:
                raise ValueError('Effect region counts do not partition valid pixels')
            if component == 'full': counts[(source, c)] = (valid, inside, outside)
            elif counts[(source, c)] != (valid, inside, outside):
                raise ValueError('Effect components use different GT masks')
            row = {k: optional_number(v) for k, v in raw.items() if k not in ('source_id', 'batch_index', 'class_id', 'component')}
            for k in EFFECT_KEYS:
                value = row[k]
                if value is not None:
                    if value < 0: raise ValueError('Negative magnitude/variance')
                    effect_values[(component, k)].append(value)
                    per_image[(source, component, k)].append(value)
                    per_class[(c, component, k)].append(value)
            for view in VIEWS:
                prefix = 'original' if view == 'original' else 'photometric'
                ix, ox = row[prefix + '_inside_abs'], row[prefix + '_outside_abs']
                if (ix is None) != (inside == 0) or (ox is None) != (outside == 0):
                    raise ValueError('Absent effect region is not explicit missing data')
                ratio = row[prefix + '_inside_outside_ratio']
                if inside and outside:
                    close(ratio, ix / (ox + 1e-12))
                elif ratio is not None: raise ValueError('Undefined localization ratio was imputed')
                if valid:
                    reconstructed = ((ix or 0.) * inside + (ox or 0.) * outside) / valid
                    close(row[prefix + '_mean_abs'], reconstructed, relative=3e-6, absolute=1e-10)
            effect_count += 1
    if effect_count != report['effect_row_count']: raise ValueError('Effect row count differs')
    for source in ids:
        if sum(counts[(source, c)][1] for c in range(19)) != counts[(source, 0)][0]:
            raise ValueError('GT class masks do not partition the source image')
    for component in COMPONENTS:
        for key in EFFECT_KEYS:
            compare_summary(effect_values[(component, key)], report['effect_summary'][component][key])
            for c in range(19):
                compare_summary(per_class[(c, component, key)], report['per_class_effect_summary'][str(c)][component][key])
            for sample in samples:
                compare_summary(per_image[(sample['source_id'], component, key)], sample['effect_summary'][component][key])
    resp_values, resp_class, paired = defaultdict(list), defaultdict(list), {}
    with (directory / 'responsibility.csv').open(newline='', encoding='utf-8') as handle:
        resp_count = 0
        for i, raw in enumerate(csv.DictReader(handle)):
            source, c, view, slot = raw['source_id'], int(raw['class_id']), raw['view'], int(raw['query_r'])
            expected_key = (ids[i // 76], (i % 76) // 4, VIEWS[(i % 4) // 2], i % 2)
            if (source, c, view, slot) != expected_key or raw['batch_index'] != '0':
                raise ValueError('Responsibility row coverage/order differs')
            count = int(raw['gt_score_pixels'])
            row = {key: optional_number(raw[key]) for key in (*RESP_KEYS, 'effective_query_count')}
            if count < 0 or any((value is None) != (count == 0) for value in row.values()):
                raise ValueError('Responsibility missing-class values differ')
            if count:
                for key in RESP_KEYS:
                    value = row[key]
                    limit = math.log(2) + 1e-6 if 'entropy' in key else .25 if key == 'cross_view_population_variance' else float('inf')
                    if not 0 <= value <= limit: raise ValueError('Responsibility/entropy out of range')
                    resp_values[(view, key)].append(value)
                    resp_class[(c, view, slot, key)].append(value)
                close(row['effective_query_count'], math.exp(row['usage_entropy_nats']))
                if slot == 0: paired[(source, c, view)] = row
                else:
                    first = paired.pop((source, c, view))
                    close(first['mean_responsibility'] + row['mean_responsibility'], 1., relative=2e-6)
                    close(first['top1_share_ties_split'] + row['top1_share_ties_split'], 1., relative=2e-6)
            resp_count += 1
    if resp_count != report['responsibility_row_count'] or paired: raise ValueError('Responsibility row count/pair differs')
    for view in VIEWS:
        for key, target in report['responsibility_summary'][view].items():
            compare_summary(resp_values[(view, key)], target)
        for c in range(19):
            for slot in range(2):
                for key in RESP_KEYS:
                    compare_summary(resp_class[(c, view, slot, key)], report['per_class_responsibility_summary'][str(c)][view][str(slot)][key])
    differences = [summary(per_image[(source, 'anchor', 'cross_view_population_variance')])['mean'] -
                   summary(per_image[(source, 'residual', 'cross_view_population_variance')])['mean'] for source in ids]
    comparison = report['census_comparison']
    close(math.fsum(differences) / len(differences), comparison['raw_anchor_minus_residual_variance_mean'])
    if sum(d < 0 for d in differences) != comparison['images_raw_anchor_variance_lower'] or sum(d == 0 for d in differences) != comparison['images_raw_anchor_variance_equal']:
        raise ValueError('Raw CSV variance ordering does not reproduce reported comparison')
    figures = []
    for name in hashes:
        if not name.endswith('.png'): continue
        with Image.open(directory / name) as image:
            if image.format != 'PNG': raise ValueError('Figure is not PNG')
            dimensions = image.size
            image.verify()
        figures.append({'path': name, 'width': dimensions[0], 'height': dimensions[1]})
    return {'schema_version': 1, 'phase': 20, 'source_count': len(ids), 'files_sha256': hashes,
        'effect_row_count': effect_count, 'responsibility_row_count': resp_count,
        'file_bytes_verified': True, 'raw_csv_independently_reviewed': True,
        'all_image_class_and_slot_summaries_reconstructed': True, 'sample_index_coverage_exact': True,
        'raw_anchor_variance_lower_image_count': sum(d < 0 for d in differences), 'png_decode_count': len(figures),
        'figures': figures, 'actual_figures_visually_reviewed': False,
        'effect_summary': {component: {key: summary(effect_values[(component, key)]) for key in EFFECT_KEYS} for component in COMPONENTS},
        'target_data_accessed': False, 'formal_training_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.directory.resolve()):
        parser.error('Preserve original evidence; choose a fresh external review output')
    result = review(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files_sha256', 'figures', 'effect_summary')}, indent=2))


if __name__ == '__main__': main()
