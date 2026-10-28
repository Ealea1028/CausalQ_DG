"""Ten-model source summary analysis; no source images, model loading or GPU."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from causalq.protocols.diagnostic_20261020 import ROOT, BASELINES, VARIANTS
from causalq.protocols.diagnostic_final_source import NEW_FINALS, final_source_report
from tools.plot_phase28_diagnostics import (
    COMPONENTS, COLORS, CLASS_NAMES, configure, value, row, ratio, weighted_usage,
)
from tools.train_strict_source import file_sha


def load_reports(baseline_root, final_root):
    reports = {};hashes = {}
    for phase, variants, directory in ((28, BASELINES, baseline_root), (31, NEW_FINALS, final_root)):
        acceptance = json.loads((ROOT / f'experiments/DIAGNOSTIC_20261020/phase{phase}_acceptance.json').read_bytes())
        for v in variants:
            path = directory / v / 'summary.json';digest = file_sha(path)
            if digest != acceptance['files_verified'][v+'/summary.json']:
                raise ValueError('Accepted summary differs: '+v)
            reports[v] = json.loads(path.read_bytes());hashes[v] = digest
    return reports, hashes


def comparison_rows(reports):
    rows = []
    for v in VARIANTS:
        r = reports[v]
        for component in COMPONENTS:
            stats = row(r, 'effects', component=component)
            rows.append({'variant': v, 'component': component,
                         **{m: stats[m]['mean'] for m in ('mean_abs', 'joint_rms', 'raw_variance',
                               'rms_normalized_variance', 'pair_raw_variance', 'near_zero_view_count')},
                         'raw_variance_definition': 'five-view_population_S5',
                         'normalized_variance_definition': 'per-view_RMS_then_five-view_population',
                         'sample_count': r['sample_count']})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, default=ROOT / 'phase28_report')
    p.add_argument('--final-root', type=Path, default=ROOT / 'phase31_export')
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/diagnostic_final_source_20261028')
    args = p.parse_args(argv);reports, hashes = load_reports(args.baseline_root, args.final_root)
    output = args.output;output.mkdir(parents=True, exist_ok=True);configure()
    models = list(VARIANTS);palette = plt.cm.tab10(np.arange(10));x = np.arange(10)
    plt.rcParams['svg.hashsalt'] = 'causalq-phase31-20261028'
    def metric(kind, name, **filters):
        return np.array([value(reports[v], kind, name, **filters) for v in models])
    def save(fig, name):
        fig.tight_layout();fig.savefig(output / (name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(output / (name+'.svg'), metadata={'Date': None}, bbox_inches='tight');plt.close(fig)
    def ticks(ax):
        ax.set_xticks(x, models);ax.axvline(4.5, color='#aaaaaa', lw=.7);ax.grid(axis='y', alpha=.16)

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    for ax, m, title in zip(axes.flat, ('mean_abs', 'joint_rms', 'raw_variance', 'rms_normalized_variance'),
                            ('平均绝对幅值', '联合RMS', '五视图原始方差', '逐视图RMS归一化方差')):
        for j, c in enumerate(COMPONENTS):
            ax.bar(x+(j-1)*.24, metric('effects', m, component=c), .23, color=COLORS[j], label=c)
        if 'variance' in m:ax.set_yscale('log')
        ax.set_title(title);ax.set_ylabel('原始数值（未乘倍率）');ticks(ax)
    axes[0, 0].legend();fig.suptitle('十模型源域宏平均：2497图 × 19类，原图+4风格；U组左 / C组右', y=.995)
    save(fig, '01_amplitude_variance')

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, c in enumerate(COMPONENTS):
        for i, m in enumerate(('raw_variance', 'rms_normalized_variance')):
            ax = axes[i, j]
            for index, v in enumerate(models):
                a = value(reports[v], 'effects', 'joint_rms', component=c)
                b = value(reports[v], 'effects', m, component=c)
                ax.scatter(a, b, color=palette[index], marker='o' if v.startswith('U') else '^', s=60)
                ax.annotate(v, (a, b), xytext=(3, 3), textcoords='offset points', fontsize=9)
            ax.set_yscale('log');ax.set_xlabel('宏平均联合RMS');ax.set_ylabel(m);ax.set_title(c);ax.grid(alpha=.15)
    save(fig, '02_amplitude_stability_scatter')

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    spatial = np.array([metric('correlations', 'anchor_residual_pearson', component='spatial', view_index=i) for i in range(5)])
    delta = np.array([metric('correlations', 'delta_anchor_residual_pearson', component='spatial', view_index=i) for i in range(1, 5)])
    for ax, data, title in ((axes[0, 0], spatial, '空间corr(Anchor, Residual)：原图及4风格'),
                            (axes[0, 1], delta, '各风格相对原图变化 corr(delta A, delta R)')):
        im = ax.imshow(data, aspect='auto', vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(x, models)
        ax.set_yticks(range(data.shape[0]), ['original', 'photo1', 'photo2', 'photo3', 'photo4'] if data.shape[0] == 5 else ['photo1', 'photo2', 'photo3', 'photo4'])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    ax = axes[1, 0]
    for j, c in enumerate(COMPONENTS):
        ax.bar(x+(j-1)*.19, metric('effects', 'raw_variance', component=c), .18, color=COLORS[j], label='Var '+c)
    ax.plot(x, 2*metric('correlations', 'style_covariance', component='variance_decomposition', view_index=-1), 'kx-', label='2Cov(style A,R)')
    ax.axhline(0, color='gray', lw=.6);ticks(ax);ax.legend(fontsize=8);ax.set_title('Var(Full)=Var(A)+Var(R)+2Cov；原始数值')
    ax = axes[1, 1]
    for m in ('addition_cancellation_ratio', 'difference_shrink_ratio'):
        ax.plot(x, metric('correlations', m, component='spatial', view_index=0), 'o-', label=m)
    ticks(ax);ax.legend(fontsize=8);ax.set_title('原图逐图/类比值的宏平均：低值提示抵消或差分收缩')
    save(fig, '03_correlation_cancellation')

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for q in (0, 1):
        axes[0, 0].bar(x+(q-.5)*.3, metric('gradients', 'query_gradient_norm', query_r=q, term='total'), .29, label=f'Query{q}')
    ticks(axes[0, 0]);axes[0, 0].legend();axes[0, 0].set_title('总目标梯度范数：固定512裁剪，eval，无优化器')
    for m in ('query_pair_cosines.1', 'gradient_cancellation_ratio'):
        axes[0, 1].plot(x, metric('gradients', m, query_r=0, term='total'), 'o-', label=m)
    ticks(axes[0, 1]);axes[0, 1].legend(fontsize=8);axes[0, 1].set_title('两Query方向与via-Anchor梯度抵消比')
    ax = axes[1, 0]
    for j, (q, m, label) in enumerate(((0, 'projection_total_anchor', 'Query0→total Anchor'),
                                      (1, 'projection_total_anchor', 'Query1→total Anchor'),
                                      (0, 'direct_projection_total', 'direct→total Anchor'))):
        ax.bar(x+(j-1)*.24, metric('gradients', m, query_r=q, term='total'), .23, label=label)
    ticks(ax);ax.axhline(0, color='gray', lw=.6);ax.legend(fontsize=8);ax.set_title('有符号投影贡献：Query项与direct项合计=1')
    grid = np.array([[value(reports[v], 'gradients', 'query_pair_cosines.1', class_id=c, query_r=0, term='total')
                      for v in models] for c in range(19)])
    im = axes[1, 1].imshow(grid, aspect='auto', vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1, 1].set_xticks(x, models);axes[1, 1].set_yticks(range(19), CLASS_NAMES, fontsize=8)
    axes[1, 1].set_title('每类别总目标Query梯度cosine');fig.colorbar(im, ax=axes[1, 1], shrink=.7)
    save(fig, '04_query_gradients')

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for m in ('mean_responsibility', 'normalized_pixel_entropy'):
        axes[0, 0].plot(x, metric('responsibility', m, query_r=0, scope='all_valid', view_index=0), 'o-', label=m)
    ticks(axes[0, 0]);axes[0, 0].legend(fontsize=8);axes[0, 0].set_title('原图Query0责任与像素责任熵：逐图/类宏平均')
    for ax, m in ((axes[0, 1], 'top1_switch_rate'), (axes[1, 0], 'js_divergence')):
        for j, v in enumerate(models):
            values = [value(reports[v], 'responsibility', m, query_r=0, scope='all_valid', view_index=i) for i in range(1, 5)]
            ax.plot(range(1, 5), values, 'o-' if v.startswith('U') else '^--', color=palette[j], label=v)
        ax.set_xticks(range(1, 5));ax.set_xlabel('风格干预序号');ax.set_title(m);ax.legend(ncol=5, fontsize=8);ax.grid(alpha=.15)
    grid = np.array([weighted_usage(reports[v], scope='gt_class') for v in models]).T
    im = axes[1, 1].imshow(grid, aspect='auto', vmin=0, vmax=1, cmap='viridis')
    axes[1, 1].set_xticks(x, models);axes[1, 1].set_yticks(range(19), CLASS_NAMES, fontsize=8)
    axes[1, 1].set_title('原图GT类别区域像素加权Query0责任');fig.colorbar(im, ax=axes[1, 1], shrink=.7)
    save(fig, '05_query_responsibility')

    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    for ax, pop in zip(axes, ('all_classes', 'gt_present')):
        grid = np.array([[ratio(value(reports[v], 'effects', 'rms_normalized_variance', class_id=c, population=pop, component='anchor'),
                               value(reports[v], 'effects', 'rms_normalized_variance', class_id=c, population=pop, component='residual'))
                          for v in models] for c in range(19)], dtype=float)
        im = ax.imshow(np.log10(grid), aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xticks(x, models);ax.set_yticks(range(19), CLASS_NAMES);ax.set_title(pop+'：log10(归一化Var A / R)')
        fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, '06_class_stratification')

    rows = comparison_rows(reports)
    with (output / 'model_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    class_rows = []
    for v in models:
        for r in reports[v]['image_macro_statistics']:
            for name, stats in r['metrics'].items():
                class_rows.append({'variant': v, 'kind': r['kind'], 'population': r['population'], 'class_id': r['class_id'],
                                   **{k: r.get(k) for k in ('component', 'view_index', 'scope', 'term', 'query_r')},
                                   'metric': name, **{k: stats.get(k) for k in ('count', 'null_count', 'mean', 'p10', 'median', 'p90', 'fisher_mean')}})
    with (output / 'class_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    plotted = {'source_summary_sha256': hashes, 'generator_sha256': file_sha(Path(__file__)), 'model_metrics': rows,
               'scope': 'single seed, source-only; image-class macro; no raw per-image recomputation; target results pending'}
    (output / 'plotted_metrics.json').write_bytes((json.dumps(plotted, indent=2, allow_nan=False)+'\n').encode())
    images = sorted(output.glob('*.png'))
    body = '<h1>十模型最终源域诊断</h1><p>2497图，五视图总体方差；原始与逐视图RMS归一化统计分开。单种子描述性结果。</p>'
    for image in images:body += f'<h2>{html.escape(image.stem)}</h2><img style="width:100%;max-width:1500px" src="{image.name}">'
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>源域诊断</title>'+body, encoding='utf-8')
    print(f'Wrote {len(images)} scientific figures and model/class tables: {output}')
    return 0


if __name__ == '__main__':raise SystemExit(main())
