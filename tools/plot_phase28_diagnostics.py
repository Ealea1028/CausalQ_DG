"""Visualize measured Phase28 summaries; no image inference, weights or GPU."""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('U0', 'C0', 'CP', 'CF')
COMPONENTS = ('full', 'anchor', 'residual')
COMPONENT_LABELS = ('Full', 'Anchor', 'Residual contribution')
COLORS = ('#365F87', '#249A7A', '#D68A39')
MODEL_COLORS = ('#446580', '#229B8C', '#DBA03D', '#A05B9D')
CLASS_NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle')
MODEL_DESCRIPTIONS = {'U0': '无中心化 / 无一致性', 'C0': '中心化 / 无一致性',
                      'CP': '中心化 / 预测KL', 'CF': '中心化 / Full CQE'}


def row(report, kind, *, population='all_classes', class_id=-1, **filters):
    selected = [r for r in report['image_macro_statistics']
                if r['kind'] == kind and r['population'] == population and r['class_id'] == class_id
                and all(r.get(k) == value for k, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f'Ambiguous/missing statistic: {kind}, {population}, {class_id}, {filters}')
    return selected[0]['metrics']


def value(report, kind, metric, *, statistic='mean', **filters):
    result = row(report, kind, **filters)[metric].get(statistic)
    return float(result) if result is not None else float('nan')


def ratio(a, b):
    return float(a / b) if math.isfinite(a) and math.isfinite(b) and b > 0 else None


def weighted_usage(report, *, scope, view_index=0, slot=0, classes=range(19)):
    result = []
    for c in classes:
        rows = [r for r in report['pixel_weighted_responsibility']
                if r['class_id'] == c and r['scope'] == scope and r['view_index'] == view_index]
        if len(rows) != 1:raise ValueError('Missing/ambiguous weighted responsibility scope')
        result.append(rows[0]['mean_responsibility'][slot] if rows[0]['pixels'] else np.nan)
    return np.asarray(result, dtype=float)


def normalized_ratio_grid(reports, population, *, classes=range(19)):
    result = []
    for c in classes:
        line = []
        for v in VARIANTS:
            a, r = [value(reports[v], 'effects', 'rms_normalized_variance', population=population,
                          class_id=c, component=k) for k in ('anchor', 'residual')]
            q = ratio(a, r)
            line.append(math.log10(q) if q is not None and q > 0 else np.nan)
        result.append(line)
    return np.asarray(result)


def load_reports(directory):
    accepted = json.loads((ROOT / 'experiments/DIAGNOSTIC_20261020/phase28_acceptance.json').read_bytes())
    reports, hashes = {}, {}
    for variant in VARIANTS:
        relative = variant + '/summary.json'
        payload = (directory / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != accepted['files_verified'][relative]:raise ValueError('Accepted summary SHA differs')
        report = json.loads(payload)
        if (report['sample_count'] != 2497 or report['views'] != 5
                or report['identity']['variant'] != variant or report['target_data_accessed'] is not False):
            raise ValueError('Phase28 summary scope differs')
        reports[variant] = report
        hashes[variant] = digest
    return reports, hashes


def configure():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'svg.hashsalt': 'causalq-phase28-20261024'})


def save(fig, output, stem, title, footer):
    fig.suptitle(title, fontsize=18, fontweight='bold', y=.985)
    fig.text(.02, .012, footer, fontsize=9, color='#53616C', va='bottom')
    fig.tight_layout(rect=(.01, .06, .99, .945), h_pad=2.5, w_pad=2.7)
    for extension in ('png', 'svg'):
        kwargs = {'dpi': 175} if extension == 'png' else {'metadata': {'Date': None}}
        fig.savefig(output / f'{stem}.{extension}', facecolor='white', **kwargs)
    plt.close(fig)


def decorate(ax, ylabel=None):
    if ylabel:ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


def effects_figure(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [('mean_abs', '平均绝对幅值', False), ('joint_rms', '跨风格联合 RMS', False),
             ('raw_variance', '原始跨风格方差', True),
             ('rms_normalized_variance', 'RMS 归一化跨风格方差', True)]
    x = np.arange(4)
    for ax, (metric, label, logarithmic) in zip(axes.flat, specs):
        maximum = 0
        for i, (component, color, legend) in enumerate(zip(COMPONENTS, COLORS, COMPONENT_LABELS)):
            y = np.array([value(reports[v], 'effects', metric, component=component) for v in VARIANTS])
            mid, low, high = [np.array([value(reports[v], 'effects', metric, component=component,
                statistic=stat) for v in VARIANTS]) for stat in ('median', 'p10', 'p90')]
            if logarithmic:
                y, mid, low, high = [z*1000 for z in (y, mid, low, high)]
            pos = x + (i-1)*.24
            ax.bar(pos, y, width=.22, color=color, label=legend, alpha=.9)
            ax.errorbar(pos, mid, yerr=np.vstack((mid-low, high-mid)), fmt='.', color='#263642',
                        alpha=.55, capsize=2, linewidth=1, markersize=4)
            for p, mean, upper in zip(pos, y, high):
                ax.text(p, max(mean, upper)*1.06, f'{mean:.3g}', ha='center', va='bottom', fontsize=8)
            maximum = max(maximum, float(np.nanmax(np.maximum(y, high))))
        ax.set_xticks(x, VARIANTS)
        ax.set_title(label)
        if logarithmic:
            ax.set_yscale('log');ax.set_ylim(top=maximum*2.1)
            ax.set_ylabel('方差（刻度值 × $10^{-3}$）')
        else:ax.set_ylim(0, maximum*1.20)
        decorate(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    save(fig, output, '01_amplitude_stability', 'Phase28｜效应幅值与稳定性',
         '每图19类宏平均，再对2497张图平均；原图+4个复合风格。柱=均值；点/线=中位数/P10–P90图像分布，非置信区间。\n'
         'Residual contribution=Full−Anchor。方差两图采用对数轴；归一化按每个图像/类别/视图分别进行。')


def correlations_figure(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for v, color in zip(VARIANTS, MODEL_COLORS):
        for ax, metric, views in [(axes[0, 0], 'anchor_residual_pearson', range(5)),
                                   (axes[0, 1], 'delta_anchor_residual_pearson', range(1, 5))]:
            ys = [value(reports[v], 'correlations', metric, component='spatial', view_index=s) for s in views]
            ax.plot(list(views), ys, marker='o', label=v, color=color, linewidth=2)
    for ax in axes[0]:
        ax.axhline(0, color='#9099A0', linewidth=1, linestyle='--')
        ax.legend(frameon=False, ncols=4, fontsize=9)
        decorate(ax, 'Pearson 算术宏平均')
    axes[0, 0].set_title('各视图：Anchor 与 Residual 空间相关')
    axes[0, 0].set_xticks(range(5), ['原图', '风格1', '风格2', '风格3', '风格4'])
    axes[0, 1].set_title('相对原图的变化量：ΔAnchor 与 ΔResidual')
    axes[0, 1].set_xticks(range(1, 5), ['风格1', '风格2', '风格3', '风格4'])
    ax = axes[1, 0]
    x = np.arange(4)
    rows = [row(reports[v], 'correlations', component='variance_decomposition', view_index=-1) for v in VARIANTS]
    a, r, cov, full = [np.array([d[k]['mean'] for d in rows])*1000 for k in
                       ('anchor_variance', 'residual_variance', 'style_covariance', 'full_variance')]
    ax.bar(x, a, color=COLORS[1], label='Var(Anchor)')
    ax.bar(x, r, bottom=a, color=COLORS[2], label='Var(Residual)')
    ax.bar(x, 2*cov, color='#B86164', label='2Cov（负值为抵消）')
    ax.scatter(x, full, marker='D', color='#172B3D', zorder=4, label='实测 Var(Full)')
    ax.axhline(0, color='#79838C', linewidth=.8)
    ax.set_xticks(x, VARIANTS);ax.set_title('跨风格方差分解：A + R + 2Cov = Full')
    decorate(ax, '原始方差 × 1000');ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    for i, (metric, label, color) in enumerate([
        ('addition_cancellation_ratio', 'RMS(Full)/(RMS(A)+RMS(R))', '#507894'),
        ('difference_shrink_ratio', 'RMS(R)/(RMS(Full)+RMS(A))', '#DA9B48')]):
        y = [value(reports[v], 'correlations', metric, component='spatial', view_index=0) for v in VARIANTS]
        bars = ax.bar(x+(i-.5)*.32, y, width=.3, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_ylim(0, 1);ax.set_xticks(x, VARIANTS)
    ax.set_title('原图：相加抵消与相减收缩');ax.legend(frameon=False, fontsize=8)
    decorate(ax, '逐图/类比值的宏平均')
    save(fig, output, '02_correlation_cancellation', 'Phase28｜相关性与差分抵消',
         '所有有效像素、图像/类别等权。上图为有符号空间Pearson；左下为风格维度协方差，两种相关口径不可混用。\n'
         '低差分幅值与负相关不等于Residual无用；方差恒等式仅核对输出分解关系。')


def gradients_figure(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x = np.arange(4)
    for q, color in enumerate(('#447EAA', '#DF9B45')):
        values = [value(reports[v], 'gradients', 'query_gradient_norm', term='total', query_r=q) for v in VARIANTS]
        bars = axes[0, 0].bar(x+(q-.5)*.32, values, width=.3, color=color, label=f'Query{q}')
        axes[0, 0].bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        projection = [value(reports[v], 'gradients', 'projection_total_anchor', term='total', query_r=q) for v in VARIANTS]
        bars = axes[1, 0].bar(x+(q-.5)*.32, projection, width=.3, color=color, label=f'Query{q}')
        axes[1, 0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[0, 0].set_title('每个 Query 对复合 q 节点的梯度范数')
    axes[1, 0].set_title('对总 Anchor 梯度的有符号投影贡献')
    axes[1, 0].axhline(.5, linestyle='--', color='#79838C', linewidth=1)
    axes[1, 0].set_ylim(0, .68)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks(x, VARIANTS);ax.legend(frameon=False);decorate(ax)
    y = [value(reports[v], 'gradients', 'query_pair_cosines.1', term='total', query_r=0) for v in VARIANTS]
    cancel = [value(reports[v], 'gradients', 'gradient_cancellation_ratio', term='total', query_r=0) for v in VARIANTS]
    ax = axes[0, 1]
    for i, (ys, color, label) in enumerate([(y, '#459A8A', '两 Query 梯度余弦'),
                                            (cancel, '#8A79AC', '合成范数 / 范数和')]):
        bars = ax.bar(x+(i-.5)*.32, ys, width=.3, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_ylim(0, 1.30);ax.set_xticks(x, VARIANTS);ax.set_title('梯度方向与合成抵消程度')
    ax.legend(frameon=False, fontsize=9);decorate(ax)
    grid = np.array([[value(reports[v], 'gradients', 'query_pair_cosines.1', class_id=c,
                            term='total', query_r=0) for v in VARIANTS] for c in range(19)])
    ax = axes[1, 1]
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(x, VARIANTS);ax.set_yticks(range(19), CLASS_NAMES, fontsize=8)
    ax.set_title('逐类：两 Query 梯度余弦均值')
    for c in range(19):
        for v in range(4):ax.text(v, c, f'{grid[c, v]:.2f}', ha='center', va='center', fontsize=7,
                                 color='white' if abs(grid[c, v]) > .85 else '#243443')
    fig.colorbar(im, ax=ax, fraction=.045, pad=.025, label='余弦 [-1, 1]')
    save(fig, output, '03_query_gradients', 'Phase28｜Query 梯度与 Anchor 贡献',
         '固定512裁剪、eval/autograd、总损失、无optimizer更新/裁剪；各模型总损失包含各自的一致性项。\n'
         '条目均为图像/类别宏平均；均值之比不同于比值均值。四旧模型没有Anchor-only直接损失路径，其直接Anchor梯度为0。')


def responsibility_figure(reports, output):
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.35))
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    heat = fig.add_subplot(gs[2, :])
    x = np.arange(4)
    p = [value(reports[v], 'responsibility', 'mean_responsibility', scope='all_valid', view_index=0, query_r=0) for v in VARIANTS]
    axes[0].bar(x, p, color='#447EAA', label='Query0')
    axes[0].bar(x, 1-np.array(p), bottom=p, color='#DF9B45', label='Query1')
    for i, n in enumerate(p):
        axes[0].text(i, n/2, f'{n:.3f}', ha='center', color='white')
        axes[0].text(i, (1+n)/2, f'{1-n:.3f}', ha='center', color='#283541')
    axes[0].set_title('原图：总体平均责任（all-valid 宏平均）')
    axes[0].legend(frameon=False, loc='upper center', ncols=2)
    axes[0].set_ylim(0, 1.16);axes[0].set_xticks(x, VARIANTS);decorate(axes[0])
    for i, (metric, factor, color, label) in enumerate([
        ('normalized_pixel_entropy', 1, '#57958A', '像素责任熵 / log(2)'),
        ('usage_entropy', 1/math.log(2), '#8D78AC', '每图类平均使用熵 / log(2)')]):
        ys = [factor*value(reports[v], 'responsibility', metric, scope='all_valid', view_index=0, query_r=0) for v in VARIANTS]
        bars = axes[1].bar(x+(i-.5)*.32, ys, width=.3, color=color, label=label)
        axes[1].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[1].set_title('平均使用均衡与像素内均衡是不同量')
    axes[1].set_ylim(0, 1.30);axes[1].set_xticks(x, VARIANTS);axes[1].legend(frameon=False, fontsize=9)
    decorate(axes[1], '归一化熵 [0, 1]')
    for ax, metric, factor, title in [(axes[2], 'top1_switch_rate', 100, '风格干预后：top1 Query 切换率'),
                                     (axes[3], 'js_divergence', 1, '风格干预后：责任分布 JS 散度')]:
        for v, color in zip(VARIANTS, MODEL_COLORS):
            ys = [factor*value(reports[v], 'responsibility', metric, scope='all_valid', view_index=s, query_r=0) for s in range(1, 5)]
            ax.plot(range(1, 5), ys, marker='o', color=color, label=v, linewidth=2)
        ax.set_title(title);ax.set_xticks(range(1, 5), ['风格1', '风格2', '风格3', '风格4'])
        ax.set_ylim(0, ax.get_ylim()[1]*1.3)
        ax.legend(frameon=False, ncols=4, fontsize=9);decorate(ax, '%' if factor == 100 else 'JS（自然对数）')
    grid = np.vstack([weighted_usage(reports[v], scope='gt_class') for v in VARIANTS])
    im = heat.imshow(grid, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=0, vcenter=.5, vmax=1))
    heat.set_xticks(range(19), CLASS_NAMES, rotation=40, ha='right', fontsize=9)
    heat.set_yticks(range(4), VARIANTS)
    heat.set_title('原图逐类：GT 类像素加权的 Query0 平均责任（原生 Query 网格）')
    for i in range(4):
        for j in range(19):heat.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=8,
                                     color='white' if abs(grid[i, j]-.5) > .32 else '#243443')
    fig.colorbar(im, ax=heat, fraction=.022, pad=.015, label='Query0 责任；Query1=1−Query0')
    save(fig, output, '04_query_responsibility', 'Phase28｜Query responsibility 与风格变化',
         '上四图：all-valid 图像/类别宏平均。下图：GT-class 全数据集像素加权，统计口径独立，不能与上图直接混合。\n'
         '接近0.5的总体平均可掩盖逐类主导；责任/熵不是因果必要性。风格1–4均为实际复合photometric干预。')


def stratified_figure(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), gridspec_kw={'height_ratios': [1, 2]})
    x = np.arange(4)
    for i, (population, label, color) in enumerate([
        ('all_classes', '全部19类', '#537F9F'), ('gt_present', '仅GT出现类', '#D79A49')]):
        a = np.array([value(reports[v], 'effects', 'rms_normalized_variance', component='anchor', population=population) for v in VARIANTS])
        r = np.array([value(reports[v], 'effects', 'rms_normalized_variance', component='residual', population=population) for v in VARIANTS])
        for ax, ys in [(axes[0, 0], a/r), (axes[0, 1], r*1000)]:
            bars = ax.bar(x+(i-.5)*.32, ys, width=.3, color=color, label=label)
            ax.bar_label(bars, fmt='%.3g', padding=3, fontsize=9)
            ax.set_yscale('log');ax.set_xticks(x, VARIANTS);ax.legend(frameon=False);decorate(ax)
    axes[0, 0].set_title('Anchor / Residual 归一化方差均值之比')
    axes[0, 0].axhline(1, linestyle='--', linewidth=1, color='#6D7780')
    axes[0, 0].set_ylim(.01, 3)
    axes[0, 1].set_title('Residual 归一化方差：类别出现分层')
    axes[0, 1].set_ylim(1, 200)
    axes[0, 1].set_ylabel('归一化方差（刻度值 × $10^{-3}$）')
    for ax, population, title in [(axes[1, 0], 'all_classes', '全部19类：逐类比值'),
                                  (axes[1, 1], 'gt_present', '仅GT出现图像：逐类比值')]:
        grid = normalized_ratio_grid(reports, population)
        im = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', cmap='RdBu_r',
                       norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=1))
        ax.set_xticks(x, VARIANTS);ax.set_yticks(range(19), CLASS_NAMES, fontsize=9);ax.set_title(title)
        for c in range(19):
            for v in range(4):ax.text(v, c, f'{10**grid[c, v]:.2g}', ha='center', va='center', fontsize=8,
                                     color='white' if grid[c, v] < -1.7 or grid[c, v] > .65 else '#243443')
        fig.colorbar(im, ax=ax, fraction=.04, pad=.025, label='log10(Anchor / Residual)')
    save(fig, output, '05_class_stratification', 'Phase28｜归一化稳定性的逐类检查',
         '颜色：蓝<1，Anchor相对变化更小；红>1，Residual相对变化更小。格内数字为线性比值，颜色为log10比值。\n'
         'GT出现分层只筛选类别/图像，效应仍在全部有效像素计算；它不同于responsibility图中的GT-class像素mask。')


def json_safe(item):
    if isinstance(item, dict):return {k: json_safe(v) for k, v in item.items()}
    if isinstance(item, list):return [json_safe(v) for v in item]
    if isinstance(item, float) and not math.isfinite(item):return None
    return item


def build(reports, hashes, output):
    output.mkdir(parents=True, exist_ok=True)
    configure()
    for plot in (effects_figure, correlations_figure, gradients_figure, responsibility_figure, stratified_figure):
        plot(reports, output)
    class_ratios = {pop: normalized_ratio_grid(reports, pop) for pop in ('all_classes', 'gt_present')}
    measurements = {v: {'model': MODEL_DESCRIPTIONS[v],
        'effects': {pop: {k: row(reports[v], 'effects', component=k, population=pop)
                          for k in COMPONENTS} for pop in ('all_classes', 'gt_present')},
        'macro_statistics': [r for r in reports[v]['image_macro_statistics'] if r['class_id'] == -1],
        'class_names': list(CLASS_NAMES),
        'class_gradient_cosine': [value(reports[v], 'gradients', 'query_pair_cosines.1', class_id=c,
                                        term='total', query_r=0) for c in range(19)],
        'class_log10_normalized_anchor_residual_ratio': {pop: grid[:, VARIANTS.index(v)].tolist()
                                                        for pop, grid in class_ratios.items()},
        'gt_class_weighted_query0': weighted_usage(reports[v], scope='gt_class').tolist()}
        for v in VARIANTS}
    payload = {'schema_version': 1, 'source_summary_sha256': hashes, 'sample_count_per_model': 2497,
        'views': 5, 'source_producer_git_sha': reports['U0']['identity']['git_sha'],
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
        'scope': 'returned summary visualization; no inference or raw per-image recomputation',
        'confidence_intervals_available': False, 'measurements': measurements}
    (output / 'plotted_metrics.json').write_bytes((json.dumps(json_safe(payload), indent=2, allow_nan=False)+'\n').encode())
    captions = [
        ('01_amplitude_stability', '1. 幅值与稳定性', 'C0的原始Anchor方差更高，但归一化后Anchor相对变化远低于Residual。CF的Full方差下降同时RMS增加，不能解释为简单幅值塌缩。'),
        ('02_correlation_cancellation', '2. 相关性与差分抵消', '四组跨风格协方差均为负。Full的方差包含分量抵消；空间Pearson与风格协方差分开报告。'),
        ('03_query_gradients', '3. Query梯度与Anchor贡献', '总损失的两Query平均梯度同向，整体合成范数比接近1。逐类平均也为正；这不排除个别图像的反向梯度。'),
        ('04_query_responsibility', '4. Query责任分布', '总体均值接近均衡可掩盖类别主导：CF的traffic light由Query0承担约95.7%，road由Query1承担约90.8%。'),
        ('05_class_stratification', '5. 类别出现分层', 'C0只有9/19个类别的Anchor归一化方差更低，总体均值不代表每个类别。CP的Residual归一化方差在全部类别宏平均下较C0更高，但在GT出现分层下略低。')]
    sections = ''.join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>'
        f'<img src="{stem}.png" alt="{html.escape(title)}"><p><a href="{stem}.svg">下载SVG矢量图</a></p></section>'
        for stem, title, text in captions)
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CausalQ Phase28诊断图册</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#243746;background:#eef2f5;margin:0}main{max-width:1200px;margin:auto;padding:30px}section{background:white;padding:24px;margin:26px 0;border-radius:12px}img{width:100%;height:auto}h1{color:#15364e}a{color:#216c91}.lead{background:#dcecf0;padding:20px;border-radius:10px}</style>
<main><h1>CausalQ · Phase28 实测诊断图册</h1><div class="lead">U0：无中心化基线；C0：中心化基线；CP：中心化+预测KL；CF：中心化+Full CQE。<br>
每模型2497张GTA5源域图像、五视图、一个已训练种子。图中量来自已接受的summary.json，无新训练或推理。
现有导出没有逐图分割预测或像素效应图，本图册不生成这类未提供的结果。图像分布分位数不是训练种子置信区间。</div>'''
    (output / 'index.html').write_bytes((text+sections+'<p>原始数值和来源：<a href="plotted_metrics.json">plotted_metrics.json</a>。仅源域描述，不据此推断泛化或因果必要性。</p></main></html>').encode())
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'phase28_report')
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/phase28_visualization_20261024')
    args = p.parse_args(argv)
    reports, hashes = load_reports(args.input)
    build(reports, hashes, args.output)
    archive = args.output.parent / (args.output.name + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.iterdir()):
            if path.is_file():z.write(path, path.name)
    print(json.dumps({'output': str(args.output), 'figures': 5, 'archive': str(archive)}, indent=2))
    return 0


if __name__ == '__main__':raise SystemExit(main())
