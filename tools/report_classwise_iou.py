"""Read-only19-class IoU report from completed DINOv3 confusion matrices."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from causalq.protocols.diagnostic_20261020 import ROOT, VARIANTS, BASELINES, baseline_source_report
from causalq.protocols.diagnostic_final_source import final_source_report
from tools.plot_phase28_diagnostics import CLASS_NAMES, configure

CHINESE = ('道路', '人行道', '建筑', '墙', '栅栏', '杆', '交通灯', '交通标志', '植被', '地形',
           '天空', '行人', '骑行者', '汽车', '卡车', '公交车', '火车', '摩托车', '自行车')


def reduce_counts(counts):
    """Rows=GT, columns=prediction; undefined union is null, not zero."""
    if (len(counts) != 19 or any(len(row) != 19 for row in counts)
            or any(type(x) is not int or x < 0 for row in counts for x in row)):
        raise ValueError('Need19x19 nonnegative integer confusion counts')
    matrix = np.asarray(counts, dtype=np.int64)
    gt = matrix.sum(1);pred = matrix.sum(0);tp = matrix.diagonal();union = gt+pred-tp
    iou = [float(tp[c]/union[c]) if union[c] else None for c in range(19)]
    defined = [x for x in iou if x is not None]
    return {'class_iou': iou, 'miou': sum(defined)/len(defined) if defined else None,
            'valid_classes': len(defined), 'gt_pixels': gt.tolist(), 'predicted_pixels': pred.tolist(),
            'true_positive_pixels': tp.tolist(), 'union_pixels': union.tolist()}


def build_report():
    path = ROOT / 'experiments/DIAGNOSTIC_20261020/phase32_acceptance.json'
    accepted = json.loads(path.read_bytes())
    if accepted['accepted'] is not True:raise ValueError('Need accepted final results')
    old_path = ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json'
    old = json.loads(old_path.read_bytes());models = {};totals = {}
    for v in VARIANTS:
        source = baseline_source_report(v) if v in BASELINES else final_source_report(v)
        target = old['reports'][BASELINES[v]] if v in BASELINES else accepted['formal_report']['reports'][v]
        models[v] = {}
        for domain, record, key in (('source', source['source_evaluation'], 'source_miou'), ('target', target, 'target_miou')):
            result = reduce_counts(record['confusion_matrix'])
            if (result['valid_classes'] != 19 or abs(result['miou']-record[key]) > 1e-12
                    or not np.allclose(result['class_iou'], record['class_iou'], atol=1e-12, rtol=0)):
                raise ValueError('Stored score differs from confusion counts: '+v+'/'+domain)
            if domain in totals and totals[domain] != result['gt_pixels']:raise ValueError('Unmatched evaluation GT totals')
            totals[domain] = result['gt_pixels'];models[v][domain] = result
    return {'study': 'completed_DINOv3_diagnostic_20261020', 'seed': 0, 'models': models,
            'class_order': [{'train_id': i, 'name': n, 'chinese': CHINESE[i]} for i, n in enumerate(CLASS_NAMES)],
            'source_samples': 2497, 'target_samples': 500,
            'accepted_target_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'old_target_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
            'limits': 'historical DINOv3 single-seed results; not Phase33 DINOv2 results; no images or weights loaded'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/classwise_iou_20261103')
    args = p.parse_args(argv);args.output.mkdir(parents=True, exist_ok=True)
    result = build_report();configure()
    plt.rcParams['svg.hashsalt'] = 'causalq-classwise-iou-20261103'
    tables = []
    for domain, title in (('target', 'Cityscapes500 目标域'), ('source', 'GTA5 holdout2497 源域')):
        grid = np.array([[100*result['models'][v][domain]['class_iou'][c] for v in VARIANTS] for c in range(19)])
        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(grid, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(10), VARIANTS)
        ax.set_yticks(range(19), [f'{i}: {CHINESE[i]} / {n}' for i, n in enumerate(CLASS_NAMES)])
        for c in range(19):
            for j in range(10):ax.text(j, c, f'{grid[c,j]:.2f}', ha='center', va='center', fontsize=8,
                                       color='white' if grid[c,j] > 65 else 'black')
        ax.set_title(title+'：各类别IoU（%），DINOv3 seed0；不是DINOv2新结果')
        fig.colorbar(im, ax=ax, label='IoU %');fig.tight_layout()
        fig.savefig(args.output / (domain+'_class_iou.png'), dpi=170, bbox_inches='tight')
        fig.savefig(args.output / (domain+'_class_iou.svg'), metadata={'Date': None}, bbox_inches='tight');plt.close(fig)
        header = '<tr><th>trainId</th><th>类别</th>'+''.join(f'<th>{v}</th>' for v in VARIANTS)+'</tr>'
        body = ''.join('<tr><td>'+str(c)+'</td><td>'+html.escape(CHINESE[c]+' / '+CLASS_NAMES[c])+'</td>'+
                       ''.join(f'<td>{number:.4f}</td>' for number in grid[c])+'</tr>' for c in range(19))
        body += '<tr><th colspan="2">19类 mIoU</th>'+''.join(f'<th>{100*result["models"][v][domain]["miou"]:.4f}</th>' for v in VARIANTS)+'</tr>'
        tables.append(f'<h2>{title}（%）</h2><img src="{domain}_class_iou.png"><div class="scroll"><table>{header}{body}</table></div>')
    page = '''<!doctype html><meta charset="utf-8"><title>19类IoU结果</title>
<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px;text-align:right}th{background:#eee}td:nth-child(2){text-align:left;white-space:nowrap}.scroll{overflow:auto}</style>
<h1>19类逐类IoU测试结果</h1><p>已完成的DINOv3十模型，seed0最终40000步；U关闭中心化，C开启。P=预测KL，F=Full CQE，A=Anchor-only，R=Residual-only。</p>
<p>IoU_c=TP_c/(TP_c+FP_c+FN_c)，mIoU是19类IoU均值。源域与目标域分开统计，ignore255不计入混淆矩阵。单训练种子点估计，不表示多种子显著性。</p>
'''+''.join(tables)
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    (args.output / 'classwise_iou.json').write_bytes((json.dumps(result, indent=2, allow_nan=False)+'\n').encode())
    print(f'Classwise source/target IoU report: {args.output}')
    return 0


if __name__ == '__main__':raise SystemExit(main())
