"""Close the predeclared single-seed diagnostic study from accepted metadata only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from causalq.protocols.diagnostic_20261020 import ROOT, VARIANTS, mechanism
from causalq.protocols.diagnostic_final_source import ACCEPTANCE
from tools.plot_phase28_diagnostics import configure, CLASS_NAMES, value
from tools.train_strict_source import file_sha

NAMES = {'none': '无一致性', 'prediction': '预测KL', 'full': 'Full CQE', 'anchor': 'Anchor-only CQE', 'residual': 'Residual-only CQE'}


def result_tables(accepted):
    rows = []
    for r in accepted['rows']:
        v = r['variant'];base = 'C0' if v.startswith('C') else 'U0'
        rows.append({'variant': v, 'centered': r['centered'], 'consistency': mechanism(v),
                     'source_miou_pct': 100*r['source_miou'], 'target_miou_pct': 100*r['target_miou'],
                     'source_delta_vs_same_parameterization_pp': 100*(r['source_miou']-accepted['source_miou'][base]),
                     'target_delta_vs_same_parameterization_pp': 100*(r['target_miou']-accepted['target_miou'][base]),
                     'seed': 0, 'training_updates': 40000, 'source_samples': 2497, 'target_samples': 500})
    comparisons = [{'comparison': key, 'source_delta_pp': val,
                    'target_delta_pp': accepted['comparisons_percentage_points']['target'][key]}
                   for key, val in accepted['comparisons_percentage_points']['source'].items()]
    interactions = [{'interaction': key, 'source_delta_pp': val,
                     'target_delta_pp': accepted['interactions_percentage_points']['target'][key]}
                    for key, val in accepted['interactions_percentage_points']['source'].items()]
    classes = []
    for v in VARIANTS:
        r = next(r for r in accepted['rows'] if r['variant'] == v)
        baseline = next(r for r in accepted['rows'] if r['variant'] == ('C0' if v.startswith('C') else 'U0'))
        for c, name in enumerate(CLASS_NAMES):
            classes.append({'variant': v, 'class_id': c, 'class_name': name,
                            'target_iou_pct': 100*r['target_class_iou'][c],
                            'delta_vs_same_parameterization_pp': 100*(r['target_class_iou'][c]-baseline['target_class_iou'][c])})
    return rows, comparisons, interactions, classes


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def markdown_table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join(
        '| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/diagnostic_final_report_20261028')
    args = p.parse_args(argv);output = args.output;output.mkdir(parents=True, exist_ok=True)
    accepted_path = ACCEPTANCE / 'phase32_acceptance.json'
    accepted = json.loads(accepted_path.read_bytes())
    if accepted['accepted'] is not True:raise ValueError('Need accepted final target evidence')
    rows, comparisons, interactions, classes = result_tables(accepted)
    for name, data in (('comparison.csv', rows), ('predeclared_contrasts.csv', comparisons),
                       ('centering_interactions.csv', interactions), ('target_class_iou.csv', classes)):
        csv_write(output / name, data)
    configure();plt.rcParams['svg.hashsalt'] = 'causalq-final-report-20261028'
    def save(fig, name):
        fig.tight_layout();fig.savefig(output / (name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(output / (name+'.svg'), metadata={'Date': None}, bbox_inches='tight');plt.close(fig)
    x = np.arange(5);labels = ['无一致性', '预测KL', 'Full CQE', 'Anchor-only', 'Residual-only']
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for group, marker in (('U', 'o'), ('C', '^')):
        group_rows = [r for r in rows if r['variant'].startswith(group)]
        for ax, domain, title in zip(axes, ('source', 'target'), ('GTA5 holdout 2497', 'Cityscapes val500')):
            y = [r[f'{domain}_miou_pct'] for r in group_rows]
            ax.plot(x, y, marker+'-', label=group+'组', lw=2)
            above = (group == 'U') if domain == 'source' else (group == 'C')
            for index, number in enumerate(y):ax.annotate(f'{number:.4f}', (index, number), xytext=(0, 7 if above else -15), textcoords='offset points', ha='center', fontsize=9)
            ax.set_xticks(x, labels);ax.set_ylabel('mIoU（%）；单训练种子点估计');ax.set_title(title);ax.grid(alpha=.2)
            ax.margins(y=.15)
    axes[0].legend();save(fig, '01_source_target_matrix')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    y = np.arange(len(comparisons))
    axes[0].barh(y-.16, [r['source_delta_pp'] for r in comparisons], .3, label='source')
    axes[0].barh(y+.16, [r['target_delta_pp'] for r in comparisons], .3, label='target')
    axes[0].set_yticks(y, [r['comparison'] for r in comparisons]);axes[0].invert_yaxis();axes[0].legend()
    axes[0].axvline(0, color='gray', lw=.8);axes[0].set_xlabel('mIoU差（百分点）');axes[0].set_title('全部预先声明的主要比较')
    for i, domain in enumerate(('source', 'target')):
        axes[1].bar(np.arange(2)+(i-.5)*.3, [r[domain+'_delta_pp'] for r in interactions], .29, label=domain)
    axes[1].set_xticks(range(2), [r['interaction'] for r in interactions], rotation=12)
    axes[1].axhline(0, color='gray', lw=.8);axes[1].set_ylabel('mIoU交互差（百分点）');axes[1].legend()
    axes[1].set_title('正值：U组的增量大于C组；不代表多种子显著性')
    save(fig, '02_contrasts_interactions')
    selected = ('UP', 'UF', 'UA', 'UR', 'CP', 'CF', 'CA', 'CR')
    grid = np.array([[next(r['delta_vs_same_parameterization_pp'] for r in classes if r['variant']==v and r['class_id']==c)
                      for v in selected] for c in range(19)])
    bound = float(np.max(np.abs(grid)))
    fig, ax = plt.subplots(figsize=(12, 10));im = ax.imshow(grid, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_yticks(range(19), CLASS_NAMES);ax.set_xticks(range(len(selected)), selected)
    for i in range(19):
        for j in range(len(selected)):
            ax.text(j, i, f'{grid[i,j]:+.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(grid[i,j])>.55*bound else 'black')
    ax.set_title('目标域每类IoU变化：U方法减U0，C方法减C0；单位百分点')
    fig.colorbar(im, ax=ax, label='IoU差（百分点）');save(fig, '03_target_class_changes')

    diagnostics = {}
    for phase in (28, 31):
        a = json.loads((ACCEPTANCE / f'phase{phase}_acceptance.json').read_bytes());diagnostics.update(a['reports'])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for group, marker in (('U', 'o'), ('C', '^')):
        group_rows = [r for r in rows if r['variant'].startswith(group)]
        base = group+'0'
        for ax, component in zip(axes, ('full', 'anchor', 'residual')):
            reference = value(diagnostics[base], 'effects', 'rms_normalized_variance', component=component)
            for r in group_rows:
                v = r['variant'];horizontal = value(diagnostics[v], 'effects', 'rms_normalized_variance', component=component)/reference
                ax.scatter(horizontal, r['target_delta_vs_same_parameterization_pp'], marker=marker, s=55,
                           color='#365F87' if group=='U' else '#D68A39', label=group+'组' if v==base else None)
                ax.annotate(v, (horizontal, r['target_delta_vs_same_parameterization_pp']), xytext=(3, -10 if v=='C0' else 3), textcoords='offset points', fontsize=9)
            ax.set_xscale('log');ax.axhline(0, color='gray', lw=.6);ax.axvline(1, color='gray', lw=.6)
            ax.set_xlabel('源域归一化方差 / 同参数化无约束基线');ax.set_ylabel('目标mIoU变化（百分点）');ax.set_title(component)
    axes[0].legend(loc='lower right')
    save(fig, '04_stability_target_association')
    source_table = markdown_table(['模型', '中心化', '约束', '源域mIoU%', '目标域mIoU%', '目标Δ同组基线pp'],
                                 [[r['variant'], '是' if r['centered'] else '否', NAMES[r['consistency']],
                                   f'{r["source_miou_pct"]:.4f}', f'{r["target_miou_pct"]:.4f}',
                                   f'{r["target_delta_vs_same_parameterization_pp"]:+.4f}'] for r in rows])
    contrast_table = markdown_table(['比较', '源域Δpp', '目标域Δpp'],
                                   [[r['comparison'], f'{r["source_delta_pp"]:+.4f}', f'{r["target_delta_pp"]:+.4f}'] for r in comparisons])
    interaction_table = markdown_table(['交互', '源域Δpp', '目标域Δpp'],
                                      [[r['interaction'], f'{r["source_delta_pp"]:+.4f}', f'{r["target_delta_pp"]:+.4f}'] for r in interactions])
    report = '''# CausalQ 20261020 Diagnostic Plan 结项报告

本方案Phase28–32已完成：复用四个旧最终模型，新增六次各40000步训练，共240000次新增更新；
十模型各2497张完整源域诊断；六个新模型各500张最终Cityscapes评估，旧四模型目标结果复用。
当前没有新的AutoDL任务，也没有新增训练或调参授权。

## 十模型性能

所有模型均为seed0固定最终40000步。U为q=a+e，C为q=a+e−mean(e)。
预测KL、Full、Anchor-only、Residual-only各实验仅加入对应一种一致性项，权重1。
效应Residual是FP32 Full−重复Anchor的输出贡献，不是独立residual向量的因果效应。

'''+source_table+'''
CA是这十个单种子结果中的最高目标域点估计66.3220%，UA为66.0753%。这是完整矩阵的
描述，不是按目标域挑选“best checkpoint”；所有模型均已提前冻结并全部报告。
Anchor-only相对同参数化基线在U/C组均为正，Residual-only在U组小幅为正、C组为负。
Full CQE在C组相对C0为正，在U组相对U0为负，不能把一种参数化的结果直接推广到另一种。

## 预先声明的对比与中心化交互

'''+contrast_table+'\n'+interaction_table+'''
交互定义为U组效应增量减C组效应增量。目标域Anchor交互+0.2040pp、Residual交互
+0.3845pp，只表示本次配对种子的差值；不能据此宣称具有训练随机性意义的交互显著性。
CA目标分数高于UA，同时UA相对U0的增量大于CA相对C0的增量，这两点并不矛盾，
因为C0本身高于U0。源域差值没有同步对应目标域差值。

## 与效应幅值和稳定性诊断联合解释

- UA相对U0的Anchor归一化方差下降81.11%，原始方差下降22.47%，但RMS增加85.32%；
  Residual RMS从0.3090增至1.5465，原图Anchor/Residual空间Pearson为−0.8070。
  Full归一化方差反而增加25.61%。UA目标域提高0.8662pp，因此本次收益不能简单归因于
  “完整效应更稳定”或“Anchor幅值被压小”；较大分量和较强抵消同时存在。
- UF的Full归一化方差下降57.49%、原始方差下降38.10%，但目标域相对U0下降0.4803pp。
  这是“该源域稳定性指标下降就足以保证目标域提升”的直接反例；不否认该指标的描述用途。
- UR/CR的Residual归一化方差分别下降75.83%/99.38%，但原始方差分别增加210.05%/130.15%，
  RMS分别增加220.39%/225.73%。目标变化分别为+0.2633/−0.1212pp。不能把归一化后的
  形状稳定写成绝对幅值波动下降，也不能据此统一解释两组泛化方向。
- CA相对C0的Anchor归一化方差下降77.81%，Full归一化方差下降69.12%，目标提高0.6622pp。
  但Anchor RMS增加57.36%，Residual的GT-present与all-class统计明显不同；分层结果须一起保留。
- 旧C0两视图原始Anchor/Residual方差比27.60的负结果仍成立；新五视图/归一化指标回答不同问题，
  不推翻旧“Anchor天然更稳定”假设未获支持的结论。选择性正则化有用与天然排序成立不是同一命题。

非中心化存在a→a+t、e→e−t的分解不唯一性；完整q保持不变时Anchor分解可改变。
因此观察到的Anchor/Residual重分配不能直接解释成被唯一识别的语义因果成分。
Full=Anchor+Residual，方差恒等式为Var(Full)=Var(A)+Var(R)+2Cov_style；空间Pearson与风格协方差不同。

## 梯度贡献与Query责任

总目标Query梯度cosine宏平均：UA/UR=0.8624/0.8290，CA/CR=0.6891/0.6512，均为正。
direct Anchor的平均投影贡献分别约5.13%/9.65%/6.19%/6.80%，说明选择性损失不全通过完整q路径
访问Anchor。中心化raw residual梯度也不等于复合q节点梯度。范数/投影是局部损失敏感度，
不包含AdamW动量、预条件和weight decay，不能代称实际优化器更新贡献或Query因果必要性。

UA责任熵较高（归一化像素熵0.7526）、photo1 top1切换率约2.879%，同时JS约0.0001414。
CR熵较低（0.3019）、切换率约0.535%，GT类别区域存在若干单slot主导。因此低JS、低切换、
平均责任接近0.5均不能单独证明更好的语义分工。各模型各类别原始统计、direct路径及null计数
保留在源域图表和逐类表中，不用一张全局平均表代替完整诊断。

## 每类性能、可视化与证据

UA和CA相对各自基线均为10/19类别IoU上升，其余9类下降，不能描述成所有类别一致改善。
UA的train/motorcycle/bus分别约+7.94/+5.81/+2.81pp，三类合计对19类均值贡献约+0.8713pp，
接近整体+0.8662pp；CA的train/motorcycle/bicycle分别约+6.65/+4.42/+3.37pp，三类贡献
约+0.7600pp，其他类别净贡献为负。增益集中在少数类别，须结合完整类别表和单种子限制解释。

- [完整源域图册](../diagnostic_final_source_20261028/index.html)：幅值、方差、相关/抵消、梯度、责任及类别分层。
- [源域详细解读](../../docs/DIAGNOSTIC_PHASE31_RESULTS.md)。三组干预图复用Phase28，不重复推理或导出。
- comparison.csv：十模型性能和同参数化基线差值；predeclared_contrasts.csv和centering_interactions.csv：全部固定对比。
- target_class_iou.csv：十模型全部19类IoU及同参数化基线差；03_target_class_changes展示八种有约束方法。
- 01_source_target_matrix：完整性能矩阵；02_contrasts_interactions：对比/交互；04_stability_target_association：
  源域归一化方差变化与目标性能变化的描述性散点。每个点是一个模型，不是独立随机重复，不作相关显著性推断。

## 范围与限制

1. 只有一个训练种子，没有训练种子方差、置信区间或显著性结论；也没有事后只给最高分方法补种子。
2. 新方案是在研究者已知旧Cityscapes结果后提出的探索性对照；新增配置与检查点在新目标评估前冻结，
   训练/诊断使用源域，最终目标评估只报告，不据此再调损失、权重或预算。
3. 本机检查导出哈希和覆盖轨迹，独立归约十份19×19整数混淆矩阵；检查点与真实数据字节留在实例。
   源域图表来自已接受摘要，没有在本机重新计算逐图明细。target_eval.jsonl只含ID、形状和GT有效数，
   相同trace哈希不代表相同预测，也不足以构造配对逐图预测bootstrap。
4. 训练为冻结DINOv3-L、原GTA5 22469/2497划分、原增强与优化器、两视图CE，最终固定40000步。
   目标评估为冻结Cityscapes500、native1024×2048整图BF16、19类trainIds/ignore255，无resize/crop/flip/TTA/sliding。

## 结项

本轮支持的有限结论是：该配对种子下，Anchor-only约束在两种参数化下均提高目标mIoU；
选择性约束、分量尺度、抵消和Query责任分布同时变化，当前证据不足以确定唯一收益机制。
源域归一化稳定性不是目标收益的充分条件。结果允许负向和混合结论，未追加新机制、训练或目标域调参。
完成当前方案后停止；保留AutoDL原始记录、最终检查点与冻结提交，不要求新的AutoDL操作。
'''
    (output / 'REPORT.md').write_bytes(report.encode())
    provenance = {'schema_version': 1, 'study_status': 'complete_single_seed_exploratory',
                  'accepted_target_sha256': file_sha(accepted_path), 'generator_sha256': file_sha(Path(__file__)),
                  'tables': {'comparison': rows, 'predeclared_contrasts': comparisons, 'interactions': interactions},
                  'source_acceptance_sha256': {str(phase): file_sha(ACCEPTANCE/f'phase{phase}_acceptance.json') for phase in (28, 29, 30, 31)},
                  'additional_training_authorized': False}
    (output / 'results.json').write_bytes((json.dumps(provenance, indent=2, allow_nan=False)+'\n').encode())
    body = '<!doctype html><meta charset="utf-8"><title>CausalQ结项结果</title><h1>CausalQ单种子最终结果</h1><p>Phase28–32完成。请结合REPORT.md阅读证据范围和限制。</p><a href="REPORT.md">完整结项报告</a>'
    for image in sorted(output.glob('*.png')):body += f'<h2>{image.stem}</h2><img style="width:100%;max-width:1500px" src="{image.name}">'
    (output / 'index.html').write_text(body, encoding='utf-8')
    print(f'Final report, four figures and full tables: {output}')
    return 0


if __name__ == '__main__':raise SystemExit(main())
