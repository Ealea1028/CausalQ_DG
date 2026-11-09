"""Verify the returned HPI target package and render the frozen study report."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META = ROOT/'experiments/HPI_DINOV2_20261031'
OUTPUT = ROOT/'analysis/hpi_dinov2_final'
TARGET_SHA = '116c782de8a080b0b09d74c68f4177da41fd9e260cad9e0943e47765be91a4a6'
CLASSES = 'road sidewalk building wall fence pole traffic_light traffic_sign vegetation terrain sky person rider car truck bus train motorcycle bicycle'.split()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_metric(metric, ids):
    if metric['ids'] != ids or metric['sample_count'] != 500:
        raise ValueError('Target membership/coverage differs')
    matrix = np.asarray(metric['confusion_matrix'])
    if matrix.shape != (19,19) or matrix.dtype.kind not in 'iu' or (matrix < 0).any():
        raise ValueError('Target confusion must be a nonnegative integer 19x19 matrix')
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.full(19,np.nan), where=union>0)
    if not np.allclose(iou, metric['class_iou'], atol=1e-14, rtol=0) or not np.isclose(np.nanmean(iou),metric['miou'],atol=1e-14,rtol=0):
        raise ValueError('Target score differs from integer confusion reduction')
    if int(matrix.sum()) != metric['valid_pixels']:
        raise ValueError('Target valid pixel count differs')
    return iou.tolist()


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected(rows, metric, **filters):
    matches = [r['metrics'][metric]['mean'] for r in rows
               if all(r.get(k)==v for k,v in filters.items()) and metric in r['metrics']]
    if len(matches) != 1:
        raise ValueError(f'Ambiguous diagnostic metric {metric} {filters}')
    return matches[0]


def main():
    from causalq.runtime.hpi_source_batch import summarize_main_path
    from causalq.runtime.hpi_target import target_contract
    OUTPUT.mkdir(parents=True,exist_ok=True)
    export = ROOT/'phase37_export/export'
    archive_path = ROOT/'phase37_export/export.zip'
    if sha(archive_path) != TARGET_SHA:
        raise ValueError('Returned target ZIP hash differs')
    files = {}
    with zipfile.ZipFile(archive_path) as archive:
        for line in (export/'SHA256SUMS').read_text().splitlines():
            expected,name = line.split('  ',1)
            if name in files or sha(export/name)!=expected or archive.read(name)!=(export/name).read_bytes():
                raise ValueError(f'Returned file differs {name}')
            files[name]=expected
        if set(archive.namelist()) != set(files)|{'SHA256SUMS'}:
            raise ValueError('Archive coverage differs')
        if archive.read('SHA256SUMS') != (export/'SHA256SUMS').read_bytes():
            raise ValueError('Archive checksum manifest differs')
    contract = target_contract()
    ids = (ROOT/contract['target_manifest']).read_text().splitlines()
    status,report = read(export/'status.json'),read(export/'target_report.json')
    identity = {'evaluation_git_sha':'f6c647e20f007f15ae1876a0d02c87746ed156b0',
                'target_freeze_sha256':sha(META/'target_freeze.json')}
    assert status['status']=='complete' and status['exit_code']==0 and status['formal_optimizer_updates']==0
    assert status['target_data_accessed'] and report['identity']==identity and status['git_sha']==identity['evaluation_git_sha']
    assert report['formal_optimizer_updates']==0 and report['sample_count_per_model']==500
    actual,locked = read(export/'runtime_actual.json'),read(META/'runtime.json')
    assert all(actual[k]==locked[k] for k in ('torch','gpu','python','cuda_runtime'))
    acceptances = {p:read(META/f'phase{p}_acceptance.json') for p in (34,35,36)}
    for phase,accepted in acceptances.items():
        for name,checksum in accepted['files_verified'].items():
            assert sha(ROOT/f'phase{phase}_export/export'/name)==checksum
    target_metrics,table,diagnostics,class_rows,routes = {},[],[],[],[]
    for name,frozen in contract['models'].items():
        metric = read(export/name/'target_metrics.json')
        iou = verify_metric(metric,ids)
        assert metric['checkpoint']==frozen and metric['geometry']==[1024,2048]
        assert metric['miou']==report['target_miou'][name] and metric['miou_percent']==100*metric['miou']
        target_metrics[name]=metric
        phase = 34 if name in ('U0','UA') else 35 if name in ('B1','B2','B3') else 36
        source = ROOT/f'phase{phase}_export/export/source'/name
        effects = read(source/'source_effect_summary.json')['image_macro_statistics']
        responsibility = read(source/'class_responsibility.json')['image_macro_statistics']
        formal = acceptances[phase]['formal_report']
        assert formal['head_updates'][name]==40000
        for row in effects+responsibility:
            for key,value in row['metrics'].items():
                diagnostics.append({'model':name,**{k:row.get(k) for k in
                    ('kind','population','class_id','component','scope','query_r','view_index','observation_unit')},
                    'metric':key,**{k:value.get(k) for k in ('mean','median','count','null_count')}})
        row = {'model':name,'backbone':'DINOv2-L HPI-p16','seed':0,
               'trainable_parameters':formal['per_head_trainable_parameters'][name],
               'source_miou_percent':100*frozen['source_miou'],'target_miou_percent':100*metric['miou']}
        for component in ('anchor','residual','full'):
            for key in ('raw_variance','rms_normalized_variance','mean_abs','joint_rms'):
                row[f'{component}_{key}'] = selected(effects,key,kind='effects',population='gt_present',class_id=-1,component=component)
        for key in ('mean_responsibility','pixel_entropy','usage_entropy','effective_query_count'):
            row[f'slot_{key}'] = selected(effects,key,kind='responsibility',population='all_classes',class_id=-1,scope='gt_class',query_r=0,view_index=0)
        for key in ('mean_responsibility','pixel_entropy','js_divergence','top1_switch_rate'):
            row[f'class_{key}'] = selected(responsibility,key,kind='class_responsibility',population='gt_present',class_id=-1,scope='all_valid',view_index=1)
        table.append(row)
        source_metrics=read(source/'source_metrics.json')
        for c,score in enumerate(iou):
            class_rows.append({'model':name,'class_id':c,'class':CLASSES[c],
                'source_iou_percent':100*source_metrics['class_iou'][c],'target_iou_percent':100*score})
        if name in ('B2','B3','B4','B5'):
            path = read(source/'main_path128.json')
            macro = summarize_main_path(path['by_image'])
            route = read(source/'routing_summary.json')
            routes.append({'model':name,'images':128,'mean_on':fmean(macro['on']),'mean_off':fmean(macro['off']),
                           **{k:macro[k] for k in ('mean_tau','variance','positive_fraction','negative_fraction','near_zero_fraction')},
                           'gamma':route['gamma'],'mean_gate':fmean(route['gate_mean']),
                           'mean_injection_rms':fmean(route['injection_feature_update_rms'])})
    contrasts=[]
    for a,b in contract['comparisons']:
        delta=100*(target_metrics[a]['miou']-target_metrics[b]['miou'])
        assert np.isclose(delta,report['comparisons_pp'][f'{a}_minus_{b}'],atol=1e-14,rtol=0)
        contrasts.append({'contrast':f'{a} - {b}','target_delta_pp':delta,
                          'source_delta_pp':100*(contract['models'][a]['source_miou']-contract['models'][b]['source_miou'])})
    assert len(set(m['valid_pixels'] for m in target_metrics.values()))==1
    acceptance={'accepted':True,'phase':37,'producer_sha':identity['evaluation_git_sha'],'archive_sha256':TARGET_SHA,
                'files_verified':files,'all_seven_integer_confusions_reduced':True,'target_ids_checked_per_model':500,
                'valid_pixels_per_model':target_metrics['U0']['valid_pixels'],'target_report':report}
    (META/'phase37_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n',encoding='utf-8',newline='\n')
    write_csv(OUTPUT/'models.csv',table)
    write_csv(OUTPUT/'contrasts.csv',contrasts)
    write_csv(OUTPUT/'class_iou.csv',class_rows)
    write_csv(OUTPUT/'diagnostics.csv',diagnostics)
    write_csv(OUTPUT/'main_path.csv',routes)
    engineering=[]
    for phase,accepted in acceptances.items():
        formal=accepted['formal_report']
        local=ROOT/f'phase{phase}_export/export'
        engineering.append({'phase':phase,'training_sha':formal['git_sha'],'formal_updates':formal['total_optimizer_updates'],
             'logical_batch':1,'physical_backbone_views':32,'workers':12,'prefetch':2,
             'timing':formal.get('training_invocation_timing'),'invocation_seconds':formal['invocation_elapsed_seconds'],
             'peak_reserved_bytes':formal['peak_reserved_bytes'],
             'training_activity':read(local/'training_activity.json') if (local/'training_activity.json').exists() else None,
             'invocation_telemetry':read(local/'gpu_telemetry_summary.json')})
    (OUTPUT/'engineering.json').write_text(json.dumps(engineering,indent=2)+'\n',encoding='utf-8')
    render(table,contrasts,class_rows,routes,report)
    print(json.dumps({'verified_files':len(files),'models':len(table),'contrasts':len(contrasts),'diagnostic_rows':len(diagnostics)},indent=2))


def render(table, contrasts, class_rows, routes, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'hpi-dinov2-final'})
    names=[r['model'] for r in table]
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].plot(names,[r['target_miou_percent'] for r in table],'o-',color='#246b93')
    ax[0].set(ylabel='Cityscapes val500 mIoU (%)',title='Fixed final models, seed 0')
    for i,r in enumerate(table):ax[0].annotate(f"{r['target_miou_percent']:.4f}",(i,r['target_miou_percent']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax[0].margins(y=.3)
    ax[1].barh([r['contrast'] for r in contrasts],[r['target_delta_pp'] for r in contrasts],color=['#287c75' if r['target_delta_pp']>=0 else '#b34d45' for r in contrasts])
    ax[1].axvline(0,color='black',lw=.7)
    ax[1].set(xlabel='Target mIoU change (percentage points)',title='Predeclared contrasts')
    fig.savefig(OUTPUT/'target_comparisons.svg',metadata={'Date':None});fig.savefig(OUTPUT/'target_comparisons.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),layout='constrained')
    values=np.array([r['target_iou_percent'] for r in class_rows]).reshape(7,19)
    im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set(yticks=range(7),yticklabels=names,xticks=range(19),xticklabels=CLASSES,title='Target per-class IoU (%)')
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
    fig.colorbar(im,ax=ax)
    fig.savefig(OUTPUT/'class_iou.svg',metadata={'Date':None});plt.close(fig)
    def fmt(value):return 'null' if value is None else f'{value:.6g}'
    lines=['# HPI DINOv2 项目最终报告','',
           '七组 seed0 模型均完成 40000 更新，共 280000 次正式更新。Phase37 对七组固定最终头各评估 Cityscapes val500，新增更新为 0。七份整数混淆矩阵和图像清单已独立核对。','',
           '| 模型 | 可训练参数 | source mIoU% | target mIoU% | Anchor raw Var | Anchor RMS-norm Var | Residual raw Var | Residual RMS-norm Var |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in table:lines.append(f"| {r['model']} | {r['trainable_parameters']} | {r['source_miou_percent']:.4f} | {r['target_miou_percent']:.4f} | "+' | '.join(fmt(r[k]) for k in ('anchor_raw_variance','anchor_rms_normalized_variance','residual_raw_variance','residual_rms_normalized_variance'))+' |')
    lines+=['','全部模型骨干为 DINOv2-L HPI-p16、seed=0。效应主表按 GT-present 类别先图内平均、再等权图像宏平均。平均绝对幅值、RMS、slot 责任与熵、class 责任与熵及 JS 见 models.csv；原始与归一化、二视图与五视图、逐类与 all-class 补充、相关与抵消及 null 计数见 diagnostics.csv。slot 主表取原图 GT-class 区域第 0 槽；class 取第 1 个干预视图的 all-valid 区域并按 GT-present 类别汇总。','',
            '| 预声明对比 | target 变化 pp | source 变化 pp |','| --- | --- | --- |']
    lines += [f"| {r['contrast']} | {r['target_delta_pp']:+.6f} | {r['source_delta_pp']:+.6f} |" for r in contrasts]
    lines += ['','B2−B1 为 +0.138558 pp，是相邻消融中最大的正变化。B3−B2 为 −0.004353 pp，B4−B3 为 −0.072754 pp，B5−B4 为 +0.011731 pp。结果没有呈现逐项叠加的单调改善。B5 相对 U0 和 UA 分别为 +0.110034 pp、+0.033311 pp。差值均为单种子点估计。','',
              '![目标比较](target_comparisons.svg)','','![逐类 IoU](class_iou.svg)','',
              '主路径干预在冻结的 128 张源图上、对 GT-present 类别和五视图计算。main_path.csv 等权汇总各图，保留 on/off 概率、带符号效应、方差、正负及近零比例、gamma、gate 和注入 RMS。原始读出效应与主路径效应分别报告。','',
              '| 模型 | mean on | mean off | mean τ | Var τ | 负效应比例 | gate | 注入 RMS |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    lines += ['| '+r['model']+' | '+' | '.join(fmt(r[k]) for k in ('mean_on','mean_off','mean_tau','variance','negative_fraction','mean_gate','mean_injection_rms'))+' |' for r in routes]
    lines += ['','B1–B5 的 Residual 是结构性零，RMS 归一化方差未定义。两个 Anchor 槽位重复，有效独立 Query 数为 1；责任高熵源自结构退化，不能解释为解耦。读出方差下降、Query 删除后的变化或 attention 图均不足以独立证明现实因果机制或泛化改善。新增模块和损失的效果归属于整个组合，未隔离为 Anchor 参数本身的效果。','',
              '运行环境为 RTX 6000D、PyTorch 2.7.0+cu128。训练逻辑 batch1，物理骨干视图 batch32；Phase35/36 使用 12 spawn worker、8 样本装载包、prefetch2 和双缓冲。U0/UA 复用 v2 已完成训练。源诊断修复后为 12 worker、实际装载 batch4，逐图恢复。工程分组耗时、峰值显存、功耗、遥测与稳态口径见 engineering.json。Phase35 稳态 GPU 活动中位数 68%，Phase36 为 62%，均低于 80% 工程目标；v3 计时提升 7.85%，低于 10% 目标。Phase34 全流程遥测包含长时间诊断，不能充当训练稳态。','',
              f"Phase37 评估段耗时 {report['invocation_evaluation_seconds']:.3f} 秒，组级峰值 allocated {report['peak_allocated_bytes']/2**30:.3f} GiB、reserved {report['peak_reserved_bytes']/2**30:.3f} GiB。该耗时口径排除初始化，含本次记录恢复/读取；显存是七头共享组峰值。目标评估采用原生 1024×2048 整图 BF16、batch2 共享骨干特征、各头 argmax。",'',
              'Phase35 由原训练 SHA c4a7b6f 完成 120000 更新，诊断在 3dfbdca 续跑，180+2317 张记录覆盖 2497 图，续跑更新为 0。最初准备曾缺少 PyYAML，已在正式运行前修复；可选 compile 失败后冻结 eager+SDPA。逐图、最终检查点和配置身份已分别冻结。Phase35 示例 ZIP 未在本地收到，保留远程哈希；Phase34/36 示例 ZIP 已核验。返回包未保留 Phase35 初次训练完整计时分项，engineering.json 用 null 保留缺失项。','',
              '| 历史背景模型 | backbone | GPU | target mIoU% |','| --- | --- | --- | --- |',
              '| 旧 U0 | DINOv3-L/16 | RTX 4090 D | 65.2091 |','| 旧 UA | DINOv3-L/16 | RTX 4090 D | 66.0753 |','',
              '历史 U0 环境见 experiments/STRICT_DG_PHASE26/final_comparison.json，旧 UA 环境见 experiments/DIAGNOSTIC_20261020/phase29_acceptance.json。硬件与骨干不同，历史分数只作背景，不构成严格配对的骨干比较。','',
              '本研究为单训练种子，并在已知旧目标反馈后制定，结论限于探索性点估计。没有跨训练种子不确定性估计。实现只迁移 HPI 的去 VLM 骨干适配路径，未复现包含 VLM 的 HPI 全模型。全部八个预声明对比及负结果均已报告。','',
              '验收依据保存在 experiments/HPI_DINOV2_20261031/phase37_acceptance.json 和 target_freeze.json。源域包分别绑定 Phase34、35、36 验收清单。模型/优化器检查点与完整逐图记录保留 AutoDL，Git 保存精简结果及身份。Phase33–38 完成，本方案已结项。']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
