"""Generate source/target tables and a descriptive seed0 figure from accepted metadata."""

import argparse
import csv
import io
import json
from pathlib import Path

from tools.audit_final_target_results import audit_final, ROOT

CLASS_NAMES=('road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
             'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')


def write_csv(path,rows,fields):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows);path.write_bytes(out.getvalue().encode())


def generate(output):
    summary=audit_final();output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    fields=['variant','seed','updates','source_samples','target_samples','source_miou_pct','target_miou_pct',
            'source_delta_vs_M0_pp','target_delta_vs_M0_pp','source_delta_vs_M1_pp','target_delta_vs_M1_pp','peak_reserved_gib']
    write_csv(output/'comparison.csv',summary['rows'],fields)
    per_class=[{'class_id':i,'class_name':name,**{r['variant']+'_iou_pct':r['target_class_iou'][i]*100 for r in summary['rows']}}
               for i,name in enumerate(CLASS_NAMES)]
    write_csv(output/'target_class_iou.csv',per_class,['class_id','class_name','M0_iou_pct','M1_iou_pct','P1_iou_pct','M2_iou_pct'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'causalq-strict-final-v1','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.6),gridspec_kw={'width_ratios':[1.1,1]})
    names=[r['variant'] for r in summary['rows']];x=list(range(4))
    for offset,domain,color,label in [(-.12,'source','#2563a6','GTA5 source holdout'),(.12,'target','#c66a16','Cityscapes target')]:
        values=[r[domain+'_miou_pct'] for r in summary['rows']]
        axes[0].scatter([v+offset for v in x],values,c=color,label=label,s=40)
        for index,value in enumerate(values):
            axes[0].annotate(f'{value:.4f}',(index+offset,value),xytext=(0,8 if domain=='source' else -15),textcoords='offset points',ha='center',fontsize=9)
        delta=[r[domain+'_delta_vs_M0_pp'] for r in summary['rows']]
        axes[1].scatter(delta,[v+offset for v in x],c=color,s=40,label=label)
    axes[0].set(xticks=x,xticklabels=names,ylim=(63.5,75.5),ylabel='mIoU (%)',title='Frozen final checkpoints (seed 0)')
    axes[1].axvline(0,color='#7d7d7d',linewidth=.9)
    axes[1].set(yticks=x,yticklabels=names,xlabel='Difference from M0 (percentage points)',title='Descriptive differences, one seed')
    axes[1].invert_yaxis()
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='center left',fontsize=9,frameon=False)
    fig.text(.5,.025,'Left y-axis is truncated. Source: 2497 images, long edge 1024. Target: 500 images, native 1024 x 2048.\n'
             'Single-seed point estimates, no confidence intervals. M3 not run: raw anchor-stability prerequisite failed.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(output/'final_comparison.png',dpi=180,metadata={'Software':'CausalQ-DG'})
    svg_path=output/'final_comparison.svg'
    fig.savefig(svg_path,metadata={'Date':None});plt.close(fig)
    svg_path.write_bytes(('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n').encode())
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'analysis/strict_dg_final')
    args=p.parse_args(argv);summary=generate(args.output_dir)
    print(json.dumps({'status':summary['study_status'],'output':str(args.output_dir)},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
