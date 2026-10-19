"""Accept the frozen final comparison and independently reduce all confusion counts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.evaluate_target import ROOT, MANIFEST_SHA, FREEZE_PATH, frozen_contract, source_report

REPORT_SHA = '564d9bed8f4a8223fc6c76c6ae12b8be39737d1df4fa2bd1721196a77bb7aaf3'
SUMS_SHA = '2270cf6cc6138e4172a3509638d163c7943f30b5a05ca6448ea3310ba610f351'
PRODUCER = '440fd499ae05fe883b9d08a56ddfbea993924529'
FREEZE_SHA = '959b031dc5112ee021b1c14dfd8af15d80c998cf7ff510d0065893f69fda95d5'
VARIANTS = ('M0','M1','P1','M2')


def validate_final(report):
    freeze=frozen_contract()
    if (hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest()!=FREEZE_SHA
            or report['ok'] is not True or report['phase']!=26 or report['git_sha']!=PRODUCER
            or report['freeze_sha256']!=FREEZE_SHA or report['target_manifest_sha256']!=MANIFEST_SHA
            or report['target_data_accessed'] is not True or report['optimizer_updates_this_run']!=0
            or report['selection_or_tuning_on_target'] is not False
            or report['m3_status']!='not_run_hypothesis_unsupported'
            or set(report['reports'])!=set(VARIANTS) or set(report['target_miou'])!=set(VARIANTS)):
        raise ValueError('Frozen final comparison protocol differs')
    rows=[];totals=[];matrices=[]
    for variant in VARIANTS:
        r=report['reports'][variant];identity=r['identity'];frozen=freeze['variants'][variant]
        source=source_report(variant)
        if (r['ok'] is not True or r['phase']!=26 or r['variant']!=variant
                or identity!={'git_sha':PRODUCER,'variant':variant,'freeze_sha256':FREEZE_SHA,
                    'checkpoint_sha256':frozen['checkpoint_sha256'],'target_manifest_sha256':MANIFEST_SHA,
                    'geometry':freeze['target_geometry'],'checkpoint_selection':'fixed_final_iteration40000'}
                or r['environment']!=source['metadata']['environment'] or r['sample_count']!=500
                or r['valid_classes']!=19 or r['optimizer_updates_this_run']!=0
                or r['target_data_accessed'] is not True or r['selection_or_tuning_on_target'] is not False
                or r['target_labels_optimized'] is not False or r['confusion_counts_exact'] is not True
                or r['model_rng_unchanged'] is not True or r['source_miou']!=frozen['source_miou']
                or not math.isfinite(r['peak_reserved_gib']) or r['peak_reserved_gib']<=0):
            raise ValueError('Final variant identity/coverage/no-update policy differs')
        matrix=r['confusion_matrix']
        if (len(matrix)!=19 or any(len(x)!=19 for x in matrix)
                or any(type(x) is not int or x<0 for row in matrix for x in row)
                or len(r['class_iou'])!=19 or sum(map(sum,matrix))!=r['valid_pixel_count']):
            raise ValueError('Final target confusion/counts differ')
        ious=[]
        for c in range(19):
            union=sum(matrix[c])+sum(row[c] for row in matrix)-matrix[c][c]
            if union<=0:raise ValueError('Final target class coverage differs')
            iou=matrix[c][c]/union
            if not math.isclose(iou,r['class_iou'][c],abs_tol=1e-12,rel_tol=0):
                raise ValueError('Final target class IoU differs from counts')
            ious.append(iou)
        mean=math.fsum(ious)/19
        if any(not math.isclose(mean,x,abs_tol=1e-12,rel_tol=0) for x in (r['target_miou'],report['target_miou'][variant])):
            raise ValueError('Final target mIoU differs from counts')
        rows.append({'variant':variant,'seed':0,'updates':40000,'source_samples':2497,'target_samples':500,
                     'source_miou_pct':r['source_miou']*100,'target_miou_pct':r['target_miou']*100,
                     'peak_reserved_gib':r['peak_reserved_gib'],'target_class_iou':ious,
                     'checkpoint_sha256':identity['checkpoint_sha256']})
        totals.append(r['valid_pixel_count']);matrices.append(matrix)
    if len(set(totals))!=1 or any([sum(matrix[c]) for c in range(19)]!=[sum(matrices[0][c]) for c in range(19)] for matrix in matrices[1:]):
        raise ValueError('Ground-truth class counts differ across frozen models')
    for row in rows:
        for baseline in ('M0','M1'):
            reference=next(r for r in rows if r['variant']==baseline)
            for domain in ('source','target'):
                row[f'{domain}_delta_vs_{baseline}_pp']=row[f'{domain}_miou_pct']-reference[f'{domain}_miou_pct']
    return {'final_target_comparison_accepted':True,'report_sha256':REPORT_SHA,'producer_git_sha':PRODUCER,
            'freeze_sha256':FREEZE_SHA,'target_manifest_sha256':MANIFEST_SHA,'rows':rows,
            'target_valid_pixels_per_model':totals[0],'study_status':'closed_with_negative_anchor_hypothesis',
            'm3_status':'not_run_hypothesis_unsupported','further_training_authorized':False,
            'scope':'hash-pinned returned metadata; independent19-class confusion reduction; no local image/checkpoint replay',
            'trace_limit':'coverage-only traces contain IDs,shape and GT valid counts; identical hashes do not mean identical predictions',
            'inference_limit':'one seed; point estimates only; no seed variance,significance or general stable improvement claim'}


def audit_final(directory=ROOT/'experiments/STRICT_DG_PHASE26'):
    raw=(directory/'final_comparison.json').read_bytes();sums=(directory/'REPORT_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REPORT_SHA or hashlib.sha256(sums).hexdigest()!=SUMS_SHA:
        raise ValueError('Final returned evidence SHA256 differs')
    expected={line.split()[1]:line.split()[0] for line in sums.decode().splitlines()}
    if expected!={'final_comparison.json':REPORT_SHA}:raise ValueError('Final checksum coverage differs')
    return validate_final(json.loads(raw))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    args=p.parse_args(argv);result=audit_final();text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(text.encode())
    print(text,end='');return 0


if __name__=='__main__':raise SystemExit(main())
