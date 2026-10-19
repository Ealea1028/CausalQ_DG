import json

import pytest

from tools.audit_final_target_results import ROOT,audit_final,validate_final
from tools.summarize_strict_dg import CLASS_NAMES,write_csv


def report():return json.loads((ROOT/'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes())


def test_final_all19_class_confusion_reduction_and_source_target_directions():
    audit=audit_final();rows={r['variant']:r for r in audit['rows']}
    assert audit['target_valid_pixels_per_model']==917018489
    assert rows['M1']['target_delta_vs_M0_pp']==pytest.approx(.450689299644)
    assert rows['M2']['target_delta_vs_M1_pp']==pytest.approx(.215755398452)
    assert rows['M2']['source_delta_vs_M1_pp']==pytest.approx(-.135957816947)
    assert audit['m3_status']=='not_run_hypothesis_unsupported' and audit['further_training_authorized'] is False
    # Coverage-only trace hashes are expected to coincide; confusion matrices differ.
    assert len({r['trace_sha256'] for r in report()['reports'].values()})==1
    assert report()['reports']['M0']['confusion_matrix']!=report()['reports']['M1']['confusion_matrix']


@pytest.mark.parametrize('change',['budget','coverage','freeze','checkpoint','target_tuning','class_iou','mean','counts','variants'])
def test_final_acceptance_rejects_inconsistent_metadata_or_metrics(change):
    r=report();m=r['reports']['M2']
    if change=='budget':m['optimizer_updates_this_run']=1
    if change=='coverage':m['sample_count']=50
    if change=='freeze':r['freeze_sha256']='0'*64
    if change=='checkpoint':m['identity']['checkpoint_sha256']='0'*64
    if change=='target_tuning':m['selection_or_tuning_on_target']=True
    if change=='class_iou':m['class_iou'][0]+=.01
    if change=='mean':m['target_miou']+=.01
    if change=='counts':m['confusion_matrix'][0][0]+=1
    if change=='variants':del r['reports']['M0']
    with pytest.raises(ValueError):validate_final(r)


def test_returned_final_bytes_cannot_be_silently_changed(tmp_path):
    original=ROOT/'experiments/STRICT_DG_PHASE26'
    for name in ('final_comparison.json','REPORT_SHA256SUMS'):(tmp_path/name).write_bytes((original/name).read_bytes())
    path=tmp_path/'final_comparison.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA256'):audit_final(tmp_path)


def test_summary_csv_preserves_units_and_19_class_names(tmp_path):
    rows=audit_final()['rows'];path=tmp_path/'table.csv'
    write_csv(path,rows,['variant','source_miou_pct','target_miou_pct','target_delta_vs_M1_pp'])
    text=path.read_text();assert len(text.splitlines())==5 and '65.87552967240946' in text
    assert len(CLASS_NAMES)==19 and CLASS_NAMES[13]=='car' and CLASS_NAMES[18]=='bicycle'
