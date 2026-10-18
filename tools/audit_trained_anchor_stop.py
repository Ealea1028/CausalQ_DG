"""Accept the returned scientific stop certificate; raw rows stay on AutoDL."""

import hashlib
import json
import math
from pathlib import Path

from tools.audit_formal_source_pair import ROOT, PAIR_SHA, audit_pair

JOB_SHA = '0184a3d57d88c284a5a18eda49ed121fb0de2a367b90de6ad228541652343460'
SUMS_SHA = 'a1ecb2cfb9155a4f12f490139f820c7d699f0470e08b70c30f392fbbb4aa356c'
PRODUCER = '148746b6e5d6a36b413a8b6d104652860cab5a36'


def validate_stop(job):
    audit_pair()
    m1 = json.loads((ROOT/'experiments/STRICT_DG_PHASE22/m1_formal_report.json').read_bytes())
    r = job['stability_report']; d = r['decision']
    if (job['ok'] is not True or job['phase'] != 25 or job['git_sha'] != PRODUCER
            or job['status'] != 'hypothesis_not_supported_stop_new_losses' or job['m3_started'] is not False
            or job['target_data_accessed'] is not False or r['ok'] is not True or r['git_sha'] != PRODUCER
            or r['sample_count'] != 2497 or r['backbone_forward_calls'] != 4994
            or r['checkpoint_sha256'] != m1['checkpoint_sha256'] or r['checkpoint_training_updates'] != 40000
            or r['baseline_pair_report_sha256'] != PAIR_SHA or r['optimizer_updates_this_run'] != 0
            or r['read_only_state_unchanged'] is not True or r['target_data_accessed'] is not False
            or r['environment'] != m1['metadata']['environment']
            or d['m3_training_authorized'] is not False or d['images_anchor_lower'] != 0
            or not 0 < d['raw_residual_variance'] < d['raw_anchor_variance']
            or not math.isclose(d['raw_anchor_variance']/d['raw_residual_variance'],d['raw_anchor_residual_ratio'],rel_tol=1e-12)):
        raise ValueError('Trained anchor scientific-stop certificate differs')
    return {'trained_anchor_hypothesis_supported':False,'m3_started':False,'job_report_sha256':JOB_SHA,
            'raw_anchor_residual_ratio':d['raw_anchor_residual_ratio'],'images_anchor_lower':0,'source_count':2497,
            'scope':'returned hash-pinned producer certificate; full sample rows not locally supplied',
            'target_data_accessed':False}


def audit_stop(directory=ROOT/'experiments/STRICT_DG_PHASE25'):
    raw=(directory/'job_report.json').read_bytes();sums=(directory/'REPORT_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=JOB_SHA or hashlib.sha256(sums).hexdigest()!=SUMS_SHA:
        raise ValueError('Scientific-stop evidence SHA256 differs')
    expected={line.split()[1]:line.split()[0] for line in sums.decode().splitlines()}
    report=json.loads(raw);embedded=(json.dumps(report['stability_report'],indent=2)+'\n').encode()
    if (set(expected)!={'job_report.json','stability/stability_report.json'} or expected['job_report.json']!=JOB_SHA
            or hashlib.sha256(embedded).hexdigest()!=expected['stability/stability_report.json']
            or (directory/'stability_report.json').read_bytes()!=embedded):
        raise ValueError('Embedded stability evidence differs')
    return validate_stop(report)
