"""One AutoDL job: formal M1 hypothesis check, then M3 training only if supported."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.inspect_gta5_source_structure import _git
from tools.trained_anchor_stability import validate_stability
from tools.train_formal_source import write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG/phase22/M0_M1_FORMAL_SEED0_2537653/M1'))
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args(argv)
    evidence = args.output_root / 'stability/stability_report.json'
    common = ['--data-root', str(args.data_root), '--pretrained-root', str(args.pretrained_root)]
    subprocess.run([sys.executable,'-m','tools.trained_anchor_stability','--output-dir',str(evidence.parent),
                    '--baseline-root',str(args.baseline_root),*common],check=True)
    report = validate_stability(evidence,_git('rev-parse','HEAD'))
    result = {'schema_version':1,'phase':25,'ok':True,'git_sha':report['git_sha'],
              'stability_report':report,'m3_started':False,'target_data_accessed':False,
              'status':'hypothesis_not_supported_stop_new_losses'}
    if report['decision']['m3_training_authorized']:
        subprocess.run([sys.executable,'-m','tools.train_formal_source','--variant','M3',
                        '--output-root',str(args.output_root),'--stability-report',str(evidence),*common],check=True)
        result.update(m3_started=True,status='m3_formal_complete',
                      formal_report=json.loads((args.output_root/'M3/formal_report.json').read_bytes()))
    write_json(args.output_root/'job_report.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
