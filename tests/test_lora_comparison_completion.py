import json
import pytest
from causalq.runtime.identity import file_sha
import tools.check_lora_l0_batching as runner


def test_finishes_original_matrix_without_retrying_rejected_head_path():
    assert runner.comparison_modes()==('baseline','batch8','combined16')
    assert runner.comparison_modes(True)==('baseline','combined16')


@pytest.mark.parametrize('field,value',[('rejected_mode','combined16'),('migration_accepted',True),('pending_original_candidate','batch8'),('repair_source_sha256','0'*64)])
def test_rejects_scope_or_repair_changes(tmp_path,monkeypatch,field,value):
    evidence={'rejected_mode':'batch8','migration_accepted':False,'pending_original_candidate':'combined16',
              'repair_source_sha256':file_sha(runner.ROOT/'causalq/runtime/lora_batched.py')}
    evidence[field]=value
    (tmp_path/'phase41_conv_failure_27ab906.json').write_text(json.dumps(evidence))
    monkeypatch.setattr(runner,'DIRECTORY',tmp_path)
    with pytest.raises(ValueError):runner.comparison_modes(True)
