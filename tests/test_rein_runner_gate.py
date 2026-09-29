import pytest
import torch

from tools.rein_runner_gate import rolling_save, validate_fresh_report
from tools.rein_training_state import capture_rng, compare_trees


def test_rolling_checkpoint_retains_only_latest_and_previous_without_rng_drift(tmp_path):
    state = capture_rng()
    for update in (5, 10, 15, 20):
        payload = dict(model={'weight': torch.tensor([float(update)])},
                       metadata={'optimizer_updates': update}, rng=state)
        result = rolling_save(tmp_path, payload)
        assert result['optimizer_updates'] == update
        assert result['bytes'] > 0
    assert {path.name for path in tmp_path.iterdir()} == {'last.pth', 'previous.pth'}
    assert torch.load(tmp_path/'last.pth', weights_only=True)['metadata']['optimizer_updates'] == 20
    assert torch.load(tmp_path/'previous.pth', weights_only=True)['metadata']['optimizer_updates'] == 15
    assert compare_trees(capture_rng(), state) == 0


def test_existing_pending_checkpoint_is_preserved(tmp_path):
    pending = tmp_path / 'pending.pth'
    pending.write_bytes(b'failed-write-evidence')
    with pytest.raises(FileExistsError):
        rolling_save(tmp_path, {})
    assert pending.read_bytes() == b'failed-write-evidence'


def test_fresh_report_gate_rejects_failed_or_wrong_producer():
    report = dict(ok=True, fresh_process_restore_replay_verified=True,
                  evaluation_git_sha='1671f1a07441868e464776cfbf9305308187c289',
                  independent_process_count=2, target_labels_optimized=False)
    validate_fresh_report(report)
    for key, value in [('ok', False), ('evaluation_git_sha', 'wrong'), ('independent_process_count', 1),
                       ('target_labels_optimized', True)]:
        with pytest.raises(ValueError):
            validate_fresh_report(dict(report, **{key: value}))
