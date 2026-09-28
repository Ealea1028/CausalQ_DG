"""CPU-only version-gate tests for the isolated REIN environment."""

import pytest
from pathlib import Path

from tools.check_rein_runtime import EXPECTED_VERSIONS, validate_versions


def test_rein_runtime_accepts_exact_pins() -> None:
    validate_versions(EXPECTED_VERSIONS.copy())


@pytest.mark.parametrize("name", ["torch", "mmcv", "mmseg", "mmdet"])
def test_rein_runtime_rejects_mixed_versions(name: str) -> None:
    versions = EXPECTED_VERSIONS.copy()
    versions[name] = "0.0.0"

    with pytest.raises(ValueError, match=f"{name} version"):
        validate_versions(versions)


def test_bootstrap_recovery_preserves_failed_prefix_and_uses_only_mirror() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_rein_phase16.sh").read_text()
    assert "--recover-conda)" in script
    assert "rein-phase16-py310-cu118-tuna-retry1" in script
    assert "rein_phase16_runtime_tuna_retry1.json" in script
    assert '--prefix "$ENV_DIR" --override-channels' in script
    assert "--no-default-packages python=3.10 pip -y" in script
    assert 'test ! -e "$ENV_DIR"' in script
    assert 'git -C "$REIN_ROOT" status --porcelain' in script
    assert "rm " not in script
    assert "ssl_verify" not in script
