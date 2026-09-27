"""CPU-only version-gate tests for the isolated REIN environment."""

import pytest

from tools.check_rein_runtime import EXPECTED_VERSIONS, validate_versions


def test_rein_runtime_accepts_exact_pins() -> None:
    validate_versions(EXPECTED_VERSIONS.copy())


@pytest.mark.parametrize("name", ["torch", "mmcv", "mmseg", "mmdet"])
def test_rein_runtime_rejects_mixed_versions(name: str) -> None:
    versions = EXPECTED_VERSIONS.copy()
    versions[name] = "0.0.0"

    with pytest.raises(ValueError, match=f"{name} version"):
        validate_versions(versions)
