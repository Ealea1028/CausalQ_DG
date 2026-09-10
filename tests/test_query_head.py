import causalq


def test_package_imports_during_phase_one() -> None:
    assert causalq.__version__ == "0.1.0"

