"""Launcher failures must not terminate a caller's interactive shell."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = "D:/Git/Git/bin/bash.exe" if Path("D:/Git/Git/bin/bash.exe").exists() else shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash unavailable")
SCRIPT = (ROOT / "scripts/launch_anchor_query_effect_phase19.sh").as_posix()


def test_failed_launcher_keeps_parent_alive():
    result = subprocess.run([BASH, "-c", '"$2" "$1" invalid; code=$?; echo "parent_alive:$code"', "test", SCRIPT, BASH],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "parent_alive:2" in result.stdout
    assert "full 40-character Git SHA" in result.stderr


def test_sourced_launcher_keeps_parent_alive():
    result = subprocess.run([BASH, "-c", 'source "$1"; code=$?; echo "parent_alive:$code"', "test", SCRIPT],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "parent_alive:2" in result.stdout
    assert "do not source" in result.stderr


def test_launcher_syntax():
    result = subprocess.run([BASH, "-n", SCRIPT], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
