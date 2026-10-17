from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').exists() else shutil.which('bash')
pytestmark = pytest.mark.skipif(BASH is None, reason='Bash unavailable')
SCRIPT = (ROOT / 'scripts/check_centered_source_resume_phase21.sh').as_posix()


@pytest.mark.parametrize('source', [False, True])
def test_failure_does_not_exit_parent_shell(source):
    command = 'source "$1"; code=$?; echo "parent_alive:$code"' if source else '"$2" "$1" invalid; code=$?; echo "parent_alive:$code"'
    result = subprocess.run([BASH, '-c', command, 'test', SCRIPT, BASH], capture_output=True, text=True)
    assert result.returncode == 0 and 'parent_alive:2' in result.stdout


def test_launcher_syntax():
    result = subprocess.run([BASH, '-n', SCRIPT], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
