from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').exists() else shutil.which('bash')
pytestmark = pytest.mark.skipif(BASH is None, reason='Bash unavailable')
SCRIPTS = [(ROOT / 'scripts' / name).as_posix() for name in
           ('analyze_query_components_phase20.sh', 'analyze_query_components_full_phase20.sh')]


@pytest.mark.parametrize('source', [False, True])
@pytest.mark.parametrize('script', SCRIPTS)
def test_launch_failure_retains_parent_shell(source, script):
    command = 'source "$1"; code=$?; echo "parent_alive:$code"' if source else '"$2" "$1" invalid; code=$?; echo "parent_alive:$code"'
    result = subprocess.run([BASH, '-c', command, 'test', script, BASH], capture_output=True, text=True)
    assert result.returncode == 0 and 'parent_alive:2' in result.stdout


@pytest.mark.parametrize('script', SCRIPTS)
def test_launcher_syntax(script):
    result = subprocess.run([BASH, '-n', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
