#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${CAUSALQ_ENV_ROOT:-/root/autodl-tmp/envs/causalq-dg}"

export CAUSALQ_DATA_ROOT="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
export CAUSALQ_PRETRAINED_ROOT="${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
export CAUSALQ_OUTPUT_ROOT="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"

mkdir -p "$(dirname "$ENV_ROOT")" "$CAUSALQ_DATA_ROOT" "$CAUSALQ_PRETRAINED_ROOT" "$CAUSALQ_OUTPUT_ROOT"

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version.split()[0]}")
print(f"Using base Python {sys.version.split()[0]}")
PY

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    python -m venv --system-site-packages "$ENV_ROOT"
fi

"$ENV_ROOT/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_ROOT/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$ENV_ROOT/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT"

cat <<EOF

CausalQ-DG environment is ready.
Activate it with:
  source "$PROJECT_ROOT/scripts/activate_autodl.sh"

Then run:
  cd "$PROJECT_ROOT"
  python tools/check_environment.py
EOF
