#!/usr/bin/env bash
# A function/subshell keeps a caller that sources this file alive on failure.
causalq_hpi_main() (
  set -euo pipefail
  if [[ $# -ne 2 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'Usage: bash scripts/run_hpi_dinov2_20261031.sh FULL_SHA prepare|train-baselines|performance-check|status|package' >&2
    return 2
  fi
  case "$2" in prepare|train-baselines|performance-check|status|package) ;; *) echo 'This action is not released' >&2; return 2 ;; esac
  local code_dir
  code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$code_dir"
  : "${PY:?Set PY to the independent hpi environment python}"
  : "${CAUSALQ_OUTPUT_ROOT:?Set the study output root}"
  export PYTHONPATH="$code_dir"
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  "$PY" -m tools.run_hpi_dinov2_20261031 "$1" "$2"
)
causalq_hpi_main "$@"
