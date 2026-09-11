#!/usr/bin/env bash

# Source this file; do not execute it as a child process.
CAUSALQ_ENV_ROOT="${CAUSALQ_ENV_ROOT:-/root/autodl-tmp/envs/causalq-dg}"

if [[ ! -f "$CAUSALQ_ENV_ROOT/bin/activate" ]]; then
  echo "CausalQ-DG environment not found: $CAUSALQ_ENV_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$CAUSALQ_ENV_ROOT/bin/activate"
export CAUSALQ_DATA_ROOT="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
export CAUSALQ_PRETRAINED_ROOT="${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
export CAUSALQ_OUTPUT_ROOT="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"

echo "Activated causalq-dg"
echo "  data:       $CAUSALQ_DATA_ROOT"
echo "  pretrained: $CAUSALQ_PRETRAINED_ROOT"
echo "  outputs:    $CAUSALQ_OUTPUT_ROOT"

