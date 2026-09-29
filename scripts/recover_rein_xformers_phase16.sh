#!/usr/bin/env bash
# Install only the missing dependency; never edit external REIN source.
set -euo pipefail
cd /root/autodl-tmp/CausalQ_DG
REIN_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
REIN_ROOT=/root/autodl-tmp/external/rein-phase16-dc063429
REIN_SHA=dc063429c4dadc0da9c6252b3db22fc55a9882ab
BASE=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_xformers_recovery_v1
CONSTRAINTS=configs/runtime/rein_phase16_constraints.txt
export OMP_NUM_THREADS=1
test -z "$(git status --porcelain)"
test -x "$REIN_PY"
test "$(git -C "$REIN_ROOT" rev-parse HEAD)" = "$REIN_SHA"
test -z "$(git -C "$REIN_ROOT" status --porcelain)"
for SUFFIX in before.txt after.txt json stderr.log; do
  test ! -e "$BASE.$SUFFIX"
done
test "$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')" -ge 1048576
"$REIN_PY" -c 'from importlib.metadata import version; from tools.check_rein_runtime import EXPECTED_VERSIONS; names={"mmseg":"mmsegmentation"}; actual={k:version(names.get(k,k)) for k in EXPECTED_VERSIONS if k!="xformers"}; assert all(actual[k]==v for k,v in EXPECTED_VERSIONS.items() if k!="xformers"),actual; print(actual)'
"$REIN_PY" -m pip freeze > "$BASE.before.txt"
# Resolve the small Python helper dependencies under the core constraints.
"$REIN_PY" -m pip install --no-cache-dir --only-binary=:all: \
  --index-url https://pypi.org/simple -c "$CONSTRAINTS" pyre-extensions==0.0.29
# Hash-verified wheel, no dependency resolution or source compilation.
"$REIN_PY" -m pip install --no-cache-dir --no-deps --require-hashes \
  -r configs/runtime/rein_phase16_xformers.txt
"$REIN_PY" -m pip check
"$REIN_PY" -m pip freeze > "$BASE.after.txt"
"$REIN_PY" -m tools.check_rein_runtime \
  --rein-root "$REIN_ROOT" --expected-rein-sha "$REIN_SHA" \
  2> "$BASE.stderr.log" | tee "$BASE.json"
cat "$BASE.stderr.log"
sha256sum "$BASE.before.txt" "$BASE.after.txt" "$BASE.json" "$BASE.stderr.log"
df -h /root/autodl-tmp
