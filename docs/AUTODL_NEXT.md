# AutoDL next step: audit Phase 16 query-residual smoke evidence

## Current action — only this section is active

The 80-microbatch / 20-update no-CQE producer smoke at
`14d3e5aa90571362eabb368c6216bb230a4421fa` returned `ok: true` and both
exit codes zero. The frozen source-only REIN seed-0 reference remains fixed at
500-image Cityscapes mIoU `0.6560509975`. The producer reported finite branch
optimization, nonzero scale/query/projection gradients, unchanged base
parameters, exact outside-class isolation, and a 17,887-byte branch checkpoint.
These are engineering observations, not target accuracy or CQE evidence.

Next, audit only the *saved* report, trace, stderr and checkpoint. This command
does not load datasets, model weights or CUDA. It creates one new JSON audit
artifact, retains all existing files and requires the clean exact Git commit
provided with this handoff. Do not rerun the smoke, start formal training,
enable CQE, or delete anything. Use a fresh terminal and run:

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
test -z "$(git status --porcelain)" || { echo dirty_worktree; exit 1; }
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1

BASE=/root/autodl-tmp/outputs/CausalQ_DG
RUN="$BASE/REIN_QUERY_RESIDUAL_NO_CQE_SMOKE_20UPDATES_14d3e5a"
REPORT="$BASE/analysis/rein_phase16_query_residual_14d3e5a_v1.json"
STDERR_LOG="$BASE/analysis/rein_phase16_query_residual_14d3e5a_v1.stderr.log"
AUDIT="$BASE/analysis/rein_phase16_query_residual_14d3e5a_saved_audit.json"
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
test -f "$RUN/train.jsonl" || exit 1
test -f "$RUN/query_residual_20updates.pth" || exit 1
test -f "$REPORT" || exit 1
test -f "$STDERR_LOG" || exit 1
test ! -e "$AUDIT" || { echo "existing_audit=$AUDIT"; exit 1; }
test -x "$PY" || exit 1

set -o pipefail
"$PY" -m tools.audit_rein_query_residual \
  --report "$REPORT" \
  --stderr-log "$STDERR_LOG" \
  --run-dir "$RUN" | tee "$AUDIT"
AUDIT_EXIT=${PIPESTATUS[0]}
echo "saved_audit_exit_code=$AUDIT_EXIT"
sha256sum "$AUDIT" "$REPORT" "$STDERR_LOG" "$RUN/query_residual_20updates.pth"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
exit "$AUDIT_EXIT"
)
```

Replace `EXPECTED_SHA` with the full commit SHA supplied in the response. Return
the complete audit JSON, audit exit code, four hashes, Git SHA/status, and disk
output. Acceptance requires `query_residual_saved_audit_ok: true`; stop there.
If the audit fails, preserve the inputs and failed audit artifact and report the
error. The producer hashes expected by the versioned auditor are report
`5b566bd4d764109e71a26b4da7abd10f267ad627377de0340ce3c616e1266510`,
stderr `ebec21c1e0c8bd257fa9353337304a3c96e6d7361e7520ba6d43b3711bf040b2`,
and branch checkpoint
`fade8f3893ca1b8e2559ec154450ad3cb6ff9efee84b17354e6f70149c68d852`.

## Completed branch smoke — do not repeat

Producer `14d3e5a` ran at seed `20260930` with accumulation 4, 80 distinct GTA5
microbatches and 20 optimizer updates. It used the frozen accepted REIN source
baseline and no CQE. Peak reserved GPU memory was 2.055 GiB. Its saved-evidence
audit is pending, and it has no target mIoU.

## Completed source-only audit — do not repeat

The audit JSON reports `source40k_saved_audit_ok: true`; producer
`d6fc52c5af9dcf0d6f57218b8b448c7df7a74ea4`; final mIoU
`0.6560509975136082`; report/stderr hashes
`e1e95369809434603017a9c0539207c8f8db662dad9f0c8c669f8b6980099187` /
`82fe80372ac64042458cc5e5f1f2aceb5a228d8bf52ae799fa843eb5b99e3330`;
final/previous checkpoint hashes
`84231e98dda68ac4b1fd4f59cc97881887a614de01b01e2697323c1eac80daf2` /
`06fe23fde947bf61687f9fde803dbc0ba7ffc8bfebc1c15f480e7841178d7003`.
The audit artifact SHA256 is
`b1a8e15345ca72dac5e9f36550dc5dadb5e4b07423e8bb56d8500fb5fde719d1`.
