# AutoDL next step: Phase 16 frozen-REIN query-residual smoke

## Current action — only this section is active

The formal source-only REIN seed-0 baseline is accepted by the read-only audit
at `1da98b5`: 40,000 optimizer updates, 160,000 microbatches and fixed-final
Cityscapes mIoU `0.6560509975`. Its final checkpoint SHA256 is
`84231e98dda68ac4b1fd4f59cc97881887a614de01b01e2697323c1eac80daf2`.

The next gate adds exactly one mechanism: an explicit class-specific query
residual over the frozen baseline's 19 semantic logits. The upstream 100
Mask2Former queries are not relabelled. Run only 80 GTA5 microbatches / 20
branch optimizer updates with accumulation 4. CQE remains disabled; no target
images or labels are used; no accuracy claim or formal branch training is
authorized. The gate must show finite branch optimization, gradient flow after
the zero-initialized scale opens, exact single-class intervention isolation,
unchanged REIN base weights, and a branch-only checkpoint roundtrip.

The gate needs little new disk space and does not copy the 283 MB baseline
checkpoint. Preserve every existing run and report. Do not clean up, resume,
start a long run, or enable CQE. Run on the exact handoff commit only:

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
test -f "$BASE/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json" || exit 1
test -f "$BASE/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth" || exit 1
test -f /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth || exit 1
test -d /root/autodl-tmp/external/rein-phase16-dc063429 || exit 1

set -o pipefail
bash scripts/check_rein_query_residual_phase16.sh
GATE_EXIT=$?
echo "outer_query_residual_gate_exit_code=$GATE_EXIT"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
exit "$GATE_EXIT"
)
```

Return the complete JSON, both exit-code lines, report/log/checkpoint hashes,
Git SHA/status and disk output. Acceptance requires `ok: true`, 80 records,
20 updates, finite loss, nonzero gradients for `alpha`, `query_bank`, and
`pixel_projection.weight`, `base_parameters_unchanged: true`, and both class
effect errors within the recorded thresholds. Stop after returning this
evidence. A smoke mIoU is intentionally absent.

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
