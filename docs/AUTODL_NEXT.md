# AutoDL next step: audit matched Phase 16 query-residual CQE smoke

## Current action — only this section is active

The paired smoke at `3644785a22a4c9391bb9076b4cdad606c63be615` completed with
`ok: true` and exit code 0. Both arms ran 20 optimizer updates over the same 80
GTA5 examples; saved input fingerprints match. Losses and gradients are finite,
the expected residual parameters received gradients, both branches changed
from identical initialization, the frozen base stayed unchanged, and both
checkpoint roundtrips passed. Peak reserved memory was 2.430 GiB. There is no
Cityscapes result; differing training losses are not an accuracy comparison.
This smoke does not authorize long training.

Next independently audit the saved report, stderr, traces and branch
checkpoints. This is read-only and does not load datasets, weights or CUDA.
Preserve failed attempts. In a fresh AutoDL terminal run:

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
git rev-parse HEAD
git status --short
bash scripts/audit_rein_query_residual_cqe_phase16.sh
```

Return the full audit JSON, artifact hashes, exit code, Git status and disk
output. Acceptance requires `query_residual_cqe_saved_audit_ok: true`; stop
after the audit. No longer run is authorized by this engineering smoke alone.

## Previous producer instructions — completed; do not rerun

The first attempt at this gate stopped during preflight with
`Compact checkpoint coverage mismatch` before any training updates. Preserve
its report and log under the `90f2495` suffix; do not rerun or overwrite them.
The cause was that REIN's trainability hook had not been activated before
checking compact-checkpoint key coverage. The corrected producer calls
`model.train(True)` and validates the exact REIN adapter/decode-head trainable
set before restoring the accepted baseline. The retry uses a new commit suffix.

That retry reached the first source microbatch but stopped while computing
CQE: the pinned PyTorch 2.0 runtime rejects tuple dimensions in `Tensor.any`.
No optimizer updates occurred. Preserve its report/log and partial run under
the `ba61199` suffix. The CQE present-class mask now uses sequential dimension
reductions compatible with the pinned runtime; unit coverage includes multiple
samples and ignored pixels. Retry only at the next new commit suffix.

The no-CQE smoke at `14d3e5a` passed its producer and saved-evidence audits.
The next gate compares fresh branches with and without the existing CQE loss.
Each arm uses the same 80 GTA5 samples, identical initialization and the same
geometry-aligned original/photometric inputs; each runs 20 optimizer updates
with accumulation four. Both use mean two-view segmentation loss. Only the
candidate adds normalized CQE at weight 1.0. The frozen REIN model stays fixed.
This smoke performs no Cityscapes evaluation and gives no accuracy result.

The script creates a new run directory, two trace files, two branch-only
checkpoints, a report and stderr log. It checks the disk/input hashes, captures
a fingerprint for every paired example, and writes final hashes to the
terminal. It needs the isolated CUDA 11.8 environment. Preserve existing runs
and stop if any expected output path already exists. Use a fresh terminal:

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
RUN="$BASE/REIN_QUERY_RESIDUAL_CQE_PAIRED_SMOKE_20UPDATES_${EXPECTED_SHA:0:7}"
REPORT="$BASE/analysis/rein_phase16_query_residual_cqe_${EXPECTED_SHA:0:7}_v1.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_${EXPECTED_SHA:0:7}_v1.stderr.log"
test -f /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth || exit 1
test -f "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" || exit 1
test -f "$BASE/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json" || exit 1
test -f "$BASE/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth" || exit 1
test ! -e "$RUN" || { echo "existing_run=$RUN"; exit 1; }
test ! -e "$REPORT" || { echo "existing_report=$REPORT"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

bash scripts/check_rein_query_residual_cqe_phase16.sh
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
)
```

Replace `EXPECTED_SHA` with the full commit SHA supplied in the response. Return
the complete report, per-arm update/loss summaries, `matched_inputs_identical`,
both checkpoint hashes, report/log hashes, exit code, Git SHA/status and disk
output. Acceptance requires exit 0, `ok: true`, 80 matched fingerprints,
20 distinct source indices and 20 updates per arm, finite values, matching
initial branch fingerprints, objective reconstruction error at most `1e-7`,
all required branch gradients, unchanged frozen base parameters and successful
checkpoint roundtrips. Stop after returning this evidence; a short smoke does
not authorize a 40k run.

## Completed branch smoke — do not repeat

Producer `14d3e5a` ran at seed `20260930` with accumulation 4, 80 distinct GTA5
microbatches and 20 optimizer updates. It used the frozen accepted REIN source
baseline and no CQE. Peak reserved GPU memory was 2.055 GiB. Its saved-evidence
audit passed at `161f077`; its audit JSON SHA256 is
`517ea9e74a1d2ad999182e957b1d20c8588d8d08fec1b70ac4eb14043fd1011b`.
It has no target mIoU.

## Completed paired CQE smoke — saved audit pending

Producer `3644785a22a4c9391bb9076b4cdad606c63be615` completed both arms with
seed `20260931`. Report/stderr hashes:
`c6a7ee96f1e2c4e2defd29e2dc07e8d439c48fde74a4f1e3f7776a4b428f5ddd` /
`15f7e575bf971c025c9cbe64b18e1051f9346458015edb8583f9965940ea591e`.
No-CQE/CQE trace hashes:
`1eb60f2459ea9a5a077303ea6cd8c0f7a66ad1caf3d013c3daecf74ed624e620` /
`f5544d5fa82e81959b9806dbc315921ecfbef4296e48f8c62aa0c8800d488bef`.
Branch checkpoint hashes:
`632929ccce21ee68e6a1b804de900ab30b4019f8391fda7710fd1b81153f78f1` /
`10410867c71843057798a45fb6a2ab86fe57c4e1b0e83491dea7836b79ff69a0`.
No target evaluation or accuracy claim is present.

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
