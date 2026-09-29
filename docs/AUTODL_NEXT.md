# AutoDL next step: isolate and probe the pinned REIN runtime

Phase 15's three-seed ViT-B/ViT-L comparison is complete. The seed-2 paired
report has `ok: true`, 500 matched Cityscapes validation images, 6,005
present-class effect maps per backbone, and report SHA256
`2761a9264593aba81712f9a5ad0ef2006c0a1b533e99a0987cd57654f3ae357b`.
Across seeds 0/1/2, ViT-L has higher final mIoU and lower normalized
query-effect style variance in every pair. See the versioned three-seed
report in `experiments/BACKBONE_SCALING_STATIC_R2_3SEED/report.md`.

The original plan's §43 DINOv2/REIN transfer check is the current single
phase. The converted DINOv2-L checkpoint passed an independent on-disk audit
at Git SHA `8ee5abf9d927743a5a88fa8b4691fd0cd4bb2301`: both pinned hashes
matched, and all 343 tensor keys, shapes, dtypes, and values matched a fresh
conversion. The audit report SHA256 is
`6cdbdf1b503f7f6c7eb58287088d99de1edde8b378e7304add140e5418a4c120`.
There were 476 GiB available system RAM and 16 GiB free data volume.

## Current AutoDL action: rerun the complete REIN runtime gate

Both CUDA diagnostics passed at `1badc850f35817352e943139baf94b12eee965f4`:
the main PyTorch 2.7/cu126 and isolated PyTorch 2.0.1/cu118 environments
each saw one RTX 4090 D (capability 8.9, driver 595.71.05) and completed
the scalar compute probe. Main/rein log hashes respectively:
`cbf3646e2b74c4205b30b9c57e7d58cb33e1d4ca888a42d7bf1784bbb878cfaf`
and `4f7736a3ac085b86ccadffcf10b47d87b079072ccb57622a4fffe3792c831a39`.
This establishes current basic GPU access, not the cause of the earlier
failure or full REIN compatibility. Preserve `NVIDIA_VISIBLE_DEVICES=void`:
the actual probes succeeded without overriding it. Both runs emitted an
invalid `OMP_NUM_THREADS` warning; set a valid value for the next process.
There are still 11 GiB free; no installation or download is needed.

Run only the existing runtime gate in the installed isolated prefix. It
checks exact package/source pins, CUDA NMS, REIN model registration and
configuration parsing. It does not build a model, load weights or datasets,
or execute deformable-attention forward. New output paths preserve all
previous reports. No training is authorized until this result is reviewed.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
REIN_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_runtime_gpu_recheck_v1
REPORT="$BASE.json"
ERROR_LOG="$BASE.stderr.log"

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] \
    || [ ! -x "$REIN_PY" ] \
    || [ -e "$REPORT" ] || [ -e "$ERROR_LOG" ]; then
    echo 'preflight_failed: wrong source, missing interpreter or existing outputs'
  else
    OMP_NUM_THREADS=1 "$REIN_PY" -m tools.check_rein_runtime \
      --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
      --expected-rein-sha dc063429c4dadc0da9c6252b3db22fc55a9882ab \
      2> "$ERROR_LOG" | tee "$REPORT"
    RUNTIME_EXIT=${PIPESTATUS[0]}
    echo "runtime_exit_code=$RUNTIME_EXIT"
    cat "$ERROR_LOG"
    sha256sum "$REPORT" "$ERROR_LOG"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the JSON, runtime exit code, stderr, hashes and source/disk evidence.
Success requires `ok: true`, CUDA NMS kept indices `[0]`, and both REIN
registrations true. If it fails, preserve the installed environment and
outputs and return the actual error; do not rerun bootstrap or edit upstream
source. Stop at this remote verification boundary.

## Completed paired CUDA diagnostic handoff (do not repeat)

At `035a3faaa90365286db06f0f728a023fcb1e47a3`, recovery installed all
seven pinned packages and `pip check` passed. The REIN source SHA matched,
but the runtime gate stopped at GPU availability, before MMCV CUDA NMS or
REIN imports. CUDA runtime was already the expected 11.8. The failed report
SHA256 is `a24a1fefce3d2a0a233010e2af567c10094c55d75e1845f0ccc86aa29e4ff44b`;
pip-freeze SHA256 is
`dd00674c5a9b5361730fdcaeaedf25e883befa0cc9d5f1e56df69f7f26d3fbfd`.
The data volume has 11 GiB free. Do not reinstall or create another environment.

The next action is a diagnostic, not a model smoke. It runs the same source
with the main and isolated interpreters, reports selected visibility/library
variables, NVIDIA device nodes and `nvidia-smi`, forces CUDA initialization
to retain its real exception, and attempts a one-scalar GPU computation.
It does not load weights/data or override visibility/library paths. An
installed CUDA wheel alone does not establish GPU access.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ] \
    && [ -z "$(git status --porcelain)" ]; then
    for NAME in main rein; do
      if [ "$NAME" = main ]; then
        DIAG_PY=/root/autodl-tmp/envs/causalq-dg/bin/python
      else
        DIAG_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
      fi
      DIAG_LOG="/root/autodl-tmp/outputs/CausalQ_DG/analysis/cuda_diagnostic_${NAME}_$(git rev-parse --short HEAD).log"
      if [ ! -x "$DIAG_PY" ] || [ -e "$DIAG_LOG" ]; then
        echo "diagnostic_preflight_failed=$NAME"
      else
        "$DIAG_PY" -m tools.diagnose_cuda 2>&1 | tee "$DIAG_LOG"
        DIAG_EXIT=${PIPESTATUS[0]}
        echo "diagnostic_${NAME}_exit_code=$DIAG_EXIT"
        sha256sum "$DIAG_LOG"
      fi
    done
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return both complete logs and exit codes. If `nvidia-smi` and both probes
fail, check the AutoDL console's GPU allocation/boot mode and contact AutoDL
support with the diagnostic; this is not evidence that package replacement
is needed. If the main probe passes but REIN fails, retain the exact init
error and library paths for environment-specific diagnosis. Do not unset
GPU allocation variables, remove drivers, disable the gate, or start training.
If both pass, the isolated REIN runtime gate still must be rerun into a fresh
report before model construction. Stop for remote diagnosis now.

## Previous Conda recovery handoff (installation completed; do not repeat)

At `893c9de308a868e2b9cbe6f9cab054c5916090b9`, the upstream REIN
checkout succeeded, but Conda metadata retrieval from `repo.anaconda.com`
failed with HTTP 000. No runtime gate or model loading was reached. Preserve
the old bootstrap log, upstream checkout, and any partial environment.

The recovery mode verifies and reuses the clean pinned REIN checkout. It
creates a **new** prefix `rein-phase16-py310-cu118-tuna-retry1`, refuses
existing retry outputs, and uses `--override-channels` with only Tsinghua's
main channel for Python/pip creation. It does not change `.condarc`, disable
TLS, delete files, or install into `causalq-dg`. PyTorch and MMCV still use
their previously pinned official wheel sources. A mirror network failure
must stop this gate; it does not authorize a model run.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_bootstrap_tuna_retry1.log

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] || [ -e "$LOG" ]; then
    echo 'preflight_failed: wrong source or existing retry log'
  else
    bash scripts/bootstrap_rein_phase16.sh --recover-conda 2>&1 | tee "$LOG"
    BOOTSTRAP_EXIT=${PIPESTATUS[0]}
    echo "bootstrap_exit_code=$BOOTSTRAP_EXIT"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi

tail -n 50 "$LOG"
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_runtime_tuna_retry1.json
if [ -f "$REPORT" ]; then cat "$REPORT"; fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the exit code, runtime JSON if created, report/freeze hashes printed by
the script, and source/disk evidence. This block does not use `exit` in the
interactive shell. Do not rerun over the retry prefix after a failure;
preserve it and return the log. Stop here until this remote gate is accepted.

## Previous bootstrap handoff (failed; do not repeat)

Use the full Git SHA from the handoff. The versioned bootstrap script checks
both checkpoint hashes and at least 14 GiB free before creating anything.
It clones only the official REIN commit
`dc063429c4dadc0da9c6252b3db22fc55a9882ab` to a separate external
directory, creates a separate Python 3.10 environment, installs pinned
PyTorch/OpenMMLab dependencies without a pip cache, and probes CUDA NMS,
REIN imports, and configuration parsing. It neither edits the upstream
checkout nor loads a segmentor or trains. If any installation fails, preserve
the partial environment and log; do not install into `causalq-dg` or rerun
over the same paths. The official REIN repository is GPL-3.0 and is not
copied into this Git project.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_bootstrap.log

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] \
    || [ -e "$LOG" ]; then
    echo 'preflight_failed: wrong source or existing log'
  else
    mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
    bash scripts/bootstrap_rein_phase16.sh 2>&1 | tee "$LOG"
    BOOTSTRAP_EXIT=${PIPESTATUS[0]}
    echo "bootstrap_exit_code=$BOOTSTRAP_EXIT"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return `bootstrap_exit_code`, the runtime JSON (if created), report and
environment-freeze SHA256 values, Git SHA/status, and disk space. On failure,
also return the last 50 log lines and preserve all created paths. A passing
runtime gate does not prove the full REIN model loads or fits RTX 4090 D;
the next action is a separate model-build and weight-load smoke after the
GTA5/Cityscapes-only configuration is versioned.

## Completed handoff: DINOv3-B Static R=2 full seed-0 training (do not repeat)

The restored ViT-B Static R=2 500-step smoke on AutoDL commit
`915e872ca080eae74eef387e7990ad6e3cecbdc5` produced a complete
`summary.json` (`ok: true`, finite losses), 500 training records, one
50-image Cityscapes validation at iteration 500 (`mIoU: 0.337162`), and the
final smoke checkpoint. Peak reserved GPU memory was 1.512 GiB; 25 GiB of
disk remained free. That short-run mIoU is diagnostic only. The next action
first audits the saved trace for contiguity and objective reconstruction,
then starts a fresh 40,000-iteration seed-0 run on all 500 validation images.
Do not resume the smoke checkpoint, overwrite an existing run, or start seed
1/2 or another target dataset at this stage.

## Completed smoke and recovery records (do not repeat)

The ViT-B weights have been restored from the distribution recorded in
`pretrained_manifest.yaml`. Their SHA256 is again
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
On AutoDL commit `d7913cf276bab6c96cfea4846dc6948bd38fb529`, the
separate backbone check passed on RTX 4090 D with bfloat16,
768-dimensional patch features, four requested intermediate maps, and zero
NaN/Inf; 25 GiB remained free. The next action is one 500-iteration training
smoke, not a 40k experiment.

## Completed recovery record (do not repeat)

The scaling smoke did not reach training. AutoDL's read-only inventory found
that `/root/autodl-tmp/pretrained/dinov3_vitb16` is absent, only the ViT-L
safetensors remains on the data volume, no likely Hugging Face cache blob was
found, and the failed smoke produced no run directory. The source commit is
`d7913cf276bab6c96cfea4846dc6948bd38fb529` with a clean worktree; the
disk has 26 GiB free. Preserve both failed-attempt logs.

The checked-in `pretrained_manifest.yaml` records the previously used
ModelScope distribution at
`https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m`.
Only the exact ViT-B safetensors with SHA256
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`
is admissible. Download into a new staging directory, verify before moving to
the configured path, and stop on any mismatch. Do not substitute ViT-L, use a
different ViT-B conversion, disable the SHA check, or launch training yet.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

DOWNLOAD_ENV=/root/autodl-tmp/envs/modelscope-download-phase15
STAGE=/root/autodl-tmp/pretrained/.dinov3_vitb16_restore_phase15
DEST=/root/autodl-tmp/pretrained/dinov3_vitb16
EXPECTED_SHA=9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b

test ! -e "$DOWNLOAD_ENV" || { echo "existing_download_env=$DOWNLOAD_ENV"; exit 1; }
test ! -e "$STAGE" || { echo "existing_stage=$STAGE"; exit 1; }
test ! -e "$DEST" || { echo "existing_destination=$DEST"; exit 1; }

python -m venv "$DOWNLOAD_ENV" || exit 1
"$DOWNLOAD_ENV/bin/python" -m pip install modelscope-hub || exit 1
"$DOWNLOAD_ENV/bin/ms-hub" download \
  facebook/dinov3-vitb16-pretrain-lvd1689m \
  --include config.json model.safetensors \
  --local-dir "$STAGE" || exit 1

test -s "$STAGE/config.json" || exit 1
test -s "$STAGE/model.safetensors" || exit 1
sha256sum "$STAGE/model.safetensors"
ACTUAL_SHA="$(sha256sum "$STAGE/model.safetensors" | cut -d' ' -f1)"
test "$ACTUAL_SHA" = "$EXPECTED_SHA" || {
  echo "weight_sha_mismatch=$ACTUAL_SHA"; exit 1;
}

python - "$STAGE/config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert config["hidden_size"] == 768, config["hidden_size"]
assert config["patch_size"] == 16, config["patch_size"]
assert config["num_register_tokens"] == 4, config["num_register_tokens"]
print("vitb16_config_ok=true")
PY
test "$?" -eq 0 || exit 1

test ! -e "$DEST" || exit 1
mv -- "$STAGE" "$DEST" || exit 1
test "$(sha256sum "$DEST/model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_SHA" || exit 1

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/backbone_check
CHECK_LOG=/root/autodl-tmp/outputs/CausalQ_DG/backbone_check/vitb16_restored_phase15.json
test ! -e "$CHECK_LOG" || { echo "existing_check_log=$CHECK_LOG"; exit 1; }
set -o pipefail
python tools/check_backbone.py \
  --model dinov3_vitb16 \
  --weights "$DEST" \
  --image-size 512 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --intermediate-indices 3 6 9 12 \
  2>&1 | tee "$CHECK_LOG"
CHECK_EXIT=${PIPESTATUS[0]}
echo "backbone_check_exit_code=$CHECK_EXIT"
test "$CHECK_EXIT" -eq 0 || exit 1
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

The recovery and GPU check above passed. These commands are retained for
provenance only; their staging and download-environment directories now exist,
so do not rerun them. Preserve the two failed smoke logs. The commands below
are historical handoffs, not the current Phase 16 action.

## Completed smoke handoff after the weight was restored

Project-plan §41's GTA5 → Cityscapes qualitative figures are provenance-matched
and closed as a diagnostic, not a causal proof. Begin §42 scaling with **one**
isolated DINOv3-B/16, Static R=2, photometric-only seed-0 smoke. The existing
DINOv3-L/16 Static R=2 three-seed result remains the reference. Do not train
on Cityscapes, run BDD100K/Mapillary/ACDC, enable CQE or learned-null, or launch
the 40k B run before this 500-step smoke is accepted.

## 1. Sync exact source and verify inputs

Replace `EXPECTED_SHA` with the full commit SHA provided with this handoff.
Run on AutoDL; never edit source files there.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || exit 1

WEIGHTS=/root/autodl-tmp/pretrained/dinov3_vitb16/model.safetensors
test -f "$WEIGHTS" || exit 1
test -d /root/autodl-tmp/datasets/gta5 || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/gtFine/val || exit 1

sha256sum "$WEIGHTS"
test "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" = \
  '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b' || exit 1
df -h /root/autodl-tmp
```

If any command fails, stop and return the output. The weight hash above is
from the previously validated local DINOv3-B safetensors. Leave at least 2 GiB
available before this smoke; do not delete data or existing checkpoints to
make space without a separate reviewed cleanup plan.

## 2. Run only the 500-iteration smoke

```bash
RUN_ID="SCALE_VITB_STATIC_R2_SEED0_SMOKE_500_RESTORED_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"

echo "run_id=$RUN_ID"
test ! -e "$RUN_DIR" || { echo "existing_run=$RUN_DIR"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

set -o pipefail
python tools/train.py \
  --config configs/scaling/gta_dinov3b_static.yaml \
  --run-id "$RUN_ID" \
  --max-iterations 500 \
  --validation-max-samples 50 \
  --seed 0 \
  2>&1 | tee "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0 || exit 1
```

Do not overwrite or resume a failed smoke directory. Keep its logs and report
the traceback for diagnosis. Smoke mIoU on 50 images is not a scaling result.

## 3. Inspect the smoke evidence

```bash
test -f "$RUN_DIR/summary.json" || exit 1
test -f "$RUN_DIR/metadata.json" || exit 1
test -f "$RUN_DIR/checkpoints/iter_000500.pth" || exit 1
wc -l "$RUN_DIR/train.jsonl"

python - "$RUN_DIR" "$EXPECTED_SHA" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = sys.argv[2]
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
assert summary["ok"] is True and summary["finite_losses"] is True
assert metadata["git_sha"] == expected and metadata["phase"] == 15
assert metadata["backbone"] == "dinov3_vitb16"
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert len(records) == 500
assert [item["iteration"] for item in records] == list(range(1, 501))
assert len(summary["validation_results"]) == 1
assert summary["validation_results"][0]["sample_count"] == 50
print("scaling_smoke_ok=true")
print("metadata:", metadata)
print("summary_without_validation:", {k: v for k, v in summary.items() if k != "validation_results"})
print("validation:", summary["validation_results"][0])
PY

sha256sum "$RUN_DIR/checkpoints/iter_000500.pth"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return `train_exit_code`, the audit output, checkpoint hash, final Git SHA and
status, and any errors. Only after checking those results may the 40k B run be
considered. §42's question about effect variance requires a later matched
full-validation analysis; this smoke cannot answer it.

## Historical action: audit smoke, then run one 40k seed-0 experiment

Replace `EXPECTED_SHA` below with the exact commit from the new handoff. Run
from a clean AutoDL checkout. The local smoke audit must pass before the full
run begins. The full run starts from random head initialization and the
verified frozen ViT-B weights; it must not resume the smoke checkpoint.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || exit 1

SMOKE=/root/autodl-tmp/outputs/CausalQ_DG/SCALE_VITB_STATIC_R2_SEED0_SMOKE_500_RESTORED_915e872
python - "$SMOKE" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
assert metadata["git_sha"] == "915e872ca080eae74eef387e7990ad6e3cecbdc5"
assert metadata["phase"] == 15 and metadata["backbone"] == "dinov3_vitb16"
assert metadata["pretrained_checkpoint_sha256"] == "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
assert metadata["seed"] == 0 and metadata["max_iterations"] == 500
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert all(key not in metadata for key in ("prediction_consistency", "causal_query_effect", "query_diversity", "learned_null"))
assert summary["ok"] is True and summary["finite_losses"] is True
assert [record["iteration"] for record in records] == list(range(1, 501))
assert all(math.isfinite(record[key]) for record in records for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"))
error = max(abs(record["loss"] - record["loss_original"] - record["loss_photometric"]) for record in records)
assert error < 1e-5, error
assert len(summary["validation_results"]) == 1
assert summary["validation_results"][0]["sample_count"] == 50
assert summary["validation_results"][0]["iteration"] == 500
assert (run / "checkpoints" / "iter_000500.pth").is_file()
print("smoke_audit_ok=true")
print("maximum_objective_reconstruction_error:", error)
print("smoke_validation:", summary["validation_results"][0])
PY
test "$?" -eq 0 || exit 1

WEIGHTS=/root/autodl-tmp/pretrained/dinov3_vitb16/model.safetensors
test -f "$WEIGHTS" || exit 1
test "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" = \
  '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b' || exit 1
test -d /root/autodl-tmp/datasets/gta5 || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/gtFine/val || exit 1
df -h /root/autodl-tmp

RUN_ID="SCALE_VITB_STATIC_R2_SEED0_40000_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"
echo "run_id=$RUN_ID"
test ! -e "$RUN_DIR" || { echo "existing_run=$RUN_DIR"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

set -o pipefail
python tools/train.py \
  --config configs/scaling/gta_dinov3b_static.yaml \
  --run-id "$RUN_ID" \
  --max-iterations 40000 \
  --validation-max-samples 500 \
  --seed 0 \
  2>&1 | tee "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0 || exit 1
```

Do not delete intermediate checkpoints before auditing the run. After the
command returns, report `train_exit_code`, the following evidence, and any
traceback. A final mIoU alone is not sufficient to accept a full run.

```bash
RUN_ID="SCALE_VITB_STATIC_R2_SEED0_40000_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"

python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
validations = summary["validation_results"]
checkpoints = sorted((run / "checkpoints").glob("iter_*.pth"))
assert summary["ok"] is True and summary["finite_losses"] is True
assert metadata["phase"] == 15 and metadata["max_iterations"] == 40000
assert metadata["backbone"] == "dinov3_vitb16" and metadata["seed"] == 0
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert [record["iteration"] for record in records] == list(range(1, 40001))
assert all(math.isfinite(record[key]) for record in records for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"))
error = max(abs(record["loss"] - record["loss_original"] - record["loss_photometric"]) for record in records)
assert error < 1e-5, error
assert len(validations) == 80
assert all(item["sample_count"] == 500 for item in validations)
assert validations[-1]["iteration"] == 40000
assert len(checkpoints) == 80 and checkpoints[-1].name == "iter_040000.pth"
print("full_run_audit_ok=true")
print("experiment_id:", metadata["experiment_id"])
print("training_git_sha:", metadata["git_sha"])
print("pretrained_checkpoint_sha256:", metadata["pretrained_checkpoint_sha256"])
print("maximum_objective_reconstruction_error:", error)
print("peak_reserved_gib:", summary["peak_reserved_gib"])
print("final_alpha:", summary["final_alpha"])
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda item: item["miou"]))
print("checkpoint_count:", len(checkpoints))
PY

sha256sum "$RUN_DIR/checkpoints/iter_040000.pth"
tail -n 20 "$LOG"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

After the 40k evidence is accepted, compare ViT-B and the fixed ViT-L Static
R=2 seed-0 result on the same 500 Cityscapes validation images. A later paired
style-effect analysis is also required before any scale-dependent variance
claim. Do not start these analyses or additional seeds as part of this handoff.
