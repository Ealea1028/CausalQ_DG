# AutoDL next step: convert the audited DINOv2-L checkpoint for REIN

Phase 15's three-seed ViT-B/ViT-L comparison is complete. The seed-2 paired
report has `ok: true`, 500 matched Cityscapes validation images, 6,005
present-class effect maps per backbone, and report SHA256
`2761a9264593aba81712f9a5ad0ef2006c0a1b533e99a0987cd57654f3ae357b`.
Across seeds 0/1/2, ViT-L has higher final mIoU and lower normalized
query-effect style variance in every pair. See the versioned three-seed
report in `experiments/BACKBONE_SCALING_STATIC_R2_3SEED/report.md`.

The original plan's §43 DINOv2/REIN transfer check is the current single
phase. The official DINOv2-L file, staged on AutoDL with SHA256
`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`,
passed the pinned CPU tensor audit at Git SHA
`26ea45f41a7b09e163ba87d2351a5c5d0108a2ff`. The audit confirms the
unconverted 14×14/37×37 layout and 24-block non-register architecture. Disk
space was 17 GiB. The next operation creates a separate REIN-compatible
16×16/32×32 tensor layout. This remains a CPU data-conversion step, not a
REIN installation, model load, GPU smoke, or CQE transfer claim.

## Current AutoDL action

Use the full Git SHA from the handoff. The original staged file and its audit
report stay untouched. Conversion refuses existing destination, temporary
file, or report paths. It verifies the source SHA before CPU-only
`weights_only=True` loading, then writes a new checkpoint and records its SHA.
Do not run this if available disk is under 5 GiB. The existing `causalq-dg`
environment is used only for tensor conversion, not REIN package installation.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
STAGE=/root/autodl-tmp/pretrained/.dinov2_vitl14_pretrain_phase16.pth.part
EXPECTED_WEIGHTS_SHA=d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
OUTPUT=/root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/dinov2_vitl14_rein_patch16_512_conversion.json

if git fetch origin main \
  && git checkout --detach "$EXPECTED_SHA" \
  && [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ] \
  && [ -z "$(git status --porcelain)" ] \
  && [ -f "$STAGE" ] \
  && [ "$(sha256sum "$STAGE" | cut -d' ' -f1)" = "$EXPECTED_WEIGHTS_SHA" ] \
  && [ ! -e "$OUTPUT" ] \
  && [ ! -e "$OUTPUT.part" ] \
  && [ ! -e "$REPORT" ]; then
  mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
  echo '===== source, weight, and capacity ====='
  git rev-parse HEAD
  stat -c '%s %n' "$STAGE"
  sha256sum "$STAGE"
  df -h /root/autodl-tmp
  AVAILABLE_KIB=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  test "$AVAILABLE_KIB" -ge 5242880 || { echo 'less_than_5GiB_free'; exit 1; }
  set -o pipefail
  python -m tools.convert_dinov2_for_rein \
    --weights "$STAGE" \
    --expected-sha256 "$EXPECTED_WEIGHTS_SHA" \
    --output "$OUTPUT" | tee "$REPORT"
  CONVERT_EXIT=${PIPESTATUS[0]}
  echo "convert_exit_code=$CONVERT_EXIT"
  if [ "$CONVERT_EXIT" -eq 0 ]; then
    stat -c '%s %n' "$OUTPUT"
    sha256sum "$OUTPUT" "$REPORT"
  fi
  git status --short
  df -h /root/autodl-tmp
else
  echo 'preflight_failed; conversion_not_started'
  git rev-parse HEAD
  git status --short
  ls -lh "$STAGE" 2>/dev/null || true
fi
```

Return the complete JSON report, `convert_exit_code`, the output and report
SHA256 values, Git SHA/status, and disk space. If conversion fails, retain
the original, any `.part` output, and the error report for diagnosis; do not
rerun over them. A passing conversion establishes tensor layout only. The
next step must still audit the converted checkpoint and specify an isolated
REIN integration and class-specific CQE intervention before GPU training.

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
