# AutoDL next step: matched ViT-B versus ViT-L style-effect analysis

The Phase-15 ViT-B Static R=2 seed-0 40k run passed the user's full audit:
40,000 contiguous finite records, 80 validations on all 500 Cityscapes-val
images, 80 checkpoints, and zero objective-reconstruction error. Its final
mIoU is `0.563382495383709`; the fixed ViT-L Static R=2 seed-0 reference is
`0.6473964462159979` (ViT-B minus ViT-L: `-8.4014` percentage points).
The ViT-B final checkpoint SHA256 is
`48d2b50d6f4ff18bcc91238074f290408ed8c63674220640c4a8cbb93a00030f`.
These figures establish a segmentation gap only. Before making the §42
scale-dependent query-effect claim, run one matched two-backbone analysis on
the same 500 validation images with deterministic original/photometric views.
Do not start another seed, target dataset, or training run at this handoff.

## Current AutoDL action

Run the commands below from a fresh shell. Replace `EXPECTED_SHA` with the
exact full Git SHA in the accompanying Codex handoff; never edit repository
source on AutoDL. A failed check means stop and report its output.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || exit 1

VITB_RUN=/root/autodl-tmp/outputs/CausalQ_DG/SCALE_VITB_STATIC_R2_SEED0_40000_10993b6
VITL_RUN=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e
VITB_CKPT="$VITB_RUN/checkpoints/iter_040000.pth"
VITL_CKPT="$VITL_RUN/checkpoints/iter_040000.pth"
test -f "$VITB_RUN/summary.json" || exit 1
test -f "$VITL_RUN/summary.json" || exit 1
test -f "$VITB_CKPT" || exit 1
test -f "$VITL_CKPT" || exit 1
test "$(sha256sum "$VITB_CKPT" | cut -d' ' -f1)" = \
  '48d2b50d6f4ff18bcc91238074f290408ed8c63674220640c4a8cbb93a00030f' || exit 1
test "$(sha256sum "$VITL_CKPT" | cut -d' ' -f1)" = \
  'a0f9f8be54bd2a1027d183f52e08937db925ccd0ca4ab19815b42635ea06d407' || exit 1
test -f /root/autodl-tmp/pretrained/dinov3_vitb16/model.safetensors || exit 1
test -f /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors || exit 1
df -h /root/autodl-tmp

REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/backbone_scaling_static_r2_seed0_style_seed20260927.json
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/backbone_scaling_static_r2_seed0_style_seed20260927.log
test ! -e "$REPORT" || { echo "existing_report=$REPORT"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
set -o pipefail
python tools/compare_backbone_scaling.py \
  --vitb-checkpoint "$VITB_CKPT" \
  --vitb-summary "$VITB_RUN/summary.json" \
  --vitl-checkpoint "$VITL_CKPT" \
  --vitl-summary "$VITL_RUN/summary.json" \
  --max-samples 500 \
  --seed 20260927 \
  --output "$REPORT" \
  2>&1 | tee "$LOG"
EVAL_EXIT=${PIPESTATUS[0]}
echo "eval_exit_code=$EVAL_EXIT"
test "$EVAL_EXIT" -eq 0 || exit 1

python - "$REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["ok"] is True and report["sample_count"] == 500
assert report["views"] == ["original", "photometric"]
assert report["models"]["vitb16"]["sample_count"] == 500
assert report["models"]["vitl16"]["sample_count"] == 500
assert report["models"]["vitb16"]["effect_class_map_count"] == \
       report["models"]["vitl16"]["effect_class_map_count"]
assert math.isclose(report["models"]["vitb16"]["final_cityscapes_miou"],
                    0.563382495383709, abs_tol=1e-9)
assert math.isclose(report["models"]["vitl16"]["final_cityscapes_miou"],
                    0.6473964462159979, abs_tol=1e-9)
for model in report["models"].values():
    assert math.isfinite(model["cross_style_effect_variance"])
print("matched_scaling_analysis_ok=true")
print("vitb_effect_variance:", report["models"]["vitb16"]["cross_style_effect_variance"])
print("vitl_effect_variance:", report["models"]["vitl16"]["cross_style_effect_variance"])
print("vitb_to_vitl_variance_ratio:", report["vitb_to_vitl_effect_variance_ratio"])
print("vitl_has_lower_effect_variance:", report["vitl_has_lower_effect_variance"])
PY
test "$?" -eq 0 || exit 1
sha256sum "$REPORT" "$VITB_CKPT" "$VITL_CKPT"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return `eval_exit_code`, the audit printout, report SHA256, Git status and disk
output. Keep the report and log; do not delete checkpoints. The result is a
single-seed diagnostic, not a multi-seed scaling law or proof of causality.

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
are the current handoff.

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

## Current action: audit smoke, then run one 40k seed-0 experiment

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
