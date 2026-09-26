# AutoDL next step: restore and verify DINOv3-B weights

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

Return the download result, both SHA256 checks, backbone-check JSON and exit
code, Git SHA/status, and disk status. If installation, download, checksum, or
GPU validation fails, stop with its output and leave the stage and logs for
diagnosis. The following smoke commands are a later handoff, **not** the
current action.

## Subsequent action after the weight is restored and verified

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
RUN_ID="SCALE_VITB_STATIC_R2_SEED0_SMOKE_500_$(git rev-parse --short HEAD)"
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
