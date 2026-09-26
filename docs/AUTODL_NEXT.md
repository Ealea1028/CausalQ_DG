# AutoDL next step: GTA5 → Cityscapes Query Effect qualitative audit

At the user's request, BDD100K, Mapillary, ACDC and other target-dataset
verification is **deferred**, not completed. This handoff is limited to the
existing GTA5-trained A0 baseline and selected Static R=2 seed-0 model on
Cityscapes **val**. It implements §41 of `CausalQ_DG_Project_Plan.md` as a
read-only diagnostic. Do not retrain, tune using Cityscapes, or claim that
qualitative similarity establishes a real-world causal effect.

The script chooses four distinct examples by ground-truth class presence only
(road, car, person, vegetation), never by model success. It produces original
image, GT, independent A0 prediction, photometric view, signed Query effect on
both views (one shared color scale per sample), and Static R=2 prediction.

## 1. Sync the exact Git source and check inputs

Use the exact implementation SHA supplied with this handoff; do not edit
source on AutoDL. Replace `EXPECTED_SHA` below with that full SHA.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"

A0_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A0_DINOV3L_BASE_SEED0_40000_REPAIRED_7eee12b/checkpoints/iter_040000.pth
STATIC_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e/checkpoints/iter_040000.pth
WEIGHTS=/root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors

test -f "$A0_CKPT"
test -f "$STATIC_CKPT"
test -f "$WEIGHTS"
sha256sum "$WEIGHTS" "$A0_CKPT" "$STATIC_CKPT"
df -h /root/autodl-tmp
```

If any `test` fails, stop and return its output. Do not substitute an arbitrary
checkpoint. At least 2 GiB available disk is recommended for this diagnostic.

## 2. Run the read-only GPU diagnostic

```bash
VIS_DIR=/root/autodl-tmp/outputs/CausalQ_DG/analysis/query_effect_static_r2_seed0_v1
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/query_effect_static_r2_seed0_v1.log

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
test ! -e "$VIS_DIR"
test ! -e "$LOG"

set -o pipefail
python tools/visualize_query_effect.py \
  --static-config configs/query_interaction/gta_dinov3l_static.yaml \
  --baseline-config configs/baseline/gta_dinov3l.yaml \
  --static-checkpoint "$STATIC_CKPT" \
  --baseline-checkpoint "$A0_CKPT" \
  --output-dir "$VIS_DIR" \
  --seed 20260926 \
  --min-class-pixels 1000 \
  2>&1 | tee "$LOG"
VIS_EXIT=${PIPESTATUS[0]}
echo "visualization_exit_code=$VIS_EXIT"
```

If `VIS_EXIT` is not zero, stop. Retain the log and any partial output; do not
rerun into the same directory or overwrite files. Return the traceback.

## 3. Verify and send back the evidence

```bash
python - "$VIS_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
assert report["ok"] is True
assert report["scope"] == "GTA5_to_Cityscapes_val"
assert [item["class"] for item in report["samples"]] == [
    "road", "car", "person", "vegetation"
]
assert len({item["sample_id"] for item in report["samples"]}) == 4
assert all(Path(item["figure"]).is_file() for item in report["samples"])
print("visualization_audit_ok=true")
print("evaluation_git_sha:", report["evaluation_git_sha"])
for item in report["samples"]:
    print(item["class"], item["sample_id"], item["gt_class_pixels"], item["figure"])
PY

git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Send back the command output, `report.json`, and four PNG panels (or a contact
sheet). Visual review must explicitly note whether style changed visibly,
whether effect localization persisted, and whether prediction improved or
degraded. The figures cannot replace quantitative Cityscapes mIoU or prove
generalization to any deferred target dataset.
