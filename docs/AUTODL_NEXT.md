# AutoDL next step: Phase 12 query-count mechanism analysis

Status: all four Phase 12 full runs are complete and stable. Final Cityscapes
mIoU for `R=1/2/3/4` is respectively
`0.603711/0.643725/0.638287/0.624083`, so R=2 is the current leader. Phase 12
is not closed until all four final checkpoints receive the same query
similarity, active-query, and cross-style effect-variance analysis.

Run only the analysis below. Do not start another training run.

## 1. Reclaim only R=4 intermediate-checkpoint space

Keep `iter_040000.pth` and delete only the 79 recorded intermediates from the
exact R=4 run directory.

```bash
R4_RUN=/root/autodl-tmp/outputs/CausalQ_DG/Q4_QUERY_COUNT_SEED0_40000_84268a1

case "$R4_RUN" in
  /root/autodl-tmp/outputs/CausalQ_DG/Q4_QUERY_COUNT_SEED0_40000_84268a1) ;;
  *) echo "refusing unexpected R4 path: $R4_RUN"; exit 1 ;;
esac

test -f "$R4_RUN/checkpoints/iter_040000.pth"

INTERMEDIATE_COUNT="$(
  find "$R4_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "intermediate_checkpoint_count=$INTERMEDIATE_COUNT"
test "$INTERMEDIATE_COUNT" -eq 79

find "$R4_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -delete

test "$(find "$R4_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
test -f "$R4_RUN/checkpoints/iter_040000.pth"
du -sh "$R4_RUN/checkpoints"
df -h /root/autodl-tmp
```

Expected: exactly one R=4 checkpoint remains and roughly 15 GiB is recovered.

## 2. Checkout the exact analysis commit

Replace no files on AutoDL. Fetch and checkout the exact Git commit below.

```bash
cd /root/autodl-tmp/CausalQ_DG

git status --short
test -z "$(git status --porcelain)"

git fetch origin
git checkout 07ef4ce71d04cac432259f10d05693cdcecfdca8
test "$(git rev-parse HEAD)" = "07ef4ce71d04cac432259f10d05693cdcecfdca8"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

python -m pytest
```

The full CPU suite must pass before GPU analysis.

## 3. Verify the four immutable final checkpoints

```bash
R1=/root/autodl-tmp/outputs/CausalQ_DG/Q1_QUERY_COUNT_SEED0_40000_9f47726/checkpoints/iter_040000.pth
R2=/root/autodl-tmp/outputs/CausalQ_DG/Q2_QUERY_COUNT_SEED0_40000_198898f/checkpoints/iter_040000.pth
R3=/root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED0_40000_d335694/checkpoints/iter_040000.pth
R4=/root/autodl-tmp/outputs/CausalQ_DG/Q4_QUERY_COUNT_SEED0_40000_84268a1/checkpoints/iter_040000.pth

test -f "$R1"
test -f "$R2"
test -f "$R3"
test -f "$R4"

sha256sum "$R1" "$R2" "$R3" "$R4"
df -h /root/autodl-tmp
```

## 4. Run the common 500-image diagnostic

This is evaluation only. It uses the original and photometric views, seed
`20260924`, and all 500 Cityscapes validation images for every model.

```bash
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/QUERY_COUNT_R1_R2_R3_R4_seed20260924.json
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/QUERY_COUNT_R1_R2_R3_R4_seed20260924.log

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
test ! -e "$REPORT"
test ! -e "$LOG"

set -o pipefail

python tools/analyze_query_count.py \
  --config configs/query_count/gta_dinov3l.yaml \
  --model R1 1 "$R1" 0.6037110117161197 \
  --model R2 2 "$R2" 0.6437254689928078 \
  --model R3 3 "$R3" 0.6382865707103561 \
  --model R4 4 "$R4" 0.6240832651012953 \
  --max-samples 500 \
  --seed 20260924 \
  --output "$REPORT" \
  2>&1 | tee "$LOG"

ANALYSIS_EXIT=${PIPESTATUS[0]}
echo "analysis_exit_code=$ANALYSIS_EXIT"
test "$ANALYSIS_EXIT" -eq 0
```

## 5. Validate and return the evidence

```bash
python - "$REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["ok"] is True
assert report["evaluation_git_sha"] == "07ef4ce71d04cac432259f10d05693cdcecfdca8"
assert report["dataset"] == "cityscapes_val"
assert report["seed"] == 20260924
assert report["views"] == ["original", "photometric"]
assert report["ranking_by_final_cityscapes_miou"] == ["R2", "R3", "R4", "R1"]

expected = {
    "R1": (1, "9f477261d4cf1bb7e9618b0d9d4fb2c1f6227d3c"),
    "R2": (2, "198898f580516e3cf508b588d299b0bd3a95c55f"),
    "R3": (3, "d335694f33b0266a338c91fe9bfe6885fb48467c"),
    "R4": (4, "84268a115400ef360270c2ed9d4a229041bfbbe5"),
}

for name, (query_count, training_sha) in expected.items():
    model = report["models"][name]
    assert model["queries_per_class"] == query_count
    assert model["training_git_sha"] == training_sha
    assert model["iteration"] == 40000
    assert model["sample_count"] == 500
    assert model["effect_class_map_count"] > 0
    assert math.isfinite(model["cross_style_effect_variance"])
    for view in ("original", "photometric"):
        similarity = model["contextual_query_similarity"][view]
        behavior = model["active_query_behavior"][view]
        assert behavior["class_map_count"] > 0
        for key, value in behavior.items():
            if key != "class_map_count":
                assert math.isfinite(value)
        if query_count == 1:
            assert similarity["pair_count"] == 0
            assert similarity["mean_cosine"] is None
        else:
            assert similarity["pair_count"] > 0
            assert math.isfinite(similarity["mean_cosine"])

    residual = model["residual_query_similarity"]
    if query_count == 1:
        assert residual["pair_count"] == 0
        assert residual["mean_cosine"] is None
    else:
        assert residual["pair_count"] > 0
        assert math.isfinite(residual["mean_cosine"])

print("query_count_analysis_ok=true")
print("ranking:", report["ranking_by_final_cityscapes_miou"])
for name in ("R1", "R2", "R3", "R4"):
    model = report["models"][name]
    print(name, {
        "mIoU": model["final_cityscapes_miou"],
        "residual_similarity": model["residual_query_similarity"],
        "contextual_similarity": model["contextual_query_similarity"],
        "active_query_behavior": model["active_query_behavior"],
        "effect_variance": model["cross_style_effect_variance"],
    })
PY

cat "$REPORT"

echo "===== final provenance ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the complete validation output and JSON report. Stop afterward: do not
delete final checkpoints and do not start another training run. The report is
needed locally to interpret the mechanism metrics and close Phase 12.
