# Next AutoDL action

Status: Phase 11 is complete. Across paired seeds 0/1/2, combined style reaches
`0.640048 +/- 0.010760` Cityscapes mIoU and photometric-only reaches
`0.636201 +/- 0.016427`. The paired advantage of combined style is small,
variable, and reverses at seed 1, so Phase 12 uses photometric-only as the
parsimonious control. Run only the three 500-iteration query-count smoke tests
below. Do not start a 40k query-count run yet.

## Reclaim the completed Phase-11 intermediates

Keep the final checkpoint and delete only the 79 intermediate checkpoints from
the completed photometric-only seed-2 run.

```bash
cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short

S1S2_RUN=/root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED2_40000_4662cba
S1S2_RESOLVED="$(realpath "$S1S2_RUN")"

case "$S1S2_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED2_40000_4662cba) ;;
  *) echo "unsafe path: $S1S2_RESOLVED"; exit 1 ;;
esac

test -f "$S1S2_RUN/summary.json"
test -f "$S1S2_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$S1S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$S1S2_RUN/checkpoints"
```

The Git SHA must be `4662cbadd4553728e521c351682da53545996a03` and status
must be empty. Only after every check above succeeds:

```bash
find "$S1S2_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$S1S2_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$S1S2_RUN/checkpoints"
df -h /root/autodl-tmp
```

## Checkout and preflight

Replace `<EXACT_SHA_FROM_HANDOFF>` with the exact SHA in the local handoff.

```bash
cd /root/autodl-tmp/CausalQ_DG
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD
git status --short

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
python tools/check_environment.py
python -m pytest
```

Git status must be empty and all tests must pass. Verify that Phase 12 changes
only query count and retains the selected photometric-only control:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_query_count, resolve_seed, style_view_weights

config = load_config(Path("configs/query_count/gta_dinov3l.yaml"))
assert config["experiment"]["phase"] == 12
assert resolve_seed(config, None) == 0
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
for query_count in (1, 2, 4):
    assert resolve_query_count(config, query_count) == query_count
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("PHASE12_SMOKE_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the three query-count smoke tests

Run the following block once. It stops on the first failed training process and
does not overwrite an existing run or log.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50
export SEED=0

test "$MAX_ITERATIONS" -eq 500
test "$VALIDATION_MAX_SAMPLES" -eq 50
test "$SEED" -eq 0

RUN_SHA="$(git rev-parse --short HEAD)"
set -o pipefail

for QUERY_COUNT in 1 2 4; do
  RUN_ID="Q${QUERY_COUNT}_QUERY_COUNT_SMOKE_500_${RUN_SHA}"
  RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
  LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

  test ! -e "$RUN_DIR"
  test ! -e "$LOG_FILE"

  QUERY_COUNT="$QUERY_COUNT" RUN_ID="$RUN_ID" \
    bash scripts/train_query_count.sh 2>&1 | tee "$LOG_FILE"
  TRAIN_EXIT=${PIPESTATUS[0]}
  echo "query_count=$QUERY_COUNT train_exit_code=$TRAIN_EXIT"
  if [ "$TRAIN_EXIT" -ne 0 ]; then
    exit "$TRAIN_EXIT"
  fi
  test -f "$RUN_DIR/summary.json"
done
```

## Extract smoke evidence

```bash
cd /root/autodl-tmp/CausalQ_DG
RUN_SHA="$(git rev-parse --short HEAD)"

python - "$RUN_SHA" <<'PY'
import json
import math
from pathlib import Path
import sys

run_sha = sys.argv[1]
output_root = Path("/root/autodl-tmp/outputs/CausalQ_DG")

for query_count in (1, 2, 4):
    run_id = f"Q{query_count}_QUERY_COUNT_SMOKE_500_{run_sha}"
    run_dir = output_root / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validations = summary["validation_results"]
    errors = [
        abs(row["loss"] - (row["loss_original"] + row["loss_photometric"]))
        for row in records
    ]
    checkpoints = list((run_dir / "checkpoints").glob("iter_*.pth"))

    assert metadata["phase"] == 12
    assert metadata["seed"] == 0
    assert metadata["max_iterations"] == 500
    assert metadata["query"]["queries_per_class"] == query_count
    assert metadata["style"]["views"] == ["original", "photometric"]
    assert "prediction_consistency" not in metadata
    assert "causal_query_effect" not in metadata
    assert "query_diversity" not in metadata
    assert summary["ok"] is True
    assert len(records) == 500
    assert [row["iteration"] for row in records] == list(range(1, 501))
    for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"):
        assert all(math.isfinite(row[key]) for row in records), key
    assert max(errors) < 1e-5
    assert len(validations) == 1
    assert validations[0]["sample_count"] == 50
    assert len(checkpoints) == 1
    assert (run_dir / "checkpoints" / "iter_000500.pth").is_file()

    print(f"===== R={query_count} metadata =====")
    print(metadata)
    print(f"===== R={query_count} summary =====")
    for key in (
        "ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
        "finite_losses", "last_gradient_norm", "final_alpha",
        "peak_allocated_gib", "peak_reserved_gib",
    ):
        print(f"{key}:", summary[key])
    print("record_count:", len(records))
    print("maximum_objective_reconstruction_error:", max(errors))
    print("final_validation:", validations[-1])
    print("checkpoint_count:", len(checkpoints))

print("PHASE12_ALL_SMOKES_OK")
PY

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

## Acceptance criteria

- The Phase-11 cleanup retains only `iter_040000.pth` and recovers roughly
  15 GiB.
- Exact handoff SHA, clean Git status, environment report, test pass, and the
  expected DINOv3 checksum are recorded.
- R=1, R=2, and R=4 each finish with exit code zero and `ok=true`.
- Each run records Phase 12, seed 0, 500 contiguous finite records, its intended
  query count, and only original/photometric style views.
- Each objective reconstructs as `loss_original + loss_photometric` within
  `1e-5`, has one 50-image validation, and produces only `iter_000500.pth`.
- No prediction consistency, CQE, or query-diversity metadata is present.

Return the cleanup output, preflight output, and complete evidence for all three
smokes. Stop after these smoke tests; do not launch any 40k query-count run.
