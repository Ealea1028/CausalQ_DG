# Next AutoDL action

Status: the Phase 5 500-iteration smoke run remains accepted, but the first full 40k attempt stopped when Pillow fully decoded a truncated GTA5 PNG. Pause training and perform a full pixel-decoding scan of all GTA5 image/label pairs before deciding how to repair the remote data.

## Diagnosis

The previous dataset validator read each image's dimensions but did not force Pillow to decode its pixel stream. A PNG can therefore have a valid header and file count while ending prematurely in its compressed image data. The updated validator calls `load()` on every image and records the exact failing path and whether it is an image or label. The dataset loader now also includes the sample ID and path in decode errors.

Do not enable Pillow's `LOAD_TRUNCATED_IMAGES`: it would silently train on incomplete pixels and would make the experiment irreproducible. Do not restart the 40k run until every pair passes full decoding.

## Commands

Run the exact Git commit supplied in the hand-off:

```bash
cd /root/autodl-tmp/CausalQ_DG
git status --short
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD
git status --short

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
```

Both Git status checks must be empty. Record where the failed full run stopped without modifying it:

```bash
FAILED_RUN_ID="A0_DINOV3L_BASE_SEED0_40000_c085de9"
FAILED_RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${FAILED_RUN_ID}"

if test -f "$FAILED_RUN_DIR/train.jsonl"; then
  echo "failed_run_records=$(wc -l < "$FAILED_RUN_DIR/train.jsonl")"
  tail -n 5 "$FAILED_RUN_DIR/train.jsonl"
fi

find "$FAILED_RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' 2>/dev/null | sort -k2 | tail -n 5
```

Run a full, zero-sample visualization scan. A nonzero scan exit code is expected while corrupt files exist, so capture it without terminating the shell:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_full_decode
set -o pipefail

python tools/check_datasets.py \
  --datasets gta5 \
  --label-scan-limit 0 \
  --samples 0 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_full_decode \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_full_decode.json

SCAN_EXIT=${PIPESTATUS[0]}
echo "gta5_full_decode_exit_code=${SCAN_EXIT}"
```

Extract a compact list of every reported unreadable path:

```bash
python - <<'PY'
import json
from pathlib import Path

report_path = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_full_decode.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))
gta5 = report["datasets"]["gta5"]
print("ok:", gta5["ok"])
print("scan_scope:", gta5["scan_scope"])
print("scanned_label_count:", gta5["scanned_label_count"])
print("unreadable_count:", gta5["unreadable_count"])
for item in gta5["unreadable"]:
    print(item)
PY
```

The JSON keeps the first 20 unreadable examples. If `unreadable_count` exceeds 20, do not attempt repair yet; return the report so the checker can be extended to publish a complete machine-readable path manifest without truncation.

Finally record provenance and space:

```bash
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Expected outcome

- The scan covers all 24,966 paired GTA5 samples.
- Each corrupt file appears with `kind: image` or `kind: label` and its exact path.
- No source files are changed by the scan.
- The failed run directory is retained for evidence but is never reused.

Return the failed-run record count and tail, full JSON report, scan exit code, compact unreadable list, exact Git SHA/status, and disk usage. Stop after the scan. The next hand-off will target only the affected source archive segment or files, then repeat the full decoder scan before creating a new 40k run ID.
