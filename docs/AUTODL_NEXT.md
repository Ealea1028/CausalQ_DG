# Next AutoDL action

Status: the full GTA5 pixel-decoding scan covered all 24,966 pairs and isolated exactly one unreadable file: `/root/autodl-tmp/datasets/gta5/images/images/20217.png`. Restore only this image from the official part-9 image archive, then repeat the full scan. Do not restart the 40k run yet.

## Accepted scan evidence

- `paired_count=24966` and `scanned_label_count=24966` with `scan_scope=all`.
- `unreadable_count=1`; the only unreadable item is image `20217.png`, reported as `image file is truncated`.
- All labels remain readable with exactly train IDs `0..18` plus `255`.
- Zero missing pairs, invalid train IDs, all-ignore labels, or geometry mismatches.
- All 62 size mismatches are scale-equivalent and remain accepted.

The official Playing for Data release contains 24,966 frames split across ten archives. Image `20217.png` belongs to part 9. Use the official TU Darmstadt archive URL already documented in `docs/DATASETS.md`.

## Commands

Use the exact Git commit supplied in the hand-off:

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

Both Git status checks must be empty. Define and validate all repair paths:

```bash
ARCHIVE_DIR="/root/autodl-tmp/uploads/gta5_repair"
ARCHIVE="${ARCHIVE_DIR}/09_images.zip"
STAGE_DIR="/root/autodl-tmp/datasets/gta5/repair_stage_09"
CORRUPT="/root/autodl-tmp/datasets/gta5/images/images/20217.png"
QUARANTINE_DIR="/root/autodl-tmp/datasets/gta5/quarantine"
QUARANTINE="${QUARANTINE_DIR}/20217.truncated.png"

realpath "$CORRUPT"
test -f "$CORRUPT"
test ! -e "$QUARANTINE"
df -h /root/autodl-tmp
```

The resolved corrupt path must remain under `/root/autodl-tmp/datasets/gta5/images/images`. Download only official image part 9 and test the entire ZIP before extraction:

```bash
mkdir -p "$ARCHIVE_DIR"

wget -c --tries=0 --timeout=60 --waitretry=10 \
  -O "$ARCHIVE" \
  "https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip"

unzip -t "$ARCHIVE"
```

Stop if `unzip -t` does not end with `No errors detected`. Locate the exact archive entry and require exactly one match:

```bash
ENTRY_LIST="$(unzip -Z1 "$ARCHIVE" | grep -E '(^|/)20217\.png$')"
ENTRY_COUNT="$(printf '%s\n' "$ENTRY_LIST" | sed '/^$/d' | wc -l)"

echo "entry_count=${ENTRY_COUNT}"
printf '%s\n' "$ENTRY_LIST"
test "$ENTRY_COUNT" -eq 1

ENTRY="$ENTRY_LIST"
```

Extract only that image into an isolated staging directory and fully decode it:

```bash
test ! -e "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
unzip -q "$ARCHIVE" "$ENTRY" -d "$STAGE_DIR"

STAGED="${STAGE_DIR}/${ENTRY}"
test -f "$STAGED"

python - "$CORRUPT" "$STAGED" <<'PY'
import hashlib
from pathlib import Path
import sys
from PIL import Image

for value in sys.argv[1:]:
    path = Path(value)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            image.load()
            print(path, "OK", image.mode, image.size, path.stat().st_size, digest)
    except Exception as exc:
        print(path, "FAILED", type(exc).__name__, str(exc), path.stat().st_size, digest)
PY
```

The current file must report `FAILED` and the staged official image must report `OK`. Preserve the corrupt file, install the staged file through a temporary path, and validate it again:

```bash
mkdir -p "$QUARANTINE_DIR"
mv -- "$CORRUPT" "$QUARANTINE"

cp -- "$STAGED" "${CORRUPT}.replacement.tmp"

python - "${CORRUPT}.replacement.tmp" <<'PY'
from pathlib import Path
import sys
from PIL import Image

path = Path(sys.argv[1])
with Image.open(path) as image:
    image.load()
    print("replacement_ok:", path, image.mode, image.size, path.stat().st_size)
PY

mv -- "${CORRUPT}.replacement.tmp" "$CORRUPT"
sha256sum "$QUARANTINE" "$CORRUPT" "$STAGED"
```

Repeat the full validation. This time the checker must exit zero:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_after_20217_repair
set -o pipefail

python tools/check_datasets.py \
  --datasets gta5 \
  --label-scan-limit 0 \
  --samples 0 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_after_20217_repair \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_after_20217_repair.json

RECHECK_EXIT=${PIPESTATUS[0]}
echo "gta5_recheck_exit_code=${RECHECK_EXIT}"
```

Record final state without deleting the archive, staged image, quarantine copy, or failed run yet:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/"
    "data_check/gta5_after_20217_repair.json"
)
gta5 = json.loads(path.read_text(encoding="utf-8"))["datasets"]["gta5"]
for key in (
    "ok",
    "paired_count",
    "scanned_label_count",
    "scan_scope",
    "unreadable_count",
    "invalid_train_ids",
    "all_ignore_label_count",
    "geometry_mismatch_count",
    "scale_equivalent_mismatch_count",
):
    print(f"{key}: {gta5[key]}")
PY

git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- Official part-9 archive passes `unzip -t`.
- It contains exactly one entry ending in `/20217.png` or equal to `20217.png`.
- The old image fails full decoding; the staged and installed images pass it.
- Installed and staged SHA-256 values match; the quarantined file has a different hash.
- Full scan covers 24,966 pairs and exits zero with top-level and GTA5 `ok=true`.
- `unreadable_count=0`, with no missing pairs, invalid IDs, all-ignore labels, or geometry mismatches.
- The 62 scale-equivalent mismatches remain warnings only.
- Final Git SHA matches the hand-off and `git status --short` is empty.

Return the ZIP integrity result, archive entry name, before/after decode output and hashes, complete recheck JSON, recheck exit code, final Git SHA/status, and disk usage. Stop after repair verification. A new 40k run ID will be issued only after this evidence is accepted.
