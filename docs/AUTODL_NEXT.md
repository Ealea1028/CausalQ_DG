# Next AutoDL action

Status: DINOv3 Phase 4 is accepted. Phase 5 is paused because its first batches exposed an invalid GTA5 raw-label conversion. Phase 3 GTA5 train-ID masks must be rebuilt and fully revalidated before baseline training resumes.

## Root cause

Official GTA5 PNG labels are palette-mode (`P`) indexed masks. The previous converter called `convert("L")`, which converted the palette colors to grayscale luminance instead of preserving class indices. Sample `22706` consequently retained only 1,142 valid pixels out of 2,002,044, and 20 of 200 sampled derived masks were entirely ignore-index 255. The raw labels remain untouched and can be used to rebuild the derived masks.

## Repair behavior

The corrected reader accepts indexed/integer mask modes and obtains their stored indices directly. Converted files are written through a temporary PNG and atomically replaced. The validator now reports and rejects all-ignore labels as well as valid-pixel coverage. Keep the broken derived directory as a temporary backup until the rebuilt dataset passes full validation.

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
python tools/check_environment.py
```

Confirm the corrected reader on sample `22706` before rebuilding anything:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

from causalq.datasets.cityscapes import label_ids_to_train_ids, read_index_mask

paths = sorted(Path("/root/autodl-tmp/datasets/gta5/labels").rglob("22706.png"))
if len(paths) != 1:
    raise RuntimeError(f"Expected one raw label, got {paths}")

with Image.open(paths[0]) as image:
    print("raw_mode:", image.mode)
    raw = read_index_mask(image)

converted = label_ids_to_train_ids(raw)
print("raw_unique_ids:", np.unique(raw).tolist())
print("converted_unique_ids:", np.unique(converted).tolist())
print("converted_valid_pixels:", int(np.count_nonzero(converted != 255)))
print("converted_valid_fraction:", float(np.mean(converted != 255)))
PY
```

The raw mode should be `P`; the raw indices should resemble Cityscapes label IDs rather than palette luminance values, and the converted mask should contain substantially more than 1,142 valid pixels. Stop if this preflight is not satisfied.

Verify the exact directories and available space before moving anything:

```bash
realpath /root/autodl-tmp/datasets/gta5/labels
realpath /root/autodl-tmp/datasets/gta5/labels_trainIds
du -sh /root/autodl-tmp/datasets/gta5/labels_trainIds
df -h /root/autodl-tmp

test ! -e /root/autodl-tmp/datasets/gta5/labels_trainIds_palette_luminance_bug
echo "backup_target_available=$?"
```

The two `realpath` results must remain inside `/root/autodl-tmp/datasets/gta5`. The backup check must print zero. Then preserve the incorrect derived masks and create a fresh destination:

```bash
mv -- \
  /root/autodl-tmp/datasets/gta5/labels_trainIds \
  /root/autodl-tmp/datasets/gta5/labels_trainIds_palette_luminance_bug

mkdir -p /root/autodl-tmp/datasets/gta5/labels_trainIds
```

Rebuild all 24,966 derived labels and perform a full scan:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/data_check
set -o pipefail

python tools/check_datasets.py \
  --datasets gta5 \
  --convert-gta5 \
  --label-scan-limit 0 \
  --samples 10 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_palette_fix \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_palette_fix.json

echo "gta5_rebuild_exit_code=${PIPESTATUS[0]}"
```

If conversion is interrupted, rerun the same command. Completed files will be skipped and each new file is atomically installed.

Inspect the corrected sample and final state:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("/root/autodl-tmp/datasets/gta5/labels_trainIds")
paths = sorted(root.rglob("22706.png"))
if len(paths) != 1:
    raise RuntimeError(f"Expected one rebuilt label, got {paths}")

with Image.open(paths[0]) as image:
    label = np.asarray(image)
    mode = image.mode

unique, counts = np.unique(label, return_counts=True)
print("path:", paths[0])
print("mode:", mode)
print("unique_ids:", unique.tolist())
print("value_counts:", {int(v): int(c) for v, c in zip(unique, counts)})
print("valid_pixels:", int(np.count_nonzero(label != 255)))
print("valid_fraction:", float(np.mean(label != 255)))
PY

find /root/autodl-tmp/datasets/gta5/labels_trainIds \
  -type f -name '*.png' | wc -l

git status --short
```

## Acceptance criteria

- `gta5_rebuild_exit_code=0` and top-level report has `"ok": true`.
- Conversion reports 24,966 source files and a total of 24,966 converted plus safely skipped existing outputs.
- `paired_count=24966`, with zero missing images or labels.
- Full scan covers all 24,966 derived labels.
- `unique_label_ids` is exactly `0..18` plus `255`.
- `invalid_train_ids`, geometry mismatches, unreadable files, and all-ignore labels are all zero.
- `valid_fraction_min` is greater than zero and mean coverage is plausible for dense semantic masks.
- Sample `22706` has substantially more than the incorrect 1,142 valid pixels.
- The corrected directory contains 24,966 PNG files.
- The old derived directory remains at `labels_trainIds_palette_luminance_bug`; do not delete it yet.
- `git status --short` is empty.

Return the complete JSON report, rebuild exit code, sample `22706` statistics, corrected file count, disk usage, and final Git status. Stop after data repair; Phase 5 training resumes only after this evidence is reviewed locally.
