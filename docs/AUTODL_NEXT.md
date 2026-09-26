# AutoDL next step: acquire and inspect BDD100K semantic segmentation data

Phase 14 closes as a negative learned-null mechanism ablation. External-domain
evaluation now returns to the three-seed-supported Static R=2 model. Work on one
target at a time, beginning with BDD100K.

This step only acquires and inspects data. Do not train, evaluate, rename raw
files, convert labels, or delete archives yet. BDD100K redistribution is
license-controlled, so use the official BDD100K portal after accepting its
terms. Generic unauthenticated direct URLs cannot be committed to this project.

Required official archives:

- `bdd100k_images_10k.zip`
- `bdd100k_sem_seg_labels_trainval.zip`

## 1. Verify repository and available storage

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

git fetch origin main
git checkout --detach 9bcc3d23e273cfcdcb36df12e8d17da25e9c3e84

test "$(git rev-parse HEAD)" = \
  "9bcc3d23e273cfcdcb36df12e8d17da25e9c3e84"
test -z "$(git status --porcelain)"

mkdir -p /root/autodl-tmp/uploads/bdd100k
mkdir -p /root/autodl-tmp/datasets/bdd100k

df -h /root/autodl-tmp
du -sh /root/autodl-tmp/uploads/bdd100k 2>/dev/null || true
du -sh /root/autodl-tmp/datasets/bdd100k 2>/dev/null || true
```

At least 15 GiB should remain before downloading. If either destination already
contains files, stop and return their listing instead of overwriting them.

## 2. Download directly on AutoDL using official authorized URLs

Sign in to the official BDD100K download portal, accept the dataset terms, and
copy the download URL for each required archive. Paste each URL only into the
corresponding silent prompt below. The URL is held in a shell variable and is
not printed.

```bash
cd /root/autodl-tmp/uploads/bdd100k

test ! -e bdd100k_images_10k.zip
test ! -e bdd100k_sem_seg_labels_trainval.zip

read -r -s -p "Paste official 10K Images URL: " BDD_IMAGES_URL
echo
wget -c --tries=0 --timeout=60 --waitretry=10 \
  -O bdd100k_images_10k.zip \
  "$BDD_IMAGES_URL"
unset BDD_IMAGES_URL

read -r -s -p "Paste official Semantic Segmentation Labels URL: " BDD_LABELS_URL
echo
wget -c --tries=0 --timeout=60 --waitretry=10 \
  -O bdd100k_sem_seg_labels_trainval.zip \
  "$BDD_LABELS_URL"
unset BDD_LABELS_URL
```

If the official portal downloads files to the local computer instead, upload
the two unchanged archives to `/root/autodl-tmp/uploads/bdd100k/` using the
AutoDL file manager, preserving the exact filenames above, then continue.

## 3. Validate the archives before extraction

```bash
cd /root/autodl-tmp/uploads/bdd100k

ls -lh \
  bdd100k_images_10k.zip \
  bdd100k_sem_seg_labels_trainval.zip

file \
  bdd100k_images_10k.zip \
  bdd100k_sem_seg_labels_trainval.zip

unzip -t bdd100k_images_10k.zip \
  | tail -n 3

unzip -t bdd100k_sem_seg_labels_trainval.zip \
  | tail -n 3

echo "===== image archive preview ====="
unzip -l bdd100k_images_10k.zip \
  | sed -n '1,25p'

echo "===== label archive preview ====="
unzip -l bdd100k_sem_seg_labels_trainval.zip \
  | sed -n '1,35p'
```

Both `unzip -t` commands must end without an error. HTML/XML login pages saved
under a `.zip` name are invalid and must not be extracted.

## 4. Extract without overwriting or restructuring

```bash
test -z "$(find /root/autodl-tmp/datasets/bdd100k -mindepth 1 -print -quit)"

unzip -q -n \
  /root/autodl-tmp/uploads/bdd100k/bdd100k_images_10k.zip \
  -d /root/autodl-tmp/datasets/bdd100k

unzip -q -n \
  /root/autodl-tmp/uploads/bdd100k/bdd100k_sem_seg_labels_trainval.zip \
  -d /root/autodl-tmp/datasets/bdd100k
```

## 5. Inspect the real layout and counts

```bash
echo "===== directories ====="
find /root/autodl-tmp/datasets/bdd100k \
  -maxdepth 6 -type d \
  | sort

echo "===== representative files ====="
find /root/autodl-tmp/datasets/bdd100k \
  -type f \
  | sort \
  | sed -n '1,80p'

echo "===== extension counts ====="
find /root/autodl-tmp/datasets/bdd100k \
  -type f \
  | sed 's/.*\.//' \
  | tr '[:upper:]' '[:lower:]' \
  | sort \
  | uniq -c \
  | sort -nr

echo "===== split/path counts ====="
for token in train val test masks images sem_seg; do
  count="$(
    find /root/autodl-tmp/datasets/bdd100k \
      -type f -path "*${token}*" \
      | wc -l
  )"
  echo "$token $count"
done

echo "===== storage ====="
du -sh /root/autodl-tmp/uploads/bdd100k
du -sh /root/autodl-tmp/datasets/bdd100k
df -h /root/autodl-tmp
```

## 6. Decode a sample of candidate masks

This inspection is deliberately layout-agnostic. It scans PNG files whose path
contains both `sem_seg` and `mask`, then reports dimensions, image modes, and
pixel IDs without modifying any file.

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path
import random

import numpy as np
from PIL import Image

root = Path("/root/autodl-tmp/datasets/bdd100k")
candidates = sorted(
    path
    for path in root.rglob("*.png")
    if "sem_seg" in str(path).lower()
    and "mask" in str(path).lower()
)

print("candidate_mask_count:", len(candidates))
assert candidates, "No semantic-segmentation mask candidates found"

random.Random(20260926).shuffle(candidates)
sample = candidates[: min(100, len(candidates))]

modes = Counter()
sizes = Counter()
unique_ids = set()
unreadable = []

for path in sample:
    try:
        with Image.open(path) as image:
            modes[image.mode] += 1
            sizes[image.size] += 1
            array = np.asarray(image)
            if array.ndim == 2:
                unique_ids.update(int(value) for value in np.unique(array))
    except Exception as error:
        unreadable.append((str(path), repr(error)))

print("sampled_mask_count:", len(sample))
print("modes:", dict(modes))
print("sizes:", {str(key): value for key, value in sizes.items()})
print("sampled_2d_unique_ids:", sorted(unique_ids))
print("unreadable_count:", len(unreadable))
print("unreadable_examples:", unreadable[:10])
print("first_20_candidates:")
for path in sorted(candidates)[:20]:
    print(path)
PY
```

Return all output from sections 3, 5, and 6, then stop. Do not delete the ZIP
archives, convert masks, implement guessed directory rules, or start model
evaluation until the observed official layout and label encoding are reviewed.
