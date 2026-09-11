# Next AutoDL action

Status: Phase 2 accepted; waiting for Phase 3 dataset validation.

## Phase 2 conclusion

The reported RTX 4090D environment is usable:

- Python 3.12.3
- PyTorch 2.7.0 with CUDA 12.6 runtime
- CUDA available and tensor probe passed
- RTX 4090D detected with 23.52 GiB VRAM
- All pinned core packages imported successfully

The earlier `ok: false` was caused only by unset storage-path variables. `scripts/activate_autodl.sh` now activates the virtual environment and exports those paths together. CUDA 12.6 is accepted; changing the image is unnecessary.

## Goal

Verify the real GTA5 and Cityscapes layouts, convert GTA5 labels when needed, and produce ten visual checks per ready dataset. Do not download DINOv3 weights or start training in this phase.

## Required data layout

See `docs/DATASETS.md`. At minimum, Phase 3 needs:

```text
/root/autodl-tmp/datasets/gta5/images
/root/autodl-tmp/datasets/gta5/labels
/root/autodl-tmp/datasets/cityscapes/leftImg8bit
/root/autodl-tmp/datasets/cityscapes/gtFine
```

Obtain Cityscapes through its official distribution channel and upload/extract it into the paths above. GTA5 can be downloaded directly on AutoDL using the commands in `docs/DATASETS.md`; do not put archives or extracted datasets inside the Git repository.

## Commands

Update the repository first:

```bash
cd /root/autodl-tmp/CausalQ_DG
git status --short
git pull --ff-only origin main
git rev-parse HEAD
bash scripts/setup_autodl.sh
source scripts/activate_autodl.sh
python tools/check_environment.py
```

Inspect the available directory layout:

```bash
find /root/autodl-tmp/datasets -maxdepth 4 -type d | sort
du -sh /root/autodl-tmp/datasets/gta5 /root/autodl-tmp/datasets/cityscapes
```

Run the first dataset check:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/data_check

python tools/check_datasets.py \
  --datasets gta5 cityscapes \
  --samples 10 \
  --label-scan-limit 200 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/before_conversion.json
```

If the GTA5 section reports `"needs_conversion": true`, run the non-destructive conversion:

```bash
python tools/check_datasets.py \
  --datasets gta5 \
  --convert-gta5 \
  --samples 10 \
  --label-scan-limit 200 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/gta5_after_conversion.json
```

If the Cityscapes section has zero `*_gtFine_labelTrainIds.png` files but does have `*_gtFine_labelIds.png` files (the normal official-archive layout), run:

```bash
python tools/check_datasets.py \
  --datasets cityscapes \
  --convert-cityscapes \
  --samples 10 \
  --label-scan-limit 200 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/cityscapes_after_conversion.json
```

This creates train-ID masks beside the raw masks and never overwrites `*_gtFine_labelIds.png`.

Then run the combined check once more:

```bash
python tools/check_datasets.py \
  --datasets gta5 cityscapes \
  --samples 10 \
  --label-scan-limit 500 \
  --output-dir /root/autodl-tmp/outputs/CausalQ_DG/data_check \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/data_check/final_report.json

git status --short
```

## Acceptance criteria

- GTA5 and Cityscapes each report `ok: true`.
- No images or labels are missing their pair.
- No unreadable files or size mismatches are reported.
- Converted/official masks contain only train IDs `0..18` and ignore index `255`.
- Ten image-plus-colored-GT previews are generated for each dataset.
- `git status --short` remains empty.

Return the directory listing, final JSON report, any conversion report, and two or three representative preview images. Stop after this check; DINOv3 backbone work begins only after the data report is reviewed locally.
