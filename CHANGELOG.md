# Changelog

## Unreleased

### Added

- Phase 1 project structure.
- Environment and data/pretrained manifests.
- CPU-only contract tests.
- AutoDL RTX 4090D setup and hand-off documentation.
- Phase 3 GTA5/Cityscapes discovery, pairing, label-ID, shape, and visualization checks.
- Non-destructive GTA5 raw-label conversion into `labels_trainIds`.
- AutoDL activation helper that exports all project storage paths.
- Phase 5 GTA5-to-Cityscapes source-only training and validation pipeline.
- Frozen DINOv3 multi-layer fusion decoder, segmentation cross-entropy, streaming mIoU, and compact head-only checkpoints.
- Geometry-only paired augmentation with scale-equivalent GTA5 label alignment.

### Changed

- Removed any dependency on historical DAFormer/MRM and QK Adapter projects.
- Environment checks now use documented AutoDL path defaults when variables have not been exported.
- GTA5 validation now unwraps one redundant archive directory such as `images/images` or `labels_trainIds/labels`.
- Editable-install `*.egg-info` metadata is ignored by Git.
- GTA5 validation now reports official resolution-only image/label differences as scale-equivalent warnings while still rejecting geometry mismatches.
- Pillow image creation no longer uses the deprecated explicit `mode` argument.
- Phase 3 hand-off now records the assumed GTA5 acceptance and isolates the remaining Cityscapes full-validation step.
- Phase 4 frozen DINOv3 ViT-B/L wrapper with dynamic prefix-token removal and dense patch maps.
- GPU backbone checker for feature shapes, finite values, parameter freezing, checkpoint hashes, and peak CUDA memory.
- Official Hugging Face DINOv3 model IDs and architecture metadata in the pretrained manifest.
- Phase 4 real-weight ViT-B/L checks passed on RTX 4090D with finite frozen features.
- CUDA memory-statistics calls now use integer device indices for PyTorch 2.7 compatibility.
- Invalid inherited `OMP_NUM_THREADS` values are normalized during AutoDL activation.
- ModelScope secondary-distribution provenance and verified checkpoint SHA-256 values are recorded.
- Phase 5 crops now retry toward valid semantic pixels and reject all-ignore batches explicitly instead of allowing a NaN mean cross-entropy.
- Training records now distinguish non-finite logits from invalid labels and include the valid-pixel count.
- GTA5 palette-mode labels now preserve their class indices instead of converting palette colors to grayscale luminance.
- Dataset validation now rejects all-ignore train-ID masks and reports valid-pixel coverage.
- Derived-label writes are atomic so an interrupted full conversion can be resumed safely.
