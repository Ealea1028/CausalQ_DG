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

### Changed

- Removed any dependency on historical DAFormer/MRM and QK Adapter projects.
- Environment checks now use documented AutoDL path defaults when variables have not been exported.
- GTA5 validation now unwraps one redundant archive directory such as `images/images` or `labels_trainIds/labels`.
- Editable-install `*.egg-info` metadata is ignored by Git.
- GTA5 validation now reports official resolution-only image/label differences as scale-equivalent warnings while still rejecting geometry mismatches.
- Pillow image creation no longer uses the deprecated explicit `mode` argument.
