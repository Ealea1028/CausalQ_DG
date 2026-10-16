# Strict-DG research results

As of 2026-10-16, there are no M0/M1/P1/M2/M3/M4 training, source mIoU,
target mIoU, effect variance/magnitude, localization or responsibility results.
Historical results remain in their existing experiment IDs and are not reused
as new strict-DG results.

## Phase 17 / Priority 0

Operator-reported source filesystem gate at
`f41dbeea92d67bffc571e2c3ef103de7b3693e00`: exit 0, 24,966 complete
image/train-ID-label pairs; zero missing/duplicate IDs; all logical filenames
flat/numeric; zero metadata candidates. See
`../experiments/STRICT_DG_PHASE17/source_inventory_review.md` for provenance,
exact hashes and limitations. Raw JSON is remote; next exporter verifies it.

Engineering decision: use the plan's deterministic configurable 10% filename
fallback, with no claim of sequence/scene independence. Expected 22,469 train
and 2,497 holdout IDs. The operator's `57a7244` export reports these exact
counts with zero overlap and valid checksums; candidate hashes are recorded in
`../experiments/STRICT_DG_PHASE17/source_split_export_review.md`.
Real manifest/ZIP bytes, independent local verification and Git freeze remain
pending transfer. No export rerun is needed. No source pixels, target data, models,
weights or GPU were used by the filesystem inventory. It provides no evidence
for anchor/residual stability or segmentation performance.
