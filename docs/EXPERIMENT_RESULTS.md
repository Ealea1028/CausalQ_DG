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
Real extracted manifests then arrived locally. Individual hashes, checksums,
report contract, canonical membership and recomputed split all pass. The
exact manifests entered Git at `4314d61`; original ZIP bytes were not verified.
See `source_split_freeze_review.md` in the same experiment directory. No source pixels, target data, models,
weights or GPU were used by the filesystem inventory. It provides no evidence
for anchor/residual stability or segmentation performance.

The returned full `cb52071` source-data report now passes its pinned local
audit: source22469/holdout2497 complete scans, zero invalid/unreadable/all-ignore
or incompatible geometry,62 allowed scale-equivalent differences,2497 holdout
and32 source-photo preprocessing checks. Saved report and review are in the
same evidence directory. No mIoU or effect metric was measured.

The initial `b16236c` smoke snapshot reports20 finite updates, nonzero branch
gradients, unchanged backbone, exact model/optimizer and zero-error prediction
roundtrip. Peak reserved is1.181641GiB. Source/target mIoU and effect metrics
were not measured; this is not a formal M0 result. Original local attachment
paths became unavailable after initial read, so the complete immutable trace
and checkpoint still require the new AutoDL raw-byte audit; do not overstate
the initial snapshot as that audit. See `gpu_smoke_return_review.md`.

Next is read-only full2497 GTA5 holdout inference from the SAME checkpoint,
preceded by that complete audit. Zero new optimizer updates and no target
access. Full-source diagnostics and independent metric checks remain pending.
