# Strict-DG research results

Latest engineering return (2026-10-17): 08c91d1 continuous worker exited 1 at
native CUDA NLL forward under strict determinism, before any optimizer.step().
No source/target/effect metric or exact continuation acceptance from this run.
The equivalent deterministic CE retry is pending AutoDL; prior evaluation
acceptance remains valid. See Phase 17 `resume_failure_08c91d1_review.md`.

As of 2026-10-17, there are no formal M0/M1/P1/M2/M3/M4 or target/effect
results. The accepted 20-update source-only engineering checkpoint has full
holdout mIoU 0.12971660656076558; this is a diagnostic, not formal M0 accuracy.
Historical experiment IDs/results are unchanged.

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

The b16236c smoke's saved report/full trace and checkpoint audit passed on
AutoDL at c7822fd. Full holdout report bytes now match the supplied SHA256 and
are pinned locally. Source coverage is all 2,497 IDs, valid pixels 1,276,971,688;
independent confusion/IoU audit passes. Source mIoU 0.12971660656076558,
image-mean CE 1.3436256861, pixel-weighted CE 1.3548140258. All eight CE replays
have zero error, model/RNG/original files unchanged; peak reserved 1.054688 GiB.
No new optimizer updates, checkpoint selection, target or effect metrics.
Twelve zero-IoU classes after 20 updates do not warrant a mechanism change.
The per-image trace remains remote; successful remote checksum output, exact
report and review are in `experiments/STRICT_DG_PHASE17/`.

Next is a separate checkpoint continuation gate: continuous 8 versus 4+4
updates across independent processes, 16 total source optimizer calls, exact
state/trace comparison. CPU synthetic acceptance is not GPU evidence; actual
AutoDL result remains pending. Formal training stays blocked.
