# Strict-DG research results

Phase18 status (2026-10-17): Centered Query CPU implementation/tests are
prepared; no M1 GPU or formal result exists. The next bounded engineering
gate does not measure mIoU or anchor-effect metrics. Phase17 source pilot is
accepted with complete pinned report/manifest; older incomplete-return notes
below are historical. No target/effect evidence is newly claimed.

Latest acceptance (2026-10-17): complete pilot report/checksum bytes are now
recovered and pinned. Full contract/worker/checksum/matrix audits pass; source
runner engineering is accepted at 482ef26. Source mIoU 35.7242% remains a
500-step diagnostic, not formal M0. Earlier missing-byte descriptions below
are historical. See Phase17 pilot500_complete_review.md. No further source
pilot repeat; formal training and final target execution remain blocked.

Pilot500 partial return (2026-10-17): 482ef26 reports exit 0, source mIoU
0.35724184600067665 (independent local recomputation 0.3572418460006766),
2,497 holdout IDs, 1,276,971,688 valid pixels, peak reserved 1.330078125 GiB.
Visible worker/metric/state-isolation checks pass. Full root report/checksum
bytes are missing from the truncated paste; complete acceptance is pending
read-only return. Do not rerun the pilot. This is not formal M0, a target
metric or anchor/effect evidence. See pilot500_partial_review.md.

Latest engineering return (2026-10-17): corrected 5dc6caa continuation is
accepted. Exact 8 versus 4+4 traces/model/optimizer/sampler/RNG, identical
final checkpoint hashes, three independent workers, CUDA CE reference loss
error 0 and gradient error 6.98e-10, peak reserved 0.951172 GiB. The report is
byte-pinned; raw checkpoint/trace bytes remain remote and were checksum-checked
there. This is reproducibility engineering evidence, not a target/effect or
formal M0 result. Previous 08c91d1 NLL failure remains recorded. Next is a
fixed 500-update source runner integration pilot (250 + 250, full source
holdout, fixed-final checkpoint), not formal training.

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

The source checkpoint continuation gate is now accepted as above. The
next source runner integration pilot remains pending AutoDL. No pilot source
mIoU, target metric or new mechanism conclusion exists yet.
