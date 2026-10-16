# Strict-DG protocol — Phase 17 / Priority 0

Specification: `CausalQ_DG_1016.txt` (2026-10-16). This phase repairs the
development protocol before changing Query representation. Phase 16 stays closed.

## Current gate: saved-artifact audit and full source-only holdout evaluation

The initial `b16236c` report/trace snapshot indicates successful20-update
optimization/freeze/serialization. See Phase17 `gpu_smoke_return_review.md` for
the reported observations and explicit limits on local file availability.
Next AutoDL operation first pins the ORIGINAL report/trace hashes and validates
all20 source IDs, finite objectives/gradients, exact40k LR sequence, alpha-zero
gradient behavior, eight source references and report summaries. It verifies
the ORIGINAL final.pth hash, metadata and trainable-state digest before inference.
No retraining or new checkpoint selection.

Evaluate all2497 frozen GTA5 holdout IDs using the already fixed resized GT
and padding. Verify full ID digest, valid-pixel counts and identical PyTorch/
NumPy int64 confusion matrices. Class IoU=TP/(GT+Pred-TP); source mIoU is the
unweighted mean over positive-union classes, with absent classes marked null.
Also report image-mean CE and valid-pixel-weighted CE. These are descriptive
diagnostics of20-update weights, not formal M0 accuracy or model-selection
thresholds. Existing eight CE references must replay within1e-5 in the new
process, and model/CPU-CUDA RNG/original smoke files must remain unchanged.
No optimizer or target dataset is constructed by this evaluator.

The ordinary M0 runner remains blocked. After evidence returns, audit it
read-only before completing long-run/source-checkpoint and final-only target
evaluation preparation. No centered Query, new loss or40k authorization now.

## Completed bounded source smoke protocol

The saved complete `cb52071` source-data report passes the pinned local audit
in `tools/audit_strict_source_data.py`; see Phase17 `source_data_review.md`.
All24966 pairs and2497 fixed holdout transforms are accepted. The62 scale-only
differences obey the declared alignment. This does not establish scene independence.

Only the dedicated smoke entry point may now optimize:20 fresh seed0 updates,
one source image per update, original+photometric CE, Static R=2 unchanged.
L=CE(Z_original,Y)+lambda_cf*CE(Z_photo,Y), lambda_cf=1. Keep AdamW1e-4/wd0.01,
clip1, BF16 AMP and the40k poly0.9 horizon. As in the legacy loop, update1
uses the initial LR; after update t, LR_next=LR0*(1-t/40000)^0.9. The smoke
uses workers0 and an explicit seed0 source permutation (engineering scope,
not an exact worker4 long-run replay). No epoch, resume or source-selection
equivalence is claimed.

The smoke verifies alpha0 gives the base path, finite forward/backward/updates,
nonzero gradients in decoder/query/scale, unchanged frozen-backbone state,
one fixed-final compact checkpoint and exact trainable/optimizer restoration.
Eight fixed source-holdout samples receive compatibility forwards; no mIoU,
accuracy threshold, intermediate checkpoint or selection. Checkpoint prediction
roundtrip must be <=1e-5. Report memory including construction/restore overhead.
The helper reuses the same frozen backbone for the restored model; it does
not reload pretrained weights or prove independent-process continuation.

`protocol.training_authorized: false` in the old data-gate config continues
to block its general training entry point. The separate smoke tool hardcodes
its own authorization at20 updates and8 holdout samples and accepts no budget
override. Its report retains `formal_training_authorized: false`. No40k run,
target evaluator, centered Query, CQE/diversity or learned-null is enabled.

## Completed source data gate

The real extracted files have been independently verified and frozen at
`4314d61`; see `source_split_freeze_review.md` in the Phase17 evidence directory.
Pinned train/val counts are 22469/2497 and exact manifest hashes are unchanged.
ZIP bytes were not locally verified; the individual member files were.

`causalq/datasets/strict_source.py` reads the frozen hashes, independently
rechecks the split and requires exact filename membership in the accepted GTA5
payload roots. It builds separate train and holdout datasets. Training retains
the legacy random scale/crop/flip. Source evaluation uses the complete frame:
resize longest side to 1024 preserving aspect (round each dimension), then
pad bottom/right to a multiple of16; bilinear image, nearest-label interpolation,
RGB padding (124,116,104), ignore255 label padding. No crop/flip/RNG/photo on
holdout. Metrics, when later enabled, refer to resized source GT, not original
GT. These choices are fixed before seeing any new source/target performance.

`tools/train.py` now dispatches explicit strict-DG configuration to these
source datasets; configurations without protocol retain historical behavior.
The prepared M0 config has `training_authorized: false` and is rejected at main
before CUDA checks or data reads. The optimizer loop, checkpoint/selection
evidence and future final-only target evaluator still need integration after
this data gate; current code does not claim a runnable strict-DG training system.

AutoDL must verify the accepted full pair/path/size digest against actual data,
decode ALL train/holdout pairs, check train IDs in0..18/255, reject unreadable,
all-ignore or incompatible geometry, exercise every holdout transform and
32 fixed training crop/photo samples. No weights, models, optimization, GPU,
target image/label or mIoU. Source-only engineering acceptance does not prove
anchor effects, segmentation quality or scene independence.

## Historical transfer and export gates — completed

The operator reports successful export at `57a7244`: 22469 train / 2497 val,
zero overlap, matching file checksums. See
`../experiments/STRICT_DG_PHASE17/source_split_export_review.md` for hashes.
At that point the actual ZIP had not returned locally, so membership was not frozen.
The local importer was prepared to verify returned ZIP or extracted files,
which checks its pinned SHA256, manifest SHA256s, report contract and canonical
membership and independently recomputes the split before writing. Only then
commit the exact `manifests/gta5_source_{train,val}.txt` and provenance JSON.
The later extracted-file receipt completed that evidence prerequisite.

The operator's inventory summary reports 24,966 complete flat-numeric pairs,
zero missing/duplicate IDs and zero metadata candidates. See
`../experiments/STRICT_DG_PHASE17/source_inventory_review.md` for exact hashes
and the evidence limitations. Full original JSON bytes have not returned locally.
Use the plan's configurable filename fallback because the inventoried subtree
provides no usable grouping. Do not claim scene/sequence independence.

The exporter first verifies the pinned inventory SHA256, producer Git, empty
Git status, pair-record digest, counts, IDs, paths and reviewed metadata fields.
Let H(i)=SHA256(ASCII("causalq-gta5-source-split-v1" + LF + exact_ID_i)).
Order by (H(i), exact_ID_i), select the first ceil(N*f) for validation and put
all remaining IDs in training. Default f=0.1; no training seed participates.
For N=24966 this gives 2497 val and 22469 train. The fraction is configurable
before freezing; it is not a parameter to tune using target results. Exact
rational arithmetic avoids ambient decimal precision affecting the rounding.

Manifests preserve exact ID spelling/leading zeros, sorted lexically with LF
line endings; require unique IDs, nonempty sets, zero overlap and full coverage.
The metadata-only `source_split_export.zip` contains both manifests, a report
and SHA256SUMS. Return these bytes for local verification and Git freeze before
data/loader checks or training. This command opens only the saved inventory,
not dataset images/labels. No pixel-level data acceptance is implied.

## Historical versus new experiments

A0–A5, Static R=2's 65.2849% three-seed result and REIN runs are historical
references. Preserve their IDs, configuration, metrics and checkpoints exactly.
They used target feedback during project development and cannot populate the
new strict-DG table. The new protocol controls future access; it cannot erase
earlier knowledge of Cityscapes. Disclose that limitation in eventual reporting.

New experiment IDs: M0 (legacy Query parameterization), M1 (centered), P1
(prediction consistency), M2 (full CQE), M3 (anchor CQE), conditional M4.
Their results are all pending. No long training is authorized by this inventory.

## Source split gate

1. AutoDL inventories ONLY the GTA5 subtree: image/converted-label filenames,
   pairing, parent directories, filename patterns and candidate metadata.
2. Return `inventory.json`, `SHA256SUMS` and console log. Review sequence/scene
   evidence before assigning any member to training or holdout. Parent folders
   and numeric filename adjacency alone do not establish sequence identities.
3. Prefer whole sequences, then scenes, then a justified deterministic filename
   group. Use configurable deterministic 10% holdout only if useful grouping is
   unavailable. Record the limitation; do not silently randomize video frames.
4. Freeze `manifests/gta5_source_train.txt` and
   `manifests/gta5_source_val.txt` in Git, with exact source inventory identity,
   split rule and hashes. Require nonempty/disjoint/complete membership and no
   group overlap. All seeds use these same memberships.

The initial inventory gate did not create provisional manifests. After reviewing
its returned summary, the current gate exports candidates from the actual remote
ID list; no assumed consecutive numeric range is used.
The inventory hashes filenames/pair metadata, not image/label content. It does
not certify decoding, label IDs, grouping quality or pixel-level duplicates.
Source label/geometry inspection follows in a separate bounded AutoDL gate.

## Development and final testing

New development may read only GTA5 train/holdout. Use source mIoU and fixed
source effect diagnostics for decisions. The current `tools/train.py` and old
shell launchers are historical entry points, with periodic Cityscapes reads;
they must not be used for M0–M4/P1. A separate explicit strict route and tests
must be implemented after source split review. Strict-DG training is not yet
implemented by this commit.

Before final testing, freeze architecture, hyperparameters, training budget,
source checkpoint rule and selected checkpoint hashes in Git. Only after all
research choices are frozen may a separate evaluator read the fixed Cityscapes
500 manifest. No target smoke, diagnostic, intermediate metric, image statistics,
early stopping or checkpoint selection. Final evaluation does not feed back
into model development. Keep an access record; do not label historical evidence
as new strict-DG evidence.

## Mathematical boundary

Current q[c,r]=a[c]+e[c,r], Z=Z_base+alpha*Delta and E_full=alpha*Delta
remain unchanged. Later centered parameterization will impose
mean_r(e_tilde[c,r])=0. Anchor stability, utility and localization are hypotheses
to test on source holdout, not conclusions of a filesystem inventory.

## Acceptance and next decision

Current engineering acceptance: exact clean Git SHA, recorded Python/platform,
exit 0, `ok: true`, positive paired count, no missing or duplicate IDs, complete
inventory/hash files, and no target access/training. Metadata review remains
pending even when `ok: true`. Preserve failed reports and stop on failure.

Scientific question for this gate: does the available GTA5 structure provide
defensible sequence/scene grouping for an independent source holdout?
After the operator returns evidence, start with read-only verification and
metadata review, then implement the deterministic split and source-only route.
