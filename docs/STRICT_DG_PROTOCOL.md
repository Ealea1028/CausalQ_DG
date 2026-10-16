# Strict-DG protocol — Phase 17 / Priority 0

Specification: `CausalQ_DG_1016.txt` (2026-10-16). This phase repairs the
development protocol before changing Query representation. Phase 16 stays closed.

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

This gate does not create provisional manifests: real GTA5 metadata is remote,
so selecting a split now would bypass the plan's required evidence.
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
