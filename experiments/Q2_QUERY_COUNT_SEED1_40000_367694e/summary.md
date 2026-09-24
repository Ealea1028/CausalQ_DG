# Phase 13 paired repeat: one-way R=2 seed 1 failed attempt

- Status: failed and excluded from all comparisons.
- Exact Git SHA: `367694ee8bc2e670be0a896f40e089760fe2177a`.
- Intended protocol: one-way R=2, photometric-only, seed 1, 40,000
  iterations.
- Progress before failure: 3,316 completed iterations and six intermediate
  checkpoints through iteration 3,000.
- Failure: random crop generation returned an all-ignore crop for sample
  `13286`; the dataset guard stopped training before the optimizer consumed
  that sample.
- Full GTA5 audit: all 24,966 pairs and labels are readable, all train IDs are
  valid, no label is entirely ignored, minimum full-label valid fraction is
  `0.134332`, and all 62 size mismatches are scale-equivalent.

The evidence rules out corrupt or unsupervised source labels. The failure is a
preprocessing edge case: all configured random crop attempts can miss the
valid region even when the full label contains supervision. Do not resume this
run. Restart from random initialization under a new Git SHA after validating a
fallback crop that is guaranteed to include a real valid pixel.
