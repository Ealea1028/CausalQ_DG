# Phase 14 isolated learned-null smoke

- Status: passed.
- Exact Git SHA: `d85db4e0dcf411681c1ddad09080dfeed1c419bd`.
- Isolation: selected Static R=2 factual path, one class-agnostic learned-null
  bank, original/photometric supervision, and only detached-centroid null
  calibration.
- Same-seed contract: the local regression suite verifies exact equality of all
  factual parameters with learned-null disabled versus enabled.
- Stability: 500 contiguous finite records; maximum objective reconstruction
  error `1.790e-7`; no runtime error.
- Learned-null signal: first/last 20-step means `2.141e-4/3.259e-5`, an
  approximately `84.8%` decrease.
- Validation: 50 Cityscapes images at iteration 500; diagnostic mIoU
  `0.329528`.
- Peak allocated/reserved VRAM: `1.972/2.713 GiB`.
- Checkpoint: one `iter_000500.pth` file (`45,641,331` bytes).

This refreshed run supersedes the mislabeled old-SHA attempt and passes the
formal Phase 14 smoke gate. Proceed to one seed-0 40k learned-null run from the
same implementation SHA. Do not add invariance, sufficiency, specificity, or
semantic-counterfactual objectives.
