# A3/A4 cross-style query-effect variance

- Status: Phase 9 second target passed.
- Evaluation Git SHA: `38c2df3470322dacd937aed17518d67dc9c5e3a1`.
- Protocol: all 500 Cityscapes validation images, 6,005 ground-truth-present class maps, and identical per-sample appearance interventions for both checkpoints.
- A3 prediction-consistency variance: `0.0019905460`.
- A4 CQE variance: `0.0000105223`.
- A4/A3 ratio: `0.00528615`.
- Relative reduction: `99.4714%`.
- Peak allocated/reserved VRAM: A3 `4.004/5.416 GiB`; A4 `4.567/5.998 GiB`.

CQE decisively reduces normalized cross-style query-effect variance. This does not reverse the separate accuracy finding: A4 remains 2.95 Cityscapes mIoU percentage points below A3 and fails the Phase 9 primary target.
