# Phase 14 learned-null smoke (superseded for formal comparison)

- Numerical status: passed.
- Exact Git SHA: `2882c87fcd825754937e5346923bd64cf9b1d676`.
- Isolation intent: selected Static R=2 factual path, a class-agnostic learned-null
  bank, original/photometric supervision, and only detached-centroid null
  calibration.
- Stability: 500 contiguous finite records; maximum objective reconstruction
  error `1.159e-7`.
- Learned-null signal: the first/last 20-step means are `1.96481e-4` and
  `2.63557e-5`, respectively, so the calibration loss decreases by about
  `86.6%`.
- Validation: 50 Cityscapes images at iteration 500; diagnostic mIoU
  `0.329539`.
- Peak allocated/reserved VRAM: `1.972/2.713 GiB`.
- Checkpoint: one `iter_000500.pth` file (`45,641,331` bytes).

This run establishes that the Phase 14 training path, loss, memory use, and
serialization are healthy. A post-smoke isolation audit found that constructing
the optional null bank before the factual Query Head consumed RNG and therefore
changed same-seed factual initialization. The implementation now constructs the
null bank after all factual modules and has an exact-parameter regression test.
Consequently, this smoke is retained as valid numerical evidence but is
superseded as the formal prerequisite: rerun the 500-step smoke on the repaired
SHA before any 40k learned-null experiment.
