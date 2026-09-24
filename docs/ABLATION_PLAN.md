# Ablation plan

| ID | Query | Style | Prediction consistency | CQE | Diversity |
|---|---:|---:|---:|---:|---:|
| A0_DINOV3L_BASE | No | No | No | No | No |
| A1_QUERY | Yes | No | No | No | No |
| A2_QUERY_STYLE | Yes | Yes | No | No | No |
| A3_PRED_CONS | Yes | Yes | Yes | No | No |
| A4_CQE | Yes | Yes | Controlled | Yes | No |
| A5_FULL | Yes | Yes | Yes | Yes | Yes |

The primary comparison is `A4_CQE`/`A5_FULL` against `A3_PRED_CONS`. Query count, interaction type, style mechanism, and effect definition are varied only after the main sequence is stable.

Phase 10 uses `lambda_div=0.01` and applies squared off-diagonal cosine
decorrelation only to within-class query residuals. Retain the diversity term in
the final method only if the controlled full run is stable and improves mIoU by
at least 0.2 percentage points under the project protocol.

The fixed A5 run improved A4 by only 0.037 percentage points. It is retained as
a negative ablation result, and diversity is removed from the final method.

The style ablation holds Query, optimizer, seed, geometry, and total
counterfactual supervision weight fixed. It compares photometric-only,
Fourier-only, and the accepted combined A2 run.

Both isolated 500-iteration smoke tests passed. Run the 40k photometric-only
experiment first, retain its final checkpoint and compact evidence, reclaim its
intermediate-checkpoint space, and only then run the Fourier-only experiment.

Photometric-only completes at `0.638287` final Cityscapes mIoU and Fourier-only
at `0.627076`, versus `0.639610` for combined-view A2. Fourier-only is clearly
inferior at seed 0. The combined-versus-photometric gap is only 0.1324 points,
so repeat those two variants at seeds 1 and 2 and report paired mean and standard
deviation before closing Phase 11.

Combined-view seed 1 is complete at `0.651021` final Cityscapes mIoU. Run the
paired photometric-only seed 1 next, then repeat the same pair at seed 2.

Photometric-only seed 1 completes at `0.651486`, 0.0465 percentage points above
its paired combined-view run. Proceed with combined seed 2 followed by
photometric-only seed 2.

Combined-view seed 2 completes at `0.629514`. Run photometric-only seed 2 next;
then compute paired three-seed mean, sample standard deviation, and per-seed
differences to close Phase 11.

Phase 11 is complete. Combined obtains `0.640048 ± 0.010760`, photometric-only
`0.636201 ± 0.016427`, and their paired difference is
`0.003847 ± 0.005987`. Because the difference is small, variable, and reverses
at seed 1, use photometric-only as the parsimonious default.

Phase 12 varies residual queries per class at `R=1,2,4`, holding the selected
photometric-only protocol fixed. The existing seed-0 `R=3` result is the
reference. Run isolated 500-iteration smokes before full training.

The `R=1,2,4` GPU smokes passed at `be450a1` with finite contiguous traces,
exact objective reconstruction, correct mechanism isolation, and less than
`2.44 GiB` peak reserved memory. Proceed to 40k runs sequentially, beginning
with `R=1`; do not interpret the 50-image smoke mIoU as an ablation result.

R=1 completes stably at `0.603711` final Cityscapes mIoU, 3.4576 percentage
points below the fixed R=3 photometric-only reference. Retain this as a negative
capacity result, reclaim its intermediate checkpoints after recording, and run
R=2 next under the identical seed-0 protocol.

R=2 completes stably at `0.643725`, improving R=1 by 4.0014 percentage points
and R=3 by 0.5439 points. This is just above the project's `<0.5 mIoU`
multi-seed repeat trigger. Keep R=2 as the current candidate, run R=4 next,
then compare the planned query similarity, active-query behavior, and effect
variance before closing Phase 12.

R=4 completes stably at `0.624083`, trailing R=2 by 1.9642 percentage points
and R=3 by 1.4203 points. The complete final-mIoU ranking is therefore
`R=2 > R=3 > R=4 > R=1`. Run the shared 500-image mechanism analysis on all
four final checkpoints before selecting the Phase 12 default.

The common mechanism analysis selects R=2. Both queries are active
(`1.91/2` effective), while R=3/R=4 contextual query states are almost
identical (mean cosine approximately `0.999`). R=3 has slightly lower effect
variance but lower final mIoU and greater query cost. Phase 12 is closed with
R=2 as the fixed query count.

Phase 13 tests query interaction one mechanism at a time. The first control is
static R=2 queries under the selected photometric-only protocol, compared with
the existing one-way R=2 result. Run a 500-iteration GPU smoke before any full
training. Bidirectional interaction is deferred until the static control is
complete.

The Static Query smoke passes at `367694e` with finite contiguous records,
exact objective reconstruction, correct mechanism isolation, and `2.258 GiB`
peak reserved memory. Proceed to its seed-0 40k run; do not interpret the
50-image smoke mIoU or add bidirectional interaction yet.

The Static Query seed-0 full run completes at `0.647396` final Cityscapes
mIoU, 0.3671 percentage points above the one-way R=2 reference (`0.643725`).
This gap is below the prespecified `<0.5 mIoU` repeat trigger. Repeat the
one-way and static variants as paired seeds 1 and 2, beginning with one-way
seed 1; defer bidirectional interaction until the paired result is resolved.

The first one-way seed-1 attempt at `367694e` is invalid and excluded: it
stopped after 3,316 iterations when all random crop attempts missed valid
pixels for sample `13286`. A full 24,966-label audit found no all-ignore or
unreadable source label, identifying a crop-sampling edge case. Commit
`3860e69` adds a valid-pixel fallback and must pass a fresh 500-iteration GPU
smoke before the one-way seed-1 40k repeat restarts from initialization.
