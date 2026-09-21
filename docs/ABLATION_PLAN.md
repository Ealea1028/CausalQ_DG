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
