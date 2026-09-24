# Method contract

## Objective

CausalQ-DG measures and distils the prediction contribution of class-specific semantic queries across appearance-only counterfactual views.

## First implementation target

- Frozen DINOv3 ViT-L/16 image backbone.
- One-way query-to-image cross-attention; image features are not modified by queries.
- Nineteen Cityscapes classes and three queries per class.
- A base segmentation path plus a query residual path.
- A learnable residual scale initialized to zero.

## Interventions

The original image, a photometric view, and a Fourier-amplitude view must share the same geometry and label. The first implementation does not use CycleGAN or geometric warping.

## Causal effect

For class `c`, intervening with `do(Q_c = 0)` removes only the class-query residual logit. It does not rerun or alter the backbone. The initial logit-level effect is the scaled class-query residual.

## Training order

Establish a credible source-only baseline before adding queries. Then add style augmentation, prediction consistency, and CQE in separate phases so every contribution has a controlled comparison.

## Prediction-consistency control

Phase 8 keeps the accepted Query + Style setup fixed. For each photometric or
Fourier view, it minimizes `KL(P_original || P_style)` on valid label pixels.
The original-view probabilities are detached, the two style losses are averaged,
and no causal-query-effect or diversity term is enabled.

## Causal-query-effect distillation

Phase 9 defines the public logit effect as `alpha * delta_logits`. The original
view is a stop-gradient reference. Each effect map is L2-normalized over valid
pixels, and SmoothL1 distances are averaged across counterfactual views and only
the semantic classes present in the current ground-truth mask. Prediction
consistency and diversity remain disabled in this phase.

## Success criterion

The decisive comparison is CQE against prediction consistency under the same query and style setup, accompanied by reduced cross-style query-effect variance.

Cross-style query-effect variance is measured on the same original,
photometric, and Fourier views for both checkpoints. Each valid-pixel class
effect map is L2-normalized over space, population variance is computed across
the three views and summed spatially, and results are averaged only over
ground-truth-present class maps. The evaluation uses a fixed per-sample seed so
the compared checkpoints receive identical interventions.

The fixed Phase 9 result passes the variance target but fails the segmentation
target: A4 reduces normalized effect variance by 99.47% relative to A3 while
losing 2.95 Cityscapes mIoU percentage points. Both outcomes are retained; the
variance result does not override the failed accuracy criterion.

## Query diversity

Phase 10 composes the already implemented prediction-consistency and CQE
objectives, then adds one new mechanism: a lightweight diversity penalty on the
learned residual queries. Within each semantic class, residual queries are
L2-normalized and the squared off-diagonal cosine similarities are averaged.
The fixed weight is `lambda_div=0.01`. The class anchors and contextualized
image-conditioned query states are not regularized by this term.

The fixed Phase 10 run is numerically stable and drives the diversity loss from
`1.11e-3` to `1.07e-7`, but reaches only `0.592124` Cityscapes mIoU. Its gain
over A4 is 0.037 percentage points, below the predefined 0.2-point retention
threshold. Query diversity is therefore excluded from the final method.

## Style ablation

Phase 11 returns to the accepted A2 Query + Style objective and disables every
consistency loss. Photometric-only and Fourier-only variants are compared with
the existing combined-view A2 result. The total counterfactual supervision
weight remains `lambda_cf=1`: a single intervention receives weight 1, while
the combined run assigns weight 0.5 to each of its two interventions.

Both isolated 500-iteration smoke tests passed at commit `b0d6e14`: their
records were contiguous and finite, their objectives reconstructed exactly,
and their metadata contained no prediction-consistency, CQE, or diversity
configuration. Full-run conclusions are intentionally deferred until both
40,000-iteration variants have been evaluated on all 500 Cityscapes validation
images.

The fixed photometric-only run reaches `0.638287` final Cityscapes mIoU,
0.1324 percentage points below the combined-view A2 reference. Its best
intermediate validation is `0.641253` at iteration 30,000, but the prespecified
final-iteration result remains the primary comparison. Fourier-only reaches
`0.627076`, 1.2534 points below combined and 1.1211 below photometric-only.
The Fourier result is decisive at seed 0. The combined-versus-photometric gap
is below the project's 0.5-point repeat threshold, so those two variants must
be repeated with seeds 1 and 2 before drawing the final style conclusion.

The combined-view seed-1 repeat is numerically stable and reaches `0.651021`
final Cityscapes mIoU. Its learned residual scale is negative, which is valid
for the unconstrained alpha parameter and does not indicate instability. The
paired photometric-only seed-1 result is required before any cross-seed style
comparison is made.

Photometric-only seed 1 reaches `0.651486`, exceeding the paired combined-view
result by 0.0465 percentage points. This reverses the seed-0 ordering, where
combined led by 0.1324 points. The seed-2 pair is therefore necessary for the
prespecified three-seed mean and standard deviation.

Combined-view seed 2 reaches `0.629514` final Cityscapes mIoU. Its last logged
gradient norm is a finite pre-clipping value; the fixed max-norm `1.0` clip is
applied before the optimizer step. The run otherwise satisfies every stability
and provenance gate. Photometric-only seed 2 is the remaining paired run.

The completed paired comparison gives combined `0.640048 ± 0.010760` and
photometric-only `0.636201 ± 0.016427` final Cityscapes mIoU. The paired
combined-minus-photometric difference is `0.003847 ± 0.005987`, changes sign at
seed 1, and is too unstable to support a superiority claim. Photometric-only is
selected as the lower-cost default; combined and Fourier remain ablations.

## Query-count ablation

Phase 12 fixes the selected photometric-only style protocol and seed 0 while
varying only the number of residual queries per class: `R=1,2,4`. The existing
photometric-only `R=3` seed-0 run is the reference. Initial GPU work is limited
to 500-iteration smoke tests before any 40k run.

All three isolated smoke tests passed at commit `be450a1`. Each produced 500
contiguous finite records, exact objective reconstruction, one 50-image
validation, and one checkpoint, while preserving the intended query count and
excluding every auxiliary consistency loss. Peak reserved memory remained
between `2.434` and `2.438 GiB`. Their short-run mIoU values are diagnostic only
and are not used for model selection. Full runs proceed one query count at a
time, beginning with `R=1`.

The R=1 full run is stable but reaches only `0.603711` final Cityscapes mIoU,
compared with `0.638287` for the fixed photometric-only R=3 reference. The
3.4576-point deficit occurs despite 40,000 finite contiguous records, exact
objective reconstruction, and the intended isolated protocol, so R=1 is
retained as a negative capacity ablation. Phase 12 continues with R=2.

The R=2 full run reaches `0.643725` final Cityscapes mIoU, improving R=1 by
4.0014 percentage points and the fixed R=3 reference by 0.5439 points. Its
40,000-record trace, objective reconstruction, isolation, and 80 full
validations all pass. R=2 is the current best candidate, but selection remains
open until the R=4 result and query-behavior diagnostics are available.

The R=4 full run is also stable and reaches `0.624083`, 1.9642 percentage
points below R=2 and 1.4203 points below R=3. Final mIoU across `R=1/2/3/4`
is `0.603711/0.643725/0.638287/0.624083`; R=2 is the current leader.

The closing diagnostic uses the same 500 Cityscapes validation images and the
deterministic original/photometric view pair for every checkpoint. It reports:

- unordered within-class cosine similarity for both learned residual queries
  and contextualized queries (the latter only for ground-truth-present
  classes);
- per-query logsumexp responsibility on downsampled ground-truth-class pixels,
  summarized by entropy-based effective query count and fraction, dominant
  mean share, and mean per-pixel top-1 responsibility;
- population variance across the two views after valid-pixel L2 normalization
  of each ground-truth-present class effect map.

These metrics distinguish redundant capacity from genuinely active and
style-stable queries. Phase 12 remains open until this common analysis is
complete.
