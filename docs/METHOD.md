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
