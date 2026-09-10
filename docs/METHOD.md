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

## Success criterion

The decisive comparison is CQE against prediction consistency under the same query and style setup, accompanied by reduced cross-style query-effect variance.

