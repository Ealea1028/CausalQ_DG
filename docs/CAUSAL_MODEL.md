# Causal model

Variables:

- `D`: domain/environment
- `S`: style
- `C`: semantic content
- `X`: image
- `F`: foundation-model feature
- `Q`: semantic query
- `Y`: segmentation label
- `Y_hat`: prediction

The working graph treats the image as a joint consequence of content and style: `C -> X <- S`. A valid style intervention changes `S` while preserving `C`, geometry, and `Y`.

The query branch reads `F` through one-way cross-attention. It contributes an additive class-specific logit residual but does not write back into the base image representation. This structural separation makes `do(Q_c = 0)` operationally explicit.

The hypothesis is not that absolute features must be identical under style change. It is that the contribution of the semantic query mechanism to the correct prediction should be stable across valid style interventions.

