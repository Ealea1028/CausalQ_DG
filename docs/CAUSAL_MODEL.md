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

The selected query branch uses static grouped queries and contributes an
additive class-specific logit residual without modifying the frozen image
representation. This structural separation makes `do(Q_c = 0)` operationally
explicit.

The hypothesis is not that absolute features must be identical under style change. It is that the contribution of the semantic query mechanism to the correct prediction should be stable across valid style interventions.

Phase 14 adds a class-agnostic learned-null bank with the same R query slots as
one semantic class. It is shared across all classes and calibrated to the
stop-gradient centroid of factual class-query states. The null intervention
replaces only class `c`'s factual query residual with this comparable shared
state. Its logit effect is `Z_c(factual) - Z_c(null)`. This phase does not yet
optimize effect invariance, sufficiency, or specificity.
