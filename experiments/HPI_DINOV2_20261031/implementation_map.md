# Phase33 implementation map

The Phase28–32 DINOv3 study is closed. This is a new exploratory study after
historical target feedback. Old target U0/UA = 65.2091/66.0753%; new scores
remain null. B0 is the new U0 alias, never an eighth training run.

Scientific recipes: `configs/strict_dg/formal/m0_seed0.yaml` (U0) and
`configs/diagnostic_20261020/formal/ua_seed0.yaml` (UA). Preserve uncentered
q=a+e, R=2, static logsumexp, alpha0, two summed CE losses, AdamW1e-4,
poly.9/40000, clip1, BF16, 512 geometry and original photometric distribution.
UA retains repeated-Anchor CQE with detached original, differentiable photo,
L2-per-class valid-map normalization, SmoothL1 beta1/weight1. Residual
Contribution means Full minus Anchor; it is not residual-only causal effect.

Reuse: BaselineDecoder, QuerySegmentor, TrainTransform, StyleInterventionBank,
GTA5 manifests/reader and loss math. Add feature-consuming QuerySegmentor API;
the image API still delegates through the same head. New protocol name grants
uncentered Anchor access without weakening legacy study or centered guards.
Main decoder reads raw block6/12/18/24 outputs; query readout uses normalized
final tokens. Existing static Query does not write into the main decoder.

Backbone: pure PyTorch DINOv2 state-compatible implementation, no MMCV/timm/VLM.
HPI commit6512993679a682422f82f675f4c6e9a2ccce4c9b CLIP models_hpi.py699–740
defines L/16 and strict bicubic kernel/position conversion. Meta DINOv2
commit7764ea0f912e53c92e82eb78a2a1631e92725fc8 defines LayerScale/LN/MLP,
CLS/mask state and historical runtime position offset0.1. Conversion changes
only kernel14→16 and spatial position37→32, preserving CLS and every other
tensor. Runtime block indices5/11/17/23 are raw, final patch tokens use LN.
SDPA uses original head scale, dropout0, noncausal. Apache license retained.

Execution changes: canonical SHA256 step/view seeds replace old global RNG
dependence while retaining distributions; randperm(seed+epoch) membership.
Worker/physical batch values come only from the controlled runtime profile.
New cohorts share one frozen backbone and original features, with independent
head parameters/optimizer/RNG/cursor. No logits or updated features are shared.
Window recovery aligns absolute steps, skips each already committed head.
CPU transforms use explicit generators; fallback threads do not reseed globals.
Logical batch remains1; batched floating point kernels need not match old bits.

Phase33 releases prepare/status/package only, zero formal optimizer updates.
Hardware/weight/runtime identities remain pending AutoDL measurement. Future
training actions reject execution until Phase34 freezes the returned profile.
No B1–B5 modules or new scientific losses are implemented in this phase.
