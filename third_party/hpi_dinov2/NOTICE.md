# DINOv2 / HPI architecture reference

The pure PyTorch state-compatible implementation in
`causalq/models/dinov2_wrapper.py` is adapted from Meta DINOv2 architecture
and layer definitions (Copyright Meta Platforms, Inc. and affiliates,
Apache License2.0). The unmodified upstream license is in LICENSE.

- Meta: https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8
- HPI conversion reference: https://github.com/MosukFate/HPI/blob/6512993679a682422f82f675f4c6e9a2ccce4c9b/models/backbones/clip/models_hpi.py#L699

Changes: remove framework/MMCV dependencies, all VLM components and training
stochastic layers; use equivalent PyTorch SDPA; expose raw post-block maps
and final normalized dense tokens; explicitly adapt patch14 to16 and
position37x37 to32x32. Retain DINOv2 LayerNorm eps1e-6, LayerScale parameter
names, MLP GELU, CLS/mask state and runtime interpolation offset0.1.
This does not reproduce the full HPI model or its performance.
