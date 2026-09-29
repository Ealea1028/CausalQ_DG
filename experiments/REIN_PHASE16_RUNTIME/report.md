# Phase 16 feasibility and runtime acceptance

Scope: GTA5 to Cityscapes only. Evidence supplied by the AutoDL operator;
no local GPU results are claimed.

## Feasibility assessment

Engineering remains feasible: the pinned DINOv2 conversion passed exact
343-tensor audit; the isolated REIN runtime now passes GPU NMS, xformers
attention and model registration. This establishes runtime compatibility,
not full-model or optimization compatibility.

The empirical project remains useful: selected Static R=2 yields final
Cityscapes mIoU `0.652849 ± 0.015138` across three seeds. The two-size study
consistently favors ViT-L in accuracy and normalized effect stability.
These are bounded protocol results, not a scaling law or causal identification.

The original central claim is not validated: A4 CQE lowers normalized style
effect variance by 99.47% but loses 2.95 accuracy percentage points versus A3.
Query diversity fails retention; learned-null worsens localization despite
seed-0 segmentation retention. REIN transfer therefore remains an exploratory
test, not an assumed positive result. Do not add CQE until a credible REIN
baseline is established. Repeated use of Cityscapes validation for development
and the single target constrain generalization claims.

## Accepted runtime evidence

- Project source: `0e0ef7c7bb1588cb0ba190c652fbce926c257121`.
- Upstream REIN: `dc063429c4dadc0da9c6252b3db22fc55a9882ab` (clean).
- Python 3.10.21; torch 2.0.1+cu118; torchvision 0.15.2+cu118;
  numpy 1.26.4; mmcv 2.1.0; mmengine 0.10.7; mmseg 1.2.2;
  mmdet 3.3.0; xformers 0.0.20.
- RTX 4090 D; CUDA 11.8; CUDA NMS indices `[0]`.
- Attention max absolute error `0.0005131959915161133` <= `0.005`.
- Backbone/head registered: true/true; runtime `ok: true`.
- Report SHA256: `2d3498fc8343927da29dfa2e73bb5fb89ad46132248d15e6a323cdef4adcdfcb`.
- stderr SHA256: `d35e74535a57ec10102ddfa16054f5f3f7d61636b7cf166085c58f6ab6fec8f2`.
- Installed freeze SHA256: `47f0e84c38075e5c2f70ffba58746844e0cc8f206f430f2b295fb928008e3245`.
- Available disk approximately 9.9 GiB; no formal training authorized yet.

Next boundary: synthetic backbone weight loading, forward and adapter backward
on AutoDL. Mask2Former execution, dataset protocol and training remain pending.
