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

## Accepted backbone evidence

- Project source: `ed3dcfe202435f4d7893681054473724e19e1ad6`.
- Upstream and all eight runtime versions unchanged from accepted runtime above.
- Converted weights SHA256: `91730ebf59fb634f5572cf5071fef8665473dcffcbef7ba4f4fa497533a8c837`.
- 343 loaded tensors; 2,990,081 trainable adapter parameters.
- Feature shapes: `[1,1024,128,128]`, `[1,1024,64,64]`,
  `[1,1024,32,32]`, `[1,1024,16,16]`; linked queries `[100,256]`.
- Synthetic objective `8.531936645507812`; 11 nonzero adapter gradients;
  frozen-backbone gradients absent. This is not a segmentation loss.
- Peak allocated/reserved memory `2.767/2.982 GiB`; exit 0 and `ok: true`.
- Report SHA256: `317dff2553f4674f9363d8660ce369245863676866fa0cf5ed0745db1bb53706`.
- stderr SHA256: `c4775f166f11b2cbb18b26ecb5487133076edf537e05943acf6e29501284ed1f`.
- Disk available 9.9 GiB. Optional ConvNeXt and TypedStorage warnings are not
  failures of this gate.

Next boundary: synthetic full Mask2Former prediction/loss/backward on AutoDL.
The pinned upstream model config is used unchanged except disabling automatic
pretrained initialization, followed by our safe explicit checkpoint load.
The test has no dataset/optimizer/checkpoints/CQE; real protocol and training
remain pending. MMSegmentation interfaces were checked against
[v1.2.2 EncoderDecoder](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/segmentors/encoder_decoder.py)
and [Mask2FormerHead](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/decode_heads/mask2former_head.py).

Local verification: 142 CPU tests pass using the existing local PyTorch
interpreter with pytest from the base Anaconda site-packages; plain
`python -m pytest` in that interpreter could not find pytest. No OpenMMLab
model/real checkpoint was loaded locally. Bash execution and full GPU model
execution are pending AutoDL verification.

## Accepted full-segmentor synthetic evidence

- Source `86fb3ffdc7a13285965f0df825472625f9e52ed4`; pinned upstream/runtime/weights unchanged.
- Upstream model-config hash `2f51a8ac089848f5e276028801983c95bf5f89e688753c1fff5a9e64bb766868`.
- 343 loaded tensors; 23,569,877 trainable parameters; 19-class scores `[1,19,512,512]`.
- Float32; all 30 losses finite; total `124.92772674560547`.
- Nonzero adapter/head/pixel-decoder gradient counts `11/291/116`; no frozen gradients.
- Peak allocated/reserved `3.400/3.662 GiB`; `ok: true`, stage complete, exit 0.
- Report hash `87ba441bb9f879fd4048c18518477e8294c06dfa27335f7811954bd0c7552301`.
- stderr hash `05a9a5cc9f11d36804cb7a20fa4832afbe547937dde1d678284a69e28cd2bca3`.
- 9.9 GiB disk free. Synthetic loss is NOT an accuracy result.

Next: 20 GTA5 optimizer updates and five Cityscapes center-crop inference
diagnostics in float32. Uses existing project preprocessing rather than
claiming exact upstream training reproduction. The inverse RGB-to-BGR bridge
must round-trip through upstream preprocessing within 1e-5; target labels
never enter optimization. No checkpoints/mIoU/CQE. Formal source-only schedule
and full-target evaluation remain pending. Local CPU tests: 145 pass; actual
isolated-runtime dataset imports and optimizer GPU memory await AutoDL.
