# Phase 16: DINOv2-L/REIN transfer feasibility and protocol

This is the original project plan's §43 follow-up, restricted by the user to
GTA5 training and Cityscapes validation. Phase 15 is closed; no new DINOv3
scaling run is authorized by this phase.

## Current resource gate

The 2026-09-27 AutoDL inventory at Git SHA
`9517909955d11dc238ccfff31d741a062d29cd15` found the GTA5 and
Cityscapes validation roots and the DINOv3-B/L weights, but no DINOv2 or
REIN directory/weight. The active environment has PyTorch `2.7.0`,
Torchvision `0.22.0`, Transformers `4.56.1`, and timm `1.0.15`; MMCV,
MMEngine, and MMSegmentation are absent. The data volume has 18 GiB free.

The [official DINOv2 repository](https://github.com/facebookresearch/dinov2)
offers `dinov2_vitl14`, and the
[official REIN repository](https://github.com/w1oves/Rein) links the original
`dinov2_vitl14_pretrain.pth` file and a GTAV-to-Cityscapes configuration. Its
documented installation uses PyTorch `2.0.1`/CUDA `11.7` plus OpenMMLab
packages, not the project's PyTorch `2.7.0`/CUDA `12.6` environment. Do not
install those requirements into the existing `causalq-dg` environment or
assume they work unchanged on RTX 4090 D. No upstream checkpoint or protocol
number may be treated as directly comparable to this project's results
without matching data, crop, validation, and checkpoint-selection rules.

## Controlled study, if the resource and compatibility gates pass

1. Pin the official DINOv2 and REIN source revisions and record the official
   DINOv2-L checkpoint SHA256 before loading any pickle-based `.pth` file.
2. In an isolated environment, reproduce a **REIN without CQE** reference
   trained on GTA5 only and evaluated on the same 500 Cityscapes validation
   images. Record full preprocessing, resolution, backbone, pretraining,
   optimizer, schedule, seed, and final-iteration selection.
3. Specify how REIN's prediction queries define a class-specific ablation
   effect without altering non-target classes. Add only the existing CQE
   objective, leaving architecture, data, augmentation, and schedule fixed.
   Add CPU tests for class isolation, gradient flow, and loss reconstruction
   before any GPU run.
4. Run a short GPU smoke before a full matched REIN/CQE pair. A lower effect
   variance without retained mIoU is a negative result, as in Phase 9.

The CQE objective failed the DINOv3-L segmentation acceptance gate. Its
transfer is therefore exploratory; it is not an established improvement and
cannot prove the method is backbone-agnostic without the matched control.
The official DINOv2-L weight was downloaded on AutoDL to
`/root/autodl-tmp/pretrained/.dinov2_vitl14_pretrain_phase16.pth.part` from
the author-linked URL. The transfer returned success, its size is
`1,217,586,395` bytes, and its SHA256 is
`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
The hash-pinned, `weights_only=True` CPU audit passed at Git SHA
`26ea45f41a7b09e163ba87d2351a5c5d0108a2ff`: 343 tensors,
304,368,640 parameters, 24 blocks, 1,024 channels, a 14×14 patch kernel,
and a 37×37 patch-position grid plus one class token. The audit report is
`/root/autodl-tmp/outputs/CausalQ_DG/analysis/dinov2_vitl14_staged_audit.json`
with SHA256
`689e52472cf0ddf1bcae257d03e3685bcc565ce73d8f071a7137f2e4dd3b9368`.
This validates the original checkpoint's structure, not REIN compatibility or
segmentation quality.

The official REIN repository is pinned for protocol inspection at commit
`dc063429c4dadc0da9c6252b3db22fc55a9882ab`. Its DINOv2 conversion
uses bicubic interpolation (`align_corners=False`) to change the 14×14 patch
kernel to 16×16 and its 37×37 position grid to 32×32 for 512×512 crops.
The repository's `tools/convert_dinov2_for_rein.py` reproduces just this
tensor conversion, checks the pinned input SHA before deserialization, uses
`weights_only=True`, refuses an existing output, and reports the converted
file SHA. A converted checkpoint is not an end-to-end REIN smoke test.

The next AutoDL step converts only this audited file into a new path and
checks its tensor layout. It does not install OpenMMLab, load REIN, train,
or overwrite the original. REIN's 100 Mask2Former prediction queries are
not intrinsically class-specific; a faithful class-specific CQE intervention
still needs an explicit design and isolated unit tests before any training.
