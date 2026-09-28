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

The conversion passed on AutoDL at Git SHA
`a3331a6520c01f8f8f0f8a65073bde85e11dda25`. The new file is
`/root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth`, size
`1,216,918,112` bytes, SHA256
`91730ebf59fb634f5572cf5071fef8665473dcffcbef7ba4f4fa497533a8c837`.
The report confirms 343 tensors, patch embedding `1024×3×16×16`, and
position embedding `1×1025×1024`; its own SHA256 is
`a77862783c247208d867f564d51b118336db4475e759f26ff198b2ce6e6b9fbb`.
The independent on-disk audit passed at Git SHA
`8ee5abf9d927743a5a88fa8b4691fd0cd4bb2301`: both fixed hashes matched,
and all 343 keys, shapes, dtypes, and tensor values matched a newly computed
conversion. The audit report SHA256 is
`6cdbdf1b503f7f6c7eb58287088d99de1edde8b378e7304add140e5418a4c120`.
AutoDL had 476 GiB available system RAM and 16 GiB free data volume.

The pinned upstream REIN commit still contains the DINOv2-L GTAV configuration
and `ReinMask2FormerHead`, despite its current README warning that a newer
feature-extractor path omits Mask2Former features. Its config uses 100
Mask2Former prediction queries, a 24-layer DINOv2-L backbone, a 16×16 patch
kernel, 512×512 crops, and 40k iterations. It also validates a concatenated
Cityscapes/BDD100K/Mapillary dataset by default, which must be replaced by
Cityscapes-only for this user-scoped study. Its GTA path and label suffix do
not match the inspected AutoDL GTA5 tree; those must be overridden in a
versioned experiment configuration, not altered in the datasets or upstream
source. The upstream repository is GPL-3.0; it is checked out separately,
not vendored into this project.

The next gate creates an isolated Python 3.10 / PyTorch 2.0.1 CUDA 11.8 /
MMCV 2.1 runtime and checks package pins, a CUDA extension, REIN imports,
and config parsing. This does not load the 1.2 GB backbone or build a
segmentor. Keep the existing `causalq-dg` environment untouched. A clean
runtime is a prerequisite, not evidence of model compatibility or quality.

The first bootstrap at `893c9de308a868e2b9cbe6f9cab054c5916090b9`
cloned the pinned REIN source successfully but stopped during Conda metadata
retrieval from `repo.anaconda.com` (HTTP 000). No PyTorch installation or
runtime compatibility result was obtained; the converted weights remain
valid. Recovery explicitly reuses the clean pinned external checkout and
creates a new `rein-phase16-py310-cu118-tuna-retry1` prefix. Conda uses
`--override-channels` and the Tsinghua main mirror for this command only,
without altering global configuration or TLS verification. All old paths
and logs are retained. The runtime gate remains pending; no training is
authorized by this network recovery.

Before a CQE experiment, define a small class-specific residual branch on
top of a fixed REIN segmentor and compare that branch **without CQE** against
the same branch **with CQE**. Native REIN remains a separate reference. The
100 native Mask2Former queries are not class-indexed; assigning class labels
to them post hoc would change the meaning of the original `do(Q_c=0)` and
cannot by itself substantiate a faithful CQE transfer. No full run is
authorized until this branch, its class-isolation tests, and a matched short
GPU smoke are ready.
REIN's 100 Mask2Former prediction queries are
not intrinsically class-specific; a faithful class-specific CQE intervention
still needs an explicit design and isolated unit tests before any training.
