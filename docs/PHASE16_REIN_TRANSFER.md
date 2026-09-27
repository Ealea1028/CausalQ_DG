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
The immediate AutoDL step only acquires the official DINOv2-L weight into a
staging path. It must **not** load, convert, install, train, or overwrite
anything until its size, SHA256, and provenance are reviewed locally.
