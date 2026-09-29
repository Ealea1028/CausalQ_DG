# CausalQ-DG

New contributors should begin with the standalone Chinese project guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It separates the implemented
method and verified results from the proposed next-generation theory roadmap.

Counterfactual Query Effect Distillation for domain-generalized semantic segmentation.

The project studies whether the prediction effect of class-specific semantic queries can remain stable under appearance-only interventions. The first implementation target is a frozen DINOv3 ViT-L/16 backbone with a lightweight segmentation head and a grouped causal-query residual branch.

## Current status

The A0--A5 core sequence and Phase 11--15 studies are complete. Static R=2 remains the supported model (`0.652849 ± 0.015138` final Cityscapes mIoU over three seeds). Phase 14's learned-null variant improves seed-0 segmentation but worsens effect localization, so it remains a negative ablation. The §41 GTA5 → Cityscapes figures are qualitative diagnostics only. The completed §42 backbone comparison finds ViT-L above ViT-B in mIoU (`0.652849` versus `0.563878` mean) and below it in normalized cross-style effect variance (`0.000567443` versus `0.000961160` mean) for all three paired seeds. This is a two-size association, not a scaling law or causal proof; CQE gain under scaling remains untested after the CQE acceptance failure. Other target datasets remain deferred. See `experiments/BACKBONE_SCALING_STATIC_R2_3SEED/report.md` and `docs/AUTODL_NEXT.md`.

## Working directories

Phase 16's pinned REIN runtime now passes CUDA NMS and xformers attention.
The backbone and full Mask2Former synthetic forward/backward gates also pass.
The 20 source-data optimizer updates returned finite losses and exit 0.
The complete report passes target inference, frozen-weight, normalization and
memory audit (3.881 GiB peak reserved). Pinned protocol inventory exits 0.
Adapted GTA5/Cityscapes data pipelines now pass the operator's gate at b645793.
The 80-microbatch / 20-update accumulated PolyLR gate passes at 3ca6731,
including exact compact checkpoint restoration (3.920 GiB peak reserved).
Five-image original-resolution slide evaluation exits 0 at 59d522b; its low
20-update diagnostic scores are not accuracy acceptance. The 500-update pilot
exits 0 at b4e292d and passes saved-evidence audit at 3601ecd (50-image
diagnostic mIoU 0.235189, checkpoint roundtrip 0, peak reserved 4.217 GiB).
Cache inventory finds no useful data-volume cache savings. Next verify bounded
checkpoint continuation including RNG and source sampler before the formal baseline;
formal source-only training and
CQE transfer remain pending.
See `experiments/REIN_PHASE16_RUNTIME/report.md` for the
feasibility assessment and `docs/AUTODL_NEXT.md` for the fixed remote boundary.

- Local development: this repository
- AutoDL project: `/root/autodl-tmp/CausalQ_DG`
- AutoDL datasets: `/root/autodl-tmp/datasets`
- AutoDL pretrained weights: `/root/autodl-tmp/pretrained`
- AutoDL outputs: `/root/autodl-tmp/outputs/CausalQ_DG`

## Local verification

```bash
python -m pytest
```

Local verification is CPU-only. Do not use a local GPU for formal DINOv3-L training.

## Project phases

1. Repository scaffold and contracts
2. AutoDL environment verification
3. Dataset inspection and conversion
4. DINOv3 backbone smoke test
5. Source-only baseline
6. Query-only model
7. Style intervention
8. Prediction-consistency control
9. CQE training
10. Query diversity
11. Style ablation
12. Query-count ablation
13. Query-interaction ablation
14. Learned-null intervention baseline
15. DINOv3-B scaling control (GTA5 → Cityscapes only)
16. DINOv2-L/REIN transfer feasibility (in progress; GTA5 → Cityscapes only)

See `CausalQ_DG_Project_Plan.md` for the complete specification and `docs/AUTODL_NEXT.md` for the next remote action.
