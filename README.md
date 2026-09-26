# CausalQ-DG

New contributors should begin with the standalone Chinese project guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It separates the implemented
method and verified results from the proposed next-generation theory roadmap.

Counterfactual Query Effect Distillation for domain-generalized semantic segmentation.

The project studies whether the prediction effect of class-specific semantic queries can remain stable under appearance-only interventions. The first implementation target is a frozen DINOv3 ViT-L/16 backbone with a lightweight segmentation head and a grouped causal-query residual branch.

## Current status

The A0--A5 core sequence and Phase 11--13 ablations are complete. Phase 12 selects R=2. In Phase 13, Static reaches `0.652849 ± 0.015138`, wins every paired seed against one-way, and also exceeds the `0.630394` seed-0 Bidirectional control. The subsequent three-seed zero-ablation audit finds an average inside/outside absolute-effect ratio of `2.913 ± 0.619`, with `86.48%` of GT-present class maps more strongly affected inside the matching region. Phase 14's isolated learned-null seed-0 run is stable and reaches `0.655364` final mIoU, `+0.7968` points over paired Static, while its calibration loss converges to `2.61e-13`. The next action is a read-only, paired zero-versus-learned-null localization analysis before repeat seeds or new objectives.

## Working directories

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

See `CausalQ_DG_Project_Plan.md` for the complete specification and `docs/AUTODL_NEXT.md` for the next remote action.
