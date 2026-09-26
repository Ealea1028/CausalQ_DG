# CausalQ-DG

New contributors should begin with the standalone Chinese project guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It separates the implemented
method and verified results from the proposed next-generation theory roadmap.

Counterfactual Query Effect Distillation for domain-generalized semantic segmentation.

The project studies whether the prediction effect of class-specific semantic queries can remain stable under appearance-only interventions. The first implementation target is a frozen DINOv3 ViT-L/16 backbone with a lightweight segmentation head and a grouped causal-query residual branch.

## Current status

The A0--A5 core sequence and Phase 11--14 mechanism studies are complete. Static R=2 remains the supported model (`0.652849 ± 0.015138` over three seeds). Phase 14's learned-null run improves seed-0 segmentation by `0.7968` points, but its paired mechanism audit is negative: the effect localization ratio falls from `3.773` with zero ablation to `1.845`, normalized contrast falls from `0.491` to `0.205`, and only `4.71%` of GT-present class maps improve. Learned-null is therefore retained as a negative ablation rather than the causal baseline. At the user's request, evaluation on datasets other than Cityscapes is deferred. Four §41 GTA5 → Cityscapes val figures have been provisionally reviewed; their AutoDL provenance report remains to be checked before closing the phase. See `docs/AUTODL_NEXT.md`.

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
