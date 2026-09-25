# CausalQ-DG

New contributors should begin with the standalone Chinese project guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It separates the implemented
method and verified results from the proposed next-generation theory roadmap.

Counterfactual Query Effect Distillation for domain-generalized semantic segmentation.

The project studies whether the prediction effect of class-specific semantic queries can remain stable under appearance-only interventions. The first implementation target is a frozen DINOv3 ViT-L/16 backbone with a lightweight segmentation head and a grouped causal-query residual branch.

## Current status

The A0--A5 core sequence and Phase 11--12 ablations are complete. Across paired seeds 0/1/2, combined style reaches `0.640048 ± 0.010760` Cityscapes mIoU and photometric-only reaches `0.636201 ± 0.016427`; the small paired advantage reverses sign at seed 1, so photometric-only is the simpler default. Phase 12 final mIoU is `0.603711/0.643725/0.638287/0.624083` for `R=1/2/3/4`. The shared mechanism analysis selects R=2: it is most accurate, uses both queries, and avoids the approximately `0.999` contextual-query cosine redundancy observed at R=3/R=4. In Phase 13, Static Query exceeds paired one-way R=2 by `0.3671/4.1601` percentage points at seeds 0/1. The direction is consistent but the magnitude is variable, so the prespecified seed-2 pair remains required; one-way seed 2 is next.

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

See `CausalQ_DG_Project_Plan.md` for the complete specification and `docs/AUTODL_NEXT.md` for the next remote action.
