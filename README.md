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
checkpoint continuation including RNG and source sampler before the formal baseline.
The a10c1f6 continuation attempt failed optimizer-state comparison; checkpoint
loading now preserves its input dictionaries and checks full restoration before replay.
The corrected same-process gate passes at 5fee730 (parameter error 1.28e-7).
Independent-process restore replay passes at 1671f1a (parameter error 1.79e-7).
The source runner integration and saved-evidence audit pass. The first formal
attempt was correctly excluded after a project-side PolyLR guard error; the
guard now matches MMEngine's 39,999-step internal horizon. The corrected
`d6fc52c` seed-0 run and its saved-evidence audit at `1da98b5` are accepted:
40,000 optimizer updates, 160,000 microbatches and fixed-final Cityscapes mIoU
`0.656051`. This is the REIN source-only reference, not a CQE result. The
class-specific residual branch over frozen REIN semantic logits passes its
80-microbatch / 20-update no-CQE smoke at `14d3e5a`: finite optimization,
nonzero branch gradients, exact class isolation and unchanged base parameters.
Native Mask2Former queries are not relabelled. The smoke's saved files pass an
independent read-only audit at `161f077`. Next is a matched 20-update smoke
that compares the same branch and photometric views with and without CQE.
Its first attempt stopped before updates on compact-checkpoint coverage; the
retry activates and verifies REIN's trainability hook before restoration.
The matched no-CQE/CQE 20-update smoke completed at `3644785`; its saved-file
first audit at `fa9c492` rejected objective reconstruction because its Python
float64 sum did not reproduce the producer's float32 tensor addition. The
audit now models float32 rounding and has a focused regression test. The v2
audit passes at `f18c2c2`; it verifies evidence integrity but provides no
target accuracy result. The user-approved formal seed-0 pair was implemented;
its first launch at `8c1396c` failed before initialization because its file-path
entry point could not import `tools`. No training occurred. The launcher now
uses module entry points with an import preflight; the paired protocol is
unchanged and awaits a fresh exact-commit AutoDL launch.
That launch completed at `8640215`, with a preliminary CQE-minus-control gain
of `+0.2760` mIoU points. Its first saved audit exposed an auditor-only
accumulation-boundary indexing error; seed-0 training must not be repeated.
The immediate next gate is a corrected read-only audit of the immutable run.
That corrected audit passes at `96b193d`: the seed-0 CQE gain is `+0.2760`
points and its paired image-bootstrap 95% interval is `[+0.2119, +0.3437]`
points. This is conditional on one training seed and remains below the original
REIN reference. Seed-1 then passes its full audit at `682d2b7`: CQE is again
positive versus its matched control, but only by `+0.0164` points and its
conditional image bootstrap interval crosses zero. Across seeds 0--1 the paired gain is
`0.1462 +/- 0.1836` points (mean +/- sample standard deviation). A fixed
seed-2 matched pair is therefore the next and final planned training-seed
replicate before deciding the CQE performance claim.
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
