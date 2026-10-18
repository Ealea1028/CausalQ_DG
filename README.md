# CausalQ-DG

## Active phase — Phase25 / conditional formal M3 — 2026-10-18

M2 completed40000 updates/source2497:73.71820470% mIoU,delta vsM1=-0.13595782pp.
No source gain or target claim. Next follows plan section46 before M3:
check RAW anchor vs residual variance on the accepted formal M1 iteration40000
checkpoint,all2497 source holdout IDs,fixed appearance seeds. This has not
been measured at the formal state; the old20-step result was unsupported.
One background job checks the trained hypothesis and automatically starts
fresh seed0 M3/40000 updates ONLY if0<raw anchor variance<raw residual variance.
Otherwise stop adding losses and return the negative result. No extra short
smoke,pilot,figures,weight search or target access. See docs/AUTODL_NEXT.md.

## Archive — Phase24 formal M2 release (completed)

P1 completed40000 updates and2497-source validation: mIoU73.84414155%,
delta vsM1=-0.01002097 percentage points. Returned hash-pinned report and
independent confusion reduction accepted; no source gain or target claim.
Next: M2 centered+original Full CQE control, fresh seed0/full40000 updates,
same two-view CE recipe plus normalized Full CQE,lambda1,beta1. No prediction
KL,Anchor-CQE or other new mechanism. Direct formal launch,1000-step recovery,
final-only full source validation. No repeated smoke/pilot/census.
See docs/AUTODL_NEXT.md. Preserve completed M0/M1/P1 outputs.

## Archive — Phase23 formal P1 release (completed)

M0/M1 seed0 each completed40000 updates and2497-image source validation at
2537653. Source mIoU: M0=73.88616267%,M1=73.85416251%,delta=-0.03200015
percentage points. Returned hashes and independent confusion reduction pass;
no target metric or generalization/decomposition improvement claim.
Next: fresh centered P1,40000 updates, same two-view recipe plus valid-pixel
KL(detached original || photometric),lambda1,T1. No extra smoke/pilot or
new census. One background launch with1000-step recovery and final-only
source evaluation. See docs/AUTODL_NEXT.md. M2/M3 remain subsequent phases.

## Archive — Phase22 formal M0-M1 release (completed)

User authorizes proceeding on assumed M1 continuation readiness; this is not
recorded as a measured CUDA pass. No more standalone readiness/pilot/census
gates. Launch fresh seed0 M0 and centered M1 sequentially,40000 updates each,
identical original+photometric CE recipe, fixed-final selection, final-only
full GTA5 source validation. Immutable1000-update recovery checkpoints;
same command resumes interrupted runs. No new CQE loss or target access.
See docs/AUTODL_NEXT.md for the single formal launch and result handoff.

## Archive — Phase21 readiness superseded by user authorization

Priority3 is closed with independent raw CSV/actual-source figure review.
Raw anchor stability is unsupported at the20-step engineering state; no new
Anchor-CQE. Prepare the no-CQE M0/M1 formal baseline comparison by checking
M1 schema2 source-runner continuation in3 fresh processes (8 vs4+4), exactly16
engineering updates, no mIoU/formal training/target access. Same model/recipe,
no new mechanism. See docs/AUTODL_NEXT.md; do not repeat M0 or source census.

## Archive — Phase20 / Priority3

Priority3 complete: returned64 files byte-verified, raw CSVs independently
recomputed, all20 fixed samples visually reviewed. No further transfer or
inference. Raw anchor stability is unsupported at the20-step checkpoint;
no new Anchor-CQE. Next local scope is M0/M1 baseline readiness. Pending
transfer statements below are historical.

Latest: full2497 census report/checksum verified;0/2497 images have lower
raw anchor variance (per-image19-class macro). Analysis archive/actual source
figures are not returned yet. Current action transfers the existing45MiB
ZIP for local raw/visual review, with no new GPU run or loss. Earlier launch
descriptions below are archived. Priority3 is not closed yet.

Current action after accepted gate20 at8500976: full2497 source-holdout census
and systematic20-image/five-class visual audit,4994 forwards and zero updates.
Same20-step engineering checkpoint; no new loss or target access. Raw and
RMS-normalized variance remain separate; no anchor-stability/DG claim from
the bounded gate. See docs/AUTODL_NEXT.md for exact full-census handoff.
Earlier gate20 launch descriptions below are historical.

Gate20 complete report/checksum now accepted; do not repeat it. Next remains
Priority3 full-source census and systematic visual audit, same early M1
checkpoint and zero updates. No raw anchor-stability claim from the sample.

Phase19 component engineering is accepted at 01f59e9 from complete pinned
report/checksum evidence. Current work is read-only responsibility/effect
diagnostics: fixed20 source holdout IDs x original/photo views, same accepted
M1 checkpoint, zero optimizer updates. No model/objective changes or target
access. See docs/QUERY_COMPONENT_DIAGNOSTICS.md and docs/AUTODL_NEXT.md.
Full-holdout scientific analysis and formal ablations remain later work.

## Archive — Phase19 / Priority2

Complete returned report/checksum bytes are now accepted with all16 rows,
exact source CE replay and unchanged model/RNG/files. Priority2 is complete;
no more inference/export is needed. Earlier pending statements are archived.

Latest f5ca5b5 terminal tails report exit0, CE error0 and unchanged model/RNG/
files. Full report/checksum acceptance remains pending. Current action in
AUTODL_NEXT.md exports existing evidence only; do not repeat inference.
The next-launch wording below is historical until that audit completes.

The `CausalQ_DG_1016.txt` Priority0 source-development engineering is accepted.
The complete 500-update pilot report/checksum is pinned and audited; source
mIoU 35.7242% remains a diagnostic, not formal M0. No repeats of that pilot.

Priority1 centered-query engineering is accepted from the complete 9583925
report: algebra errors below 8e-9, zero gradient-sum/alpha-zero/prediction
roundtrip error and peak reserved 1.417969 GiB. No M1 mIoU or formal result.
Priority2 now adds opt-in anchor-only components to the same model/head,
sharing one backbone pass per view. Full/anchor effects retain the pre-addition
logit contract; residual contribution is their FP32 difference, not an
independent residual causal effect. Default forward and legacy APIs stay fixed.
Next AutoDL gate loads the SAME immutable M1 checkpoint and checks eight
source images under original/photometric views, with zero optimizer updates.
No new loss, formal training, target access or effect-stability claim. Final
target evaluation remains blocked until architecture and weights freeze.
Old launchers read Cityscapes during development
and must not be used for new M0–M4/P1 experiments. Static R=2's 65.2849% remains
a historical reference. Phase 16 remains closed.
The historical status and working-directory narrative below is retained for
provenance; the active action is exclusively [`docs/AUTODL_NEXT.md`](docs/AUTODL_NEXT.md).

New contributors should begin with the standalone Chinese project guide:
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It separates the implemented
method and verified results from the proposed next-generation theory roadmap.

Counterfactual Query Effect Distillation for domain-generalized semantic segmentation.

The project studies whether the prediction effect of class-specific semantic queries can remain stable under appearance-only interventions. The first implementation target is a frozen DINOv3 ViT-L/16 backbone with a lightweight segmentation head and a grouped causal-query residual branch.

## Current status

The A0--A5 core sequence and Phase 11--16 studies are complete. Static R=2 remains the supported DINOv3 model (`0.652849 ± 0.015138` final Cityscapes mIoU over three seeds). Phase 14's learned-null variant improves seed-0 segmentation but worsens effect localization, so it remains a negative ablation. The §41 GTA5 → Cityscapes figures are qualitative diagnostics only. The completed §42 backbone comparison finds ViT-L above ViT-B in mIoU (`0.652849` versus `0.563878` mean) and below it in normalized cross-style effect variance (`0.000567443` versus `0.000961160` mean) for all three paired seeds. Phase 16's REIN transfer gives a three-seed matched CQE gain of only `+0.0986 ± 0.1538` percentage points; its training-seed 95% interval crosses zero and mean CQE mIoU remains below the unmodified REIN reference. This is weak directional evidence, not a statistically resolved or practically meaningful CQE improvement. Other target datasets remain deferred. See `experiments/REIN_PHASE16_RUNTIME/report.md` and `docs/AUTODL_NEXT.md`.

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
uses module entry points with an import preflight; the paired protocol remained
unchanged and the repaired launch described below superseded that failure.
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
seed-2 matched pair therefore completed the final planned training-seed
replicate used to decide the CQE performance claim.
Seed-2 passes at `e00d5f4` with another positive but negligible `+0.0035`
point delta and an image-bootstrap interval crossing zero. Across all three
seeds CQE gains `+0.0986 +/- 0.1538` points, with a training-seed t interval
that crosses zero; all CQE variants remain below the original REIN reference.
The hash-pinned aggregate passes at `2e096bd`, records the bounded conclusion,
and sets `no_additional_long_training_authorized: true`. Phase 16 is closed;
the current plan contains no further authorized AutoDL experiment.
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
