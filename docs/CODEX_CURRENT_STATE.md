# CausalQ_DG_1016 repository audit

Latest Priority 0 update (2026-10-16): the operator returned an exit-0 inventory
summary for 24,966 complete flat-numeric pairs with no metadata candidates.
The original audit below remains a snapshot of `0549881`. Current work adds
`causalq/protocols/source_split.py`, `tools/build_gta5_source_split.py`,
`scripts/build_gta5_source_split_phase17.sh` and `tests/test_source_split.py`.
The existing model, data adapter and trainer remain unchanged. The new gate
verifies pinned inventory bytes then exports the deterministic 10% filename
fallback as candidate manifests for return to local Git; nothing is frozen yet.
See `STRICT_DG_PROTOCOL.md`, `EXPERIMENT_RESULTS.md` and `AUTODL_NEXT.md`.

Audit date: 2026-10-16. This is Phase 17 / Priority 0 only: strict-DG protocol
preparation. No centered Query change, new loss, training or target evaluation.

## Repository snapshot before changes

- HEAD: `0549881480828bada4fd84b8613d72faaadecd04` (`docs: close phase 16 CQE study`).
- Original branch: `codex/close-phase16`; work branch: `codex/strict-dg-source-audit`.
- `git status --short`: no tracked modifications. Pre-existing untracked items:
  `.test-tmp-phase11-handoff-fix/`, `CausalQ_DG_1016.txt`,
  `CausalQ_DG_Modification_Report.md`, the Chinese experiment report DOCX,
  `analysis/{car,person,road,vegetation}.png`, `artifacts/`,
  `tools/build_causalq_analysis_report.py`. Preserve these user assets. Only the
  explicitly requested 1016 research plan is added as a versioned specification.
- Inspected status, HEAD, last 20 commits, all branches and tracked-file inventory;
  read README, AGENTS, original plan, PROJECT_GUIDE, METHOD, AUTODL_NEXT and
  `CausalQ_DG_1016.txt`, then followed actual model/data/trainer/config code.

## Implementation and evidence

| Item | Actual HEAD behavior / source |
|---|---|
| DINOv3 | `causalq/models/dinov3_wrapper.py`; frozen ViT-L/16, D=1024, layers 6/12/18/24; HF local weights with architecture validation |
| Weight identity | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |
| Selected config | `configs/query_interaction/gta_dinov3l_static.yaml`, Phase 13 I1 |
| Query Bank | `GroupedCausalQueryBank` in `causalq/models/query_segmentor.py`; q[c,r]=a[c]+e[c,r], residuals NOT centered |
| Query count / interaction | R=2, 19 classes, Static, no cross-attention layers; constructor default R=3/one_way is not selected mainline |
| Residual Head | normalized projected pixel/query similarity / temperature 0.07, logsumexp over R, bilinear upsampling |
| Alpha | learned scalar, initialized 0; Z=Z_base+alpha*Delta; signed, not constrained positive |
| Effect | `get_query_effect` returns alpha*Delta; `counterfactual_logits(c)` assigns exactly base[c], leaves other channels unchanged |
| Components | no anchor-only effect or residual-contribution API yet; `get_query_residuals` returns raw parameters |
| Style | `causalq/interventions/style.py`; selected trainer calls only `photometric_view`; original plus aligned photometric, total lambda_cf=1 |
| Photometric settings | brightness/contrast/saturation 0.7–1.3; gamma 0.8–1.2; temperature 0.9–1.1; grayscale probability 0.1; ImageNet normalization |
| CQE | `causalq/losses/causal_query_effect.py`; valid-pixel L2 normalization per class map, stopped original reference, SmoothL1 spatial sum averaged over per-image GT-present classes; disabled for Static baseline |
| Diversity | `causalq/losses/query_diversity.py`; within-class squared off-diagonal residual cosine; historical negative ablation, disabled in selected baseline |
| Learned-null | optional Static-only bank initialized after factual path; negative localization ablation; remains disabled |
| Training | `tools/train.py`, historical Phase 5–15 only; `scripts/train_static_query.sh` invokes it |
| Schedule | 40k optimizer iterations, 512x512 crop, batch1/accum1, BF16 AMP, AdamW 1e-4, wd0.01, poly0.9, clip norm1; scale0.5–2.0, flip0.5, valid crop fallback |
| Target access | train.py constructs Cityscapes val dataset/loaders before weights; validates every 500 steps AND final; saves checkpoints at validation; no strict source split or target access guard |
| Source data | `gta5_dataset` in `causalq/datasets/segmentation.py` pairs all images and converted labels by relative stem; optional redundant archive directory unwrapped; all pairs train |
| Inspection | `tools/check_datasets.py` defaults to BOTH GTA5 and Cityscapes; unsuitable as the new default development gate |
| Outputs | CAUSALQ_OUTPUT_ROOT/run_id: config.yaml, metadata.json, train.jsonl, checkpoints/iter_*.pth, summary.json; compact checkpoints exclude frozen backbone |
| Analysis | tools/compare_effect_variance.py, analyze_query_count.py, analyze_zero_effect.py, analyze_learned_null_effect.py, visualize_query_effect.py; several default to target data and need later source-only routing |
| Tests | CPU tests for Query Bank/head/interventions/effects/losses/checkpoints/datasets/configs; REIN tests use synthetic fixtures, remote runtime tests guarded; no strict-DG split tests at audited HEAD |
| Evidence | experiments/registry.csv and small per-run summaries/reports in Git; full runs/checkpoints/data remain on AutoDL |
| Phase 16 | closed; three REIN CQE deltas +0.2760/+0.0164/+0.0035 pp; mean +0.0986 +/-0.1538 pp, training-seed interval crosses zero; no new REIN run |

Historical A0–A5 remain unchanged: 60.17, 61.95, 63.96, 62.12, 59.18,
59.21 percent. A4 versus A3 loses 2.9474 pp while normalized cross-style
effect variance drops 99.4714%. These diagnostics used Cityscapes.
Static R=2's three-seed **historical reference** is 65.2849% +/-1.5138 pp.
It is not a strict-DG M0 result. No M0–M4/P1 result exists yet.

## CODE/DOC CONFLICT

1. PROJECT_GUIDE model sections and METHOD's first-implementation target describe
   R=3/one_way; actual selected config and evidence use R=2/Static.
2. PROJECT_GUIDE's map names `causalq_dg/` and `style/`; real package is
   `causalq/`, interventions are in `causalq/interventions/`.
3. Old “target not optimized” language permits recurring target validation;
   the 1016 plan forbids all development-time target access. Old experiments
   remain historical; existing trainer must not launch new M experiments.
4. Previous AUTODL_NEXT says no next phase authorized. The user's 1016 request
   now authorizes a new isolated Priority 0 source inventory; it does not reopen
   Phase 16 or authorize a long run before protocol gates.
5. Data sequence/scene availability cannot be inferred from repository code or
   old flat-file examples. Actual AutoDL inventory is required before splitting.

## Next Modification Impact Map

- Current change: add a GTA5-only filesystem inventory and focused tests;
  version this audit, the 1016 plan, strict-DG protocol boundary and AutoDL handoff.
- Next, AFTER inventory review: freeze source train/val membership by sequence,
  otherwise scene, otherwise validated filename/group. Configurable deterministic
  10% fallback only if no useful grouping exists. Commit exact ID manifests and
  hashes; seeds change sampling/initialization only.
- Later strict trainer/evaluator: source holdout during development; model and
  hyperparameters frozen before fixed Cityscapes-500 final evaluation. Record
  target access and freeze evidence. New configs use M0/M1/P1/M2/M3/M4 IDs.
- Reuse backbone, decoder, Query/head APIs, geometry, photo bank, CE, metrics,
  compact checkpoint utilities and focused CPU test fixtures.
- Do not change old configs, A0–A5 labels/results, accepted checkpoints, REIN
  code/evidence, or Query parameterization in this gate.
- Centered parameterization is a later isolated change:
  q[c,r]=a[c]+e[c,r]-mean_r(e[c,r]); anchor and residual effects remain hypotheses.

## Verification and stopping boundary

The new inventory reads filenames/sizes and bounded source-side metadata only;
it does not decode images, validate label values, load weights, import torch,
select a split, train, or touch target data. Its success means filesystem
pairing is usable for split review, not dataset/model/accuracy acceptance.
Stop for AutoDL GTA5 inventory before continuing the dependent split decision.
Local test results and the executable handoff are recorded in AUTODL_NEXT.
