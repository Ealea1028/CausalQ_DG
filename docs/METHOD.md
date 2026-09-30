# Method contract

## Phase 16 boundary

REIN transfer remains exploratory after CQE's failed accuracy acceptance.
The isolated runtime passes package pins, CUDA NMS and xformers attention.
The upstream backbone gate passes complete 343-tensor weight coverage,
feature/query shapes and adapter-only finite backward gradients at `ed3dcfe`.
The synthetic full-segmentor gate also passes at `86fb3ff`: all 30 Mask2Former
losses and head/adapter/pixel-decoder gradients are finite in float32, with
3.662 GiB peak reserved memory. Next is a bounded 20-step source-data optimizer
diagnostic using existing project GTA5 geometry/crop logic and an audited
normalization bridge. Five target center-crop predictions are compatibility
checks only; Cityscapes labels are never optimized and no mIoU is reported.
There is no CQE, checkpoint or formal training in this smoke.
The operator's 20-step trace at `17883de` is finite and exits 0; full saved-report
audit passes at `b85f105`, with normalization error `4.768e-7` and peak reserved
memory `3.881 GiB`. High norms are pre-clipping. The real-data engineering gate
is accepted, not accuracy. Next inspect pinned upstream effective data/optimizer/
schedule/evaluation settings before versioning a project source-only protocol.
Smoke settings are not exact upstream reproduction; train-ID conversion and
full Cityscapes-only evaluation must be explicit before formal training.
An accepted real-data smoke and a source-only REIN baseline are prerequisites
for any later transfer claim. The main implementation remains DINOv3 Static R=2.

The upstream protocol inventory exits 0 at `ede170c`. The adapted baseline keeps
the upstream model, photometric distortion and paramwise 40k PolyLR protocol,
with physical batch 1/accumulation 4, clip 1, workers 0, seed 0 and one retained
checkpoint. GTA5 converted `.png` train IDs must not be remapped; scale-only
alignment and valid-pixel crop fallback are declared safety deviations. Both
target dataset and metric are narrowed to Cityscapes; full original GT is kept
after input resize, unlike the earlier cropped inference smoke. Adapted data
pipelines require AutoDL acceptance before any scheduled GPU run. This is not
an exact paper reproduction or evidence of successful CQE transfer.

The adapted data gate is accepted at `b645793`. Initial scheduled accumulation
attempts `d5558e1` and `9ac1f80` failed on configuration
types lost in JSON, before source optimization. Runtime configuration must be
rebuilt from the pinned typed upstream source through the unchanged adapter,
then checked for complete JSON-value equality with the accepted report before
model/dataset construction. Saved JSON remains provenance evidence, not a
lossless executable configuration.

The corrected scheduled gate passes at `3ca6731`: 80 source microbatches and
20 accumulated optimizer updates, PolyLR checked per update, one 283,250,458-byte
compact checkpoint and exact prediction roundtrip. Peak reserved memory is
3.920 GiB. This is engineering acceptance only. Next verify standard sliding
inference/postprocessing against original Cityscapes GT and independent versus
official IoUMetric counts on five images. No target optimization or accuracy
acceptance is permitted by that short evaluation. Formal REIN training remains
deferred.

The five-image slide gate exits 0 at `59d522b`. The reported zero/low IoUs
after only 20 updates are not a successful segmentation baseline; absent-class
metric NaNs must be distinguished from non-finite logits. Before proceeding,
the saved report is hash-pinned and checked for complete original geometry.
The next bounded pilot starts fresh at seed 0, uses 2,000 source microbatches
and 500 updates with the SAME 40k-update PolyLR horizon, retains one compact
checkpoint, and diagnoses 50 original-resolution target images. Target labels
are never optimized. It is neither an exact-resume run nor formal 40k training,
and has no accuracy threshold, target checkpoint selection, or CQE mechanism.

The pilot operator reports exit 0 at `b4e292d`, with road 95.15%, building
76.04%, vegetation 81.15% and car 51.88% IoU on the 50-image diagnostic.
Many minority classes remain zero. This supports optimization feasibility,
not formal accuracy or comparison with the 500-image DINOv3 experiments.
The full saved report, trace and compact checkpoint need a read-only audit
before formal training. Disk free is 9.3 GiB; inventory cache/mount usage first.
Never delete datasets, current environments, final checkpoints or evidence.

The saved pilot audit passes at `3601ecd`: 2,000 contiguous source records,
500 updates, 50 original-resolution targets, diagnostic mIoU `0.235189`,
first/last-20 loss means `122.3003/57.5609`, checkpoint roundtrip zero and
4.217 GiB peak reserved memory. The checkpoint SHA is
`b9747a71b1871e201a7730d7b92ca1daf836c7fdc0bcf61b06bd2c9b40fbfda0`.
Cache savings are negligible and on the system overlay, not the data volume.
The next engineering gate stores source permutation/cursor/generator and
Python/NumPy/CPU/CUDA RNG with model/optimizer/scheduler state. It executes
20 source updates then replays update 21 twice from the same disk checkpoint,
requiring matching source indices, losses, parameters and optimizer state.
This is SAME-process continuation verification, not yet fresh-process resume,
bitwise CUDA determinism, target accuracy or formal 40k authorization.

The `a10c1f6` attempt FAILED at optimizer scalar comparison after 20 finite
updates and zero prediction-roundtrip error. The roundtrip loader passed the
checkpoint optimizer dictionary directly to MMEngine BaseOptimWrapper, which
pops `base_param_settings` from its input. Subsequent replay therefore lacked
the saved base LR and retained the previous branch's value. The loader now
clones optimizer/scheduler trees for every restore, rejects missing base
settings, and checks complete restored state before executing either branch.
Scalar equality and GPU tolerances are unchanged; mismatch errors include the
nested path. The corrected bounded gate PASSES at `5fee730`: loss difference
zero, parameter maximum difference `1.2759119272232056e-7`, optimizer maximum
difference `1.3969838619232178e-8`. Next two independent processes use distinct
construction seeds before restoring saved RNG/state and replaying update 21.
Snapshots include post-update RNG. This is fresh-process restore replay only,
not uninterrupted-versus-resumed equivalence or formal accuracy. No target
optimization, CQE or 40k training is introduced.

The independent-process replay PASSES at `1671f1a`: loss difference zero,
parameter maximum difference `1.7881393432617188e-7`, optimizer maximum
difference `1.0710209608078003e-8`; saved sampler and post-update RNG agree.
Next exercise a fresh-source runner with 20 updates, epoch-aware sampling and
atomic rolling saves every five updates. It keeps latest/previous only in a
new run, validates complete checkpoint serialization, then diagnoses five
original-resolution Cityscapes images. Prior artifacts remain untouched.
This is the long-run integration gate, not another accuracy experiment; formal
40k and full 500-image final evaluation remain pending its acceptance.

The runner operator returns exit 0 at `7a3a1c6` and two 283,336,151-byte files;
the saved-evidence audit was passed before the formal attempt. One fresh
source-only REIN seed-0 baseline was then permitted: 40k optimizer
updates, 160k microbatches, accumulation4, unchanged typed source protocol and
FP32. Latest/previous rolling checkpoints are saved every 1k updates; source
sampler spans epochs. Final checkpoint alone receives full 500-image original-GT
Cityscapes evaluation; no target-based checkpoint selection or CQE. Compared
with DINOv3, backbone/head/augmentation/evaluation and training exposure differ,
so a metric gap cannot isolate an adapter or CQE effect. Formal results and
arbitrary interruption equivalence remain unverified until real evidence returns.

The first formal attempt at `ef91451` stopped at the LR guard after update 1,760.
Its logged LR (`9.603104465931869e-05`) agrees with MMEngine 0.10.7 PolyLR's
`total_iters=end-begin-1=39,999`; the project-side expected schedule had divided
by 40,000. This is an implementation-check failure, not evidence of model
instability or accuracy. Preserve that partial run as failed evidence and
rerun from fresh seed-0 initialization only after the corrected schedule gate
is committed and deployed. No resume is authorized.

The corrected `d6fc52c` seed-0 run passes the read-only saved-evidence audit at
`1da98b5`: 40,000 optimizer updates / 160,000 microbatches, six complete source
epochs plus 10,204 samples, 40 checkpoint-history records, fixed-final
Cityscapes mIoU `0.656051`, and no target-label optimization. This establishes
the adapted REIN source-only reference; it does not establish CQE transfer.

The next single-mechanism step adds an explicit class-specific residual branch
over the fixed REIN segmentor's 19 semantic logits. It uses a
`[class, query, channel]` bank with two queries per class, normalized similarity,
log-sum-exp aggregation, and a scalar residual scale initialized to zero. Native
Mask2Former prediction queries remain unchanged and are not assigned semantic
class identities. The intervention `do(Q_c=0)` restores only output channel
`c` to its fixed base logit. A bounded 20-update source smoke must demonstrate
finite optimization, delayed gradient flow through the zero-initialized scale,
class isolation, compact branch-only checkpoint roundtrip, and unchanged base
parameters before any matched long run or CQE loss is considered.

## Objective

CausalQ-DG measures and distils the prediction contribution of class-specific semantic queries across appearance-only counterfactual views.

## First implementation target

- Frozen DINOv3 ViT-L/16 image backbone.
- One-way query-to-image cross-attention; image features are not modified by queries.
- Nineteen Cityscapes classes and three queries per class.
- A base segmentation path plus a query residual path.
- A learnable residual scale initialized to zero.

## Interventions

The original image, a photometric view, and a Fourier-amplitude view must share the same geometry and label. The first implementation does not use CycleGAN or geometric warping.

## Causal effect

For class `c`, intervening with `do(Q_c = 0)` removes only the class-query residual logit. It does not rerun or alter the backbone. The initial logit-level effect is the scaled class-query residual.

## Training order

Establish a credible source-only baseline before adding queries. Then add style augmentation, prediction consistency, and CQE in separate phases so every contribution has a controlled comparison.

## Prediction-consistency control

Phase 8 keeps the accepted Query + Style setup fixed. For each photometric or
Fourier view, it minimizes `KL(P_original || P_style)` on valid label pixels.
The original-view probabilities are detached, the two style losses are averaged,
and no causal-query-effect or diversity term is enabled.

## Causal-query-effect distillation

Phase 9 defines the public logit effect as `alpha * delta_logits`. The original
view is a stop-gradient reference. Each effect map is L2-normalized over valid
pixels, and SmoothL1 distances are averaged across counterfactual views and only
the semantic classes present in the current ground-truth mask. Prediction
consistency and diversity remain disabled in this phase.

## Success criterion

The decisive comparison is CQE against prediction consistency under the same query and style setup, accompanied by reduced cross-style query-effect variance.

Cross-style query-effect variance is measured on the same original,
photometric, and Fourier views for both checkpoints. Each valid-pixel class
effect map is L2-normalized over space, population variance is computed across
the three views and summed spatially, and results are averaged only over
ground-truth-present class maps. The evaluation uses a fixed per-sample seed so
the compared checkpoints receive identical interventions.

The fixed Phase 9 result passes the variance target but fails the segmentation
target: A4 reduces normalized effect variance by 99.47% relative to A3 while
losing 2.95 Cityscapes mIoU percentage points. Both outcomes are retained; the
variance result does not override the failed accuracy criterion.

## Scaling control

The project-plan §42 scaling check starts with an isolated DINOv3-B/16
comparison against the selected DINOv3-L/16 Static R=2 model. The GTA5 source,
Cityscapes-val target, frozen-backbone training, two static queries per class,
photometric-only view, segmentation losses, crop, optimizer, schedule, and seed
remain fixed. Only the pretrained backbone, its four intermediate layer
indices, and resulting model dimensions change. Prediction consistency, CQE,
diversity, and learned-null stay disabled because those mechanisms failed their
earlier acceptance gates. A 500-step, 50-image GPU smoke is diagnostic only;
full 40k results and matched style-effect analysis are required before making
any scale claim. No other target dataset is included while its evaluation is
deferred by the user.

The ViT-B seed-0 40k trace, 80 full Cityscapes validations, and 80 checkpoints
passed audit. Its final mIoU is `0.563382` versus `0.647396` for the matched
ViT-L Static R=2 seed-0 run (ViT-B minus ViT-L: `-8.4014` percentage points).
On the same 500 validation images and deterministic original/photometric
views, normalized query-effect variance is `0.001145` for ViT-B and `0.000581`
for ViT-L; the ViT-L value is 49.2% lower. The matching class-map coverage is
6,005 per backbone. This supports the scale hypothesis for this seed pair but
does not establish a multi-seed scaling law. The audited ViT-B seed-1 run
reaches `0.555294` mIoU versus `0.669958` for ViT-L seed-1 (a gap of
`11.4664` percentage points). The seed-1 matched style-effect variance is
`0.000999` for ViT-B and `0.000620` for ViT-L, with 6,005 present-class maps
each; ViT-L is 37.9% lower. Both seed pairs point in the same direction.
The ViT-B seed-2 run passed its 40k audit and the matched 500-image effect
evaluation. Its final mIoU is `0.572957` versus `0.641193` for ViT-L;
normalized effect variance is `0.000740077` versus `0.000501017` (32.30%
lower for ViT-L). Across three seeds, ViT-B and ViT-L respectively reach
`0.563878 ± 0.008842` and `0.652849 ± 0.015138` final mIoU; the paired
ViT-L advantage is `8.8971 ± 2.3608` percentage points. Their normalized
effect variances are `0.000961160 ± 0.000204922` and
`0.000567443 ± 0.000060839`, with ViT-L lower for all three seeds. This is
evidence for a consistent two-size association in this protocol, not a
scaling law or causal proof. The original scaling question about CQE gain
remains unanswered because CQE failed its earlier segmentation acceptance
gate and was not added to the selected Static model. See the versioned
three-seed report in `experiments/BACKBONE_SCALING_STATIC_R2_3SEED/report.md`.

## Query diversity

Phase 10 composes the already implemented prediction-consistency and CQE
objectives, then adds one new mechanism: a lightweight diversity penalty on the
learned residual queries. Within each semantic class, residual queries are
L2-normalized and the squared off-diagonal cosine similarities are averaged.
The fixed weight is `lambda_div=0.01`. The class anchors and contextualized
image-conditioned query states are not regularized by this term.

The fixed Phase 10 run is numerically stable and drives the diversity loss from
`1.11e-3` to `1.07e-7`, but reaches only `0.592124` Cityscapes mIoU. Its gain
over A4 is 0.037 percentage points, below the predefined 0.2-point retention
threshold. Query diversity is therefore excluded from the final method.

## Style ablation

Phase 11 returns to the accepted A2 Query + Style objective and disables every
consistency loss. Photometric-only and Fourier-only variants are compared with
the existing combined-view A2 result. The total counterfactual supervision
weight remains `lambda_cf=1`: a single intervention receives weight 1, while
the combined run assigns weight 0.5 to each of its two interventions.

Both isolated 500-iteration smoke tests passed at commit `b0d6e14`: their
records were contiguous and finite, their objectives reconstructed exactly,
and their metadata contained no prediction-consistency, CQE, or diversity
configuration. Full-run conclusions are intentionally deferred until both
40,000-iteration variants have been evaluated on all 500 Cityscapes validation
images.

The fixed photometric-only run reaches `0.638287` final Cityscapes mIoU,
0.1324 percentage points below the combined-view A2 reference. Its best
intermediate validation is `0.641253` at iteration 30,000, but the prespecified
final-iteration result remains the primary comparison. Fourier-only reaches
`0.627076`, 1.2534 points below combined and 1.1211 below photometric-only.
The Fourier result is decisive at seed 0. The combined-versus-photometric gap
is below the project's 0.5-point repeat threshold, so those two variants must
be repeated with seeds 1 and 2 before drawing the final style conclusion.

The combined-view seed-1 repeat is numerically stable and reaches `0.651021`
final Cityscapes mIoU. Its learned residual scale is negative, which is valid
for the unconstrained alpha parameter and does not indicate instability. The
paired photometric-only seed-1 result is required before any cross-seed style
comparison is made.

Photometric-only seed 1 reaches `0.651486`, exceeding the paired combined-view
result by 0.0465 percentage points. This reverses the seed-0 ordering, where
combined led by 0.1324 points. The seed-2 pair is therefore necessary for the
prespecified three-seed mean and standard deviation.

Combined-view seed 2 reaches `0.629514` final Cityscapes mIoU. Its last logged
gradient norm is a finite pre-clipping value; the fixed max-norm `1.0` clip is
applied before the optimizer step. The run otherwise satisfies every stability
and provenance gate. Photometric-only seed 2 is the remaining paired run.

The completed paired comparison gives combined `0.640048 ± 0.010760` and
photometric-only `0.636201 ± 0.016427` final Cityscapes mIoU. The paired
combined-minus-photometric difference is `0.003847 ± 0.005987`, changes sign at
seed 1, and is too unstable to support a superiority claim. Photometric-only is
selected as the lower-cost default; combined and Fourier remain ablations.

## Query-count ablation

Phase 12 fixes the selected photometric-only style protocol and seed 0 while
varying only the number of residual queries per class: `R=1,2,4`. The existing
photometric-only `R=3` seed-0 run is the reference. Initial GPU work is limited
to 500-iteration smoke tests before any 40k run.

All three isolated smoke tests passed at commit `be450a1`. Each produced 500
contiguous finite records, exact objective reconstruction, one 50-image
validation, and one checkpoint, while preserving the intended query count and
excluding every auxiliary consistency loss. Peak reserved memory remained
between `2.434` and `2.438 GiB`. Their short-run mIoU values are diagnostic only
and are not used for model selection. Full runs proceed one query count at a
time, beginning with `R=1`.

The R=1 full run is stable but reaches only `0.603711` final Cityscapes mIoU,
compared with `0.638287` for the fixed photometric-only R=3 reference. The
3.4576-point deficit occurs despite 40,000 finite contiguous records, exact
objective reconstruction, and the intended isolated protocol, so R=1 is
retained as a negative capacity ablation. Phase 12 continues with R=2.

The R=2 full run reaches `0.643725` final Cityscapes mIoU, improving R=1 by
4.0014 percentage points and the fixed R=3 reference by 0.5439 points. Its
40,000-record trace, objective reconstruction, isolation, and 80 full
validations all pass. R=2 is the current best candidate, but selection remains
open until the R=4 result and query-behavior diagnostics are available.

The R=4 full run is also stable and reaches `0.624083`, 1.9642 percentage
points below R=2 and 1.4203 points below R=3. Final mIoU across `R=1/2/3/4`
is `0.603711/0.643725/0.638287/0.624083`; R=2 is the current leader.

The closing diagnostic uses the same 500 Cityscapes validation images and the
deterministic original/photometric view pair for every checkpoint. It reports:

- unordered within-class cosine similarity for both learned residual queries
  and contextualized queries (the latter only for ground-truth-present
  classes);
- per-query logsumexp responsibility on downsampled ground-truth-class pixels,
  summarized by entropy-based effective query count and fraction, dominant
  mean share, and mean per-pixel top-1 responsibility;
- population variance across the two views after valid-pixel L2 normalization
  of each ground-truth-present class effect map.

The common 500-image analysis selects R=2. It reaches the highest final mIoU
(`0.643725`), uses `1.91/2` effective queries, and has residual/contextual mean
cosine `0.320/0.931`. R=3 and R=4 have contextual cosine near `0.999`; their
near-uniform responsibilities therefore describe duplicated states rather than
clear specialization. R=3 has 14.5% lower cross-style effect variance than
R=2 (`7.88e-4` versus `9.03e-4`) but lower accuracy and higher query cost.
R=1 and R=4 are worse in both accuracy and effect variance than R=2. Phase 12
is complete and fixes `queries_per_class=2` for subsequent experiments.

## Query-interaction ablation

Phase 13 starts the interaction comparison from one isolated control: static
queries. It keeps the selected R=2 photometric-only protocol fixed but removes
Query-to-Image cross-attention. The grouped class anchors and residual queries
still form dense logits through the same normalized pixel/query similarity
head; only the image-conditioned query update is absent. No prediction
consistency, CQE, diversity, learned-null, or bidirectional interaction is
enabled. The existing Phase 12 R=2 result is the one-way reference.

The isolated 500-iteration Static Query smoke passes at commit `367694e`:
500 contiguous finite records, exact objective reconstruction, the intended
zero-layer interaction metadata, one 50-image validation, and one checkpoint.
Peak reserved memory is `2.258 GiB`. Its short-run mIoU is diagnostic only;
the seed-0 40k run is required for the interaction comparison.

The seed-0 Static Query full run reaches `0.647396` final Cityscapes mIoU,
0.3671 percentage points above the existing one-way R=2 reference. It has
40,000 contiguous finite records, 80 full 500-image validations, exact
objective reconstruction, and `2.258 GiB` peak reserved memory. Because the
gap is below the plan's 0.5-point repeat threshold, no interaction winner is
declared. One-way and static must be repeated as paired seeds 1 and 2 before
introducing the bidirectional control.

The first one-way seed-1 attempt at commit `367694e` stopped after 3,316
iterations because all ten random crop candidates for sample `13286` contained
only ignore pixels. A subsequent audit of all 24,966 GTA5 labels found no
all-ignore or unreadable label and a minimum full-image valid fraction of
`0.134332`, confirming a crop-sampling edge case rather than corrupt data.
Training preprocessing therefore keeps the existing random attempts but, only
when all attempts are empty, samples a real valid label pixel and constructs a
crop guaranteed to contain it. The failed run is excluded and must restart
from random initialization after GPU verification of the fix.

The repaired one-way seed-1 path passes a fresh 500-iteration GPU smoke at
commit `3860e69`: it reaches the final iteration, writes one checkpoint, and
completes a finite 50-image validation at `0.341777` mIoU. This short-run score
is diagnostic only. The original 3,316-iteration attempt remains excluded;
the formal seed-1 repeat restarts from random initialization at the repaired
commit.

The repaired one-way R=2 seed-1 full run completes at `0.628357` final
Cityscapes mIoU, with a best intermediate result of `0.637596` at iteration
23,500. All 40,000 records and 80 full validations pass. Its last pre-clipping
gradient norm is `72.1414`, but it is finite and is clipped by the fixed
max-norm `1.0` before the optimizer step. The paired Static Query seed-1 run is
required before interpreting this result.

The paired Static Query seed-1 run completes at `0.669958` final Cityscapes
mIoU, with a best intermediate result of `0.676172` at iteration 37,000. Its
40,000 finite contiguous records, exact objective reconstruction, 80 complete
500-image validations, and intended mechanism isolation all pass. Static leads
one-way by `4.1601` percentage points at seed 1, compared with `0.3671` points
at seed 0. Both paired differences favor Static, but their magnitude is highly
variable; complete the prespecified seed-2 pair before selecting the interaction
or adding the bidirectional control.

The one-way R=2 seed-2 member completes at `0.628523` final Cityscapes mIoU,
with a best intermediate result of `0.632286` at iteration 21,000. Its 40,000
records, 80 complete validations, exact objective reconstruction, isolation,
and checkpoint evidence pass. Run the paired Static Query seed-2 member before
computing the final three-seed interaction statistics or introducing the
bidirectional control.

The paired Static Query seed-2 member completes at `0.641193` final mIoU and
passes the same trace, validation, isolation, and checkpoint gates. Across
seeds 0/1/2, one-way obtains `0.633535 ± 0.008826` and Static obtains
`0.652849 ± 0.015138`. The paired Static-minus-One-way difference is
`0.019314 ± 0.019819` and is positive for every seed. Static is therefore
selected over one-way. Phase 13 continues only with the remaining planned
bidirectional self-attention control; this does not reopen the completed
Static-versus-One-way comparison.

The bidirectional control concatenates the R=2 grouped query tokens with the
DINOv3 patch tokens and applies one joint self-attention/FFN layer. The updated
query and image streams are split and used only by the additive query-residual
head; the frozen backbone and base segmentation decoder remain unchanged. This
isolates bidirectional interaction from the already fixed architecture and
losses. Its isolated 500-iteration GPU smoke passes at commit `1366716` with
500 contiguous finite records, exact objective reconstruction, the intended
metadata, one 50-image validation, and `2.455 GiB` peak reserved memory. The
short-run mIoU is diagnostic only. Run one seed-0 40k experiment before deciding
whether bidirectional interaction merits paired repeats; learned-null remains
deferred.

The Bidirectional seed-0 full run reaches `0.630394` final mIoU, 1.3332
percentage points below One-way and 1.7003 below Static. Its best intermediate
result (`0.636628` at iteration 33,000) also remains below both final
references. The deficit is outside the 0.5-point repeat band, so Bidirectional
is retained as a negative structural ablation without seed-1/2 repeats. Phase
13 is complete and selects Static interaction.

Before learned-null changes the intervention baseline, the selected Static
three-seed checkpoints undergo a zero-ablation semantic audit. For every
ground-truth-present class, the audit compares the absolute
factual-minus-zero class-logit effect inside that class's valid GT pixels with
the same effect on other valid pixels. This is a diagnostic only: it changes
neither the model nor its training objective.

The three-seed audit passes: the inside/outside absolute-effect ratios are
`2.794/3.583/2.363`, with an across-seed mean of `2.913 ± 0.619`; `86.48%` of
GT-present class maps have stronger effect magnitude inside their matching
region. This supports a learned-null comparator while not yet establishing
sufficiency or specificity.

## Learned-null intervention baseline

Phase 14 keeps the selected Static R=2 photometric-only factual path unchanged.
It adds one class-agnostic `R x D` null bank shared by all classes. A
stop-gradient SmoothL1 calibration fits its slots to the across-class centroid
of the current factual query states; this gives the null parameters an explicit
training signal without pulling factual queries toward the null. Replacing
only class `c`'s residual with the calibrated null defines
`Z_c(factual) - Z_c(null)`. Prediction consistency, CQE, diversity, effect
invariance, sufficiency, and specificity remain disabled so the new
intervention baseline is tested in isolation.

The first 500-step GPU smoke at `2882c87` is numerically healthy: all records
are finite, objective reconstruction is exact to `1.159e-7`, the learned-null
loss falls from a first-20 mean of `1.965e-4` to a last-20 mean of `2.636e-5`,
and peak reserved memory is `2.713 GiB`. A stricter isolation audit then found
that constructing the optional null bank before the factual Query Head consumed
random numbers and changed its same-seed initialization. The null bank is now
constructed only after every factual-path module, and a regression test checks
bitwise equality of all shared parameters with learned-null disabled. The old
smoke remains numerical evidence but must be refreshed on the repaired SHA
before a 40k comparison.

The repaired smoke at `d85db4e` passes: all 500 records are finite and
contiguous, reconstruction error is `1.790e-7`, peak reserved memory is
`2.713 GiB`, and the null calibration loss decreases by approximately `84.8%`.
This authorizes a single seed-0 40k learned-null run. Its segmentation metric
tests whether the added baseline leaves the selected factual path intact; it
does not by itself establish that learned-null effects are more causal.

The seed-0 run reaches `0.655364` final mIoU (`+0.7968` points versus paired
Static) with finite 40k optimization and null calibration converged to
`2.614e-13`. The next diagnostic computes zero and learned-null class-logit
effects from the same forward output and compares their localization on every
GT-present class map. This separates segmentation retention from the stronger
claim that the learned baseline yields a better-localized intervention effect.

That diagnostic is negative. Learned-null reduces the mean inside/outside
absolute-effect ratio from `3.773` to `1.845` and normalized contrast from
`0.491` to `0.205`; only `4.71%` of paired present-class maps improve. Its
outside-region magnitude also rises from `0.656` to `1.080`. Thus the improved
segmentation score is not evidence for a better causal baseline. Learned-null
is retained as a controlled negative ablation, zero remains a diagnostic, and
the project does not proceed to sufficiency/specificity optimization on this
mechanism.
