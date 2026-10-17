# Phase20 / Priority3 diagnostic contract

This phase adds analysis only, after accepted Phase19 at 01f59e9. Existing
model, parameters, APIs, checkpoint, objective and intervention ranges are
unchanged. No optimizer, training, target access, mIoU selection or new loss.
The 20-step checkpoint is an engineering artifact, not a formal M1 result.

## Responsibility and parameters

rho[c,r,p] = softmax_R(score[c,r,p]) using existing temperature-scaled score
maps. This is competition/attribution, not causal effect. Use native score
grid; source labels resized with nearest interpolation. Ignore255; each
class's rows use only its own GT pixels. Empty classes return null/blank,
not zero. Tiny classes can disappear at this resolution; explicit pixel
counts record that limitation.

For every class/slot/view: mean responsibility, mean pixel entropy in nats,
entropy of mean usage, exp(usage entropy), activation fraction rho>1/R,
top1 fraction with ties split equally, paired-view population variance.
Contribution means mean(logsumexp(all slots)-logsumexp(without slot r)),
before alpha/base. It is a score aggregation deletion diagnostic, not an
independent residual effect; R=1 deletion is undefined. Duplicate slots
retain positive deletion drops via log R, so a nonzero drop does not prove
semantic specialization. CSV includes absent classes explicitly.

Parameter rows record raw query pair cosine for a+centered residuals, anchor
norm, centered residual norms and anchor/mean-residual norm ratio. Zero-norm
ratios/cosines and R=1 pairs are null. No projected-vector interpretation is
substituted for parameter-space cosine.

## Effect statistics

Full/anchor effects retain pre-addition alpha*delta. Residual contribution
is their FP32 difference. On valid resized source pixels, record per-image/
class/component magnitude (mean absolute), RMS, population variance over
the original/photo pair = mean((x-y)^2/4), and RMS-normalized pair MSE.
Normalized MSE is null if either map has zero energy. Record inside/outside
GT-class mean absolute effects and their ratio (outside epsilon1e-12);
absent regions are null. This descriptive localization is not a new
necessity intervention or a claim about causal utility. Magnitude/energy
must accompany variance to expose trivial near-zero effects.

Summaries are macro means over defined image-class maps with defined counts.
Responsibility summaries duplicate class-level entropy for each R slot;
fixed R means equal class weights. Raw CSV rows retain region/pixel counts;
no missing values are imputed. Neither lower variance nor parameter cosine
is a correctness acceptance criterion or proof of specialization.

## Frozen sampling and artifact contract

tools/analyze_query_components.py accepts checkpoint, frozen source manifest,
source/weight roots and original Phase19 directory. No budget override. Gate
selects 20 of 2497 source holdout IDs by ascending SHA256 of
causalq-components-v1:ID, independently of labels/results; fixed seed is first
8 hex digits of SHA256(causalq-components-photo-v1:ID). It is independent of
model initialization and sample processing order. SourceEvalTransform gives
long-edge1024/pad16 full-frame resized source GT, not original-resolution GT.
The photometric config, per-image seed and input hashes enter the report.
Replay the photometric image exactly with the same seed; isolate/restore RNG.

Original Phase18 report/trace/checkpoint and Phase19 report/trace checksums
are audited before CUDA/data and again afterward. Load model state only.
Exactly40 forwards, one backbone encoding per view. All model/style/RNG state
must remain unchanged; all original files remain immutable. Runtime stays
torch2.7.0+cu126/transformers4.56.1, BF16, workers0, strict deterministic,
CUBLAS :4096:8, TF32true/cuDNNbenchmarkfalse. No target module/data import.

Outputs: component_effect_report.json, query_similarity.csv,
responsibility.csv, effect_statistics.csv and SHA256SUMS. Report includes
per-image summary, seed/input hashes, parameter rows, coverage, artifact
hashes, environment and state checks; full CSVs stay on AutoDL. This gate
verifies analysis engineering on a bounded sample only. Full2497 analysis,
systematic20-sample heatmaps and scientific comparisons remain subsequent
work after this gate; do not infer Var(anchor)<Var(residual) over the entire
holdout from these20 inputs, or use this early checkpoint to choose a method.
