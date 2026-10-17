# Centered Query parameterization — Phase18 / Priority1

The first and only new research factor is explicit centering in the existing
GroupedCausalQueryBank. Static R=2, frozen DINOv3-L, logsumexp aggregation,
temperature .07, style/loss/optimizer/schedule stay fixed. No new loss,
anchor-effect computation, necessity mechanism or target access in this change.

For raw learnable residuals e[c,r,d] and anchors a[c,d]:

    e_bar[c,d] = mean_r e[c,r,d]
    e_tilde[c,r,d] = e[c,r,d] - e_bar[c,d]
    q[c,r,d] = a[c,d] + e_tilde[c,r,d]

Thus mean_r(e_tilde)=0 and mean_r(q)=a, up to floating-point arithmetic.
Gradients through centering are projected by I - 11^T/R along the slot axis,
so the raw residual CE gradient sums to zero across slots. A shared residual
shift cancels. For R=1 the effective residual and its task gradient are zero;
queries equal anchors. Raw parameter means need not be zero and are never
projected or rewritten in place. Existing AdamW decay still acts on raw
parameters; no regularizer or optimizer change is introduced.

This fixes a parameterization convention. It does not establish causal
identifiability, semantic disentanglement or a performance gain. In particular,
mean(q)=a before the unchanged head normalization does not mean the mean of
normalized queries equals a, nor that anchor logits equal mean full-query
logits. Anchor-only effect remains later work.

Compatibility:

- GroupedCausalQueryBank/QuerySegmentor accept centered_residuals=False by default.
  Existing constructors and raw a+e forward behavior are unchanged.
- Parameter names/shapes and initialization RNG consumption are unchanged.
- get_query_residuals() returns the raw Parameter in either mode.
- get_anchors() returns the anchor Parameter; get_centered_residuals() computes
  e-mean(e) without mutating raw storage. Centered forward uses it only when
  explicitly enabled.
- No checkpoint tensor key records the Python mode flag. Config and checkpoint
  metadata MUST pin centered_residuals. The new loader rejects wrong/missing
  semantic mode/provenance and does not silently reinterpret legacy checkpoints.
  Historical factories default false; no old checkpoint is converted.
- Same raw initialization with centering changes nonzero-alpha query outputs;
  this is the intended M1 factor. At alpha=0, factual logits equal the baseline
  path exactly in eval mode. No claim of optimizer-trajectory equivalence.

CPU tests cover R=1/2/3, batch 1/>1, algebra, shift invariance, projected
gradients, legacy initialization/state/forward, alpha-zero, mode-aware compact
checkpoint roundtrip and config guard. The initial M1 config changes only the
centered flag plus phase/ID/status from strict M0 and keeps training false.

Next AutoDL gate is exactly 20 fresh source optimizer updates and eight fixed
source holdout compatibility inputs. It audits accepted Phase17 metadata first,
checks CUDA CE, alpha-zero, initial/final algebra (1e-6), finite/nonzero branch
gradients and raw residual gradient sum (1e-6), frozen backbone, fresh-head
model/optimizer roundtrip and predictions <=1e-5. No mIoU/selection threshold,
long training, target access or effect claim. The restored head shares the
same frozen backbone; independent M1 process continuation is not verified here.
