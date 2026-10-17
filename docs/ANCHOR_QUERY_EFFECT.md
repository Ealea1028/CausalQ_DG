# Anchor-only computation — Phase19 / Priority2

This is the proposed model-internal intervention in CausalQ_DG_1016 sections
16–17, not a historical CausalQ implementation or proof of causal semantics.
Only centered Static queries without learned-null are supported in this scope.

For R slots, Q_full[c,r]=a[c]+e_tilde[c,r] and Q_anchor[c,r]=a[c]. Preserve R,
temperature, projections, per-query normalization and logsumexp aggregation.
The anchor map is NOT the average of full-query logits or an independently
normalized/aggregated alternate head. Both paths share the existing head.
forward_components(images, include_anchor=True) encodes the image once and
reuses its tokens; the default call computes no anchor path.

Conceptually, Z_zero is the common base path, Z_full=Z_zero+E_full and
Z_anchor=Z_zero+E_anchor. Each class-specific intervention changes only that
class's residual contribution and keeps other class logits unchanged.

The canonical full effect retains the historical pre-addition alpha*delta
definition. The anchor effect is the same alpha times the anchor-only head
delta. get_anchor_query_effect() returns that map in FP32;
get_residual_contribution() returns FP32(E_full)-FP32(E_anchor). Therefore
E_full=E_anchor+E_residual up to FP32 arithmetic. The residual contribution
includes normalization/logsumexp interactions; do not call it a standalone
residual causal effect. get_query_effect() remains unchanged.

BF16 factual and anchor logits retain the existing output precision.
Subtracting an already quantized base addition can differ from the pre-addition
effect through cancellation. The numerical gate reports this roundoff, without
changing old logits or relaxing the FP32 component-identity threshold. It does
not assert bitwise real-arithmetic identities on quantized base sums.

Zero reference means removing class c's query branch, as existing
counterfactual_logits(c) does. It does not mean sending zero query vectors
through the head: for zero normalized query scores the R-slot logsumexp is
log(R), not generally zero. Anchor-only repeats a[c] in all R slots;
anchor_counterfactual_logits(c) substitutes its class map only.

No parameters/checkpoint keys are added and no cached output is mutated.
The opt-in output contains anchor_logits, anchor_scaled_delta_logits and
anchor_query_states; default outputs leave them None. Anchor-only gradients
reach anchors/shared head, not raw residuals. Current scope adds no new loss,
detach policy, optimizer update or model-selection feedback. Existing CQE and
learned-null APIs keep their behavior; unsupported anchor modes fail before
the backbone is called.

CPU tests cover R=1/2/3, batch1/>1, zero alpha, zero effective residual,
raw-residual independence/gradient exclusion, repeated-anchor shape, exact
class routing, default factual/RNG preservation, one backbone call, scope
rejection, BF16 roundoff, artifact-audit ordering and model-only checkpoint
restoration. Source-only AutoDL gate uses the accepted 20-step centered
checkpoint, eight fixed holdout images and original/photo views. It verifies
original CE replay, finite effects, component/routing identities and unchanged
model/RNG/files with zero optimizer updates. No mIoU or stability/utility claim.
