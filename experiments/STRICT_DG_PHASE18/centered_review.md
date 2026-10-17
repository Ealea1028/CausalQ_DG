# Centered Query engineering acceptance — 2026-10-17

The complete compressed return decodes byte-exactly to the report and original
checksum manifest pinned by tools/audit_centered_query.py. Producer is clean
958392541519cdad5f80d628c18959176966c6c8, seed0, Python3.12.3 / torch2.7.0+cu126 /
transformers4.56.1 / RTX4090D. The operator reports all three file checksums OK.
Raw train trace/checkpoint bytes remain remote; no independent local raw audit.

Twenty updates finish. Initial residual/query-mean errors are 1.86e-9/7.45e-9;
final errors 1.86e-9/3.73e-9. Alpha-zero error 0, raw residual gradient sum error
0, finite nonzero decoder/anchor/residual/projection/alpha gradient groups,
frozen backbone, exact model/optimizer roundtrip and prediction error0 on eight
source references. Peak reserved 1.41796875 GiB. No target access, mIoU,
independent-process centered continuation or formal training was measured.

This accepts parameterization engineering, not performance/causal semantics or
anchor-effect stability. No comparison of the eight CE values establishes a
M1 accuracy gain. Default legacy behavior is unchanged. The next separately
scoped mechanism is anchor-only effect computation, with zero optimizer updates
and the same immutable centered checkpoint. No Anchor-CQE/new loss yet.
