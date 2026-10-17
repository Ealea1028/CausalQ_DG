# Phase19 complete evidence accepted — 2026-10-17

Supersedes the earlier partial-return status. Gzip/base64 decodes to exact
report SHA e45bd7bf4cb1b334ad02938c7052668b977c0ada0d3dea6a3c566ba0c3fc010c
and original checksum SHA 902ac0da623b8ee7d569d0fd18bae222a909ee768c883c241b04b48521f6b6d4.
The report happens to share the previously printed console-log hash because
the CLI printed the same serialized JSON bytes. The report identity is now
established from its own returned bytes, not inferred from the old log hash.

tools/audit_anchor_query_effect.py verifies clean producer f5ca5b5ac604bdae0d233d631001e95ee4b6ac36,
pinned runtime and accepted Phase18 lineage, 8 frozen source IDs x 2 views,
16 backbone calls, zero optimizer updates, exact original CE replay, correct
class routing, unchanged model/RNG/files, no target/mIoU/new loss/formal run.
Embedded rows reconstruct the original components.jsonl SHA exactly; raw
checkpoint and trace remain remote. Maximum decomposition error 5.82e-11,
base-addition roundoff 2.384185791015625e-7. Peak reserved1.517578125GiB.

This accepts component computation only. It does not establish responsibility
specialization, cross-style stability, localization or utility. Priority2 is
complete; next separately scoped Phase20/ Priority3 is diagnostic analysis,
without training, new losses, target access or checkpoint selection.
