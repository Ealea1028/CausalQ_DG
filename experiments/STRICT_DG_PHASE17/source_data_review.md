# Phase 17 source-data acceptance

Read-only audit of the returned JSON matches the supplied SHA256SUMS:
`cf01d813f6125c42a8e6712e449bfdba72ce5f287222357c975780d7cad8fab3`.
Exact producer `cb5207101915c97b23da82792608b2e2c24c123e`, clean checkout;
Python3.12.3, torch2.7.0+cu126, NumPy2.3.2 in the existing DINOv3 runtime.
The small immutable report is versioned alongside this review; it contains
no raw data, weights or checkpoints. `tools/audit_strict_source_data.py`
checks its pinned bytes, protocol, full coverage and acceptance fields.

- Complete train22469/holdout2497 scans; manifest and pair/path/size hashes match.
- Both splits cover labels0..18 and255; no invalid labels, unreadable pairs,
  all-ignore labels or incompatible geometry.
- 57 train and5 holdout size differences are scale-equivalent under the already
  declared tolerance/alignment. Do not reinterpret them as scene shifts.
- All2497 deterministic holdout transforms pass, with shapes560x1024 or576x1024;
  minimum valid fraction0.261315 and unchanged torch RNG.
- 32 train crop/photo checks pass; mean absolute photo change0.327037.

Source mIoU, target mIoU, effect variance/magnitude, localization and
responsibility were not measured. No optimizer updates occurred. This accepts
the source data engineering gate only; it does not prove independence of
correlated scenes/frames, anchor hypotheses or segmentation generalization.

Next isolated gate: 20 fresh seed0 optimizer updates on this same Static R=2
model with original+photometric CE and the fixed40k schedule horizon; bounded
source holdout compatibility only, final compact checkpoint roundtrip, frozen
backbone verification and memory measurement. No target access, long training,
new Query parameterization, CQE, diversity, learned-null or REIN changes.
