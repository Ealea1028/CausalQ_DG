# Full-source census complete — 2026-10-17

Completion update: extracted artifacts supplied locally at project
analysis_export/. All64 SHA256 hashes match; independent CPU/math.fsum
reconstruction reproduces all142329 effect/189772 responsibility rows and
all2497 image/class/slot/global summaries, GT masks/denominators/ratios and
raw-variance comparison.60 PNGs decode. All20 fixed cases reviewed in
overview/effect/responsibility contact sheets; native-size first21312 effects
and last16752 overview inspected for scales, ignore regions and alignment.
No mismatched geometry or scale substitution observed. Coarse predictions,
broad car/person effects and asymmetric slot usage are expected diagnostics
of this early state, not proof of semantic modes or causal usefulness.

raw_review.json pins independent recomputation; completion_review.json pins
the manual visual scope separately (the automated record does not claim
visual judgment). Priority3 engineering is complete. No further file transfer
or source-analysis run is required. Original artifacts are ignored by Git.
The earlier pending-file narrative below is retained for provenance.

Complete return report SHA88dd5291bc163165f7f174c9820f5b2f8924fffb82c230de76d8b9e1dbcd4de5
and checksum SHA53e6740705c8b3ad88e66cfd98d70a3c2514b9eed00f4e64af77c3a302e2b4b1
match the compressed attachment bytes. Producer79e440b5e78b34e39ef1792e67807ba164d6752e,
clean source tree, original20-step M1 checkpoint,2497 source IDs/4994 forwards,
zero updates/target access/new losses, same model/style/RNG and original files.

tools/audit_query_component_census.py verifies full source membership,
prior20 overlap replay,19 parameter rows, all19-class summary denominators,
weighted responsibility/effect aggregate reconstruction, fixed20/five-class
figure paths and every returned manifest entry. Report counts189772
responsibility rows/142329 effect rows/60 PNGs; peak reserved1.650390625GiB.
Max decomposition error1.1641532182693481e-10. This is report verification,
not independent recomputation of the remote raw CSVs or actual visual QA.

| Component | Original mean absolute effect | Paired population variance | RMS-normalized pair MSE |
| --- | ---: | ---: | ---: |
| full | 0.00151511 | 6.34305e-10 | 0.00280743 |
| anchor | 0.00176851 | 8.42671e-10 | 0.00253016 |
| residual contribution | 0.000298867 | 4.51276e-11 | 0.00303172 |

Raw anchor/residual variance ratio18.6731. For each image, using its macro
mean over19 class maps,0/2497 images have lower raw anchor variance; none tie.
Mean raw difference7.975434000961372e-10. The raw stability inequality in
CausalQ_DG_1016 section46 is not supported for this early engineering state.
Normalized ordering is a distinct diagnostic and must not replace that
hypothesis. Original/photo responsibility pixel entropy0.554449/0.554566
nats,27723 present image-class maps/view; no semantic-specialization claim.

Limitations:20 training updates only, alpha0.00121144, one seeded photo/image,
resized source GT, filename split without scene-independence evidence. No
formal M1 accuracy, DG utility or trained-method rejection follows. Do not
add an Anchor-CQE loss based on normalized variance alone.

Attachment contains text only. Analysis archive is not locally available:
remote path /root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_CENSUS2497_79e440b/analysis_export.zip,
reported size45MiB, SHA3010b35769c33408eae2af2b57c491d4e65b70880dffb24da35108e935be8d6b.
It contains3 complete CSVs,2497 sample summaries and60 PNGs. Next action is
transfer that existing archive, verify all64 members and recompute/review
raw statistics and figures locally. Never put source previews/archive in Git.
Priority3 remains open until this review. No GPU rerun or formal training.
