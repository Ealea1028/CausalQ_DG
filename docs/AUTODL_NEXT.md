# AutoDL next step: verify existing Cityscapes visualization provenance

Four §41 panels have been received and provisionally reviewed, but the
`report.json` and run log are missing locally. This is a **read-only** evidence
handoff, not a new GPU run. Do not repeat visualization into an existing
directory, retrain, evaluate another dataset, or delete checkpoints. The only
active target is GTA5 → Cityscapes val; BDD100K, Mapillary, ACDC, and others
remain deferred.

Run the following commands on AutoDL. They do not alter any project source or
experiment output. If a required file is absent, stop and report that fact;
do not recreate it merely to satisfy this audit.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

VIS_DIR=/root/autodl-tmp/outputs/CausalQ_DG/analysis/query_effect_static_r2_seed0_v1
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/query_effect_static_r2_seed0_v1.log
REPORT="$VIS_DIR/report.json"

echo '===== repository ====='
git rev-parse HEAD
git status --short

echo '===== existing files ====='
test -f "$REPORT" || { echo "missing_report=$REPORT"; exit 1; }
test -f "$LOG" || { echo "missing_log=$LOG"; exit 1; }
for name in road car person vegetation; do
  test -f "$VIS_DIR/$name.png" || { echo "missing_figure=$name"; exit 1; }
done
ls -lh "$REPORT" "$LOG" "$VIS_DIR"/*.png

echo '===== report ====='
cat "$REPORT"

echo '===== output integrity ====='
sha256sum "$VIS_DIR"/road.png "$VIS_DIR"/car.png \
  "$VIS_DIR"/person.png "$VIS_DIR"/vegetation.png

echo '===== errors and disk ====='
grep -E 'Traceback|AssertionError|FloatingPointError|CUDA out of memory' "$LOG" || true
df -h /root/autodl-tmp
```

Send back the complete `report.json` text, the four SHA-256 lines, the Git SHA
and status, and any error lines. The four PNGs copied locally have SHA-256
values recorded in `analysis/QUERY_EFFECT_VISUAL_REVIEW.md`; matching them
against the AutoDL originals establishes that the reviewed panels came from
this run. The report must show the exact evaluation Git SHA, pretrained and
model checkpoint hashes, sample IDs, and the intended Cityscapes-val scope.

If the report or hashes do not match, the §41 visualization remains
unverified. Once provenance is established, record the qualitative outcome
without claiming causality or multi-target generalization; only then choose
the next single project phase.
