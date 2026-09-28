#!/usr/bin/env bash
# Set up and probe the pinned external REIN runtime without touching causalq-dg.
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/CausalQ_DG
EXTERNAL_ROOT=/root/autodl-tmp/external
REIN_ROOT="$EXTERNAL_ROOT/rein-phase16-dc063429"
REIN_SHA=dc063429c4dadc0da9c6252b3db22fc55a9882ab
ENV_DIR=/root/autodl-tmp/envs/rein-phase16-py310-cu118
ANALYSIS_ROOT=/root/autodl-tmp/outputs/CausalQ_DG/analysis
REPORT="$ANALYSIS_ROOT/rein_phase16_runtime.json"
FREEZE="$ANALYSIS_ROOT/rein_phase16_pip_freeze.txt"
MODE=${1:-fresh}
case "$MODE" in
  fresh) ;;
  --recover-conda)
    # Preserve the failed prefix, original log, and already pinned source.
    ENV_DIR=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1
    REPORT="$ANALYSIS_ROOT/rein_phase16_runtime_tuna_retry1.json"
    FREEZE="$ANALYSIS_ROOT/rein_phase16_pip_freeze_tuna_retry1.txt"
    ;;
  *) echo 'Usage: bootstrap_rein_phase16.sh [--recover-conda]' >&2; exit 2 ;;
esac
test "$#" -le 1
SOURCE=/root/autodl-tmp/pretrained/.dinov2_vitl14_pretrain_phase16.pth.part
SOURCE_SHA=d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
CONVERTED=/root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth
CONVERTED_SHA=91730ebf59fb634f5572cf5071fef8665473dcffcbef7ba4f4fa497533a8c837

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
test -f "$SOURCE"
test -f "$CONVERTED"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(sha256sum "$CONVERTED" | cut -d' ' -f1)" = "$CONVERTED_SHA"
if [ "$MODE" = fresh ]; then
  test ! -e "$REIN_ROOT"
else
  test -d "$REIN_ROOT/.git"
  test "$(git -C "$REIN_ROOT" rev-parse HEAD)" = "$REIN_SHA"
  test -z "$(git -C "$REIN_ROOT" status --porcelain)"
fi
test ! -e "$ENV_DIR"
test ! -e "$REPORT"
test ! -e "$FREEZE"
AVAILABLE_KIB=$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')
test "$AVAILABLE_KIB" -ge 14680064 || {
  echo 'Need at least 14 GiB free before isolated REIN installation' >&2
  exit 1
}
command -v conda >/dev/null
mkdir -p "$EXTERNAL_ROOT" "$ANALYSIS_ROOT"

if [ "$MODE" = fresh ]; then
  git clone https://github.com/w1oves/Rein.git "$REIN_ROOT"
  git -C "$REIN_ROOT" checkout --detach "$REIN_SHA"
fi
test "$(git -C "$REIN_ROOT" rev-parse HEAD)" = "$REIN_SHA"
test -z "$(git -C "$REIN_ROOT" status --porcelain)"

# Ignore .condarc/defaults for this command only; retain TLS verification.
conda create --prefix "$ENV_DIR" --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --no-default-packages python=3.10 pip -y
PYTHON="$ENV_DIR/bin/python"
"$PYTHON" -m pip install --no-cache-dir numpy==1.26.4
"$PYTHON" -m pip install --no-cache-dir \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install --no-cache-dir --only-binary=mmcv \
  numpy==1.26.4 mmcv==2.1.0 mmengine==0.10.7 \
  mmsegmentation==1.2.2 mmdet==3.3.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
"$PYTHON" -m pip install --no-cache-dir \
  ftfy==6.2.0 scipy==1.11.4 prettytable==3.10.0 \
  matplotlib==3.8.4 regex==2024.4.16 \
  timm==0.9.12 einops==0.7.0
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$FREEZE"
"$PYTHON" -m tools.check_rein_runtime \
  --rein-root "$REIN_ROOT" \
  --expected-rein-sha "$REIN_SHA" | tee "$REPORT"
test "${PIPESTATUS[0]}" -eq 0
echo 'rein_phase16_runtime_gate_passed=true'
sha256sum "$REPORT" "$FREEZE"
df -h /root/autodl-tmp
