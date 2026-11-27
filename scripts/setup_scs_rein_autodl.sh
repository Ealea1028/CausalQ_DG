#!/usr/bin/env bash
# Build the pinned MMCV CUDA wheel in an isolated SCS environment, then run P0.
set -Eeuo pipefail

SHA=${1:?usage: setup_scs_rein_autodl.sh FULL_PROJECT_SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
cd "$(git rev-parse --show-toplevel)"
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"

BASE=/root/autodl-tmp
SOURCE_ENV="$BASE/envs/causalq-lora1114"
REIN="$BASE/external/rein-phase16-dc063429"
REIN_SHA=dc063429c4dadc0da9c6252b3db22fc55a9882ab
test -x "$SOURCE_ENV/bin/python"
test "$(git -C "$REIN" rev-parse HEAD)" = "$REIN_SHA"
test -z "$(git -C "$REIN" status --porcelain --untracked-files=no)"
command -v conda >/dev/null
OUT=$(mktemp -d "$BASE/outputs/scs_p0_setup_${SHA:0:7}_XXXXXX")
ENV="$OUT/env"
PY="$ENV/bin/python"
echo "OUT=$OUT"
exec > >(tee "$OUT/setup.log") 2>&1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
unset PYTHONPATH
printf '%s\n' "$SHA" > "$OUT/project_sha.txt"
printf '%s\n' "$REIN_SHA" > "$OUT/rein_sha.txt"
printf '%s\n' "$PY" > "$OUT/python_path.txt"

echo '== Clone DINOv3 runtime =='
conda create --yes --prefix "$ENV" --clone "$SOURCE_ENV"
export PATH="$ENV/bin:$PATH"
"$PY" -c 'import torch, transformers; assert torch.__version__ == "2.7.0+cu128" and torch.version.cuda == "12.8" and transformers.__version__ == "4.56.1"'

echo '== Install MMCV build and REIN runtime dependencies =='
"$PY" -m pip install --disable-pip-version-check \
  setuptools==80.9.0 wheel==0.45.1 ninja==1.11.1.4 numpy==1.26.4 \
  mmengine==0.10.7 mmsegmentation==1.2.2 mmdet==3.3.0 \
  ftfy==6.2.0 regex==2024.4.16 timm==0.9.12 einops==0.7.0
if ! "$PY" -c 'import cv2'; then
  "$PY" -m pip install --disable-pip-version-check opencv-python-headless==4.11.0.86
fi

echo '== Locate the CUDA 12.8 build toolkit =='
valid_toolkit() {
  [[ -n "$1" && -x "$1/bin/nvcc" ]] &&
    "$1/bin/nvcc" --version | grep 'release 12\.8,' >/dev/null
}
toolkit=''
for candidate in "${CUDA_HOME:-}" /usr/local/cuda-12.8 /usr/local/cuda; do
  if valid_toolkit "$candidate"; then toolkit="$candidate"; break; fi
done
if [[ -z "$toolkit" ]]; then
  toolkit="$OUT/cuda-toolkit"
  CONDA_PKGS_DIRS="$BASE/conda-pkgs-scs" conda create --yes --prefix "$toolkit" \
    --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge \
    cuda-nvcc cuda-libraries-dev
  valid_toolkit "$toolkit"
fi
export CUDA_HOME="$toolkit"
"$CUDA_HOME/bin/nvcc" --version
if ! command -v g++ >/dev/null || ! command -v gcc >/dev/null; then
  CONDA_PKGS_DIRS="$BASE/conda-pkgs-scs" conda install --yes --prefix "$ENV" \
    --override-channels -c conda-forge gcc_linux-64=12 gxx_linux-64=12
  export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
fi
"${CXX:-g++}" --version

echo '== Build CUDA MMCV 2.1.0 for RTX 4090 =='
MMCV_SOURCE='https://files.pythonhosted.org/packages/a7/19/f97f0fd4d432113a232e75f965335be6b7bb6557ae9faf8051e9a012cadd/mmcv-2.1.0.tar.gz#sha256=d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f'
mkdir "$OUT/wheels" "$OUT/build-tmp"
TMPDIR="$OUT/build-tmp" CUDA_VISIBLE_DEVICES='' \
  MAX_JOBS="${SCS_BUILD_JOBS:-4}" MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9 \
  "$PY" -m pip wheel --verbose --no-deps --no-build-isolation \
    --wheel-dir "$OUT/wheels" "mmcv @ $MMCV_SOURCE"
wheel=("$OUT"/wheels/mmcv-2.1.0-*.whl)
test "${#wheel[@]}" -eq 1
test -s "${wheel[0]}"
sha256sum "${wheel[0]}" > "$OUT/mmcv_wheel.sha256"
"$PY" -m pip install --disable-pip-version-check "${wheel[0]}"
"$PY" -m pip list --format=freeze > "$OUT/pip_list.txt"
conda list --explicit --prefix "$ENV" > "$OUT/conda_explicit.txt"

echo '== Probe REIN head on GPU =='
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt"
exec 9>>"$BASE/outputs/CausalQ_DG/lora_causalq_1114/gpu0.lock"
# Keep the built environment and wait briefly if the GPU is busy.
flock -w 60 9
RC=0
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.probe_scs_rein_runtime --expected-sha "$SHA" \
  --rein-root "$REIN" --report "$OUT/report.json" > "$OUT/probe.log" 2>&1 || RC=$?
if [[ -f "$OUT/report.json" ]]; then cat "$OUT/report.json"; else cat "$OUT/probe.log"; fi
echo "exit=$RC OUT=$OUT PY=$PY"
exit "$RC"
