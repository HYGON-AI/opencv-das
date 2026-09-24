#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

# DTK environment scripts append to variables that may not exist in a clean
# Actions container. Keep nounset disabled only while sourcing them.
set +u
source /opt/dtk/env.sh
for cuda_env in /opt/dtk/cuda/env.sh /opt/dtk/cuda/cuda/env.sh; do
  if [[ -f "${cuda_env}" ]]; then
    source "${cuda_env}"
  fi
done
set -u

nvcc_path="$(command -v nvcc || true)"
if [[ -z "${nvcc_path}" ]]; then
  nvcc_path="$(find /opt/dtk -type f -path '*/bin/nvcc' -perm /111 -print -quit)"
fi
if [[ -z "${nvcc_path}" ]]; then
  echo "DTK CUDA compiler (nvcc) was not found under /opt/dtk" >&2
  exit 1
fi

cuda_root="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
export PATH="$(dirname "${nvcc_path}"):${PATH}"
export CUDA_PATH="${cuda_root}"
export CUDA_BIN_PATH="${cuda_root}"
export CUDACXX="${nvcc_path}"
export CUDAToolkit_ROOT="${cuda_root}"

"${nvcc_path}" --version
cmake --version

cmake -S . -B build-dtk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${nvcc_path}" \
  -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
  -DCUDA_TOOLKIT_ROOT_DIR="${cuda_root}" \
  -DCUDAToolkit_ROOT="${cuda_root}" \
  -DOPENCV_EXTRA_MODULES_PATH="${GITHUB_WORKSPACE}/opencv_contrib/modules" \
  -DWITH_CUDA=ON \
  -DBUILD_LIST=core,imgproc,cudev,cudaarithm \
  -DBUILD_TESTS=ON \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python3=OFF

cmake --build build-dtk --parallel "${MAX_JOBS:-4}"
ctest --test-dir build-dtk --output-on-failure --no-tests=error \
  -R 'opencv_test_(core|cudaarithm)$'
