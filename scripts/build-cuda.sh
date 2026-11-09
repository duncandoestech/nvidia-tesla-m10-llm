#!/usr/bin/env bash
# Build mainline llama.cpp for Tesla M10 (sm_50) inside the CUDA 12.9 / gcc 11 image.
# CUDA 13 dropped Maxwell, so 12.x is required. Usage: build-cuda.sh <srcdir> <builddir> [extra cmake args]
set -euo pipefail
SRC=${1:-/data/m10/llama.cpp}; BLD=${2:-build-sm50}; shift 2 || true
docker run --rm --gpus all -v /data/m10:/data/m10 -w "$SRC" m10-llamacpp:cuda12.9 bash -c "
  cmake -B $BLD -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF -DGGML_NATIVE=ON $* &&
  cmake --build $BLD -j 12 --target llama-bench llama-server llama-cli test-backend-ops llama-perplexity"
