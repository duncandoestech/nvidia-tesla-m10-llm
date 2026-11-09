#!/usr/bin/env bash
# Build llama.cpp at the last commit before DMMV removal (c3ea58aca^), with and without GGML_CUDA_FORCE_DMMV.
set -e
cd /data/m10
[ -d llama.cpp-predmmv ] || git -C llama.cpp worktree add --detach /data/m10/llama.cpp-predmmv c3ea58aca^
for v in mmvq dmmv; do
  X=""; [ $v = dmmv ] && X="-DGGML_CUDA_FORCE_DMMV=ON"
  docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-predmmv m10-llamacpp:cuda12.9 bash -c \
    "cmake -B build-$v -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $X && cmake --build build-$v -j 12 --target llama-bench" > logs/build-predmmv-$v.log 2>&1
  ls -la llama.cpp-predmmv/build-$v/bin/llama-bench
done
echo BUILD_DMMV_AB_DONE
