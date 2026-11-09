#!/usr/bin/env bash
# Build mmvq launch-shape variants (ncols_dst==1 path) for sm_50 from the llama.cpp-tune worktree.
cd /data/m10
until grep -q BUILD_DMMV_AB_DONE logs/build-dmmv-ab.log 2>/dev/null; do sleep 20; done
for v in "1 1" "2 1" "8 1" "4 2" "2 2" "1 2"; do
  set -- $v; name=nw$1rpb$2
  docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
   "cmake -B build-$name -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='-DM10_NW1=$1 -DM10_RPB1=$2' && cmake --build build-$name -j 12 --target llama-bench" > logs/build-tune-$name.log 2>&1
  ls llama.cpp-tune/build-$name/bin/llama-bench || echo "BUILD FAIL $name"
done
echo TUNE_BUILDS_DONE
