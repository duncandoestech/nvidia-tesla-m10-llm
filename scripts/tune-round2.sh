#!/usr/bin/env bash
# Round 2: push rows-per-block further, then check correctness of the winner.
cd /data/m10
for v in "1 4" "1 8" "2 4"; do
  set -- $v; name=nw$1rpb$2
  docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
   "cmake -B build-$name -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='-DM10_NW1=$1 -DM10_RPB1=$2' && cmake --build build-$name -j 12 --target llama-bench test-backend-ops" > logs/build-tune-$name.log 2>&1 || echo "BUILD FAIL $name"
done
export M10_IDS=2 REPS=5
M=models/Qwen3.5-2B-Q4_K_M.gguf
for round in 1 2; do
  for v in stock nw1rpb2 nw1rpb4 nw1rpb8 nw2rpb4; do
    B=/data/m10/llama.cpp-tune/build-$v; [ $v = stock ] && B=build-sm50
    ./bench.sh tune2-$v-r$round $B -m $M -ngl 99 -fa 1 -p 0 -n 128
  done
done
for v in nw1rpb4 nw1rpb8; do
  echo "== correctness $v"
  docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=2 -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 \
    /data/m10/llama.cpp-tune/build-$v/bin/test-backend-ops -b CUDA0 -o MUL_MAT -p "n=1," 2>&1 | grep -E "FAIL|passed|ASSERT" | tail -3
done
echo TUNE2_DONE
