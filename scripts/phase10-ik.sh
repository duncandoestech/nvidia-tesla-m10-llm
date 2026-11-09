#!/usr/bin/env bash
# Phase 10: ik_llama.cpp (needs CMake >= 3.25 for the CUDA20 dialect) vs our best mainline build on the M10.
cd /data/m10
CM=/data/m10/tools/cmake-3.31.6-linux-x86_64/bin/cmake
BEST=/data/m10/llama.cpp-tune/build-best; IK=/data/m10/ik_llama.cpp/build-m10
M2=models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf; L7=models/llama-2-7b.Q4_0.gguf
MQ4=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf; D27=/data/models/Qwen3.8-27B-UD-Q5_K_XL.gguf
echo "== build ik_llama.cpp $(git -C ik_llama.cpp log -1 --format='%h %cd')"
docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/ik_llama.cpp m10-llamacpp:cuda12.9 bash -c \
  "$CM -B build-m10 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && $CM --build build-m10 -j 12 --target llama-bench llama-server" > logs/build-ik.log 2>&1 && echo "ik built" || { echo "IK BUILD FAIL"; grep -iE "error" logs/build-ik.log | head -5; }
IKOK=0
if [ -x $IK/bin/llama-bench ]; then
  M10_IDS=0 REPS=1 ./bench.sh p10-ik-smoke $IK -m $M2 -ngl 99 -fa 1 -p 0 -n 8; grep -q '"avg_ts"' results/p10-ik-smoke.jsonl && IKOK=1
  echo "ik smoke ok=$IKOK"; [ $IKOK = 0 ] && tail -5 results/p10-ik-smoke.err
fi
[ $IKOK = 1 ] || { echo PHASE10_DONE; exit 0; }
echo "== ik vs best, one GPU (2 rounds)"
for round in 1 2; do for m in $M2 $L7; do for b in best ik; do B=$BEST; [ $b = ik ] && B=$IK
  M10_IDS=0 REPS=3 ./bench.sh p10-$b-$(basename $m .gguf)-r$round $B -m $m -ngl 99 -fa 1 -p 512 -n 128; done; done; done
echo "== MoE Qwen3.6-35B-A3B Q4, 4 GPUs"
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p10-best-moe-layer $BEST -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p10-ik-moe-layer $IK -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p10-ik-moe-graph $IK -m $MQ4 -ngl 99 -sm graph -ts 1/1/1/1 -fa 1 -p 512 -n 128
echo "== dense Qwen3.8-27B Q5, 4 GPUs: best tensor vs ik graph"
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p10-best-27b-tensor $BEST -m $D27 -ngl 99 -sm tensor -ts 1/1/1/1 -fa 1 -p 512 -n 64
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p10-ik-27b-graph $IK -m $D27 -ngl 99 -sm graph -ts 1/1/1/1 -fa 1 -p 512 -n 64
echo PHASE10_DONE
