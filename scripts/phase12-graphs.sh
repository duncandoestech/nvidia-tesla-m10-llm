#!/usr/bin/env bash
# Phase 12: CUDA graphs on Maxwell (mainline disables them below Volta) on top of the best build.
cd /data/m10
until grep -q PHASE11_DONE logs/phase11.log; do sleep 30; done
F="-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4 -DM10_VMAD -DM10_MOE_NOSPILL -DM10_GRAPHS"
docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
  "cmake -B build-best-graphs -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='$F' && cmake --build build-best-graphs -j 12 --target llama-bench test-backend-ops" > logs/build-best-graphs.log 2>&1 && echo built || { echo BUILD FAIL; tail -5 logs/build-best-graphs.log; echo PHASE12_DONE; exit 0; }
BEST=/data/m10/llama.cpp-tune/build-best; G=/data/m10/llama.cpp-tune/build-best-graphs
M10_IDS=0 REPS=1 ./bench.sh p12-smoke $G -m models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf -ngl 99 -fa 1 -p 0 -n 8
grep -q '"avg_ts"' results/p12-smoke.jsonl || { echo "SMOKE FAIL"; tail -5 results/p12-smoke.err; echo PHASE12_DONE; exit 0; }
for round in 1 2; do for m in models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf models/llama-2-7b.Q4_0.gguf; do for b in best graphs; do B=$BEST; [ $b = graphs ] && B=$G
  M10_IDS=0 REPS=3 ./bench.sh p12-$b-$(basename $m .gguf)-r$round $B -m $m -ngl 99 -fa 1 -p 0 -n 128; done; done; done
for b in best graphs; do B=$BEST; [ $b = graphs ] && B=$G
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p12-$b-moe $B -m /data/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 0 -n 128; done
echo PHASE12_DONE
