#!/usr/bin/env bash
# Phase 7: vmad dp4a A/B, then MoE models, MoE ubatch, and long-prompt flash attention on the M10.
cd /data/m10
until grep -q PHASE6_DONE logs/phase6.log; do sleep 30; done
T=/data/m10/llama.cpp-tune/build-nw1rpb4; V=/data/m10/llama.cpp-tune/build-vmad; S=build-sm50
echo "== build vmad"
docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
  "cmake -B build-vmad -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='-DM10_NW1=1 -DM10_RPB1=4 -DM10_VMAD' && cmake --build build-vmad -j 12 --target llama-bench test-backend-ops" > logs/build-vmad.log 2>&1 && VOK=1 || { echo "VMAD BUILD FAIL"; tail -5 logs/build-vmad.log; VOK=0; }
if [ $VOK = 1 ]; then
  echo "== vmad correctness"
  docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 \
    $V/bin/test-backend-ops -b CUDA0 -o MUL_MAT -p "n=1," 2>&1 | grep -E "passed|FAIL" | tail -2
  echo "== vmad A/B (GPU 0, tg128)"
  for round in 1 2; do for m in models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf models/Qwen3.5-2B-Q4_K_M.gguf models/Qwen3.5-2B-Q8_0.gguf models/llama-2-7b.Q4_0.gguf; do
    for b in base vmad; do B=$T; [ $b = vmad ] && B=$V
      M10_IDS=0 REPS=3 ./bench.sh p7-$b-$(basename $m .gguf)-r$round $B -m $m -ngl 99 -fa 1 -p 0 -n 128; done; done; done
fi
echo "== MoE sweep (4 GPUs, layer split)"
for m in Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL; do
  for b in tuned stock; do B=$T; [ $b = stock ] && B=$S
    [ $b = stock ] && [[ $m == *Q2_K_XL ]] && continue
    M10_IDS=0,1,2,3 REPS=2 ./bench.sh p7-moe-$b-$m $B -m /data/models/$m.gguf -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128; done; done
echo "== MoE ubatch sweep (Q3)"
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p7-moe-ub $T -m /data/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -ub 128,256,512,1024 -p 2048 -n 0
echo "== long-prompt flash attention (head size 256)"
M10_IDS=0 REPS=2 ./bench.sh p7-fa-2b $T -m models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf -ngl 99 -fa 0,1 -p 512,2048,4096 -n 0
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p7-fa-moe $T -m /data/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf -ngl 99 -sm layer -ts 1/1/1/1 -fa 0,1 -p 4096 -n 0
echo PHASE7_DONE
