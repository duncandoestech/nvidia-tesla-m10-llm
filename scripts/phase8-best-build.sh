#!/usr/bin/env bash
# Phase 8: combined best build (single+batch mmvq tuning, vmad, MoE spill fix), MoE rows variant, verification, head-to-heads.
cd /data/m10; source multi.sh
until grep -q PHASE7_DONE logs/phase7.log; do sleep 30; done
BEST="-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4 -DM10_VMAD -DM10_MOE_NOSPILL"
bld(){ docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
  "cmake -B $1 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='$2' && cmake --build $1 -j 12 --target $3" > logs/build-$1.log 2>&1 && echo "built $1" || { echo "BUILD FAIL $1"; tail -5 logs/build-$1.log; }; }
bld build-best "$BEST" "llama-bench llama-batched-bench test-backend-ops llama-server llama-speculative-simple llama-perplexity llama-cli"
bld build-best-moerpb4 "$BEST -DM10_MOE_RPB=4" "llama-bench llama-batched-bench test-backend-ops"
TB=/data/m10/llama.cpp-tune
echo "== spill check (mmvq kernels with a stack frame)"
for b in build-best build-best-moerpb4; do echo "$b: $(docker run --rm -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 cuobjdump -res-usage $TB/$b/bin/libggml-cuda.so 2>/dev/null | grep -A1 mul_mat_vec_q | grep -oE 'STACK:[0-9]+' | sort | uniq -c | tr '\n' ' ')"; done
echo "== correctness"
for b in build-best build-best-moerpb4; do for op in MUL_MAT MUL_MAT_ID; do
  echo "$b $op: $(docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 $TB/$b/bin/test-backend-ops -b CUDA0 -o $op 2>&1 | grep -E 'tests passed|ASSERT|FAIL' | head -2 | tr '\n' ' ')"; done; done
echo "== dense head-to-head (GPU 0): stock vs single-tuned vs best"
for round in 1 2; do for m in models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf models/llama-2-7b.Q4_0.gguf; do
  for b in stock nw1rpb4 best; do B=$TB/build-$b; [ $b = stock ] && B=build-sm50
    M10_IDS=0 REPS=3 ./bench.sh p8-$b-$(basename $m .gguf)-r$round $B -m $m -ngl 99 -fa 1 -p 512 -n 128; done; done; done
echo "== MoE head-to-head (4 GPUs, layer): nw1rpb4 vs best vs best-moerpb4"
MQ=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf
for b in nw1rpb4 best best-moerpb4; do M10_IDS=0,1,2,3 REPS=2 ./bench.sh p8-moe-$b $TB/build-$b -m $MQ -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128; done
echo "== MoE batched decode (4 GPUs): best vs best-moerpb4 vs nw1rpb4"
for b in nw1rpb4 best best-moerpb4; do echo "-- $b"; bbench p8-moebb-$b 0,1,2,3 $TB/build-$b/bin -m $MQ -ngl 99 -fa on -c 8192 -npp 128 -ntg 64 -npl 1,2,4,8; done
echo PHASE8_DONE
