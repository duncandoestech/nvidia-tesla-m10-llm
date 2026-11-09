#!/usr/bin/env bash
# Phase 9: ik_llama.cpp vs our best mainline build, MoE MTP speculative decoding, MoE ubatch 2048, M10 power-cap sweep.
cd /data/m10
until grep -q PHASE8_DONE logs/phase8.log; do sleep 30; done
restore_pl(){ sudo -n nvidia-smi -i 0,1,2,3 -pl 53 >/dev/null; echo "power limits restored: $(nvidia-smi -i 0,1,2,3 --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"; }
trap restore_pl EXIT
BEST=/data/m10/llama.cpp-tune/build-best; IK=/data/m10/ik_llama.cpp/build-sm50
M2=models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf; L7=models/llama-2-7b.Q4_0.gguf
MQ4=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf; D27=/data/models/Qwen3.8-27B-UD-Q5_K_XL.gguf
echo "== A: build ik_llama.cpp ($(git -C ik_llama.cpp log -1 --format='%h %cd'))"
docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/ik_llama.cpp m10-llamacpp:cuda12.9 bash -c \
  "cmake -B build-sm50 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build-sm50 -j 12 --target llama-bench llama-server" > logs/build-ik.log 2>&1 && echo "ik built" || { echo "IK BUILD FAIL"; tail -5 logs/build-ik.log; }
IKOK=0
if [ -x $IK/bin/llama-bench ]; then
  M10_IDS=0 REPS=1 ./bench.sh p9-ik-smoke $IK -m $M2 -ngl 99 -fa 1 -p 0 -n 8 && grep -q '"avg_ts"' results/p9-ik-smoke.jsonl && IKOK=1
  echo "ik smoke ok=$IKOK"; [ $IKOK = 0 ] && tail -5 results/p9-ik-smoke.err
fi
if [ $IKOK = 1 ]; then
  echo "== B: ik vs best, one GPU"
  for round in 1 2; do for m in $M2 $L7; do for b in best ik; do B=$BEST; [ $b = ik ] && B=$IK
    M10_IDS=0 REPS=3 ./bench.sh p9-$b-$(basename $m .gguf)-r$round $B -m $m -ngl 99 -fa 1 -p 512 -n 128; done; done; done
  echo "== B2: MoE Qwen3.6-35B-A3B Q4 on 4 GPUs"
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-best-moe-layer $BEST -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-ik-moe-layer $IK -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-ik-moe-graph $IK -m $MQ4 -ngl 99 -sm graph -ts 1/1/1/1 -fa 1 -p 512 -n 128
  echo "== B3: dense Qwen3.8-27B Q5 on 4 GPUs: best tensor vs ik graph"
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-best-27b-tensor $BEST -m $D27 -ngl 99 -sm tensor -ts 1/1/1/1 -fa 1 -p 512 -n 64
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-ik-27b-graph $IK -m $D27 -ngl 99 -sm graph -ts 1/1/1/1 -fa 1 -p 512 -n 64
fi
echo "== C: MoE MTP speculative decoding (best build, 4 GPUs, llama-server)"
./spec-server-test.sh p9-moe-nospec $BEST/bin 0,1,2,3 64 -m $MQ4 -sm layer -ts 1/1/1/1 2>&1 | tail -4
for n in 1 2 3; do ./spec-server-test.sh p9-moe-mtp$n $BEST/bin 0,1,2,3 64 -m $MQ4 -sm layer -ts 1/1/1/1 --spec-type draft-mtp --spec-draft-n-max $n 2>&1 | tail -4; done
echo "== D: MoE ubatch 2048 (best)"
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-moe-ub $BEST -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -b 4096 -ub 1024,2048 -p 2048 -n 0
echo "== E: power-cap sweep (all 4 M10 GPUs)"
for cap in 53 45 38 30; do
  sudo -n nvidia-smi -i 0,1,2,3 -pl $cap >/dev/null; echo "-- cap $cap W: $(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader)"
  M10_IDS=0 REPS=3 ./bench.sh p9-pl$cap-2b $BEST -m $M2 -ngl 99 -fa 1 -p 512 -n 128
  M10_IDS=0,1,2,3 REPS=2 ./bench.sh p9-pl$cap-moe $BEST -m $MQ4 -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -p 512 -n 128
done
restore_pl; trap - EXIT
echo PHASE9_DONE
