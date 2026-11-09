#!/usr/bin/env bash
# Phase 5: multi-GPU and batched decode for small models on the M10.
cd /data/m10; source multi.sh
cool(){ for i in $(seq 1 120); do [ $(nvidia-smi -i 0,1,2,3 --query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '$1>48' | wc -l) = 0 ] && return; sleep 5; done; }
T=/data/m10/llama.cpp-tune/build-nw1rpb4/bin; S=/data/m10/llama.cpp/build-sm50/bin
M2=models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf; L7=models/llama-2-7b.Q4_0.gguf
nvidia-smi -i 0,1,2,3 --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader -l 1 > results/p5.smi.csv & SMI=$!
echo "== A: four independent copies (aggregate throughput)"
for m in $L7; do cool; run4 p5-4copy-$(basename $m .gguf) $T -m $m -ngl 99 -fa 1 -p 512 -n 128 -r 3; done
echo "== B: one model, tensor split over 4 GPUs vs 1 GPU (single stream)"
for m in $M2 $L7; do cool; M10_IDS=0,1,2,3 REPS=3 ./bench.sh p5-tensor4-$(basename $m .gguf) /data/m10/llama.cpp-tune/build-nw1rpb4 -m $m -ngl 99 -sm tensor -ts 1/1/1/1 -fa 1 -p 512 -n 128; done
echo "== C: batched decode on one GPU, tuned vs stock"
for b in tuned stock; do B=$T; [ $b = stock ] && B=$S; cool
  bbench p5-batched-$b 0 $B -m $M2 -ngl 99 -fa on -c 16384 -npp 128 -ntg 128 -npl 1,2,4,8,16,32; done
kill $SMI
echo PHASE5_DONE
