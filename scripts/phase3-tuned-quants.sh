#!/usr/bin/env bash
# Phase 3: every quant on the tuned mmvq builds (1 warp, 4 or 2 rows/block) vs stock, GPU 0, generation only.
cd /data/m10
export M10_IDS=0 REPS=3
for f in Qwen3.5-2B-Q4_0 Qwen3.5-2B-Q4_K_M Qwen3.5-2B-Q5_K_M Qwen3.5-2B-Q6_K Qwen3.5-2B-Q8_0 Qwen3.5-2B-IQ4_XS Qwen3.5-2B-IQ4_NL Qwen3-1.7B-Q4_0 Qwen3-1.7B-Q8_0 llama-2-7b.Q4_0; do
  for v in stock nw1rpb4 nw1rpb2; do
    B=/data/m10/llama.cpp-tune/build-$v; [ $v = stock ] && B=build-sm50
    ./bench.sh p3-$f-$v $B -m models/$f.gguf -ngl 99 -fa 1 -p 0 -n 128
  done
done
echo PHASE3_DONE
