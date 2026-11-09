#!/usr/bin/env bash
# mmvq launch-shape sweep (single-token generation path) on one GM107 (GPU 2), interleaved, 2 rounds.
cd /data/m10
export M10_IDS=2 REPS=5
M=models/Qwen3.5-2B-Q4_K_M.gguf
for round in 1 2; do
  for v in stock nw1rpb1 nw2rpb1 nw8rpb1 nw4rpb2 nw2rpb2 nw1rpb2; do
    B=/data/m10/llama.cpp-tune/build-$v; [ $v = stock ] && B=build-sm50
    ./bench.sh tune-$v-r$round $B -m $M -ngl 99 -fa 1 -p 0 -n 128
  done
done
echo TUNE_SWEEP_DONE
