#!/usr/bin/env bash
# Phase 1: wait for downloads, re-fetch Q4_0, verify checksums, then a quant sweep on one M10 GPU (GPU 0).
set -u
cd /data/m10
while pgrep -f download-models.s[h] >/dev/null; do sleep 20; done
bash /data/m10/redl.sh
./verify-models.sh | tee logs/verify.log
export M10_IDS=0
for f in Qwen3.5-2B-Q4_0 Qwen3.5-2B-Q4_K_M Qwen3.5-2B-Q5_K_M Qwen3.5-2B-Q6_K Qwen3.5-2B-Q8_0 Qwen3.5-2B-IQ4_XS Qwen3.5-2B-IQ4_NL Qwen3.5-2B-BF16 Qwen3-1.7B-Q4_0 Qwen3-1.7B-Q8_0; do
  grep -q "OK   $f.gguf" logs/verify.log || { echo "skip $f (checksum)"; continue; }
  ./bench.sh p1-$f build-sm50 -m models/$f.gguf -ngl 99 -fa 1 -p 512 -n 128
done
# community reference (llama.cpp discussion #15013 format) for comparison with Tesla M40 numbers
grep -q "OK   llama-2-7b.Q4_0.gguf" logs/verify.log && ./bench.sh p1-ref-llama2-7b-q4_0 build-sm50 -m models/llama-2-7b.Q4_0.gguf -ngl 99 -fa 0,1
echo PHASE1_DONE
