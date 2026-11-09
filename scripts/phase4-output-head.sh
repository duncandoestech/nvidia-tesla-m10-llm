#!/usr/bin/env bash
# Phase 4: output-head (lm_head) type for Qwen3.5-2B Q4_0 on one GM107. Self-quantized from BF16 so only the head type differs.
set -u
cd /data/m10
docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
  "cmake --build build-nw1rpb4 -j 12 --target llama-quantize llama-perplexity" > logs/build-quantize.log 2>&1 || { echo BUILD_FAIL; exit 1; }
mkdir -p models/self data
[ -s data/wiki.test.raw ] || { curl -sfL -o data/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && (cd data && python3 -c "import zipfile; zipfile.ZipFile('wikitext-2-raw-v1.zip').extractall('.')") && cp data/wikitext-2-raw/wiki.test.raw data/; }
ls -la data/wiki.test.raw || exit 1
Q="docker run --rm --gpus all -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 /data/m10/llama.cpp-tune/build-nw1rpb4/bin/llama-quantize"
for h in q6_K q4_0 q8_0; do
  f=models/self/Qwen3.5-2B-Q4_0-head_$h.gguf
  [ -s $f ] || $Q --output-tensor-type $h /data/m10/models/Qwen3.5-2B-BF16.gguf /data/m10/$f Q4_0 12 > logs/quant-head_$h.log 2>&1 || echo "QUANT FAIL $h"
  ls -la $f
done
export M10_IDS=0 REPS=3
for round in 1 2; do for h in q6_K q4_0 q8_0; do
  ./bench.sh p4-head_$h-r$round /data/m10/llama.cpp-tune/build-nw1rpb4 -m models/self/Qwen3.5-2B-Q4_0-head_$h.gguf -ngl 99 -fa 1 -p 512 -n 128
done; done
for h in q6_K q4_0 q8_0; do
  f=models/self/Qwen3.5-2B-Q4_0-head_$h.gguf; [ $h = BF16 ] && f=models/Qwen3.5-2B-BF16.gguf
  echo "== ppl $h"
  docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -v /data/m10:/data/m10 -w /data/m10 m10-llamacpp:cuda12.9 \
    /data/m10/llama.cpp-tune/build-nw1rpb4/bin/llama-perplexity -m $f -f data/wiki.test.raw -ngl 99 -fa on -c 512 --chunks 150 2>&1 | grep -E "Final estimate|error" | tail -2
done
echo PHASE4_DONE
