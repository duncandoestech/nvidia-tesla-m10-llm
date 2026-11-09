#!/usr/bin/env bash
# Download the small-Qwen quant sweep plus the community reference model into /data/m10/models.
set -u
cd /data/m10/models
HF=https://huggingface.co
get(){ curl -sfL -C - --retry 5 -o "$(basename $1)" "$HF/$1" || echo "FAIL $1"; }
for q in Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0 IQ4_XS IQ4_NL BF16; do get unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-$q.gguf; done
for q in Q4_0 Q8_0; do get unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-$q.gguf; done
get TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
ls -la; echo DONE
