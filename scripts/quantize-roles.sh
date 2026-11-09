#!/usr/bin/env bash
# Self-quantize role models on a second machine (CPU) once downloads finish, then copy results to the M10 host.
D=~/m10-models; Q=$D/build-cpu/bin/llama-quantize
until grep -q DL_ROLES_DONE $D/download.log; do sleep 30; done
mkdir -p $D/self
q(){ [ -s $D/self/$2 ] || $Q $3 $D/$1 $D/self/$2 Q4_0 8 > $D/self/$2.log 2>&1 && echo "quantized $2 ($(du -h $D/self/$2 | cut -f1))" || echo "QUANT FAIL $2"; }
q Qwen3.5-4B-BF16.gguf Qwen3.5-4B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
q Qwen3.5-9B-BF16.gguf Qwen3.5-9B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
q Qwen2.5-Coder-7B-Instruct-F16.gguf Qwen2.5-Coder-7B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
rsync -a --info=stats1 $D/self/*.gguf $M10_HOST:/data/m10/models/roles/ | tail -2
echo QUANT_ROLES_DONE
