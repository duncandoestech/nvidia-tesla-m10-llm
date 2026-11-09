#!/usr/bin/env bash
# Qwen3.5-4B with MTP heads: download BF16 on a second machine, quantize Q4_0 + Q4_0 head, copy after the main pipeline.
D=~/m10-models; R=unsloth/Qwen3.5-4B-MTP-GGUF; f=Qwen3.5-4B-BF16.gguf; out=$D/mtp/Qwen3.5-4B-MTP-BF16.gguf; mkdir -p $D/mtp
[ -s $out ] || curl -sfL --retry 5 -o $out https://huggingface.co/$R/resolve/main/$f
want=$(curl -s https://huggingface.co/api/models/$R/tree/main | python3 -c "import json,sys; print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$f'))")
[ "$want" = "$(sha256sum $out | cut -d' ' -f1)" ] && echo "OK mtp bf16" || { echo "BAD mtp bf16"; exit 1; }
$D/build-cpu/bin/llama-quantize --output-tensor-type q4_0 $out $D/mtp/Qwen3.5-4B-MTP-Q4_0-head_q4_0.gguf Q4_0 8 > $D/mtp/quant.log 2>&1 && echo "quantized mtp" || { echo "QUANT FAIL"; tail -3 $D/mtp/quant.log; exit 1; }
grep -iE "nextn|mtp" $D/mtp/quant.log | head -3
until grep -q PIPELINE_DONE $D/pipeline.log; do sleep 30; done
rsync -a --partial $D/mtp/Qwen3.5-4B-MTP-Q4_0-head_q4_0.gguf $M10_HOST:/data/m10/models/roles/ && echo "COPIED mtp 4b"
echo MTP4B_DONE
