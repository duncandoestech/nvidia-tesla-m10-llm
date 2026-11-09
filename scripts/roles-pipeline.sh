#!/usr/bin/env bash
# One pipeline on a second machine: remaining downloads -> quantize -> copy to the M10 host in priority order (Wi-Fi is ~2.5 MB/s).
set -u
D=~/m10-models; Q=$D/build-cpu/bin/llama-quantize; H=$M10_HOST:/data/m10/models/roles/; mkdir -p $D/self; cd $D
get(){ local repo=$1 f=$2 out=${3:-$2}; mkdir -p "$(dirname "$out")"
  [ -s "$out" ] || curl -sfL --retry 5 -o "$out" "https://huggingface.co/$repo/resolve/main/$f" || { echo "FAIL $f"; return; }
  want=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | python3 -c "import json,sys; print(next((x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$f' and 'lfs' in x), 'nolfs'))")
  [ "$want" = nolfs ] && { echo "OK(small) $out"; return; }
  [ "$want" = "$(sha256sum "$out" | cut -d' ' -f1)" ] && echo "OK $out" || echo "BAD $out"; }
qz(){ [ -s $D/self/$2 ] || $Q $3 $D/$1 $D/self/$2 Q4_0 8 > $D/self/$2.log 2>&1 && echo "quantized $2" || echo "QUANT FAIL $2"; }
cp1(){ rsync -a --partial "$@" && echo "COPIED $(basename "${@: -2:1}")" || echo "COPY FAIL $*"; }
qz Qwen3.5-4B-BF16.gguf Qwen3.5-4B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
cp1 $D/self/Qwen3.5-4B-Q4_0-head_q4_0.gguf $H
get unsloth/Seed-Coder-8B-Instruct-GGUF Seed-Coder-8B-Instruct-BF16.gguf
qz Seed-Coder-8B-Instruct-BF16.gguf Seed-Coder-8B-Instruct-Q4_0.gguf ""
cp1 $D/self/Seed-Coder-8B-Instruct-Q4_0.gguf $H
cp1 $D/qwen2.5-coder-7b-instruct-q4_0.gguf $H
get ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
get unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf Qwen3.5-4B-mmproj-F16.gguf
cp1 $D/qwen3-reranker-0.6b-q8_0.gguf $D/Qwen3.5-4B-mmproj-F16.gguf $D/embeddinggemma-300M-Q8_0.gguf $H
for f in adapter_config.json adapter_model.safetensors head.pt provenance.json result.json tokenizer.json tokenizer_config.json training_config.json; do get jaredpalmer/kev-0.8b $f kev/kev-0.8b/$f; done
for f in config.json merges.txt model.safetensors-00001-of-00001.safetensors model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do get Qwen/Qwen3.5-0.8B-Base $f kev/Qwen3.5-0.8B-Base/$f; done
ssh $M10_HOST 'mkdir -p /data/m10/kev'; cp1 $D/kev/ $M10_HOST:/data/m10/kev/
[ -s $D/kev-image.tar.gz ] || docker save m10-kev:latest | gzip -1 > $D/kev-image.tar.gz; echo "kev image $(du -h $D/kev-image.tar.gz | cut -f1)"
cp1 $D/kev-image.tar.gz $M10_HOST:/data/m10/kev/
ssh $M10_HOST 'gunzip -c /data/m10/kev/kev-image.tar.gz | docker load | tail -1'
qz Qwen3.5-9B-BF16.gguf Qwen3.5-9B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
cp1 $D/self/Qwen3.5-9B-Q4_0-head_q4_0.gguf $H
qz Qwen2.5-Coder-7B-Instruct-F16.gguf Qwen2.5-Coder-7B-Q4_0-head_q4_0.gguf "--output-tensor-type q4_0"
cp1 $D/Qwen3.5-4B-Q4_0.gguf $D/Qwen3.5-4B-IQ4_XS.gguf $D/Qwen3.5-4B-Q8_0.gguf $H
echo PIPELINE_DONE
