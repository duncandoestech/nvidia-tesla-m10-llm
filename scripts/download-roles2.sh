#!/usr/bin/env bash
# Batch 2: Seed-Coder, reranker, Qwen3.5-4B mmproj, Kev-0.8B + its base. Download on a second machine, verify, quantize, copy.
set -u
D=~/m10-models; cd $D
until grep -q QUANT_ROLES_DONE $D/quantize.log 2>/dev/null; do sleep 20; done
get(){ local repo=$1 f=$2 out=${3:-$2}
  mkdir -p "$(dirname "$out")"
  [ -s "$out" ] || curl -sfL --retry 5 -o "$out" "https://huggingface.co/$repo/resolve/main/$f" || { echo "FAIL $f"; return; }
  want=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | python3 -c "import json,sys; print(next((x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$f' and 'lfs' in x), 'nolfs'))")
  [ "$want" = nolfs ] && { echo "OK(small) $out"; return; }
  [ "$want" = "$(sha256sum "$out" | cut -d' ' -f1)" ] && echo "OK $out" || echo "BAD $out"; }
for q in IQ4_XS BF16; do get unsloth/Seed-Coder-8B-Instruct-GGUF Seed-Coder-8B-Instruct-$q.gguf; done
get ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
get unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf Qwen3.5-4B-mmproj-F16.gguf
for f in adapter_config.json adapter_model.safetensors head.pt provenance.json result.json tokenizer.json tokenizer_config.json training_config.json; do get jaredpalmer/kev-0.8b $f kev/kev-0.8b/$f; done
for f in config.json merges.txt model.safetensors-00001-of-00001.safetensors model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do get Qwen/Qwen3.5-0.8B-Base $f kev/Qwen3.5-0.8B-Base/$f; done
$D/build-cpu/bin/llama-quantize $D/Seed-Coder-8B-Instruct-BF16.gguf $D/self/Seed-Coder-8B-Instruct-Q4_0.gguf Q4_0 8 > $D/self/seed-q4_0.log 2>&1 && echo "quantized Seed-Coder Q4_0" || echo "QUANT FAIL seed"
rsync -a --info=stats1 $D/Seed-Coder-8B-Instruct-IQ4_XS.gguf $D/qwen3-reranker-0.6b-q8_0.gguf $D/Qwen3.5-4B-mmproj-F16.gguf $D/self/Seed-Coder-8B-Instruct-Q4_0.gguf $M10_HOST:/data/m10/models/roles/ | tail -1
ssh $M10_HOST 'mkdir -p /data/m10/kev' && rsync -a $D/kev/ $M10_HOST:/data/m10/kev/ && echo "kev files copied"
echo DL2_DONE
