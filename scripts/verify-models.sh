#!/usr/bin/env bash
# Check every downloaded GGUF in /data/m10/models against the sha256 Hugging Face publishes.
cd /data/m10/models
declare -A REPO=( [Qwen3.5-2B]=unsloth/Qwen3.5-2B-GGUF [Qwen3-1.7B]=unsloth/Qwen3-1.7B-GGUF [llama-2-7b]=TheBloke/Llama-2-7B-GGUF )
for f in *.gguf; do
  for k in "${!REPO[@]}"; do [[ $f == $k* ]] && r=${REPO[$k]}; done
  want=$(curl -s "https://huggingface.co/api/models/$r/tree/main" | python3 -c "import json,sys; print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$f'))")
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "OK   $f" || echo "BAD  $f"
done
