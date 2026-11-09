#!/usr/bin/env bash
# Download role-model candidates to a second machine (fast link), verify sha256, then copy to the M10 host /data/m10/models/roles.
set -u
D=~/m10-models; mkdir -p $D; cd $D
get(){ local repo=$1 f=$2
  [ -s "$f" ] || curl -sfL --retry 5 -o "$f" "https://huggingface.co/$repo/resolve/main/$f" || { echo "FAIL $f"; return; }
  want=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | python3 -c "import json,sys; print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$f'))")
  [ "$want" = "$(sha256sum "$f" | cut -d' ' -f1)" ] && echo "OK $f" || echo "BAD $f"; }
for q in Q4_0 IQ4_XS Q8_0 BF16; do get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-$q.gguf; done
for q in Q4_0 IQ4_XS BF16; do get unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-$q.gguf; done
get Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_0.gguf
get unsloth/Qwen2.5-Coder-7B-Instruct-GGUF Qwen2.5-Coder-7B-Instruct-F16.gguf
get Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf
get unsloth/embeddinggemma-300m-GGUF embeddinggemma-300M-Q8_0.gguf
echo "== copy to the M10 host"
ssh $M10_HOST 'mkdir -p /data/m10/models/roles' && rsync -a --info=stats1 --exclude="*BF16*" --exclude="*F16*" --exclude="*f16*" $D/*.gguf $M10_HOST:/data/m10/models/roles/ | tail -3
echo DL_ROLES_DONE
