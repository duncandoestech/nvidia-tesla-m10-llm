#!/usr/bin/env bash
# Final role-model layout on the M10 host's M10 (one role per GPU). Usage: layout.sh [general|coder|kev|embed|all]
# Docker-published ports bypass firewalld, so other LAN machines reach 8090-8095 directly.
cd /data/m10; T=/data/m10/llama.cpp-tune; R=/data/m10/models/roles; S=evals/serve.sh
general(){ docker rm -f m10-role-gv >/dev/null 2>&1
  $S general 1 8090 $T/build-best/bin -m $R/Qwen3.5-4B-MTP-Q4_0-head_q4_0.gguf --mmproj $R/Qwen3.5-4B-mmproj-F16.gguf \
    -fa on -c 16384 --spec-type draft-mtp --spec-draft-n-max 1 --alias general --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5; }
coder(){ $S coder 2 8091 $T/build-best-graphs/bin -m $R/qwen2.5-coder-7b-instruct-q4_0.gguf -fa on -c 16384 --alias coder --temp 0.2; }
embed(){ $S embed 3 8093 $T/build-best/bin -m $R/embeddinggemma-300M-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 --alias embed
  $S rerank 3 8094 $T/build-best-rr/bin -m $R/qwen3-reranker-0.6b-q8_0.gguf --reranking -c 8192 -b 8192 -ub 8192 --alias rerank; }
kev(){ docker rm -f m10-role-kev >/dev/null 2>&1
  docker run -d --name m10-role-kev --oom-score-adj 1000 --memory 6g --memory-swap 8g --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0 \
    -e KEV_DTYPE=fp32 -e KEV_FUSED=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v /data/m10/kev/Qwen3.5-0.8B-Base:/work/Qwen/Qwen3.5-0.8B-Base:ro -v /data/m10/kev/kev-0.8b:/work/kev-0.8b:ro -w /work -p 0.0.0.0:8095:8095 \
    m10-kev:latest python -m kev.serve --run /work/kev-0.8b --host 0.0.0.0 --port 8095 --device cuda >/dev/null
  for i in $(seq 120); do curl -sf localhost:8095/v1/models >/dev/null 2>&1 && { echo "kev up on :8095"; return; }; sleep 5; done; echo "kev TIMEOUT"; }
case ${1:-all} in all) kev; general; coder; embed;; *) $1;; esac
