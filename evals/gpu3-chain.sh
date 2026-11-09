#!/usr/bin/env bash
# GPU 3 chain: OCR (Qwen3-VL-4B, Qwen3.5-4B+mmproj), retrieval (embeddinggemma; Qwen3-Embedding + reranker).
cd /data/m10; B=/data/m10/llama.cpp-tune/build-best/bin; R=/data/m10/models/roles
PY="docker run --rm --network host --oom-score-adj 1000 -v /data/m10:/data/m10 --entrypoint python3 m10-pyeval:latest"
evals/serve.sh vl4b 3 18113 $B -m /data/models-extra/qwen3-vl-4b/Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf --mmproj /data/models-extra/qwen3-vl-4b/mmproj-F16.gguf -fa on -c 8192 && $PY /data/m10/evals/ocr.py 18113 qwen3-vl-4b-q4kxl
docker rm -f m10-role-vl4b >/dev/null
evals/serve.sh q35v 3 18113 $B -m $R/Qwen3.5-4B-Q4_0-head_q4_0.gguf --mmproj $R/Qwen3.5-4B-mmproj-F16.gguf -fa on -c 8192 && $PY /data/m10/evals/ocr.py 18113 qwen35-4b-q4h-mmproj --no-think
docker rm -f m10-role-q35v >/dev/null
evals/serve.sh embg 3 18111 $B -m $R/embeddinggemma-300M-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 && $PY /data/m10/evals/retrieval.py 18111 embeddinggemma-300m-q8 gemma
docker rm -f m10-role-embg >/dev/null
evals/serve.sh embq3 3 18111 $B -m $R/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 && evals/serve.sh rerank 3 18114 $B -m $R/qwen3-reranker-0.6b-q8_0.gguf --reranking -c 8192 -b 8192 -ub 8192 && $PY /data/m10/evals/retrieval.py 18111 qwen3-emb-0.6b-q8+rerank qwen3 18114
docker rm -f m10-role-embq3 m10-role-rerank >/dev/null
echo GPU3_CHAIN_DONE
