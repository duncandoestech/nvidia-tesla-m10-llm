#!/usr/bin/env bash
# Round 2: GPU2 MTP sweep (4B), GPU3 reranker on rr build over embeddinggemma, GPU1 9B bench.
cd /data/m10; B=/data/m10/llama.cpp-tune; R=models/roles
PY="docker run --rm --network host --oom-score-adj 1000 -v /data/m10:/data/m10 --entrypoint python3 m10-pyeval:latest"
gpu2(){ M=$R/Qwen3.5-4B-MTP-Q4_0-head_q4_0.gguf
  evals/serve.sh mtp 2 18102 $B/build-best/bin -m $M -fa on -c 4096 && $PY /data/m10/evals/mtp_speed.py 18102 q35-4b-nomtp
  for k in 1 2 3; do evals/serve.sh mtp 2 18102 $B/build-best/bin -m $M -fa on -c 4096 --spec-type draft-mtp --spec-draft-n-max $k && $PY /data/m10/evals/mtp_speed.py 18102 q35-4b-mtp$k; done
  docker rm -f m10-role-mtp >/dev/null; echo GPU2_DONE; }
gpu3(){ evals/serve.sh embg 3 18111 $B/build-best/bin -m $R/embeddinggemma-300M-Q8_0.gguf --embedding -c 8192 -b 8192 -ub 8192 && \
  $PY /data/m10/evals/retrieval.py 18111 embeddinggemma-300m-q8+rerank-rr gemma 18114
  docker rm -f m10-role-embg m10-role-rerank >/dev/null; echo GPU3_DONE; }
gpu1(){ M10_IDS=1 ./bench.sh role-9b-q4_0 $B/build-best-graphs -m $R/Qwen3.5-9B-Q4_0.gguf -p 512 -n 128 -fa 1; echo GPU1_DONE; }
gpu1 > logs/r2-gpu1.log 2>&1 & gpu2 > logs/r2-gpu2.log 2>&1 & gpu3 > logs/r2-gpu3.log 2>&1 &
wait $(jobs -p); echo R2_23_DONE >> logs/r2-gpu3.log
