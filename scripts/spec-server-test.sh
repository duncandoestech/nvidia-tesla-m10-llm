#!/usr/bin/env bash
# Speculative decoding through llama-server (handles recurrent-state checkpoints for Qwen3.5 hybrids).
# Usage: spec-server-test.sh <tag> <bindir> <gpus> <n_predict> <server args...>
TAG=$1 B=$2 G=$3 N=$4; shift 4
PORT=18090; NAME=m10-spec-$$
docker run -d --rm --name $NAME --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$G -p 127.0.0.1:$PORT:$PORT \
  -v /data/m10:/data/m10 -v /data/models:/data/models:ro -w /data/m10 m10-llamacpp:cuda12.9 \
  $B/llama-server --host 0.0.0.0 --port $PORT -ngl 99 -fa on -c 8192 --parallel 1 "$@" > /dev/null
for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && break; sleep 2; done
python3 - $PORT $N $TAG <<'PY'
import json,sys,urllib.request
port,n,tag=sys.argv[1],int(sys.argv[2]),sys.argv[3]
prompts=["Write a Python function that checks whether a number is prime, with a docstring and three test cases.",
         "Explain how a CPU cache works to a high school student, in about 200 words.",
         "List ten countries in Europe with their capitals, one per line, formatted as 'Country: Capital'."]
tot_tok=tot_t=dn=da=0
for p in prompts:
    req=urllib.request.Request(f"http://localhost:{port}/completion",data=json.dumps({"prompt":p,"n_predict":n,"temperature":0,"cache_prompt":False}).encode(),headers={"Content-Type":"application/json"})
    t=json.load(urllib.request.urlopen(req,timeout=600))["timings"]
    tot_tok+=t["predicted_n"]; tot_t+=t["predicted_ms"]; dn+=t.get("draft_n",0); da+=t.get("draft_n_accepted",0)
    print(f"  {tag} {t['predicted_per_second']:.2f} t/s  draft {t.get('draft_n',0)} accepted {t.get('draft_n_accepted',0)}")
print(f"  {tag} TOTAL {1000*tot_tok/tot_t:.2f} t/s  acceptance {da/max(dn,1):.2%} ({da}/{dn})")
PY
docker logs $NAME 2>&1 | grep -iE "error|fail" | head -3
docker stop $NAME > /dev/null
