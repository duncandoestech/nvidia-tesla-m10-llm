#!/usr/bin/env bash
# Start a llama-server container for one model. Usage: serve.sh <name> <gpus> <port> <bindir> <server args...>
# Stops an existing container of the same name first. Waits up to 10 min for /health.
NAME=m10-role-$1 G=$2 PORT=$3 B=$4; shift 4
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --oom-score-adj 1000 --memory 4g --memory-swap 6g --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$G -p 0.0.0.0:$PORT:$PORT \
  -v /data/m10:/data/m10 -v /data/models:/data/models:ro -v /data/models-extra:/data/models-extra:ro -w /data/m10 \
  m10-llamacpp:cuda12.9 $B/llama-server --host 0.0.0.0 --port $PORT -ngl 99 --jinja --cache-ram 0 "$@" > /dev/null
for i in $(seq 1 300); do c=$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health); [ "$c" = 200 ] && { echo "$NAME up on :$PORT (gpu $G)"; exit 0; }
  docker ps -q -f name=$NAME | grep -q . || { echo "$NAME DIED"; docker logs $NAME 2>&1 | tail -5; exit 1; }; sleep 2; done
echo "$NAME TIMEOUT"; exit 1
