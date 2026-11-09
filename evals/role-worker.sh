#!/usr/bin/env bash
# Per-GPU worker: wait for a model file, then speed (best vs graphs) + HumanEval (+ optional perplexity).
# Usage: role-worker.sh <gpu> <port> <tag> <model-file> <graphs:0|1> <he-args> [ppl:0|1]
G=$1 PORT=$2 TAG=$3 M=$4 GR=$5 HEARGS=$6 PPL=${7:-0}
cd /data/m10
until [ -s "$M" ]; do sleep 30; done
BEST=/data/m10/llama.cpp-tune/build-best; GRB=/data/m10/llama.cpp-tune/build-best-graphs
echo "== $TAG on GPU $G ($(du -h $M | cut -f1))"
M10_IDS=$G REPS=3 ./bench.sh r1-$TAG-best $BEST -m $M -ngl 99 -fa 1 -p 512 -n 128
[ $GR = 1 ] && M10_IDS=$G REPS=3 ./bench.sh r1-$TAG-graphs $GRB -m $M -ngl 99 -fa 1 -p 512 -n 128
SB=$BEST; [ $GR = 1 ] && SB=$GRB
if [ $PPL = 1 ]; then
  echo "ppl $TAG: $(docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$G -v /data/m10:/data/m10 -w /data/m10 m10-llamacpp:cuda12.9 $BEST/bin/llama-perplexity -m $M -f data/wiki.test.raw -ngl 99 -fa on -c 512 --chunks 60 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+')"
fi
evals/serve.sh $TAG $G $PORT $SB/bin -m $M -fa on -c 4096 && python3 evals/humaneval.py $PORT $TAG 164 $HEARGS
docker rm -f m10-role-$TAG > /dev/null
echo "WORKER_DONE $TAG"
