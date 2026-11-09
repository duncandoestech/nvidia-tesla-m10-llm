#!/usr/bin/env bash
# Kernel-time breakdown with nvprof (nsys 2026 does not trace Maxwell). Usage: profile-tg.sh <tag> <builddir> <model> [extra llama-bench args]
TAG=$1; BLD=$2; M=$3; shift 3
docker run --rm --gpus all --privileged -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=${M10_IDS:-0} -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 \
  nvprof --print-gpu-summary --csv --log-file /data/m10/results/$TAG.nvprof.csv \
  $( [[ $BLD == /* ]] && echo $BLD || echo /data/m10/llama.cpp/$BLD )/bin/llama-bench -m $M -ngl 99 -r 1 "$@" > /dev/null 2>&1
python3 - /data/m10/results/$TAG.nvprof.csv <<'PY'
import csv,sys
rows=[r for r in csv.reader(open(sys.argv[1])) if len(r)>7 and r[0] in ("GPU activities",)]
for r in rows[:14]: print(f"{float(r[1]):5.1f}% {r[3]:>7} calls {r[4]:>9}  {r[7][:100]}")
PY
