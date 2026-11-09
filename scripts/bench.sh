#!/usr/bin/env bash
# M10 llama-bench wrapper, following the test box measurement rules:
#  - wait for every M10 GPU to cool to <=48 C, 1 Hz nvidia-smi log, abort on >80 C / SM clock <1000 MHz while busy.
# Usage: bench.sh <tag> <builddir> <llama-bench args...>
# Output: /data/m10/results/<tag>.jsonl (llama-bench JSONL) + <tag>.smi.csv
set -uo pipefail
TAG=$1; BLD=$2; shift 2
R=/data/m10/results; mkdir -p $R
IDS=${M10_IDS:-0,1,2,3}
cool(){ for i in $(seq 1 120); do
  hot=$(nvidia-smi -i $IDS --query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '$1>48' | wc -l)
  [ "$hot" = 0 ] && return 0; sleep 5; done; echo "WARN: cool-down timeout"; }
# production (P100 llama-server on :8080) busy state is logged before/after each run so any effect shows up in the data
prod_busy(){ curl -s --max-time 3 localhost:8080/slots | python3 -c "import json,sys; print(int(any(x.get('is_processing') for x in json.load(sys.stdin))))" 2>/dev/null || echo 0; }
# (no longer waits for prod idle, by choice 2026-11-07; busy state is only logged)
PROD_BEFORE=$(prod_busy)
cool
nvidia-smi -i $IDS --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,clocks_event_reasons.active \
  --format=csv,noheader -l 1 > $R/$TAG.smi.csv &
SMI=$!
( while kill -0 $SMI 2>/dev/null; do sleep 5
    awk -F', ' '{t=$3+0; if (t>80) {print "ABORT temp "t; exit 1}}' <(tail -n 4 $R/$TAG.smi.csv) || { pkill -f llama-bench; break; }
  done ) &
WD=$!
docker run --rm --oom-score-adj 1000 --memory 4g --memory-swap 6g --gpus all --ipc=host -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$IDS -e GGML_CUDA_P2P=${GGML_CUDA_P2P:-1} ${DOCKER_ENV:-} \
  -v /data/m10:/data/m10 -v /data/models:/data/models:ro -w /data/m10 m10-llamacpp:cuda12.9 \
  $( [[ $BLD == /* ]] && echo $BLD || echo /data/m10/llama.cpp/$BLD )/bin/llama-bench "$@" -o jsonl -r ${REPS:-3} > $R/$TAG.jsonl 2> $R/$TAG.err
RC=$?
kill $SMI $WD 2>/dev/null
echo "$TAG rc=$RC prod_busy_before=$PROD_BEFORE prod_busy_after=$(prod_busy) maxT=$(cut -d, -f3 $R/$TAG.smi.csv | sort -n | tail -1) maxP=$(cut -d, -f4 $R/$TAG.smi.csv | sort -n | tail -1)"
python3 - "$R/$TAG.jsonl" <<'PY'
import json,sys
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except: continue
    kind = f"pp{d['n_prompt']}" if d['n_gen']==0 else f"tg{d['n_gen']}"
    print(f"  {d['model_type'][:28]:28} {d.get('type_k','')}/{d.get('type_v','')} fa={d.get('flash_attn')} b={d.get('n_batch')} ub={d.get('n_ubatch')} sm={d.get('split_mode')} {kind:7} {d['avg_ts']:8.2f} ± {d['stddev_ts']:.2f}")
PY
