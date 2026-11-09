#!/usr/bin/env bash
# Helpers for multi-GPU M10 tests. Source this file.
# run4 <tag> <bindir> <llama-bench args...>: one llama-bench per M10 GPU at the same time; prints per-GPU and summed tg/pp.
run4(){ local TAG=$1 B=$2; shift 2; local R=/data/m10/results
  local pids=""
  for g in 0 1 2 3; do
    docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$g -v /data/m10:/data/m10 -v /data/models:/data/models:ro -w /data/m10 \
      m10-llamacpp:cuda12.9 $B/llama-bench "$@" -o jsonl > $R/$TAG-gpu$g.jsonl 2> $R/$TAG-gpu$g.err &
    pids="$pids $!"
  done; wait $pids   # only our four (a bare wait would also wait on a caller's background nvidia-smi logger)
  python3 - $R/$TAG <<'PY'
import json,sys,glob
tot={}
for f in sorted(glob.glob(sys.argv[1]+"-gpu*.jsonl")):
    for l in open(f):
        try: d=json.loads(l)
        except: continue
        k=f"pp{d['n_prompt']}" if d['n_gen']==0 else f"tg{d['n_gen']}"
        tot.setdefault(k,[]).append(d['avg_ts'])
for k,v in tot.items(): print(f"  {sys.argv[1].split('/')[-1]} {k}: per-GPU {[round(x,2) for x in v]} sum {sum(v):.2f}")
if not tot: print("  NO RESULTS", sys.argv[1])
PY
}
# bbench <tag> <gpu-list> <bindir> <args...>: llama-batched-bench, keeps the table
bbench(){ local TAG=$1 G=$2 B=$3; shift 3
  docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=$G -e GGML_CUDA_P2P=1 -v /data/m10:/data/m10 -v /data/models:/data/models:ro -w /data/m10 \
    m10-llamacpp:cuda12.9 $B/llama-batched-bench "$@" > /data/m10/results/$TAG.txt 2> /data/m10/results/$TAG.err
  grep -E "^\|" /data/m10/results/$TAG.txt || { echo "  NO RESULTS $TAG"; tail -3 /data/m10/results/$TAG.err; }
}
