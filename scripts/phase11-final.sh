#!/usr/bin/env bash
# Phase 11: confirm recommended configs on the best build.
cd /data/m10
restore_pl(){ sudo -n nvidia-smi -i 0,1,2,3 -pl 53 >/dev/null; echo "power limits restored: $(nvidia-smi -i 0,1,2,3 --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"; }
trap restore_pl EXIT
BEST=/data/m10/llama.cpp-tune/build-best; MQ4=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
NEM=/data/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf
echo "== Nemotron-3.5 30B-A3B Q4, best build, 4 GPUs"
M10_IDS=0,1,2,3 REPS=2 ./bench.sh p11-nemotron-best $BEST -m $NEM -ngl 99 -sm layer -ts 1/1/1/1 -fa 1 -b 4096 -ub 512,2048 -p 2048 -n 128
echo "== recommended MoE serving config: Q4 + MTP 2 + ub 2048 (llama-server), 53 W then 38 W"
for cap in 53 38; do sudo -n nvidia-smi -i 0,1,2,3 -pl $cap >/dev/null; echo "-- cap $cap W"
  ./spec-server-test.sh p11-moe-rec-pl$cap $BEST/bin 0,1,2,3 64 -m $MQ4 -sm layer -ts 1/1/1/1 -b 4096 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2 2>&1 | tail -4
done
restore_pl; trap - EXIT
echo PHASE11_DONE
