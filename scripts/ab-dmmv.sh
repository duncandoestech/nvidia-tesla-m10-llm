#!/usr/bin/env bash
# A/B: float DMMV vs emulated-dp4a MMVQ for token generation on one GM107 (GPU 1), same pre-removal commit.
cd /data/m10
until grep -q PHASE1_DONE logs/phase1.log && grep -q BUILD_DMMV_AB_DONE logs/build-dmmv-ab.log && grep -q "llama-2-7b.Q4_0.gguf" logs/verify.log 2>/dev/null; do sleep 30; done
grep -q "OK   llama-2-7b.Q4_0.gguf" logs/verify.log || { echo "llama-2 checksum bad"; exit 1; }
export M10_IDS=1
for rep in 1 2; do for v in mmvq dmmv; do
  ./bench.sh ab-dmmv-$v-r$rep /data/m10/llama.cpp-predmmv/build-$v -m models/llama-2-7b.Q4_0.gguf -ngl 99 -fa 0 -p 512 -n 128
done; done
echo AB_DMMV_DONE
