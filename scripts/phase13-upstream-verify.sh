#!/usr/bin/env bash
# Phase 13: verify the upstream-style patches (01 table, 02 vmad, 03 MoE no-local-mem) against f477f795c.
cd /data/m10
until grep -q PHASE10_DONE logs/phase10.log 2>/dev/null; do sleep 30; done
for v in up01 upall; do
  [ -d llama.cpp-$v ] || git -C llama.cpp worktree add -q --detach /data/m10/llama.cpp-$v f477f795c
  git -C llama.cpp-$v apply /data/m10/01-cuda-maxwell-mmvq-table.patch
  [ $v = upall ] && git -C llama.cpp-$v apply /data/m10/02-cuda-maxwell-vmad-dp4a.patch /data/m10/03-cuda-mmvq-moe-no-local-mem.patch
  git -C llama.cpp-$v diff --stat | tail -1
  docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-$v m10-llamacpp:cuda12.9 bash -c \
    "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j 12 --target llama-bench llama-batched-bench test-backend-ops" > logs/build-$v.log 2>&1 && echo "built $v" || { echo "BUILD FAIL $v"; grep -iE "error" logs/build-$v.log | head -5; }
done
echo "== spills"; for v in up01 upall; do echo "$v: $(docker run --rm -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 cuobjdump -res-usage /data/m10/llama.cpp-$v/build/bin/libggml-cuda.so 2>/dev/null | grep -A1 mul_mat_vec_q | grep -oE 'STACK:[0-9]+' | sort | uniq -c | tr '\n' ' ')"; done
echo "== correctness"; for v in up01 upall; do for op in "MUL_MAT -p type_b=f32,m=[0-9]+,n=[1-8]," "MUL_MAT_ID"; do
  echo "$v $op: $(docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 /data/m10/llama.cpp-$v/build/bin/test-backend-ops -b CUDA0 -o $op 2>&1 | grep -E 'tests passed|ASSERT|FAIL' | head -2 | tr '\n' ' ')"; done; done
echo "== speed (GPU 0, tg128 + batched)"
M2=models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf
for round in 1 2; do for v in stock up01 tuned-exp upall best-exp; do
  case $v in stock) B=build-sm50;; up01|upall) B=/data/m10/llama.cpp-$v/build;; tuned-exp) B=/data/m10/llama.cpp-tune/build-b-nw1rpb4;; best-exp) B=/data/m10/llama.cpp-tune/build-best;; esac
  M10_IDS=0 REPS=3 ./bench.sh p13-$v-r$round $B -m $M2 -ngl 99 -fa 1 -p 0 -n 128; done; done
source multi.sh
for v in up01 upall; do echo "-- batched $v"; bbench p13-bb-$v 0 /data/m10/llama.cpp-$v/build/bin -m $M2 -ngl 99 -fa on -c 8192 -npp 128 -ntg 128 -npl 1,2,4,8; done
echo PHASE13_DONE
