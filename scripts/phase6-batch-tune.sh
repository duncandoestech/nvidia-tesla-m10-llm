#!/usr/bin/env bash
# Phase 6: tune the mmvq path for 2-8 tokens per step (batched/parallel decode, speculative verify).
cd /data/m10; source multi.sh
until grep -q PHASE5_DONE logs/phase5.log; do sleep 30; done
VARS="1_2 1_4 2_4 1_8 2_2"
for v in $VARS; do nw=${v%_*}; rb=${v#*_}; name=b-nw${nw}rpb${rb}
  docker run --rm --gpus all -v /data/m10:/data/m10 -w /data/m10/llama.cpp-tune m10-llamacpp:cuda12.9 bash -c \
   "cmake -B build-$name -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS='-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=$nw -DM10_RPBB=$rb' && cmake --build build-$name -j 12 --target llama-batched-bench llama-bench test-backend-ops" > logs/build-$name.log 2>&1 || echo "BUILD FAIL $name"
done
cool(){ for i in $(seq 1 120); do [ $(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits) -le 48 ] && return; sleep 5; done; }
M2=models/self/Qwen3.5-2B-Q4_0-head_q4_0.gguf
for round in 1 2; do for v in base $VARS; do
  B=/data/m10/llama.cpp-tune/build-nw1rpb4/bin; [ $v != base ] && B=/data/m10/llama.cpp-tune/build-b-nw${v%_*}rpb${v#*_}/bin
  cool; echo "== $v r$round"; bbench p6-$v-r$round 0 $B -m $M2 -ngl 99 -fa on -c 8192 -npp 128 -ntg 128 -npl 1,2,3,4,6,8
done; done
for v in $VARS; do echo "== correctness $v"
  docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=1 -v /data/m10:/data/m10 m10-llamacpp:cuda12.9 \
    /data/m10/llama.cpp-tune/build-b-nw${v%_*}rpb${v#*_}/bin/test-backend-ops -b CUDA0 -o MUL_MAT -p "type_b=f32,m=[0-9]+,n=[1-8]," 2>&1 | grep -E "passed|FAIL|ASSERT" | tail -2
done
echo PHASE6_DONE
