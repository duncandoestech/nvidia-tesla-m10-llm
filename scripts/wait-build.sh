#!/usr/bin/env bash
# Wait until no cmake/nvcc process is running (max 40 min), then report the build result.
for i in $(seq 1 240); do pgrep -x cmake >/dev/null || pgrep -x nvcc >/dev/null || break; sleep 10; done
tail -3 /data/m10/logs/${1:-build-sm50}.log; ls /data/m10/llama.cpp/${1:-build-sm50}/bin/ 2>&1 | tr '\n' ' '; echo; ls -la /data/m10/models
