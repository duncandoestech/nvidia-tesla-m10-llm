# Tesla M10 LLM

An ongoing effort to optimize the NVIDIA Tesla M10 for local LLM inference. Originally designed for VDI, this often-overlooked Maxwell card packs 32 GB of VRAM (4x 8 GB). Through experimentation, unconventional configurations and performance tuning, this project pushes aging hardware beyond its intended purpose.

The M10 is one board with four GM107 Maxwell GPUs, each with 8 GB and ~83 GB/s, compute capability 5.0. CUDA 13 dropped it, and almost nobody has published inference numbers for it. This repo has a full set, a small llama.cpp patch that makes token generation on Maxwell **up to 1.8x faster** than current llama.cpp, and a setup that turns the four GPUs into a team of specialist models for a larger AI agent.

## Headline numbers (one GM107 unless noted)

| Model | stock llama.cpp | this repo's best build |
|---|---|---|
| Qwen3.5-2B Q4_0 (Q4_0 output head) | 23.4 t/s | **41.2 t/s** |
| Llama-2-7B Q4_0 (community reference) | 8.1 t/s | **13.4 t/s**, **15.1 t/s** with CUDA graphs |
| Qwen3.6-35B-A3B Q4_K_XL, all 4 GPUs | 11.1 t/s | **15.8 t/s**, **17.7 t/s** with MTP |
| Qwen3.8-27B Q5_K_XL dense, all 4 GPUs, `-sm tensor` | | 6.65 t/s |
| Four independent Qwen3.5-2B copies (one per GPU) | | ~150 t/s total |

See [RESULTS.md](RESULTS.md) for every table, and [research/RESEARCH.md](research/RESEARCH.md) for what others have published.

## What makes the difference

1. **CUDA 12.x only.** CUDA 13 dropped sm_50. Driver branch 580 is the last that supports Maxwell.
2. **mmvq launch shape:** 1 warp and 4 rows per block (stock: 4 warps, 1 row) for both the 1-token and 2–8-token paths. GM107 has no `dp4a`, so the quantized mat-vec kernel is compute-bound. Fewer warps avoid the shared-memory reduction, and more rows reuse the activation vector.
3. **`vmad` dp4a:** PTX `vmad.s32.s32.s32` byte-selects compile to a native `VMAD.S8.S8` on sm_50 (+11 % on Q4_0). Idea from shinbunbun/llama-cpp-p100-patches #01.
4. **MoE kernel spill fix:** `mul_mat_vec_q_moe` indexed its per-row sums with `threadIdx.x`, which spills to local memory on sm_50 (18 kernels). An unrolled select removes it (+11–18 % on batched MoE decode). Same fix as ik_llama.cpp #2721.
5. **CUDA graphs below Volta** (`-DM10_GRAPHS`): mainline disables them on pre-Volta GPUs. Enabling them gives +16 % on Llama-2-7B with byte-identical output, and no change on hybrid Qwen3.5.
6. **Model choices:** simple quants (Q4_0, IQ4_XS, Q8_0) beat K-quants. A Q4_0 output head gives +15 % on Qwen3.5's 248k-vocab models (perplexity +1.3 %). For MoE prompts use `-b 4096 -ub 2048`.

The patch is [patches/m10-mmvq-tunables.patch](patches/m10-mmvq-tunables.patch) against llama.cpp `f477f795c` (2026-11-08). Every change is behind a compile-time define, so the default build is unchanged:

```
-DCMAKE_CUDA_ARCHITECTURES=50 \
-DCMAKE_CUDA_FLAGS='-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4 -DM10_VMAD -DM10_MOE_NOSPILL'   # add -DM10_GRAPHS for dense models
```

## Four GPUs, four jobs

Each GPU runs one small model as its own OpenAI-compatible server (`evals/layout.sh`):

| GPU | Role | Model | Speed | Quality |
|---|---|---|---|---|
| 0 | decisions | Kev-0.8B ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)), PyTorch FP32 | ~1 s/request | classifier / scorer |
| 1 | chat + vision | Qwen3.5-4B Q4_0 + mmproj, MTP draft n=1 | 21.8 t/s | HumanEval 76.8 %, OCR 24/24 |
| 2 | code | Qwen2.5-Coder-7B-Instruct Q4_0, CUDA graphs | 12.9 t/s | HumanEval 89.0 % |
| 3 | search | embeddinggemma-300M Q8_0 (+ Qwen3-Reranker-0.6B) | ~7 docs/s | SciFact nDCG@10 0.789 |

A larger agent ([Hermes Agent](https://github.com/NousResearch/hermes-agent) on Qwen3.8-27B, on other GPUs) delegates work to them with plain HTTP calls. In real tasks (`results/delegation/`) the coder wrote a correct log parser after two fix rounds, the vision model transcribed build commands from a web-page screenshot exactly, Kev triaged 30 GitHub issues in 46 s, and a four-lane job with every GPU busy at once ran 1.98x faster than the lanes one after another. The small models keep numbers exact but sometimes attach facts to the wrong source or fill gaps with guesses, so the agent checks their output.

## Layout

- `RESULTS.md`: all measurements, summary first
- `research/`: published data and sources. `hermes-*.md` are raw research reports from an AI research agent; unverified items are marked.
- `patches/`: the experiment patch (`m10-mmvq-tunables.patch`) and clean upstream-style patches (`upstream/`, with an explanation of each)
- `scripts/`: build, benchmark and sweep scripts (`bench.sh` is the harness: cool-down to 48 °C, 1 Hz nvidia-smi log, llama-bench JSONL)
- `evals/`: role-model serving (`serve.sh`, `layout.sh`) and quality evals (HumanEval in a no-network sandbox, SciFact retrieval, synthetic OCR, MTP speed)
- `experiments/dp4a/`: SASS comparison of dp4a emulations on sm_50
- `results/`: raw llama-bench JSONL, nvidia-smi logs, eval outputs (`roles/`), agent delegation reports (`delegation/`)
- `logs/`: phase logs

## Reproducing

The scripts assume two local Docker images and a few paths; adjust them to your setup.

- `m10-llamacpp:cuda12.9`: Ubuntu 22.04 with the CUDA 12.9 toolkit and gcc 11 (for example `nvidia/cuda:12.9.1-devel-ubuntu22.04` plus `cmake`, `git`, `build-essential`). llama.cpp is built inside it with `scripts/build-cuda.sh`.
- `m10-pyeval:latest`: any Python 3.11+ image with `numpy` and `Pillow`, used for the evals and as the HumanEval sandbox (run with `--network none --read-only`).
- `m10-kev:latest`: built from `scripts/kev.Dockerfile`.
- Paths: work dir `/data/m10`, models under `/data/models` and `/data/m10/models/roles`. Scripts that copy models use `$M10_HOST` for the GPU machine.
- `evals/data/` holds HumanEval (OpenAI, MIT). The retrieval eval uses BEIR SciFact, which is CC BY-NC 2.0 and not included: run `evals/get_scifact.sh` to download it (checksum-verified, the same files as the published results).

## Test system

Tesla M10 on an x8 upstream link (its four GPUs report no P2P support to each other: `nvidia-smi topo -p2p r` = GNS) in an X79 desktop (i7-3930K, 16 GB RAM), case fans at 100 %, next to two P100s running other work (the big agent's model). Ubuntu 22.04 container with CUDA 12.9 and gcc 11, driver 580.178.04.
