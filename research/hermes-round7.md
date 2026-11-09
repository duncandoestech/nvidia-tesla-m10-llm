# M10 / Maxwell LLM inference — research round 7

Delivered: /tmp/m10-research-round7.md
Focus: everything needed to benchmark ik_llama.cpp against a tuned mainline llama.cpp on the Tesla M10 — build, `-sm graph`, architecture support, and mainline expert-parallel status.
Rules followed: research only; one file; no other hosts/commands. All quotes verbatim from pages fetched in full this round (source URLs inline); [SNIPPET] = rests on search-result text only.
ik_llama.cpp state at fetch time: **branch `main`, HEAD `8a9bf106ed25e4e6d63b51e48a459d0cbbb1e613`** ("Do not diosable FA with LoRA adaptors (#2867)", ikawrakow, 2026-11-08T11:23:20Z) — https://api.github.com/repos/ikawrakow/ik_llama.cpp/commits?per_page=1

---

## 1. Task 1 — exact current build instructions for ik_llama.cpp + CUDA for sm_50

### 1.1 The two canonical sources

**(a) README quickstart** (https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/README.md):

```
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

plus: "If you use Docker: `ghcr.io/ikawrakow/ik-llama-cpp:cu12-swap` … `cu12-server` … `cu12-full`"; build details in `docs/build.md`.

**(b) `docs/pascal-volta.md`** (https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/docs/pascal-volta.md) — "This page lists what the CUDA backend does on NVIDIA Pascal and Volta GPUs: which kernels run, what falls back to the CPU, and what fails. It describes commit `3dbc2c9f8` (2026-10-13)." Its Build section, verbatim:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build build --config Release
```

- "Use the architectures of the GPUs you have."
- "Without `CMAKE_CUDA_ARCHITECTURES`, the build uses `native` when `GGML_NATIVE` is on (the default unless cross-compiling), the CUDA toolkit is 11.6 or newer, and CMake is 3.24 or newer. Otherwise it uses `50;61;70;75;80`, or `60;61;70;75;80` with `GGML_CUDA_F16`. The first list has no `60`, so for a P100 set the list."
- "**CUDA 13.0 removed offline compilation for Maxwell, Pascal and Volta. Use a CUDA 12.x toolkit. NVIDIA recommends 12.9 and the 580 driver branch, which is the last branch that supports these GPUs.** The P100 and Titan V tests for this page used CUDA 12.4."
- "`GGML_CUDA_FA_ALL_QUANTS` (default OFF) adds vec FA kernels for more KV types. `GGML_IQK_FA_ALL_QUANTS` (default ON) adds q4_0, q4_1 and iq4_nl to the CPU FA kernel."

### 1.2 Recommended command for the M10 (sm_50) — synthesis of (a)+(b)

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50
cmake --build build --config Release
```

- Explicit `CMAKE_CUDA_ARCHITECTURES=50` matters if you build on a Turing-or-newer box (native would pick that box's arch, not 50). On a box with no native detection (CUDA < 11.6 or CMake < 3.24), the default list `50;61;70;75;80` already includes 50 — but don't rely on it.
- **CUDA 12.x toolkit is required** for an sm_50 build going forward (CUDA 13 removed Maxwell offline compilation). The ik team's Pascal/Volta tests used CUDA 12.4; NVIDIA's recommendation quoted above is 12.9 + 580 driver branch (for Pascal/Volta).
- **Host caveat (from our earlier rounds, not from the ik docs):** the M10 host is CUDA 11-locked, last driver 550.54.10. The ik docs do not state a minimum CUDA version for building; a CUDA 11.x toolkit still compiles sm_50 (it is in the default list above), so a CUDA 11 build remains possible on that host — but that is outside what the ik docs' Pascal/Volta page tested. Decision point before the benchmark.
- README positioning of old GPUs: "The only fully functional and high performance compute backends are the CPU (AVX2 or better, ARM_NEON or better) and CUDA (Turing or newer, **though recently there has been progress in catching up Pascal and Volta — see docs/pascal-volta.md**)." Note: the doc covers sm_60/61/70; Maxwell (sm_50) runs the pre-Volta code paths and is not in the doc's tables (the M10-specific evidence is mb8565's PRs, §2.6).

### 1.3 GGML_CUDA options relevant to Pascal/Maxwell (from `docs/build.md` + `docs/pascal-volta.md`)

| option / env | default | note (source) |
|---|---|---|
| `-DGGML_CUDA=ON` | off | "Build with CUDA support. By default it builds to native CUDA." (build.md) |
| `-DCMAKE_CUDA_ARCHITECTURES=…` | native / `50;61;70;75;80` | see 1.1 (pascal-volta.md) |
| `-DGGML_NCCL=OFF` | auto | "To disable usage of NCCL." (build.md); auto-detect + runtime fallback described in §2.2 (#1080) |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | OFF | "adds vec FA kernels for more KV types. Builds with this option were not checked." (pascal-volta.md) |
| `GGML_IQK_FA_ALL_QUANTS` | ON | CPU FA kernel gains q4_0/q4_1/iq4_nl (pascal-volta.md) |
| `-DGGML_MAX_SRC=N` | - | "The maximum number of GPUs." (build.md) |
| `-DGGML_MAX_CONTEXTS=2048` | - | only for Thireus-suite quants (build.md) |
| `-DGGML_CUDA_USE_GRAPHS=ON` | (build.md Windows example) | CUDA *graph capture* toggle — note: "CUDA graphs are not used below Ampere (`cc < CC_AMPERE` in `ggml-cuda.cu`)" (pascal-volta.md), so on the M10 capture is off regardless; this is unrelated to split-mode `graph` |
| `GGML_CUDA_NO_PINNED` / `GGML_CUDA_NO_PINNED_WEIGHTS` | - | env vars; the `_WEIGHTS` form keeps CPU-resident weights mmapped (hybrid CPU/GPU MoE) (build.md) |
| `CUDA_VISIBLE_DEVICES` | - | per-die pinning for the 4-die M10 (build.md) |

### 1.4 What sm_50-class GPUs actually do (pascal-volta.md, commit 3dbc2c9f8)

- "With all layers offloaded (`-ngl 99`), the matrix multiplications run on the GPU. Flash attention (FA) is on by default (`-fa on`). On these GPUs it runs on the GPU only for some models, KV cache types and batch sizes. Otherwise the FA step runs on the CPU."
- FA placement table rows that matter for the benchmark models (head size 256 = Qwen3.5/Qwen3.6 family): "**Gemma 1, Gemma 2 and 3 up to 12B, Qwen3.5, Qwen3.6, Qwen3-Next, Falcon3 (256) | GPU (P100) | GPU (P40, GTX 10xx) | GPU (V100, Titan V)**". Gemma-4 (512/256): GPU on Pascal, "partly CPU" on V100 (global layers). MLA models (576/512): "partly CPU" everywhere on Pascal/Volta.
- Pascal kernel paths: decode (n_q ≤ 8) = `vec_f32` (sm_60 uses vec_f32, not the fp16 vec kernel — #2144; sm_61 always f32); prompt (n_q > 8) at D=256 = `tile_f32` (**#2580** — the M10 PR, verified on M10 in its body: f16 pp2048 39.3→47.3 t/s, pp4096 32.6→44.1, q8_0 pp4096 13.7→44.0).
- KV cache: "Use an f16 or q8_0 KV cache. Other types can move FA to the CPU, also in the 'GPU' rows. With q5_0, q5_1, bf16 or f32, the program then stops … **q8_KV crashes on these GPUs.**" At D=256, vec_f32 (decode) runs on GPU only for f16 and q8_0; "The tile kernels (n_q > 8 …) … convert K and V to f16 before the launch. So f32, f16, bf16, q8_0, q6_0, q5_0, q5_1, q4_0, q4_1 and iq4_nl run on the GPU there, in any pair."
- "If the CPU cannot run the KV cache type either, the program stops." (q4_0/q4_1/iq4_nl on CPU FA need `GGML_IQK_FA_ALL_QUANTS`, ON by default.)
- "-fa 0 turns FA off. Attention then runs as matrix multiplications and a softmax, which the GPU runs at every head size. A quantized V cache needs FA, so use -fa 0 with an f16 cache."
- **MUL_MAT:** sm_60: n ≤ 8 → MMVQ; n > 8 → "dequantize to f32 + cuBLAS SGEMM" — "`ggml_cuda_should_use_mmq` returns false below cc 610 (`MIN_CC_DP4A`), so sm_60 never takes MMQ. `GGML_CUDA_FORCE_MMQ` does not change this, because this check comes first." (sm_50 has no dp4a either; it is not listed in the table — flagged as doc gap, behavior expected same class.)
- "The code uses fp16 cuBLAS (`CUBLAS_COMPUTE_16F`) only at cc >= 700 and with `GGML_PREC_DEFAULT`. On Pascal, F16 weights and quantized weights that reach cuBLAS are converted to f32 and use SGEMM."
- "With several GPUs and `-sm graph`, the MLA models and Gemma-4 need FA."

### 1.5 README warnings that bite on old-GPU / hybrid setups (verbatim)

- "**Do not use quantized models from Unsloth that have `_XL` in their name. These are likely to not work with `ik_llama.cpp`.** … the Unsloth `_XL` models that are likely to not work are those that contain `f16` tensors (which is never a good idea in the first place). All others are fine." (Matters for the Qwen3.6 GGUFs — the `_XL` variants exist, §3.)
- "When running hybrid CPU/GPU inference for MoE models with some or all of the experts left on the CPU, **do not use `-rtr`** unless you know what you are doing. The `-rtr` option causes all tensors left in RAM to be repacked to row-interleaved format while loading the model. As not all quantization types have a CUDA implementation, this will result in matrix multiplications with these tensors to be **always done on the CPU** … Most notably, the k-quants (K2_K, Q3_K, Q4_K, Q5_K, Q6_K) do not have row-interleaved implementations for CUDA."
- "Some users have reported issues with **graph parallel (a.k.a. split mode `graph`)** and partial GPU offload (using `--cpu-moe` or `--n-cpu-moe` or tensor overrides). If you are using/want to use split mode graph and observe gibberish/incoherent responses, try adding **`-cuda graphs=0`** to your command line."

---

## 2. Task 2 — what `-sm graph` is in ik_llama.cpp, its flags and limits

### 2.1 Definition (docs/parameters.md, https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/docs/parameters.md)

"`-sm, --split-mode SPLIT_MODE` | How to split the model across multiple GPUs | none | When you have more than one GPU, how to split the model across multiple GPUs, one of: - `none` use one GPU only. - **`graph` split model tensors and computation graph across GPUs. `graph` is exclusive here and extremely effective for dense and MoE [PR 1080]**. - `layer` split layers and KV across GPUs Example: `-sm graph`"

So ik has three split modes (`none`/`graph`/`layer`); `graph` is the tensor-and-graph level split (mainline's experimental `tensor` mode analogue, but a first-class ik feature).

### 2.2 NCCL and P2P (PR #1080 "Graph parallel: the next generation", ikawrakow, 2025-12-22, merged — https://github.com/ikawrakow/ik_llama.cpp/pull/1080)

"This PR improves split mode "graph" performance by using NCCL." Caveats, verbatim:

- "NCCL is required to get the improved performance (but it should still work the same as without this PR if NCCL is not available). On Debian/Ubuntu getting NCCL is as simple as `sudo apt install libnccl-dev`. … **Update**: after the latest changes, **NCCL can be disabled via `cmake -DGGML_NCCL=OFF`**. **For two GPUs, the difference between NCCL and no NCCL is basically negligible**"
- "The `--max-gpu` option does not work with NCCL enabled. It does work without NCCL. **Update**: it now works by falling back to non-NCCL communication."
- "PP performance for more than 4 GPUs is likely to be bad. Why? It looks like I'm not using NCCL correctly. PP and TG performance are both excellent for 2 GPUs, but for 3 or more GPUs the straightforward NCCL usage … results in a horrible PP performance (2X or more lower compared to not using NCCL). Hence, I have implemented a workaround that uses pairwise communicators, but that workaround is only available for 3 and 4 GPUs … **Update**: With more than 4 GPUs it is very likely that disabling NCCL will give better performance."
- "The final version of this PR that was merged will check of NCCL is available on the system and, if yes, enable it by default. To disable usage of NCCL, use `cmake -DGGML_NCCL=OFF`."
- "Performance of smallish MoE models is better than before, but still not as good as with split mode 'layer' for short context."
- Test system: "a 4x3090 system with a Ryzen-3995WX CPU, no NVlink" — i.e. graph mode was developed and validated **without NVLink/P2P**. Headline result: "With 4 GPUs we now get almost 1500 t/s PP and 50 t/s TG!" (Llama-3-70B Q4_0).
- **P2P: not required.** mb8565 ran 2x P40 (a no-P2P pair) in graph mode: #2580 2-GPU pp8192 table — "2x P40 (no P2P) … graph: f16 456.2 → 778.5 t/s; q4_0 (aborts on main) → 774.4 t/s. In graph mode the profiler shows the tile kernel on both GPUs."
- Known old-GPU-era P2P corruption report in the fork: issue #1489 "Using P2P enabled 3090 drivers results in corrupted output" (closed 2026-03-22) — relevant if `GGML_CUDA_P2P`-style direct transfers are enabled on the M10.
- **NCCL on Maxwell (sm_50): [UNVERIFIED]** — NVIDIA's "supported GPUs" NCCL page 404s this round and the ik docs never state a CC floor for their NCCL use. Empirically, mb8565's 2x M10 `-sm graph` runs work (§2.6); whether NCCL or the fallback transport served them is not stated in the PR.

### 2.3 Architecture limits (the `is_model_split_supported` list)

From parameters.md: "Models architectures [supported](https://github.com/ikawrakow/ik_llama.cpp/blob/022bd00aab9ec8428c4811275de89796c677d278/src/llama.cpp#L3056) by `--split-mode graph`":

```
LLM_ARCH_LLAMA, LLM_ARCH_QWEN3MOE, LLM_ARCH_GLM4_MOE, LLM_ARCH_MISTRAL3,
LLM_ARCH_COMMAND_R, LLM_ARCH_COHERE2, LLM_ARCH_COHERE2_MOE, LLM_ARCH_MIMO2,
LLM_ARCH_QWEN3, LLM_ARCH_QWEN3VL, LLM_ARCH_HUNYUAN_MOE, LLM_ARCH_OPENAI_MOE,
LLM_ARCH_ERNIE4_5_MOE, LLM_ARCH_MINIMAX_M2, LLM_ARCH_SEED_OSS, LLM_ARCH_STEP35,
LLM_ARCH_QWEN35, LLM_ARCH_QWEN35MOE, LLM_ARCH_GEMMA4, LLM_ARCH_DEEPSEEK2,
LLM_ARCH_GLM_DSA, LLM_ARCH_MISTRAL4, LLM_ARCH_MELLUM, LLM_ARCH_LAGUNA
```

- **Qwen3.5 (dense, `qwen35`) and Qwen3.5-MoE / Qwen3.6 (`qwen35moe`) are both on the list** → `-sm graph` is supported for the unsloth Qwen3.5 and Qwen3.6 GGUFs (both are hybrid linear-attention models — `llm_arch_is_hybrid()` in `src/llama-arch.cpp` at HEAD returns true for `QWEN35` and `QWEN35MOE`).
- Not on the list: Nemotron-H (arch doesn't exist in ik yet — §3), Mamba/SSM models, Llama-3-Nemotron etc. For those, per PR #2860's body (merged 2026-11-08): "These archs are not in `is_model_split_supported`, so `-sm graph` changes to `-sm layer`. On 2x P100, `-sm layer` gives the same PPL as one GPU." — i.e. **silent fallback to layer mode**.
- MoE-specific `-sm graph` correctness work landed in 2026 (all merged, from the R6 inventory): #2778 "Refuse router tensor overrides with split routed experts in split mode graph" (2026-10-28), #2775 "Fix -sm graph abort and Cohere partial drop with uneven splits" (2026-10-28), #2773 "Gemma4: fix -sm graph crash with routed experts on the CPU" (2026-10-28), #2744 "Fix -sm attn assert for Flint and Qwen3-MoE" (2026-10-24), #2722 "Refuse -ot layouts that break split mode graph" (2026-10-21), #2719 (2026-10-21), #2709 "CUDA: do not fuse the MoE down projection when its output overlaps the up/gate inputs" (2026-10-20), #2706 "CUDA: do not clear padding inside the next expert when experts are offloaded" (2026-10-20). Plus the #1080 tail commit: "With -sm graph, an uneven tensor split can leave a device with no share of a layer's attention or FFN … In Cohere2-MoE, a device with an attention share but no expert share was skipped … The output is then wrong without an error, e.g. with 3 GPUs and -ts 2,1,1."
- Hybrid-specific limits: #1288's caveats for Qwen3.5-MoE (same class as Qwen3-Next): "one cannot have more than one sequence. … if you try to e.g. calculate perplexity for context of 512 using u-batches > 512 (as one usually does for hybrid CPU/GPU inference), that will not work. Neither will pipeline parallelism."
- Partial-offload caveat: README (verbatim in §1.5) — graph mode + `--cpu-moe`/`--n-cpu-moe`/`-ot` may produce gibberish; workaround `-cuda graphs=0`.
- "With several GPUs and `-sm graph`, the MLA models and Gemma-4 need FA." (pascal-volta.md)

### 2.4 `-sm graph` companion flags (all ik-only, parameters.md, verbatim values)

| flag | default | description (abridged verbatim) |
|---|---|---|
| `-smf16, --split-mode-f16` | **1** | "Use f16 for data exchange between GPUs [PR 1087]" |
| `-smf32, --split-mode-f32` | 0 | "Use f32 for data exchange between GPUs [PR 1087]" |
| `-grt, --graph-reduce-type` | f16 | "q8_0 / bf16 / f16 / f32. Reduce the data transferred between GPUs [PR 1154]" |
| `-smgs, --split-mode-graph-scheduling` | 0 | "Force Split Mode Graph Scheduling [PR 1068]" |
| `--max-gpu N` | - | "Define (and use) a maximum number of GPUs per layer with split mode 'graph' … using all of them leads to a lower performance than using just 2 (or using the default split mode 'layer') [PR 1051]" — works via non-NCCL fallback (#1080) |
| `-ts, --tensor-split` | auto | per-GPU fractions (also in mainline) |
| `-wgt, --worst-graph-tokens N` | - | "Control compute buffer sizes for large batches [PR 1560]" |
| `-cuda graphs=N` | (on) | CUDA graph capture toggle; README: add `-cuda graphs=0` for graph-mode + partial offload; #2715 "fixes the draft (MTP/external) context ignoring `-cuda graphs=0`" |
| `-cuda max-graphs=N` | 0 (unlimited) | "Cap on the number of cached CUDA graphs [PR 2715]" |
| `-cuda, --cuda-params` | - | "Comma-separated list of cuda parameters … Powerful way to tweak Fusion, GPU offload threshold, and MMQ-ID threshold [PR 910, PR 181…]"; also `"offload-batch-size=8"` (used in #2850) |
| `-cuda fa-fabsum=1` | off | "replaces tile_f16 with its FABsum variant on sm_60 (#2510, see parameters)" (pascal-volta.md) |

### 2.5 Other ik-only flags that matter for old GPUs (parameters.md, verbatim)

- `-fa, --flash-attn` — **default `on` in ik** ("auto / on / off. Improves t/s and reduces memory usage.") vs mainline default `auto` (mainline multi-gpu.md). For a fair comparison, pin `-fa on` on both.
- `-fmoe or --fused-moe` — "Fused MoE ffn_up and ffn_gate … Speedup for MoE models. [PR 229]" — **enabled by default** (`--no-fmoe` disables: "Disable fused MoE | enabled"). No mainline equivalent — an ik advantage for MoE decode/prompt.
- `--no-fug, --no-fused-up-gate` — "Disable fused up-gate | enabled | Turn off the speedup for dense models. [PR 741]"
- `-ger, --grouped-expert-routing` — "Enable grouped expert routing | disabled | For BailingMoeV2 architecture (Ling/Ring models). [PR 836, PR 838]"
- `-rtr, --run-time-repack` — "Repack tensors if interleaved variant is available | - | May improve performance on some systems. [PR 147]" (+ README CPU-MoE warning, §1.5)
- `-mla, --mla-use` — "0 / 1 / 2 / 3, default 3. For DeepSeek models, and other recent models that are using MLA. [PR 188, PR 205, PR 235, PR 243, PR 252…]"; pascal-volta.md: "The MLA models also switch to their second head size at 128 tokens (default `-mla 3`)"
- `-amb, --attention-max-batch` — "Max batch size for attention computations | 256 | Specifies the maximum K*Q size in MB we want to tolerate. Default changed from 0 to 256 in PR 2312 to avoid very large temporary compute buffers on MLA models. [PR 237, PR 2743]"
- `-ser, --smart-expert-reduction` — "Experts reduction Kmin,t | -1, 0 | Use a custom number of active experts. Powerful, basically REAP from just command line. If we set t = 1, we use a fixed number of experts K_min (`-ser 1,6` will use 6 experts instead of the model default). [PR 239]"
- `--offload-only-active-experts / -ooae` (ON) / `-no-ooae` — "tells the scheduler to offload only the activated experts … reduces the amount of RAM -> VRAM data transfer [PR 405, PR 698]"
- `-ot or --override-tensor` — "Override where model weights are stored using regular expressions. This allows for example to keep the MoE experts on the CPU"; parameters.md example: `-ot "blk.(?:[0-9]|[1-7][0-9]|[8][0-7]).ffn._exps.=CPU"`
- `--ctx-checkpoints N` (32) + `--ctx-checkpoints-interval` (512) + `--ctx-checkpoints-tolerance` (5) + `--ctx-checkpoints-eviction` (`variance`/`fifo`/`auto`) + `--ctx-ckpt-spill-dir` + `--ctx-ckpt-live-n` (2) — "Enable checkpoint for recurrent models **Qwen3-Next and Qwen3.5-MoE**. [PR 1310]"; tolerance: "Creates the checkpoint N tokens before the prompt is fully processed to reduce prompt process for **Qwen 3.5 thinking models**. [PR 1346]"; eviction [PR 2020]; spill [PR 2446].
- `-fit` / `--gpu-fit-margin` / `--fit-margin` (1024) — VRAM auto-fit (also mainline, ik-tuned).
- `--dsa / --dsa-top-k / -ictk / -fidx` (GLM-DSA sparse attention), `--swa-compress`, `--dsv41-replay-keep` — newer hybrid-model flags (not needed for Qwen3.5/3.6, listed for completeness).

### 2.6 Old-GPU / M10 empirical numbers for `-sm graph` (from the R6-verified mb8565 PRs)

- **#2740** (merged 2026-10-23, 2-warp MMVQ): "2x M10, `-sm layer` | 5.573 → 6.506 t/s (**+16.7 %**)" and "2x M10, `-sm graph` | **10.25 → 11.16 t/s (+8.9 %)**" (Llama-3.1-8B-Instruct Q3_K_M, tg128, `llama-bench -p 0 -n 128 -r 5 -ngl 99 -fa 1`, `-r 3` on M10). → **`-sm graph` demonstrably works on the M10 (2 dies).**
- **#2701** (merged 2026-10-19): 2x P100 `-sm graph` 66.44 → 73.90 t/s (+11.2 %), `-sm layer` 42.23 → 46.71 (+10.6 %) (Q4_K_M, tg128).
- **#2580** (merged 2026-10-02): 2x P40 no-P2P `-sm graph` pp8192 f16 456.2 → 778.5 t/s, q4_0 (aborts on main) → 774.4; 2x P100 P2P graph f16 350.9 → 480.5, q4_0 311.8 → 483.1.
- #2860 (merged 2026-11-08): unsupported archs silently fall back to `-sm layer` ("On 2x P100, -sm layer gives the same PPL as one GPU").

---

## 3. Task 3 — architecture support in ik_llama.cpp as of November 2026

Arch table of record: `src/llama-arch.cpp` at HEAD `8a9bf106` (2026-11-08), https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/src/llama-arch.cpp — `LLM_ARCH_NAMES` contains (extract, Qwen family): `"qwen3"`, `"qwen3moe"`, `"qwen3next"`, `"qwen3vl"`, `"qwen3vlmoe"`, **`"qwen35moe"` (LLM_ARCH_QWEN35MOE)**, **`"qwen35"` (LLM_ARCH_QWEN35)**, `"qwen4exp"`, `"flint"`, plus gpt-oss, deepseek2/4/41, glm-dsa, minimax-m2/m3, gemma4(+mtp), mistral3/4, hunyuan-moe, ernie4_5(-moe), cohere2(-moe), command-r, bailingmoe2/3, dflash/dflash2/dflash-draft, etc. **There is no `nemotron_h` / `nemotron_h_moe` entry.**

### 3.1 `qwen35` (Qwen3.5 dense — 0.8B/2B/4B, hybrid linear attention)

**Supported.** PR #1326 "Adding support for dense Qwen-3.5 models (text only)" — https://github.com/ikawrakow/ik_llama.cpp/pull/1326, **merged 2026-02-26** (ikawrakow). Body: "Adding support for dense Qwen-3.5 models (text only). Tested with `unsloth/Qwen3.5-27B-GGUF` Q4_K_S quant." + "Unsloth are not providing models for i-quants [for dense Qwen3.5] … they do have I-quants for the new Qwen3.5 MoE models." (So for the 0.8B/2B/4B use unsloth's UD-Q*/K quants, not the i-quant line.)
- `llm_arch_is_hybrid()`: true for `QWEN35` (and `QWEN35MOE`) → hybrid/recurrent path, hence the `--ctx-checkpoints` flags (§2.5).
- `-sm graph`: supported (on the §2.3 list).
- pascal-volta.md FA table: "Qwen3.5, Qwen3.6 (256) → GPU on P100/P40" (decode vec_f32 f16/q8_0; prompt tile_f32 per #2580 — the M10-verified PR).

### 3.2 `qwen35moe` (Qwen3.5-MoE family — includes the Qwen3.6 GGUFs)

**Supported.** PR #1288 "Qwen3.5-MoE support" — https://github.com/ikawrakow/ik_llama.cpp/pull/1288, **merged 2026-02-21** (ikawrakow). Bench (mainline build 8111 (11c325c6e) vs ik, 2026-02-19): "qwen35moe 397B.A17B IQ4_XS: CPU pp512 38.98 → 88.53 t/s (2.271x), tg256 3.81 → 7.58 (1.990x); CUDA (3090, all MoE tensors in RAM) pp4096 175.37 → 312.39 (1.781x), tg256 19.50 → 20.22 (1.037x)." Caveats (verbatim): "one cannot have more than one sequence. … perplexity for context of 512 using u-batches > 512 … will not work. Neither will pipeline parallelism."

**The Qwen3.6 GGUFs are exactly this arch.** HF API for `unsloth/Qwen3.6-35B-A3B-GGUF` (https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF, fetched in full):

- `"gguf": {"architecture": "qwen35moe", "context_length": 262144, ...}` — the GGUF files carry `general.architecture = qwen35moe`.
- repo tags include `"qwen3_5_moe"`, `"base_model:Qwen/Qwen3.6-35B-A3B"`; created 2026-04-16, last modified 2026-04-20, 1,631,083 downloads.
- Available quants: BF16, `Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, **UD-IQ4_NL_XL**, UD-IQ4_XS, **UD-Q2_K_XL**, UD-Q3_K_M/S/**XL**, UD-Q4_K_M/S/**XL**, UD-Q5_K_M/S/**XL**, UD-Q6_K, **UD-Q6_K_XL**, **UD-Q8_K_XL**, plus `mmproj-F16/F32/BF16.gguf` and `imatrix_unsloth.gguf`.
- **Pick the non-`_XL` quants for ik** (README warning §1.5 — the `_XL` ones contain f16 tensors and "are likely to not work").

Naming note (to avoid confusion in the benchmark write-up): mainline's MTP PR #22673 (merged 2026-06-28) says "I tested this on **Qwen3.6 27B and Qwen3.6 35BA3B**", and mainline MoE work names "Qwen3.5-35B-A3B" (e.g. PR #29963/#29184 benches) — the 35B-A3B MoE ships under both generations' marketing names, but the **GGUF arch string is `qwen35moe`** in both ik and mainline. ik has no separate `qwen36` arch.

### 3.3 Nemotron-3.5 (`nemotron_h_moe`)

**Not supported in ik mainline as of 2026-11-08** (no arch in `LLM_ARCH_NAMES` at HEAD). The support is in flight:

- **PR #2850 "Nemotron-H (Nemotron 3.5) and Mamba-2 support"** — https://github.com/ikawrakow/ik_llama.cpp/pull/2850, **OPEN**, created 2026-11-05 (TitanMamo). Body (fetched in full): "This adds **Nemotron-H support** - the architecture the Nemotron 3.5 family uses (Bolt, Super, ...) - together with the Mamba-2 state-space ops it needs: `ssm_conv` and `ssm_scan` in the Mamba-2 tensor layout on CPU and CUDA … Tested with the Unsloth `UD-IQ1_S` quant of **Nemotron 3.5 Bolt 74B-A9B** and with NVIDIA's **Nemotron3-Nano-4B (Q4_K_M)**." Limits (from its `docs/README-nemotron.md`, quoted in the PR): "**one sequence per context, no `-sm graph`**". Measured on an RTX 3060 (sm_86), CUDA build: "4B dense: **85.2 t/s decode, ~2 900 t/s prefill** at 2k"; "Bolt 74B-A9B `UD-IQ1_S` with `--n-cpu-moe 28`: **25.1 t/s decode**; prefill **145 t/s** with the host experts computed on the CPU (the default), and **360 / 518 / 682 t/s** with `--cuda-params "offload-batch-size=8"` at `-ub 512 / 1024 / 2048`." "NVFP4 is deliberately not here … being implemented separately." "This supersedes #2836."
- Related state (ik repo search, this round): PR #2836 (closed unmerged, superseded); **PR #1593 "Mamba-2 + Nemotron-H MoE backport (Phase 3.x)" — OPEN since 2026-04-06** (older, incomplete backport); issue #2689 "NemotronH architecture support / Nemotron 3.5 Super VL and 74B A9B Bolt" (closed 2026-10-18, as work moved to PRs); issue #1075 "Feature Request: Support NVIDIA Nemotron 3 Nano" (still open since 2025-12-19).
- Merged groundwork: PR #2858 "ggml: NVFP4 type with row scales and loading of mainline NVFP4 GGUFs" — **merged 2026-11-08** (needed for NVIDIA's NVFP4 Nemotron exports).
- For contrast, mainline already has the arch: mainline's `docs/multi-gpu.md` (fetched this round) lists "Nemotron-H, Nemotron-H-MoE" among architectures — i.e. mainline loads nemotron_h_moe GGUFs; on multi-GPU it can only use `-sm layer` (tensor mode is not implemented for it, §4).

### 3.4 Bottom line for the benchmark

| model (GGUF arch) | ik (HEAD 8a9bf106) | mainline |
|---|---|---|
| Qwen3.5-0.8B/2B/4B (`qwen35`) | **yes** (#1326, 2026-02-26); `-sm graph` OK; ctx-checkpoints supported | yes |
| Qwen3.6-35B-A3B / Qwen3.5-MoE (`qwen35moe`) | **yes** (#1288, 2026-02-21); `-sm graph` OK; one-sequence limit; no PP; use non-`_XL` quants | yes |
| Nemotron-3.5 Bolt/Super (nemotron_h_moe) | **no** (PR #2850 open, 2026-11-05; "no -sm graph" even when merged) | yes (arch present; `-sm layer` only) |

---

## 4. Task 4 — mainline llama.cpp: expert-parallel multi-GPU for MoE

**Direct answer: mainline has no dedicated "expert parallel" mode as of November 2026** — no PR/discussion landed that places different experts of the same layer on different GPUs automatically. Search evidence: `repo:ggml-org/llama.cpp "expert parallel"` → **23 items, 0 of them adding such a mode** (full inventory below). What mainline actually offers for MoE across multiple GPUs:

### 4.1 The official multi-GPU doc (master, fetched in full this round)

https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md — split modes: `none` / `layer` (**default**, pipeline) / `row` ("Deprecated. Older row-split tensor-parallel path … Splits only dense weights across GPUs. Superseded by `tensor`") / `tensor` ("**EXPERIMENTAL.** Tensor parallelism that splits both weights and KV across the participating GPUs via a 'meta device' abstraction").

Key constraints (verbatim):
- tensor mode: "`--flash-attn off` … is a hard error. KV cache types must be non-quantized: f32, f16, or bf16 … `--fit` … Not supported with tensor … Performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise."
- **"`--split-mode tensor` is not implemented for all architectures. The following will fail with 'LLAMA_SPLIT_MODE_TENSOR not implemented for architecture …': MoE / hybrid: Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, Nemotron-H-MoE, Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1; State-space / RWKV-style: Mamba, Mamba2 …; Other: PLAMO2, MiniCPM3, Gemma-3n, OLMo2, BitNet, T5"** → **every MoE model of interest (Qwen3.5-MoE-class, Nemotron-H-MoE) is blocked from tensor parallelism in mainline.**
- NCCL: "There's no runtime flag for NCCL - it's selected at build time (`-DGGML_CUDA_NCCL=ON`, this is the default) … When llama.cpp is compiled with NCCL support it uses it automatically for cross-GPU reductions in `tensor` mode."
- P2P: "`GGML_CUDA_P2P` … opt-in at runtime … P2P requires driver support (usually restricted to workstation/datacenter GPUs) and **may cause crashes or corrupted outputs on some motherboards or BIOS configurations** (e.g. when IOMMU is enabled)."

### 4.2 The mechanisms that do exist

1. **`-ot` / `--override-tensor` (manual per-tensor — and therefore per-expert — placement).** PR #11397 "llama : add option to override model tensor buffers" — https://github.com/ggml-org/llama.cpp/pull/11397, **merged 2025-04-02**: "Adds command line parameter `--override-tensor` (`-ot`) that allows changing the buffer type where a model tensor is allocated. This gives user fine grained control over what tensors are to offloaded to each device." This is the only way to expert-parallel a MoE model in mainline today: regex-pin `blk.N.ffn_(up|gate|down)_exps` to specific devices (ik's parameters.md documents the exact pattern style: `-ot "blk.(?:[0-9]|[1-7][0-9]|[8][0-7]).ffn._exps.=CPU"`).
   - Correction to my earlier note: in Round 4 I cited this as "PR #42563" — that number 404s (checked this round: https://github.com/ggml-org/llama.cpp/pull/42563 → 404). The correct reference is **#11397**.
2. **Pipeline parallelism with host-RAM MoE experts (open, in review).** Discussion #29964 (2026-10-04, drrros) + **PR #29963 "ggml : pipeline parallelism with MoE experts in host RAM"** — https://github.com/ggml-org/llama.cpp/pull/29963, opened 2026-10-04, **open** (reviewers: ggerganov + JohannesGaessler awaiting, sasa7812 commented, aminya requested changes). Core idea (verbatim): "when part of the MoE weights sits in host RAM, llama.cpp silently switches off pipeline parallelism between GPUs (src/llama-context.cpp, `!has_tensor_overrides()`) … All the experts are then uploaded through the first device that can (GPU0 on CUDA) and the GPUs run strictly one after another. This PR keeps pipeline parallelism working with the weights in host RAM. **Each GPU uploads the experts of its own layers**, consecutive ubatches overlap across devices. Long prompt processing is x1.5-2 faster on my 3x RTX PRO 4000." Bench (3x RTX PRO 4000 Blackwell, PCIe5 x16 P2P, EPYC 9274F, 384 GB DDR5, CUDA 13.3): "Qwen3.5-35B-A3B, `-ncmoe 999`, server: PP 2705 / 2725 → **5486 / 5901 t/s**"; "Qwen3.8-Flash-Next Q8_0 (fit): 1009 / 1019 → 1808 / 2102"; "Qwen3.8 llama-bench fit pp8192/pp32768/tg64: 1464 / 1334 / 46.2 → 1749 / 2474 / 47.1". Important counter-evidence from the discussion: eapache (2026-10-05) on a heterogeneous 3060 + 4500 pair: "On master the 4500 peaks at RX of ~10GiB/s … I get about 1100 t/s pp. With #29963 the 3060 peaks at RX of about 3GiB/s … I get only 600 t/s pp. … Explicitly hard-code-disabling `pipeline_parallel` on the PR branch recovers full performance" and "even … all transformer layers on the 4500 and only the output layer on the 3060, enabling the pipeline still imposed a roughly 14% penalty." → pipeline mode helps when the cards are similar and the interconnect is fast; it can hurt on skewed pairs (relevant to a 4× GM107 board with slow cross-die links).
3. **Expert offload + caching (single-GPU-centric, but the pieces):** `--cpu-moe` / `--n-cpu-moe N` (put N experts in host RAM), `--moe-cache-mib` (LRU VRAM cache of hot CPU-resident experts; RFC discussion #24528, merged for the 0.7.x line). Open extensions: **PR #30594 (open, 2026-10-15) "llama : hybrid MoE expert cache with profile prefilling and zero-upload inference"** — "`--moe-cache-profile`: Pre-populates the GPU expert cache using offline-profiled routing distributions so the hottest experts …"; **PR #30908 (closed unmerged, 2026-10-21) "llama : MoE hybrid mode concurrent GPU resident and CPU missed experts"** — "based on Strata's dual-device offloading architecture … (such as Qwen 3.5 MoE and Qwen 3.6 MoE)".
4. **Performance-side MoE kernels (not parallelism, but lands in the benchmark):** PR #29184 "CUDA: fuse shared experts into MMVQ" (merged 2026-10-02) — "Qwen3.5-35B-A3B-Q8_0 on RTX Pro 6000: microbatch 1: 268.96 → 275.05 t/s (+2.26 %), microbatch 2: 448.21 → 468.5…"; PR #20905 (MoE GEMV rewrite, merged 2026-03-29, 1.07–1.34× on Blackwell/Ampere, no Maxwell rows — R6 §2.2).
5. **History:** issue #4667 (2023-12-28, closed completed 2024-04-02) "[FEATURE REQUEST] Make MoE offloading offload real layers rather than groups of layers" — the original expert-granularity offload ask (closed when `--cpu-moe`/`-ot` matured); issue #11532 (closed 2025-04-08) "MoE only load activated expert(s) to GPU" (→ `--cpu-moe` family + `-ooae` in ik).

### 4.3 Full "expert parallel" search inventory (23 items, repo:ggml-org/llama.cpp)

| # | type | state | dates | relevance |
|---|---|---|---|---|
| 4667 | issue | closed completed | 2023-12-28 → 2024-04-02 | expert-granularity offload FR (origin of the idea) |
| 11397 | PR | **merged** | 2025-01-24 → 2025-04-02 | **`-ot` tensor override — the manual EP mechanism** |
| 11532 | issue | closed | 2025-01-30 → 2025-04-08 | load-only-active-experts FR |
| 16000 | PR | open | 2025-09-15 | `--numa mirror` (CPU side) |
| 18039 | PR | merged | 2025-12-14 → 2026-06-12 | EAGLE3 spec decoding (MoE benches mention EP-adjacent setups) |
| 19822 | PR | closed unmerged | 2026-02-23 | LLaDA2 arch |
| 21147 | PR | merged | 2026-03-29 → 2026-04-06 | WebGPU MUL_MAT_ID (MoE op) |
| 21996 | PR | closed unmerged | 2026-04-16 | SYCL batched mul_mat_id MMVQ |
| 22105 | PR | merged | 2026-04-19 → 2026-06-28 | DFlash spec decoding |
| 22673 | PR | merged | 2026-05-04 → 2026-05-16 | MTP ("tested on Qwen3.6 27B and Qwen3.6 35BA3B") |
| 25117 | issue | closed | 2026-06-29 → 2026-10-13 | DFlash regression, quantized MoE |
| 25209 | PR | closed unmerged | 2026-07-01 | CPU tensor parallelism RFC (not GPU EP) |
| 26185 | PR | merged | 2026-07-27 → 2026-08-15 | Kimi-K3 (latent MoE) |
| 26467 | PR | open | 2026-08-02 | Instella-MoE arch |
| 27306 | issue | open | 2026-08-18 | draft-mtp DeviceLost bug |
| 27411 | PR | open | 2026-08-19 | "cuda : prefer MMQ for MXFP4 on CDNA2" — "the clause also applies to **expert-parallel MoE tensors**" (AMD wording; kernel selection, not a mode) |
| 27853 / 27854 | issues | closed | 2026-08-28 | off-topic "layer access parallelism" posts |
| 29128 | issue | closed | 2026-09-19 → 2026-11-04 | LoRA expert-count mismatch OOB in MUL_MAT_ID |
| 29184 | PR | merged | 2026-09-20 → 2026-10-02 | **CUDA: fuse shared experts into MMVQ** (Qwen3.5-35B-A3B numbers above) |
| 29535 | PR | merged | 2026-09-27 → 2026-10-06 | K2 Horizon dense + MoVA |
| 29963 | PR | **open (in review)** | 2026-10-04 | **pipeline parallelism with host-RAM MoE experts** (§4.2.2) |
| 30594 | PR | open | 2026-10-15 | **hybrid MoE expert cache, profile prefilling, zero-upload** |
| 30908 | PR | closed unmerged | 2026-10-21 | MoE hybrid GPU/CPU concurrent experts (Strata) |

### 4.4 M10-specific synthesis (ik vs mainline for MoE on 4 dies)

- **mainline on M10 + MoE:** `-sm layer` (pipeline) is the only sane split mode (tensor mode blocked for MoE); experts can be manually distributed per die with `-ot` regexes (EP-by-hand), or kept in host RAM with `--n-cpu-moe` + `--moe-cache-mib`. No automatic expert distribution exists.
- **ik on M10 + MoE:** adds `-sm graph` — tensor+graph level split that is explicitly "extremely effective for dense and MoE" and is on the supported-arch list for both Qwen3.5 and Qwen3.5-MoE/Qwen3.6 (`qwen35`/`qwen35moe`); works on M10 empirically (#2740: 2x M10 -sm graph 11.16 t/s vs -sm layer 6.51 t/s, Q3_K_M Llama-3.1-8B — note the ~1.7x gap between graph and layer on the M10); plus `-fmoe` (default on) fused MoE up/gate, `-ooae`, `-ser`. For a fair A/B: mainline `-sm layer` vs ik `-sm layer` first (same mode), then ik `-sm graph` as the ik-only upside.

---

## 5. Practical benchmark checklist (synthesis, all sources above)

1. **ik build:** `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50` + CUDA 12.x toolkit (CUDA 13 can't compile Maxwell); HEAD `8a9bf106` (2026-11-08). Decide first: CUDA 11.x toolkit on the locked M10 host vs driver/toolkit upgrade (ik docs recommend 12.x; NVIDIA's quoted recommendation is 12.9 + 580 branch for Pascal/Volta; M10's own last driver branch is 550 — untested territory in the ik docs).
2. **Models:** Qwen3.5-0.8B/2B/4B = arch `qwen35` (ik OK since 2026-02-26); Qwen3.6-35B-A3B / Qwen3.5-MoE = arch `qwen35moe` (ik OK since 2026-02-21; use non-`_XL` unsloth quants on ik); Nemotron-3.5 = ik **blocked** (PR #2850 open; and "no -sm graph" even when it lands) — mainline only.
3. **Matched flags for the A/B:** `-fa on` (ik default is on, mainline default auto — pin both), `-ctk f16 -ctv f16` (or q8_0 — never q5_0/q5_1/bf16/f32 on Pascal-class FA, never q8_KV: crash), `-ngl 99`, same `-b/-ub/-r`, same quant file. ik-only deltas to declare in the write-up: `-fmoe` on (mainline has no fused MoE up/gate), `--no-fug` on, `-ooae` on, `-mla 3`, `-amb 256`, `--ctx-checkpoints 32` (qwen35/qwen35moe only), i-quants (ik-only quant types).
4. **Multi-die:** ik `-sm layer` (apples-to-apples with mainline) and ik `-sm graph` (ik-only; `-smf16` default 1, `-grt f16`, `--max-gpu`, `-DGGML_NCCL` choice; P2P not required; watch the README partial-offload warning → `-cuda graphs=0` if mixing with `--n-cpu-moe`/`-ot`). mainline: `-sm layer` + manual `-ot` expert placement (and the open #29963 pipeline-with-host-experts PR, with the eapache skew caveat for non-uniform cards).
5. **Qwen3.5-MoE limits to respect in the harness:** one sequence at a time (#1288); no pipeline parallelism for that model on ik; perplexity with ubatch > 512 on ctx-512 won't work.

---

## 6. Source index (fetched in full this round)

- ik repo: api.github.com/repos/ikawrakow/ik_llama.cpp (metadata: branch main, pushed 2026-11-08) + /commits?per_page=1 (HEAD 8a9bf106)
- raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/: README.md, docs/build.md, docs/parameters.md, docs/pascal-volta.md, src/llama-arch.cpp, convert_hf_to_gguf.py (334 KB — arch mapping context)
- GitHub PR/discussion pages (firecrawl HTML): ik #1080 (full, incl. NCCL caveats), ggml-org #29963 (full, incl. all bench tables + review state), ggml-org discussions/29964 (full)
- GitHub search API (JSON): ik repo "nemotron" (26 items, all parsed); ggml-org "expert parallel" (23 items, all parsed)
- raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md (full; parameters.md → 404, file not present on master)
- huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF (full JSON: gguf.architecture=qwen35moe, quant list)
- 404s checked (negative results): ggml-org PR #42563 (does not exist — correct -ot ref is #11397), docs.nvidia.com NCCL "supported-gpus" page (moved; NCCL-on-Maxwell left [UNVERIFIED])
- Carried from R6 (re-quoted where used): ik #2580/#2701/#2721/#2726/#2740/#2860 M10 numbers; mainline #20905/#23539/#30524.

[SNIPPET]-only this round: nothing material — one search-result line from web_search ("NVIDIA CUDA 13.0 Maxwell Compatibility Guide PDF exists at docs.nvidia.com/cuda/archive/13.0.0") corroborates the pascal-volta.md CUDA-13 statement; NCCL Maxwell support status is marked [UNVERIFIED] rather than guessed.

Known gaps: NCCL CC floor for sm_50 (docs page 404; empirical 2x M10 -sm graph works, transport type unstated); ik sm_50 MUL_MAT table cell (pascal-volta.md only lists sm_60/61/70; sm_50 has no dp4a either — expected same "MMVQ ≤8, dequant+SGEMM >8" class, unconfirmed in doc); whether mainline's Qwen3.6 35B-A3B ships with any different arch string than qwen35moe in a later GGUF (unsloth's current files say qwen35moe).
