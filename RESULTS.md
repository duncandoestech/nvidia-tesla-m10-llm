# Tesla M10 inference results

## TLDR

- **Speed:** tuned llama.cpp takes Qwen3.5-2B from 17.6 to about 42 tokens/s on one of the M10's four GPUs (2.4x). Qwen3.6-35B-A3B (MoE) on all four GPUs does 15.8 tok/s, or 17.7 with its built-in MTP drafting.
- **Role models, one per GPU:** Qwen3.5-4B for chat + vision (21.8 tok/s with MTP), Qwen2.5-Coder-7B for code (HumanEval 89.0 %), embeddinggemma-300M for search (SciFact nDCG@10 0.789), and the Kev-0.8B decision model (about 1 s per request).
- **Real-world test:** a larger agent (Hermes on Qwen3.8-27B) delegated real tasks to the four role models, including one job with all four GPUs working at once (1.98x faster than running the lanes one after another). The small models keep numbers exact but sometimes attach facts to the wrong source, so the big agent should check their work.
- **Surprises:** the M10's four GPUs cannot send data to each other directly; small separate draft models make generation slower, not faster; a 38 W power cap costs nothing for MoE serving; a reranker on top of a good embedder did not help.
- **Measured vs guessed:** every speed here is measured, in at least two interleaved rounds, with correctness checked by llama.cpp's test suite, perplexity and byte-identical output. Why each kernel change helps is explained, not proven.

Hardware: Tesla M10 (4x GM107, 8 GB each) in an X79 box (i7-3930K, 16 GB RAM), card on an x8 upstream link, next to 2x P100.
Software: mainline llama.cpp `f477f795c` (2026-11-08), CUDA 12.9 inside an Ubuntu 22.04 / gcc 11 container, driver 580.178.04, `CMAKE_CUDA_ARCHITECTURES=50`.
Run rules: cool to 48 C or below before each run, llama-bench with 3 repetitions, nvidia-smi logged at 1 Hz, stop above 80 C.

## Correctness on sm_50 (test-backend-ops, one GM107 vs CPU)

| Op | Result |
|---|---|
| FLASH_ATTN_EXT | all pass |
| ARGSORT | all pass (includes the PR #28389 fix) |
| MUL_MAT | **aborts**: `GGML_ASSERT(convert_func != nullptr)` at ggml-cuda.cu:1710 for **Q4_K, Q5_K, Q6_K weights given as non-contiguous views** (test cases with `k_v=1280`, n ≥ ~9). Other types pass. |

The MUL_MAT abort is in the batched-cuBLAS fallback, which only runs on GPUs without dp4a/MMQ. It has no non-contiguous dequantize function for K-quants. Normal weight matmuls are contiguous and work (the Q4_K_M model runs). The risk is any graph that hands a K-quant *view* to mul_mat. Candidate for an upstream issue.

## Phase 0: first run (Qwen3.5-2B Q4_K_M, GPU 0)

| fa | pp512 t/s | tg128 t/s |
|---|---|---|
| 0 | 295.7 ± 6.4 | 17.60 ± 0.04 |
| 1 | 289.1 ± 3.8 | 17.70 ± 0.02 |

- Power sat at the 53 W per-GPU cap. SM clock was 1124–1137 MHz (max 1202). Throttle reason `0x2` (applications clocks), but application clocks show N/A and cannot be changed on this card. Peak temperature was 53 C.
- Generation at 17.6 t/s is ~27 % of the bandwidth ceiling (83 GB/s ÷ 1.28 GB ≈ 65 t/s). **Generation is compute-bound, not memory-bound.**
- nvprof kernel breakdown for generation (`profile-tg.sh`; nsys 2026 records nothing on Maxwell, nvprof works):
  - `mul_mat_vec_q` Q4_K 39.8 %, Q6_K 31.5 %, Q5_K 15.0 % → **86 % of GPU time is quantized mat-vec**
  - host-to-device copies 6.1 %, gated DeltaNet 1.4 %, flash attention 0.3 %
## Phase 1: quant sweep, stock build (GPU 0, fa 1, 3 reps)

| Model | Quant | file GB | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| Qwen3.5-2B | Q4_0 | 1.21 | 285.9 ± 2.3 | **20.58** ± 0.36 |
| Qwen3.5-2B | IQ4_XS | 1.17 | 283.7 ± 1.9 | 19.89 ± 0.01 |
| Qwen3.5-2B | IQ4_NL | 1.21 | 281.4 ± 4.1 | 18.55 ± 0.14 |
| Qwen3.5-2B | Q8_0 | 2.01 | 290.0 ± 1.2 | 18.05 ± 0.17 |
| Qwen3.5-2B | Q4_K_M | 1.28 | 292.6 ± 1.7 | 17.70 ± 0.07 |
| Qwen3.5-2B | Q5_K_M | 1.44 | 306.8 ± 6.5 | 16.87 ± 0.04 |
| Qwen3.5-2B | Q6_K | 1.57 | 304.9 ± 5.9 | 16.57 ± 0.43 |
| Qwen3.5-2B | BF16 | 3.78 | 315.5 ± 9.9 | 12.64 ± 0.21 |
| Qwen3-1.7B | Q4_0 | 1.06 | 278.2 ± 3.9 | 23.55 ± 0.13 |
| Qwen3-1.7B | Q8_0 | 1.83 | 284.8 ± 2.0 | 19.88 ± 0.13 |

- **Q8_0 beats Q4_K_M** despite a 57 % larger file. The decode math costs more than bandwidth on GM107, and the simple formats (Q4_0, IQ4_XS, Q8_0) win.
- Prompt processing is ~280–315 t/s for every format. It all goes through dequantize + FP32 cuBLAS, so the format barely matters.
- Production was busy for every run (logged).

**Community reference** (llama.cpp #15013 format: Llama-2-7B Q4_0, `-ngl 99`):

| fa | pp512 t/s | tg128 t/s |
|---|---|---|
| 0 | 64.85 ± 0.56 | 8.16 ± 0.33 |
| 1 | 64.38 ± 0.29 | 8.71 ± 0.42 |

Tesla M40 in the same thread: 282.7 / 38.0 (fa 0). One GM107 = **~23 % of an M40** on both prompt and generation. My estimate from bandwidth and core count was 9–11 tg and 50–60 pp; measured 8.2–8.7 and 65.

## DMMV (float mat-vec) A/B: dead end

At the last commit before DMMV was removed (`c3ea58aca^`, Nov 2024), Llama-2-7B Q4_0, GPU 1, two interleaved rounds:

| Build | pp512 t/s | tg128 t/s |
|---|---|---|
| MMVQ (emulated dp4a) | 65.60 / 65.19 | **8.72 / 8.44** |
| `GGML_CUDA_FORCE_DMMV=ON` (float) | 65.27 / 65.22 | 2.76 / 2.70 |

The old float path is **3.2x slower** on Maxwell, so reviving it is not worth it. The integer path with emulated dp4a is the right base to tune.

## Phase 2: mmvq launch shape (single-token generation path)

On sm_50, llama.cpp uses the "GENERIC" mmvq table: 4 warps per block, 1 output row per block (2 for a few low-bit types). A small patch (`patches/m10-mmvq-tunables.patch`) makes both compile-time overridable for `ncols_dst == 1` (`-DM10_NW1=<warps> -DM10_RPB1=<rows>`). Qwen3.5-2B Q4_K_M, GPU 2, fa 1, tg128, 5 reps per run, two interleaved rounds. Production was busy during most runs, the same for every variant.

| warps / rows per block | round 1 t/s | round 2 t/s |
|---|---|---|
| stock (4 / 1) | 18.98 ± 0.66 | 17.72 ± 0.10 |
| 8 / 1 | 15.79 ± 0.03 | 15.81 ± 0.02 |
| 2 / 1 | 21.10 ± 0.23 | 20.63 ± 0.47 |
| 1 / 1 | 22.79 ± 0.54 | 22.22 ± 0.29 |
| 4 / 2 | 22.08 ± 0.24 | 22.67 ± 1.10 |
| 2 / 2 | 24.81 ± 0.43 | 25.16 ± 0.45 |
| **1 / 2** | **26.32 ± 0.30** | **26.32 ± 0.25** |

**+44 % generation from launch shape alone** (stock avg 18.35 → 26.32). The trend is monotonic: fewer warps per block and more rows per block both help. My reading (inferred, not profiled): with only 5 SMs and slow emulated dp4a, splitting one row across 4 warps adds a shared-memory reduction and sync per row for little extra memory parallelism. Giving one warp several rows reuses the activation vector from registers. Round 2, same setup (production busy in all runs):

| warps / rows per block | round 1 t/s | round 2 t/s |
|---|---|---|
| stock (4 / 1) | 19.15 ± 0.39 | 18.00 ± 0.47 |
| 1 / 2 | 27.13 ± 0.38 | 26.55 ± 0.31 |
| **1 / 4** | **28.09 ± 0.22** | **27.95 ± 0.39** |
| 1 / 8 | 27.02 ± 0.28 | 26.78 ± 0.22 |
| 2 / 4 | 27.81 ± 0.46 | 27.48 ± 0.17 |

**Best: 1 warp, 4 rows per block. Stock 18.6 → 28.0 t/s, +51 %.** It plateaus past 4 rows. Correctness: `test-backend-ops -o MUL_MAT -p "n=1,"` passes 684/684 for the 1/4 and 1/8 builds. This was tuned on Q4_K_M only; the best value per quant type is still to be checked.

## Phase 3: tuned launch shape on every quant (GPU 0, fa 1, tg128, 3 reps)

| Model | stock | 1 warp / 4 rows | 1 warp / 2 rows | gain (best) |
|---|---|---|---|---|
| Qwen3.5-2B Q4_0 | 19.81 ± 0.06 | **32.43 ± 0.04** | 29.95 ± 0.89 | +64 % |
| Qwen3.5-2B Q4_K_M | 17.54 ± 0.10 | **27.60 ± 0.18** | 25.51 ± 0.21 | +57 % |
| Qwen3.5-2B Q5_K_M | 16.96 ± 0.19 | **26.38 ± 1.29** | 26.02 ± 0.91 | +56 % |
| Qwen3.5-2B Q6_K | 17.08 ± 0.55 | **24.38 ± 0.65** | 23.19 ± 0.27 | +43 % |
| Qwen3.5-2B Q8_0 | 18.20 ± 0.15 | **24.41 ± 0.28** | 23.31 ± 0.36 | +34 % |
| Qwen3.5-2B IQ4_XS | 19.90 ± 0.02 | **31.07 ± 0.26** | 29.78 ± 0.66 | +56 % |
| Qwen3.5-2B IQ4_NL | 18.48 ± 0.07 | **30.66 ± 0.06** | 28.80 ± 0.15 | +66 % |
| Qwen3-1.7B Q4_0 | 23.52 ± 0.21 | **39.32 ± 1.60** | 36.33 ± 1.81 | +67 % |
| Qwen3-1.7B Q8_0 | 19.83 ± 0.05 | **27.44 ± 0.75** | 26.62 ± 0.55 | +38 % |
| Llama-2-7B Q4_0 | 7.92 ± 0.12 | **12.33 ± 0.34** | 11.49 ± 0.33 | +56 % |

1 warp / 4 rows wins for every format. Best small Qwen on one GM107: **Qwen3.5-2B Q4_0 at 32.4 t/s** (stock 19.8), which is ~39 GB/s, about 47 % of the 83 GB/s bandwidth. Llama-2-7B goes from 23 % of an M40 to 32 %.

## Phase 4: output-head (lm_head) format (tuned build 1 warp / 4 rows, GPU 0)

Profiling the tuned build put the Q6_K output head at 25 % of generation time: Qwen3.5's 248k vocab makes it ~417 MB of a 1.2 GB model, read every token. I quantized from the BF16 file myself (Q4_0 layers, only `--output-tensor-type` differs, no imatrix). Two interleaved rounds; perplexity on wikitext-2 test, ctx 512, 150 chunks.

| Output head | file GB | pp512 t/s | tg128 t/s (r1 / r2) | PPL |
|---|---|---|---|---|
| BF16 model (reference) | 3.78 | | | 12.05 ± 0.17 |
| Q6_K (control, usual default) | 1.20 | 288.4 / 282.9 | 33.12 / 31.96 | 13.15 ± 0.19 |
| **Q4_0** | 1.07 | 288.9 / 283.5 | **37.61 / 37.21** | 13.32 ± 0.19 |
| Q8_0 | 1.33 | 283.3 / 282.9 | 33.25 / 32.76 | 13.14 ± 0.19 |

- **Q4_0 head: +15 % generation (32.5 → 37.4 t/s) for +1.3 % perplexity** (13.15 → 13.32, within one error bar of each other).
- A Q8_0 head runs at the same speed as Q6_K (the head kernel is near the bandwidth limit, and Q8_0's cheaper decode offsets its larger size) with no quality gain. No reason to pick it.
- Q4_0 layers themselves cost +9 % perplexity vs BF16 (12.05 → 13.15). An imatrix quant would likely narrow that. Not tested yet.
- Production was busy during the speed runs (logged).

**Best single-GM107 config so far: Qwen3.5-2B, Q4_0 layers + Q4_0 head, tuned mmvq → ~37.4 t/s generation, ~285 t/s prompt** (stock build + unsloth Q4_K_M: 17.6 t/s). That is **2.1x the starting point**.

## Phase 5: using all four GM107s (tuned build 1w/4r)

**A. Four independent copies** (one llama-bench per GPU at the same time, 3 reps):

| Model | per-GPU pp512 | sum pp512 | per-GPU tg128 | **sum tg128** |
|---|---|---|---|---|
| Qwen3.5-2B Q4_0 + Q4_0 head (tg16 smoke run) | | | 35.9–38.6 | **~150** |
| Llama-2-7B Q4_0 | 65.0–66.1 | 261.9 | 12.13–12.33 | **48.8** |

The copies do not slow each other down (per-GPU = single-GPU numbers), despite sharing one x8 upstream link.

**B. One model, `-sm tensor` over all 4 GPUs** (single stream, 3 reps):

| Model | pp512 t/s | tg128 t/s | vs one GPU |
|---|---|---|---|
| Qwen3.5-2B Q4_0 + Q4_0 head | **624.6** ± 1.1 | 28.78 ± 0.63 | pp 2.2x, tg **0.77x** (37.4 on one GPU) |
| Llama-2-7B Q4_0 | 189.0 ± 0.0 | **29.44** ± 0.05 | pp 2.9x, tg **2.4x** (12.3 on one GPU) |

Tensor split pays off once each GPU's share of a layer is big enough to hide the per-layer sync. It is a loss for a 2B model and a big win for 7B and up.

**C. Batched decode on one GPU** (llama-batched-bench, Qwen3.5-2B Q4_0 + Q4_0 head, pp128 tg128 per sequence, tuned build):

| parallel sequences | prompt t/s | generation t/s (total) |
|---|---|---|
| 1 | 187 | 37.1 |
| 2 | 248 | 46.2 |
| 4 | 285 | 55.4 |
| 8 | 291 | **64.5** |
| 16 | 279 | 26.2 |
| 32 | 283 | 49.4 |

**Cliff past 8 sequences:** mmvq handles at most 8 columns. Beyond that the CUDA backend falls back to dequantize + FP32 cuBLAS (no MMQ on sm_50), which is slow at small batches. For many parallel users, four copies with ≤ 8 sequences each beat one big batch. Phase 6 tunes the 2–8 column path.

## Phase 6: tuning the 2–8 column mmvq path (batched decode, speculative verify)

Patch defines `M10_NWB` / `M10_RPBB` set warps / rows per block for `2 <= ncols_dst <= 8`. Stock is 4 warps for 2–4 columns, 2 warps for 5–8, 2 rows. All builds keep the single-token tuning (1 warp / 4 rows). llama-batched-bench, Qwen3.5-2B Q4_0 + Q4_0 head, GPU 0, pp128 tg128 per sequence, average of 2 interleaved rounds. Columns are parallel sequences; values are total generation t/s.

| batch path warps / rows | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| stock batch path | 38.0 | 44.0 | 49.4 | 57.7 | 63.5 | 64.5 |
| 1 / 2 | 37.6 | 52.2 | 58.6 | 63.0 | 71.2 | 66.1 |
| **1 / 4** | 37.6 | **56.8** | **64.2** | 67.9 | **76.6** | **78.1** |
| 2 / 4 | 37.2 | 53.6 | 62.0 | 66.4 | 72.4 | 77.1 |
| 1 / 8 | 37.0 | 56.9 | 65.8 | **73.6** | 69.2 | 54.9 |
| 2 / 2 | 37.5 | 47.3 | 54.7 | 58.1 | 62.2 | 66.1 |

- **1 warp / 4 rows also wins for batches:** +29 % at 2 sequences, +30 % at 3, +21 % at 8.
- 8 rows is better at 3–4 columns but falls apart at 6–8 (likely register pressure). A per-width table (8 rows for ≤ 4 columns, 4 rows for 5–8) could gain a little more at 4 columns. Not built.
- Correctness: `test-backend-ops -o MUL_MAT` for n = 1..8 passes **1637/1637** on all five variants.
- **New best build: `build-b-nw1rpb4`** (`-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4`).

## Phase 7a: vmad dp4a (PTX `vmad` byte-select → native `VMAD.S8.S8` on sm_50)

Idea from shinbunbun/llama-cpp-p100-patches #01, which is gated to sm_60 there. Patch define `M10_VMAD`. Both builds have the 1 warp / 4 row single-token tuning. GPU 0, tg128, 3 reps, two interleaved rounds:

| Model | base r1 / r2 | vmad r1 / r2 | gain (avg) |
|---|---|---|---|
| Qwen3.5-2B Q4_0 + Q4_0 head | 37.95 / 37.60 | **42.33 / 41.87** | **+11.4 %** |
| Qwen3.5-2B Q4_K_M | 27.18 / 27.45 | 28.32 / 29.69 | +6.2 % |
| Qwen3.5-2B Q8_0 | 25.09 / 24.43 | 26.06 / 24.63 | +2.4 % |
| Llama-2-7B Q4_0 | 12.44 / 12.23 | 12.58 / 13.92 | +7.4 % (noisy r2: ± 0.97) |

- Correctness: `test-backend-ops -o MUL_MAT -p "n=1,"` 684/684 pass.
- Biggest gain on Q4_0, where the dot product is a larger share of the work. Q8_0 is close to its memory limit.
- **Best Qwen3.5-2B on one GM107 is now ~42 t/s** (start of the night: 17.6).

## Phase 7b: MoE models on all four GM107s (`-sm layer -ts 1/1/1/1 -fa 1`, 2 reps)

| Model | file GB | build | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q2_K_XL | 12.6 | tuned (1w/4r) | 65.8 ± 0.5 | 14.48 ± 0.24 |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 17.2 | stock | 68.4 ± 1.2 | 9.27 ± 0.02 |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 17.2 | tuned | 68.9 ± 1.2 | **14.29** ± 0.02 (+54 %) |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 22.9 | stock | 84.6 ± 1.0 | 11.08 ± 0.00 |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 22.9 | tuned | 85.3 ± 1.1 | **15.60** ± 0.01 (+41 %) |
| Nemotron-3.5-Lightning-30B-A3B UD-Q4_K_XL | 25.5 | tuned | 70.2 ± 0.0 | **16.90** ± 0.01 |

- MoE decode with one token goes through the regular mmvq kernel (`has_ids && ncols_dst > 1` is the only MoE-kernel path), so **the single-token tuning applies directly: +41–54 %**.
- **Q4 is faster than Q3 and Q2** despite being bigger: low-bit K-quant decode costs more compute than it saves in bandwidth on GM107.
- Smoke runs: `-sm tensor` on Qwen3.6-35B-A3B Q2 gave 8.4 t/s vs 14.5 with layer split (the per-layer sync costs more than the parallel reads save for small experts). `-ncmoe 8` (8 layers' experts on the CPU) gave 5.0 t/s.
- **MoE prompt processing wants a big ubatch** (Qwen3.6-35B-A3B Q3, pp2048, 4 dies): `-ub` 128 → 42.5, 256 → 52.6, 512 → 66.3, **1024 → 81.6 t/s** (+23 % over the default). More tokens per expert per step amortizes the dequantize + cuBLAS cost. 2048 not tried yet (needs `-b` > 2048).
- **Flash attention on long prompts (head size 256)** in mainline on sm_50 is no win for these hybrid models. Qwen3.5-2B fa 0 / fa 1: pp512 284.8 / 280.9, pp2048 289.3 / 282.0, pp4096 283.2 / 268.9. Qwen3.6-35B-A3B pp4096: 66.5 / 63.8. Only a few layers are full attention (the rest are Gated DeltaNet), so the D=256 prefill issue fixed in ik_llama.cpp #2580 barely shows here.
- For scale: the dense Qwen3.8-27B on the same four dies does 6.65 t/s (tensor split). An A3B MoE is **2.3–2.5x faster** on this card.

## Phase 8: combined best build

`build-best` = `-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4 -DM10_VMAD -DM10_MOE_NOSPILL`. `build-best-moerpb4` adds `-DM10_MOE_RPB=4`.

**Verification**
- Spill check (`cuobjdump -res-usage`): stock has 18 `mul_mat_vec_q_moe` kernels with a 16-byte stack. **build-best: 317/317 mmvq kernels with STACK:0.** The `M10_MOE_NOSPILL` unrolled select removes all of them.
- `test-backend-ops -o MUL_MAT_ID`: 931/931 pass (both builds). `-o MUL_MAT` full suite stops at the same pre-existing K-quant non-contiguous assert as stock (see top). The n = 1..8 subset passed 1637/1637 in phase 6.

**Dense, GPU 0, two interleaved rounds (tg128 t/s):**

| Model | stock | single-token tuned (1w/4r) | **best** |
|---|---|---|---|
| Qwen3.5-2B Q4_0 + Q4_0 head | 23.40 / 23.40 | 37.57 / 37.21 | **41.69 / 40.75** |
| Llama-2-7B Q4_0 | 8.09 / 8.40 | 12.07 / 13.58 | **13.59 / 13.26** |

Prompt processing is unchanged by any of this (pp512 ≈ 284–289 and ≈ 64.5).

**MoE Qwen3.6-35B-A3B UD-Q3_K_XL, 4 GPUs layer split:**

| Build | pp512 | tg128 |
|---|---|---|
| single-token tuned | 68.7 | 14.51 |
| **best** | **72.2** | **15.01** |
| best + MoE 4 rows | 70.1 | 15.00 |

**MoE batched decode** (llama-batched-bench, 4 GPUs, pp128 tg64 per sequence; total generation t/s), the path that uses `mul_mat_vec_q_moe`:

| parallel seqs | single-token tuned | best (spill fix) | best + MoE 4 rows |
|---|---|---|---|
| 1 | 14.00 | 14.78 | 14.60 |
| 2 | 18.93 | 20.92 (+11 %) | **21.91** (+16 %) |
| 4 | 22.26 | 24.67 (+11 %) | **26.24** (+18 %) |
| 8 | 20.42 | 24.11 (+18 %) | **24.39** (+19 %) |

The spill fix is worth +11–18 % on the multi-token MoE path. 4 MoE rows per block adds a few percent more at 2–4 sequences.

## Phase 9: MoE MTP speculative decoding, MoE ubatch, power caps (best build)

**MTP speculative decoding** (Qwen3.6-35B-A3B-MTP UD-Q4_K_XL, built-in MTP heads, `--spec-type draft-mtp`, 4 GPUs layer split, llama-server, temp 0, 3 prompts × 64 tokens):

| `--spec-draft-n-max` | generation t/s | acceptance |
|---|---|---|
| off | 14.66 | n/a |
| 1 | 16.99 (+16 %) | 86.1 % (87/101) |
| **2** | **17.72 (+21 %)** | 75.8 % (113/149) |
| 3 | 15.90 (+8 %) | 60.3 % (120/199) |

Unlike a separate 0.8B draft on the 2B (−56 %), the MTP heads are nearly free and MoE verify batches are cheap. **Use `--spec-type draft-mtp --spec-draft-n-max 2`.**

**MoE prompt processing ubatch** (pp2048, `-b 4096`): `-ub 1024` 99.3, **`-ub 2048` 109.4 t/s** (vs ~85 at the default 512).

**Power-cap sweep** (all four M10 GPUs capped; restored to 53 W afterwards and verified). Mean power is per GPU over 1 Hz samples where that GPU was > 50 % busy:

| cap | 2B pp512 | 2B tg128 | 2B mean W | MoE pp512 | MoE tg128 | MoE mean W |
|---|---|---|---|---|---|---|
| 53 W (stock) | 286.8 | 41.03 | 51.4 | 87.2 | 15.83 | 45.3 |
| 45 W | 265.8 | 35.75 | 43.4 | 77.5 | 15.79 | 37.2 |
| 38 W | 204.2 | 28.08 | 36.5 | 66.7 | 15.36 | 34.8 |
| 30 W | 155.9 | 22.12 | 28.6 | 48.2 | 15.10 | 29.3 |

- A single GM107 running a small dense model is **power-limited at the stock 53 W**: speed falls almost in proportion to the cap. Tokens per joule stays ~0.77–0.82, so there is no efficiency sweet spot.
- **MoE decode across 4 dies barely cares:** 30 W costs 4.6 % of decode speed while each die draws ~29 W instead of ~45 W. Each die only works ~1/4 of the time with a layer split. For a decode-heavy MoE service, a 30–38 W cap (the-mighty-m10 uses 30 W) saves about a third of the power. Prompt processing does pay (−45 % at 30 W).

## Phase 11: recommended configs confirmed (best build, 4 GPUs)

**Nemotron-3.5-Lightning-30B-A3B UD-Q4_K_XL**, layer split, `-b 4096`:

| `-ub` | pp2048 t/s | tg128 t/s |
|---|---|---|
| 512 | 71.4 ± 0.3 | 17.39 ± 0.06 |
| 2048 | **99.5** ± 4.6 | 17.31 ± 0.01 |

**Recommended Qwen3.6-35B-A3B serving config** (llama-server, UD-Q4_K_XL, `-sm layer -ts 1/1/1/1 -b 4096 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2`, temp 0, 3 prompts × 64 tokens):

| M10 power cap | generation t/s | MTP acceptance |
|---|---|---|
| 53 W (stock) | 16.83 | 75.8 % |
| **38 W** | **16.90** | 75.8 % |

**A 38 W cap costs nothing for MoE + MTP serving.** (16.8 here vs 17.7 in phase 9 with the default ubatch is within run-to-run variation on 3 short prompts.) Power limits restored to 53 W and verified.

## Phase 12: CUDA graphs on Maxwell

Mainline turns CUDA graphs off below Volta (`ggml_cuda_graph_set_enabled`: `cc < GGML_CUDA_CC_VOLTA`). Patch define `M10_GRAPHS` skips that gate. On top of the best build, GPU 0, tg128, 3 reps, two interleaved rounds, production idle:

| Model | best r1 / r2 | best + graphs r1 / r2 | gain |
|---|---|---|---|
| Llama-2-7B Q4_0 | 13.29 / 12.75 | **15.26 / 14.98** | **+16 %** |
| Qwen3.5-2B Q4_0 + Q4_0 head | 42.53 / 42.17 | 41.72 / 42.27 | 0 % |
| Qwen3.6-35B-A3B Q4 (4 GPUs) | 12.48 ± 3.15 | 13.05 ± 3.66 | inconclusive (both runs noisy; phase 9 measured 15.8 ± 0.01 for best) |

- **Correctness:** Llama-2-7B wikitext-2 perplexity (40 chunks) is 5.5968 ± 0.12438 with and without graphs, identical per chunk. A 96-token greedy generation (`llama-simple`) gives **byte-identical text** with and without graphs.
- Why Qwen3.5 gains nothing (inferred): its hybrid recurrent graph is not reused between tokens, so graphs fall back to normal launches. Not checked with the graph debug log.
- Recommendation: enable for dense non-hybrid models. Neutral elsewhere.

## Speculative decoding with a small draft (first test)

- `llama-speculative-simple` cannot run Qwen3.5: "speculative decoding will use checkpoints (context does not support partial sequence removal)" then "failed to initialize speculative decoding". The Gated DeltaNet recurrent state cannot be partly rolled back. llama-server handles it with checkpoints (`scripts/spec-server-test.sh`).
- llama-server, one GM107, tuned build (1w/4r), Qwen3.5-2B Q4_0 + Q4_0 head target, temp 0, 3 prompts × 32 tokens:

| Setup | generation t/s | draft acceptance |
|---|---|---|
| no draft | **38.07** | n/a |
| + Qwen3.5-0.8B Q4_0 draft, `--spec-type draft-simple --spec-draft-n-max 4`, same GPU | 16.91 | 62.2 % (56/90) |

- **Speculative decoding with a small draft is a large loss on the M10** (−56 %) despite decent acceptance. Likely causes (inferred, not profiled): the 0.8B draft costs a big share of a 2B step on a compute-bound chip, and each verify pays for a recurrent-state checkpoint. Built-in MTP on the bigger Qwen3.6-35B-A3B is still to test (first attempt starved while a build ran in parallel).

## Just for fun: Qwen3.8-27B on all four M10 GPUs

UD-Q5_K_XL (20.9 GB), tuned build (1 warp / 4 rows), `-ngl 99 -fa 1 -ts 1/1/1/1`, all 4 GM107s, 2 reps. Production was busy on the P100s.

| Split mode | pp512 t/s | tg64 t/s |
|---|---|---|
| `-sm layer` (GPUs take turns) | 18.55 ± 0.07 | 2.43 ± 0.02 |
| **`-sm tensor`** (all 4 GPUs work on each layer) | **58.50 ± 0.14** | **6.65 ± 0.01** |
| `-sm row` | fails: "device CUDA0 does not support split buffers" (replaced by tensor mode in mainline) | |

- Tensor split is **2.7x faster at generation** than layer split: all four GPUs read their share of the weights at once (~139 GB/s effective vs ~51 GB/s).
- For scale: the same model family on 2x P100 (production, Q6_K_XL, MTP) does ~37 t/s. The 4x M40 12GB report on Reddit had Qwen3.6-27B at 8 t/s TG, 26 t/s PP.
- Gotcha: in llama-bench, `-ts 1,1,1,1` means four separate tests of `ts=1` (everything lands on GPU 0 → OOM). The per-GPU split needs slashes: `1/1/1/1`.

## Why generation is compute-bound

- Why: compute capability 5.0 has no `dp4a`. llama.cpp's fallback (common.cuh) does 4 byte multiplies. nvcc compiles that to 4 `XMAD.S16` + ~6 extract/shift ops (checked in SASS, `experiments/dp4a/`), about 10 instructions where Pascal uses 1. MMQ is disabled below cc 6.1 (mmq.cu:446), so prompt processing goes through dequantize + FP32 cuBLAS.

## Role models (2026-11-08): first results, one GM107 each

Builds: `build-best` (or `build-best-graphs` for the dense non-hybrid coders). Servers run with `--cache-ram 0` (the default host prompt cache grew a server to 4.1 GB RSS and got it OOM-killed). HumanEval = all 164 problems, greedy, pass@1, code run in a no-network sandbox (sandbox validated with reference solutions). OCR = 24 synthetic images (sentences, numbers, code, receipts; clean/blur/rotate/noise). Retrieval = BEIR SciFact (5,183 docs, 300 test queries).

| Role | Model / quant | pp512 t/s | tg128 t/s | Quality |
|---|---|---|---|---|
| General | Qwen3.5-4B Q4_0 + Q4_0 head | 111.3 | 18.3 | wikitext PPL 9.69; **HumanEval 76.8 %** (thinking off); thinking on, first 40 problems: 82.5 % (33/40) at ~10x the tokens |
| Coder | Seed-Coder-8B-Instruct Q4_0 | 61.6 (graphs 67.8) | 11.3 (**graphs 13.1**) | **HumanEval 82.9 %** (136/164) |
| Coder | **Qwen2.5-Coder-7B-Instruct Q4_0** (chosen) | | 12.9 (graphs) | **HumanEval 89.0 %** (146/164) |
| General (bigger) | Qwen3.5-9B Q4_0 (stock) | 66.2 | 10.6 (graphs) | not evaluated: 4B is ~1.8x faster and fits beside mmproj |
| Vision | Qwen3-VL-4B UD-Q4_K_XL + mmproj F16 | | | OCR 24/24 exact, 6.4 s/image |
| Vision | Qwen3.5-4B Q4_0 + its mmproj F16 | | | OCR 24/24 exact, 6.1 s/image |
| Embedding | Qwen3-Embedding-0.6B Q8_0 | | | SciFact nDCG@10 0.701, R@100 0.947, **1.7 docs/s** |
| Embedding | embeddinggemma-300M Q8_0 | | | **SciFact nDCG@10 0.789**, R@100 0.987, **7.0 docs/s** |
| Reranker | Qwen3-Reranker-0.6B Q8_0 over Qwen3-Embedding top 20 | | | nDCG@10 0.701 → **0.759**, 13.4 s/query (stock build) |
| Reranker | same, over embeddinggemma top 20, `build-best-rr` (#30422 fix) | | | nDCG@10 0.789 → 0.782 (**no gain**), 13.5 s/query (**fix gives no speedup here**) |

**MTP on the dense 4B** (`unsloth/Qwen3.5-4B-MTP`, requantized Q4_0 + Q4_0 head, llama-server, 4 chat prompts x 256 tokens, greedy, thinking off, GPU 2):

| Setting | Mean t/s | Draft acceptance |
|---|---|---|
| no MTP (same file) | 18.8 | |
| `--spec-type draft-mtp --spec-draft-n-max 1` | **21.8 (+16 %)** | 83 % |
| n-max 2 | 21.0 | 69 % |
| n-max 3 | 18.6 | 55 % |
| n-max 1 + mmproj loaded | 21.0 | |

MTP works on Maxwell for this dense hybrid model (Pascal is reported unsupported elsewhere; not an issue on sm_50 here). n-max 1 is best on a dense model, unlike the MoE (n-max 2). MTP and the vision projector load together.

- **Kev-0.8B** (decision model, jaredpalmer/kev @ 2074cc4): runs on one GM107 in PyTorch 2.8.0+cu126 (Python 3.12 image `m10-kev`, built on a second machine and shipped as a 3.8 GB gzip). `KEV_DTYPE=fp32` (Maxwell has no BF16/FP16 math), `KEV_FUSED=0`. The Gated DeltaNet and causal-conv layers use transformers' slow PyTorch reference path (flash-linear-attention / causal_conv1d not installed). README ticket example (3 questions in one pass): **0.36 s warm**, 1.1 s cold, deterministic. Answers: department shipping 0.46 / billing 0.28 / returns 0.26, escalate 0.55, frustration 1.23 (Kev-4B in the README picks returns). The base model loads offline by mounting it at `./Qwen/Qwen3.5-0.8B-Base` in the working directory.
- The synthetic OCR set is too easy to separate the two vision models. Qwen3.5-4B with its projector matches Qwen3-VL-4B, so one GPU can serve both general and vision.
- embeddinggemma beats Qwen3-Embedding on both quality and speed here. Hermes round 13 found the likely Qwen3-Embedding cause: llama.cpp issue #14234, where ggerganov's fix is to append the EOS token (`<|endoftext|>`) to each input by hand. Not re-run (a full SciFact pass takes ~50 min at 1.7 docs/s, and embeddinggemma is already 4x faster).
- The reranker only helps the weaker embedder. With embeddinggemma it slightly hurts and costs ~13 s per 20-doc query, so for retrieval use embeddinggemma alone. The #30422 logits fix does not change reranker speed on this card (the time is in the forward passes, not the logits copy).
- CUDA graphs help the dense coders (+16 % tg, +10 % pp), as on Llama-2.

## Phase 13: upstream-style patches verified (f477f795c)

`patches/upstream/01` (Maxwell MMVQ table), `02` (vmad), `03` (MoE no local memory). These are clean versions without the `M10_*` defines. 01 is on branch `cuda-maxwell-mmvq` (uncommitted). GPU 0, tg128, two rounds:

| Build | r1 | r2 |
|---|---|---|
| stock | 23.42 | 23.40 |
| upstream 01 | 37.89 | 37.59 |
| experiment equivalent (`build-b-nw1rpb4`) | 37.64 | 37.28 |
| upstream 01+02+03 | 42.45 | 41.37 |
| experiment equivalent (`build-best`) | 42.19 | 41.78 |

Batched (01+02+03): 1 → 41.7, 2 → 59.0, 4 → 75.8, 8 → 84.5 t/s total. Correctness: MUL_MAT n=1..8 1637/1637, MUL_MAT_ID 931/931 for both builds. Stack frames: 01+02+03 has 317/317 mmvq kernels at STACK:0. 01 alone leaves the 18 MoE kernels (fixed by 03) plus 6 low-bit kernels at 5–8 columns (Q2_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) with 72–160-byte spills, which vmad removes.

## ik_llama.cpp: cannot run on this host CPU

Approved and attempted 2026-11-08 (`/data/m10/ik_llama.cpp`, `d2488475`):
- The build needs CMake >= 3.25 (CUDA20 dialect). Used Kitware CMake 3.31.6 inside the container.
- **Native build fails:** ik_llama.cpp's CPU code needs AVX2. `ggml_vec_simd_sigmoid_f32` / `ggml_vec_sigmoid_mul_f32` exist only for NEON, AVX-512 and AVX2+FMA, and `tq-lut-i8.h` also assumes AVX2. The test box's i7-3930K (Sandy Bridge) has AVX but no AVX2. A local scalar fallback for the two functions got past the first error but not `tq-lut-i8.h`.
- **AVX2 build** (`-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON`) compiles, but `llama-bench` dies with **"Illegal instruction"** (rc 132) right after CUDA init, even with every layer on the GPU.
- Conclusion: comparing ik_llama.cpp on the M10 needs a host CPU with AVX2 (Haswell or newer). mb8565's published M10 numbers (+16.8 % with 2 rows, etc.) remain the only ik data on this card.
