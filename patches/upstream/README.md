# Upstream-style patches for llama.cpp (Maxwell / sm_50)

Clean versions of the M10 tuning, without the experiment `M10_*` defines, against upstream llama.cpp `f477f795c` (2026-11-08). Each patch has one purpose, so they can be reviewed separately:

- `01-cuda-maxwell-mmvq-table.patch`: Maxwell entry in the MMVQ launch-parameter table (the big win)
- `02-cuda-maxwell-vmad-dp4a.patch`: native `VMAD.S8.S8` in place of the missing `dp4a`
- `03-cuda-mmvq-moe-no-local-mem.patch`: removes a local-memory spill in the MoE mat-vec kernel (not Maxwell-specific)

Apply with `git apply` on a llama.cpp checkout at or near `f477f795c`. Results and raw data are in `../../RESULTS.md` (Phase 13) and `../../results/`.

## Patch 01: Maxwell entry in the MMVQ parameter table

**What it does:** adds `MMVQ_PARAMETERS_MAXWELL`, chosen for NVIDIA devices whose highest compiled arch is below sm_60 (device side: `__CUDA_ARCH__ < GGML_CUDA_CC_PASCAL`). For that table, `calc_rows_per_block` returns 4 for 1–8 columns. `calc_nwarps` has no Maxwell case, so it falls through to its existing default of 1 warp.

**Why it helps:** sm_50 has no `dp4a`, so `mul_mat_vec_q` is compute-bound, not bandwidth-bound (stock reached ~27 % of the 83 GB/s on a GM107; nvprof showed 86 % of decode time in mmvq). With 4 warps on one row, each row ends in a shared-memory reduction and `__syncthreads`, and each warp re-reads the q8_1 activation from L2 for every row. With 1 warp and 4 rows, there is no cross-warp reduction and the activation loads are reused across 4 rows. Only 5 SMs, so occupancy is not the limit.

**Evidence** (all in RESULTS.md, raw JSONL in results/):
- Sweep of warps x rows on Qwen3.5-2B Q4_K_M: 8w/1r 15.8, stock 4w/1r 18.4, 1w/1r 22.5, 2w/2r 25.0, 1w/2r 26.3–27.1, **1w/4r 28.0**, 1w/8r 26.9 t/s.
- Every quant type gained: Q4_0 +64 %, IQ4_NL +66 %, Q4_K_M +57 %, Q6_K +43 %, Q8_0 +34 %, Llama-2-7B Q4_0 +56 %.
- Batches of 2–8 columns: +29 % at 2, +21 % at 8 (1w/4r best; 8 rows collapses at 6–8 columns).
- MoE decode (1 column, has ids, goes through the same kernel): Qwen3.6-35B-A3B +41–54 %.
- `test-backend-ops -o MUL_MAT` n=1..8: 1637/1637 pass. Phase 13 re-verifies this exact upstream diff.

**Verified as an upstream diff (phase 13, f477f795c + patch 01 only):** Qwen3.5-2B Q4_0 tg128 37.89 / 37.59 t/s vs stock 23.42 / 23.40, the same as the experiment build (37.64 / 37.28). MUL_MAT n=1..8 1637/1637, MUL_MAT_ID 931/931. **Known wart:** with patch 01 alone, `cuobjdump -res-usage` shows 6 extra kernels with 72–160-byte stack frames (register spills): `mul_mat_vec_q` for Q2_K at 7–8 columns, and IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S at 5 columns. With patch 02 (vmad) also applied, all 317 mmvq kernels have STACK:0. Options if a reviewer asks: keep 2 rows for those types above 4 columns, or land 02 first. Their speed impact was not measured (rare types, batches of 5+).

**Likely review questions:**
- *Only measured on sm_50?* Yes, Tesla M10 (GM107). sm_52 (GTX 9xx, M40, M60) is gated in too and was not measured. The gate could be narrowed to `== 500` if needed.
- *Prior art?* ik_llama.cpp #2701 (2 rows on sm_5x/sm_60, +16.8 % on an M10) and #2740 (2 warps for Q2_K/Q3_K only). Our sweep shows 1 warp is the bigger win on Maxwell for all types. shinbunbun/llama-cpp-p100-patches #02 used 4 rows with 4 warps on P100.
- *Why not change GENERIC?* GENERIC covers Pascal and others, where 1 warp was not measured. ik #2701 found sm_61 neutral or slower with forced rows.

## Patch 02: `vmad` dp4a for Maxwell (separate PR)

In `ggml_cuda_dp4a`, the non-dp4a fallback below sm_60 becomes 4 PTX `vmad.s32.s32.s32` with byte selectors. ptxas maps them to native `VMAD.S8.S8` on sm_50 (checked in SASS: a test loop goes from 42 to 30 instructions; `experiments/dp4a/`). Measured on top of patch 01: Q4_0 +11.4 %, Q4_K_M +6.2 %, Llama-2-7B +7.4 %, Q8_0 +2.4 %. test-backend-ops passes. The idea comes from shinbunbun's P100 patch #01 (gated >= sm_60 there). Credit to that patch.

## Patch 03: MoE mmvq kernel without local memory (separate PR)

`mul_mat_vec_q_moe` reads `tmp[threadIdx.x]` / `tmp_gate[threadIdx.x]`. Dynamic indexing forces the arrays into local memory. On sm_50, `cuobjdump -res-usage` shows 18 `mul_mat_vec_q_moe` kernels with STACK:16, and 0 after the fix (an unrolled compare with a compile-time index). The fix is bit-identical and not Maxwell-specific: check `-res-usage` on a newer arch to see whether it spills there too. Measured: MoE batched decode +11–18 % (2–8 sequences, Qwen3.6-35B-A3B on 4 GM107s). MUL_MAT_ID tests 931/931 pass. Same idea as ik_llama.cpp #2721 for their fused kernel. Mainline has no issue for this (searched).

## Not proposed upstream

- `M10_GRAPHS` (CUDA graphs below Volta): +16 % on dense Llama-2 with byte-identical output, no change on Qwen3.5. Upstream disables graphs below Volta on purpose, and open PR #30727 touches this. Discuss in an issue first.
- The K-quant non-contiguous MUL_MAT assert (`convert_func != nullptr`, ggml-cuda.cu:1710, Q4_K/Q5_K/Q6_K views on sm_50) is a pre-existing bug. Worth an issue report on its own.
