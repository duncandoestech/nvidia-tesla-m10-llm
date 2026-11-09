# M10 / Maxwell LLM inference — research round 6

Delivered: /tmp/m10-research-round6.md
Tools: firecrawl-backed web_extract (api.github.com issue/PR search + raw file fetch, huggingface.co raw config.json, docs.litellm.ai, markaicode.com, raw.githubusercontent.com), web_search. One file written, no other hosts/commands.
Rule note: [SNIPPET] = rests on search-result text only. Everything else was fetched in full this round (source URL given). All PR numbers/tables quoted verbatim from API-fetched PR bodies.

---

## 1. Task 1 — mb8565: complete 2026 inventory (ik_llama.cpp + ggml-org/llama.cpp)

### 1.1 Totals (search API, this round)

- ik_llama.cpp, author:mb8565, created 2026-01-01..2026-11-08: **total_count = 59** (incomplete_results: false). All 59 are PRs; there are no issues by mb8565 in the repo in this window.
  Query: https://api.github.com/search/issues?q=repo:ikawrakow/ik_llama.cpp+author:mb8565+created:2026-01-01..2026-11-08&per_page=100
- ggml-org/llama.cpp, same author + date range: **total_count = 0**.
  Query: https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+author:mb8565+created:2026-01-01..2026-11-08&per_page=100
- ggml-org/llama.cpp, same author, **all time**: **total_count = 0**.
  Query: https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+author:mb8565&per_page=30&sort=created&order=desc
  -> mb8565 has never opened a PR or issue against mainline ggml-org/llama.cpp. All 2026 work (and all of their public PR history visible to the search API) is against ikawrakow/ik_llama.cpp.
- Discussions by mb8565 in either repo: **0** (search API, `is:discussion` qualifier, both repos, total_count 0 each). Note: the search API's discussion coverage is the same index as issues; treat "0 discussions" as "none found", and both mb8565's forks report has_discussions=false.
- mb8565's repos (api.github.com/users/mb8565/repos): exactly two —
  - mb8565/ik_llama.cpp (fork of the ik fork; created 2026-06-24, pushed 2026-11-06; description "llama.cpp fork with additional SOTA quants and improved performance")
  - mb8565/llama.cpp (fork of mainline; created 2026-10-05, pushed 2026-10-05)
- Activity window: first 2026 PR created 2026-06-24 (#2019), last 2026-11-06 (#2860, merged 2026-11-08).
- Merge state of the 59: **55 merged, 3 closed-unmerged (#2082, #2151, #2192), 1 open (#2859)**.

### 1.2 The 59 PRs (ik_llama.cpp, mb8565, 2026)

State = merged (date) / closed-unmerged (date) / open.

| # | Created | State | Title |
|---|---|---|---|
| 2860 | 2026-11-06 | merged 11-08 | graphs : restore the FFN LayerNorm for starcoder2, starcoder, gpt2, bloom, mpt, codeshell, orion, jais |
| 2859 | 2026-11-06 | **open** | iqk : take the multi-head indexed FA gather only when heads share a row |
| 2793 | 2026-10-29 | merged 10-29 | convert : add MiniMax-M3 and M3.1 |
| 2792 | 2026-10-29 | merged 11-02 | minimax-m3 : convert exact F32 indexer projections to BF16 at load |
| 2790 | 2026-10-29 | merged 11-02 | minimax-m3 : sparse attention (MSA) on the CPU backend, behind --dsa |
| 2789 | 2026-10-29 | merged 10-29 | ggml : run pool_1d on all threads |
| 2788 | 2026-10-29 | merged 10-29 | minimax-m3 : load M3.1 GGUFs by skipping the indexer tensors on every layer |
| 2778 | 2026-10-28 | merged 10-28 | Refuse router tensor overrides with split routed experts in split mode graph |
| 2775 | 2026-10-28 | merged 10-29 | Fix -sm graph abort and Cohere partial drop with uneven splits |
| 2773 | 2026-10-28 | merged 10-28 | Gemma4: fix -sm graph crash with routed experts on the CPU |
| 2744 | 2026-10-24 | merged 10-25 | Fix -sm attn assert for Flint and Qwen3-MoE |
| 2740 | 2026-10-23 | merged 10-23 | CUDA: use 2 warps in MMVQ for Q2_K and Q3_K with one column on older GPUs |
| 2726 | 2026-10-22 | merged 10-22 | sampling : normalize probabilities in the DIST sampler |
| 2722 | 2026-10-21 | merged 10-23 | Refuse -ot layouts that break split mode graph |
| 2721 | 2026-10-21 | merged 10-21 | CUDA: keep the fused MMVQ per-row sums in registers |
| 2720 | 2026-10-21 | merged 10-21 | server : do not add the tail probability mass twice in n_probs |
| 2719 | 2026-10-21 | merged 10-21 | Fix -sm graph for Cohere2 and Command-R when -ot moves a dense FFN |
| 2710 | 2026-10-20 | merged 10-20 | iqk: reserve the MLA flash-attention work buffer for every consumer branch |
| 2709 | 2026-10-19 | merged 10-20 | CUDA: do not fuse the MoE down projection when its output overlaps the up/gate inputs |
| 2708 | 2026-10-19 | merged 10-20 | CUDA: two correctness fixes for IQ1_S_R4 / IQ1_M_R4 |
| 2707 | 2026-10-19 | merged 10-20 | minimax-m3 : load mainline GGUFs that carry the indexer tensors |
| 2706 | 2026-10-19 | merged 10-20 | CUDA: do not clear padding inside the next expert when experts are offloaded |
| 2701 | 2026-10-19 | merged 10-19 | CUDA: use 2 rows per block in MMVQ for one column on sm_5x, sm_60 and sm_75 |
| 2696 | 2026-10-18 | merged 10-19 | HIP: make __vsub4 wrap per byte like CUDA |
| 2686 | 2026-10-16 | merged 10-16 | CUDA: fix the IQ5_KS MMQ scale stride on GPUs without int8 MMA |
| 2670 | 2026-10-14 | merged 10-14 | docs : add Pascal and Volta support status |
| 2669 | 2026-10-14 | merged 10-14 | iqk: do not use the float GEMM kernels when K is below one SIMD step |
| 2668 | 2026-10-14 | merged 10-15 | CUDA: use MMVQ for up to 16 columns on sm_60 |
| 2647 | 2026-10-11 | merged 10-12 | CUDA Pascal: run head size 512 flash attention on the GPU |
| 2646 | 2026-10-10 | merged 10-12 | CUDA: apply attention sinks in the wmma FA kernel |
| 2645 | 2026-10-10 | merged 10-12 | gpt-oss: fix race in fused CUDA expert selection |
| 2581 | 2026-10-01 | merged 10-02 | Fix V cache store for 3-D V tensors without flash attention (Gemma-4 -fa 0) |
| 2580 | 2026-10-01 | merged 10-02 | CUDA: route head size 256 FA prefill to tile_f32 on NVIDIA GPUs before Volta |
| 2575 | 2026-09-30 | merged 10-01 | CUDA: run IQ6_K MMQ on the Q6_K integer-scale MMA kernel |
| 2574 | 2026-09-30 | merged 10-02 | CUDA: requantize blocks of 16 to Q8_0 in the MMQ loader |
| 2566 | 2026-09-29 | merged 09-30 | CUDA: index the FA mask by its row stride in the wmma-f16 kernel |
| 2553 | 2026-09-28 | merged 09-28 | cuda : IQ4_K mmq tile load via __byte_perm instead of a paired table gather |
| 2510 | 2026-09-23 | merged 09-23 | cuda : opt-in FABsum variant of the fp16 tile flash-attention kernel for sm_60 |
| 2509 | 2026-09-23 | merged 09-23 | cuda : IQ4_K mmvq via __byte_perm instead of a scalar byte gather |
| 2442 | 2026-09-13 | merged 09-15 | iqk: reserve the flash-attention work buffer for the narrowed nek1 |
| 2291 | 2026-08-11 | merged 08-11 | llama: pass rope freq factors to build_std_attention |
| 2290 | 2026-08-11 | merged 08-11 | Fix rope type for 39 architectures |
| 2192 | 2026-07-27 | **closed-unmerged** 07-29 | ggml : fix vec_dot/vec_dot_type mismatch in non-IQK builds |
| 2188 | 2026-07-26 | merged 07-26 | sampling: fix out-of-bounds logits read when the vocab has no newline token |
| 2151 | 2026-07-18 | **closed-unmerged** 09-23 | CUDA sm_60: bound the fp16 flash-attention error on P100 prefill instead of disabling the fast path |
| 2150 | 2026-07-17 | merged 07-18 | perplexity: fix int overflows in large-context buffer sizing |
| 2149 | 2026-07-17 | merged 07-18 | Build the MROPE/IMROPE position sections in the legacy no-pos decode path |
| 2144 | 2026-07-16 | merged 07-17 | CUDA: use fp32 flash-attention vec kernel on P100 (sm_60) decode, fixes silent fp16 accumulation error |
| 2142 | 2026-07-16 | merged 07-17 | CUDA: faster exact-fp32 flash-attention tile kernel on Pascal prefill (S3 retile) |
| 2130 | 2026-07-14 | merged 07-14 | sm graph: reshape folded-head MLA tensors to 3D before the per-head split |
| 2087 | 2026-07-05 | merged 07-06 | CUDA: fix flash attention for gpt-oss (SWA + attention sinks) on the tile kernels (no-tensor-core GPUs) |
| 2082 | 2026-07-04 | **closed-unmerged** 07-06 | CUDA: guard the MoE row-scatter kernels against out-of-range routing indices |
| 2079 | 2026-07-03 | merged 07-06 | CUDA: enable MLA flash-attention decode on Pascal (vec_f32 for K=576/V=512), incl. Q8_0 KV |
| 2069 | 2026-07-02 | merged 07-02 | GLM-DSA: fix -fa 0 garbage perplexity on the batch>8 indexer path (private seed, do not corrupt KQ_mask) |
| 2045 | 2026-06-27 | merged 07-02 | deepseek2 : GLM-DSA sparse attention (lightning indexer), --dsa off by default |
| 2038 | 2026-06-26 | merged 06-26 | ggml : fix set_rows CPU crash when the destination is F32 |
| 2037 | 2026-06-26 | merged 06-26 | llama : fix CPU-only load crash on a CUDA build (device_mem out-of-bounds) |
| 2031 | 2026-06-25 | merged 06-25 | cuda : fix MLA flash-attn vec decode for asymmetric K/V head sizes |
| 2019 | 2026-06-24 | merged 06-24 | eval-callback : sum over the full tensor, not just the printed slice |

All URLs: https://github.com/ikawrakow/ik_llama.cpp/pull/<number>

### 1.3 Every M10 / sm_50 number they report

Only six PRs contain M10 or sm_50 content. All numbers below are verbatim from the API-fetched PR bodies.

**#2701 — "CUDA: use 2 rows per block in MMVQ for one column on sm_5x, sm_60 and sm_75"** (merged 2026-10-19)
https://github.com/ikawrakow/ik_llama.cpp/pull/2701
- "With one column, MMVQ computes one row per block. On sm_50, sm_60 and sm_75, two rows per block are faster. TG-128 of Llama-3.1-8B Q4_K_M goes up 12 % on a P100 and on a Quadro RTX 4000. On sm_61 and sm_70, forced two rows were neutral or slower."
- M10 row (TG-128, Llama-3.1-8B Q4_K_M, llama-bench): "M10 (sm_50), two rows forced | 6.60 | 7.70 | **+16.8 %**"
- 2x P100: -sm layer 42.23→46.71 (+10.6 %) / -sm graph 66.44→73.90 (+11.2 %); DeepSeek-V2-Lite REAP48 +10.9/+8.8 %. RTX 4000 (sm_75): 63.40→70.99 (+12.0 %). P40 forced: +0.3 / -4.5 %; Titan V forced: -15.2 / -5.4 %.
- Register-pressure data point: "Q3_K keeps one row, because with two rows its fused up/gate kernel goes from 116 to 195 registers on sm_60 and gets slower."
- "The result does not follow dp4a: sm_61, sm_70 and sm_75 all have dp4a, and only sm_75 gets faster."
- P100 `llama-sweep-bench -c 16384 -ub 512 -ngl 99 -fa 1`, Llama-3.1-8B Q4_K_M: S_TG +10.8 % at N_KV 0 down to +7.3 % at 15872; S_PP unchanged (477.4→476.6). "The PR saves 2.3 ms per token at every context."
- Host/device caveat: "I know a host/device mismatch here is what #446 fixed. … With `70-real;75-virtual`, an sm_75 GPU runs the sm_70 code but the host uses 750, and test-backend-ops fails for one column."
- PPL (ctx 512, -ub 1, 4 chunks), P100: 4.9096 Q4_K_M, 5.1215 Q3_K_M (main = PR). "I have not tested sm_52, sm_53 or other Turing cards."

**#2721 — "CUDA: keep the fused MMVQ per-row sums in registers"** (merged 2026-10-21) — the sm_50 local-memory-spill fix
https://github.com/ikawrakow/ik_llama.cpp/pull/2721
- "#2701 uses two rows per block in MMVQ for one column on sm_5x, sm_60 and sm_75. With two rows, `fused_mul_mat_vec_q` gets a **32-byte local memory stack frame on sm_5x and sm_60**. The cause is `tmp_u[j][threadIdx.x]`, `tmp_g[j][threadIdx.x]` and the two `local_bias` arrays. They are indexed with `threadIdx.x`, so the compiler puts them in local memory."
- "In my sm_60 build (CUDA 12.4), **34 of the 66 one-column fused kernels had a stack frame. With the PR, none has one.** The Q4_K kernel loses its 16 local loads and stores. **My sm_50 build for the M10 is the same, 34 before and none after.** sm_75 also uses two rows, but had no stack frame. Its Q4_K fused kernels lose 64 to 72 instructions. TG on the RTX 4000 changes by less than the run-to-run spread on main (1.0 % and 1.3 %)."
- "With one row per block the SASS does not change. All 66 one-column fused kernels are identical to main on sm_61 (CUDA 12.9) and on sm_70 (CUDA 12.4)."
- -ub 1 PPL (ctx 512, 4 chunks; "2 on the M10"), main = PR: Llama-3.1-8B Q4_K_M: P100 4.9128, 2x P100 -sm layer / -sm graph 4.9128 / 4.9011, RTX 4000 4.9125, P40 4.9109, Titan V 4.9032, **M10 (2 chunks) 5.3221**; DeepSeek-V2-Lite REAP48 Q4_K_M: 15.2952 (P100), 15.0585 (2x P100 graph); gpt-oss-20b Q4_K_M: 7.1946 (P100). `--kl-divergence-base` files byte-identical main vs PR on P100 (Llama-3.1-8B, DeepSeek-V2-Lite, gpt-oss-20b).

**#2740 — "CUDA: use 2 warps in MMVQ for Q2_K and Q3_K with one column on older GPUs"** (merged 2026-10-23)
https://github.com/ikawrakow/ik_llama.cpp/pull/2740
- "With one column, `mul_mat_vec_q` uses 4 warps. This PR uses 2 for Q3_K below Ampere and for Q2_K below Turing."
- Bench: "tg128 t/s, Llama-3.1-8B-Instruct, `llama-bench -p 0 -n 128 -r 5 -ngl 99 -fa 1` (`-r 3` on M10). Runs alternate main 55d5dd9f and this patch on it."
- M10 rows: "M10 (sm_50) | Q3_K_M 5.80 → 6.38 (**+10.0 %**) | Q2_K 8.445 → 9.410 (**+11.4 %**)"
- **Multi-GPU -sm on M10**: "2x M10, `-sm layer` | 5.573 → 6.506 (**+16.7 %**)" and "2x M10, `-sm graph` | 10.25 → 11.16 (**+8.9 %**)" (Q3_K_M).
- Controls: P100 +9.7/+9.9 %, P40 +7.9/+4.7 %, Titan V +7.3/+6.5 %, RTX 4000 +5.6 % (Q3_K_M) / "same kernel" (Q2_K); "The Q4_K_M control launches the same kernels as main and changes by 0.12 % or less."
- Upstream cross-references in the body: "llama.cpp #29753 (sm_70) and #25479 (sm_61, open) use 2 warps for more types. Here, Q4_K, Q5_K, Q6_K (P40, Titan V), Q4_0 and Q8_0 (P40) were not faster."

**#2580 — "CUDA: route head size 256 FA prefill to tile_f32 on NVIDIA GPUs before Volta"** (merged 2026-10-02)
https://github.com/ikawrakow/ik_llama.cpp/pull/2580
- "On Pascal and Maxwell (P100 sm_60, P40 sm_61, **M10 sm_50**), flash attention prefill at head size 256 runs the vec kernel on main. … This PR sends D=256 prefill (batch > 8) to `flash_attn_tile_ext_f32`, which uses each K/V tile for 16 query columns and accumulates in fp32. Decode (batch <= 8) does not change."
- **M10 table (verbatim): "Tesla M10 (Maxwell sm_50, one GM107 die), Falcon3-10B, 32cddbfc vs PR, same flags, arms main, PR, PR, main. With f16 KV, pp2048 goes from 39.3 to 47.3 t/s (+20 %) and pp4096 from 32.6 to 44.1 (+35 %). With q8_0 KV, pp4096 goes from 13.7 to 44.0. With q4_0 KV, main runs this FA op on the host CPU (40 threads, AVX-512). There the PR is 1.6 % (pp2048) and 7.0 % (pp4096) slower than main. KLD against the CPU with f16 KV, n_ctx 2048, 4 chunks: 0.000679 on main, 0.000680 with the PR."**
- P100 table (main 0821d62a, `llama-bench -ub 512 -ngl 99 -fa 1 -r 3`), Falcon3-10B Q4_K_M f16: pp2048 272.4→296.7 (+9.0 %), pp8192 169.4→262.7 (+55.1 %), pp16384 116.5→226.7 (+94.5 %); q8_0: +20.5/+83.6/+147.8 %; q4_0 (CPU on main): +9.1/+19.7/+29.5 %. Gemma-4-12B Q4_K_M f16: +9.3/+10.2/+9.4 %; q8_0: +21.6/+24.2/+22.7 %; q4_0: +8.0/+3.7/+3.1 %. Llama-3.1-8B D=128 control: -0.1 %/0.0 %. "In the same runs, tg64 moves by 0.1 % or less with f16 and q8_0 KV."
- Single P40, Falcon3-10B, pp8192: f16 246.0→448.7 t/s; q8_0 32.7→445.8 t/s; "f16 KV goes … five quantized KV type pairs now run on the GPU instead of the CPU" (q4_0, iq4_nl, q8_0/q6_0, q8_0/iq4_nl, q6_0/q5_0 abort on main because that P40 host has no AVX2; "With the PR they run at about 446 t/s").
- Two-GPU pp8192 (Falcon3-10B, `-w 0`): 2x P100 (P2P) layer f16 295.9→457.5, q4_0 208.4→458.9 (CPU FA on main); graph f16 350.9→480.5, q4_0 311.8→483.1; 2x P40 (no P2P) layer f16 426.9→778.8, q4_0 aborts→774.6; graph f16 456.2→778.5, q4_0 aborts→774.4. "tg64 with f16 KV does not change in any of these. In graph mode the profiler shows the tile kernel on both GPUs."
- PPL/KLD (P100, n_ctx 8192): Falcon3 f16 PPL 2.5151→2.5149, KLD 0.000665→0.000666; Gemma-4-12B f16 13.874→13.761.
- Test-setup line, verbatim: **"I have tested on P100 (one and two GPUs), P40 (one and two GPUs) and a Tesla M10 (sm_50). The route also applies to sm_62, which I have not tested."**

**#2668 — "CUDA: use MMVQ for up to 16 columns on sm_60"** (merged 2026-10-15) — no M10 numbers
https://github.com/ikawrakow/ik_llama.cpp/pull/2668
- Extends the sm_60 MMVQ column limit to 16 (spec-decode verify batches up to 16 tokens on P100); P40 unchanged (already has MMQ). All benchmark numbers are P100 (sm_60); the PR explicitly notes it "tested a `50;61` build on the P100" — i.e. the sm_50 code path compiles but is not perf-tested. (No M10 t/s in the body.)

**#2670 — "docs : add Pascal and Volta support status"** (merged 2026-10-14)
https://github.com/ikawrakow/ik_llama.cpp/pull/2670
- Adds a Pascal/Volta support-status doc. sm_50 is covered via the sm_60 shim (P40 sm_61 likewise); tested on P100 + Titan V; the doc states P40 gets the FA kernel but not the MMQ path. No M10 hardware.

### 1.4 Their M10 / test setup as stated in the PRs

Assembled only from verbatim PR statements (no guessing):
- GPUs: **Tesla M10 (sm_50)**, P100 (one and two), P40 (one and two), Titan V, Quadro RTX 4000, RTX 3060-adjacent sm_75, plus a 60 W-capped RTX A2000 (sm_86, excluded from results). "I have tested on P100 (one and two GPUs), P40 (one and two GPUs) and a Tesla M10 (sm_50)" (#2580).
- M10 usage: single die for `-ngl 99` runs (captioned "one GM107 die" in #2580) and **2x M10 multi-GPU with `-sm layer` and `-sm graph`** (rows in #2740, PPL rows in #2721, forced-rows row in #2701).
- CPUs: "2x Xeon E5-4669 v4" (CPU baseline runs, `CUDA_VISIBLE_DEVICES="" -ngl 0`, #2860); "Xeon Platinum 8260, one socket, 24 threads" (#2726 CPU device).
- Tools: `llama-bench -p 0 -n 128 -r 5 -ngl 99 -fa 1` with **`-r 3` on M10** (#2740); `llama-bench -ub 512 -ngl 99 -fa 1 -r 3` (#2580); `llama-sweep-bench -c 16384 -ub 512 -ngl 99 -fa 1` (#2701) and `-c 32768` (#2580); `llama-perplexity -c 512 --chunks 4 -b 512 -ub 1` (+ `--no-escape`, `-fa 1`); KLD via `--kl-divergence-base`.
- Builds: sm_60 build on CUDA 12.4, sm_61 on CUDA 12.9, sm_70 on CUDA 12.4 (#2721); the M10 sm_50 build's CUDA version is not stated in the bodies; "Prepared with AI assistance; I ran the measurements and reviewed the change" (#2580, #2726).

### 1.5 Theme index (for the topics you named)

- **MoE**: #2778 (split-mode graph + split routed experts), #2773 (Gemma4 -sm graph crash, routed experts on CPU), #2744 (Flint/Qwen3-MoE -sm attn assert), #2709 (don't fuse MoE down-proj when output overlaps up/gate inputs), #2706 (expert offload padding), #2645 (gpt-oss fused expert selection race), #2082 (MoE row-scatter OOB guard — closed unmerged), #2045/#2069 (GLM-DSA sparse attention), #2793/#2792/#2790/#2788/#2707 (MiniMax-M3/MSA).
- **Flash attention**: #2859 (open: multi-head indexed FA gather), #2710 (MLA FA work buffer), #2647 (Pascal D=512 FA on GPU), #2646 (attention sinks in wmma FA), #2580 (pre-Volta D=256 FA prefill → tile_f32), #2566 (FA mask row stride), #2510 (sm_60 fp16 FA FABsum opt-in), #2442 (FA work buffer nek1), #2151 (sm_60 fp16 FA error bound — closed unmerged), #2144 (P100 fp32 FA vec decode), #2142 (Pascal prefill S3 retile), #2087 (gpt-oss FA on tile kernels), #2079 (Pascal MLA FA decode).
- **Prompt processing**: #2580 (pp2048–pp16384 tables, incl. M10), #2142/#2151 (P100 prefill), #2668 (16-column verify batches), #2701 (S_PP unchanged / S_TG +7.3..+10.8 %).
- **Multi-GPU -sm graph on M10**: the only explicit M10 multi-GPU numbers are in #2740 (2x M10: -sm layer 5.573→6.506 +16.7 %, -sm graph 10.25→11.16 +8.9 %, Q3_K_M) and the #2721 PPL rows (M10 -ub 1 = main). The -sm graph correctness cluster (#2778, #2775, #2773, #2744, #2722, #2719, #2130) was benchmarked on 2x P100 / P40, not M10; #2860 notes these archs fall back `-sm graph` → `-sm layer` and "On 2x P100, -sm layer gives the same PPL as one GPU".

---

## 2. Task 2 — mul_mat_vec_q_moe in mainline: spill / register pressure / rows per block on pre-Volta, and PR #20905

### 2.1 Every item mentioning `mul_mat_vec_q_moe` in ggml-org/llama.cpp (13 total)

Query: https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+%22mul_mat_vec_q_moe%22&per_page=30 (total_count 13)

| # | Kind | Title | State | Relevance |
|---|---|---|---|---|
| 28476 | PR | SYCL: Add IQ type handling for MoE | merged 2026-09-09 | SYCL only (Arc B60; tg512 11.17→17.92 +60.4 %) |
| 28113 | issue | Eval bug: CUDA/HIP: MoE models produce garbage output on RDNA3.5 (gfx1151) since #27621 | closed 2026-09-04 | HIP/RDNA3.5 regression |
| 27621 | PR | CUDA: extend MOE fusion to specdec, earlier MOE glu fusion and topk-router fusion were restricted to 1 token | merged | fusion scope |
| 24087 | PR | CUDA: enroll mul_mat_vec_q_moe into pdl | merged | PDL = sm_80+ feature |
| 27861 | PR | llama: GPU-resident LRU cache for host-offloaded MoE expert weights | merged | expert caching |
| 23539 | issue | mmvq.cu - mul_mat_vec_q: out-of-bounds read of vx in last block when nrows_x % rpb != 0 | closed not_planned (stale) 2026-07-07 | **the only rows-per-block item** |
| 25620 | issue | Eval bug: MMVQ host/device nwarps mismatch causes NaN on gfx11-generic / RDNA3.5 | closed | host/device dispatch mismatch class |
| 21996 | PR | sycl: add batched mul_mat_id MMVQ kernel for MoE token generation | merged | SYCL only |
| 25618 | issue | Eval bug: Speculative decoding (draft-mtp / draft-dspark): greedy output diverges on quantized targets | closed | specdec |
| 21494 | issue | [M3 Ultra] Metal: 13 % throughput regression in tg128 vs March 2026 builds | closed | Metal |
| 24182 | issue | Misc. bug: ~4-5x performance regression with NVFP4 + tensor split after hparams refactor | closed | NVFP4 |
| **20905** | PR | **Optimize MOE GEMV kernel for BS > 1** | **merged 2026-03-29** | **the kernel rewrite (below)** |
| 22319 | issue | Model request: DeepSeek V4 Series | closed 2026-06-29 | passing mention |

**Finding (direct answer): there is NO mainline issue or PR about mul_mat_vec_q_moe spilling to local memory, register pressure, or rows-per-block tuning on pre-Volta (Maxwell/Pascal) GPUs.** The closest three:

1. **#23539** (https://github.com/ggml-org/llama.cpp/issues/23539, closed not_planned, stale-bot): "In `mul_mat_vec_q` (and `mul_mat_vec_q_moe`), the grid is launched with ceiling division … the accumulation loop can be out-of-bounds when the kernel try to read some rows at the end." Suggested fix: bounds-check the row. Author's own assessment: "For all of the models I know, numbers of rows is always even and `rbp` is 1 or 2 ==> out-of-bounds never happens." i.e. rows-per-block in the MoE kernel is 1–2 in practice; correctness-only, not performance, and it was never fixed upstream.
2. **#30524** (https://github.com/ggml-org/llama.cpp/issues/30524, closed 2026-11-05) — the host/device dispatch-mismatch bug, the pre-Volta-adjacent one:
   - Setup: 11x GTX 1660 Super (TU116, cc 7.5), CUDA 12.9, build 11616 (e7268bf6a), `-DGGML_CUDA_FORCE_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES="61-virtual;80-virtual"`, Qwen3.8-Flash-Next IQ3_XXS GGUF.
   - Symptom: `CUDA error: invalid argument … ggml_cuda_mul_mat_vec_q(...)` for MoE models only.
   - Cause: "the host-side `get_mmvq_mmid_max_batch(type, cc)` uses the device's real compute capability (7.5 → turing_plus, IQ3_XXS = 7), while the device-side uses `__CUDA_ARCH__`. With PTX for 6.1 JIT-compiled on Turing, that's 610, which falls through to pascal_older (IQ3_XXS = 4). For MoE batches of 5–7 tokens, the host launches a configuration the compiled kernel doesn't support."
   - Fix (one line, quoted): `cc = ggml_cuda_highest_compiled_arch(cc);` — "no speed loss". Also: "Other host-side decisions keyed on cc, where the device code uses `__CUDA_ARCH__`, may have the same mismatch."
   - M10 relevance: for a pure `-DCMAKE_CUDA_ARCHITECTURES=50` build, host (cc 50) and device (__CUDA_ARCH__ 500) both land on the pascal_older tables, so this specific mismatch does not apply to a M10-only build — but it is the documented failure mode for any mixed/virtual-arch build, and it proves the dispatch tables (`get_mmvq_mmid_max_batch_*`) are the knob that decides rows/batch on pre-Volta.
3. **#25620** (https://github.com/ggml-org/llama.cpp/issues/25620): same host/device mismatch class on SYCL (NaN on RDNA3.5) — confirms the pattern is cross-backend, not CUDA-specific.

sm_50 mentions in ggml-org/llama.cpp (query: https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+%22sm_50%22&per_page=50, total_count 6): #31756, #30524, #28389, #12325, #12104, #3670. Of these, the two that matter:
- **#31756** (https://github.com/ggml-org/llama.cpp/issues/31756, closed "completed" 2026-11-06): "Compile bug: CUDA11 sm_50 fails in lightning-indexer.cu" — clean CUDA 11.7 build targeting sm_50; `get_dequantize_V()` triggers 5 static assertions in fattn-common.cuh ("bad type"/"unsupported type"); first bad commit 15e50990 (PR #29050, MFMA lightning indexer for CDNA2). CMake: `cmake -B build -DGGML_CUDA=ON -DGGML_CCACHE=OFF -DCMAKE_CUDA_ARCHITECTURES=50`.
- **#30524** (above).
The other four (#28389 CUB argsort, #12325, #12104, #3670) mention sm_50 only incidentally (old/unrelated).

### 2.2 PR #20905 — "Optimize MOE GEMV kernel for BS > 1" (the current mainline MoE MMVQ)

https://github.com/ggml-org/llama.cpp/pull/20905
- Author: gaugarg-nv (NVIDIA contributor). Created 2026-03-23, **merged 2026-03-29** (merged_by JohannesGaessler, merge commit ec16a072f06c9c44d33513405a83068b15ae1b2c). +358/−59, 3 files, 4 commits, 14 comments. Follow-up to PR #20635 (the original MoE kernel).
- What it changed (verbatim):
  - "The previous MOE kernel for BS > 1 had too many thread blocks (nrows_x, nchannels_dst, ncols_dst), with very little work per threadblock. threadblock of (32, 4) was doing inner dot product for a single row."
  - "New mul_mat_vec_q_moe kernel is dedicated for MoE multi-token kernel with **grid (ceil(nrows_x/rpb), nchannels_dst), block (warp_size, ncols_dst). Each warp handles two rows independently with warp-level reduction only (no shared memory sync)**."
  - "This change doesn't increase any compilation time as a single template instance is needed per type. This also simplifies the original GEMV kernel and gets rid of `is_multi_token_id` specialization."
  - AI disclosure (verbatim): "AI used to write comments, add tests in test-backend-ops for BS=2,3 … benchmark, and **auto-tuning to finalize `rows_per_block` value**."
- Numbers (n_prompt 512, pp = prompt t/s; no Maxwell/Pascal/P40/P100 rows — test GPUs are RTX 5090, RTX PRO 6000 Blackwell, RTX 3090):
  - RTX 5090, qwen3moe 30B.A3B Q4_0: n_ubatch 2: 450.37→492.92 (1.09x); 3: 521.35→604.39 (1.16x); 4: 598.15→726.81 (1.22x). Q4_K_M: 383.56→414.14 (1.08x), 496.53→575.06 (1.16x), 550.21→665.28 (1.21x). Q8_0: 333.99→391.59 (1.17x), 430.30→476.49 (1.11x), 494.93→573.60 (1.16x).
  - RTX 5090, gpt-oss 20B MXFP4: 539.64→630.79 (1.17x), 694.81→773.40 (1.11x), 789.34→908.48 (1.15x). gpt-oss 20B Q4_K_M: 564.72→600.57 (1.06x), 721.70→804.74 (1.12x), 802.84→922.95 (1.15x).
  - RTX PRO 6000 Blackwell, qwen3next 80B.A3B Q4_K_M (ub2): 258.84→316.96 (1.22x); gpt-oss 120B MXFP4 (ub2): 347.87→397.08 (1.14x); full ranges 1.07–1.22x.
  - RTX 3090, qwen3moe 30B.A3B Q4_0: 310.20→394.19 (1.27x) (ub3), 337.44→450.56 (1.34x) (ub4); Q4_K_M up to 1.27x; gpt-oss 20B Q4_K_M 328.90→363.22 (1.10x) … 389.53→472.39 (1.21x).
- M10 relevance: the kernel's structure (rows-per-block rpb, 2 rows per warp, warp-only reduction, no shared-memory sync) is exactly the design space where sm_50's register file and lack of dp4a bite. **No Maxwell numbers exist upstream for this kernel — every reported speedup is Blackwell/Ampere.** The pre-Volta rows-per-block/dispatch tables (`get_mmvq_mmid_max_batch_pascal_older` etc.) remain the upstream tuning point (see #30524), and the ik-fork equivalent work (mb8565 #2701/#2721/#2740) is dense-MMVQ, not the MoE variant — so an sm_50-tuned MoE MMVQ would not duplicate existing upstream work.

---

## 3. Task 3 — four llama-server instances (one per M10 die) behind a router

### 3.1 Which tools actually load-balance across identical backends

- **llama-server itself: no router mode.** There is no llama.cpp built-in load balancer across separate server processes. What exists is in-process concurrency on one device: `--parallel` (slots) + `--cont-batching` — documented in ggml-org/llama.cpp discussion https://github.com/ggml-org/llama.cpp/discussions/4130 ("Could you have an explanation of how the --parallel and --cont-batching options function?", referencing PR #3589 "server : parallel decoding and multimodal" and #3677). That is one process / one model / one (or more, via -dev) device — not four independent instances. (Note: examples/server/README.md returned 404 on raw.githubusercontent this round — the file moved in a repo restructure; the flag is corroborated by the discussion above and by current server builds.)
- **llama-swap: NOT a load balancer.** Repo (the real one is mostlygeek, not justinmetzger): https://github.com/mostlygeek/llama-swap (6,207 stars, Go, MIT, active — pushed 2026-11-08; homepage llama-swap.ai). README, verbatim: "Run multiple generative AI models on your machine and **hot-swap between them on demand**." and "In the most basic configuration llama-swap handles one model at a time. For more advanced use cases, using a `matrix` allows multiple models to be loaded at the same time." It routes by the `model` field in the request and swaps the upstream server to match — the opposite of LB across identical backends (four entries with the same model name is not a documented LB mode). Useful adjacent facts from the README: it proxies OpenAI/Anthropic/llama-server endpoints, and its Docker unified images include `unified-cuda` = "NVIDIA Pascal through Ada: P40, P100, GTX 10xx, RTX 20xx/30xx/40xx. Built with CUDA 12, for cards CUDA 13 dropped" (and the unified image bundles ik-llama-server). It is the right tool if you want *different* models served from the same port; it does not spread one model's traffic across four GPUs.
- **LiteLLM: YES — this is the purpose-built option.** Docs: https://docs.litellm.ai/docs/proxy/load_balancing — "Load balance multiple instances of the same model. The proxy will handle routing requests (using LiteLLM's Router). **Set `rpm` in the config if you want maximize throughput.**" Routing strategy table (verbatim names): `simple-shuffle` (recommended/default — "Randomly distributes requests"), `least-busy` ("Routes to deployment with fewest active requests — High concurrency scenarios"), `usage-based-routing` ("bad for perf"), `latency-based-routing` ("Routes to fastest responding deployment"), `cost-based-routing". Mechanism: list several `model_list` entries with the **same `model_name`** and different `api_base`/`api_key`; the proxy serves an OpenAI-compatible API on :4000 and checks `model_id` in response headers to confirm distribution. Reliability layer: retries, cooldowns, fallbacks, `num_retries`, `timeout`; Redis for shared state across proxy instances; Kubernetes pod discovery (Cache-Aware Routing) — "Requests without an explicit session ID use round robin." llama-server is an OpenAI-compatible backend, so each die's instance goes in as an `openai/`-style deployment with its api_base.
- **nginx: YES — the simple option.** Official docs: https://nginx.org/en/docs/http/load_balancing.html (upstream block; round-robin default, `least_conn`, `ip_hash`, weights, `backup`). The how-to with llama.cpp-specific config and measured proxy overhead: https://markaicode.com/integrate/nginx-with-llamacpp/ ("nginx + llama.cpp: Production Reverse Proxy Setup Guide", checked against primary sources Aug 2026). Its load-balancing step (Section 5), verbatim:
  ```
  upstream llama_backend {
      least_conn;
      server 127.0.0.1:8080;
      server 192.0.2.2:8080;
      server 192.0.2.3:8080 backup;
  }
  ```
  plus: "Each backend needs its own model pinned and its own `--api-key` if you're authenticating; version-drift between backends (one node on an older build than another) is a real failure mode."
  Streaming requirements (verbatim): "`proxy_http_version 1.1` and `proxy_buffering off` are required for streaming; the default HTTP/1.0 proxying buffers the whole response before forwarding, which breaks token-by-token output. `proxy_read_timeout 300s` prevents nginx from cutting off long generations (the default is 60s)."
  Measured proxy overhead (their mock-backend test, 300 requests): mean 0.499 ms direct → 0.681 ms via nginx (+0.182 ms); p95 +0.117 ms. "the proxy hop itself adds well under 1ms of pure networking overhead."
  Its honest limitation (verbatim): "If you need real load-balancing intelligence. nginx's `least_conn` is coarse. If you need queue-depth-aware routing across GPU nodes, a purpose-built inference router will outperform it."
  Other how-tos found: https://deepwiki.com/alexziskind1/llama-throughput-lab/9-round-robin-load-balancing ("multiple llama-server instances, an nginx reverse proxy that distributes requests, and client code … that sends concurrent requests to the proxy endpoint") and https://markaicode.com/integrate/nginx-with-llamacpp/ FAQ ("Define an upstream block with least_conn; and list each backend's address; point proxy_pass at the upstream name instead of a single host.").

### 3.2 Recommended wiring for the M10 (4 dies = 4 CUDA devices)

- Start four llama-server processes, each pinned to one die (e.g. `CUDA_VISIBLE_DEVICES=0..3` / `--cuda-devices` per instance) on ports 8080–8083, all bound to 127.0.0.1, same model + same quant + same flags (version-drift is the #1 failure mode per both guides above).
- Front it with **nginx** (`least_conn` upstream over 127.0.0.1:8080–8083, `proxy_http_version 1.1`, `proxy_buffering off`, `proxy_read_timeout 300s`+), or **LiteLLM proxy** if you want retry/cooldown/least-busy semantics and one OpenAI-compatible API with per-deployment rpm. llama-swap is the wrong tool here (model swapping, not LB).
- Caveats specific to identical-4-backend LB: round-robin/least_conn do not know a slot's in-flight KV length — for long-context workloads `least_conn` (nginx) or `least-busy`/`latency-based` (LiteLLM) beats blind round-robin; per-instance `-np/--parallel` still applies inside each process if a die should serve >1 concurrent slot.

---

## 4. Task 4 — Qwen3.5-0.8B vs 2B vs 4B: tokenizer/vocab identity

Raw config.json (HuggingFace, branch main, fetched in full this round):
- https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json

Exact values (text_config, quoted):

| field | 0.8B | 2B | 4B |
|---|---|---|---|
| **vocab_size** | **248320** | **248320** | **248320** |
| tie_word_embeddings | true | true | true |
| hidden_size | 1024 | 2048 | 2560 |
| intermediate_size | 3584 | 6144 | 9216 |
| num_hidden_layers | 24 | 24 | 32 |
| full attention layers (full_attention_interval 4, layer_types) | 6 of 24 | 6 of 24 | 8 of 32 |
| num_attention_heads / num_key_value_heads | 8 / 2 | 8 / 2 | 16 / 4 |
| head_dim | 256 | 256 | 256 |
| linear_num_key_heads / linear_num_value_heads | 16 / 16 | 16 / 16 | 16 / 32 |
| max_position_embeddings | 262144 | 262144 | 262144 |
| eos_token_id | 248044 | 248044 | 248044 |
| image_token_id / video_token_id | 248056 / 248057 | 248056 / 248057 | 248056 / 248057 |
| vision_start_token_id / vision_end_token_id | 248053 / 248054 | 248053 / 248054 | 248053 / 248054 |
| rope (theta / partial_rotary_factor / mrope_section) | 1e7 / 0.25 / [11,11,10] | 1e7 / 0.25 / [11,11,10] | 1e7 / 0.25 / [11,11,10] |
| mtp_num_hidden_layers | 1 | 1 | 1 |
| architectures | Qwen3_5ForConditionalGeneration | Qwen3_5ForConditionalGeneration | Qwen3_5ForConditionalGeneration |
| model_type (text) / dtype / transformers_version | qwen3_5_text / bfloat16 / 4.57.0.dev0 | same | same |

**Conclusion: identical.** All three share `vocab_size = 248320`, `tie_word_embeddings = true`, and byte-identical special-token IDs (eos 248044; vision 248053/248054; image 248056; video 248057) and rope setup. That is the tokenizer-identity test llama.cpp's draft path cares about (draft and target must share the tokenizer; identical vocab size + special ids + tied embeddings is the strong form of that). So **Qwen3.5-0.8B is a valid llama.cpp draft model for the 2B and 4B targets** (e.g. `--spec-type draft-simple -md Qwen3.5-0.8B-….gguf --spec-draft-n-max N`), with `--spec-draft-device`/`--spec-draft-ngl` to pin it to a spare M10 die. Caveats: (a) these are VLM configs (Qwen3_5ForConditionalGeneration with a vision_config; the text model is `qwen3_5_text`) — convert/load the text side; (b) no public measured acceptance rate exists yet for 0.8B-draft → 2B/4B-target specifically (that gap is unchanged from round 5).
Closest public acceptance datapoint (fetched this round, ik #2726 below): **Qwen3.5-2B (Q6_K) as draft for a Qwen3.5 MoE target** — see 4.1.

### 4.1 Measured Qwen3.5 draft datapoint (ik_llama.cpp #2726, merged 2026-10-22)

https://github.com/ikawrakow/ik_llama.cpp/pull/2726 — "sampling : normalize probabilities in the DIST sampler"
- The bug: "Since #1261, `llama_sample_dist` calls `llama_sample_softmax_impl` with `normalize = false`. After the sampler chain the top candidate always has `p = 1`. The draft-model check `cur_p->data[0].p < p_min` never fires, so every `-md` draft runs to `n_max`." Fix: normalize, "as mainline's dist sampler does."
- Measured effect (setup: Llama-3.1-8B-Instruct Q4_K_M target, Llama-3.2-1B-Instruct Q4_K_M draft, llama-server, `n_max=16`, `p_min=0.75`, 5 prompts × 256 tokens, seed 42): "Acceptance is 23 to 31% on main and **76 to 84% with the PR**." t/s (temp 0/0.8): P100 17.1/15.2 → 41.7/41.1 (no-spec 46.2/46.4); Titan V 77.8/65.3 → 129.5/125.9; RTX 4000 65.5/56.3 → 75.4/69.2; CPU 11.0/11.4 → 26.1/25.9.
- **Qwen3.5 row (verbatim)**: "`-md` hybrid: Qwen3.5-122B-A10B IQ3_XXS, experts on CPU, rest on one P100, **Qwen3.5-2B Q6_K draft** | 2.55 / 2.43 | 15.23 / 15.09 | 15.37 / 15.41" (main / PR / no-spec, temp 0 / temp 0.8). "In the hybrid run, main drafts 16 tokens every time and accepts 3.0 to 3.5% of them." — i.e. on main, a Qwen3.5-2B draft for a Qwen3.5 target gained nothing (2.55 t/s); after the sampler fix it matches no-spec (15.23 vs 15.37). This is the same-tokenizer family working as intended, and it is the strongest public evidence that Qwen3.5-small drafts for Qwen3.5 targets only pay off once the DIST-sampler normalization is present (ik has it since 2026-10-22; mainline equivalent status not established this round — check before relying on it).
- Other spec rows: MTP (Ornith-1.5-9B Q8_0, Titan V): 89.31→89.18 (identical output); DFlash2 (Qwen3.8-27B, P100): 12.81/14.72 → 12.81/11.92 (slower at temp 0.8 because the PR's standard accept test `min(1, p/q)` is stricter than main's `min(1, p/(p_top*q))`; "At temp 0 main and the PR give identical output"); distribution check (1500 seeds, chi-square): main p = 7.7e-72 / TV 0.28 vs PR p = 0.64 / TV 0.05 — main's sampling was measurably broken, the PR's is not. `llama-speculative` (Llama 8B/1B, temp 0.8): acceptance 10.2% → 17.8%.

---

## 5. Corrections to earlier rounds

1. **Round 4 §1 (PR 2580) M10 absolute values were mis-transcribed.** R4 wrote "f16 KV: pp2048 main ~3245 -> PR ~3932 (+21.2 %); pp4096 ~3334 -> ~4502 (+35.0 %); tg 225.9 -> 235.2 (+4.1 %)". The full PR body (re-fetched this round) shows the M10 table is: f16 pp2048 **39.3 → 47.3 t/s (+20 %)**, f16 pp4096 **32.6 → 44.1 (+35 %)**, q8_0 pp4096 **13.7 → 44.0 (3.2x)**; there is **no M10 tg table** in the PR ("tg64 moves by 0.1 % or less" is stated for the P100 runs only). Percentages in R4 were right; absolutes were not. Use this round's numbers.
2. Round 4 §0's core-count correction (5 SMs × 32 cores) was itself wrong per your round-5 correction — already superseded in the round-5 report §0 (128 cores/SM; 640/die; 2560/M10; 1.672 TF per die). No change this round.

---

## 6. Source index (fetched in full this round)

- api.github.com searches: ik_llama.cpp author:mb8565 created:2026 (59, full bodies); ggml-org author:mb8565 created:2026 (0); ggml-org author:mb8565 all-time (0); is:discussion both repos (0+0); repo:ggml-org "mul_mat_vec_q_moe" (13, full bodies for the 13); repo:ggml-org "sm_50" (6); api.github.com/users/mb8565/repos (2)
- PR/issue bodies: ik #2860, #2740, #2726 (incl. tail), #2721, #2701 (incl. M10 table + S_TG/S_PP sweep), #2580 (full body incl. M10 table, 2x-GPU tables, PPL/KLD), #2668, #2670; ggml-org #20905 (full), #23539, #30524 (full)
- huggingface.co raw config.json: Qwen/Qwen3.5-0.8B, -2B, -4B
- https://docs.litellm.ai/docs/routing and https://docs.litellm.ai/docs/proxy/load_balancing
- https://markaicode.com/integrate/nginx-with-llamacpp/ (incl. Section 5 upstream config, measured overhead, failure-mode table)
- https://nginx.org/en/docs/http/load_balancing.html (search-result level + official docs link)
- https://github.com/mostlygeek/llama-swap README (raw main, full)
- web_search corroboration: llama-server --parallel/--cont-batching (ggml-org discussion #4130, PRs #3589/#3677)

[SNIPPET]-only items this round: the deepwiki "round-robin load balancing" llama-throughput-lab page (search snippet); markaicode FAQ one-liner on least_conn (confirmed in the full page, cited there). No Zhihu/CSDN re-attempts (still blocked, unchanged).
Known gaps: no sm_50 numbers for the mainline MoE MMVQ kernel (#20905) exist anywhere; no 0.8B→2B/4B measured acceptance rate; mainline status of the DIST-sampler normalization (ik #2726) not established.
