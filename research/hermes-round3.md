# M10 / Maxwell LLM inference research — Round 3

Delivered for Claude (the coding agent). Date: 2026-11-08. Tools: firecrawl (web_extract) + web_search + GitHub REST search API (api.github.com, fetched as JSON). All numbers quoted with source. Items marked [SNIPPET] come from search-result text, not fetched full content — treat as unverified.

Already known (not repeated here): mmvq compute-bound on sm_50 (no dp4a); 1 warp with 4 rows per block = +50–65% generation.

============================================================
1. llama.cpp / ik_llama.cpp GitHub — Maxwell, sm_50/sm_52, mmvq, dp4a
============================================================

1.1 ik_llama.cpp PR #2580 — THE HEADLINE FIND
URL: https://github.com/ikawrakow/ik_llama.cpp/pull/2580 (merged 2026-10-02, author mb8565)
Title: "CUDA: route head size 256 FA prefill to tile_f32 on NVIDIA GPUs before Volta"

This PR was explicitly tested on a Tesla M10 (sm_50, one GM107 die) — the only Maxwell-specific FA number found anywhere in this round.

Problem: at head size D=256 (Falcon3-10B, Gemma-4-12B sliding-window layers), FA prefill used the vec kernel, which re-reads the entire K/V range for every query column. For KV types other than F16 and Q8_0, no D=256 vec instance exists, so on Maxwell/Pascal the whole FA op silently fell back to CPU.

Fix: D=256 prefill (batch > 8) routed to flash_attn_tile_ext_f32 — each K/V tile reused across 16 query columns, fp32 accumulate.

Measured on M10 (single GM107), Falcon3-10B Q4_K_M, `llama-bench -ub 512 -ngl 99 -fa 1 -r 3`, main 32cddbfc vs PR:
- f16 KV:  pp2048 39.3 -> 47.3 t/s (+20%);  pp4096 32.6 -> 44.1 t/s (+35%)
- q8_0 KV: pp4096 13.7 -> 44.0 t/s (3.2x)
- q4_0 KV: main runs FA on host CPU (40 threads, AVX-512); there the PR is 1.6% (pp2048) and 7.0% (pp4096) SLOWER — a beefy CPU with AVX-512 beats M10 for q4_0 KV FA prefill
- Quality: KLD vs CPU 0.000679 (main) vs 0.000680 (PR) — unchanged

For context (same PR): P100 pp16384 116.5 -> 226.7 t/s f16 (+94.5%), q8_0 +147.8%; P40 pp8192 246.0 -> 448.7 (f16), 32.7 -> 445.8 (q8_0); 2x P40 pp8192 426.9 -> 778.8 (layer split). tg64 (decode) moves <= 0.1%.

Related PR in the same series: #2510 ("I kept head sizes 64 and 128 on tile_f16, where tile_f32 was slower") — on P100, D=64/128 prefill does NOT benefit from tile_f32. M10 L2 differs from P100, so whether D=128 FA prefill on M10 also improves with tiling is an open experiment worth running.

Practical relevance: any model with head dim 256 on your M10s (Falcon3, Gemma sliding-window layers) is leaving 20–35% prefill on the table on current main; and if you're doing long-context prefill, the vec-vs-tile FA choice is a bigger lever than most mmvq tweaks.

1.2 shinbunbun/llama-cpp-p100-patches — a measured patch catalog for pre-dp4a NVIDIA
URL: https://github.com/shinbunbun/llama-cpp-p100-patches
     https://github.com/shinbunbun/llama-cpp-p100-patches/blob/main/docs/patches.md
     https://deepwiki.com/shinbunbun/llama-cpp-p100-patches/2.2-mmvq-tuning-for-sm_60

43 individually-measured patches against llama.cpp v0.6.0, authored for P100 (sm_60, also no dp4a). The "no DP4A" scope in the repo is explicitly "GP100 and Maxwell/Kepler — takes its own MMQ tile table", so this is the closest published body of work to your exact situation. ALL numbers below are P100-sm_60; each needs re-measurement on M10, but the failure modes are architecture-class (pre-Turing, no dp4a, 4-warp blocks, L2-bound matvec), not chip-specific.

Most relevant to your mmvq compute-bound finding:

- Patch 02 "mmvq-rows-per-block-sm60" (pre-Turing): stock MMVQ processes 1 output row per block; each block re-reads the q8_1 activation row from L2 once per row. The load is the bottleneck (DRAM only 32–34% of peak; instructions not the limit). Sweep at ncols_dst=1 (9B Q4_0, llama-bench tg32): rows=1 -> 46.9, rows=2 -> 54.1, rows=4 -> 57.7 (+23.0%), rows=8 -> 57.2 t/s. This independently reproduces your 4-rows-per-block result (they measure +23% on a 9B model; your +50–65% is presumably on a smaller model / different batch, where the effect is larger).
- Patch 01 "vmad-dp4a-sm60": replaces the 4x-int8 dot-product emulation with a chain of 4 PTX `vmad` ops (integer multiply-add with byte selectors). ~15 SASS instructions per 4-dot -> ~6.75. +6.5–6.8% decode. sm_60-specific in the patch, but PTX vmad with byte selectors is old PTX — a 1-hour compile test on sm_50 would establish whether it assembles; if so this is a ~45% instruction-count cut in exactly the kernel you found compute-bound. (Flag: I have not verified byte-select vmad in the sm_50 PTX table — verify before assuming.)
- Patch 07 "mmvq-moe-rows-sm60": the fused MoE matvec launcher hardcodes rows_per_block=2; raising it gives +1.9% decode. Not architecture-gated — applies to Maxwell. (Matters only if you run MoE.)
- Patch 09 "mmvq-nwarps-small-k-sm60" (pre-Turing): the MMVQ block is 4 warps wide regardless of K; for small-K types the tail warps sit idle (IQ3_XXS K=512: only 16 of 128 threads do work). "Just enough warps to cover K": IQ3_XXS 64.03 -> 36.64 us; Q3_K 106.59 -> 48.27 us; IQ2_XS 24.00 -> 20.76 us; +1.29% MoE decode. On M10 (16 SMs) occupancy math is tighter, so this could be bigger — worth a direct test.
- Patch 08 "mmvq-mmid-batch-sm60" (pre-Volta): MUL_MAT_ID batch-size cap table is keyed on cc < VOLTA; above the cap it falls back to a sorted-gather path. Raising the cap: +2.2%. The gate (pre-Volta) includes sm_50, so the same fallback likely fires on M10. Runtime knob added in patch 35: env LLAMA_MMVQ_MMID_MAX.
- Patch 05 "mmvf-f32-pascal" (pre-Turing): the F32 matvec path caps ne11 at 3; speculative decoding sets ne11 = n_draft+1 = 4, which silently pushes ALL F32 mul_mat to cuBLAS SGEMM (measured 5.9% of decode GPU time). Fix: +3.7–4.3%. Directly relevant if you want llama.cpp-native speculative decoding on M10 — there is a known cap in exactly that path.
- Patch 12 "mmvq-f16-sm60": Q4_1 GEMV on HFMA2 — nibbles expanded to half2 via LOP3 magic constants (0x6400/0x5400), fp16 accumulate, no int->float conversion; +9.4–9.5%. GP100-only mechanism (full-rate HFMA2 + LOP3, which is Pascal+; no fp16 path exists on sm_50). The Maxwell analogue is nibble expansion via PRMT/LUT (PRMT exists on Maxwell) accumulating in fp32 — an original experiment, but the pattern (avoid the I2F chain, expand weights to float in registers) is what to copy.
- Patch 10 "mmvq-q8-1-activation-cache": cache the q8_1 activations in device memory; gate/up projections re-quantize the same activation every time. Cheap, arch-independent.
- Patches 32–47: MoE expert caching in host memory (the strata-style pattern). Highlights: 60 expert cache slots -> 51.0/52.4 ms/token at 40k/211k context; expert replan period 128 vs 32: 47.45 vs 48.65 ms/token (−2.5%); shortening replan for the first 32 tokens raises expert-cache hit rate 7.2% -> 46.6%, 32-token generation 94.2 -> 83.7 ms/token (−11%).

The maintainer's README says: "the most useful contribution is measurements on hardware I don't have — P40, GTX 10xx, Maxwell, or anything Turing onward." Your M10 benchmark numbers, mapped onto this patch list, would be a directly publishable contribution.

1.3 llama.cpp PR #31787 "CUDA: fix Maxwell build" (JohannesGaessler, MERGED 2026-11-06 — two days before this report)
URL: https://github.com/ggml-org/llama.cpp/pull/31787 (fixes #31756)
Verbatim from the PR body: "I think the correct way to handle this is to not immediately fail the build if FP16 isn't available (which only happens on Maxwell). This means that issues may only be detected at runtime but given that so few people use Maxwell I think that is preferable over failed builds."
Implication: current llama.cpp main broke the Maxwell build and the fix landed 2026-11-06. If your fork base predates that commit, rebuild from >= it.

1.4 Other llama.cpp Maxwell-relevant issues found via GitHub API search (repo:ggml-org/llama.cpp "maxwell" = 37 total hits, mostly old build issues):
- #25593 "Eval bug: SM_60 Quality Loss, FP32 math silently done in FP16, fix merged in two forks" (https://github.com/ggml-org/llama.cpp/issues/25593) — a silent-precision-downgrade regression on a pre-Pascal-arch GPU; the pattern to suspect if you ever see quality loss that "shouldn't" be possible. (Title/URL verified; did not fetch body.)
- #25680 "Enhance ggml_cuda_mul_mat_cublas with compute type fallback" (https://github.com/ggml-org/llama.cpp/pull/25680) — cuBLAS compute-type fallback for GPUs that lack the requested compute mode.
- #4395 (https://github.com/ggml-org/llama.cpp/issues/4395) — crash log in the issue: `cuBLAS error 15 at ggml-cuda.cu:7956: the requested functionality is not supported` in ggml_cuda_mul_mat_mat_batched_cublas. The classic sm_50 trap: cuBLAS mat-mat paths that simply do not exist on Maxwell. General rule: on M10, prefer native GGML kernels and keep cuBLAS off the decode hot path.
- #10849 "Compile bug: Compiling on Maxwell architecture 52 cuda12.7" (https://github.com/ggml-org/llama.cpp/issues/10849, Dec 2024, closed stale) — sm_52 build fragility history.
- #21438 (https://github.com/ggml-org/llama.cpp/pull/21438) — "docker: lower cuda12 floor to 12.8.1" — context: modern llama.cpp builds assume CUDA 12.8+ drivers; M10 tops out at driver 550.54.10 (CUDA 11.x runtime class per your earlier finding; CUDA 12.x compatibility through 12.x minor).

1.5 llama.cpp discussion #13103 (https://github.com/ggml-org/llama.cpp/discussions/13103)
Walkthrough of the MMVQ kernel internals by JohannesGaessler (qk/qi/vdr register layout, why the kernel is shaped as it is). The best public reference for surgically editing the kernel you're already tuning.

1.6 Build/runtime environment facts for sm_50
- GitHub gist "llama.cpp + CUDA 13.2, GTX 1650 Ti" (https://gist.github.com/1e92b52551f2f1b1f26f0a1e23d3f0f0 — [SNIPPET, URL from search results, do not cite without re-fetch]): documents arch-selection build failure and the workaround pattern of pinning -DGGML_CUDA_ARCH to the card's arch (for you: 50).
- runaihome.com blog "llama.cpp + CUDA 13.2 GTX 1650 Ti" (https://runaihome.com/llama-cpp-cuda-13-2-gtx-1650-ti/): references upstream issue #13271 for the sm_86 build failure; same arch-list modernization that drops sm_50.
- timoruohomaki/theworkstation repo (https://github.com/timoruohomaki/theworkstation, fetched in full): HP Z240 Tower + Xeon E3-1230 v5 + 32GB DDR4, K2200 for display, Tesla M10 for compute. Verbatim: "NVIDIA driver 535.x — supports sm_50/sm_52 ... CUDA Toolkit 12.6 (pinned — Maxwell unsupported in CUDA 13+) ... Ollama serving, M10 appears as 4 CUDA devices (640 cores, 8GB each = 2560 total cores)"; "Stock 400W PSU is insufficient for the M10 — using 1000W." Includes device_query / cuda_bench (GB/s + GEMM GFLOPS) / torch_bench scripts you can borrow as a measurement harness. [Repo claim on the CUDA-13-drop: consistent with NVIDIA's announced deprecation, but this is the source I can cite.]

============================================================
2. NVIDIA forums / papers — fast int8 & 4-bit GEMV on Maxwell
============================================================

2.1 NVIDIA developer forum "GEMV Library for NVIDIA GPU"
URL: https://forums.developer.nvidia.com/t/gemv-library-for-nvidia-gpu/335717 (Dec 2024, fetched)
Threads a library of fast GEMV routines; links: "How to speed up matrix-vector product" (CUDA blog, cuda_hgemv) and the FastGEMV project. All FP16/A100-era — no Maxwell-specific content. No Maxwell int8 GEMV library exists on the NVIDIA forums.

2.2 Instruction-level notes (for your kernel design)
- LOP3: Pascal (sm_60) and later ONLY. The P100 patch-12 nibble trick (LOP3 with 0x6400/0x5400) cannot transfer to sm_50. Maxwell's byte-permute primitive is PRMT — that is your tool for nibble expansion.
- __vabsdiff4 (int4 absolute-difference, SAD-style): available on Maxwell; the standard VADD building block. The P100 patch set does NOT use it for dot products (uses vmad chains) — dot product on Maxwell is not expressible as SAD, so vabsdiff4 helps distance work, not GEMV.
- PTX vmad with byte selectors (vmad.s32.s8/s.u8...): the mechanism behind P100 patch 01 (+6.5–6.8% decode, ~15 -> ~6.75 SASS per 4-dot). Old PTX, but I have not verified the sm_50-specific form in the PTX ISA matrix — [VERIFY before coding] — a single nvcc smoke test settles it.
- No academic paper found that targets int8/4-bit GEMV specifically on Maxwell. Closest general material is FastGEMV (FP16, A100). Gap, not absence of evidence — this space is simply not written about publicly; the shinbunbun patch catalog is the de-facto reference.

============================================================
3. Chinese sources (zhihu / bilibili / CSDN / v2ex / Tieba)
============================================================

3.1 Zhihu p/702874971 — BLOCKED (firecrawl antibot, two attempts, different URL forms).
[SNIPPET ONLY] Search-result description: an M10 (CC 5.0, "相当于 GTX750 级别") with 32GB (4x8GB) running llama3-70B under Ubuntu 20.04, author spent a day ("花了一天") getting it working. No numbers visible in any snippet. If the OP matters to you, it needs a manual browser read.

3.2 CSDN blog.csdn.net/savage2k/article/details/147398819 — BLOCKED (HTTP 521, two attempts). [SNIPPET ONLY] Described as a cheap-GPU llama.cpp article; no numbers available.

3.3 bilibili: search for "Tesla M10 推理 显卡" returns M40 GPU review videos (e.g. https://www.bilibili.com/video/BV1yP4y1u7fC — "M40显卡评测" [SNIPPET]); no M10+LLM video found.

3.4 v2ex: "M10" hits are a smart-door-lock (京造 M10) and a tablet — zero GPU/LLM content. Tieba: nothing found.

Conclusion: no usable first-hand Chinese M10 LLM numbers obtained this round. The English m10dis thread (r/LocalLLaMA 1xnl82r, from round 2) remains the highest-value real-world M10 LLM artifact found anywhere.

============================================================
4. ServeTheHome / Level1Techs — M10 multi-GPU, P2P, power, cooling
============================================================

4.1 STH "Asus x-399-a and Tesla M10 can't finish POST" (fetched in full)
URL: https://forums.servethehome.com/index.php?threads/asus-x-399-a-and-tesla-m10-cant-finish-post.48335/ (Jul 2025, 3 replies)
M10 stuck POST on an Asus x399-a (TR4) board, RGB stuck on "firmware update". Working fix from the poster: "tape over the SMBUS pins on the M10's PCIe connector" — the M10's LED-firmware SMBUS traffic conflicts with the board; "I could not believe in that, but it actually works!" Confirmed working on Xeon x5680 boards, i9-9900KS, Ryzen 7 7800X3D. Poster's stated purpose: "for ollama/llm inference".
This is the second distinct M10-on-modern-PCIe quirk (the first being the 64 KiB legacy-I/O exhaustion boot hack in the 1xnl82r m10dis thread). Both are board/BIOS-level, not driver-level — expect them when moving M10s between machines.

4.2 STH M10 tag: exactly ONE thread (the one above). No M10 multi-GPU P2P / PEX8747 threads exist on STH.

4.3 L1T "Gaming on my Tesla" thread, page 17 (fetched)
URL: https://forum.level1techs.com/t/gaming-on-my-tesla-more-likely-than-you-think/171185?page=17 (Jun 2021)
- M10 discussed as a 4x coprocessor rig paired with a 3090; no working solution found in-thread.
- K40/K80 note: "Above 4G Decoding" must be enabled in BIOS on consumer boards for these cards — the classic large-BAR pitfall; relevant to M10-class cards on some boards.
- M40 price reference at the time: $220.
No LLM content on the page; L1T value for this task is historical. No L1T thread on M10 LLM inference exists.

4.4 M10 P2P through the on-card PEX8747: no direct forum reports on STH/L1T. The only working evidence in the public record is the 1xnl82r m10dis build (Tinygrad GPU-P2P over the PLX switch, PLX PCIe 2.1 multicast, 29 us multicast publish + 132 us M10 round-trip per MoE layer, 52.5 t/s decode / ~277 t/s prefill) plus the tinygrad open-gpu-kernel-modules fork referenced there.

============================================================
5. Follow-up on r/LocalLLM 1wuinuf (the M10-as-subagent-card thread)
============================================================

5.1 Re-fetched 2026-11-08: URL https://www.reddit.com/r/LocalLLM/comments/1wuinuf/tesla_m10_a_dubious_card_with_a_weird_plan/
NO new first-hand results from the OP. The OP is still in planning; latest OP comment: "I'm currently running 3, soon to be 4 p100's. I'm ready for the adventure!" Thread remains discussion-stage (~1 month old at time of fetch).
Useful comments captured:
- jjusko20: "locked to cuda 11... CRAWLING... my main model has to wait 6 hours... i would recommend a p40 — newer generation, can run cuda 12, 24GB, 3x+ the bandwidth"
- mtbMo: confirms the Quadro M4000 (4GB, also Maxwell) works for their use
- silent-curious-dev: warns about M10 passive-cooling in a hot chassis
Status to carry forward: this thread will not produce numbers without the OP buying a card.

============================================================
6. Speculative decoding / multi-GPU tricks on old Maxwell cards
============================================================

6.1 llama.cpp-native speculative decoding on Maxwell is gated by a known cap:
P100 patch 05 (see 1.2): pre-Turing F32 matvec caps ne11 at 3; SD with >= 3 drafts pushes the whole F32 matvec path to cuBLAS SGEMM, costing 3.7–4.3% decode. On M10 that cap is live — lifting it (one-line arch-conditional change) is a prerequisite for llama.cpp SD on your cards.

6.2 Speculative decoding PROVEN on M10s (outside llama.cpp): the Strata-maxwell bench (below) runs spec-6 with min-p 0.5 on Qwen3.8-Flash-Next over 4x M10 + RTX 3060 and gets 69.5–70.1 t/s decode with 234/258 and 202/285 drafts accepted (~1.5 accepted per step is not the win — the win is the expert math).

6.3 Multi-GPU expert offload on M10 — the best real-world data found this round:
URL: https://github.com/BitMakerMan/Strata-maxwell/blob/main/bench/results/2026-10-25-m10-vs-standard/README.md (fetched in full, 2026-10-25 record)
Host: RTX 3060 (12GB) + 4x Tesla M10, 4-core AVX2 CPU, 39GB RAM. Model: Qwen3.8-Flash-Next, full 256K context; KV cache fills most of the 3060's 12GB, ~2.5GiB for expert cache -> 92% of routed experts offloaded. 22,702 experts in M10 VRAM (5676+5676+5675+5675 per die; 7.61–7.63 GiB per die), 1874 expert-cache slots (2.53 GiB) on the 3060.
- Config A (experts in M10 VRAM, strata fork, spec 6 + min-p 0.5): prefill 2.7k = 148.5/144.5/154.0 t/s; prefill 10k = 154.3 t/s; decode (300 tokens) = 69.5 / 70.1 t/s (234/258 drafts accepted)
- Config B (experts in CPU RAM, stock strata v0.2.6, spec 4): prefill 2.7k = 99.9/430.3/434.9 t/s; prefill 10k = 275.3 t/s; decode = 35.9 / 52.5 t/s (202/300, 207/285 accepted)
- Verbatim conclusion: "expert compute for the verification window on the 4 M10 dies beats a 4-core AVX2 pool even with experts page-locked in RAM" (decode 1.3x when warm); prefill in CPU RAM is 1.8–2.8x FASTER (M10 tiers prefill in 64-token windows).
- Production env recipe from the record: MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 FASTCOMPLETE=1
- Quality gate green, deterministic.

Implication for your 4-die M10: as a whole-machine strategy, "M10s as a big expert pool under a small modern GPU" is now a demonstrated architecture (here 70 t/s decode on a 256K-context MoE on a 4-core CPU host). As a per-kernel strategy it confirms M10 VRAM-resident expert compute beats even a weak CPU's AVX2 for verification-window work.

============================================================
7. Ranked recommendations for your next M10 experiments
============================================================

1. Test PTX byte-select vmad on sm_50 (1-hour smoke test). If it assembles, port P100 patch 01: ~45% fewer instructions in your compute-bound kernel, +6.5–6.8% on P100 as the expected lower bound.
2. Re-measure P100 patches 07 (MoE rows), 09 (nwarps vs K — likely larger on M10's 16 SMs), 08 (mmid batch cap; pre-Volta gate includes sm_50; env knob LLAMA_MMVQ_MMID_MAX exists), 10 (q8_1 activation cache) on M10. 02 is already validated by you.
3. FA D=256 tile routing (ik PR #2580): +20–35% prefill measured on M10, free if your base is recent ik_llama.cpp; and benchmark D=128 FA vec-vs-tile on M10 (open question the PR explicitly leaves: "A D=256 prefill run, main vs PR, from someone with another Pascal or Maxwell card would help" — you ARE that someone, and D=128 is the bonus experiment).
4. If running MoE: q8_1 activation caching + the Strata/maxwell expert-pool pattern (M10 VRAM experts > 4-core AVX2 CPU for verification-window compute).
5. Build hygiene: base on llama.cpp >= the 2026-11-06 "CUDA: fix Maxwell build" merge (PR #31787); CUDA <= 12.x (Maxwell unsupported in CUDA 13 per the themeworkstation repo, consistent with NVIDIA's deprecation); keep cuBLAS off decode (issue #4395 crash pattern: cuBLAS error 15, functionality not supported).
6. If doing llama.cpp-native speculative decoding: lift the ne11<=3 F32 matvec cap first (patch 05 pattern) — otherwise 3.7–4.3% is left on the table in cuBLAS SGEMM.
7. Hardware: SMBUS POST fix (tape pins) for TR4/x399 boards; "Above 4G Decoding" BIOS option on older consumer boards; PSU >= ~500W-class headroom (stock 400W insufficient per themeworkstation); passive-cooling caution in hot chassis (silent-curious-dev, 1wuinuf).

============================================================
8. Gaps / not found
============================================================

- No NVIDIA forum thread or paper on int8/4-bit GEMV tuned specifically for Maxwell. The public record for "pre-dp4a GEMV optimization" is effectively: shinbunbun's patch catalog + the ik_llama.cpp mb8565 PR series (#2510, #2562, #2566, #2567, #2580).
- Zhihu/CSDN M10 LLM posts exist but are bot-walled (zhihu antibot, CSDN 521). Snippets only — see section 3.
- STH/L1T contain no M10 multi-GPU P2P or LLM-inference threads beyond the single POST/POST-fix thread (4.1).
- 1wuinuf OP has not posted results as of 2026-11-08.
- ik PR #2580's D=128 FA question on Maxwell is genuinely open (PR #2510 says tile_f32 was SLOWER than vec for D=64/128 on P100 — M10's smaller L2 may invert that).
