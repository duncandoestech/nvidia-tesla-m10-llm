# M10 research — Round 5

Tools: firecrawl `web_extract` (api.github.com, huggingface.co, reddit.com) + `web_search`.
Rules: research only; this one file; no other hosts. Numbers quoted exactly;
[SNIPPET] = seen in search snippet only, could not re-fetch.

## 0. Hardware model correction (authoritative, per your correction)

Accepted and now used as ground truth in this report:

- Maxwell SM (SMM) = 128 CUDA cores (4 SMSP x 32).
- GM107 die = 5 SMs x 128 = 640 cores PER GPU (per die).
- M10 card = 4 dies x 640 = 2560 cores total.
- 1.672 TFLOPS FP32 is PER DIE (640 x 2 x 1.306 GHz).
- theworkstation repo was right.

Flagging my own errors so they don't propagate:
- Round 3 report: "16 SMs" — wrong (5 SMs per die).
- Round 4 report section 0: I wrote "5 SMs x 32 cores = 160 per die, 640 total" —
  wrong. That was 32 cores/SM (Kepler-era figure). Correct: 5 x 128 = 640 per die,
  2560 on the M10 card. This round's numbers below use the corrected model.

Also correcting a misattribution I nearly repeated: **llama.cpp (ggml-org) PR #2740
is NOT an MMVQ PR** — it is a 2023 PR "Explain the user that GGML isn't supported
anymore" (IgnacioFDM, closed unmerged, 2023-10-08), verified via
https://api.github.com/repos/ggml-org/llama.cpp/pulls/2740 . The 2-warp MMVQ PR with
M10 numbers is **ik_llama.cpp #2740** (different repo, 2026). Do not cite "llama.cpp
#2740" for MMVQ work.

---

## 1. TASK 1 — Power efficiency on Maxwell (t/s vs `nvidia-smi -pl`)

Bottom line up front: **I found no published tokens-per-second vs power-cap sweep
for M10, M40, M60, or GTX 9xx.** If you build one, it appears to be novel. What
does exist:

### Minimum/maximum power limits (measured by owners)

Source: r/LocalLLaMA "Nvidia Tesla power limits" (3y ago),
https://www.reddit.com/r/LocalLLaMA/comments/17a401e/nvidia_tesla_power_limits/
(the OP asked for min `-pl` on P4, P40, P100, M40, M60; full thread fetched):

- K80: "got tesla k80, 100w min" (u/Good-Juggernaut-740)
- P40: "P40 shows min 125W, max 250W" (u/harrro)
- P100: "P100, 125W min" (u/Potential-Net-9375)
- M40 / M60: **no answers in the thread** (8 comments, none addressed them).

So for the Maxwell Teslas you asked about (M40, M60) there is not even a
community-confirmed min/max `-pl` — worth measuring first, since whether
`-pl` is even honored on M40/M60/M10 (PCIe-only cards) is undocumented in my
sources.

### Old-Tesla speed data found (context for a power sweep)

Source: r/LocalLLaMA "Anyone have speeds for old Tesla cards?" (3y ago),
https://www.reddit.com/r/LocalLLaMA/comments/1bgk0wh/anyone_have_speeds_for_old_tesla_cards/
(full thread fetched):

- u/MachineZer0: "P40 is great for GGUF. P100 for EXL2. M40 is still supported but
  a really slow version of P40, but prevalent model has 12gb. The 24gb version
  doubles the price."
- u/firearms_wtf: 4xP40, dual Xeon E5-2667, 128GB DDR3-1600:
  - Falcon-180 (Q3_K) ~5 T/s @ 512 CTX
  - Goliath-120 (Q5_K) ~7 T/s @ 512 CTX
  - Senku-70B-Full (Q6_K) ~7 T/s @ 512 CTX
- u/noneabove1182: "I benchmarked mixtral q3_k_m on the p40 and was getting 20t/s"
  (single P40).
- u/Mass2018: 5xP100, ExLlamaV2 4.25bpw Goliath 120B, 12k context, RoPE 4.5
  (imgur chart in thread): <1000 ctx 4.3405 t/s; 1000-1999 3.3141; 2000-2999
  2.7245; 3000-3999 2.514; 4000+ 2.248.

### Adjacent data point (power cap on an older-gen card, not Maxwell)

ik_llama.cpp PR #2701 (see section 4) reports an RTX A2000 benchmarked "at a 60 W
power cap, with the SM clock at 600-800 MHz and the memory clock at 5701 MHz" —
shows what a cap does to clocks on a datacenter card, but it is Ampere and the
author deliberately excluded it from the merged change (run-to-run spread 3.8%).
No Maxwell equivalent found.

### M10 power context (from Round 1 rig data, re-stated for completeness)

The 6xM10 rig (reddit 1xnl82r): 700-1300 W whole system incl. RTX 3070 Ti.
M10 itself is a 75 W PCIe-power-only card. No per-cap M10 t/s data exists in any
source I could reach.

GTX 9xx: my `web_search` for "GTX 980" OR "GTX 970" power limit LLM inference
returned only a 386-char snippet set with no usable numbers [SNIPPET].

**Recommendation:** the sweep itself (M10 at, e.g., 25/35/45/55/75 W with
`llama-bench -p 0 -n 128` + `nvidia-smi dmon` clock readout, Q4_K_M 8B on a
single die vs `-sm layer`/`-sm graph` across dies) is publishable; nothing to
dedup against exists.

---

## 2. TASK 2 — Draft models for Qwen3.5-2B / Qwen3.5-4B targets

### 2.1 Qwen3.5-0.8B as a draft for 2B/4B

- Family/tokenizer: all Qwen3.5 sizes (0.8B/2B/4B/9B) come from the same Qwen
  family; llama.cpp PR #31535 lists them side by side as
  `ggml_Qwen3.5-{0.8B,2B,4B,9B}-Q4_0[.gguf]` and `...-imatembd...` variants.
  llama.cpp requires draft and target to share a vocabulary — same-family, so
  0.8B should load as `-md` draft for 2B/4B, but **no public confirmation or
  acceptance-rate number for 0.8B-draft vs 2B/4B-target exists** (I searched
  llama.cpp issues and HF). Verify at load time; measure acceptance locally
  (`speculative acceptance` in the server log).
- Relative size evidence (NOT NVIDIA — Snapdragon SM8850 HTP, llama.cpp PR
  #31535, https://github.com/ggml-org/llama.cpp/pull/31535), Q4_0, tg128:
  - Qwen3.5-0.8B: 88.80 +/- 0.15 t/s (imatembd) / 82.10 (plain)
  - Qwen3.5-2B: 49.27 / 44.21
  - Qwen3.5-4B: 22.40 / 21.51
  - Qwen3.5-9B: 12.24 / 12.38
  So the 0.8B draft runs ~2x the 2B and ~4x the 4B — the right ratio to make
  basic-draft spec-decoding pay off on a GEMV-bound card.

### 2.2 EAGLE-3 for small Qwen3.5

HF API search `Qwen3.5 eagle3` (fetched 2026-11-08). For **2B or 0.8B: none
found**. Available:

- yuyijiong/Qwen3.5-4B-Eagle3 (2026-08-07, 199 dl, base Qwen/Qwen3.5-4B,
  transformers + TGI endpoint tags)
- yuyijiong/Qwen3.5-9B-Eagle3 (2026-08-07)
- BLR2/Qwen3.5-9B-Eagle3-ShareGPT (2026-03-07, 5 likes) + GGUF:
  dobriak/Qwen3.5-9B-Eagle3-ShareGPT, dobriak/Eagle3-Qwen3.5-9B (gguf tag)
- jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge (2026-03-14, 6 likes)
- VirVen/Qwen3.5-27B-EAGLE3-v2 (2026-07-07)
- circulus/Qwen3.5-4B-eagle3-ov-int4 (2026-08-09, OpenVINO)

**Runtime caveat:** llama.cpp mainline docs (docs/speculative.md, fetched
round 4 and re-checked) document draft-model and MTP spec decoding; I found no
EAGLE-3 section in the fetched doc. The EAGLE-3 weights above target
sglang/vLLM/transformers/OpenVINO — they are not usable through llama.cpp's
`-md` path as-is.

### 2.3 DFlash (z-lab) for Qwen3.5 — official drafts, measured numbers

DFlash = block-diffusion draft model, paper arXiv:2602.06036, code
github.com/z-lab/dflash, joint Z-Lab + Modal retrain (40k seq len, sliding-window
attention). HF API search `Qwen3.5 dflash` (fetched 2026-11-08). Official z-lab
drafts: **Qwen3.5-4B** (z-lab/Qwen3.5-4B-DFlash, 44 likes, 14,529 dl,
2026-03-05), Qwen3.5-9B (7,833 dl), Qwen3.5-27B (1,959 dl), Qwen3.5-35B-A3B
(1,546 dl). **No 2B or 0.8B DFlash exists** — 4B is the smallest target with an
official draft.

z-lab/Qwen3.5-4B-DFlash README (full card fetched,
https://huggingface.co/z-lab/Qwen3.5-4B-DFlash):
- "It is not a standalone language model. It is intended to be paired with the
  target model in a speculative decoding server."
- Usage is SGLang: `--speculative-algorithm DFLASH --speculative-draft-model-path
  z-lab/Qwen3.5-4B-DFlash --speculative-dflash-block-size 8` (block 8 = default
  for concurrency; 16 = longer accept lengths, strong concurrency-1). vLLM via
  vllm-project/vllm#40898, "will be updated to make it merge-ready soon".
  **llama.cpp does not support DFlash.**

Measured (SGLang, 1x NVIDIA B200, bf16, greedy, thinking on, max 4096 out, 5
runs; output tok/s with speedup in parentheses):

Concurrency 1 (baseline ~356 tok/s):
| workload | MTP s=3 | DFlash b=4 | MTP s=7 | DFlash b=8 | MTP s=15 | DFlash b=16 |
|---|---|---|---|---|---|---|
| gsm8k | 739.3 (2.08x) | 859.6 (2.41x) | 772.0 (2.17x) | 1226.8 (3.45x) | 585.3 (1.64x) | 1387.4 (3.90x) |
| math500 | 763.7 (2.12x) | 899.4 (2.50x) | 832.5 (2.31x) | 1355.7 (3.76x) | 645.1 (1.79x) | 1636.5 (4.54x) |
| humaneval | 739.7 (2.08x) | 892.2 (2.51x) | 803.0 (2.26x) | 1325.2 (3.72x) | 594.7 (1.67x) | 1634.9 (4.60x) |
| mbpp | 723.9 (2.01x) | 895.6 (2.49x) | 737.2 (2.05x) | 1314.9 (3.65x) | 557.4 (1.55x) | 1494.9 (4.15x) |
| mt-bench | 708.8 (1.99x) | 806.7 (2.26x) | 699.0 (1.96x) | 1085.3 (3.04x) | 528.7 (1.48x) | 1211.0 (3.40x) |

Concurrency 32: DFlash b=8 best: gsm8k 17613.7 (2.35x), math500 19759.4 (2.61x),
humaneval 18792.5 (2.58x), mbpp 17908.0 (2.53x), mt-bench 14623.7 (2.15x);
MTP s=15 degrades badly at concurrency (1.10-1.28x).

Mean accept length, concurrency 1 (this is the hardware-independent part — most
transferable number to your M10 expectations):
| workload | MTP s=3 | DFlash b=4 | MTP s=7 | DFlash b=8 | MTP s=15 | DFlash b=16 |
|---|---|---|---|---|---|---|
| gsm8k | 3.422 | 3.427 | 5.133 | 5.299 | 6.175 | 6.748 |
| math500 | 3.502 | 3.528 | 5.345 | 5.650 | 6.468 | 7.478 |
| humaneval | 3.448 | 3.551 | 5.193 | 5.684 | 6.147 | 7.719 |
| mbpp | 3.272 | 3.418 | 4.611 | 5.236 | 5.326 | 6.527 |
| mt-bench | 3.266 | 3.234 | 4.626 | 4.704 | 5.610 | 5.933 |

### 2.4 Practical read for llama.cpp on the M10

For Qwen3.5-2B/4B on M10 via llama.cpp mainline, the usable options are:
1. Qwen3.5-0.8B (Q4_0 or Q5_K_M) as `-md` basic draft — same tokenizer family,
   acceptance unmeasured publicly, must be measured locally;
2. MTP, if the target GGUF preserves MTP layers (Round 4: the SC117 Qwen3.6
   GGUF keeps native MTP; check the Qwen3.5-2B/4B GGUFs you load for
   `nextn`/MTP tensors before assuming).
EAGLE-3/DFlash weights exist for the 4B target but only for sglang/vLLM
runtimes, not llama.cpp.

---

## 3. TASK 3 — Quant choices for small models with huge vocab (Qwen3.5, 248k vocab)

### 3.1 Tooling state (Nov 2026)

- Per-tensor override exists in `llama-quantize` (`-ot name=type`) for
  `output.weight` and `tok_embeddings.weight` (established behavior; used
  everywhere in the GGUF ecosystem).
- **Open PR llama.cpp #31387** (Dorijan10, opened 2026-10-29, 0 comments,
  label "testing"): "quant : allow manual tensor types with F32, F16 and BF16
  targets" — would let you pin output.weight (and others) to F32/F16/BF16 while
  quantizing the rest. If your plan is "keep output.weight high-precision,
  quant the rest", this PR is the upstream vehicle; it is not merged.
  URL: https://github.com/ggml-org/llama.cpp/pull/31387
- imatrix-with-embeddings variants already exist in the wild: llama.cpp PR
  #31535 (Hexagon Qwen3.5 fusion) tests `ggml_Qwen3.5-2B-Q4_0-imatembd.gguf`
  etc. ("imatembd" = imatrix including the embedding tensor).
  On that backend (SM8850 HTP, Q4_0), imatembd vs non-imatembd tg128:
  0.8B 88.80 vs 82.10 (+8.2%), 2B 49.27 vs 44.21 (+11.4%), 4B 22.40 vs 21.51
  (+4.1%), 9B 12.24 vs 12.38 (worse). Not NVIDIA, but directionally: embedding
  imatrix helps small Qwen3.5 models most at the small sizes.

### 3.2 Measured PPL data found (all from fetched PR bodies)

- llama.cpp PR #30035 (hexagon q2_k, merged 2026-10-12,
  https://github.com/ggml-org/llama.cpp/pull/30035), PPL wikitext-2, 16 chunks,
  -c 512:
  - Llama-3.2-1B Q2_K: 37.4741 (default) / 37.5124 (HVX only)
  - Llama-3.2-3B Q2_K: 17.8196 / 17.8175
  - Qwen3.5-2B `--pure` Q2_K (imatrix): value present in the table but the
    number was truncated in my fetch [PARTIAL] — only the "-0.0%" delta visible
  - Qwen3.5-2B Q2_K (bartowski): 18.6689 -> 18.6667
  So a Qwen3.5-2B at Q2_K lands around PPL 18-19 on wikitext-2; the imatrix
  pure-Q2_K variant is in the same table (value to be re-fetched if you want it
  exactly).
- ik_llama.cpp PR #2721 (fetched full, section 4): `-ub 1` PPL table, ctx 512,
  4 chunks (2 on M10), Llama-3.1-8B Q4_K_M: P100 4.9128, 2xP100 layer/graph
  4.9128/4.9011, RTX 4000 4.9125, P40 4.9109, Titan V 4.9032, **M10 5.3221**.
  The M10 PPL is ~8% higher than the P40's for the same Q4_K_M file — an
  on-hardware observation of Maxwell-side precision difference (no FP16
  accumulate on sm_50) worth knowing when you compare PPL across your cards.

### 3.3 The actual gap

No public PPL-vs-speed matrix exists for Qwen3.5 (or any 248k-vocab model)
varying output.weight type (F16 vs Q8_0 vs Q4_0) or tok_embeddings type at
Q4_0/IQ4_XS, on any GPU. With the output+embedding pair being ~a third of a 2B
model's weights (your figure), that is a big lever and nobody has published it
for this family. Your planned sweep would be the first. One structural note for
the sweep: on the M10, output-weight GEMV is n = 248k rows — it is dominated by
the same MMVQ geometry work as section 4, so output-tensor type changes will
interact with nwarps/rows-per-block choices, not be independent.

---

## 4. TASK 4 — MMVQ nwarps / rows_per_block upstream state (dedup map)

Everything below was fetched in full via api.github.com today (2026-11-08), not
snippets.

### 4.1 ik_llama.cpp (mb8565) — all three merged Oct 2026

**ik #2701** — https://github.com/ikawrakow/ik_llama.cpp/pull/2701
"CUDA: use 2 rows per block in MMVQ for one column on sm_5x, sm_60 and sm_75",
created 2026-10-19, merged 2026-10-19 (closed by ikawrakow).
- "With one column, MMVQ computes one row per block. On sm_50, sm_60 and sm_75,
  two rows per block are faster. TG-128 of Llama-3.1-8B Q4_K_M goes up 12 % on
  a P100 and on a Quadro RTX 4000. On sm_61 and sm_70, forced two rows were
  neutral or slower."
- "So the result does not follow dp4a: sm_61, sm_70 and sm_75 all have dp4a,
  and only sm_75 gets faster. The change applies only when the device code was
  compiled for an arch below 610, or for sm_75. For sm_61, sm_70, sm_80 and
  sm_86 the PTX of the one-column kernels is identical to main."
- "Q3_K keeps one row, because with two rows its fused up/gate kernel goes from
  116 to 195 registers on sm_60 and gets slower."
- TG-128 (t/s), Llama-3.1-8B unless noted:
  | GPU (arch) | model | main | PR | change |
  |---|---|---|---|---|
  | P100 (sm_60) | Q4_K_M | 42.36 | 47.49 | +12.1 % |
  | P100 | DS-V2-Lite REAP48 Q4_K_M | 82.31 | 91.34 | +11.0 % |
  | P100 | Q3_K_M | 36.21 | 38.18 | +5.5 % |
  | P100 | IQ3_S / IQ1_M | 35.73 / 48.96 | 37.29 / 53.04 | +4.4 / +8.3 % |
  | 2x P100 -sm layer | Llama / DS | 42.23 / 82.00 | 46.71 / 90.97 | +10.6 / +10.9 % |
  | 2x P100 -sm graph | Llama / DS | 66.44 / 101.26 | 73.90 / 110.17 | +11.2 / +8.8 % |
  | RTX 4000 (sm_75) | Q4_K_M | 63.40 | 70.99 | +12.0 % |
  | RTX 4000 | DS-V2-Lite Q4_K_M | 108.85 | 117.99 | +8.4 % |
  | **M10 (sm_50), two rows forced** | **Q4_K_M** | **6.60** | **7.70** | **+16.8 %** |
  | P40 (sm_61), two rows forced | Llama / DS | 47.82 / 92.03 | 47.94 / 87.91 | +0.3 / -4.5 % |
  | Titan V (sm_70), two rows forced | Llama / DS | 104.27 / 96.78 | 88.42 / 91.53 | -15.2 / -5.4 % |
- M10 baseline from this PR: Llama-3.1-8B Q4_K_M single-die decode = 6.60 t/s
  main, 7.70 t/s with 2 rows (sm_50 is covered by the merged "<610" rule, so
  the gain ships on the M10).
- Bit-identical results ("each row is summed by the same threads in the same
  order"); -ub1 PPL unchanged: P100 4.9096 (Llama Q4_K_M) / 5.1215 (Q3_K_M);
  RTX 4000 4.9111 / DS 15.1494.
- Sweep, `llama-sweep-bench -c 16384 -ub 512`, one P100: TG gain shrinks with
  context (N_KV 0: +10.8% ... 15872: +7.3%); "The PR saves 2.3 ms per token at
  every context"; PP unchanged (~477 -> 341 t/s as ctx grows, both builds).

**ik #2721** — https://github.com/ikawrakow/ik_llama.cpp/pull/2721
"CUDA: keep the fused MMVQ per-row sums in registers", created 2026-10-21,
merged 2026-10-21. This is the **sm_50 local-memory spill fix for #2701**:
- "#2701 uses two rows per block in MMVQ for one column on sm_5x, sm_60 and
  sm_75. With two rows, `fused_mul_mat_vec_q` gets a 32-byte local memory stack
  frame on sm_5x and sm_60. The cause is `tmp_u[j][threadIdx.x]`,
  `tmp_g[j][threadIdx.x]` and the two `local_bias` arrays. They are indexed
  with `threadIdx.x`, so the compiler puts them in local memory."
- Fix: "selects the row with an unrolled compare. The bias goes into a scalar,
  because each thread only reads the bias of its own row."
- "In my sm_60 build (CUDA 12.4), 34 of the 66 one-column fused kernels had a
  stack frame. With the PR, none has one. ... My sm_50 build for the M10 is
  the same, 34 before and none after."
- PPL identical (M10 -ub1 Llama Q4_K_M: 5.3221 main = PR).
- Lesson for any new sm_50 geometry work: **never index per-row temp arrays
  with `threadIdx.x` in the fused kernel** — it spills to local memory on
  Maxwell and eats the win.

**ik #2740** — https://github.com/ikawrakow/ik_llama.cpp/pull/2740
"CUDA: use 2 warps in MMVQ for Q2_K and Q3_K with one column on older GPUs",
created 2026-10-23, merged 2026-10-23.
- "With one column, `mul_mat_vec_q` uses 4 warps. This PR uses 2 for Q3_K below
  Ampere and for Q2_K below Turing. On sm_75 the Q2_K gain was inside the
  noise. llama.cpp #29753 (sm_70) and #25479 (sm_61, open) use 2 warps for more
  types. Here, Q4_K, Q5_K, Q6_K (P40, Titan V), Q4_0 and Q8_0 (P40) were not
  faster."
- tg128 t/s, Llama-3.1-8B-Instruct, `llama-bench -p 0 -n 128 -r 5 -ngl 99 -fa 1`
  (`-r 3` on M10):
  | GPU | Q3_K_M main | Q3_K_M PR | change | Q2_K main | Q2_K PR | change |
  |---|---|---|---|---|---|---|
  | **M10 (sm_50)** | **5.80** | **6.38** | **+10.0 %** | **8.445** | **9.410** | **+11.4 %** |
  | P100 (sm_60) | 38.17 | 41.86 | +9.7 % | 54.65 | 60.05 | +9.9 % |
  | P40 (sm_61) | 41.35 | 44.62 | +7.9 % | 61.14 | 64.03 | +4.7 % |
  | Titan V (sm_70) | 90.92 | 97.58 | +7.3 % | 134.84 | 143.62 | +6.5 % |
  | RTX 4000 (sm_75) | 51.70 | 54.61 | +5.6 % | 80.38 | 80.32 | same kernel |
  | 2x M10, -sm layer | 5.573 | 6.506 | +16.7 % | | | |
  | 2x M10, -sm graph | 10.25 | 11.16 | +8.9 % | | | |
  | 2x P100, -sm layer | 38.086 | 41.764 | +9.7 % | | | |
  | 2x P100, -sm graph | 62.26 | 66.95 | +7.5 % | | | |
  | 2x P40, -sm layer | 42.25 | 46.06 | +9.0 % | | | |
  | 2x P40, -sm graph | 56.57 | 59.72 | +5.6 % | | | |
  | 2x Titan V, -sm layer | 90.16 | 96.56 | +7.1 % | 133.68 | 142.38 | +6.5 % |
  | 2x Titan V, -sm graph | 127.88 | 133.66 | +4.5 % | 166.35 | 172.85 | +3.9 % |
- "The Q4_K_M control launches the same kernels as main and changes by 0.12 %
  or less." At -ub 2/-ub 4 logits bit-identical; at -ub 1 summation order
  changes (KLD table present in PR, truncated in my fetch).

### 4.2 llama.cpp mainline (ggml-org)

- **PR #16715** (commit f77c13b, per issue #30334's citation): added the MMVQ
  gate/up **fusion** and the Pascal rule in
  `ggml_cuda_should_fuse_mul_mat_vec_q`:
  `if (cc <= GGML_CUDA_CC_PASCAL) { return false; }` — the constant is 600, so
  610 (P40) is NOT covered and fusion stays on there.
- **PR #20905** (gaugarg-nv, merged 2026-03-29): "Optimize MOE GEMV kernel for
  BS > 1" — dedicated MoE multi-token `mul_mat_vec_q_moe` kernel,
  "grid (ceil(nrows_x/rpb), nchannels_dst), block (warp_size, ncols_dst). Each
  warp independently handles two rows, with only warp-level reduction (no
  shared memory synchronization)." RTX 5090, qwen3moe 30B.A3B / qwen35moe
  35B.A3B, pp512: Q4_0 ub4 598 -> 726.8 t/s (+21.7%); overall +7%..+22% across
  Q4_0/Q4_K_M/Q8_0, ub2-ub4. Maintainer (per #30334): "On NVIDIA Pascal or
  older disable for iq3_s, q2_k, and q3_k" (cc < 700), plus per-card batch
  limit "4 for Q2_K and Q3_K for pascal".
- **PR #27828** (per #30334 citation): "P100 (cc == 600) always use MMVQ for MoE
  multiply up to batch 8, 6.1 cards not included."
- **PR #29753** (per ik #2740 citation): 2 warps for more types, sm_70.
- **PR #25479** (per #30334 citation, **unmerged draft**): "2 warps instead of
  4 at batch 1 for 6.1 and 6.2 cards, gtx1060. unmerged draft and it shows how
  many warps MMVQ kernel uses at batch 1, 6.1 cards".
- **Issue #30334** (ymr-gif, OPEN, 2026-10-11, 0 comments, label
  bug-unconfirmed): "Misc. bug: CUDA: P40 and the MMVQ fusion check for pascal.
  the fusion check misses 6.1, so it stays on for P40".
  https://github.com/ggml-org/llama.cpp/issues/30334 (full body fetched).
  Environment: "Tesla P40, compute capability 6.1", driver 550.163.01, CUDA
  12.4, "version: 0.6.0-dev (build 33, commit 1623d8c)".
  PR history table from the issue:
  | PR | check | result |
  |---|---|---|
  | #16715 | <= 600 | misses 610 |
  | #20905 | < 700 | covers 610 |
  | #27828 | == 600 | misses 610 |
  Fusion off vs stock, tg128, tok/s median of 6 (Mixtral-8x7B Q2_K / Q3_K_M,
  Llama-3.1-8B Q4_0 / Q4_K_M):
  | file | d0 | d496 | d2016 |
  |---|---|---|---|
  | Mixtral Q2_K | 32.42 -> 30.65, -5.46% | 32.03 -> 30.30, -5.41% | 31.37 -> 29.69, -5.35% |
  | Mixtral Q3_K_M | 25.76 -> 27.12, +5.29% | 25.54 -> 26.83, +5.06% | 25.06 -> 26.35, +5.15% |
  | Llama Q4_0 | 52.28 -> 54.75, +4.72% | 51.39 -> 53.69, +4.47% | 49.72 -> 51.87, +4.33% |
  | Llama Q4_K_M | 48.94 -> 48.34, -1.21% | 48.04 -> 47.47, -1.19% | 46.53 -> 46.05, -1.04% |
  Small-K rule (from the issue): `use = nwarps > 1 && blocks_per_row_x <
  nwarps * blocks_per_iter_1warp` (K = 4096, limit under 2048) + a
  slow_pascal list. Fusion off + forced small-K, Mixtral 8x7B Q2_K, P40,
  tg128:
  | depth | stock | fusion off | fusion off + small-K |
  |---|---|---|---|
  | 0 | 32.42 | 30.65, -5.46% | 36.33, +12.04% |
  | 496 | 32.03 | 30.30, -5.41% | 35.85, +11.91% |
  | 2016 | 31.37 | 29.69, -5.35% | 35.07, +11.78% |
  pp512, -ub 1/2/4/8: +12.15% / -2.78% / -1.42% / -0.06% (small-K arm).
  "fusion off + small_k(forced) gives about 12% gain on tok/sec, but with
  fusion off only there is a reduction of 5%, at 1 batch".

### 4.3 Dedup verdict for your work

Already merged for the M10 (sm_50) in ik_llama.cpp:
- 2 rows per block: +16.8% (Q4_K_M, forced; active via the <610 arch rule) —
  ik #2701, with the sm_50 spill fixed by ik #2721.
- 2 warps for Q2_K/Q3_K decode: +10.0% / +11.4% — ik #2740.

Not upstream, i.e. genuinely open territory:
- **1 warp / 4 rows per block** (your +50-65% result): nobody upstream has
  proposed this geometry. Upstream settled on 2 rows (per block) and 2 warps
  (per row); your 4-rows-in-1-warp is a different partition and, if the +50-65%
  holds, a larger win than either merged change. An ik PR proposing 4 rows for
  sm_50 (with #2721-style register discipline — no threadIdx.x-indexed per-row
  temps, or it will spill 32+ bytes on sm_50) would not duplicate existing
  work.
- Pascal 610 (P40) is the mainline gap: #30334 open, #25479 unmerged, #27828
  explicitly excludes 6.1. If you also work P40 numbers, that is where mainline
  is asking for help.
- No Maxwell- or Pascal-specific **rows_per_block table** exists in mainline:
  the small-K rule is size-based (blocks_per_row_x vs nwarps*blocks_per_iter)
  plus a type list; there is no cc-keyed table to extend, so a "Maxwell/Pascal
  table" PR would be new.

---

## 5. Gap summary (what does not exist publicly)

1. t/s vs `-pl` on M10/M40/M60/GTX 9xx — nothing.
2. Confirmed min/max `-pl` for M40/M60/M10 — nothing (K80 100W, P40 125/250W,
   P100 125W are the only Maxwell-and-older answers found).
3. Acceptance rate for Qwen3.5-0.8B draft -> 2B/4B target — nothing.
4. EAGLE-3 or DFlash drafts for Qwen3.5-2B or 0.8B — none exist (4B is the
   smallest with an official z-lab DFlash; 4B/9B/27B/35B-A3B have EAGLE-3).
5. PPL-vs-speed for output.weight / tok_embeddings types on 248k-vocab models
   — nothing; PR #31387 (manual F16/BF16/F32 tensor targets) is the relevant
   open upstream vehicle.
6. A cc-keyed rows_per_block or nwarps table for Maxwell/Pascal in mainline —
   nothing; the open items are #30334 (P40 fusion check) and #25479 (P40 2
   warps).
