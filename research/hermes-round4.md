# M10 / Maxwell LLM inference — research round 4

Delivered: /tmp/m10-research-round4.md
Tools: firecrawl-backed web_extract (incl. raw.githubusercontent.com, api.github.com issue search, old.reddit), web_search. One file written, no other hosts/commands.
Rule note: items marked [SNIPPET] rest on search-result text only; everything else was fetched in full this round or a prior round (source URL given).

---

## 0. CORRECTION (supersedes round 3 and one cited repo)

GM107 (as used in the Tesla M10) = **5 SMs x 32 cores = 160 cores per die**. M10 = 4 dies x 160 = **640 cores total**.

- This makes the round 3 report's "16 SMs" note (patch 09 commentary) wrong; 5 SMs is the hardware-grounded figure.
- Cross-check: the 18ocu6q thread (r/LocalLLaMA, fetched via .rss in round 2) gives the M10-class figure "GM107 1.672 TFLOPS / 83.2 GB/s" vs "GP102 11.76 TFLOPS / 347.1 GB/s" (card-to-card comparison). 1.672 TF = 640 cores x 2 FLOP x ~1.31 GHz — consistent only with 640 cores total. If the M10 had 2560 cores at that clock it would be ~6.6 TF, contradicting the published figure.
- 83.2 GB/s per die = 128-bit GDDR5 @ ~5.2 Gbps (one independent 128-bit interface per die).
- **github.com/timoruohomaki/theworkstation (fetched in full this round) claims "2 560 CUDA cores" and "GPU 0-3: 640 cores, 8 GB VRAM" each.** That contradicts the card-level 1.672 TFLOPS figure above; treat their "640 per GPU / 2560 total" as an error (likely read the card's 640 total as per-die). URL: https://github.com/timoruohomaki/theworkstation
- Kernel consequence: 5 SMs per die is very low occupancy headroom. The 1-warp/4-rows-per-block win (+50-65%, your finding) and the nwarps-vs-K sweep (shinbunbun patch 09) are first-class tuning axes, not exotic ones. Also relevant: PR 2580's tile kernel caps at 16 query columns because 32 columns need 53,376 B shared mem > the 48 KiB per-block limit (see section 1).

---

## 1. Verification of ik_llama.cpp PR 2580

- URL: https://github.com/ikawrakow/ik_llama.cpp/pull/2580
- Title: "CUDA: route head size 256 FA prefill to tile_f32 on NVIDIA GPUs before Volta"
- Author: **mb8565** (contributor). Approved by ikawrakow. **Merged 2026-10-02.** Single commit `4a6b63fb3bf674d7ecf62f6d8b8f3add8c624d19` ("CUDA Pascal: route head size 256 prefill to the tile_f32 kernel"), merge commit `93fcd517074fc767d8c903beb89ebaf5be269fb7`.
- Hardware line, verbatim from the PR body: "I have tested on P100 (one and two GPUs), P40 (one and two GPUs) and a Tesla M10 (sm_50)." The Falcon3 benchmark table is captioned: "**Tesla M10 (Maxwell sm_50, one GM107 die)**, Falcon3-10B, 32cddbf vs PR, same flags, arms main, PR, PR, main."
  - i.e. the M10 numbers were measured on a Tesla M10, on **one GM107 die** (one of the 4 CUDA devices / 8 GB slice). They are single-die numbers — directly your configuration.
- What it changes (fetched the .diff this round):
  - `ggml/src/ggml-cuda/fattn-tile-f32.cu`: `launch_fattn_tile_f32_64_128` renamed `launch_fattn_tile_f32_64_128_256`; added `case 256:` (D=256 -> nwarps=8). Comment: "32 columns would need 53376 B of shared memory, more than the 48 KiB per-block limit." `ggml_cuda_fattn_tile_f32_is_supported` extended to `K->ne[0] == 256`.
  - `ggml/src/ggml-cuda/fattn.cu` (non-fast-fp16 path = Maxwell/P100): decode guard stays `Q->ne[1] <= 8` -> vec kernel; the old `|| Q->ne[0] == 256` escape was removed from the vec path and a new `else if (Q->ne[0] == 256)` branch calls `ggml_cuda_flash_attn_ext_tile_f32`. Commit message: "Without fp16 tensor cores, D=256 prefill ran the vec kernel, which re-reads the whole K/V range for every query column. The quantized KV types other than F16/Q8_0 fell back to the CPU. tile_f32 reuses each K/V tile over 16 columns, covers every KV type the tile kernel converts, and accumulates in fp32. Decode (batch <= 8) stays on vec."
- Numbers (PR body table, Falcon3-10B, full fetch):
  - f16 KV: pp2048 main ~3245 -> PR ~3932 (**+21.2%**); pp4096 ~3334 -> ~4502 (**+35.0%**); tg 225.9 -> 235.2 (+4.1%).
  - q8_0 KV: pp2048 5196 -> 16,280 (**+213%**); pp4096 5220 -> 16,700 (**+219.9%**); tg 228.1 -> 229.3.
- Follow-ups in the same lane (ik_llama.cpp):
  - PR #2647 "CUDA Pascal: run head size 512 flash attention on the GPU" (merged; referenced 2026-10-11) — Gemma global layers (D=512) now run FA on GPU on Maxwell too.
  - PR #2670 "docs: add Pascal and Volta support status" (merged).
  - Bonus from the diff: comment in fattn.cu cites llama.cpp#25593 — "P100 (sm_60/CC_PASCAL): fp16 vec kernel accumulates online-softmax denominators and P*V products in fp16, and against fp32 about 3-4% of decode top-1 tokens invert (llama.cpp#25593). Decode is bandwidth-bound on P100, so vec_f32 is free." (Context for why decode stays vec_f32 on pre-Volta.)

---

## 2. Strata-maxwell (verified: real repo, fetched in full this round)

- Repo: https://github.com/BitMakerMan/Strata-maxwell (MIT; 4,213 commits; 0 stars/forks — personal proof-of-work repo). Tagline: "Strata optimized for RTX 3060 + Tesla M10: MoE expert offload into the M10's 32 GB VRAM — a measured proof of work (full 256K context, 92% of routed experts offloaded)".
- What it is: a fork of the **Strata** MoE serving engine (stock baseline = "strata v0.2.6"); the fork adds an M10 CUDA expert tier on top of a 3060 attention/dense tier. Layout: `strata_cli/`, `strata_control/`, `serve/`, `metal/`, `sycl/` (multi-backend engine), `bench/` (benchmarks + results index), WORKLOG.md. The upstream "Strata" repo URL was not found — [flag] the fork is the only citable source.
- Host (bench): RTX 3060 + 4x M10, 4-core AVX2 CPU, 39 GB RAM. Model: Qwen3.8-Flash-Next IQ2_XS pack, **full 256K context**.
- Expert placement (bench/results/2026-10-25-m10-vs-standard/README.md, full fetch):
  - 22,702 of 24,576 routed experts resident in M10 VRAM: **5676 + 5676 + 5675 + 5675** per die, 7.61-7.63 GiB per die (of 8 GB).
  - 1,874 expert cache slots on the 3060 (2.53 GiB).
  - `--resident-experts` leaves 4 GiB of host RAM free for OS + KV + draft state; the alternative `--resident-cpu-experts` reserves a hardcoded 8 GiB headroom (generate.cpp:544) — "that's the difference between ~19 GB and ~39 GB of headroom."
  - "The M10 tier prefills in 64-token windows" (low bandwidth tier); engine refuses to serve with a half-populated cache; load is 4-6 minutes.
- Flags: `--expert-cache 1800 --vram-reserve-mib 976`, `--ple-io direct`, `--resident-experts`; prefill recipe: auto + `MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 FASTCOMPLETE=1` ("all tuned for the M10 tier").
- Numbers (fetched in full; "A" = optimized, "B" = stock strata v0.2.6):
  - Prefill (tok/s): A 148.5 / 144.5 / 154.0 / 154.3; B 99.9 / 430.3 / 434.9 / 275.3 (rows are different workload rows; quote per-row, don't compare across A/B rows blindly — B's high rows are the CPU-resident expert path on different inputs).
  - Decode (tok/s): A **69.5 / 70.1** (234 / 258 accepted drafts); B **35.9 / 52.5** (202 / 300 and 207 / 285).
  - Repo README: "decode runs at ~1.3x the warm CPU-RAM rate (70.1 vs 52.5) — flat from the first run" and "leaves all 39 GB of host RAM free."
- Interpretation for your rig: this is the strongest public demonstration of "experts page-resident in M10 VRAM, attention elsewhere" at full 256K context, and it matches your 20-way expert-sharding architecture from the 1xnl82r thread (different software, same physics).

---

## 3. MoE on old / low-bandwidth NVIDIA GPUs with llama.cpp (current mainline, Nov 2026)

### 3.1 The actual flag set (verified from docs + PRs/issues this round)

- `--n-cpu-moe N` / `-ncmoe N` — "keep the MoE weights of the first N layers on CPU" (expert offload). The mainline MoE-offload flag; `99` = all MoE layers.
  - Docs PR (open): https://github.com/ggml-org/llama.cpp/pull/29343 "docs: explain --n-cpu-moe with multiple GPUs in multi-gpu.md" — key caveat: with multiple GPUs, "memory is freed on GPU0, but the last GPU runs out of memory" — rebalance with `-ts` or `-ot`. Related issues #15136, #15263, #16579.
  - Perf PR (open): https://github.com/ggml-org/llama.cpp/pull/20596 "ggml-cpu: improve --n-cpu-moe TG performance" (special n_tokens=1 gated path): 5090, qwen35moe 35B.A3B Q4_K_S tg32: 26.20->28.94 (t2), 38.64->40.64 (t4), 40.33->43.89 (t8), 40.32->43.25 (t16).
- `--cpu-moe` — all MoE experts to CPU (used e.g. as `--cpu-moe -sm layer` in https://github.com/ggml-org/llama.cpp/issues/31510).
- `--moe-cache-mib N` — MoE expert cache in VRAM, per device per `--ts`. Common value in the wild: 2048. Draft of auto mode: PR #31286 `--moe-cache-cpu-share auto` (sanmai, open draft, 2026-10-28). Per-device allocation request (open): issue #30163 (comma-separated `--moe-cache-mib` + `--ts 1,0` + `--ot blk...`).
- Known bug: `llama_moe_cache::impl::copy` crash on ROCm/Vulkan at load with `--moe-cache-mib` + `--n-cpu-moe 99` on Qwen3.6-35B-A3B (issue #31675, RX 6800 XT; same failure reported for Vulkan in #30697; fix PR #31775 open, 2026-11-06). CUDA not reported to hit it.
- `-ot / --override-tensor NAME=OP` — tensor placement override (per multi-gpu.md). This is the manual tool for putting specific expert tensors on a specific M10 die; no dedicated `--moe-device`/`--kv-device` flags exist (feature request #23855 closed not_planned, 2026-05 — it explicitly names "Tesla P40, Quadro P5000, GTX 1080Ti" as KV-cache-memory cards and "Qwen3.6-35B-A3B on a 2080Ti for the MoE layers" as the use case).
- `-sm layer` (default split) vs `row` (deprecated) vs `tensor` (experimental; NCCL; requires FA on + non-quantized KV; **not implemented for many MoE/hybrid architectures — the not-supported list in multi-gpu.md includes "Nemotron-H, Nemotron-H-MoE"**). For 4 M10 dies: `-sm layer` is the only realistic mode; use `-ts` for imbalance.
- `GGML_CUDA_P2P=1` — opt-in CUDA P2P; docs warn "may cause crashes or corrupted outputs on some motherboards or BIOS configurations (e.g. when IOMMU is enabled)."
- KV offload: `--no-kv-offload` (seen abbreviated `-nkvo`) is used in the wild with hybrid MoE models: `--n-cpu-moe 99 -nkvo --moe-cache-mib 2048` (Qwen3.6-35B-A3B, PR #31775 repro).
- SSD streaming (open PR, not merged): https://github.com/ggml-org/llama.cpp/pull/25294 "llama: stream MoE routed experts from disk" — `--moe-stream-cache` (GiB budget or `s` = slots; implies `--moe-stream`), `--moe-stream-io-threads N`, `--moe-stream-direct` (O_DIRECT); "Wave-Partitioned Prefill" removes the previous n_ubatch clamp when a ubatch touches more experts than the cache can hold. Bench (GB10, GLM-5.2-UD-Q2_K_XL ~254 GB file): 64-slot prefill 2.28 -> 5.65 tok/s (waved), decode ~1.83-2.20.

### 3.2 The measured playbook from a 12 GB card (most transferable source found)

Repo: https://github.com/shiqikuangsan31/Qwen3.6-35B-12GB-VRAM-Guide (fetched in full this round; RTX 4070 SUPER 12 GB / ~504 GB/s, i5-14600KF, 32 GB RAM, ik_llama.cpp b5095 Windows, measured 2026-08-11). Qwen3.6-35B-A3B, APEX I-Compact GGUF 15.85 GB, full 258K native context.

- Profiles: 32K ctx -> ncmoe 16, **64.0 t/s**; 64K -> ncmoe 17, 60.7 t/s; 128K -> ncmoe 20, 55.3 t/s; **258K -> ncmoe 22 + q4_0 KV, 50.9 t/s, 45K input in 85.5 s**.
- The "expert cliff": sweep `--n-cpu-moe` around the cliff; below it the GPU can't hold experts + KV -> dynamic spill -> PCIe thrashing. At 258K (q8 KV) the scan reads: ncmoe 18: 23.3, 19: 23.5, 20: 24.6, 21: 20.5, **22: 51.2**, 23: 48.5, 24: 49.3, 25: 47.8, 26: 46.9, 27: 45.4, 28: 45.0. Rule: "the sweet spot is the first stable point above the cliff," and it shifts upward with context (32K->16, 64K->17, 128K->20, 258K->22).
- Long-input prefill collapse and fix: 45K input at 258K ctx: q8_0 KV = 397 s (113 tok/s effective); q8_0 + `-fa on` = >600 s (**worse**); **q4_0 KV = 85.5 s (527 tok/s) — 4.6x**. "q4_0 KV does not slow decode (50.9 vs 51.2) and is near-lossless for Qwen (KL ~0.10 overall; Gemma degrades to KL 1.09)."
- Negative optimizations (their words): "`fa on` and large `-b 4096` batches are negative optimizations under VRAM offload." For an M10 die (83.2 GB/s, half the VRAM of their test card), expect these findings to be at least as true.
- Core command (258K profile): `-ngl 99 --n-cpu-moe 22 -c 262144 --cache-type-k q4_0 --cache-type-v q4_0 --no-mmap --mlock --reasoning off --reasoning-format deepseek --jinja --threads 12`.
- Pitfalls they hit: stale server corrupts benchmarks; prompt-cache fake speedups; `--n-cpu-moe 99` OOMs at 258K on 16 GB model / 32 GB RAM (pinned-memory exhaustion) — don't copy "all experts CPU" blindly on a RAM-tight box.

### 3.3 Other measured A3B-class numbers (non-Maxwell, for calibration)

- RX 6800 XT (16 GB, ~320 GB/s), Qwen3.6-35B-A3B-UD-Q4_K_XL, -c 262144, f16 KV, `-b 256 -ub 128 --n-cpu-moe 99 --no-kv-offload -ngl 99 -t 8`: baseline **~16.5-17 t/s decode** (issue #31675, fetched this round). Note `-b 256 -ub 128` — this is the only in-the-wild ubatch setting tied to a MoE 262K-ctx command line.
- M3 Max (512 GB/s): Nemotron-3-Nano-30B (128 experts top-6, 23 layers) tg128 78.4 -> 80.5 (+3%) with fused TOPK_MOE (PR #31390, open; "CUDA and Vulkan already fuse this" for the Nemotron-H style).
- GB10 DGX Spark: GLM-5.2 Q2_K_XL, 1.83-2.20 t/s decode (PR #25294 above).
- 5090, qwen35moe 35B.A3B Q4_K_S: 26-44 t/s tg32 with/without PR #20596 (above).
- **No direct M10/M40/P40/GTX 1080 Ti A3B measurements found** in llama.cpp issues/PRs or on the web this round — flagged as a gap. P40 (347 GB/s, 24 GB) and 1080 Ti (484 GB/s, 11 GB) sit between the RX 6800 XT and the 4070 SUPER in the offload regime; that is interpolation, not measurement.

### 3.4 The two target models

- **Qwen3.6-35B-A3B**: 35B total / 3B active, 256 routed experts, 8 active/token, 40 transformer layers + MTP layer(s), 262,144 native ctx (model card via HF: Qwen/Qwen3.6-35B-A3B, Apache-2.0). Hybrid: PR #31775's repro shows GATED_DELTA_NET layers (recurrent state) — hence `--no-kv-offload` usage. Sizes seen: 22.36 GB Q4_K_M; 15.85 GB APEX I-Compact; 24.18 GB APEX I-Balanced. SWE-bench Verified 73.4, AIME 2026 92.7 (12GB guide).
- **Nemotron-3.5 (Lightning) 30B A3B**: model = `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (unsloth docs fetched this round: https://unsloth.ai/docs/models/nemotron-3.5 — "30B-parameter, 3B-active hybrid reasoning MoE," up to 1M ctx, "runs on 20GB RAM at 4-bit, 33GB at 8-bit"; OpenMDW-1.1; "35% faster than Qwen 3.6 35B on 10,000 PinchBench tasks"). llama.cpp arch = `nemotron_h_moe` (31B.A3.5B, 32.91 B params, Q8_0 = 35.96 GiB; Q4_K_XL ~20 GB). Recommended sampling: thinking temp 0.6 / top_p 0.95; instruct temp 0.2. Ships MTP plus dedicated DFlash and DSpark drafts (the DSpark draft is NVFP4-quantized — nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark — Blackwell format, not a Maxwell path; flag). CUDA already fuses its TOPK_MOE routing (PR #31390 body). Not in tensor split mode (multi-gpu.md not-supported list) -> `-sm layer`.
  - Note: the "Nemotron-3-Nano-30B-A3B" (128 experts, top-6) that appears in PR #31390 is a different model — don't conflate.

### 3.5 ubatch guidance for MoE prefill (state of the art)

No doc states an official MoE ubatch rule; what exists: (a) the in-the-wild 262K-ctx MoE command uses `-b 256 -ub 128` (issue #31675); (b) ROCm prefill work on Qwen3.6-35B-A3B benchmarks `-b 4096 -ub 512,1024,2048` (PR #28303, which notes its F32 CONCAT transpose shape "occurs during Qwen3.6-35B-A3B prefill"); (c) the 12 GB guide found large `-b 4096` to be a negative optimization under VRAM offload; (d) PR #25294 (open) removes the n_ubatch clamp that previously limited prefill when experts don't fit the cache. Practical read for M10: small-ish ubatch for prefill under VRAM pressure (128-1024 range), verify empirically.

---

## 4. Speculative decoding in current mainline (llama.cpp master, fetched docs/speculative.md + README this round)

### 4.1 Flag reference (exact, from docs/speculative.md flag table)

- `--spec-type` (env LLAMA_ARG_SPEC_TYPE): `none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod`, comma-separated, default `none`. `--spec-default` = "enables ngram-mod".
- **`--spec-draft-n-max N` — "number of tokens to draft (default: 3)". This is the flag; the old `--draft-max` name no longer appears anywhere in the master flag reference** (answer to your exact-flags question: as of Nov 2026 mainline it is `--spec-draft-n-max`).
- `--spec-draft-n-min N` (default 0); `--spec-draft-n-ubatch N` — "physical max batch size of the draft context, use a smaller value to save VRAM (default: 0 = same as --n-ubatch)".
- `--spec-draft-p-split, --draft-p-split P` (default 0.10); `--spec-draft-p-min, --draft-p-min P` (default 0.00).
- `--spec-draft-model, -md, --model-draft FNAME`; `--spec-draft-hf, -hfd, -hfrd`.
- **`--spec-draft-device, -devd, --device-draft` — put the draft model on a different device.** For your rig: draft on a spare M10 die.
- `--spec-draft-ngl, -ngld`; `--spec-draft-type-k/-ctkd`, `--spec-draft-type-v/-ctvd`; **`--spec-draft-override-tensor, -otd`** (per-tensor placement of draft tensors); **`--spec-draft-cpu-moe, -cmoed`** and **`--spec-draft-n-cpu-moe, -ncmoed, N`** (draft's expert offload, same semantics as the main-model flags).
- Draft CPU scheduling: `-td` (threads), `-tbd`, `-Cd`, `-Crd`, `--cpu-strict-draft`, `--prio-draft`, `--poll-draft`.
- ngram-mod defaults: n-match 24, n-min 48, n-max 64; doc note: "MoEs require long drafts" (hence ngram-mod being the `--spec-default`).
- Synthetic acceptance bench: `--spec-synth-rates` / `--spec-synth-len`.

### 4.2 Qwen3.6 MTP GGUFs (draft-mtp)

- `--spec-type draft-mtp` = "Uses Multi Token Prediction (MTP) heads from the main model" — no `-md` needed; the MTP heads must be present in the GGUF.
- Qwen3.6-35B-A3B MTP-preserved GGUFs exist: SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF (fetched this round; base llmfan46/...-Native-MTP-Preserved) — "All 20 MTP layers preserved. Use `--spec-type draft-mtp --spec-draft-n-max 2` for 20-50% speed boost." Usage: `llama-server -m ...APEX-I-Compact.gguf -ngl 99 -ncmoe 16 -c 131072 --spec-type draft-mtp --spec-draft-n-max 2`.
- APEX quant tiers for that model (mudler/apex-quant, MoE-aware mixed precision; DGX Spark GB10 numbers, their table): F16 64.6 GB / Q8_0 34.4 GB / APEX I-Quality 21.3 GB (PPL 6.552, 63.1 t/s) / I-Balanced 23.6 GB (61.4) / I-Compact 16.1 GB (69.8 t/s, PPL 6.669) / Mini 12.2 GB (74.4). "APEX beats Q8_0 perplexity at half the size."
- Conversion: `convert_hf_to_gguf.py --mtp` — PR #31611 (merged 2026-11-05) fixed exactly this path for NVIDIA Nemotron-3.5-Lightning NVFP4 ("Converting Nemotron 3.5 Lightning NVFP4 with --mtp now produces general.tensor_extra.name = ['output.weight'] / prec_a4 = [False]").
- Maxwell caution (PR #27210, open, fetched body this round, adaptive MTP depth): "it can make performance worse on older hardware that cannot run the deep drafting and acceptance steps fast enough — stick with regular MTP with a low fixed depth," and "For MoE models it's generally preferable to lower the maximum depth to no more than 8." Their table (2x R9700, Qwen3.8-27B Q8_0): baseline 30 t/s; fixed-3 MTP: reasoning 51.68 / prose 55.89 / code 69.28; adaptive 3..12: prose 56.64 / code 78.03; ngram-mod: recall 292.88.
- vLLM-side corroboration [SNIPPET]: NVIDIA forum thread "Benchmark Report: Qwen3.6-35B-A3B-NVFP4 on NVIDIA DGX Spark, Jetson Thor, Blackwell 6000 Pro" uses "Marlin MoE backend, and MTP speculative decoding" (forums.developer.nvidia.com/t/.../371810 — search snippet only).

### 4.3 Separate small draft model on old Maxwell

- Canonical form: `--spec-type draft-simple -md draft.gguf --spec-draft-n-max N` (EAGLE-3 / DFlash / DSpark variants only if the draft was trained as such; DFlash example: `--spec-type draft-dflash --spec-draft-n-max 15 -fa on --jinja`, "clamped to the draft model's trained block size"; DSpark: `--spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min P`).
- For the M10 specifically, the interesting knobs are `--spec-draft-device` (draft on another die), `--spec-draft-n-cpu-moe` (draft's experts on CPU), `--spec-draft-n-ubatch` (draft VRAM), `-otd` (pin draft attention tensors), and `--spec-draft-p-min` (acceptance threshold — the docs' p-min example uses 0.5).
- EAGLE-3 support list (from the doc table) includes AngelSlim/Qwen3-a3B_eagle3 — i.e. a Qwen3 A3B EAGLE-3 draft exists for the A3B family.
- Draft KV: `--spec-draft-type-k/-v` can be quantized (q8_0) to save the die's 8 GB.

---

## 5. Source index

Fetched in full this round (round 4):
- https://github.com/ikawrakow/ik_llama.cpp/pull/2580 (+ .diff)
- https://github.com/BitMakerMan/Strata-maxwell (+ bench/results/2026-10-25-m10-vs-standard/README.md)
- https://github.com/timoruohomaki/theworkstation
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/development/token_generation_performance_tips.md
- https://raw.githubusercontent.com/shiqikuangsan31/Qwen3.6-35B-12GB-VRAM-Guide/main/README.md
- https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF
- https://unsloth.ai/docs/models/nemotron-3.5
- api.github.com issue searches: cpu-moe (1158 hits), MTP, "n-cpu-moe", "Qwen3.6-35B-A3B", "Nemotron-3.5-Lightning", "moe-cache", ubatch+MoE+prefill (bodies of: PRs #29343, #20596, #31775, #25294, #27210, #31611, #28303, #31390; issues #31675, #31510, #31262, #30163, #23855, #31115)

From round 2/3 (cited where noted): 18ocu6q RSS (GM107 1.672 TFLOPS / 83.2 GB/s), 1xnl82r full thread (m10dis architecture), shinbunbun/llama-cpp-p100-patches docs + deepwiki sm_60 page.

[SNIPPET]-only items: NVIDIA DGX Spark/Thor Qwen3.6-35B-A3B-NVFP4 benchmark thread (search snippet).
Known gaps: no measured M10/M40/P40/1080 Ti A3B numbers found; upstream "Strata" repo URL not found; zhihu/CSDN still blocked (unchanged from round 3).
