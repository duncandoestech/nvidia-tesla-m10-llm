# Tesla M10: research notes

Collected 2026-11-07. Sources are listed at the bottom. Each point says whether it is **published** (someone else reported it), **spec** (vendor or chip data), **source** (read in the llama.cpp code) or **inferred** (my reasoning, not yet measured).

## What the card is

- **Spec:** the M10 is one board with **four separate GM107 Maxwell GPUs**. Each GPU has 8 GB GDDR5 on a 128-bit bus, about 83 GB/s, and 640 CUDA cores at a 1033 MHz base / 1306 MHz boost clock. An on-board PCIe switch connects the four. The card is passively cooled and made for VDI (virtual desktops).
- **Measured on the test box:** the four GPUs show up as `04:00.0`–`07:00.0` behind a PCIe switch on root port `00:02.2`. The upstream link is **x8** (max x16), the same lane-split situation the P100s had. Each GPU's power cap is 53 W (min 26.5 W), compute capability 5.0, and driver 580.178.04.
- **Inferred:** the card is not a 32 GB GPU. It is four 8 GB GPUs, each with about a ninth of a P100's bandwidth (83 vs 732 GB/s). Token generation is limited by memory bandwidth, so one GM107 can at most read a 1.2 GB model about 69 times per second. That is the hard ceiling for single-GPU token generation on a ~1.2 GB quant. Real kernels usually reach 60–75 % of it.

## Software support (the biggest trap)

- **Published:** CUDA 13 dropped Maxwell (sm_50/52). A llama.cpp build made with CUDA 13 compiles without complaint, then fails at runtime with "no kernel image". **Use CUDA 12.x** (the test box has 12.9, used here inside an Ubuntu 22.04 CUDA 12.9 container). Driver branch 580 is the last one that supports Maxwell. Do not upgrade the driver past it while the M10 is in the box.
- **Published (ollama #9953):** since ollama 0.5.13, its CUDA 12 build includes sm_50, so ollama will spread a model onto an M10 next to faster cards and slow everything down. There is no runtime switch: pin the devices with `CUDA_VISIBLE_DEVICES` or run separate servers.
- **Published (llama.cpp PR #28389, merged 2026-09-19, commit b23701f77):** on sm_50 and sm_52, the CUB radix-sort argsort path corrupted memory when it sorted in place ("illegal memory access", Xid 31). That path only runs when argsort/top-k has more than 1024 columns, which means GPU-side top-k sampling over a big vocab, as used by DFlash speculative decoding. The reporter ran **4x Tesla M10 with Qwen3.8-27B Q4_K_M, `-sm layer -ts 1,1,1,1 -fa on -ctk q4_0 -ctv q4_0`**, the only other M10 multi-GPU LLM setup found anywhere. **Any build used here must include that commit.**
- **Source/published (ik_llama.cpp docs, to be checked in mainline):** the quantized matmul kernels (MMQ) and fast dot products need `dp4a`, which arrived in compute capability 6.1. Maxwell (5.0) has no dp4a and no fast FP16 math, so llama.cpp falls back to dequantize + FP32 cuBLAS for prompt processing, and to a software dp4a for generation. Flash attention support on old cards depends on head size and KV type, and unsupported combinations fall back to the CPU.

## Published numbers from the same architecture (Maxwell)

No one has published M10 tok/s anywhere I could reach (Reddit was not fetchable; GitHub, ServeTheHome and the llama.cpp discussions were searched). The closest data:

| GPU (chip, bandwidth) | Backend | Model | pp512 t/s | tg128 t/s | Source |
|---|---|---|---|---|---|
| Tesla M40 (GM200, 288 GB/s) | CUDA, fa 0 | Llama-2-7B Q4_0 | 282.7 | 38.0 | llama.cpp #15013 |
| Tesla M40 | CUDA, fa 1 | Llama-2-7B Q4_0 | 290.2 | 40.0 | #15013 |
| Tesla M40 | Vulkan, fa 1 | Llama-2-7B Q4_0 | 93.4 | 41.7 | #10879 |
| GTX 980 (GM204, 224 GB/s) | Vulkan, fa 0 | Llama-2-7B Q4_0 | 186.2 | 33.9 | #10879 |
| GTX 970 (GM204) | CUDA, fa 1 | Llama-2-7B Q3_K_S | 191.3 | 14.8 | #15013 |

### Reddit reports (found by Hermes, 2026-11-07, from search snippets only)

Reddit blocked full-page reads from both hosts, so these quotes come from search-engine snippets. Threads not opened, numbers **unverified**:

- **M10 + ollama, 4 GPUs** (r/ollama 1fjn7e8): Llama 3.1 q8 went from "0.6 tokens/s to over 6 tokens/s" vs 2x Xeon 2673v4. A commenter on a Dell R630 reports "with llama 3.1 8b I have 9tk/s" (quant not stated). FP16 models "don't work on this card".
- **M10 in an R740** (r/ollama 1k8jubs): llama.cpp built from source, "M10 performs worse than the cpus" (dual Xeon Gold 6126). Single-GPU env vars and NUMA tweaks did not help.
- **M10 verdict thread** (r/LocalLLaMA 18ocu6q): "a p40 will be at least 4 times faster".
- **4x M40 12GB** (r/LocalLLaMA 1suii0z): Qwen3.6-27B "fp4: 26tok/s PP 8tok/s TG".
- **2x M40 + Quadro RTX 4000 draft** (r/LocalLLaMA 1sitlcj): Qwen3-30B-A3B Q4 with a Qwen3-0.6B draft model, "around a 70% acceptance rate and 35t/s". Speculative decoding on a faster card helps the slow Maxwell cards.
- **M40 ollama sweep** (est.tech "LLM drag racing", page read directly): gemma3:4b 32.1, qwen3.5:9b 20.5, qwen3.5:27b 5.6 t/s. The M40 "hits its 85 °C thermal ceiling", and they gate runs at ≤60 °C.

### Full threads and two M10 projects (Hermes, via Reddit RSS; repos verified by reading them 2026-11-08)

- r/ollama 1fjn7e8, full thread: one GPU q6_K ~5 t/s, q5_K_M 6 t/s; R630 Llama-3.1-8B 9 t/s, Gemma2-9B 6 t/s; a 70B across all 4 GPUs 0.5 t/s ("unusable").
- **github.com/crgregersen/m10dis** (MIT): 6x M10 (24 GM107) + RTX 3070 Ti behind a PLX PEX8748 switch, custom stack, not llama.cpp. The RTX runs attention and the router (vLLM). The M10s hold FP8 MoE experts and the LM head, running custom persistent kernels. Qwen3.8-Flash-Next FP8: **52.5 t/s decode, ~277 t/s prefill**, up to 96 t/s with MTP speculative decoding on code. Driver 580.178.04, CUDA 12.4 sm_50 (same as ours). Maxwell findings stated in their RESULTS.md:
  - expert gate/up GEMV reaches **≈ 60 GB/s of the 83 GB/s peak**
  - best variant: "**2 rows per warp**, transposed down layout, up to 40 warps per SM"
  - unrolling the lane loop made it *slower* (62 → 81 µs)
  - **This independently matches our Phase 2 result** (1 warp, several rows per block), and their 72 % of bandwidth vs our 47 % shows the headroom left.
- **github.com/garydavenport73/the-mighty-m10** (MIT): 3x M10, ollama wrapper scripts only, no kernel work. It caps each GPU at **30 W** (`nvidia-smi -pl 30`) and spreads layers over all 12 GPUs (`OLLAMA_SCHED_SPREAD=1`). Generation t/s: deepseek-coder-v2-16b 12.4, gpt-oss-20b 8.1, qwen3.5-35b 6.5, gpt-oss-120b 5.8, llama3.3-70b 1.2.

### Round 3 (Hermes with firecrawl, 2026-11-08; full report `hermes-round3.md`)

- **shinbunbun/llama-cpp-p100-patches** (also the "29-patches" build used for the P100s on the same box): 43 individually measured patches for P100, which also has no dp4a. Relevant here (read from the patch files):
  - **02 mmvq rows per block:** 4 rows/block for ncols=1 on pre-Turing (P100 9B Q4_0 tg: 1 → 46.9, 2 → 54.1, 4 → 57.7, 8 → 57.2 t/s), and 4 rows for the ncols 2–4 verify batch. It keeps 4 warps. **Our sweep adds that 1 warp is better on Maxwell** (4w/2r 22.1 vs 1w/2r 26.3 t/s).
  - **01 vmad dp4a:** replaces byte-extract emulation with PTX `vmad.s32.s32.s32 .bN` (+6.5 % on P100, gated `>= sm_60`). **Checked: on sm_50 it compiles to native `VMAD.S8.S8`**; the test loop drops from 42 to 30 SASS ops. Throughput on Maxwell is unmeasured. Added to the patch behind `-DM10_VMAD`.
  - 07 (MoE rows/block), 08 (mmid batch cap, env `LLAMA_MMVQ_MMID_MAX`), 09 (nwarps vs small K), 10 (q8_1 activation cache): to test in the MoE phase.
- ik_llama.cpp PR #2580 (FA tile routing for D=256) is reported to give +20–35 % prefill, with numbers attributed to the M10. Hermes is verifying the hardware line in round 4.
- llama.cpp PR #31787 "CUDA: fix Maxwell build" merged 2026-11-06. Our base `f477f795c` (2026-11-08) already includes it.
- Hermes reports a "strata" fork running 4x M10 + RTX 3060 as an MoE expert pool: 69.5–70.1 t/s decode with speculative decoding. Unverified, being checked in round 4.
- Chinese forums (zhihu, CSDN) blocked or empty. ServeTheHome and Level1Techs have no M10 LLM or P2P threads.
- Hermes errors to discount: it said GM107 has 16 SMs (it has 5), then "corrected" it to 160 cores per die (wrong: Maxwell SMMs have 128 cores, so 640 per GM107, 2560 per card).

### Round 4 (Hermes, 2026-11-08; full report `hermes-round4.md`)

- **ik_llama.cpp PR #2580 verified** (merged 2026-10-02, mb8565): routes head-size-256 FA prefill to `tile_f32` on pre-Volta. The PR body says "tested on ... a Tesla M10 (sm_50)", table captioned "one GM107 die". Falcon3-10B: f16 KV pp +21–35 %; q8_0 KV pp ~3x (q8_0 KV otherwise fell back to the CPU). Qwen3.5/3.6 use head size 256, so mainline's equivalent path should be measured on long prompts (todo).
- **BitMakerMan/Strata-maxwell** verified (MIT): Strata MoE engine fork, RTX 3060 + 4x M10. 22,702 of 24,576 routed experts resident in M10 VRAM (~5676 per die). Qwen3.8-Flash-Next IQ2_XS at 256K context: decode 69.5–70.1 t/s with speculative decoding; experts in CPU RAM instead: 35.9–52.5 t/s.
- Mainline MoE flags: `-ncmoe/--n-cpu-moe N`, `--cpu-moe`, `--moe-cache-mib`, `-ot`. Tensor split "not implemented for many MoE/hybrid architectures" per multi-gpu.md. Speculative flags: `--spec-type draft-simple|draft-mtp|...`, `-md`, `--spec-draft-n-max`, **`--spec-draft-device/-devd`** (draft on another GPU), `-ngld`, `-ctkd/-ctvd`.
- PR #27210 (adaptive MTP depth, open) warns deep drafting can be slower on older hardware.

### Round 5 (Hermes, 2026-11-08; full report `hermes-round5.md`): upstream mmvq work, so we don't duplicate it

- **An ik_llama.cpp contributor (mb8565) owns an M10** and tuned the same kernel in Oct 2026 (all merged in ik_llama.cpp, not in mainline):
  - **ik #2701:** 2 rows per block for one column on sm_5x/sm_60/sm_75. **M10 Llama-3.1-8B Q4_K_M 6.60 → 7.70 t/s (+16.8 %)**, P100 +12.1 %. Neutral or slower on sm_61/sm_70.
  - **ik #2721:** with >1 row, the *fused* gate/up mmvq kernel spilled per-row temps (`tmp_u[j][threadIdx.x]`) to local memory on sm_5x/sm_60 (34 of 66 kernels). Fixed with an unrolled compare.
  - **ik #2740:** 2 warps for Q2_K and Q3_K only. M10 Q3_K_M 5.80 → 6.38, Q2_K 8.45 → 9.41. "Q4_K, Q5_K, Q6_K, Q4_0 and Q8_0 were not faster" (measured on P40 and Titan V, *not* on the M10).
- **Our result goes further:** 1 warp for *all* types, plus 4 rows, gives **+57 % on Q4_K_M and +64 % on Q4_0** on GM107. The 1-warp gain for K-quants and Q4_0 was never tested on Maxwell upstream.
- Mainline: PR #20905 (MoE GEMV optimization, merged 2026-03), #27828 (P100 always MMVQ for MoE), #29753 (2 warps on sm_70), #25479 (2 warps on sm_61, unmerged draft). No Maxwell table exists in mainline.
- **Checked our builds (cuobjdump -res-usage):** stock and tuned mainline both have 299 mmvq kernels with no stack and **18 with a 16-byte stack, all `mul_mat_vec_q_moe`** (the MoE path). The single-token path we tuned does not spill. The MoE kernel is the next target.
- No public M10/Maxwell power-cap data and no measured acceptance rate for a Qwen3.5-0.8B draft. Both must be measured here.

### Round 6 (Hermes, 2026-11-08; full report `hermes-round6.md`)

- mb8565: 59 ik_llama.cpp PRs in 2026, none in mainline. M10 numbers appear only in #2701, #2721, #2740 and #2580. Corrected #2580 M10 numbers (Falcon3-10B, one GM107): f16 KV **pp2048 39.3 → 47.3 t/s (+20 %)**.
- **2x M10 with ik_llama.cpp `-sm graph`** (#2740, Llama-3.1-8B Q3_K_M): 10.25 → 11.16 t/s vs `-sm layer` 5.57 → 6.51. ik's graph split nearly doubles a dense model on two dies.
- **Mainline has no issue or PR about `mul_mat_vec_q_moe` spilling or rows/block on pre-Volta.** Our spill finding (18 MoE kernels with 16-byte stack on sm_50) is new. PR #20905 (MoE GEMV rewrite, 2026-03) has no Maxwell or Pascal data.
- Qwen3.5-0.8B / 2B / 4B share `vocab_size = 248320` and special tokens: the 0.8B is a valid llama.cpp draft.
- (round 8) **NCCL ships no code for sm_50/52**: official binaries cover CC ≥ 6.0 (CUDA 12) or ≥ 7.5 (CUDA 13). Mainline `-sm tensor` on the M10 therefore runs its non-NCCL fallback, and still gives 2.7x on the dense 27B.
- (round 8) Cooling: Cults3D "Side-mounted cooling fan duct for Nvidia Tesla M10" (https://cults3d.com/en/3d-model/gadget/side-mounted-cooling-fan-duct-for-nvidia-tesla-m10). m10dis fan curve: PWM 20 up to 55 °C, rising to PWM 200 at 70 °C. No published sustained-load M10 temperatures; the GM107 throttle point (~98 °C, commonly cited) is unverified. Ours peaked at 66 °C in a tower with case fans at 100 %.
- (round 8) Vulkan works on sm_50 with recent drivers, but no Maxwell Vulkan vs CUDA numbers are published.
- (round 8) Prior art for Maxwell/Pascal GEMV tuning: ik_llama.cpp mb8565 PRs (#2580, #2701, #2721, #2740), shinbunbun P100 patches, m10dis (custom kernels, "2 rows per warp"). No other llama.cpp fork found.
- (round 9) Mainline `GGML_CUDA_P2P` is opt-in by presence. m10dis needed a patched nvidia-uvm for GPU-to-GPU routes; no public M10 `nvidia-smi topo` or P2P bandwidth existed. **Measured here (2026-11-08):** `nvidia-smi topo -m` shows the four M10 dies as `PIX` to each other (one PCIe switch), but `nvidia-smi topo -p2p r` reports **`GNS` (GPU not supported) between every pair of M10 dies**, while the two P100s show `OK`. All M10 multi-GPU traffic goes through host memory.
- (round 9) Mainline disables CUDA graphs below Volta (`ggml_cuda_graph_set_enabled`: `cc < GGML_CUDA_CC_VOLTA` → `disable_due_to_gpu_arch`). Open PR #30727 reports ~25 % from enabling graphs on older GPUs. Being tested here as `M10_GRAPHS` (phase 12).
- (round 9) MTP: `--spec-draft-p-min` defaults to 0, and no published best value. PR #22673 notes MTP costs some prompt speed (D2H embedding transfers).
- (round 9) Gated DeltaNet: the scalar `gated_delta_net.cu` kernel runs on any CC; the MMA path is gated to Ampere and later. GDN was 1.4–2.8 % of generation time in our profiles, so not a bottleneck.
- Serving four dies: llama-server has no router mode, and llama-swap is a model swapper, not a load balancer. Use nginx `least_conn` or the LiteLLM proxy (simple-shuffle / least-busy) over four llama-server instances.

**What this says for us (inferred):** the "9 tk/s for an 8B" report fits the bandwidth math for one GM107 (83 GB/s ÷ ~4.9 GB Q4 ≈ 17 t/s ceiling, ~55 %). Nobody reports tuning beyond ollama defaults. Speculative decoding with a draft model on a faster card is the one trick reported to help Maxwell.

**Inferred scaling to one GM107:** the M40 has ~3.5x the bandwidth and ~4.8x the cores (3072 vs 640) of one M10 GPU. Llama-2-7B Q4_0 (3.56 GiB) would then do roughly 9–11 t/s generation and ~50–60 t/s prompt on one M10 GPU. Our run of the same model on one GPU is the check on this guess.

## Plan this suggests

1. Build mainline llama.cpp with CUDA 12.9, `CMAKE_CUDA_ARCHITECTURES=50`, including the argsort fix.
2. Measure the community reference (Llama-2-7B Q4_0, `-sm none`) so our numbers line up with the tables above.
3. Small Qwen first (Qwen3.5-2B), one GPU: sweep quant types. On Maxwell the dequant cost differs a lot between formats, so the smallest file is not automatically the fastest. Then sweep flash attention, batch / ubatch sizes, the KV cache type, and the `GGML_CUDA_FORCE_MMQ` / `FORCE_CUBLAS` build options.
4. Then the 4-GPU questions: layer split vs row split vs four independent copies (4 servers = 4x throughput for parallel users). Also P2P through the on-board switch.
5. Power and clocks: is it power-limited at 53 W? Do application clocks help? Is a passive card in this chassis thermally limited?
6. Optionally a Vulkan build for comparison (the published M40 data says Vulkan is ~3x slower on prompt and equal on generation).

## Sources

- llama.cpp CUDA scoreboard: https://github.com/ggml-org/llama.cpp/discussions/15013
- llama.cpp Vulkan scoreboard: https://github.com/ggml-org/llama.cpp/discussions/10879
- Vulkan vs CUDA thread (no Maxwell): https://github.com/ggml-org/llama.cpp/discussions/23109
- sm_50 argsort fix, 4x M10 reporter: https://github.com/ggml-org/llama.cpp/pull/28389
- ollama spreading onto M10: https://github.com/ollama/ollama/issues/9953
- ollama rig with M10 among other cards (no numbers): https://github.com/ollama/ollama/issues/13152
- CUDA 13 dropping Maxwell/Pascal: https://www.hardware-corner.net/local-llm-cuda-deprecation-p40-v100-20250507/ and https://localaimaster.com/blog/pascal-gpus-after-cuda-13
- ik_llama.cpp notes on pre-dp4a cards: https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md
- M10 general overview (low quality, specs only): https://retrivium.ai/journal/nvidia-tesla-m10-llm-rendering/
- Reddit threads (snippets via Hermes, unverified): https://www.reddit.com/r/ollama/comments/1fjn7e8 , https://www.reddit.com/r/ollama/comments/1k8jubs , https://www.reddit.com/r/LocalLLaMA/comments/18ocu6q , https://www.reddit.com/r/LocalLLaMA/comments/1ngdwnk , https://www.reddit.com/r/LocalLLaMA/comments/1suii0z , https://www.reddit.com/r/LocalLLaMA/comments/1sitlcj
- M40 ollama sweep: https://est.tech/news/llm-drag-racing-security-edition
- Not reachable: Reddit full pages (blocked for both Claude and Hermes), ServeTheHome search (empty).
