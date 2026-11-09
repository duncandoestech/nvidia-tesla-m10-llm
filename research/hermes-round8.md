# M10 / Maxwell research — round 8

Rules: research only, one file. Quotes verbatim from pages fetched in full this round (URLs inline); [UNVERIFIED] = no fetchable primary source this round.

## 1. NCCL minimum compute capability, and NCCL on Maxwell (M10/M40/M60)

Authoritative source: NVIDIA/nccl `master`, `makefiles/common.mk` (https://raw.githubusercontent.com/NVIDIA/nccl/master/makefiles/common.mk, fetched today):

```
CUDA8_GENCODE = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61
# "Prior to SM75 is deprecated from CUDA13.0 onwards" → CUDA 13 builds: sm_75/80/90/100/110/120 only
```

- The oldest architecture current NCCL builds for is **sm_60 (Pascal, CC 6.0)** (with CUDA 11.8–12.7 toolkits: sm_60/61/75/80/90(+100,120 on 12.8+)). With CUDA 13.x toolkits the floor rises to **sm_75 (Turing, CC 7.5)**.
- Current release: **v2.32.3-1, published 2026-09-17** (https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1); release notes contain no Maxwell/Pascal/Kepler content; the current user guide (2.32.3) has no "supported GPUs" page (candidate URLs 404).

Verdicts:
- **M10 (sm_50) and M40 (sm_52): no current NCCL release ships code for them.** Official binaries only cover CC ≥ 6.0 (CUDA ≤12.x builds) or ≥ 7.5 (CUDA 13+ builds). A from-source build targeting sm_50 with CUDA 12.x is technically possible but unsupported/undocumented by NVIDIA — treat NCCL as unavailable on Maxwell.
- **M60 (Kepler sm_30): never** — below even the historical CC 3.5 floor of early NCCL 2.x.
- History: early NCCL 2.x (2016–2019 era) supported CC 3.5+ (Kepler/Maxwell); the exact release that dropped Maxwell is [UNVERIFIED] (archived user-guide "Supported GPUs" page 404s, release notes searched make no mention).
- Practical consequence (from round 7): ik_llama.cpp's `-sm graph` explicitly works **without** NCCL (build auto-detect, runtime fallback; "For two GPUs, the difference between NCCL and no NCCL is basically negligible" — PR #1080), and mb8565's 2x M10 `-sm graph` runs were measured without NVLink/P2P (#2740). A 4-die M10 should plan on the non-NCCL path.

## 2. Practical cooling for a passive Tesla M10 in a desktop tower

Nature of the card: "Tesla M10s are passive server cards and need forced air through their fins." (crgregersen/m10dis, `docs/HARDWARE.md`, fetched in full: https://raw.githubusercontent.com/crgregersen/m10dis/main/docs/HARDWARE.md)

3D-printed hardware (all fetched):
- **Thingiverse thing:6791537** "NVIDIA Tesla M10 fan duct" (KungfuPancake): https://www.thingiverse.com/thing:6791537 — "air duct for a NVIDIA Tesla M10 card … to direct the air flow of two 40mm fans through the passive cooler. The flange has a lip with a very tight spacing so it sits snugly in the fan shroud. The three M3 screws are sufficient … no supports are necessary." BOM (via the MakerWorld remix): 2× Delta FFB0412VHN 40 mm fans, 8× M3 heat-set inserts, 3× M3x14, 8× M3x30.
  - **MakerWorld 2839079** (Jgl1740, released 2026-05-23, CC BY-SA; https://makerworld.com/en/models/2839079-nvidia-tesla-m10-fan-duct) — remix that "corrected a slicing error in the original model that lead to an unwanted opening." Contains the honest caveat: "a) this card needs a fair amount of cooling and b) **consumer 40mm fans generally don't move that much air**."
- **Thingiverse thing:7005879** "Nvidia K80, M60, M10, P40, P100 Tesla GPU fan shroud & external riser": https://www.thingiverse.com/thing:7005879 — blower-style: "Model: **7530 Blower Fan 75x75x30mm, 12V** … The total length of Fan+GPU is ~400mm. The riser frame can be printed on a 250x250 bed by putting it diagonally."
- **Cults3D** "Side-mounted cooling fan duct for Nvidia Tesla M10" (Tomblarom): https://cults3d.com/en/3d-model/gadget/side-mounted-cooling-fan-duct-for-nvidia-tesla-m10 — side mount, "Cooling Fan: J50GH-A00 PVB120G12H (Dell OptiPlex 390/790/990)", 3× M3x8mm + 1× M4x5mm.
- Commercial kits: eBay 155573881561 (M10/M60 blower kit) and 313527520178 (K20…P100 kit).

The one fully-published production M10 cooling setup (crgregersen/m10dis — 6× M10 + RTX 3070 Ti on an ASRock B250M Pro4 / i5-6400, six shroud fan banks):
- "Six fan banks, one per card, all follow the **hottest** M10, whichever card it is: 2 banks on the motherboard's Nuvoton NCT6683 (driver: the patched `nct6687` in `fan/drivers/nct6687/`; **never `pwm2`, which is the CPU fan**), 4 banks on an ARCTIC USB fan controller (driver: `fan/drivers/arctic-fan-controller/`)."
- "`fan/m10-fancontrol.conf` drives all six with the same curve, **PWM 20 up to 55 °C rising to PWM 200 at 70 °C**." The systemd unit sets all fans full when it stops; `m10-fan-emergency`/`m10-fan-full` scripts; probe-mapping and idle-calibration scripts in `fan/`.
- Clock context (same file): "Clocks | capped by the application-clocks setting at **1137–1189 of 1202 MHz SM**, memory 2600 MHz; parked workers idle at 135/405 MHz." `docs/LESSONS.md` (fetched): "Parked workers drop to 135/405 MHz, so every timing run needs a warm-up and a clock check first."

Reported temperatures under sustained load: no explicit °C measurement of a loaded M10 is published in any source I could fetch. The only published sustained-load thermal design point is the m10dis curve above — i.e. keep the hottest die **≤ ~70 °C** with clocks already capped below the 1202 MHz max (1137–1189). Anything hotter is uncharted in public data.

GM107 thermal throttle threshold: GM107 parts (e.g. GTX 750 Ti) are commonly cited at **98 °C max GPU temperature** (the point where the clock is reduced). I could **not** verify a primary source this round — NVIDIA's GeForce spec pages now 404 (both URL patterns tried) and TechPowerUp is antibot-blocked — so treat 98 °C as [UNVERIFIED]. Sourced anchor points: the 70 °C fan-curve target and the sub-max clock caps above.

Reliability note (not thermal, but relevant to sustained passive cooling): "A faulting CUDA op can wedge an M10 die (2026-10-25 incident: **Xid 31 → 8× Xid 158**, machine died hard; ~40 earlier Xid 31s had all recovered cleanly, so the escalation is possible but not the rule). Recovery that worked: full host power-off." (Strata-maxwell `WORKLOG.md`.)

## 3. llama.cpp Vulkan on Maxwell in 2026 (incl. 580 driver)

Works on sm_50 — direct evidence:
- **Issue #18743** (theIvanR, 2026-01-10, closed not_planned 2026-02-25; https://github.com/ggml-org/llama.cpp/issues/18743) — llama.cpp 7651 (ccbc84a53), Windows, CUDA 11.4 toolchain, 3× Tesla K40c (CC 3.5) + Quadro K620 (CC 5.0): "Selecting the Maxwell device (CC 5.0) works fine." and "**Vulkan and CPU backends work fine on all devices.**" (The Kepler-side CUDA crash in that issue — "CUDA error: no kernel image is available for execution on the device" on SCALE — was the user's build, not a backend defect.)

On the 580 driver specifically:
- **Issue #20762** (zkrx, 2026-03-19, closed 2026-05-20 as not_planned/stale; https://github.com/ggml-org/llama.cpp/issues/20762) — llama.cpp 8407 (bcdc4eb), Arch Linux headless, **nvidia-580xx-dkms 580.142**, 2× GTX 1070 + **GTX TITAN X (Maxwell GM20, 12 GB)**:
  - Device line: "NVIDIA GeForce GTX TITAN X (NVIDIA) | uma: 0 | **fp16: 0 | bf16: 0** | warp size: 32 | shared memory: 49152 | **int dot: 0** | matrix cores: none" — i.e. the backend runs f32 kernels on Maxwell (no fp16/int-dot path).
  - It runs long headless sessions (Qwen3-Coder-Next Q4_K_M, ctx 150k, `--no-mmap`, "30 minutes – hours"), then: "dmesg shows an Xid 13 error on the same GPU (nvidia GTX Titan X). This time **after 61,702 token**." dmesg: "NVRM: Xid (PCI:0000:04:00): 13, pid=1458, name=llama-server, **Graphics Exception**: channel 0x0000000b". Crash: "terminate called after throwing an instance of 'vk::DeviceLostError' … vk::Device::waitForFences: ErrorDeviceLost".
  - "I limited the power to all GPUs to around 100W and **could still reproduce**. … gpu_burn on all 3 cards at the same time, at full power, for 15 minutes with no error."
  - Closed by the stale bot, **no fix landed**. → Vulkan + 580 on Maxwell-class works, but there is an open, unaddressed long-run stability failure on Maxwell; plan for watchdog/restart or keep the CUDA path for unattended serving.
- General NVIDIA-driver flakiness on the Vulkan path: **#10528** (open since 2024-11-26, 83 comments): inconsistent SIGSEGV inside `libnvidia-eglcore.so.535.183.01` in the "[vkrt] Analysis" thread.
- Vulkan GEMM work that also applies to Maxwell: **#16056** (merged 2025-09-20) "vulkan: use vec dot for matrix matrix multiplications" — vec2 dot accumulations, shared-memory/register caching.

Measured Vulkan vs CUDA numbers on Maxwell (M40, GTX 970/980, 750 Ti): **none found.** Exhaustive this round: GitHub issue searches `vulkan M40` (1 hit — #10528), `vulkan "GTX 980"` (unrelated), `vulkan maxwell` (10 hits, no Maxwell throughput), plus web search. #18743 has no t/s at all. There is no public Vulkan-vs-CUDA comparison on any Maxwell card in the llama.cpp context — a measurement would be new data, not a confirmation of existing numbers.

## 4. Other forks/projects tuning MMVQ/GEMV for Maxwell or Pascal (prior art)

1. **Niko1221/Strata** — https://github.com/Niko1221/Strata (C++, MIT; created 2026-09-24; at fetch: 41,787 stars / 4,580 forks / 1,157 open issues). "Qwen3.8-Flash-Next on any consumer hardware: one-click install … Strata inference engine, OpenAI/Anthropic API on localhost." **Not a llama.cpp fork** — its own engine; official support list is RTX 20/30/40/50, AMD, Arc, Apple, Strix Halo, DGX Spark (no Maxwell). But it does ship Pascal/Maxwell-adjacent kernel tuning, documented in `docs/COMMUNITY_BENCHMARKS.md` (fetched):
   - 2× **P40 (sm_61)**, Xeon E5-2699 v3, 125 GB RAM, Strata 0.1.41 CUDA 12 source, IQ3_S, 131,072 ctx, layer split: "**decode 35.0 tok/s, prompt 350/558 tok/s** at 3.9K/21.9K, needles 6/6".
   - P40 (sm_61), Strata `a0116b2` (0.1.43.1) CUDA 12.4, IQ3_S, 262,144 ctx: "**the profile of a prompt and of a decode window on Pascal and four bitwise changes for sm_61: a 565-token prompt 205 -> 255 tok/s, decode 34-35 -> 37-38 tok/s, and with the opt-in `STRATA_DENSE_F32=1` a 7K prompt 360-432 -> 464-474 tok/s**".
   - V100 (sm_70): opt-in `STRATA_SM70_TABLE=1` — "decode **+9.8% and +15.0%** … (28.16/28.64 → 30.92/32.94 tok/s), prompt throughput unchanged, and the generated tokens bit-identical".
2. **nuke1984/Strata-maxwell** — https://github.com/nuke1984/Strata-maxwell (fork of Strata; created 2026-10-18; branch `maxwell-0.2.6`, tag `maxwell-0.2.6-port-production-OK`; further fork BitMakerMan/Strata-maxwell, pushed 2026-11-08). "RTX 3060 12 GB + Tesla M10", serving Qwen3.8-Flash-Next MoE (24,576 routed experts, ~33.7 GB quantized): 1,874-expert cache (2.53 GiB) on the 3060; the other **22,702 experts (92%) resident in the M10's 30.5 GB** (4× ~7.6 GiB usable). Full 256K context, fully-resident A/B vs page-locked CPU RAM, equal expert split: prefill 144.5–154.0 (2.7k) / 154.3 (10k) vs 430.3/434.9 and 275.3; **decode 69.5/70.1 vs 35.9/52.5** → "M10 wins decode ~1.3× warm (70.1 vs 52.5); RAM wins prefill ~2.8×". Kernel-level sm_50 facts from `WORKLOG.md` (fetched in full): "sm_50 (GM107) has no dp4a/fp16-math/tensor cores: **our dequant-dot runs on 4×`vmad` PTX, measured within 1.07× of the best alternative primitive (326.8 vs 350.7 G MAC/s)**"; "the gup is instruction-bound at 27× DRAM headroom — that is the silicon's price for 2-bit dequant, not a tuning gap"; prefill ladder **75.8 → 157.9 tok/s (+108%)** via six `STRATA_MW_*` knobs (MT gup E=4, MT down, TWOPHASE, MT_T=256x4, AD2H, FASTCOMPLETE); generation 62.5 → ~72 (+15%) (`--spec 6`, MT=1, MT_T=256x4); "NEVER enable `STRATA_MW_DRED`" (the `--maxwell-split tensor` regime produced real illegal-memory-access crashes; cross-device peer copies crashed historically, the two-hop form verified clean over 250 rounds); "Decode beyond ~72: measured at its floor for this rig's serial verification loop (waitCPU ~14.3 ms/window)"; "In October 2026 the Tesla M10 sells for about **$100-120 on the backmarket** and carries 32 GB of GDDR5." MIT.
3. **crgregersen/m10dis** — https://github.com/crgregersen/m10dis (created 2026-11-04, MIT, 36 stars, 5 forks, CUDA). "Qwen3.8-Flash-Next-FP8 (48-layer MoE, 512 experts) on one RTX 3070 Ti plus **24 Maxwell GPUs from six Tesla M10 cards**. PCIe multicast, GPU page-table remapping, 96 tok/s decode." README results table (fetched in full): **Decode 52.5 tok/s (single stream); Prefill ≈ 277 tok/s** (pipelined batches of 32 rows, two in flight); context 94,631 KV tokens; **MMLU-Redux 5-shot 90.98 ± 0.56** (official base 90.68); **LiveCodeBench v6 87.5 %** (official 91.9); "MoE layer, decode ≈ 150 µs per token per layer (publish 29 + round trip 132 + combine 9 µs)". Each of 20 routed GM107 holds 32 of every expert's 640 rows (FP8, ≈ 6 GB per GPU); hidden states **multicast** via each M10's on-board PEX8747; card-leader GPUs sum partials over on-card P2P and write back via **remapped UVM page-table entries**; one PLX PEX8748 switch, all links Gen3 x4 ≈ 3.9 GB/s. "Tested with Linux 7.0.14, **NVIDIA driver 580.178.04, CUDA 12.4 (sm_50 for the M10s)**, vLLM 0.32.0, PyTorch 2.13 + cu132, Python 3.14." `docs/LESSONS.md` kernel timings: gate/up 174 µs → 39 µs after restructure (v2), barriers ≈ 13 µs, polling 4 µs, shared expert 303 µs → 76 µs, slowest worker 146 µs → 103 µs with card-leader sums (RTX-side sum of 24 partials was 0.83×); "The first progress window lies. A decode rate taken from the first 10-second window … showed 25 tok/s against a steady 52.8."
4. **Mainline llama.cpp (not a fork, for the credit list)**: PR **#25680** (theIvanR, opened 2026-07-14, closed unmerged 2026-07-15) "Enhance ggml_cuda_mul_mat_cublas with compute type fallback" — "Adds hardware capability checks for CUDA compute type selection to prevent crashes on older NVIDIA architectures when using MTP models … **Tested with Qwen3.6 35B-A3B Q4XL on Tesla K40 (Kepler): Stock: ~22 tok/s, MTP disabled: ~17.5 tok/s, MTP enabled (2 draft tokens): ~23.5 tok/s**." Plus the R6-credited Pascal items: #20905 (MoE GEMV rewrite, merged), #30334 (Pascal gap, open), #25479 (unmerged), #11397 (`-ot`).
5. **ik_llama.cpp mb8565 M10 PRs** (credited in R3–R7, unchanged): #2580 (tile_f32 for sm_50-class, M10-verified), #2701, #2721 (sm_50 spill fix), #2740 (2-warp MMVQ, M10), #2726, #2860.

Negative result: no other public llama.cpp fork specifically tuning `mul_mat_vec_q`/GEMV for Maxwell or Pascal surfaced this round (GitHub repo + issue searches; the only Maxwell/Pascal-specific GEMV-class work found is in the Strata family above, which is a separate engine).

## Gaps (honest list)
- Exact NCCL release that dropped Maxwell: [UNVERIFIED] (archived docs page gone; release notes silent).
- GM107 98 °C throttle point: [UNVERIFIED] this round (NVIDIA pages 404, TechPowerUp antibot); sourced anchors are the ≤70 °C fan-curve design point and the 1137–1189/1202 MHz clock caps.
- No explicit sustained-load °C for a loaded M10 exists in public sources I could fetch.
- No measured Vulkan-vs-CUDA numbers on any Maxwell card (llama.cpp): none public.
