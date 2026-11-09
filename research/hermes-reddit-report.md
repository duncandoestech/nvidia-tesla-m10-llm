# Tesla M10 / Maxwell LLM Inference — First-Hand Reports (research v2, 2026-11-08)

Method note: full threads read via firecrawl web_extract (full post + comments, saved to disk).
One exception: r/LocalLLaMA/18ocu6q was blocked by firecrawl (antibot on .json, login wall on
old.reddit, all engines failed on www) — that thread's data came via the public .rss feed instead.

Context for the benchmark at hand: 4x GM107 Maxwell GPUs, 8 GB each (= one Tesla M10 card).
Per GM107: 83.2 GB/s memory bandwidth, ~1.672 TFLOPS FP32, CC 5.0, no FP16 (no 16-bit float),
CUDA 11 locked, last supporting driver 550.54.10, ~53 W max per chip, ~$80 per card.
GM107 is the same die as the GTX 750 Ti.

=====================================================================
1) r/ollama/1fjn7e8 — "NVIDIA TESLA M10" (u/Jarlsvanoid, ~2y ago)
   https://www.reddit.com/r/ollama/comments/1fjn7e8/nvidia_tesla_m10/
=====================================================================
- Software: ollama. M10 = "four GTX 750 Ti cards with 8GB each"; ollama unifies the memory
  and distributes the model across the 4 GPUs (32GB pool).
- OP (CPU baseline: 2x Xeon 2673v4):
    * Llama 3.1 q8: 0.6 tokens/s (CPU) -> "over 6 tokens/s" (M10)
    * Non-quantized (FP16) models do NOT work — no 16-bit float support
    * Nice side effect: each of the 4 GPUs can be assigned to its own VM
- R630 user (ollama + docker + openwebui, M10 "on top of" the chassis):
    * llama 3.1 8b: 9 tk/s
    * Gemma 2 9b: 6 tk/s
    * llama 3.1 70b across several of the M10 GPUs: 0.5 tk/s — "wasn't usable at all"
    * "For inference it's not that bad at all!"
    * Stated next step: Tesla T4 16GB (75 W TDP)
- Another commenter quoting their 8b result: "meta-llama-3.1-8b-instruct
  10.86 tok/sec, 165 tokens, 0.31s to first token" (comparing CPU vs this GPU speed)
- Takeaway: ~5-11 t/s for 7-9B quants is the realistic per-card band; 32GB pool only
  matters if you accept 0.5 t/s-class throughput for 70B.

=====================================================================
2) r/ollama/1k8jubs — "I installed a Tesla M10 into an r740" (u/dopey_se, ~2y ago)
   https://www.reddit.com/r/ollama/comments/1k8jubs/m10_performs_worse_than_the_cpus/
=====================================================================
- Hardware: Dell R740, dual Xeon Gold 6126, ~512GB RAM, PM3YD riser, 750W PSU (1 of 2
  connected), TR5TP cable too short so a PCIe 8-pin extension was used. Arch + Ubuntu;
  llama.cpp compiled from source on Ubuntu.
- Verbatim benchmark output (model: Anxioom/llama-3-8b-Instruct-Q6_K-GGUF):
    llama.cpp ON GPU (M10):
      prompt eval: 1028.66 ms / 17 tokens  = 16.53 tokens/s
      eval:        4358.45 ms / 20 runs    = 4.59 tokens/s (217.92 ms per token)
    llama.cpp CPU (dual Xeon Gold 6126):
      prompt eval: 414.58 ms / 17 tokens   = 41.01 tokens/s
      eval:        10234.78 ms / 61 runs   = 5.96 tokens/s (167.78 ms per token)
    -> CPU won BOTH prompt eval and generation on this model.
- ollama tinyllama, GPU:
      prompt eval rate: 92.29 tokens/s (40 tokens / 433.4 ms)
      eval rate:        37.78 tokens/s (202 tokens / 5.35 s)
  ollama tinyllama, CPU (CUDA_VISIBLE_DEVICES=-1):
      prompt eval rate: 237.41 tokens/s
      eval rate:        45.71 tokens/s (67 tokens / 1.47 s)
    -> CPU won again. "I've yet to find a model the M10 outperforms my CPU :)"
- Tuning tried without help: env vars to pin a single GPU, NUMA affinity to one CPU
  socket. GPU usage visible in nvidia-smi, just slow.
- OP also ordered a P100 (officially supported in the R740) and expects it to beat the CPUs.
- Comment (whateverworks325): "How about T10? I got one the other day and it works great,
  16G, performance comparable to 4060, and half the price of 4060. The downside is the
  passive cooling design; I need to add a fan, otherwise it will crash due to overheat."
- Caveat: CPU here is a dual Xeon Gold 6126 (24C/48T) with ~500GB RAM — a strong CPU
  baseline. A consumer CPU would likely close the gap less, but the direction (M10 slower
  than a decent CPU for 8B) is consistent with report 5's warning.

=====================================================================
3) r/LocalLLaMA/18ocu6q — "Couuld llama2 70b be run on a tesla m10?" (~3y ago, 16 comments)
   https://www.reddit.com/r/LocalLLaMA/comments/18ocu6q/couuld_llama2_70b_be_run_on_a_tesla_m10/
   [BLOCKED: firecrawl failed (antibot + login wall). Data via .rss feed.]
=====================================================================
- Per-chip numbers quoted in-thread: GM107 = 1.672 TFLOPS / 83.2 GB/s vs GP102 (GTX 970/980
  / M40 family) = 11.76 TFLOPS / 347.1 GB/s
- Even all 4 GM107s in parallel would not reach one P40; "a P40 will be at least 4 times
  faster, despite having 8 GB less"
- Community consensus: Maxwell CC 5.0 is aged out / unsupported by several AI tools
  (older runtimes, no FP16), though ollama does work on it.

=====================================================================
4) r/LocalLLM/1xnl82r — "Low cost $1500 AI rig — QWEN 3.8 Flash Next, 52.5 t/s decode in
   FP8 — 6xM10 = 192GB VRAM" (u/WiseAbalone4021, 4d ago)   << THE BIG ONE
   https://www.reddit.com/r/LocalLLM/comments/1xnl82r/low_cost_1500_ai_rig_qwen_38_flash_next_525_ts/
=====================================================================
- System: ASRock B250M Pro4, i5-6400 (4C @ 2.7GHz), 32GB DDR4, RTX 3070 Ti (attention /
  GDN / router), 6x Tesla M10 = 24x GM107 as "expert GPUs", PEX8748 PCIe switch with each
  M10 on Gen3 x4 (on-card PEX8747 multicasts to its 4 GPUs), 1TB NVMe, two PSUs 650W +
  1650W. M10s ~$80 each, $500 for six.
- Aggregate memory bandwidth: 4 x 6 x 83 = 1992 GB/s (≈ one RTX PRO 6000 Blackwell, which
  only has 96GB, for $10-20k).
- Model: Qwen 3.8 Flash Next (MoE: 512 routed experts + shared expert), 180GB, FP8.
  Hidden dim 2580 / 20 = 128, so expert work slices into 20 parts across 20 GM107s;
  5x4=20 GPUs compute 1/20 slice of the 8 routed experts (FP8); a 6th M10 runs the shared
  expert (FP16). Load balance worked out: 1/20*8/2=0.2 vs 1/4=0.25.
- Interconnect trick: PLX switches implement PCIe 2.1 multicast; GPU-P2P methods from
  Tinygrad (github.com/rioncarter/tinygrad.open-gpu-kernel-modules) proved GPU multicast
  works. Boot problem: 25 PCI devices exhaust the 64 KiB legacy I/O port space (4 KiB per
  PCI bridge) -> modified kernel holding the PCIe tree back; boot on PSU #1, suspend,
  power on PSU #2, wake; all GPUs initialized MMIO-only.
- 3-level result aggregation (thread credits "Solution (Claude wrote this)"):
    L1: each GPU sums its own experts into a 2560-value partial row
    L2: card leader sums its 4 cards over on-card P2P (PTE-remapped routes) + flags,
        fixed order for reproducibility
    L3: the 3070 Ti combines 5 routed leaders + gate * shared
- Attention quantized to FP8 (3070 Ti has only 8GB) -> just under 100k tokens of context.
  PLE + LM-Head also moved onto M10s.
- RESULTS (exact, from the post):
    Decode: 52.5 tok/s.
      MoE layer ~150 us = 29 us multicast publish + 132 us M10 round trip
      (gate/up 39 us, down 18-23 us, PCIe return + card sum the rest) + 9 us combine.
      Only 2.5 MB of weights read per token (~31 us at 83 GB/s) -> decode is
      LATENCY-BOUND, not bandwidth-bound. 48 layers x 150 us = 7.4 ms of a 19 ms token.
    Prefill: "~277 tok/s prefill" (pipelined batches of 32 rows, two in flight; chunk
      period 2.21 ms; per-chunk: host copy 0.21 ms, gate/up 0.67 ms, down 0.54 ms,
      finish/handshake 0.72 ms; 1024 rows: 72.9 ms pipelined / 98.4 ms batched /
      157.2 ms one-row-at-a-time). OP: "You are correct prefill is 'slowish'."
      M10s are the critical path; the handshake ("finish") is the biggest single phase.
- Benchmarks run: LCB6 and MMLU, "no noticeable degradation" (MMLU higher, LCB6 lower).
- Power: 700-1300 W total rig; each GM107 30-40 W under load, self-throttles (max 53 W).
- Code: github.com/crgregersen/m10dis — custom shard-and-combine stack, NOT llama.cpp /
  ollama / vLLM. OP: "not a project for people with fairly good experience with hardware
  and some low level understanding."
- Single-M10 fork: github.com/garydavenport73/the-mighty-m10 (commenter gr1mlok: "I was
  able to use a single M10 with a setup that uses a modified version of this guy's script")
- Forward plan in-thread: MTP targeting 100+ t/s; commenter (ketosoy): a beefier "trunk"
  card (e.g. V100 16GB) could take the shared expert or run pipeline parallel between trunks.

=====================================================================
5) r/LocalLLM/1wuinuf — "Tesla M10: A questionable card with a weird plan"
   (u/FizzyDuncDizzel, 1mo ago, 337 upvotes / 94 comments)
   https://www.reddit.com/r/LocalLLM/comments/1wuinuf/tesla_m10_a_dubious_card_with_a_weird_plan/
=====================================================================
- Plan: dedicate the M10 to Hermes sub-agents — one small model per 8GB GPU; the main
  larger model delegates to them. Bought lowballed on eBay (~40 euros). "I'll post results
  once I get it running."
- OP's own analysis: four separate 8GB GPUs on one board; GPUs talk over a slow PCIe link
  so memory doesn't stack well for one big model; Maxwell CC 5.0 aging out of support;
  card was built for VDI, not inference. "Four independent 8GB GPU is a much better fit
  [for sub-agents] than trying to treat the card as one 32GB pool."
- jjusko20: "dont do it bro. memory bandwidth on these things is 83 gbps (per gpu).
  You'll have 4 8gb cards that all decode more than 10 times slower than my v100s.
  This thing is going to be like decoding on a cpu. ... It isn't aging out of support,
  it's beyond aged out. It's locked to cuda 11."
- Force-Name (on M40s, same family problem): "I bought 2 m40$ and they are slow as all.
  You can shard your llm across it and if you don't care that you wait they work fine.
  I'm replacing mine soon for better compute time."
- mtbMo: "Running 2x m4000 8GB GPUs, serving an embedding model and 3B ministral.
  Slow as hell, but fine for background tasks, and faster than CPU."
- silent-curious-dev: passive cooling needs a good thermal solution; expect tinkering.
- Note: this is the same idea as the benchmark at hand — worth checking this thread's
  comment stream for OP's eventual results (posted 1mo ago; may have follow-ups).

=====================================================================
6) r/LocalLLaMA/1lybqtw — "Should I buy Tesla K80 for 70€ or Tesla M10 for 110€?"
   (u/Similar-Republic149, 1y ago, 24 comments)
   https://www.reddit.com/r/LocalLLaMA/comments/1lybqtw/should_i_buy_tesla_k80_for_70_or_tesla_m10_for_110/
=====================================================================
Community verdicts on M10/K80, with upgrade-path pricing (all quoted):
- offlinesir (Top 1% commenter): both cards fanless (need a cooling solution) and very
  old; "I would recomend against buying them"; that money -> 3060 12GB or API credits.
  "it was released in 2016, based on the Maxwell architecture, which was released in 2014.
  ... the tech behind the card is... 11 years old. It's just not supported anymore by some
  AI tools. VRAM is not everything, just because it's got two thirds the vram of the
  3060 12gb doesn't mean it's even half as good."
- EnvironmentalRow996: "P40 is a newer option. Or M40."
- OutlandishnessIll466: "They [P40s] used to be 150-200 euros. They work, I have 3, but
  not that fast and you need to spend another 30 euros for a fan. A RTX 3090 costs 2x the
  price of a p40 for 4x the performance."
- a_beautiful_rhind: "Better off getting an Mi50 than those ancient cards. Any cuda
  advantage for something that obsolete is long gone." Also: "His M10 has less b/w than
  my xeon. Bandwidth 83.20 GB/s"
- No-Refrigerator-1672: "M40 is still considerably faster than CPU; and if it were selling
  for 50 eur per piece, it would be a good deal. But yeah, as of today, Mi50 is definetly
  better bank/buck."
- AppearanceHeavy6724: "Tesla M10 is e-waste, a retro device, and not worth $10, let alone
  110 euro. If you want the cheapest possible 16 GiB buy two p104-100, $25-$40 each ...
  They still work fine, and way less hassle to use."

=====================================================================
7) r/LocalLLaMA/1hpg2e6 — "Budget AKA poor man Local LLM" (u/Boricua-vet, 2y ago)
   https://www.reddit.com/r/LocalLLaMA/comments/1hpg2e6/budget_aka_poor_man_local_llm/
   [Reference build: 2x P102-100 — PASCAL, not Maxwell, but the budget path the
   community above points to. P102-100: 10GB, 320-bit bus, 440.3 GB/s.]
=====================================================================
- $130 total build: Crosshair V Formula-Z + FX-8350 + 32GB DDR3 + 2x P102-100 ($80 for
  both). Cards power-limited to 150W via nvidia-smi (250W TDP) to run cool.
- ollama, all Q4_K_M (rounded): llama3.2:1b 112 TK/s; phi3.5:3.8b 62; mistral:7b 39;
  llama3.1:8b 37; mistral-nemo:12b 26; nexusraven:13b 24; qwen2.5:14b 20; Phi-4 (14.7B) 20;
  phi3:14b 22; mistral-small:22b 14; gemma2:27b 12; qwen 32B Q4 11-12 TK/s.
- Runs home assistant + faster-whisper + Phi-4-14B + llama3.2-3b, sub-1s responses, solar.
- Weakness: ComfyUI image gen, 1024x768 > 2 minutes (FP8).

=====================================================================
8) Maxwell M40 / GTX 970 references (earlier round, est.tech leaderboard + Reddit)
=====================================================================
- est.tech M40 (24GB, ollama) leaderboard: gemma3:4b 32.1, gpt-oss:20b 29.9,
  mistral-nemo:12b 21.6, qwen3.5:9b 20.5, mixtral:8x7b 9.4, qwen3.5:27b 5.6,
  qwen3.6:27b 5.4 t/s. M40 hits its 85C thermal ceiling (jet-engine fan); gate runs <=60C.
- r/LocalLLaMA/16somxc: 30B on M40 24GB < 10 t/s; "fan sounds like a jet engine."
- r/LocalLLaMA/1hh4dwn: 2x M40 + 2080 Ti for 70B: 3 t/s first prompt -> 1.7 t/s after;
  ollama refuses the 2080 Ti when M40s present; LocalAI ~50% better (uses all 3 GPUs).
- r/LocalLLaMA/1suii0z: 4x M40 12GB, Qwen3.6-27B fp4: 26 tok/s prompt, 8 tok/s generation.
- r/LocalLLaMA/1sitlcj: 2x M40 spec-decode: Qwen3-30B-A3B Q4 split across 2 M40s +
  Qwen3-0.6B draft on Quadro RTX 4000; ~70% acceptance, 35 t/s; 20-25 t/s at 128k context;
  "VRAM speed on the m40s is not great, at 288 GB/s."
- r/ollama/1kggwwo: GTX 970 4GB as second GPU with an RTX 4080: Gemma3 27B 5 t/s ->
  14 t/s (970 used for VRAM only).

=====================================================================
VERDICT for the benchmark at hand (4x GM107 / one M10)
=====================================================================
Realistic numbers to expect from first-hand reports:
- ollama, 32GB pool: 7-9B Q4-Q8 ~ 6-11 t/s (6+, 9, 10.86 reported); 9B ~6 t/s;
  70B ~0.5 t/s (unusable). FP16 is out entirely.
- llama.cpp pinned to one GM107: ~4.6 t/s on 8B Q6_K — slower than a dual Xeon Gold 6126
  CPU (5.96 t/s) in one report; a single decent CPU may do better. Single-GPU pinning and
  NUMA tuning made no difference in that report.
- Prompt eval is the better use (16.5 t/s vs 4.6 t/s generation in that report; tinyllama
  92 t/s prompt on GPU), but even prompt eval lost to the big CPU there.
- The two architectures that actually win on GM107:
    (a) one small model per GPU (sub-agent pattern) — the 1wuinuf plan;
    (b) heavy custom MoE sharding with PCIe multicast — 52.5 t/s decode / ~277 t/s
        prefill on a 180GB MoE with 6x M10 + a 3070 Ti (m10dis), latency-bound decode,
        needs a custom kernel stack, not stock llama.cpp/ollama.
- Power/thermal: 30-40W per GM107 under load, self-throttles; fanless card, add airflow.
- If the goal is more tokens per dollar instead: T10 16GB (~4060-class, add a fan),
  P102-100/P104-100 ($25-40 each), P40 (used 150-200 euros + 30-euro fan), M40 (deal at
  ~50 euros), Mi50 (best bank/buck per thread 6), T4 16GB 75W.
