# M10 specialist model selection — Round 10 (2026-11-08)

Target: 4x 8 GB CC5.0 GM107, ~83 GB/s each, llama.cpp CUDA 12.9 (f32 compute, no FP16/BF16 math), ~6 GB per model. Sources fetched 2026-11-08: HF model cards/API, llama.cpp master docs, kev GitHub. [EST] = estimate; [UNVER] = not verified this round.

## 0. llama.cpp server capabilities (master, tools/server/README.md — verbatim)
- `--embedding, --embeddings`: "restrict to only support embedding use case; use only with dedicated embedding models"; `--pooling {none,mean,cls,last,rank}`; OpenAI-compatible /v1/embeddings.
- `--rerank, --reranking`: "enable reranking endpoint on server"; POST /reranking is "Similar to https://jina.ai/reranker/ ... Requires a reranker model (such as bge-reranker-v2-m3) and the `--embedding --pooling rank` or `--reranking` options" (PR #9510; aliases /rerank, /v1/rerank); multimodal rerankers (e.g. Qwen3-VL-Reranker) supported.
- KV cache `-ctk/-ctv {f32,f16,q8_0,q4_0,...}` default f16 (storage only — on CC5.0 compute stays f32; q8_0 saves VRAM).
- `-sm {none,layer,row,tensor}` (Maxwell caveats: Round 9).

## 1. General chat (<=6 GB) — pick: Qwen3.5-4B
Qwen3.5-4B (Feb 2026, Apache-2.0, huggingface.co/Qwen/Qwen3.5-4B): "Causal Language Model with Vision Encoder" (natively multimodal, pipeline tag image-text-to-text); GDN hybrid: 32 layers = 8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN)), hidden 2560; context 262,144 native, "extensible up to 1,010,000"; "MTP: trained with multi-steps".
Card benchmarks (4B / 9B): MMLU-Pro 79.1 / 82.5; MMLU-Redux 88.8 / 91.1; GPQA Diamond 76.2 / 81.7; IFEval 89.8 / 91.5; LiveCodeBench v6 55.8 / 65.6; HMMT Feb 25 74.0 / 83.2; BFCL-V4 50.3 / 66.1; TAU2-Bench 79.9 / 79.9. Same table: GPT-OSS-20B MMLU-Pro 74.8; GPT-5-Nano OCRBench 75.3. -> 4B beats GPT-OSS-20B on MMLU-Pro; 9B beats it everywhere. Best per-GB among candidates with verified numbers (no Gemma 3 4B / Llama 3.1 8B / Phi-4 numbers found in fetched sources — gap flagged below).
GGUF: unsloth/Qwen3.5-4B-GGUF (1,530,019 downloads) + unsloth/Qwen3.5-4B-MTP-GGUF (110,447); unsloth/Qwen3.5-9B-GGUF (1,787,774) + -MTP (99,620). ggml-org publishes only 0.8B + 35B-A3B GGUF (llama.cpp quickstart: `llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF`).
Sizes [EST, anchor: Q4_0 4B = 2.6 GB per jica98 card]: 4B Q4_K_M ~2.7-2.8 GB (fits with ~3 GB headroom), Q8_0 ~4.2 GB; 9B Q4_K_M ~5.5-5.8 GB — workable in 8 GB but tight (-ctk/-ctv q8_0, modest ctx).
Maxwell note: GDN layers run the base scalar CUDA kernel (the fast MMA kernel of PR #29353 is Ampere+ only — Round 9); all-f32 compute.

## 2. Coder <10B
Reference (official numbers, from ByteDance-Seed/Seed-Coder-8B-Instruct card table): Qwen2.5-Coder-7B-Instruct (7.61B, 128K, Apache-2.0): HumanEval 88.4, MBPP 83.5, MHPP 26.7, BigCodeBench 48.8 / Hard 20.3, LiveCodeBench (2410-2502) 17.3.
- Seed-Coder-8B-Instruct (ByteDance, Apr 2025, MIT, arXiv:2506.03524, 32K): HumanEval 84.8, MBPP 85.2, MHPP 36.2, BigCodeBench 53.3 / 26.4, LCB (2410-2502) 24.7 — beats Qwen2.5-Coder-7B on everything except HumanEval. Same table: Qwen3-8B 84.8 / 77.0 / 32.8 / 51.7 / 23.0 / 23.5; OpenCoder-8B 83.5 / 79.1 / 30.5 / 50.9 / 18.9 / 17.1; Yi-Coder-9B 82.3 / 82.0 / 26.7 / 49.0 / 17.6 / 17.5.
- Newer-generation numbers: Qwen3.5-4B LiveCodeBench v6 (2026 window) = 55.8; Qwen3.5-9B = 65.6 (Qwen3.5 card). LCB v6 for Qwen2.5-Coder-7B / Seed-Coder-8B not found -> no same-benchmark comparison [UNVER].
- ccore/Qwen3.5-4B-3D-Coder (Oct 2026, 254,053 downloads): three.js/GLB 3D code only.
- jica98/qwen3.5-4B-super-coder (Q4_0 GGUF ~2.6 GB): self-reported HumanEval+ 0.00 and MBPP+ 0.00 — "the Q4_0 quant may degrade code generation significantly" — excluded.
Verdict: Qwen2.5-Coder-7B Q4 (~4.4 GB [EST]) = verified workhorse under 6 GB; Seed-Coder-8B = newer challenger (higher LCB/BCB, lower HumanEval); Qwen3.5-4B Q4 (~2.6 GB) = best code+chat+vision all-in-one (LCB v6 55.8).

## 3. Embedding — pick: Qwen3-Embedding-0.6B
- Qwen3-Embedding-0.6B (Apache-2.0, 32K, <=1024-d MRL, 100+ languages, arXiv:2506.05176): MTEB (Eng v2) Mean(Task) **70.70** — vs NV-Embed-v2 69.81 (7.8B), gte-Qwen2-7B 70.72 (7.6B), stella_en_1.5B 69.43, multilingual-e5-large 65.53 (0.6B); MTEB multilingual 64.33 (vs NV-Embed-v2 56.29, GritLM-7B 60.92, Cohere-m3 61.12); C-MTEB 66.33. 4B = 74.60, 8B = 75.22 (eng v2; 8B was No.1 multilingual, 70.58, as of 2025-06-05).
- embeddinggemma-308m (Google, Sept 4 2025 blog, arXiv:2509.20354): 308M (~100M model + ~200M embedding), Gemma 3 arch, 2K context, MRL 768->128, sub-200 MB RAM QAT; "highest ranking open multilingual text embedding model under 500M on MTEB" (exact score in chart image only, not text). Official ggml-org GGUF: huggingface.co/collections/ggml-org/embeddinggemma-300m-68b2a87d78ca52408f7918f3.
- EmbeddingGemma 2: natively multimodal (code/vision/audio); "MTEB Code ... from 68.76 to 78.68" (+9.92) [blog snippet].
- llama.cpp: --embedding verified (s.0). Official GGUF: **Qwen/Qwen3-Embedding-0.6B-GGUF** (168,018 downloads). Q8_0 ~0.65 GB [EST].
Verdict: Qwen3-Embedding-0.6B — best verified MTEB at this size, 32K ctx (EmbeddingGemma: 2K).

## 4. Reranker (llama.cpp --rerank) — pick: Qwen3-Reranker-0.6B
Qwen3-Reranker-0.6B (Apache-2.0, 32K, instruction-aware, 100+ languages): MTEB-R (Eng) **65.80**, CMTEB-R 71.31, MMTEB-R 66.36, MLDR 67.28, MTEB-Code 73.42, FollowIR 5.41 — vs BGE-reranker-v2-m3 57.03/72.16/58.36/59.51/41.38/-0.01, gte-multilingual-reranker-base 59.51/74.08/59.44/66.33/54.18, jina-multilingual-reranker-v2-base 58.22/63.37/63.73/39.66/58.98. 4B = 69.76 (MTEB-R best), 8B = 69.02. Format: yes/no cross-encoder on Qwen3-0.6B-Base (judge prompt, "yes"/"no" logit difference) — compatible with the --rerank endpoint.
GGUF: no official Qwen GGUF; community: dean2155/Qwen3-Reranker-0.6B-Q8_0-GGUF (482 dl) and -Q4_K_M (319), DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF (529). Q8_0 ~0.65 GB [EST].

## 5. Vision (OCR/screenshots) — pick: Qwen3-VL-4B (or fold into Qwen3.5-4B)
- Qwen3-VL-4B-Instruct (Sept 2025, Apache-2.0, arXiv:2505.09388): 262,144 ctx (expandable 1M); "Expanded OCR: Supports 32 languages (up from 19); robust in low light, blur, and tilt". Official GGUF: Qwen/Qwen3-VL-4B-Instruct-GGUF (730,217 dl); also ggml-org/Qwen3-VL-2B-Instruct-GGUF (29,283 dl).
- Numbers (third-party: codersera.com Qwen3-VL-4B-vs-8B guide, updated June 2026 — NOT official Qwen): DocVQA 4B-Instruct ~91% (8B 96.1%, 4B-Thinking 94.2%), OCRBench 4B ~85% (8B 89.6%), ScreenSpot 4B ~90% (8B 94.4%), MMBench ~84%; Qwen3-VL-4B Q4_K_M ~3.3 GB, "Minimum 6 GB VRAM | Comfortable at 8 GB"; 2B ~1.9 GB. Official card tables live in images only (aliyun CDN unreachable from this host) [UNVER].
- llama.cpp: official GGUFs run via llama-mtmd-cli / llama-server -mm with a separate mmproj vision file (per codersera; mtmd is llama.cpp's multimodal tool).
- One-GPU shortcut: Qwen3.5-4B is natively multimodal — official card OCR rows: OCRBench 85.0, OmniDocBench 1.5 86.2, CC-OCR 76.7, MMLongBench-Doc 54.2, CharXiv(RQ) 70.8 (Qwen3.5-4B; 9B: 89.2/87.7/79.3).

## 6. Kev (github.com/jaredpalmer/kev)
What: prefill-only decision models — frozen Qwen3.5 base + LoRA adapter + pointer head; answers choice/noul/score questions as calibrated probabilities (no text generation); System One API (POST /v1/systemone); ships as adapter+head, base pulled from Hub; Kev 1.0 = 0.8B/4B/9B/27B (published 2026-09-24).
- Kev-0.8B: Qwen3.5-0.8B-Base (24 layers: 18 GDN + 6 attention), LoRA r16/a32 = 11.3M params, T=2.35; "runs on a laptop or a 4 GB GPU"; accuracy (new sources, dev/test) 0.648/0.697, Brier 0.481/0.416.
- Kev-4B: Qwen3.5-4B-Base (32 layers: 24 GDN + 8 attention, hidden 2560), LoRA 33.8M, T=2.41; "small enough for one 24 GB GPU or a 32 GB Apple Silicon Mac"; CUDA: "one L40S, H100 or any GPU with about 16 GB free"; accuracy 0.817/0.838, Brier 0.269/0.242.
Requirements (pyproject.toml, verbatim): requires-python ">=3.12,<3.14"; deps: torch>=2.6,<2.9; transformers>=5.17,<6; accelerate>=1.15.0; datasets>=3.0; peft>=0.21; numpy>=2.5.3; pydantic>=2.9; scikit-learn>=1.9.1; serve extra: fastapi>=0.115, typesafe-sdk>=0.6.0, uvicorn>=0.30 (mlx-lm 0.31.x darwin-arm64 only). Install from source — "pip install kev installs a different package" (K.E.V. ORM, 2016). No CUDA pin in pyproject (standard torch cu12x wheels).
Runtime (serve.py/checkpoint.py, verbatim): `--device auto` probes in a child process ("tests an allocation and matrix multiplication in a separate process"), CUDA first, then MPS, then CPU; KEV_DEVICE_PROBE_TIMEOUT default 30 s. GPU serving dtype default = bf16 ("2-4.5x faster than fp32 on an L4, same answers"; Kev-4B: 209 -> 118 ms at 101 tokens, 850 -> 189 ms at 330 tokens); KEV_DTYPE=fp32 = "the exact path used for evaluation"; fp16 accepted. CUDA defaults: cuda_graphs=True; fused Triton kernels "needs flash-linear-attention fused_qwen35.FLA_VERSION and refuses any other ... not in the serve extra, pinned in the Modal images"; overrides KEV_FUSED=0/1, KEV_CUDA_GRAPHS=0/1.
M10 fit (8 GB, sm_50):
- Kev-0.8B: weights ~1.6 GB bf16 / ~3.3 GB fp32 [EST = params x bytes] — fits easily (card claims any 4 GB GPU).
- Kev-4B: ~8.2 GB bf16 / ~16.4 GB fp32 — does NOT fit 8 GB (validated at ~16 GB free).
- FP32 on old GPU: supported (KEV_DTYPE=fp32). On sm_50 there is no native bf16/fp16 math (PyTorch upcasts to fp32), so expect bf16 ~= fp32 speed with the memory benefit; the "2-4.5x" figure is an L4 (tensor-core) claim, not a Maxwell claim.
- Fused FLA kernels + CUDA graphs on sm_50: untested in the repo (validated on L4/L40S/H100/H200/MLX) [UNVER]; if they error, disable via KEV_FUSED=0 / KEV_CUDA_GRAPHS=0 (eager is the validated path — "the eager path every reported number uses").
- torch 2.6-2.8 CUDA wheels include sm_50 [UNVER — wheel matrix not checked this round]; which cuXXX wheel matches the host's CUDA 12.9 / driver needs checking at setup time.

## Unverified / gaps
- embeddinggemma exact MTEB score (chart image only, not text).
- Official Qwen3-VL-4B OCR/DocVQA numbers (card images; CDN unreachable) — third-party codersera numbers used instead.
- GGUF byte sizes (HF repo file listings bot-blocked; [EST] anchored on jica98 Q4_0=2.6 GB and Qwen3-VL-4B Q4_K_M ~3.3 GB).
- LCB v6 for Qwen2.5-Coder-7B / Seed-Coder-8B (same-benchmark coder comparison).
- Per-GB vs Gemma 3 4B / Llama 3.1 8B / Phi-4 (not in fetched sources).
- torch sm_50 wheel matrix; FLA/Triton + CUDA graphs on sm_50; exact PyTorch cuXXX build vs host driver.
