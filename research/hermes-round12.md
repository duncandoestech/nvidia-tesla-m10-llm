# M10 specialist ops — Round 12 (2026-11-08)

Sources: llama.cpp issues/PRs + master server README (fetched today), unsloth MTP README, arXiv papers, HF cards. [UNVER] = not verified this round.

## 1. Issue #30422 — reranker copies full-vocab logits D2H every token
Status as of 2026-11-08: **OPEN** — created 2026-10-12, label bug-unconfirmed, 0 comments, no linked PR (checked via API) → **no fix merged**. github.com/ggml-org/llama.cpp/issues/30422
Root cause: #14208 (b5681) dropped the `t_logits = cparams.embeddings ? nullptr : res->get_logits()` condition; with pooling on, every input token becomes an output and `llama_context::decode()` extracts logits for all of them — `ggml_backend_tensor_get_async(..., n_outputs*n_vocab*sizeof(float))` — a full-vocab D2H copy (~600 KB/token for Qwen3's 151,669 vocab) that nothing reads. Decoder-derived embedding/rerank models (Qwen3-Reranker, Qwen3-Embedding, F2LLM) build an lm_head, so t_logits is set.
File/function: the logits-extraction block inside `llama_context::decode()` — src/llama-context.cpp (the issue quotes the decode loop; file name inferred from the function, not stated in the issue).
Proposed patch (verbatim, 2 lines, from the issue):
```diff
-        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
+        const bool pooled_embd = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;
+        if (!pooled_embd && logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
```
Mixed-server safety: the server toggles `llama_set_embeddings()` per batch (`slot->need_embd()`), so completion batches still get logits. Issue notes a fuller fix would also skip building lm_head in the graph for pooled batches. Related: #29388 (logits buffer allocation — its fix keys on whether the graph produces logits, so it does NOT cover decoder models).
Measurements (issue, RTX 3060 on PCIe 3.0 x1, Qwen3-Reranker-0.6B Q8_0, -ngl 99, 25 docs): 25×139 tok 3.10 s → 0.53 s; 25×1541 tok 35.1 s → 5.3 s; scores identical to 5 decimals; nsys: 2,119 MB DtoH in 2.57 s vs ~70 ms kernels. M10 action: apply this patch at build time until it merges (~6x gain).

## 2. Qwen3.5-4B/9B-MTP GGUF + --spec-type draft-mtp
The official Qwen3.5 smalls are MTP-native: "MTP: trained with multi-steps" (Qwen/Qwen3.5-4B card, Model Overview). GGUFs: **unsloth/Qwen3.5-4B-MTP-GGUF** (110,447 dl), **unsloth/Qwen3.5-9B-MTP-GGUF** (99,620 dl), plus 0.8B/27B/35B-A3B MTP variants (HF search, today).
llama.cpp: `--spec-type draft-mtp` is the supported mode for these dense models — llama-server accepts `--spec-type`, `-md`, `-ngld` (issue #30424); MTP PR #22673 **merged 16 May 2026**; `--spec-type mtp` was renamed to `draft-mtp` on 2026-05-13 (unsloth MTP repo README).
Official command (unsloth README, verbatim):
`llama-server -hf unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL -ngl 99 -c 8192 -fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 6`
Speedups: unsloth claims "**~1.5-2x faster generation**". Measured acceptance data found (llama.cpp issues, today):
- Qwen3.5-122B-A10B MoE, embedded MTP layer: 19.3 → 32.3 t/s, **acceptance 0.76, mean accepted len 3.3** (llama.cpp issue thread).
- Adaptive-MTP PR **#27210** ("spec: add adaptive MTP draft depth (draft-mtp-adaptive)"), dense Qwen3.8-27B Q8_0, 2× R9700 HIP: baseline ~30.0 t/s → fixed depth 3: 51.68 (reasoning) / 55.89 (prose) / 69.28 (code) / 81.94 (recall) t/s; adaptive 3..12 up to 78.03 (code) / 147.46 (recall). Flags: `--spec-draft-n-max`, `--spec-draft-n-min-adaptive`, `--spec-draft-p-min`.
Caveats for M10 (Maxwell, older hardware):
- PR #27210 "Horses for Courses": Adaptive MTP "can make performance worse on older hardware that cannot run the deep drafting and acceptance steps fast enough" → "stick with regular MTP with a low fixed depth"; "**Dense Models generally receive the most benefit**".
- **Pascal is explicitly unsupported**: issue #29174 "MTP draft GGUFs fail to load … no MTP path on Pascal". Maxwell: no public report [UNVER — test at setup; a model without MTP layers errors "the model has no MTP layers (needed by --spec-type draft-mtp)" per #30427].
- `-np > 1` and `--mmproj` are **not yet supported with MTP** (unsloth README) → the MTP chat GPU runs single-slot, and the vision model cannot use an MTP path at all.
- A mismatched MTP draft (wrong base) gives **acceptance 0.00**, gibberish thinking loops, pegged throughput (issue: mtp-Qwen3.8-27B vs UD quant) → always pair the matching official MTP GGUF with its base.
- #31495 (open, 2026-11-01): draft-mtp cut prefill ~80% on HIP (523.9→135.99 t/s) while decode gained +30% (16.16→21.02); Vulkan unaffected; author still nets 20–25% wall time (77/23 gen/prefill workload). A single-GPU CUDA log in the same thread shows prefill overhead nearly vanishes on one GPU (765→742 t/s). [Backend-specific; M10 unverified.]

## 3. SciFact nDCG@10 — published values
**Neither vendor publishes a per-task SciFact number.** Verified by full-text check (0 hits): Qwen3-Embedding paper v3 (arXiv 2506.05176 — Tables 7–9 are per-type + code only), Qwen3-Embedding-0.6B HF card (per-type tables only), Qwen3-Embedding blog (MTEB-R rerank table only), EmbeddingGemma paper v3 (arXiv 2509.20354 — Table 11 lists the FULL MTEB(English, v2) task set: **no SciFact row**), EmbeddingGemma-300M HF card (per-type/per-dim only).
Why: **SciFact is not in the MTEB(English, v2)** task set (confirmed from both papers' per-task tables; v2 English retrieval = ArXiv, CQADupstack*, FiQA, HotpotQA-HN, MLQA, SCIDOCS, TRECCOVID, …). SciFact exists only as a standalone MTEB task: huggingface.co/datasets/mteb/scifact (test: 300 queries, 5,183 docs, 339 relevant, nDCG@10).
The only per-model per-task source is the MTEB leaderboard (huggingface.co/spaces/mteb/leaderboard) — a JS app whose data backend (datasets-server) blocks scraping; no published SciFact score for either model could be extracted. **No published number to compare your 0.701 against exists in vendor material** — flagged rather than guessed.
Closest published anchors:
- Qwen3-Embedding-0.6B MTEB(Eng v2): Mean(Task) 70.70, **Retrieval type 61.83** (paper Table 7; 4B: 74.60/68.46, 8B: 75.22/69.44).
- EmbeddingGemma-300M: MTEB(Eng v2) 69.67 (768d, HF card); paper Table 11 per-task retrieval (300M): TRECCOVID 80.35, FEVER-HN 80.75, MLQA 78.95, HotpotQA-HN 71.48, ArXiv-HierClust 63.59, CQADupstackGaming 59.52, FiQA2018 47.74, CQADupstackUnix 41.52, ClimateFEVER-HN 26.71, SCIDOCS 18.43.
- EmbeddingGemma QAT (card): Q8_0 (768d) MTEB-Multilingual 60.93, Q4_0 60.62.
- FYI: EmbeddingGemma 2 (740M, multimodal, Apache 2.0, 8192 ctx) released 2026-10-27 (arXiv 2610.37364) — also fits an 8 GB die.
- [Inference, labeled] your 0.701 on SciFact sits plausibly above Qwen's 61.83 retrieval-type mean (SciFact is easier than SCIDOCS-class tasks, harder than TRECCOVID-class) — but that is calibration, not a published comparison.

## 4. Embedding throughput settings (llama.cpp server, compute-limited GPU)
Verbatim from tools/server/README.md (master, fetched today, github.com/ggml-org/llama.cpp):
- `--embedding, --embeddings` — "restrict to only support embedding use case; use only with dedicated embedding models (default: disabled)"
- `-b, --batch-size N` — "logical maximum batch size (default: 2048)"
- `-ub, --ubatch-size N` — "physical maximum batch size (default: 512)"
- `-np, --parallel N` — "number of server slots (default: -1, -1 = auto)"
- `-cb, --cont-batching` — "whether to enable continuous batching (a.k.a dynamic batching) (default: enabled)"
- `-fa, --flash-attn [on|off|auto]` — default 'auto'
- POST `/embeddings` (non-OAI): "supports all poolings, including `--pooling none` … For all other pooling types, only the pooled embeddings are returned, normalized using Euclidean norm." (OAI-compatible: /v1/embeddings.)
- Rerank NOTE (verbatim): "A bidirectional-encoder reranker (e.g. `bge-reranker-v2-m3`) must fit the whole prompt within `--ubatch-size`; if the input is too large to process, increase `--ubatch-size`. Causal decoder rerankers (e.g. `Qwen3-VL-Reranker`) have no such limitation." → Qwen3-Reranker-0.6B is uncapped by ubatch, but long embedding docs are still chunked at ubatch size.
Recommended line for the embedding GPU:
`llama-server -m Qwen3-Embedding-0.6B-Q8_0.gguf --embedding -ngl 99 -c 4096 -b 2048 -ub 1024 -np 4 -fa auto`
1. **-ub 1024 (or 2048)**: default 512 undersizes a compute-bound prefill. Data point: #31495 ruled-variables table, -ub 128 → 443.5 t/s vs -ub 2048 → 467.6 t/s (MTP off, 27B on HIP).
2. **-b 2048** (default) keeps multiple long docs in one logical request; raise only for huge per-call doc batches.
3. **-np 4** with continuous batching ON: concurrent requests overlap (no MTP restrictions apply here).
4. **-c/--ctx-size** = your real max input (0 = from model = 32,768 for Qwen3-Embedding). A 32k ctx on a 512-token doc wastes memory/graph — 2048–4096 is plenty.
5. Rerank mode: `--reranking` (or `--embedding --pooling rank`), and apply the #30422 patch — that is the single biggest throughput win (~6x, section 1).
6. Keep doc-per-request counts modest so batching can interleave.
No published llama.cpp embedding-specific benchmark could be fetched; the above is flag-level.

## Unverified / gaps
- #30422 open as of 2026-11-08 (0 comments, no linked PR); "src/llama-context.cpp" inferred from the `llama_context::decode()` name.
- MTP on Maxwell (sm_50): Pascal explicitly unsupported (#29174); no Maxwell MTP report found [UNVER — test at setup].
- MTP decode gains on M10: all published numbers are HIP/Vulkan/newer CUDA; 1.5–2x is unsloth's general claim [UNVER on M10].
- Per-model SciFact nDCG@10: not published in vendor material (verified negative); leaderboard is JS-only, backend blocked [no citable published number].
- Embedding throughput: no published llama.cpp benchmark found; guidance is flag-level (README) + one ubatch data point (#31495).

All URLs quoted inline above.
