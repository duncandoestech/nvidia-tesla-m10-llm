# M10 specialist ops — Round 11 (2026-11-08)

Sources: official HF cards + Qwen3.5-4B chat_template.jinja, llama.cpp tools/server/README.md + issues #30422/#29216, hermes-agent v0.23 local source/docs. [UNVER] = not verified this round.

## 1. Sampling settings and prompt formats

### Qwen3.5-4B (hf.co/Qwen/Qwen3.5-4B)
Thinking is ON by default ("Qwen3.5 models operate in thinking mode by default, generating thinking content signified by `\n...\n`").
Toggle: the official chat template (chat_template.jinja) branches on a jinja variable — `{% if enable_thinking is defined and enable_thinking is false %}` changes the assistant prompt. In vLLM/SGLang per card: extra_body `{"chat_template_kwargs": {"enable_thinking": false}}`. In llama.cpp: per-request field `chat_template_kwargs: {"enable_thinking": false}` (llama.cpp server README's own example is exactly `{"enable_thinking": false}`), or server-wide `--chat-template-kwargs '{"enable_thinking":false}'`; `reasoning_effort: none` also disables thinking.
Official sampling (card, verbatim):
- Thinking, general: temp 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0
- Thinking, precise coding: temp 0.6, top_p 0.95, top_k 20, min_p 0.0, pp 0.0, rep 1.0
- Instruct (non-thinking), general: temp 0.7, top_p 0.8, top_k 20, min_p 0.0, pp 1.5, rep 1.0
- Instruct, reasoning: temp 1.0, top_p 1.0, top_k 40, min_p 0.0, pp 2.0, rep 1.0 (Best Practices section; the Quickstart Tip lists top_p 0.95/top_k 20/pp 1.5 — two official variants, flagged)
- "you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions... a higher value may occasionally result in language mixing."
Output length: 32,768 for most queries; 81,920 for math/programming competition tasks.
llama.cpp flag equivalents: -t/--temp, --top-p, --top-k, --min-p, --penalty-p (presence), --repeat-penalty.

### Qwen2.5-Coder-7B-Instruct (full card fetched)
No sampling section exists in the official card (verified on the complete README) — Qwen publishes none for this model. Prompt format: standard chat template; card quickstart system prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." Context 131,072 (config.json 32,768 default + YaRN factor 4.0 for longer). Low temperature (~0.2) is common community practice for code models [community, UNVER]. Note: QwenLM/Qwen2.5-Coder repo main now redirects to Qwen3-Coder (30B-A3B and up — no sub-10B official Qwen coder).

### Seed-Coder-8B-Instruct (full card fetched)
No sampling section in the card (verified complete). Context 32,768; LLaMA-style arch, plain chat template, no system-prompt requirement shown. No official recommendation [verified]; low temp is community practice [UNVER].

### Qwen3-VL-4B-Instruct (card "Generation Hyperparameters", verbatim)
- VL: greedy=false, top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0, presence_penalty=1.5, out_seq_length=16384
- Text: greedy=false, top_p=1.0, top_k=40, repetition_penalty=1.0, presence_penalty=2.0, temperature=1.0, out_seq_length=32768
Instruct edition = non-thinking (Thinking is a separate model). OCR/screenshot prompt: standard chat template with image content blocks (`{"type":"image",...}` then text instruction); ask for verbatim text.

## 2. Embedding + reranker in llama.cpp

### Qwen3-Embedding-0.6B
Query prefix (official QwenLM/Qwen3-Embedding repo, verbatim): `Instruct: {task_description}\nQuery:{query}`; "Each query must come with a one-sentence instruction that describes the task"; default task: 'Given a web search query, retrieve relevant passages that answer the query'; "No need to add instruction for retrieval documents"; write instructions in English; instructions help ~1–5%.
Pooling: LAST token (EOS) — official code `last_token_pool(last_hidden_states, attention_mask)` then L2 normalize.
llama.cpp: `--pooling {none,mean,cls,last,rank}` — "use model default if unspecified". The official Qwen3-Embedding GGUFs carry `qwen3.pooling_type = 3` (= last; llama.cpp #29216, Qwen3-Embedding-8B-Q8_0). → `--pooling last` is NOT needed for the official GGUF; the model default applies. Add it only for third-party conversions.
Known issue #29216 (open): 8B Q8_0 on Metal returns all-null embeddings for some inputs and poisons the slot until restart (workaround -ngl 34); "Qwen3-Embedding-0.6B-Q8_0 on Metal: finite" — 0.6B unaffected there. Nothing reported on CUDA.

### Qwen3-Reranker-0.6B (ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF)
Serve: `llama-server -m qwen3-reranker-0.6b-q8_0.gguf --reranking` (command from #30422) or `--embedding --pooling rank`.
Request (server README, verbatim): POST /v1/rerank (aliases /rerank, /v1/reranking, /v1/reranking); Jina format uses key `documents`: `{"query": "...", "documents": ["...", ...], "top_n": N, "model": "..."}`; TEI format uses key `texts` (+`return_text`). Jina response: `{model, object, usage{prompt_tokens}, results:[{index, relevance_score}]}`; TEI: bare `[{index, score}]`. "Both are sorted descending by score and truncated to top_n; the score is the match probability in [0, 1]." Endpoint is "not OAI-compatible. Similar to https://jina.ai/reranker/ but might change in the future."
Model mechanics: Qwen3-Reranker-0.6B is a causal decoder reranker → "have no such limitation" (no ubatch-size cap, server README note). P(yes) is computed server-side from "yes"/"no" logits; official judge prompt (QwenLM repo): system "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."; user `: {instruction}\n: {query}\n: {doc}` with default instruction = the web-search sentence above.
KNOWN ISSUE #30422 (open, 2026-10-12, exactly this model/quant): with `--reranking`, the server copies n_vocab logits (151,669×4 B ≈ 600 KB/token) D2H for every token though nothing reads them (regression from #14208) → ~6x slower: 25 docs×139 tok 3.10 s vs 0.53 s with the proposed patch; 25×1,541 tok 35.1 s vs 5.3 s; rerank prefill ~1,100 tok/s vs ~9,300 tok/s plain completion. "A narrow PCIe link makes it dramatic" — directly relevant to M10. One-line patch in the issue (skip the logits copy when pooled embeddings are on).

## 3. Hermes v0.23: extra OpenAI-compatible endpoints without editing config.yaml
Verified in local source (~/.hermes/hermes-agent, v0.23) + docs tree.
1. One-shot CLI, zero config: `hermes chat -q "..." --provider custom --base-url http://HOST:PORT/v1 --api-key KEY -m MODEL`. hermes_cli/_parser.py: `--base-url` = "OpenAI-compatible endpoint for --provider custom (e.g. http://127.0.0.1:8000/v1)"; `--api-key` = "API key for this launch only; not persisted"; `-q` single query, `-t/--toolsets` enables tools, `-Q` quiet, `--format stream-json`. A skill can wrap this via terminal: a full agent subtask against a local llama.cpp endpoint, per-call base_url, no config change.
2. delegate_task per-call base_url: NOT available in v0.23. `provider/model/base_url/api_key/api_mode` are RESERVED_TASK_PARAMETERS (agent/delegation_provider.py: "the raw route fields (provider/model/base_url/api_key) never become model-facing"); the default backend routes every child from the `delegation.*` section or the parent's route (plugins/delegation/default/__init__.py).
3. Delegation provider plugin — the documented per-subagent route: implement DelegationProvider.task_parameters() (e.g. a `local` enum: chat/coder/vision) and route_task() returning `{"provider":"custom","model":...,"base_url":"http://HOST:PORT/v1","api_key":...}`; route dicts may carry model/provider/base_url/api_key/api_mode/request_overrides/reasoning_effort/profile (ROUTE_KEYS); switch backend inside a running session with `/delegation <name>`; backends fail closed. Full interface + worked "tiers" example: website/docs/developer-guide/delegation-provider-plugin.md.
4. MCP: an MCP server wrapping chat.completions exposes a per-call tool; base_url is fixed per server instance (register one server per endpoint). No first-party openai-chat MCP catalog entry verified in this tree [UNVER].
5. Subagent lifecycle API (developer/programmatic): per-launch `model` override; `provider` precedence "request > delegation config > parent" (website/docs/developer-guide/subagent-lifecycle-api.md).

## 4. 4B model as a sub-agent for a larger orchestrator — measured guidance
Verified data:
- Qwen3.5-4B official card, General Agent rows: BFCL-V4 50.3 (9B: 66.1); TAU2-Bench 79.9 (9B: 79.1) [card footnote: airline domain evaluated "applying the fixes proposed in the Claude Opus 4.5 system card"]; VITA-Bench 22.0 (9B: 29.8); DeepPlanning 17.6 (9B: 18.0); LiveCodeBench v6 55.8.
- Qwen3 blog (qwenlm.github.io/blog/qwen3): "even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct"; "Qwen3-1.7B/4B/8B/14B/32B-Base performs as well as Qwen2.5-3B/7B/14B/32B/72B-Base, respectively"; post-training stage 4 RL covers "agent capabilities"; Qwen-Agent example points `model_server` at any OpenAI-compatible endpoint (works with llama.cpp).
- τ-bench (arXiv:2406.12045): even gpt-4o gets pass^1 ≈61% τ-retail / ≈35% τ-airline, pass^8 <25% retail; agents "struggle with complex reasoning over databases, understanding and following ad-hoc policies, and handling compound (more than one) requests."
- τ²-bench (arXiv:2506.07982): dual-control pass^1 gpt-4.1 34%, o4-mini 42%, claude-3.7-sonnet 49%; "substantial performance decrease (around 20% pass^1) when agents must shift from autonomous operation to guiding a user" — communication/coordination is the critical bottleneck.
- Cognition, "Don't Build Multi-Agents" (cognition.ai/blog/dont-build-multi-agents): Principle 1 "Share context, and share full agent traces, not just individual messages"; Principle 2 "Actions carry implicit decisions, and conflicting decisions carry bad results"; Claude Code subagents "are usually only tasked with answering a question, not writing any code" (they lack the main agent's context); 2024 "edit apply model" practice: a small model rewriting a whole file from the big model's markdown spec beat the big model emitting diffs, "However, these systems would still be very faulty... the most slight ambiguities" — small-model failure mode = ambiguity, not effort; Cognition fine-tuned a small model for context compression ("this is in fact something we've done at Cognition").
Synthesis (each point grounded above):
- GOOD for a 4B sub-agent: single-shot, tightly specified, small-output, verifiable tasks — classify/route/label; extract to a strict JSON schema (delegate_task output_schema); OCR → clean text; summarize to a fixed template; mechanical rewrite from a complete spec (Cognition edit-apply); single-step tool call with the full tool schema in context (BFCL-V4 50.3 ≈ ~half of single-call tool accuracy).
- BAD: multi-turn multi-step tool chains (frontier pass^1 <50% on τ-bench; 4B well below); open-ended edits from ambiguous specs (Cognition failure mode); parallel subagents making independent decisions (Principle 2); long-horizon planning (DeepPlanning 17.6, VITA 22.0 ≈ near-zero); anything needing the full conversation in memory (context-starvation, Claude Code example).
- Operating rules: pass full task context + tool docs (Principle 1); one well-defined question per sub-agent; strict output schema; verify every result (self-report ≠ fact); use the card's instruct-mode sampling (temp 0.7, top_p 0.8, top_k 20, pp 1.5) with thinking OFF for latency-sensitive sub-calls (thinking ON multiplies output tokens).

## Unverified / gaps
- Qwen2.5-Coder-7B + Seed-Coder-8B: no official sampling published (full cards checked); low-temp community practice [UNVER].
- Qwen3.5-4B card's instruct/reasoning row exists in two variants (both quoted).
- unsloth Qwen3.5 GGUF chat template assumed identical to the official jinja (enable_thinking) [UNVER — check at setup].
- Hermes: no first-party openai-chat MCP catalog entry verified [UNVER].
- #30422 slowdown measured on RTX 3060 PCIe x1; M10 magnitude unmeasured [UNVER].

(All URLs quoted inline above.)
