# Role models on the M10 host's Tesla M10 (for Hermes)

Each model runs on its own M10 GPU (Maxwell, slow: treat them as cheap helpers, not as your main brain).
All servers are OpenAI-compatible llama.cpp servers unless noted. Host: `$M10_HOST` (the M10 host's LAN IP; Docker-published ports, no firewall change needed).

| Port | Role | Model | Speed | Measured quality |
|---|---|---|---|---|
| 8090 | general + vision | Qwen3.5-4B Q4_0 (+mmproj) | ~18 t/s decode | HumanEval 76.8 %, OCR 24/24 |
| 8091 | coder | Qwen2.5-Coder-7B-Instruct Q4_0 | ~12 t/s | HumanEval 89.0 % |
| 8093 | embeddings | embeddinggemma-300M Q8_0 | ~7 docs/s | SciFact nDCG@10 0.789 |
| 8094 | reranker | Qwen3-Reranker-0.6B Q8_0 | see notes | +0.06 nDCG@10 on SciFact |
| 8095 | decisions (Kev) | kev-0.8b (not llama.cpp) | ~0.4 s/request | classifier / scorer |

## How to call them (no config change needed)

One-off chat through the Hermes CLI:

    hermes chat -q "..." --provider custom --base-url http://$M10_HOST:8091/v1 -m coder

Or plain HTTP:

    curl -s http://$M10_HOST:8091/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"..."}],"temperature":0.2,"max_tokens":1024,
           "chat_template_kwargs":{"enable_thinking":false}}'

- **8090 general/vision**: always send `"chat_template_kwargs":{"enable_thinking":false}` (thinking on is ~10x slower for a small gain). Images: OpenAI `image_url` content part with a `data:image/png;base64,...` URL.
- **8093 embeddings**: `POST /v1/embeddings {"input": [...]}`. Prefix queries with `task: search result | query: ` and documents with `title: none | text: `.
- **8094 reranker**: `POST /v1/rerank {"query": "...", "documents": [...], "top_n": 5}`; returns `results[].index` + `relevance_score`. Rerank only the top 20 or so from embeddings.
- **8095 Kev**: `POST /v1/systemone` with `{"state": "<text>", "model": "kev-latest", "questions": {"<id>": {"type": "noul"|"choice"|"score", "instructions": "...", "criteria": ...}}}`. noul = yes-probability; choice criteria = `{"option": "description"}`; score criteria = list of levels, low to high. Several questions per request is fine.

## Delegation rules that work for small models

- One tightly specified, checkable task per call. Give the exact output format.
- Do not hand them multi-step tool chains. You plan and verify; they draft, classify, extract, embed.
- Always check their output (run the code, spot-check classifications, compare against the source).
