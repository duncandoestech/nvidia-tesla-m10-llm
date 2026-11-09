# T6: all four M10 GPUs at once (from Claude, 2026-11-08)

Goal: one real job where all four helper models work AT THE SAME TIME, each on its own M10 GPU.
Job: an "llama.cpp CUDA issue digest" for the 40 most recent open llama.cpp issues mentioning CUDA
(GitHub search `repo:ggml-org/llama.cpp is:issue is:open CUDA`, title + first 2000 chars of body).
Fetch the issues FIRST (before starting any lane), save to /tmp/m10-deleg/t6-issues.json.

Run four lanes concurrently: one Python script, one thread (or process) per lane, all started together.
Each lane works through its whole list on its own; lanes do not wait for each other.
Log for every call: lane, issue/item id, start and end timestamps (time.time()), ok/error.

- Lane A, GPU 0, Kev (8095): for each issue ask `old_gpu` (noul), `kind` (choice bug/performance/feature request/question),
  `severity` (score: cosmetic / minor / breaks some setups / crash or wrong output for many users).
  Then, in the same lane, score every chunk in /tmp/m10-deleg/t5-corpus.json with one `actionable` noul question:
  "Does this note contain a concrete setting or change someone could apply to speed up LLM inference on a Tesla M10?"
- Lane B, GPU 1, general (8090, thinking off): for each issue, a 2-sentence plain-English summary (what broke, on what hardware).
- Lane C, GPU 2, coder (8091): for each issue, extract the exact command line(s) the reporter ran (llama-server / llama-cli /
  llama-bench / cmake) as JSON `{"commands": [...]}`; empty list if none. Copy verbatim, do not invent flags.
- Lane D, GPU 3, embeddings (8093): embed every issue (doc prefix); find near-duplicate pairs (cosine >= 0.85);
  for each issue, the single most similar chunk from t5-corpus.json (reuse its stored vectors) with its cosine.

After all lanes finish (only then), merge everything into /tmp/m10-deleg/t6-digest.md: one row per issue
(number, title, Kev kind/severity/old_gpu, summary, commands, duplicate-of, closest research note),
plus the top 10 `actionable` research chunks by Kev's probability.

Verify like before: spot-check 8 issues yourself (summary faithful? commands verbatim? Kev sensible? dup pairs real?).

Report: append a "T6: four GPUs in parallel" section to /tmp/m10-delegation-report.md:
- per-lane: calls, lane wall time, mean s/call, errors
- overall wall time vs the sum of the four lane times (the parallel speedup)
- the time window in which all four lanes were running at once
- spot-check verdicts with one concrete example per lane
Reply DONE-T6 when finished. Do not write "DONE-T6" anywhere before then.
