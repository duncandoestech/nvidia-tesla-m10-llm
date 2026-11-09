# Real-world delegation tasks for Hermes (from Claude, 2026-11-08)

Read /tmp/m10-role-models.md first (endpoints and how to call each model).
You are the orchestrator: plan, delegate the bounded piece to the named model, then VERIFY its output yourself.
Do not do the delegated piece yourself first. Count retries. Time each delegated call.
Write all scratch files under /tmp/m10-deleg/ . Final report: /tmp/m10-delegation-report.md

## T1. Coder (8091): log summarizer
Have the coder write `smi_summary.py`: reads an nvidia-smi CSV with NO header, columns
`timestamp, index, temperature.gpu, power.draw ("51.30 W"), clocks.sm ("1137 MHz"), clocks.mem, utilization.gpu ("87 %"), memory.used, clocks_event_reasons.active`
and prints one line per GPU index: max temperature, mean power, max power, minimum SM clock among rows with utilization >= 50 %, and the number of such rows.
Input: /tmp/m10-sample.smi.csv . Run it. Check its numbers with your own independent one-liner. Up to 2 fix rounds with the coder.

## T2. Vision (8090): read a real web page screenshot
Use firecrawl to take a screenshot of https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md (the CUDA section if you can scroll/crop to it).
Send the image to 8090 and ask for the exact CUDA build commands shown. Compare with the page's markdown from firecrawl. Report exact/near/wrong.

## T3. General (8090, text, thinking off): faithful summaries
For /tmp/m10-research-round10.md, round11.md, round12.md: ask 8090 for 5 bullets each, keeping every number exact.
Check every bullet against the source. Report: bullets total, bullets with a factual error, examples.

## T4. Kev (8095): issue triage
Fetch the 30 most recent open llama.cpp issues that mention CUDA (GitHub API search, e.g. `repo:ggml-org/llama.cpp is:issue is:open CUDA`).
For each, one Kev request with state = title + first ~1500 chars of body and questions:
- `old_gpu` noul: "Is this issue specific to older NVIDIA GPUs (Maxwell, Pascal, compute capability below 7.0)?"
- `kind` choice: bug / performance / feature request / question (give each a one-line description)
- `severity` score: ["cosmetic", "minor", "breaks some setups", "crash or wrong output for many users"]
Then judge 10 of them yourself (blind to Kev's answer first) and report agreement per question, and Kev's total time.

## T5. Embeddings + reranker + general (8093, 8094, 8090): RAG over your own research
Corpus: /tmp/m10-research-round3.md ... round13.md. Split into ~800-character chunks with filename + heading.
Embed all chunks (8093, document prefix). For each question below: embed the query (query prefix), take top 20 by cosine, rerank with 8094, keep top 4, and ask 8090 to answer ONLY from those chunks, citing filenames.
Questions:
1. What rows-per-block / warps setting did the ik_llama.cpp PR for Maxwell use, and what gain did it report?
2. What is the cause and fix for bad Qwen3-Embedding output in llama.cpp?
3. Which MTP settings were reported for Qwen3.6 MoE and on what hardware?
4. What power limit findings were reported for the Tesla M10?
5. What is the recommended sampling for Qwen3.5 small models with thinking off?
Grade each answer (correct / partly / wrong / unsupported) against the source files. Note whether the reranker changed the top-4 vs. embeddings alone.

## Report format (/tmp/m10-delegation-report.md)
TLDR first (one line per model: "use it for X / don't use it for Y"). Then per task: model, number of calls, wall time, retries, verdict, one concrete example.
When done, reply in this pane: DONE-DELEG
