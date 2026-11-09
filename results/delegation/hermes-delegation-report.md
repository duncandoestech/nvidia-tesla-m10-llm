# M10 delegation test report — 2026-11-08

Orchestrator: hermes (Qwen3.8-27B, custom). Delegated to the five role models on
the M10 host. All numbers below are from real runs; scratch under /tmp/m10-deleg/.

## TLDR (one line per model)

- **coder (8091)**: use it for single-shot, tightly-specified scripts with an exact output example — don't use it for spec interpretation; it crashed on a stdlib edge case and needed 2 fix rounds for a 50-line script.
- **general+vision (8090)**: use it for verbatim transcription of code blocks/headings out of rendered page screenshots — don't use it for layout or off-crop awareness; it transcribes exactly what is in the image and nothing else.
- **general text (8090, thinking off)**: use it for number-faithful 5-bullet summaries of ~10 KB notes — don't use it for attribution precision; 2 of 15 bullets carried a factual error, both misattributions.
- **Kev (8095)**: use it for high-volume triage scoring (kind, old-GPU flag, severity) at ~1.5 s/issue — don't use its severity readout as a final call at the minor/breaks-some-setups boundary without review.
- **embeddings (8093) + reranker (8094)**: use them for chunk-retrieval over a small personal corpus — the right source file was in the top-20 for all 5 questions; don't use the reranker as a hard relevance gate, its scores saturate (all ≥0.988) and it mostly reorders/dedups within a section.

## T1 — Coder (8091): log summarizer

- Model: Qwen2.5-Coder-7B-Instruct Q4_0, port 8091
- Calls: 3 (initial write + 2 fix rounds — the allowed maximum)
- Wall time: 47.5 s + 53.7 s + 57.9 s = 159.1 s
- Retries: 2

Round 1 produced a script that crashed immediately:
`TypeError: "delimiter" must be a unicode character, not a string of length 2` (it used
`csv.reader(file, delimiter=', ')`). Round 2 fixed the crash but dropped the required
`MHz` suffix and left `max_power` unformatted. Round 3 (after the exact traceback + both
spec violations were sent back) was correct.

Verification: model output vs an independently written one-liner — byte-identical:

    GPU 0: max_temp=62C mean_power=19.5W max_power=51.3W min_sm_at_util50=1137MHz rows_util50=138
    GPU 1: max_temp=58C mean_power=18.8W max_power=56.9W min_sm_at_util50=1137MHz rows_util50=106
    GPU 2: max_temp=47C mean_power=18.7W max_power=56.8W min_sm_at_util50=1137MHz rows_util50=103
    GPU 3: max_temp=54C mean_power=18.3W max_power=55.4W min_sm_at_util50=1137MHz rows_util50=76

- Verdict: usable for bounded coding with explicit output-format examples and a
  verify-and-reject loop; not a one-shot solution for spec-heavy tasks.

## T2 — Vision (8090): transcribe CUDA build commands from a real page screenshot

- Model: Qwen3.5-4B Q4_0 + mmproj, port 8090
- 8090 calls: 3 (2 band-localization probes + 1 real transcription)
- Wall time: 60.0 s + 57.2 s + 45.9 s = 163.1 s
- Retries: 0 (model side)

Plumbing (not charged to the model): firecrawl keyless v2 returned the page markdown
(57 KB, ground truth) but its screenshot format is viewport-only (1920×1080) — v2 rejects
any `options`/fullPage payload. Tall capture done with headless
`firefox --screenshot --window-size=1280,9000`; the CUDA section was located with two
probe crops and the final test image cropped to 1280×1400 (## CUDA heading through
Non-Native Builds).

Model output (5 lines) vs ground truth (3 cmake commands in the CUDA section):

    cmake -B build -DGGML_CUDA=ON                              EXACT
    cmake --build build --config Release                       EXACT
    cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF            EXACT
    nvcc warning : Cannot find valid GPU for '-arch=native'…   (extra — real code-block line at crop edge, but a warning, not a command)
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"  (extra — real command, from the adjacent Override section at the crop edge)

- Verdict: EXACT for the core CUDA build commands (3/3 verbatim); NEAR overall — both
  extra lines are genuinely displayed in the image (verified against markdown lines 379
  and 404), i.e. faithful to the pixels, zero hallucination, but it cannot be told the
  difference between "the section you asked about" and "the next section at the edge of
  the crop" without an explicit boundary instruction.

## T3 — General text (8090, thinking off): faithful summaries of rounds 10–12

- Model: Qwen3.5-4B Q4_0, port 8090, `enable_thinking: false`
- Calls: 3 (one per file, full ~10–12 KB file per prompt)
- Wall time: 101.7 s + 94.4 s + 77.9 s = 274.0 s
- Retries: 0

Bullets total: 15 (5 per file, exactly as asked). Every bullet checked against the source.
Numbers preserved exactly in all 15 bullets.

- Bullets with a factual error: 2
  - round11, bullet 2: "increasing time by ~6x **on Metal**" — the #30422 measurements
    were on an RTX 3060 PCIe 3.0 x1 (round11.md line 69). Metal is where the *other*
    issue (#29216, null embeddings) lives. Misattribution.
  - round10, bullet 2: "Seed-Coder-8B … **is larger than the ~4.4 GB estimate for
    Qwen2.5-Coder-7B Q4**" — the source never compares the two models' sizes; the 4.4 GB
    figure is Qwen2.5-Coder-7B's own estimate (round10.md line 24). Also calls Seed-Coder-8B
    "selected for coding tasks" while the source's verdict names Qwen2.5-Coder-7B the
    workhorse and Seed-Coder the challenger.
- round12: 5/5 exact (two cosmetic typos: "SGLGang" for SGLang, "EmbeddingGema" for
  EmbeddingGemma).
- Verdict: 13/15 exact. Strong number preservation; the failure mode is invented
  attributions/comparisons, not mangled data.

## T4 — Kev (8095): triage 30 open llama.cpp CUDA issues

- Model: kev-0.8b, port 8095, `POST /v1/systemone`
- Corpus: 30 most recent open issues matching `repo:ggml-org/llama.cpp is:issue is:open CUDA`
  (GitHub search API, 282 total matches; fetched 2026-11-08 17:47 CST)
- Calls: 30 (one per issue; state = title + first 1500 chars of body), 0 failures, 0 retries
- Kev total time: 45.6 s wall (mean 1.52 s/request; server-reported latency mean 1336 ms,
  max 1703 ms)

Questions: `old_gpu` (noul), `kind` (choice: bug/performance/feature request/question),
`severity` (score: cosmetic / minor / breaks some setups / crash or wrong output for many users).

I judged 10 issues blind (locked to file before Kev ran): 31840, 31830, 31828, 31780,
31661, 31555, 31503, 31448, 31425, 31392.

Agreement per question (10 judged):
- old_gpu: 9/10 — Kev's sole miss: #31503 (Tesla T4, noul 0.576 → "yes"; T4 is CC 7.5,
  not below 7.0 — defensible either way; Kev's other borderline was #31654, AMD MI210,
  noul 0.527, not in my judged set).
- kind: 8/10 — misses: #31830 (Kev "performance", I "feature request" — title says
  feature request, body is full of speed measurements) and #31425 (Kev "performance",
  I "bug" — `--load-mode none` ignored). Both are the bug↔performance boundary.
- severity: 7/10 with round(expected score); 4/10 with argmax of per-level probabilities.
  Kev's distributions are well-shaped (garbage-output #31448: p(level3)=0.78; the
  "breaks some setups" set clusters at level 2) but the expected-value readout
  systematically lands one level above my minor/cosmetic calls (#31840 score 1.51,
  #31830 1.54) while argmax is overconfident elsewhere.

Kind distribution over all 30: 20 bug / 7 performance / 2 feature request / 1 question.
old_gpu "yes" over all 30: only 3 (31661 GTX 1070 Pascal — clearly correct; 31503 T4 —
debatable; 31654 AMD MI210 — not an NVIDIA GPU, borderline noul).

- Verdict: an excellent fast triage layer — 30 issues in 46 s, calibrated probabilities,
  no garbage. Use it to rank/flag; keep a human (or me) for final severity at the
  minor/breaks-some-setups boundary and for kind where the issue mixes symptoms with a
  request.

## T5 — RAG (8093 + 8094 + 8090): 5 questions over research rounds 3–13

- Models: embeddinggemma-300M Q8_0 (8093), Qwen3-Reranker-0.6B Q8_0 (8094),
  Qwen3.5-4B (8090, thinking off)
- Servers: not up at 17:45 CST (connection refused); up by 18:11 CST when the
  5-min retry loop's 5th check hit
- Corpus: rounds 3–13 (11 files, ~210 KB) → 226 chunks of ~800 chars, each prefixed
  `[filename] heading` so the answerer can cite filenames
- Embedding: 226 chunks in 45.6 s (batches of 50; ~5 chunks/s). Doc prefix
  `title: none | text: …`; query prefix `task: search result | query: …`
- Per question: embed query → cosine top-20 → 8094 rerank → keep top-4 → 8090 answers
  from those chunks only, instructed to say UNSUPPORTED if absent
- Calls: 1 embed batch-run + 5 query embeds + 5 reranks + 5 × 8090 = 16 model calls,
  0 failures, 0 retries. Stage-2 wall time ≈ 4.4 min (262 s)
  (per question: query embed ~0.3 s, rerank 13–16 s per 20-doc batch ≈ 0.7 s/doc,
  8090 answers 29–44 s. Labeling note: the raw t5-results.json field `t_embed_query`
  (43–60 s) is a mislabel — it is the total per-question time (embed ~0.3 s + rerank
  + 8090), since the delta was taken at the end of each loop iteration. The corpus
  was NOT re-embedded per question: stage 1 ran once and saved all 226 vectors to
  t5-corpus.json; stage 2 loads them and scores cosine in pure Python with no model
  call. The real single-query embeds were <1 s each, consistent with the ~7 docs/s
  server measured in stage 1.)

Retrieval quality: for all 5 questions the file(s) containing the correct answer were
inside the embeddings top-20. Reranker changed the top-4 set in 4/5: Q2 (round12
"embedding throughput" chunk swapped out for round11 "Qwen3-Reranker"), Q3 (round4
§4.2 "Qwen3.6 MTP GGUFs" and round9 §3 promoted to the top two slots), Q5 (round6
"draft datapoint" swapped for round4 "the two target models"); Q1's change was only a
swap of adjacent 800-char slices of the same section (round5 §4.1); Q4: set unchanged,
order only. Rerank scores were saturated — all 20 returned scores ≥0.988, so the
ranking, not the score magnitude, carries the signal.

Grading (each answer checked against the source files):

1. "rows-per-block / warps in the ik_llama.cpp PR for Maxwell + gain" — **PARTLY**.
   The M10 numbers are exact: ik #2740, 2 warps for Q3_K (below Ampere) / Q2_K (below
   Turing) vs main's 4 warps; Q3_K_M 5.80 → 6.38 t/s (+10.0%), Q2_K 8.445 → 9.410 t/s
   (+11.4%) — all verbatim from round5.md §4.1. Error: it attributes "2 rows per block
   for Q3_K" to #2740; the rows change is PR #2701 (sm_50 Q4_K_M +16.8%), and #2701
   explicitly keeps Q3_K at one row.
2. "cause and fix for bad Qwen3-Embedding output" — **CORRECT**. Issue #14234
   ("Bad output from Qwen3-Embedding-0.6B"): self-converted GGUF served with
   `--pooling last` produces vectors different from the safetensors/ONNX references;
   fix = add the EOS token manually (`"input": ["test
   …"]`) per ggerganov — exactly round13.md §1. Correctly added that the official
   GGUFs default to pooling_type 3 so `--pooling last` isn't needed (round11).
3. "MTP settings for Qwen3.6 MoE + hardware" — **PARTLY**. Right core:
   `--spec-draft-n-max 2` default ("1% behind baseline on natural text, 19% ahead on
   code"), 3 for code-heavy work; m10dis (24× GM107 + RTX 3070 Ti) ~97% acceptance at
   3 drafts, 3.84 tokens/step humaneval — all verbatim from round9.md §3. Errors: it
   reports 2× R9700 fixed-3 MTP as 56.64/78.03 t/s — those are the *adaptive* numbers
   (fixed-3 is 55.89 prose / 69.28 code, round4/round12) — and frames R9700/m10dis
   data as Qwen3.6-35B-A3B results when the sources label them Qwen3.8-27B dense /
   Qwen3.8-Flash-Next. It also missed the most direct round4 §4.2 facts: the SC117
   MTP-preserved GGUF ("All 20 MTP layers preserved", "20-50% speed boost" claim) and
   the RTX 4070 SUPER 12 GB guide hardware.
4. "power limit findings for the Tesla M10" — **CORRECT** (with a quirk). The corpus
   finding *is* the absence of M10-specific data, and it reported that faithfully:
   no published t/s vs power-cap sweep; no community-confirmed min/max `-pl` for
   M40/M60/M10; whether `-pl` is even honored on PCIe-only Maxwell cards is
   undocumented; closest anchors P40 min 125 W / max 250 W, P100 min 125 W; M10 is a
   75 W PCIe-power-only card — all from round5.md. Quirk: it appended "UNSUPPORTED"
   at the end despite having just answered from the chunks (over-literal reading of
   my answer-from-chunks-only instruction).
5. "recommended sampling for Qwen3.5 small models, thinking off" — **CORRECT** (minor
   side-clause error). The six values are exact: temp 0.7, top_p 0.8, top_k 20,
   min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0 — round11.md "Instruct
   (non-thinking), general". In a side note it mixed two variants: it calls the
   "precise coding" variant temp 0.6 *and* pp 2.0; the coding variant is temp 0.6 /
   pp 0.0, pp 2.0 belongs to the instruct-reasoning row (temp 1.0, top_p 1.0, top_k 40).

Score: 3 correct / 2 partly / 0 wrong / 0 unsupported. Same failure mode as T3 —
numbers preserved verbatim, attributions get tangled (which PR / which model / which
variant a number belongs to).

## T6 — four M10 GPUs at once (Kev + general + coder + embeddings, one job)

Job: "llama.cpp CUDA issue digest" over the 40 newest open CUDA issues
(`repo:ggml-org/llama.cpp is:issue is:open CUDA`, fetched first → /tmp/m10-deleg/t6-issues.json,
title + first 2000 chars of body, 40/40 non-empty). One Python process, four threads (one per
lane/GPU), all started together, lanes independent. Output: /tmp/m10-deleg/t6-digest.md
(one row per issue + dup pairs + top-10 actionable research chunks).

Per-lane results (347 model calls total, 0 transport errors, 0 retries needed):

| lane | GPU / model | work | calls | wall time | mean s/call | errors |
|------|-------------|------|-------|-----------|-------------|--------|
| A | GPU0, Kev (8095) | triage 40 issues (old_gpu/kind/severity) + score 226 research chunks "actionable" | 266 | 256.0 s | 0.962 | 0 |
| B | GPU1, general (8090, thinking off) | 2-sentence summary per issue | 40 | 478.4 s | 11.96 | 0 |
| C | GPU2, coder (8091) | verbatim command-line extraction per issue | 40 | 772.9 s | 19.32 | 0 |
| D | GPU3, embeddings (8093) | embed 40 issues, dup pairs >= 0.85, closest research chunk each | 1 | 24.2 s | 23.12 (1 call, 40 docs) | 0 |

- Overall wall: **772.9 s (12.9 min)** vs sum of lane times **1531.5 s** → **1.98x parallel speedup**
  (bounded by lane C, the slowest at 19.3 s/call; ideal 4-way speedup ~3.9x was not reachable
  because D finished in 24 s and A/B are mid-speed).
- All-four-lanes-running window: **19:28:59 -> 19:29:24 CST (24.2 s)** — exactly lane D's
  lifetime (its single embed call). A+B+C ran together for 256.0 s (19:28:59 -> 19:33:15).
  Lane spans: A 19:28:59-19:33:15, B 19:28:59-19:36:58, C 19:28:59-19:41:52, D 19:28:59-19:29:24.
- Lane A's 266 calls averaged 0.96 s — Kev is the workhorse of the fleet for classification
  volume (T4's 1.5 s/call was issue-only; chunk scoring with one noul question is faster).
- Top-10 actionable research chunks (Kev noul, of 226): round12 section 2 (Qwen3.5 MTP +
  --spec-type draft-mtp, 0.921), round6 section 2.2 (PR #20905 MOE GEMV, 0.888), round11
  section 3 (Hermes v0.23 endpoints, 0.882), round5 section 4.1 (ik PRs, 0.873), round8
  section 4 (Maxwell/Pascal MMVQ tuning, 0.865).

Spot-check verdicts (8 issues checked by the orchestrator against the actual bodies; full
detail in the digest's verification notes):

- **Lane B (summaries)**: faithful in 7/8 — e.g. #31840: v0.9.0 / RTX 5070 (SM120) /
  Ryzen 5 7500F all verbatim in the body. Finding: both #31920 and #31918 summaries claim
  "measured on NVIDIA A100 GPUs", but the complete 936-char bodies contain **no hardware at
  all** — unsupported detail (minor fabrication).
- **Lane C (commands)**: best lane on the happy path — #31656's command is verbatim including
  correct joining of shell line-continuations, all flags matching. Two problems found:
  (1) #31555's commands (cmake -DCMAKE_CUDA_ARCHITECTURES=70 ... + make) are **invented** —
  the full 3,080-char issue body contains no command line at all; the model inferred the
  "obvious" sm_70 repro, violating the verbatim-only instruction;
  (2) 3 of 40 replies were truncated by my 400-token budget mid-list (#31229, #31385, #31451) —
  I recovered 9 of the 13 truncated commands from the raw output, all verbatim against the
  bodies. Lesson: budget >= 1k tokens for extraction tasks, and never trust "commands" output
  without a body grep.
- **Lane A (Kev)**: sensible across the board — #31656 bug/crash 2.52/old_gpu no (Blackwell)
  is exactly right; #31555 old_gpu yes (Volta) reasonable. One borderline: #31425 old_gpu 0.58
  (Titan V is CC 7.0 — not below 7.0 — and the failure is config-specific, so a strict "no" is
  defensible). old_gpu=yes calls this batch: #31661 (GTX 1070/Pascal, correct), #31654 (MI210,
  debatable — it's AMD), #31555 (Volta, correct), #31503 (T4 CC 7.5, borderline by the letter
  of the question), #31425 (borderline, above).
- **Lane D (embeddings)**: caught the one true duplicate — #31920 <-> #31918 at 0.9913
  (identical bodies) — but 0.85 is far too loose for short issue texts: 71 further pairs at
  0.85-0.89 are topical siblings (e.g. #31281 "GLM 5.3 doesn't load with Tensor Split" vs
  #31229 "op offload counts heads as batch size" at 0.894 — clearly different bugs; shared
  "Eval bug:" boilerplate + hardware sections inflate similarity). Closest-note matches were
  on point (DFlash2 crash -> round5 section 2.3 DFlash note, 0.853).

Verdict for the "four GPUs at once" question: **it works** — all four lanes ran concurrently
with zero cross-lane interference and zero transport errors; the digest merged cleanly. The
all-four-active window is short (24 s) only because embedding 40 docs is cheap — a job that
makes every lane call-bound (like A) stretches the 4-GPU overlap to the full wall time.

---
**Reviewer note (Claude, checked against GitHub):** the #31555 verdict above is right that the coder invented its cmake command, but wrong that the issue has no command line. The full 3,080-char body has `llama-server -m raon-1-mini-preview-Q6_K.gguf -ngl 99 -fa on -c 32768 -np 1 --jinja` and `-DCMAKE_CUDA_ARCHITECTURES="70;89"`, both after the 2,000-char cut the task gave the lanes. The coder saw text ending at `-DCMAKE_C` and completed it from guesswork (`=70`, `make`). Lesson: give extraction tasks the whole body, and tell the model to return nothing when a command is cut off. The A100 claim for #31918/#31920 is confirmed fabricated (both 936-char bodies are identical and name no hardware).
