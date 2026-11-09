#!/usr/bin/env python3
"""BEIR SciFact retrieval eval for a llama-server embedding model (+ optional reranker).
Usage: retrieval.py <embed_port> <tag> <style:qwen3|gemma> [rerank_port]"""
import json, math, os, sys, time, urllib.request
import numpy as np
port, tag, style = sys.argv[1], sys.argv[2], sys.argv[3]
rport = sys.argv[4] if len(sys.argv) > 4 else None
D = os.environ.get("SCIFACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scifact"))
if not os.path.exists(f"{D}/corpus.jsonl"):
    sys.exit(f"SciFact not found in {D}: run evals/get_scifact.sh first (or set SCIFACT_DIR)")
corpus = [json.loads(l) for l in open(f"{D}/corpus.jsonl")]
queries = {q["_id"]: q["text"] for q in map(json.loads, open(f"{D}/queries.jsonl"))}
qrels = {}
for l in list(open(f"{D}/qrels/test.tsv"))[1:]:
    qid, did, s = l.split("\t"); qrels.setdefault(qid, {})[did] = int(s)
qids = sorted(qrels)
def post(p, path, body):
    req = urllib.request.Request(f"http://localhost:{p}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))
def embed(texts):
    out = []
    for i in range(0, len(texts), 16):
        r = post(port, "/v1/embeddings", {"input": [t[:2000] for t in texts[i:i+16]]})
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    v = np.array(out, dtype=np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True)
if style == "qwen3":
    dtext = [(d.get("title", "") + "\n" + d["text"]).strip() for d in corpus]
    qtext = ["Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: " + queries[q] for q in qids]
else:
    dtext = ["title: " + (d.get("title") or "none") + " | text: " + d["text"] for d in corpus]
    qtext = ["task: search result | query: " + queries[q] for q in qids]
import os
cache = f"/data/m10/results/emb-cache-{tag.split('+')[0]}.npy"
t0 = time.time()
if os.path.exists(cache): DV = np.load(cache); t_doc = float("nan")
else: DV = embed(dtext); t_doc = time.time() - t0; np.save(cache, DV)
QV = embed(qtext)
S = QV @ DV.T
def ndcg10(ranked, rel):
    dcg = sum((rel.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    return dcg / sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
ids = [d["_id"] for d in corpus]
nd, rc, nd_rr = [], [], []
t_rr = 0
for qi, q in enumerate(qids):
    order = np.argsort(-S[qi]); ranked = [ids[j] for j in order[:100]]
    nd.append(ndcg10(ranked, qrels[q])); rc.append(len(set(ranked) & set(qrels[q])) / len(qrels[q]))
    if rport:
        top = [ids[j] for j in order[:20]]; docs = [dtext[j][:2000] for j in order[:20]]
        t1 = time.time(); r = post(rport, "/v1/rerank", {"query": queries[q], "documents": docs, "top_n": 20}); t_rr += time.time() - t1
        rr = [top[x["index"]] for x in sorted(r["results"], key=lambda x: -x["relevance_score"])]
        nd_rr.append(ndcg10(rr + ranked[20:], qrels[q]))
res = {"tag": tag, "ndcg10": float(np.mean(nd)), "recall100": float(np.mean(rc)), "docs_per_s": len(dtext) / t_doc}
if rport: res.update({"ndcg10_reranked": float(np.mean(nd_rr)), "rerank_s_per_query": t_rr / len(qids)})
print(json.dumps(res)); json.dump(res, open(f"/data/m10/results/ret-{tag}.json", "w"))
