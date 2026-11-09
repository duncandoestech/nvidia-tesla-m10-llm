#!/usr/bin/env python3
"""OCR eval on the synthetic set: character similarity, exact match, seconds per image. Usage: ocr.py <port> <tag> [--no-think]"""
import base64, difflib, json, sys, time, urllib.request
port, tag = sys.argv[1], sys.argv[2]; no_think = "--no-think" in sys.argv
import os
D = "/data/m10/evals/data/ocr"; truth = json.load(open(f"{D}/truth.json"))[:int(os.environ.get("OCR_N", "24"))]
norm = lambda s: "\n".join(" ".join(l.split()) for l in s.strip().strip("`").splitlines() if l.strip())
sims, exact, secs, rows = [], 0, [], []
for t in truth:
    b64 = base64.b64encode(open(f"{D}/{t['file']}", "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Transcribe all text in this image exactly, keeping line breaks. Output only the text."}]}],
        "temperature": 0, "max_tokens": 400}
    if no_think: body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    secs.append(time.time() - t0)
    got = norm(r["choices"][0]["message"]["content"] or ""); want = norm(t["text"])
    s = difflib.SequenceMatcher(None, got, want).ratio(); sims.append(s); exact += got == want
    rows.append({"file": t["file"], "sim": round(s, 4), "exact": got == want, "got": got})
res = {"tag": tag, "char_similarity": sum(sims) / len(sims), "exact": exact, "n": len(truth), "sec_per_image": sum(secs) / len(secs),
       "by_effect": {e: round(sum(r["sim"] for r in rows if e in r["file"]) / max(1, sum(e in r["file"] for r in rows)), 4) for e in ["clean", "blur", "rotate", "noise"]}}
print(json.dumps(res)); json.dump({**res, "rows": rows}, open(f"/data/m10/results/ocr-{tag}.json", "w"))
