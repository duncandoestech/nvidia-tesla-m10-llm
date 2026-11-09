"""Decode speed via llama-server /completion: mtp_speed.py <port> <tag> [n_predict]"""
import json, sys, time, urllib.request
port, tag = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
prompts = ["Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and docstring.",
           "Explain how a B-tree index works in a relational database, step by step.",
           "List 20 practical tips for reducing household energy use, one sentence each.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle."]
out = []
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900)); t = r.get("timings", {})
    out.append({"tps": t.get("predicted_per_second"), "n": t.get("predicted_n"), "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")})
res = {"tag": tag, "mean_tps": sum(o["tps"] for o in out) / len(out), "runs": out}
print(json.dumps(res)); json.dump(res, open(f"/data/m10/results/mtp-{tag}.json", "w"))
