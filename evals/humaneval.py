#!/usr/bin/env python3
"""HumanEval pass@1 (greedy) against an OpenAI-compatible llama-server.
Generated code runs in a throwaway container: no network, read-only, 512 MB, 1 CPU, 10 s timeout.
Usage: humaneval.py <port> <tag> [n_problems] [--no-think]"""
import gzip, json, re, subprocess, sys, time, urllib.request
port, tag = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 164
no_think = "--no-think" in sys.argv
probs = [json.loads(l) for l in gzip.open("/data/m10/evals/data/HumanEval.jsonl.gz", "rt")][:n]
SANDBOX = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--memory", "512m", "--cpus", "1",
           "--entrypoint", "timeout", "m10-pyeval:latest", "10", "python3", "-"]
def ask(prompt):
    body = {"messages": [{"role": "system", "content": "You are an expert Python programmer."},
                         {"role": "user", "content": "Complete the following Python function. Reply with the complete function, including the signature, in one ```python code block.\n\n" + prompt}],
            "temperature": 0, "max_tokens": int(__import__("os").environ.get("HE_MAXTOK", "768"))}
    if no_think: body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"]["content"] or "", r.get("timings", {})
passed = 0; toks = 0; ms = 0; rows = []
t0 = time.time()
for p in probs:
    text, tm = ask(p["prompt"])
    toks += tm.get("predicted_n", 0); ms += tm.get("predicted_ms", 0)
    m = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    imports = "\n".join(l for l in p["prompt"].splitlines() if l.startswith(("import ", "from ")))
    if f"def {p['entry_point']}" not in code: code = p["prompt"] + code
    prog = f"{imports}\n{code}\n\n{p['test']}\n\ncheck({p['entry_point']})\n"
    ok = subprocess.run(SANDBOX, input=prog.encode(), capture_output=True).returncode == 0
    passed += ok; rows.append({"task_id": p["task_id"], "passed": ok})
print(f"{tag}: pass@1 {passed}/{len(probs)} = {100*passed/len(probs):.1f}%  gen {1000*toks/max(ms,1):.1f} t/s  wall {time.time()-t0:.0f}s")
json.dump({"tag": tag, "passed": passed, "n": len(probs), "gen_tps": 1000*toks/max(ms,1), "rows": rows}, open(f"/data/m10/results/he-{tag}.json", "w"))
