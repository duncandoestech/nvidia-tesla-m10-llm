#!/usr/bin/env python3
"""Check the HumanEval sandbox: reference solutions pass, a failing program fails, network is blocked."""
import gzip, json, subprocess
SANDBOX = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--memory", "512m", "--cpus", "1",
           "--entrypoint", "timeout", "m10-pyeval:latest", "10", "python3", "-"]
run = lambda src: subprocess.run(SANDBOX, input=src.encode(), capture_output=True).returncode
ps = [json.loads(l) for l in gzip.open("/data/m10/evals/data/HumanEval.jsonl.gz", "rt")][:10]
ok = sum(run(p["prompt"] + p["canonical_solution"] + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n") == 0 for p in ps)
bad = run("assert 1 == 2\n")
net = run("import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)\n")
print(f"canonical {ok}/10 pass; failing program rc={bad} (expect nonzero); network attempt rc={net} (expect nonzero)")
