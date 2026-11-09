#!/usr/bin/env bash
# Download BEIR SciFact (CC BY-NC 2.0, not redistributed here) into evals/data/scifact for retrieval.py.
# Source: BEIR (https://github.com/beir-cellar/beir), the same files the published results used.
set -euo pipefail
URL=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
SHA256=536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
DATA="$(cd "$(dirname "$0")" && pwd)/data"
[ -f "$DATA/scifact/corpus.jsonl" ] && { echo "SciFact already in $DATA/scifact"; exit 0; }
mkdir -p "$DATA"; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fL --retry 3 -o "$TMP/scifact.zip" "$URL"
echo "$SHA256  $TMP/scifact.zip" | sha256sum -c -
unzip -q "$TMP/scifact.zip" -d "$DATA"
echo "SciFact ready: $DATA/scifact ($(wc -l < "$DATA/scifact/corpus.jsonl") docs)"
