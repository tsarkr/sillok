#!/usr/bin/env bash
# Ensure Sillok-LM background daemons (artifact server + pretraining) are alive and print status.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/07_launch_daemons.sh" configs/sillok_tiny_debug.yaml 50000

echo "=== Active Processes ==="
ps -ef | grep -E 'cloudflared|06_serve_artifacts|04_pretrain' | grep -v grep || true

echo "=== Tunnel & Telemetry ==="
"$ROOT_DIR/.venv/bin/python3" -c '
import json, urllib.request
from sillok.remote import detect_runtime
info = detect_runtime()
print("Public Tunnel:", info.host_processes.cloudflared_public_url or "(check /home/nsua/tunnel.log)")
with urllib.request.urlopen("http://127.0.0.1:8000/status.json", timeout=5) as r:
    st = json.loads(r.read().decode())
print("Dataset tokens:", st.get("dataset_meta", {}).get("train_tokens"), "train /", st.get("dataset_meta", {}).get("val_tokens"), "val")
for line in st.get("recent_training_log", [])[-5:]:
    print(" ", line)
with urllib.request.urlopen("http://127.0.0.1:8000/job.json", timeout=5) as r:
    job = json.loads(r.read().decode())
print("Colab Job:", job.get("job_name"), "->", job.get("config"))
'
