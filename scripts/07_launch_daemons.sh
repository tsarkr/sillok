#!/usr/bin/env bash
# Launch and maintain detached Sillok-LM daemons (Artifact HTTP Server on :8000 + Pretraining)
# Survives VS Code Remote-SSH disconnection (uses nohup + setsid).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs data/processed artifacts/tokenizer checkpoints

PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi
export PYTHONPATH="$ROOT_DIR/src"

# 1. Start read-only Artifact & Telemetry HTTP Server on 127.0.0.1:8000 (for cloudflared tunnel)
if [[ -f logs/artifact_server.pid ]] && kill -0 "$(cat logs/artifact_server.pid)" 2>/dev/null; then
    echo "[daemon] Artifact server already running (PID=$(cat logs/artifact_server.pid))"
else
    nohup setsid "$PYTHON_BIN" -u scripts/06_serve_artifacts.py \
        --host 127.0.0.1 \
        --port 8000 \
        > logs/artifact_server.log 2>&1 < /dev/null &
    echo $! > logs/artifact_server.pid
    echo "[daemon] Started Artifact HTTP server on 127.0.0.1:8000 (PID=$(cat logs/artifact_server.pid))"
fi

# 2. Launch detached autonomous training pipeline if requested
CONFIG_PATH="${1:-configs/sillok_tiny_debug.yaml}"
MAX_STEPS="${2:-500}"

if [[ -f logs/pretrain_daemon.pid ]] && kill -0 "$(cat logs/pretrain_daemon.pid)" 2>/dev/null; then
    echo "[daemon] Pretraining daemon already running (PID=$(cat logs/pretrain_daemon.pid))"
else
    nohup setsid "$PYTHON_BIN" -u scripts/04_pretrain.py \
        --config "$CONFIG_PATH" \
        --max-steps "$MAX_STEPS" \
        --no-compile \
        >> logs/pretrain_daemon.log 2>&1 < /dev/null &
    echo $! > logs/pretrain_daemon.pid
    echo "[daemon] Started autonomous pretraining daemon (PID=$(cat logs/pretrain_daemon.pid), config=$CONFIG_PATH, max_steps=$MAX_STEPS)"
fi
