#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
TRIES="${TRIES:-40}"
SLEEP_SECS="${SLEEP_SECS:-30}"

echo "[vllm_wait] Waiting for vLLM on :$PORT ..."
for i in $(seq 1 "$TRIES"); do
  if curl -sSf "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1 || \
     curl -sSf "http://127.0.0.1:${PORT}/health"    >/dev/null 2>&1; then
    echo "[vllm_wait] vLLM is up."
    exit 0
  fi
  sleep "$SLEEP_SECS"
done

echo "[vllm_wait] Timed out waiting for vLLM on :$PORT" >&2
exit 1
