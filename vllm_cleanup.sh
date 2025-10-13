#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-}"          # optional; if set we’ll scope by port first
RETRIES="${RETRIES:-5}"   # final verification retries
SLEEP_SECS="${SLEEP_SECS:-1}"

pids=()

# 1) Prefer killing the process that OWNS the listening socket on $PORT (if provided)
if [ -n "${PORT}" ]; then
  if command -v ss >/dev/null; then
    while read -r line; do
      # extract pid=NNN from ss output
      pid=$(sed -n 's/.*pid=\([0-9]\+\).*/\1/p' <<<"$line")
      [ -n "${pid:-}" ] && pids+=("$pid")
    done < <(ss -ltnp "( sport = :$PORT )" 2>/dev/null | awk 'NR>1 {print $0}')
  fi
  if [ "${#pids[@]}" -eq 0 ] && command -v lsof >/dev/null; then
    while read -r pid; do pids+=("$pid"); done < <(lsof -t -iTCP:"$PORT" -sTCP:LISTEN -nP 2>/dev/null || true)
  fi
fi

# 2) Also find *any* vLLM "serve" trees for this user, regardless of port
#    This catches cases where port changed or vLLM is mid-initialization.
mapfile -t serve_pids < <(pgrep -u "$USER" -f '(^|[/ ])vllm( |$).*serve' || true)
if [ "${#serve_pids[@]}" -gt 0 ]; then
  pids+=("${serve_pids[@]}")
fi

# De-dup PIDs
if [ "${#pids[@]}" -gt 0 ]; then
  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u)
fi

if [ "${#pids[@]}" -eq 0 ]; then
  echo "[vllm_cleanup] No vLLM PIDs found for user $USER (port='${PORT:-unset}'). Nothing to do."
else
  # echo "[vllm_cleanup] Target PIDs: ${pids[*]}"
  # Map to unique PGIDs
  mapfile -t pgids < <(ps -o pgid= -p "${pids[@]}" | tr -d ' ' | sort -u)
  # echo "[vllm_cleanup] Target PGIDs: ${pgids[*]}"

  # TERM whole groups first
  for g in "${pgids[@]}"; do
    # echo "[vllm_cleanup] TERM process group PGID=$g"
    kill -TERM -"$g" 2>/dev/null || true
  done

  sleep 2

  # Any survivors? KILL the groups
  for g in "${pgids[@]}"; do
    if ps -o pgid= -p "${pids[@]}" 2>/dev/null | grep -q "$g"; then
      # echo "[vllm_cleanup] KILL process group PGID=$g (still present)"
      kill -KILL -"$g" 2>/dev/null || true
    fi
  done
fi

# Verify (brief retries)
for i in $(seq 1 "$RETRIES"); do
  alive=false
  if [ -n "${PORT}" ]; then
    if ss -ltnp "( sport = :$PORT )" 2>/dev/null | awk 'NR>1 {exit 0} END{exit 1}'; then
      alive=true
    fi
  fi
  if pgrep -u "$USER" -f '(^|[/ ])vllm( |$).*serve' >/dev/null 2>&1; then
    alive=true
  fi
  $alive || break
  sleep "$SLEEP_SECS"
done

echo "[vllm_cleanup] Clean up done."