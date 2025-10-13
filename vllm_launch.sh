#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/scr/zyanzhe/vllm}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm.out}"
MODEL_KEY="${1:-Qwen/Qwen3-32B-AWQ}"
DATA_PARALLEL_SIZE="${2:-1}"

mkdir -p "$LOG_DIR"

# Helper: launch one model
_launch() {
  # New session/process group so cleanup can nuke group safely later
  setsid sh -c "$1" >>"$LOG_FILE" 2>&1 &
}

(
  cd /nlp/scr/zyanzhe/vllm || exit 1

  case "$MODEL_KEY" in
    Qwen/Qwen3-32B-AWQ-Thinking|Qwen/Qwen3-32B-AWQ)
      _launch "VLLM_USE_V1=1 HF_HOME=/scr/zyanzhe VLLM_CACHE_ROOT=/scr/zyanzhe/vllm \
        vllm serve Qwen/Qwen3-32B-AWQ \
          --port $PORT \
          --enable-auto-tool-choice \
          --tool-call-parser hermes \
          --gpu-memory-utilization 0.95 \
          --max-num-seqs 1 \
          --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
          --max-model-len 100000 \
          --reasoning-parser qwen3 \
          --data-parallel-size $DATA_PARALLEL_SIZE"
      ;;
    Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)
      _launch "VLLM_USE_V1=1 HF_HOME=/scr/zyanzhe VLLM_CACHE_ROOT=/scr/zyanzhe/vllm \
        vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
          --port $PORT \
          --enable-auto-tool-choice \
          --tool-call-parser hermes \
          --gpu-memory-utilization 0.95 \
          --max-num-seqs 1 \
          --max-model-len 100000 \
          --reasoning-parser qwen3 \
          --data-parallel-size $DATA_PARALLEL_SIZE"
      ;;
    Qwen/Qwen3-30B-A3B-Thinking-2507-FP8)
      _launch "VLLM_USE_V1=1 HF_HOME=/scr/zyanzhe VLLM_CACHE_ROOT=/scr/zyanzhe/vllm \
        vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
          --port $PORT \
          --enable-auto-tool-choice \
          --tool-call-parser hermes \
          --gpu-memory-utilization 0.95 \
          --max-num-seqs 1 \
          --max-model-len 100000 \
          --reasoning-parser qwen3 \
          --data-parallel-size $DATA_PARALLEL_SIZE"
      ;;
    openai/gpt-oss-20b)
      _launch "VLLM_USE_V1=1 HF_HOME=/scr/zyanzhe VLLM_CACHE_ROOT=/scr/zyanzhe/vllm \
        vllm serve openai/gpt-oss-20b \
          --port $PORT \
          --gpu-memory-utilization 0.9 \
          --max-num-seqs 8 \
          --max-model-len 100000 \
          --tool-call-parser openai \
          --reasoning-parser openai_gptoss \
          --enable-auto-tool-choice"
      ;;
    openai/gpt-oss-120b)
      _launch "VLLM_USE_V1=1 HF_HOME=/juice2/scr2/nlp/pix2code/huggingface VLLM_CACHE_ROOT=/juice2/scr2/nlp/pix2code/vllm \
        vllm serve openai/gpt-oss-120b \
          --port $PORT \
          --gpu-memory-utilization 0.9 \
          --max-num-seqs 5 \
          --max-model-len 100000 \
          --tool-call-parser openai \
          --reasoning-parser openai_gptoss \
          --enable-auto-tool-choice"
      ;;
    *)
      echo "Unknown model key: $MODEL_KEY" >&2
      exit 2
      ;;
  esac
)

echo "[vllm_launch] Launched $MODEL_KEY on port $PORT, logging to $LOG_FILE"
