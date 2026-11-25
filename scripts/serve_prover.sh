#!/usr/bin/env bash
# Serve DeepSeek-Prover-V1.5 with vLLM's OpenAI-compatible server ON the EC2 g6.xlarge,
# after scripts/setup_ec2.sh. bf16, context 4096; a 7B model in bf16 needs about 14 GB
# of the L4's 24 GB, the rest goes to the KV cache.
#
#   bash scripts/serve_prover.sh            # background; logs to ~/vllm.log
#   FOREGROUND=1 bash scripts/serve_prover.sh
#
# By default it listens on 127.0.0.1 only: run the search on the same instance, or reach it
# from your laptop with an SSH tunnel (printed below). To listen on all interfaces set
# HOST=0.0.0.0 and VLLM_API_KEY, and open port 8000 in the security group to your IP only.
# It never calls the AWS API and never creates, starts or stops instances.
set -euo pipefail
VLLM_ENV=${VLLM_ENV:-$HOME/vllm-env}
MODEL=${MODEL:-deepseek-ai/DeepSeek-Prover-V1.5-RL}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.90}
LOG=${LOG:-$HOME/vllm.log}

if [ "$HOST" != "127.0.0.1" ] && [ -z "${VLLM_API_KEY:-}" ]; then
  echo "HOST=$HOST exposes the server; set VLLM_API_KEY too." >&2
  exit 1
fi

ARGS=(serve "$MODEL" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_UTIL" --host "$HOST" --port "$PORT")
[ -n "${VLLM_API_KEY:-}" ] && ARGS+=(--api-key "$VLLM_API_KEY")

if [ -n "${FOREGROUND:-}" ]; then
  exec "$VLLM_ENV/bin/vllm" "${ARGS[@]}"
fi

nohup "$VLLM_ENV/bin/vllm" "${ARGS[@]}" >"$LOG" 2>&1 &
PID=$!
echo "vLLM starting (pid $PID, log $LOG). The first start downloads about 14 GB of weights."
AUTH=()
[ -n "${VLLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")
for _ in $(seq 1 180); do
  if curl -sf "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "vLLM exited; last lines of $LOG:" >&2; tail -30 "$LOG" >&2; exit 1
  fi
  sleep 10
done
curl -sf "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM not up after 30 min; see $LOG" >&2; exit 1; }

# Public address from instance metadata (IMDSv2), for the tunnel hint only.
TOKEN=$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
PUBLIC=$( [ -n "$TOKEN" ] && curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)

echo
echo "Prover endpoint: http://$HOST:$PORT  (model $MODEL, pid $PID)"
echo "On this instance:  .venv/bin/python -m markov_lean.search --endpoint http://127.0.0.1:$PORT --limit 5"
if [ -n "$PUBLIC" ]; then
  echo "From your laptop:  ssh -N -L $PORT:127.0.0.1:$PORT ubuntu@$PUBLIC   then use http://127.0.0.1:$PORT"
fi
echo "Stop it:           kill $PID"
