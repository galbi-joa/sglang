#!/usr/bin/env bash
# End-to-end CP profiling: launch a DeepSeek MLA server with context-parallel
# prefill enabled, capture a torch profiler trace, and isolate the attention
# kernels per phase (prefill vs decode).
#
# WHY e2e *in addition* to the microbench:
#   The microbench (bench_mla_kernels.py) times the kernels in isolation, which
#   is the cleanest apples-to-apples comparison. But the mentor's claim is about
#   a *CP deployment*. CP is wired only on the model side (deepseek_v2.py latent
#   all-gather) and -- per attention_backend_handler.py -- only fa3 dispatches
#   the CP-prefill path today. This script lets you confirm, with CP actually
#   on, which backend each phase lands on and where the time goes.
#
# Run on the B300 box. Requires a DeepSeek MLA checkpoint reachable by --model.
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V3}"
TP="${TP:-8}"
DP="${DP:-2}"                       # MLA CP derives attn_cp_size = TP // DP
BACKEND="${BACKEND:-flashmla}"      # flashmla | trtllm_mla | fa3
PORT="${PORT:-30000}"
OUT="${OUT:-./e2e_cp_${BACKEND}}"
IN_LEN="${IN_LEN:-32768}"           # long context => exercises CP + long decode
OUT_LEN="${OUT_LEN:-512}"
mkdir -p "$OUT"

echo ">>> backend=$BACKEND  TP=$TP DP=$DP  in=$IN_LEN out=$OUT_LEN  model=$MODEL"

# 1) launch server with CP prefill on. SGLANG_TORCH_PROFILER_DIR makes the
#    /start_profile endpoint dump traces there.
SGLANG_TORCH_PROFILER_DIR="$OUT" \
python -m sglang.launch_server \
  --model "$MODEL" \
  --trust-remote-code \
  --tp "$TP" --dp "$DP" --enable-dp-attention \
  --attention-backend "$BACKEND" \
  --enable-prefill-context-parallel \
  --context-length 40960 \
  --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# 2) wait for health
echo ">>> waiting for server..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo ">>> server up"; break
  fi
  sleep 5
done

# 3) profiled single-batch run (one request keeps phases cleanly separable;
#    MLA CP requires batch_size==1 anyway per server_args.py:1933).
curl -sf "http://127.0.0.1:${PORT}/start_profile" >/dev/null
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random \
  --random-input-len "$IN_LEN" \
  --random-output-len "$OUT_LEN" \
  --num-prompts 1 --max-concurrency 1 \
  2>&1 | tee "$OUT/bench_serving.log"
curl -sf "http://127.0.0.1:${PORT}/stop_profile" >/dev/null

echo ">>> trace(s) written under: $OUT"
echo ">>> next: hand the trace to the llm-torch-profiler-analysis skill, or grep"
echo "    the kernel names: flash_fwd / fwd_kvcache_mla (flashmla) vs"
echo "    trtllm_* / xqa (trtllm-gen) vs BatchPrefillRagged (flashinfer prefill)."
