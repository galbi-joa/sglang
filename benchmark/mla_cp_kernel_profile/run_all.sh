#!/usr/bin/env bash
# Full MLA kernel benchmark sweep on B300 (FlashMLA vs trtllm-gen).
#
# Runs every regime and tees each into results/. Designed for B300 (SM10.3)
# after rebuilding sgl_kernel with CUDA 13 (see README "Running on B300").
#
# Usage:
#   bash run_all.sh                      # default sweep
#   HEADS=128 BATCH="1 4" bash run_all.sh
set -uo pipefail   # NOT -e: a failing regime should not abort the whole sweep

HEADS="${HEADS:-128}"
BATCH="${BATCH:-1 4 16}"
SEQ_K="${SEQ_K:-4096 16384 32768}"
SEQ_Q="${SEQ_Q:-2048 8192}"
OUT="${OUT:-results}"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)

run() {
  local name="$1"; shift
  echo ""
  echo "############################################################"
  echo "# $name"
  echo "# python bench_mla_kernels.py $*"
  echo "############################################################"
  python bench_mla_kernels.py "$@" 2>&1 | tee "$OUT/${STAMP}_${name}.log"
}

echo "=== Full MLA kernel sweep  ($STAMP) ==="
python -c "import torch; print('device:', torch.cuda.get_device_name(0), \
'SM', '.'.join(map(str, torch.cuda.get_device_capability(0))), \
'torch', torch.__version__)"

# 1) decode, bf16 -- documents the missing FlashMLA dense-decode kernel on B300
#    (flashmla n/a, trtllm runs).
run "decode_bf16" --mode decode --batch $BATCH --seq-k $SEQ_K --heads $HEADS --dtype bf16

# 2) decode, fp8 -- fp8 dense also absent on B300 (flashmla n/a).
run "decode_fp8" --mode decode --batch $BATCH --seq-k $SEQ_K --heads $HEADS --dtype fp8

# 3) absorbed-MLA prefill (FlashMLA decode kernel reused for prefill), bf16.
run "prefill_absorbed_bf16" --mode prefill_absorbed --batch 1 --seq-q $SEQ_Q --heads $HEADS --dtype bf16

# 4) pure ragged prefill fallback, bf16.
run "prefill_ragged_bf16" --mode prefill_ragged --batch 1 --seq-q $SEQ_Q --heads $HEADS --dtype bf16

# 5) sparse decode (DSA/V3.2) -- the FlashMLA decode path that DOES run on B300
#    after the CUDA-13 rebuild. FlashMLA sparse vs trtllm-gen sparse.
run "sparse_decode" --mode sparse_decode --batch $BATCH --seq-k $SEQ_K --heads $HEADS

echo ""
echo "=== Done. Per-regime logs under $OUT/${STAMP}_*.log ==="
echo "Summary of the headline numbers:"
grep -hE "^\s*[0-9]+\s+[0-9]+" "$OUT"/${STAMP}_*.log 2>/dev/null || true
