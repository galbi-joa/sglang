# MLA attention kernel profiling: FlashMLA vs trtllm-gen (CP scenario)

Goal: turn this claim — *"on B300, FlashMLA is much slower than
trtllm-gen in the CP scenario"* — into measured numbers, split cleanly by
phase (prefill vs decode).

## The key insight that shapes this benchmark

CP (Context Parallel) is **not** a kernel feature. The latent-KV all-gather
that *is* CP lives at the model level (`deepseek_v2.py: rebuild_cp_kv_cache`
→ `cp_all_gather_rerange_output`), invoked from the absorbed-MLA prepare in
`forward_mla.py:384`. Once CP has reassembled the full KV, what runs is an
ordinary MLA attention kernel that has no idea CP happened.

So "slow under CP" really means **"the attention kernel that runs in a
CP deployment is slow"**, and the fair test is kernel-vs-kernel:

| phase | FlashMLA backend runs | trtllm_mla backend runs |
|-------|-----------------------|-------------------------|
| decode | `sgl_kernel.flash_mla.flash_mla_with_kvcache` (DeepSeek FlashMLA) | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (backend="trtllm-gen") |
| prefill (absorbed-MLA branch) | `flash_mla_with_kvcache` — the **same decode kernel**, q_len>1 | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` |
| prefill (pure ragged branch) | falls back to `flashinfer` ragged (`super().forward_extend`) | `flashinfer.prefill.trtllm_ragged_attention_deepseek` |

Two facts worth keeping in front of you while reading results:

1. **Architecture gating.** `sgl-kernel/cmake/flashmla.cmake` builds FlashMLA
   **dense decode only for sm90 (Hopper)** (`sm90/decode/dense/instantiations/
   {fp16,bf16}.cu`); the sm100 list is **sparse-only**. The trtllm_mla unit
   test gates on `_REQUIRED_MAJOR = 12` (Blackwell). So on B300 (SM10x) FlashMLA
   dense decode may have no native fast path while trtllm-gen does — this is the
   #1 hypothesis for the slowdown.
2. **Prefill is NOT just flashinfer-vs-flashinfer.** SGLang reuses the FlashMLA
   *decode* kernel for prefill whenever dispatch returns
   `AttnForwardMethod.MLA` (piecewise CUDA graph, prefill-CP,
   `flashinfer_mla_disable_ragged`, or a prefix with chunked-prefix-cache
   disabled). So absorbed-MLA prefill lands on the very same SM90-gated FlashMLA
   decode kernel. Only *pure ragged* prefill defers to the flashinfer ragged
   wrapper. The two regimes are benchmarked separately as `prefill_absorbed`
   and `prefill_ragged`. (See benchmarkno1.md sections 3, 5b.)

## Files

- `bench_mla_kernels.py` — microbenchmark. Times each kernel in isolation with
  CUDA events + L2 flush, cross-checks outputs (cosine diff) so you know the
  comparison is valid, and can emit a Chrome trace. **Start here.**
- `profile_e2e_cp.sh` — launches a real DeepSeek server with
  `--enable-prefill-context-parallel`, runs one long-context request under the
  torch profiler, and tells you which kernels showed up per phase. Use this to
  confirm the microbench reflects the real CP deployment.

## Quick start (on the B300 box)

```bash
cd benchmark/mla_cp_kernel_profile

# 1) Decode sweep long KV, q_len=1
python bench_mla_kernels.py --mode decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --dtype bf16

# 2) Absorbed-MLA prefill: SGLang reuses the FlashMLA *decode* kernel for
#    prefill (q_len>1) whenever dispatch picks AttnForwardMethod.MLA.
python bench_mla_kernels.py --mode prefill_absorbed \
    --batch 1 --seq-q 2048 8192 16384 --heads 128 --dtype bf16

# 3) Pure ragged prefill (FlashMLA's fallback) vs trtllm-gen ragged
python bench_mla_kernels.py --mode prefill_ragged \
    --batch 1 --seq-q 2048 8192 16384 --heads 128 --dtype bf16

# 4) Dump a chrome trace of the long-context decode case
python bench_mla_kernels.py --mode decode --seq-k 32768 --batch 1 \
    --trace trace_decode_32k.json
#    -> feed trace_decode_32k.json to the `llm-torch-profiler-analysis` skill

# 5) Run everything at once
python bench_mla_kernels.py --mode all \
    --batch 1 --seq-k 4096 16384 32768 --seq-q 2048 8192 --heads 128

# 6) End-to-end with CP actually enabled, per backend
BACKEND=flashmla    bash profile_e2e_cp.sh
BACKEND=trtllm_mla  bash profile_e2e_cp.sh
```

`--heads` is q-heads after TP: 128 for TP=1, 16 for TP=8, etc. Pick the value
matching the deployment you're comparing.

## How to read it

- **`speedup` column** = `flashmla_ms / trtllm_ms`. >1 means FlashMLA is slower
  (confirms the claim); the magnitude per shape tells you *where* it's worst
  (likely large `seq_k`).
- **`cos_diff`** should be ~1e-3 or smaller. If it's large, the two kernels
  aren't computing the same thing for that shape (e.g. one fell back / errored)
  and the timing comparison for that row is meaningless — investigate before
  trusting the number.
- A row printing `n/a` means that backend's kernel was unavailable or raised on
  this device/shape — itself a finding (e.g. FlashMLA has no path for B300).

## Caveats

- bf16 path uses `bmm1_scale = softmax_scale` (q/k descale = 1). For an FP8-KV
  comparison you'd add per-tensor descales and the `mla_quantize_and_rope_for_fp8`
  preprocessing the trtllm backend uses; left out here to keep the first cut
  apples-to-apples in bf16.
- flashinfer API kwargs (`trtllm_ragged_attention_deepseek`,
  `BatchPrefillWithRaggedKVCacheWrapper.plan`) track the pinned
  `flashinfer==0.6.11.post1`. If you bump flashinfer, re-check signatures.
