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

# 5) Run everything at once, and save a CSV
python bench_mla_kernels.py --mode all \
    --batch 1 --seq-k 4096 16384 32768 --seq-q 2048 8192 --heads 128 \
    --csv results.csv

# 6) Full sweep helper: writes per-regime logs + CSVs and a combined CSV
bash run_all.sh                 # results/<stamp>_*.log, *.csv, *_combined.csv

# 7) End-to-end with CP actually enabled, per backend
BACKEND=flashmla    bash profile_e2e_cp.sh
BACKEND=trtllm_mla  bash profile_e2e_cp.sh
```

### CSV output (`--csv`)

Any run can append `--csv <path>` to dump every benchmarked shape as a table.
Columns: `mode, batch, s_q, s_kv, heads, dtype, a_backend, a_ms, a_note,
b_backend, b_ms, b_note, speedup_a_over_b, cos_diff`. A backend that had no
kernel for the shape leaves `*_ms` empty and records the reason in `*_note`
(e.g. `Dense decode MLA is only supported on SM90a`), so the CSV doubles as a
record of which paths exist on this device. `run_all.sh` writes one CSV per
regime plus a merged `<stamp>_combined.csv`.

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

## Running on B300 (Blackwell / SM 10.3) — required setup

Measured 2026-06-01. Two environment facts bite here:

1. **trtllm-gen needs CUDA toolkit 13.** flashinfer JIT-compiles the trtllm-gen
   kernel for `compute_103a`; nvcc 12.8 does not know SM103 and fails with
   `Unsupported gpu architecture 'compute_103a'`. The driver (580.x) already
   supports CUDA 13, so install the toolkit only:
   ```bash
   # Ubuntu 24.04, driver already CUDA-13-capable
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
   apt-get install -y cuda-toolkit-13-0     # NOT 'cuda' / 'cuda-13-0' (those touch the driver)
   export PATH=/usr/local/cuda-13.0/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
   rm -rf /root/.cache/flashinfer        # drop the failed 12.8 JIT cache
   ```

2. **FlashMLA needs sgl_kernel rebuilt from source with CUDA 13.** A pip
   prebuilt `sgl_kernel` wheel is typically built under CUDA <= 12.8, so it has
   **no sm100/sm103 binaries** and every flashmla call raises
   `no kernel image is available`. `sgl-kernel/cmake/flashmla.cmake` *does*
   compile SM100/SM103 when `CUDA_VERSION > 12.8` / `>= 13.0`, so rebuild:
   ```bash
   pip uninstall -y sgl-kernel
   cd sgl-kernel && pip install --no-build-isolation .   # long; needs CUDA 13 nvcc
   # success markers in the log: "Patched utils.h for SM103a support"
   ```
   After the rebuild, on B300 you get **sparse decode/prefill and dense
   prefill**, but **dense decode is still absent** — `flashmla.cmake`'s SM100
   block has no dense-decode source. That kernel would have to be written in the
   external `sgl-project/FlashMLA` repo.

So on B300 today: `--mode decode --dtype bf16/fp8` shows flashmla `n/a` (no
dense decode), trtllm-gen runs. `--mode sparse_decode` is the FlashMLA path
that B300 can support after the rebuild. See `benchmarkno1.md` section 4b and
`worklog_2026-06-01.md`.

## Caveats

- bf16 path uses `bmm1_scale = softmax_scale` (q/k descale = 1). For an FP8-KV
  comparison you'd add per-tensor descales and the `mla_quantize_and_rope_for_fp8`
  preprocessing the trtllm backend uses; left out here to keep the first cut
  apples-to-apples in bf16.
- flashinfer API kwargs (`trtllm_ragged_attention_deepseek`,
  `BatchPrefillWithRaggedKVCacheWrapper.plan`) track the pinned
  `flashinfer==0.6.11.post1`. If you bump flashinfer, re-check signatures.
