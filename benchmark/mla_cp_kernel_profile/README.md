# MLA Attention Kernel Benchmark: FlashMLA vs trtllm-gen (B300 / DSA)

Microbenchmark comparing **FlashMLA** (`sgl_kernel`) and **trtllm-gen** (`flashinfer.decode`)
for DeepSeek V3.2 DSA (sparse attention) on NVIDIA B300 SXM6 (SM 10.3, Blackwell).

Environment: torch 2.12.0+cu130, `sgl_kernel` rebuilt from source with CUDA 13,
flashinfer 0.6.x. Timing: CUDA events, median of 50 iterations, L2 flush between each.

---

## Result summary

**On B300 with fp8 deployment, trtllm-gen is faster across all measured paths.**

| path | winner | margin |
|------|--------|--------|
| sparse decode, q_len=1 (fp8) | trtllm | **2.4–2.75×** |
| sparse decode, q_len>1 (fp8, unrolled) | trtllm | **2.9–3.5×** |
| sparse prefill / extend (fp8) | trtllm | **1.7–2.0×** (vs flashmla bf16) |
| sparse prefill via flashmla_kv fp8 | trtllm | **2.7–3.5×** (+ smem fallback at B≥16) |
| dense decode on B300 | trtllm only | flashmla has no SM100 kernel |

The only case where FlashMLA has an edge is bf16-vs-bf16 prefill (~10% faster), but
this configuration does not exist in a real fp8 deployment.

### Why trtllm wins

1. **FlashMLA's sparse prefill kernel (`flash_mla_sparse_fwd`) is bf16-only.**
   The cmake SM100 prefill sources (`phase1_k512.cu`, `phase1_k576.cu`) have no fp8
   variant; the function signature has no `is_fp8_kvcache`; the docstring says
   `kv: bfloat16`. trtllm natively supports fp8 for prefill, which halves KV read
   bandwidth.

2. **FlashMLA's only fp8 prefill path is the decode kernel reused (`flashmla_kv`).**
   `_forward_flashmla_kv` (dsa_backend.py:1776) folds each new token into a q_len=1
   decode call (`b × s_q` rows). At large batch×s_q this hits the **shared memory
   limit** (280 KB needed vs 232 KB max on B300) and falls to a slow fallback kernel.

3. **FlashMLA has no dense decode kernel on B300.** `flashmla.cmake` builds dense
   decode only for SM90 (Hopper). On SM100 (Blackwell) a bf16/fp16 query raises
   `Dense decode MLA is only supported on SM90a architecture`. Only sparse decode
   (SM100 `v32`/`model1` kernels, fp8) exists on B300 after the CUDA-13 rebuild.

4. **trtllm's dispatch overhead is lower for q_len=1.** At decode time the
   computation is GEMV-bound with very low arithmetic intensity; fixed overhead
   (kernel launch, metadata processing) dominates. trtllm's kernel is more optimized
   for this regime.

---

## File layout

```
bench_mla_kernels.py    main benchmark script
benchmark_final.md      final results report (Chinese)
docs/
  benchmarkno1.md         initial analysis and hypothesis write-up
  bench_implementation.md math comparison of 3 regimes + per-test implementation
  bench_implementation_v2.md  fp8 sparse_decode alignment story (3 stages)
  dsa_dispatch_report.md  full DSA V3.2 dispatch call graph (Chinese, with Mermaid)
  changelog.md            change log
  worklog_2026-06-01.md   day-1 work log (B300 setup, initial findings)
  worklog_2026-06-05.md   day-2 work log (fp8 fix, bug fixes, dispatch verification)
  worklog_2026-06-08.md   day-3 work log (prefill claim confirmed, docs cleanup)
```

---

## Quick start (B300 box, after setup below)

```bash
cd benchmark/mla_cp_kernel_profile

# Sparse decode: the FlashMLA path that actually runs on B300
python bench_mla_kernels.py --mode sparse_decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --csv sd.csv

# Sparse prefill / extend: batch sweep, 875 new + 69125 cached tokens
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 2 4 8 16 32 --seq-q 875 --cached-len 69125 --heads 64 --csv prefill.csv

# Sparse prefill / extend: 90k cached + 10k new tokens
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 --seq-q 10000 --cached-len 90000 --heads 128 --csv extend.csv

# With context parallel (CP=8): per-rank q = 10000/8, full KV
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 --seq-q 10000 --cached-len 90000 --cp-size 8 --heads 128 --csv extend_cp8.csv

# Dense decode (documents missing FlashMLA kernel on B300 — all n/a)
python bench_mla_kernels.py --mode decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --dtype bf16
```

`--heads` = q-heads after TP (128 for TP=1, 16 for TP=8, etc.).

`speedup` column = `flashmla_ms / trtllm_ms`; values > 1 mean FlashMLA is slower.

---

## Benchmark modes

| `--mode` | q_len | what runs |
|----------|-------|-----------|
| `decode` | 1 | `flash_mla_with_kvcache` (dense) vs `trtllm_batch_decode_with_kv_cache_mla` |
| `prefill_absorbed` | > 1 | same as decode but causal; FlashMLA decode kernel reused for prefill |
| `prefill_ragged` | > 1 | `flashinfer` ragged wrapper vs `trtllm_ragged_attention_deepseek` |
| `sparse_decode` | 1 | `flash_mla_with_kvcache(indices=…, is_fp8_kvcache=True)` vs trtllm sparse |
| `sparse_prefill` | > 1 | `flash_mla_sparse_fwd` (bf16) + `flashmla_kv` (fp8) vs trtllm fp8, plus standalone-FlashMLA columns (below) |
| `sparse_prefill_v4` | > 1 | V4/MODEL1 shapes (d_qk=512): sgl bf16 vs standalone native fp8 vs trtllm (expected n/a) |

`sparse_prefill` is the DSA / V3.2 path and the realistic long-context scenario.
It outputs five columns: `trtllm_fp8`, `flashmla_sparse_bf16`, `flashmla_kv_fp8`,
`sa_bf16`, `sa_fp8v32`.

---

## Standalone FlashMLA columns (fp8 prefill)

The `sa_*` columns and the `sparse_prefill_v4` mode use the **standalone**
`flash_mla` package instead of sgl_kernel's vendored copy (which is bf16-only
for sparse prefill — finding #1 above). No sgl-kernel rebuild is needed; the
two coexist (`flash_mla` vs `sgl_kernel.flash_mla`):

```bash
git clone -b fp8-sparse-prefill https://github.com/galbi-joa/FlashMLA.git
cd FlashMLA && git submodule update --init csrc/cutlass
FLASH_MLA_DISABLE_SM90=1 pip install -v --no-build-isolation .   # CUDA >= 12.9
python tests/test_flash_mla_sparse_prefill_fp8_v32.py            # correctness first
python tests/test_flash_mla_sparse_prefill_fp8.py                # (V4/MODEL1 kernel)
```

If the package is missing, the new columns just print `n/a` — existing columns
are unaffected.

- `sa_bf16` — the same bf16 call through the standalone build (deepseek main,
  newer than the sgl-project pin `df022eb`): isolates code-version effects.
- `sa_fp8v32` — V3.2 fp8 prefill: the wrapper takes the **same 656B-per-token
  V32 cache** this script already builds for the other fp8 columns, dequantizes
  the pool into a reused bf16 workspace (<1% of the attention's own KV traffic)
  and runs the bf16 d=576 kernel. Same shape + same cache bytes as the trtllm
  column → the first apples-to-apples FlashMLA-vs-trtllm fp8 prefill row.
  The timed region includes the per-call (= per-layer) dequant.
- `sparse_prefill_v4` — DeepSeek V4 ships with `attention_backend=dsv4` and a
  packed FP8-nope/BF16-rope cache = FlashMLA's MODEL1 format (d_qk = 448+64).
  The standalone build has a **native** fp8 prefill kernel for it (PrefillFp8:
  raw fp8 + e8m0 scales TMA-gathered, dequantized in shared memory). trtllm-gen
  has no kernel for this head layout, so its column documents that gap.
  V4-realistic sweeps: add `--topk 1024` (or 512).

```bash
# V3.2 shapes with the standalone columns (same command as before):
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 --seq-q 10000 --cached-len 90000 --heads 128 --csv extend.csv

# V4/MODEL1 shapes:
python bench_mla_kernels.py --mode sparse_prefill_v4 \
    --batch 1 --seq-q 10000 --cached-len 90000 --heads 128 --topk 1024 --csv v4.csv
```

---

## Context parallel (CP) note

CP is a model-level concern, not a kernel feature. The latent-KV all-gather
(`rebuild_cp_kv_cache` → `cp_all_gather_rerange_output` in `deepseek_v2.py`) runs
before the attention kernel. Once CP has assembled the full KV, each rank calls a
plain MLA attention kernel with no knowledge of CP. `--cp-size N` shards the new
tokens across N ranks, so the per-rank q length = `seq_q // N`; KV stays full.

---

## Setup on B300 (required)

Two issues arise on Blackwell (SM10.3):

**1. CUDA toolkit 13 for trtllm-gen JIT compilation.**
flashinfer JIT-compiles the trtllm-gen kernel for `compute_103a`; nvcc 12.8 does
not know SM103 and fails with `Unsupported gpu architecture 'compute_103a'`.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y cuda-toolkit-13-0   # NOT 'cuda' / 'cuda-13-0' (avoids touching the driver)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
rm -rf /root/.cache/flashinfer         # drop the failed 12.8 JIT cache
```

**2. Rebuild `sgl_kernel` from source with CUDA 13.**
A pip-installed wheel is typically built under CUDA ≤ 12.8 and has no sm100/sm103
binaries; every flashmla call raises `no kernel image is available`.
`sgl-kernel/cmake/flashmla.cmake` compiles SM100/SM103 when CUDA ≥ 13.0:

```bash
pip uninstall -y sgl-kernel
cd sgl-kernel && pip install --no-build-isolation .   # needs CUDA-13 nvcc; takes a few minutes
# look for: "Patched utils.h for SM103a support"
```

After the rebuild, B300 gets sparse decode and sparse/dense prefill kernels.
Dense decode remains absent (no SM100 dense-decode source in the FlashMLA repo).
