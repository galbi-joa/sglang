"""
MLA attention kernel microbenchmark: FlashMLA (sgl_kernel) vs trtllm-gen (flashinfer).

WHY THIS EXISTS
---------------
There is a claim: "On B300, FlashMLA is much slower than trtllm-gen in the CP
scenario." CP (context parallel) itself is a *model-level* concern (the latent-KV
all-gather lives in deepseek_v2.py, not in either kernel). Once CP has gathered
the KV, what actually runs is a plain MLA attention kernel. So the real question
reduces to a kernel-vs-kernel comparison under long-context shapes.

KEY CORRECTION (matches benchmarkno1.md sections 3, 5b)
-------------------------------------------------------
FlashMLA does NOT only run on decode. SGLang reuses the **same** FlashMLA decode
kernel (`flash_mla_with_kvcache` -> `fwd_kvcache_mla`) for a large part of
*prefill* too: whenever the dispatch in
`attention_backend_handler.py:_handle_attention_backend` returns
`AttnForwardMethod.MLA` (absorbed) instead of `MHA_*` -- i.e. under piecewise
CUDA graph, prefill-CP, `flashinfer_mla_disable_ragged`, or a prefix with
chunked-prefix-cache disabled. Only *pure ragged* prefill (forward_mode==EXTEND,
no prefix, ragged allowed) defers to a flashinfer ragged wrapper. So this script
benchmarks three regimes:

    decode            : flashmla `flash_mla_with_kvcache` (q_len=1)
                        vs trtllm-gen `trtllm_batch_decode_with_kv_cache_mla`
    prefill_absorbed  : flashmla `flash_mla_with_kvcache` (q_len>1, causal)  <-- decode kernel reused for prefill
                        vs trtllm-gen `trtllm_batch_decode_with_kv_cache_mla`
    prefill_ragged    : flashinfer ragged wrapper (flashmla's pure-prefill fallback)
                        vs trtllm-gen `trtllm_ragged_attention_deepseek`

This script times each kernel in isolation, cross-checks numerical agreement so
the comparison is apples-to-apples, and (optionally) emits a Chrome trace you can
feed to the `llm-torch-profiler-analysis` skill.

SHAPES (DeepSeek-V3 MLA)
------------------------
    kv_lora_rank      = 512
    qk_nope_head_dim  = 128
    qk_rope_head_dim  = 64
    page/block size   = 64   (both backends support 64)
    h_q               = num_attention_heads // TP   (128 for TP=1)
    h_kv              = 1    (MLA = MQA against the shared latent)

  Absorbed form (decode AND absorbed-MLA prefill -- the kernel sees the latent):
    head_dim (q / kv) = kv_lora_rank + qk_rope_head_dim = 576
    head_dim_v (out)  = kv_lora_rank                   = 512
    softmax_scale     = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5 = 192**-0.5

  Non-absorbed form (pure ragged prefill only):
    q/k head_dim = 192, v head_dim = 128.

B300 NOTE (important)
---------------------
FlashMLA has NO bf16/fp16 dense-decode kernel on Blackwell (SM100/SM103). A bf16
query raises "BF16 Dense MLA is not supported on SM100" -- this is a *finding*,
not a bug: it confirms hypothesis #1. The only dense-decode path that exists on
B300 is FP8 (`fwd_kvcache_mla_fp8`). The script captures the error and prints an
`n/a` row + the message instead of crashing. For a real head-to-head on B300,
run with `--dtype fp8`.

USAGE
-----
    # decode sweep -- on B300 use --dtype fp8 (bf16 has no FlashMLA dense kernel)
    python bench_mla_kernels.py --mode decode \
        --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --dtype fp8

    # the bf16 run is still useful to *document* the missing-kernel finding:
    python bench_mla_kernels.py --mode decode --seq-k 4096 --heads 128 --dtype bf16

    # absorbed-MLA prefill: FlashMLA decode kernel doing prefill work (q_len>1)
    python bench_mla_kernels.py --mode prefill_absorbed \
        --batch 1 --seq-q 2048 8192 --heads 128 --dtype bf16

    # pure ragged prefill (flashmla's fallback) vs trtllm ragged
    python bench_mla_kernels.py --mode prefill_ragged \
        --batch 1 --seq-q 2048 8192 --heads 128 --dtype bf16

    # everything, and dump a chrome trace for the profiler skill
    python bench_mla_kernels.py --mode all --trace trace_mla.json

Run this ON THE B300 box (needs CUDA + torch + sgl_kernel + flashinfer).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# ----- MLA constants (DeepSeek-V3) -----
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_CKV = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576  (decode q & kv last dim)
HEAD_DIM_V = KV_LORA_RANK  # 512  (decode output)
HEAD_DIM_QK = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192 (prefill q/k)
SOFTMAX_SCALE = HEAD_DIM_QK ** -0.5  # scaling is on the *original* 192, not 576
PAGE_SIZE = 64

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn}

# On Blackwell (SM100/B300) FlashMLA has NO bf16/fp16 dense-decode kernel
# (sgl-kernel/cmake/flashmla.cmake compiles dense decode only for sm90; the
# sm100 list is sparse-only). The only dense-decode path that exists on B300 is
# the FP8 one (fwd_kvcache_mla_fp8). So `--dtype fp8` is the realistic regime
# for comparing FlashMLA vs trtllm-gen decode on B300.
FP8_DTYPE = torch.float8_e4m3fn


# --------------------------------------------------------------------------- #
# Capability detection / optional imports
# --------------------------------------------------------------------------- #
def device_cap() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def try_import_flashmla():
    try:
        from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

        return flash_mla_with_kvcache, get_mla_metadata
    except Exception as e:  # noqa: BLE001
        print(f"[skip] flashmla unavailable: {e}")
        return None, None


def try_import_flashinfer():
    try:
        import flashinfer

        return flashinfer
    except Exception as e:  # noqa: BLE001
        print(f"[skip] flashinfer unavailable: {e}")
        return None


# --------------------------------------------------------------------------- #
# Timing harness: CUDA events, warmup, L2 flush between iters
# --------------------------------------------------------------------------- #
@dataclass
class TimeResult:
    name: str
    ok: bool
    ms: float = float("nan")
    note: str = ""


def _l2_flush(buf: torch.Tensor) -> None:
    # write the whole buffer to evict L2 so each iter starts cold (kernel-fair)
    buf.zero_()


def time_kernel(
    name: str,
    fn: Callable[[], torch.Tensor],
    iters: int = 50,
    warmup: int = 10,
) -> TimeResult:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            _l2_flush(flush_buf)
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        # median is robust to the occasional scheduling hiccup
        return TimeResult(name=name, ok=True, ms=times[len(times) // 2])
    except Exception as e:  # noqa: BLE001
        return TimeResult(name=name, ok=False, note=repr(e))


# --------------------------------------------------------------------------- #
# Shared paged-KV builder (so both backends attend to identical data)
# --------------------------------------------------------------------------- #
def build_decode_inputs(b, s_q, s_k, h_q, dtype, device="cuda", seed=0):
    torch.manual_seed(seed)
    cache_seqlens = torch.full((b,), s_k, dtype=torch.int32, device=device)
    max_seqlen_pad = ((s_k + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    num_blocks_per = max_seqlen_pad // PAGE_SIZE

    # Generate in a float type, then cast (randn does not support fp8 directly).
    gen_dtype = torch.bfloat16 if dtype == FP8_DTYPE else dtype

    # absorbed-form query: head_dim = 576
    q = (torch.randn(b, s_q, h_q, HEAD_DIM_CKV, dtype=gen_dtype, device=device) / 10).clamp_(-1, 1)

    block_table = torch.arange(
        b * num_blocks_per, dtype=torch.int32, device=device
    ).view(b, num_blocks_per)

    # latent KV, h_kv = 1, last dim 576. h_kv axis is size-1 so the two backend
    # layouts (.., block, 1, 576) and (.., 1, block, 576) share memory.
    kv_paged = (
        torch.randn(b * num_blocks_per, PAGE_SIZE, HEAD_DIM_CKV, dtype=gen_dtype, device=device) / 10
    ).clamp_(-1, 1)

    if dtype == FP8_DTYPE:
        q = q.to(FP8_DTYPE)
        kv_paged = kv_paged.to(FP8_DTYPE)
    return q, kv_paged, block_table, cache_seqlens, max_seqlen_pad


# --------------------------------------------------------------------------- #
# DECODE runners
# --------------------------------------------------------------------------- #
def make_flashmla_decode(q, kv_paged, block_table, cache_seqlens, h_q):
    """Build a FlashMLA decode runner. Returns (run_fn, note).

    run_fn is None when the kernel is unavailable for this dtype/device -- e.g.
    on B300 a bf16 query raises 'BF16 Dense MLA is not supported on SM100',
    because FlashMLA only ships an FP8 dense-decode kernel for Blackwell. The
    error is captured here (not raised) so it surfaces as an `n/a` + note row.
    """
    flash_mla_with_kvcache, get_mla_metadata = try_import_flashmla()
    if flash_mla_with_kvcache is None:
        return None, "flashmla unavailable"

    b, s_q = q.shape[0], q.shape[1]
    is_fp8 = q.element_size() == 1
    k_cache = kv_paged.unsqueeze(2)  # (num_blocks, PAGE_SIZE, 1, 576)
    descale = None
    if is_fp8:
        # per-tensor descale = 1.0 (inputs already in the fp8 range); both must
        # be passed together per the wrapper's assertion.
        descale = torch.ones((1,), dtype=torch.float32, device=q.device)

    try:
        tile_md, num_splits = get_mla_metadata(
            cache_seqlens, s_q * h_q // 1, 1, h_q, is_fp8, None
        )
    except Exception as e:  # noqa: BLE001  -- kernel/dtype not built for this SM
        return None, _short_err(e)

    def run():
        out, _ = flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            HEAD_DIM_V,  # head_dim_v = 512
            tile_md,
            num_splits,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            descale_q=descale,
            descale_k=descale,
            is_fp8_kvcache=is_fp8,
        )
        return out  # (b, s_q, h_q, 512)

    return run, ""


def make_trtllm_decode(q, kv_paged, block_table, cache_seqlens, max_seq_len):
    """Build a trtllm-gen decode runner. Returns (run_fn, note)."""
    fi = try_import_flashinfer()
    if fi is None:
        return None, "flashinfer unavailable"
    kv_cache = kv_paged.unsqueeze(1)  # (num_blocks, 1, PAGE_SIZE, 576)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    bmm1_scale = SOFTMAX_SCALE  # q_scale * k_scale * softmax_scale; descale=1

    def run():
        return fi.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,  # (b, s_q, h_q, 576)
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=int(max_seq_len),
            bmm1_scale=bmm1_scale,
            # backend defaults to "trtllm-gen"
        )

    return run, ""


# --------------------------------------------------------------------------- #
# PREFILL runners (non-absorbed form: q/k head_dim=192, v head_dim=128)
# --------------------------------------------------------------------------- #
def build_prefill_inputs(b, s_q, h_q, dtype, device="cuda", seed=0):
    torch.manual_seed(seed)
    total_q = b * s_q
    q = (torch.randn(total_q, h_q, HEAD_DIM_QK, dtype=dtype, device=device) / 10).clamp_(-1, 1)
    k = (torch.randn(total_q, h_q, HEAD_DIM_QK, dtype=dtype, device=device) / 10).clamp_(-1, 1)
    v = (torch.randn(total_q, h_q, QK_NOPE_HEAD_DIM, dtype=dtype, device=device) / 10).clamp_(-1, 1)
    cu = torch.arange(0, (b + 1) * s_q, s_q, dtype=torch.int32, device=device)
    seq_lens = torch.full((b,), s_q, dtype=torch.int32, device=device)
    return q, k, v, cu, seq_lens


def make_trtllm_prefill(q, k, v, cu, seq_lens, b, s_q):
    fi = try_import_flashinfer()
    if fi is None:
        return None
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    def run():
        return fi.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=s_q,
            max_kv_len=s_q,
            bmm1_scale=SOFTMAX_SCALE,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=b,
            window_left=-1,
            cum_seq_lens_q=cu,
            cum_seq_lens_kv=cu,
            enable_pdl=False,
            is_causal=True,
            return_lse=False,
        )

    return run


def make_flashinfer_ragged_prefill(q, k, v, cu, seq_lens, b, s_q):
    """What the flashmla backend actually falls back to for pure prefill."""
    fi = try_import_flashinfer()
    if fi is None:
        return None
    try:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr=cu,
            kv_indptr=cu,
            num_qo_heads=q.shape[1],
            num_kv_heads=k.shape[1],
            head_dim_qk=HEAD_DIM_QK,
            head_dim_vo=QK_NOPE_HEAD_DIM,
            causal=True,
            sm_scale=SOFTMAX_SCALE,
            q_data_type=q.dtype,
        )

        def run():
            return wrapper.run(q, k, v)

        return run
    except Exception as e:  # noqa: BLE001
        print(f"[skip] flashinfer ragged wrapper setup failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# SPARSE decode runners (DeepSeek Sparse Attention / V3.2)
# --------------------------------------------------------------------------- #
# This is the ONLY native FlashMLA decode path that exists on B300 (SM100):
# sgl-kernel/cmake/flashmla.cmake compiles sm100/decode/.../{v32,model1}.cu
# (sparse, fp8) but no dense decode. FlashMLA runs it via
# flash_mla_with_kvcache(..., indices=topk, is_fp8_kvcache=True). trtllm-gen
# handles the same sparse decode as a dense decode over a *reduced* page table
# (dsa_backend.py:2152). So this is the apples-to-apples sparse comparison.
DSA_TOPK = 2048  # DeepSeek V3.2 indexer top-k


def quantize_k_cache(input_k_cache, dv=KV_LORA_RANK, tile_size=128):
    """Port of test_flashmla.quantize_k_cache: (num_blocks, block, 1, 576) bf16
    -> (num_blocks, block, 1, dv + 4*(dv/tile) + 2*(d-dv)) uint8 fp8 layout."""
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1 and dv % tile_size == 0
    num_tiles = dv // tile_size
    x = input_k_cache.squeeze(2)  # (num_blocks, block, d)
    elem = x.element_size()
    out = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + elem * (d - dv)),
        dtype=torch.float8_e4m3fn, device=x.device,
    )
    nope = out[..., :dv]
    scale = out[..., dv:dv + num_tiles * 4].view(torch.float32)
    rope = out[..., dv + num_tiles * 4:].view(x.dtype)
    rope[:] = x[..., dv:]
    for t in range(num_tiles):
        sl = slice(t * tile_size, (t + 1) * tile_size)
        inv = x[..., sl].abs().max(dim=-1).values / 448.0  # (num_blocks, block)
        scale[:, :, t] = inv
        nope[..., sl] = (x[..., sl].float() / inv.unsqueeze(-1).float()).to(torch.float8_e4m3fn)
    return out.view(num_blocks, block_size, 1, -1)


def build_sparse_inputs(b, s_k, h_q, topk, device="cuda", seed=0):
    torch.manual_seed(seed)
    cache_seqlens = torch.full((b,), s_k, dtype=torch.int32, device=device)
    max_pad = ((s_k + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    nblk = max_pad // PAGE_SIZE
    # DSA sparse decode keeps q in bf16; only the K cache is fp8-quantized.
    # (dsa_backend.py: q_all stays bf16 at :1792, quantize_k_cache only on KV.)
    q = (torch.randn(b, 1, h_q, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device) / 10).clamp_(-1, 1)
    block_table = torch.arange(b * nblk, dtype=torch.int32, device=device).view(b, nblk)
    kv = (torch.randn(b * nblk, PAGE_SIZE, 1, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device) / 10).clamp_(-1, 1)
    kv_q = quantize_k_cache(kv)  # (nblk, PAGE_SIZE, 1, bytes), fp8
    # absolute indices into the flattened kv cache (valid range [0, s_k)).
    idx = torch.randint(0, s_k, (b, 1, topk), dtype=torch.int32, device=device)
    return q, kv_q, block_table, cache_seqlens, idx


def make_flashmla_sparse_decode(q, kv_q, cache_seqlens, idx, h_q):
    flash_mla_with_kvcache, get_mla_metadata = try_import_flashmla()
    if flash_mla_with_kvcache is None:
        return None, "flashmla unavailable"
    topk = idx.shape[-1]

    # flashmla.cmake only instantiates SM100 sparse decode for head64 / head128
    # (sm100/decode/head64/{v32,model1}.cu). dsa_backend.py pads num_heads up to
    # a multiple of 64 (Hopper) or 128 (Blackwell) for exactly this reason; we
    # mirror that here so a non-{64,128} head count doesn't hit a missing kernel.
    maj = device_cap()[0]
    pad_to = 128 if maj >= 10 else 64
    h_pad = ((h_q + pad_to - 1) // pad_to) * pad_to
    if h_pad != h_q:
        q = torch.nn.functional.pad(q, (0, 0, 0, h_pad - h_q))

    try:
        tile_md, num_splits = get_mla_metadata(
            cache_seqlens, 1 * h_pad // 1, 1, h_pad, True, topk
        )
    except Exception as e:  # noqa: BLE001
        return None, _short_err(e)

    def run():
        # block_table is unused by the sparse path but must NOT be None -- the
        # wrapper asserts `block_table is not None`. dsa_backend.py:1826 passes a
        # (batch, 0) empty int32 tensor for exactly this reason; mirror it.
        empty_block_table = torch.empty(
            (q.shape[0], 0), dtype=torch.int32, device=q.device
        )
        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_q,
            block_table=empty_block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=HEAD_DIM_V,
            tile_scheduler_metadata=tile_md,
            num_splits=num_splits,
            softmax_scale=SOFTMAX_SCALE,
            is_fp8_kvcache=True,
            indices=idx,
            # causal omitted -> defaults to False (sparse path requires False)
        )
        return out

    return run, ""


def make_trtllm_sparse_decode(kv_q_dim_kv, cache_seqlens, idx, b, s_k, h_q, kv_bf16):
    """trtllm-gen sparse decode. dsa_backend.py:2152 calls
    trtllm_batch_decode_with_kv_cache_mla with sparse_mla_top_k=topk and a
    block_tables that already encodes the topk positions. We feed the topk
    indices directly as the (page_size=1) block table."""
    fi = try_import_flashinfer()
    if fi is None:
        return None, "flashinfer unavailable"
    topk = idx.shape[-1]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    # trtllm-gen MLA keeps q in bf16 and a bf16/fp8 paged kv cache laid out as
    # (num_blocks, 1, page_size, kv_cache_dim). Use the same bf16 latent KV.
    q = torch.randn(b, 1, h_q, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    # block_tables for sparse: (batch, 1, topk) of kv positions, page_size 1.
    block_tables = idx.view(b, 1, topk).to(torch.int32)

    def run():
        return fi.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_bf16,
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=cache_seqlens,
            max_seq_len=int(s_k),
            sparse_mla_top_k=topk,
            bmm1_scale=SOFTMAX_SCALE,
            backend="trtllm-gen",
        )

    return run, ""


def run_sparse_decode(args):
    """DSA / V3.2 sparse decode: FlashMLA native sparse kernel (the path B300
    can run after the CUDA-13 rebuild) vs trtllm-gen sparse decode."""
    rows = []
    print(f"\n=== SPARSE DECODE (DSA/V3.2, fp8 KV, topk={DSA_TOPK}, q_len=1) ===")
    print(f"{'B':>4} {'S_K':>7} {'H':>4} | {'flashmla(ms)':>13} {'trtllm(ms)':>12} "
          f"{'speedup':>8}")
    for b in args.batch:
        for s_k in args.seq_k:
            for h in args.heads:
                if DSA_TOPK > s_k:
                    print(f"{b:>4} {s_k:>7} {h:>4} | {'skip':>13} {'skip':>12} "
                          f"{'-':>8}   (S_K < topk={DSA_TOPK})")
                    continue
                q, kv_q, bt, cs, idx = build_sparse_inputs(b, s_k, h, DSA_TOPK)
                # bf16 paged latent KV for the trtllm path (it does not take the
                # fp8-packed layout FlashMLA uses).
                nblk = kv_q.shape[0]
                kv_bf16 = (torch.randn(nblk, 1, PAGE_SIZE, HEAD_DIM_CKV,
                                       dtype=torch.bfloat16, device="cuda") / 10).clamp_(-1, 1)

                fm, fm_note = make_flashmla_sparse_decode(q, kv_q, cs, idx, h)
                tg, tg_note = make_trtllm_sparse_decode(
                    kv_q.shape, cs, idx, b, s_k, h, kv_bf16
                )
                r_fm = time_kernel("flashmla_sparse", fm) if fm else TimeResult("flashmla_sparse", False, note=fm_note or "unavailable")
                r_tg = time_kernel("trtllm_sparse", tg) if tg else TimeResult("trtllm_sparse", False, note=tg_note or "unavailable")

                sp = (r_fm.ms / r_tg.ms) if (r_fm.ok and r_tg.ok) else float("nan")
                print(f"{b:>4} {s_k:>7} {h:>4} | {_fmt(r_fm):>13} {_fmt(r_tg):>12} "
                      f"{sp:>8.2f}")
                _print_note_lines(b, f"B={b} S_K={s_k} H={h}", r_fm, r_tg)
                rows.append((b, s_k, h, r_fm, r_tg, sp))
    return rows


# --------------------------------------------------------------------------- #
# Correctness cross-check (cosine similarity, fp-tolerant)
# --------------------------------------------------------------------------- #
def cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double().flatten(), y.double().flatten()
    return (1 - 2 * (x * y).sum() / max((x * x + y * y).sum().item(), 1e-12)).item()


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def _bench_absorbed_pair(b, s_q, s_k, h, dtype):
    """Build + time the FlashMLA and trtllm-gen absorbed-MLA runners for one
    shape. Returns (r_fm, r_tg, speedup, cos_diff). Handles the (run, note)
    tuples and turns a missing kernel into an `n/a` row carrying the note (e.g.
    'BF16 Dense MLA is not supported on SM100')."""
    q, kv, bt, cs, _ = build_decode_inputs(b, s_q, s_k, h, dtype)
    fm, fm_note = make_flashmla_decode(q, kv, bt, cs, h)
    tg, tg_note = make_trtllm_decode(q, kv, bt, cs, s_k)

    r_fm = time_kernel("flashmla", fm) if fm else TimeResult("flashmla", False, note=fm_note or "unavailable")
    r_tg = time_kernel("trtllm", tg) if tg else TimeResult("trtllm", False, note=tg_note or "unavailable")

    # Only fetch outputs for the cross-check if BOTH timed cleanly. A failed run
    # (e.g. trtllm JIT failure / flashmla 'no kernel image') already recorded its
    # error in r_*.note; calling it again would just re-raise and abort the sweep.
    cd = float("nan")
    if r_fm.ok and r_tg.ok:
        out_fm = _safe_call(fm)
        out_tg = _safe_call(tg)
        if out_fm is not None and out_tg is not None:
            cd = cos_diff(out_fm.reshape(b, h, -1), out_tg.reshape(b, h, -1))
    sp = (r_fm.ms / r_tg.ms) if (r_fm.ok and r_tg.ok) else float("nan")
    return r_fm, r_tg, sp, cd


def _print_note_lines(b, shape_str, r_fm, r_tg):
    """Surface the kernel-unavailable reason underneath an n/a row."""
    if not r_fm.ok and r_fm.note:
        print(f"    [{shape_str}] flashmla: {r_fm.note}")
    if not r_tg.ok and r_tg.note:
        print(f"    [{shape_str}] trtllm:   {r_tg.note}")


def run_decode(args):
    dtype = _DTYPES[args.dtype]
    rows = []
    print(f"\n=== DECODE (q_len=1 absorbed MLA, head_dim 576 -> 512, dtype={args.dtype}) ===")
    print(f"{'B':>4} {'S_K':>7} {'H':>4} | {'flashmla(ms)':>13} {'trtllm(ms)':>12} "
          f"{'speedup':>8} {'cos_diff':>10}")
    for b in args.batch:
        for s_k in args.seq_k:
            for h in args.heads:
                r_fm, r_tg, sp, cd = _bench_absorbed_pair(b, 1, s_k, h, dtype)
                print(f"{b:>4} {s_k:>7} {h:>4} | {_fmt(r_fm):>13} {_fmt(r_tg):>12} "
                      f"{sp:>8.2f} {cd:>10.2e}")
                _print_note_lines(b, f"B={b} S_K={s_k} H={h}", r_fm, r_tg)
                rows.append((b, s_k, h, r_fm, r_tg, sp, cd))
    return rows


def run_prefill_absorbed(args):
    """Absorbed-MLA prefill: SGLang reuses the FlashMLA *decode* kernel for
    prefill (q_len>1, causal) whenever dispatch returns AttnForwardMethod.MLA.
    Same kernels as decode, only s_q>1 -- this is the regime the earlier draft
    wrongly excluded."""
    dtype = _DTYPES[args.dtype]
    rows = []
    print(f"\n=== PREFILL (absorbed MLA: FlashMLA decode kernel reused, q_len>1, "
          f"head_dim 576 -> 512, causal, dtype={args.dtype}) ===")
    print(f"{'B':>4} {'S_Q':>7} {'S_K':>7} {'H':>4} | {'flashmla(ms)':>13} "
          f"{'trtllm(ms)':>12} {'speedup':>8} {'cos_diff':>10}")
    for b in args.batch:
        for s_q in args.seq_q:
            for h in args.heads:
                # s_k == s_q: pure prefill (no cached prefix), causal self-attn.
                r_fm, r_tg, sp, cd = _bench_absorbed_pair(b, s_q, s_q, h, dtype)
                print(f"{b:>4} {s_q:>7} {s_q:>7} {h:>4} | {_fmt(r_fm):>13} "
                      f"{_fmt(r_tg):>12} {sp:>8.2f} {cd:>10.2e}")
                _print_note_lines(b, f"B={b} S_Q={s_q} H={h}", r_fm, r_tg)
                rows.append((b, s_q, h, r_fm, r_tg, sp, cd))
    return rows


def run_prefill_ragged(args):
    """Pure ragged prefill: what FlashMLA falls back to when dispatch does NOT
    pick absorbed MLA (forward_mode==EXTEND, no prefix, ragged allowed). Here it
    is genuinely flashinfer-ragged vs trtllm-gen-ragged."""
    dtype = _DTYPES[args.dtype]
    print("\n=== PREFILL (pure ragged fallback, non-absorbed head_dim 192/128, "
          "causal) ===")
    print(f"{'B':>4} {'S_Q':>7} {'H':>4} | {'fi_ragged(ms)':>14} {'trtllm(ms)':>12} "
          f"{'speedup':>8} {'cos_diff':>10}")
    rows = []
    for b in args.batch:
        for s_q in args.seq_q:
            for h in args.heads:
                q, k, v, cu, sl = build_prefill_inputs(b, s_q, h, dtype)
                fi_run = make_flashinfer_ragged_prefill(q, k, v, cu, sl, b, s_q)
                tg_run = make_trtllm_prefill(q, k, v, cu, sl, b, s_q)

                r_fi = time_kernel("fi_ragged", fi_run) if fi_run else TimeResult("fi_ragged", False, note="unavailable")
                r_tg = time_kernel("trtllm", tg_run) if tg_run else TimeResult("trtllm", False, note="unavailable")

                out_fi = fi_run() if fi_run else None
                out_tg = tg_run() if tg_run else None
                cd = float("nan")
                if out_fi is not None and out_tg is not None:
                    o_tg = out_tg[0] if isinstance(out_tg, (tuple, list)) else out_tg
                    cd = cos_diff(out_fi.reshape(b * s_q, h, -1), o_tg.reshape(b * s_q, h, -1))

                sp = (r_fi.ms / r_tg.ms) if (r_fi.ok and r_tg.ok) else float("nan")
                print(f"{b:>4} {s_q:>7} {h:>4} | {_fmt(r_fi):>14} {_fmt(r_tg):>12} "
                      f"{sp:>8.2f} {cd:>10.2e}")
                rows.append((b, s_q, h, r_fi, r_tg, sp, cd))
    return rows


# small helpers
def _fmt(r: TimeResult) -> str:
    return f"{r.ms:.3f}" if r.ok else "n/a"


def _short_err(e: Exception) -> str:
    """One-line error string for the note column (e.g. the SM100 dense-MLA msg)."""
    s = str(e).strip().splitlines()
    return s[0][:80] if s else type(e).__name__


def _safe_call(fn):
    """Call a runner once for the cross-check, swallowing any kernel error."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def maybe_trace(args, fn):
    """Wrap a single representative call in torch.profiler -> chrome trace."""
    if not args.trace:
        fn()
        return
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(args.trace)
    print(f"\n[trace] wrote {args.trace}  "
          f"-> feed to the `llm-torch-profiler-analysis` skill")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["decode", "prefill_absorbed", "prefill_ragged",
                 "sparse_decode", "all"],
        default="decode",
        help="decode | prefill_absorbed (FlashMLA decode kernel reused for "
        "prefill) | prefill_ragged (pure flashinfer fallback) | sparse_decode "
        "(DSA/V3.2 fp8 sparse -- the only native FlashMLA decode path on B300) "
        "| all",
    )
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--seq-k", type=int, nargs="+", default=[4096, 16384, 32768],
                   help="decode KV length (long-context regime)")
    p.add_argument("--seq-q", type=int, nargs="+", default=[2048, 8192],
                   help="prefill query length")
    p.add_argument("--heads", type=int, nargs="+", default=[128],
                   help="q heads after TP (128 for TP=1, 16 for TP=8, ...)")
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    p.add_argument("--trace", type=str, default=None,
                   help="export a chrome trace of one representative iter")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Run on the B300 box.")
    maj, minr = device_cap()
    print(f"device: {torch.cuda.get_device_name()}  SM {maj}.{minr}  "
          f"torch {torch.__version__}")
    if maj >= 10 and args.dtype in ("bf16", "fp16"):
        print(f"WARNING: on SM{maj}.{minr} (Blackwell) FlashMLA has NO "
              f"{args.dtype} dense-decode kernel -- flashmla rows will show n/a "
              f"with 'Dense MLA is not supported on SM100'. Re-run with "
              f"--dtype fp8 for the path that actually exists on B300.")
    else:
        print(f"note: FlashMLA dense decode is built for SM90 (bf16/fp16) and "
              f"SM90+ FP8; on Blackwell only the FP8 dense path exists. "
              f"trtllm-gen MLA targets SM100/120.")

    if args.mode in ("decode", "all"):
        run_decode(args)
    if args.mode in ("prefill_absorbed", "all"):
        run_prefill_absorbed(args)
    if args.mode in ("prefill_ragged", "all"):
        run_prefill_ragged(args)
    if args.mode in ("sparse_decode", "all"):
        run_sparse_decode(args)

    if args.trace:
        # Trace one representative case. For prefill_* modes use s_q (q_len>1);
        # otherwise the long-context decode case.
        dtype = _DTYPES[args.dtype]
        if args.mode.startswith("prefill"):
            s_q = s_k = args.seq_q[-1]
        else:
            s_q, s_k = 1, args.seq_k[-1]
        q, kv, bt, cs, _ = build_decode_inputs(args.batch[0], s_q, s_k, args.heads[0], dtype)
        fm, fm_note = make_flashmla_decode(q, kv, bt, cs, args.heads[0])
        tg, tg_note = make_trtllm_decode(q, kv, bt, cs, s_k)
        target = fm or tg
        if target is None:
            print(f"[trace] no runnable kernel for this shape/dtype "
                  f"(flashmla: {fm_note or 'n/a'}; trtllm: {tg_note or 'n/a'})")
        else:
            for _ in range(5):
                target()
            torch.cuda.synchronize()
            maybe_trace(args, target)


if __name__ == "__main__":
    main()
