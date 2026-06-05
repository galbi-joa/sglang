# bench_implementation_v2.md —— sparse_decode 的 fp8/fp8 对齐：原始状态、修复过程与结果

> 本文是 `bench_implementation.md` 的补充（v2），专门记录 **`sparse_decode` 模式**从
> "精度不对等"到 "fp8 query + fp8 KV 公平对比"的完整过程：**原来哪一端是 bf16、对齐时
> 踩了什么坑、根因是什么、最终如何修、以及修后在 B300 上的实测结果。**
>
> 测试硬件：NVIDIA B300（Blackwell，SM 10.3），torch 2.12.0+cu130，`sgl_kernel` 由 CUDA 13
> 源码重编译，flashinfer 0.6.x（trtllm-gen kernel 来源）。
> 被比较对象：FlashMLA（`sgl_kernel.flash_mla`）vs trtllm-gen（`flashinfer.decode`）。
> speedup = `flashmla_ms / trtllm_ms`，**大于 1 表示 FlashMLA 更慢**。

---

## 1. 一句话结论

`sparse_decode` 原来 **trtllm 一端用 bf16 KV**（FlashMLA 用 fp8），精度不对等；把 trtllm KV
改成 fp8 时**只改了 KV、漏改 query**，导致 `Missing TRTLLM-GEN kernel`。根因是
**trtllm 要求 query dtype 必须等于 KV dtype**。把 query 也改成 fp8 后，两端均 fp8 q + fp8 KV，
对比成立，且与 production 的 DSA fp8 路径一致。

---

## 2. sparse_decode 的 dtype 演化（三阶段，带 commit）

| 阶段 | commit | flashmla（q / KV） | trtllm（q / KV） | 状态 |
|------|--------|--------------------|-------------------|------|
| A 原始 | `f0ddd71` → `1b1632d` → `d58a91b` | **bf16** / **fp8**(packed) | **bf16** / **bf16** | 能跑，但**精度不对等** |
| B 对齐尝试 | `81a768a` | bf16 / fp8(packed) | **bf16** / **fp8**(平铺) | ❌ `Missing TRTLLM-GEN kernel` |
| C 正确修复 | `3a9567a` | bf16 / fp8(packed) | **fp8** / **fp8**(平铺) | ✅ 两端 fp8，可比 |

> 说明："原来是 bf16 的"指 **trtllm 的 KV**（以及两端的 query 起初都是 bf16）。FlashMLA 的 KV
> 从一开始就是 fp8（`quantize_k_cache`），query 一直是 bf16——这是 FlashMLA 稀疏 decode 的
> native 形态（commit `1b1632d`："keep q in bf16, only K cache is fp8"）。问题出在把这个
> "**q 保持 bf16**" 的假设错误地套用到了 trtllm 一端。

---

## 3. 阶段 A —— 原始状态：为什么不公平

- FlashMLA：`flash_mla_with_kvcache(q=bf16, k_cache=fp8 packed, is_fp8_kvcache=True, indices=topk)`。
  这是 DSA decode 的真实路径（`dsa_backend.py:_forward_flashmla_kv` :1816，KV 用 `quantize_k_cache`）。
- trtllm：`trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=topk)`，但 **KV 是 bf16**。
- **不公平点**：FlashMLA 读 fp8 KV（每行半字节量级，内存流量减半），trtllm 读 bf16 KV（全字节）。
  这是 memory-bound 的稀疏 decode，KV 精度直接影响读带宽，故时间对比有偏。

阶段 A 实测（旧 `changelog.md` 3.4，trtllm 仍 bf16 KV）：

| B | S_K | flashmla(ms) | trtllm(ms, **bf16 KV**) | speedup |
|---|-----|------|------|------|
| 1 | 4096 | 0.039 | 0.016 | 2.37 |
| 1 | 16384 | 0.037 | 0.016 | 2.25 |
| 1 | 32768 | 0.037 | 0.016 | 2.26 |
| 4 | 4096 | 0.045 | 0.018 | 2.44 |
| 4 | 16384 | 0.045 | 0.020 | 2.21 |
| 4 | 32768 | 0.045 | 0.021 | 2.20 |
| 16 | 4096 | 0.049 | 0.023 | 2.18 |
| 16 | 16384 | 0.049 | 0.025 | 1.99 |
| 16 | 32768 | 0.049 | 0.025 | 2.00 |

---

## 4. 阶段 B —— 对齐尝试与踩的坑

为了消除精度不对等，把 `make_trtllm_sparse_decode` 的 **KV cast 成 fp8**（平铺 576 fp8），
但 **query 仍保持 bf16**（沿用阶段 A 对 FlashMLA 的 "q 保持 bf16" 假设）。结果 B300 报错：

```
Missing TRTLLM-GEN kernel (decode): ... headDimQk=576, headDimV=512,
sparseMlaType=1, numTokensPerPage=1, multiCtasKvMode=2 ...
```

当时一度想得出 **"fp8/fp8 不可能，退回 bf16"** 的结论——**这是错的**。

---

## 5. 根因分析（修复过程中真正搞清楚的事）

1. **`Missing TRTLLM-GEN kernel` 是 kernel 派发未命中（找不到匹配 cubin），不是 KV 布局错误。**
   布局错只会触发 shape 断言或算出错误数值，绝不会报 "Missing kernel"。所以"两端 fp8 布局
   不兼容"的方向是错的。

2. **真正的差异是 query dtype。** flashinfer 的 `trtllm_fmha_kernel_launcher.cu` 强制
   **query dtype 必须等于 KV dtype**（`ICHECK_EQ(kv_data_type, q_data_type)`，且只能是
   BF16 或 FP8 E4M3）。`bf16 query + fp8 KV` 是 trtllm-gen 根本没编译过的混合精度组合 → 派发失败。
   阶段 A 的 bf16 之所以能跑，正是因为那时 query 与 KV **都是 bf16**（一致）。

3. **production 的 DSA fp8 路径本就是 fp8 query + fp8 KV。** `dsa_backend.py:_forward_trtllm`
   在 fp8 分支调用 `mla_quantize_and_rope_for_fp8`（:2079），把 **query 也量化成 fp8**
   （返回的 `merged_q_out` 为 `float8_e4m3fn`，见 `utils.py:387` 的 docstring :425）。

4. **trtllm 的 fp8 KV 不是 packed 布局。** `calculate_mla_kv_cache_dim`
   （`model_runner_kv_cache_mixin.py:182`）对 TRTLLM 后端返回 `kv_lora_rank + qk_rope_head_dim
   = 576`（:193–201，注释 "excluding TRTLLM"），即平铺 576 fp8，不是 FlashMLA 的
   `quantize_k_cache` packed（656B）。所以基准给 trtllm 的"平铺 fp8 cast"恰好与 production 一致。

---

## 6. 阶段 C —— 正确修复

把 `make_trtllm_sparse_decode` 的 query 也 cast 到 fp8，与 fp8 KV 对齐（commit `3a9567a`）：

```python
# query dtype 必须等于 fp8 KV 的 dtype；production 的 q 是 mla_quantize_and_rope_for_fp8 的 fp8 输出
q = torch.randn(b, 1, h_q, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda").to(FP8_DTYPE)
```

修复后两端的 native fp8 路径：

| backend | query | KV |
|---------|-------|-----|
| flashmla sparse decode | **bf16** | fp8（quantize_k_cache packed，656B） |
| trtllm  sparse decode  | **fp8**  | fp8（平铺 576） |

> query dtype 不同**不是"不公平"**，而是两个 kernel 在 fp8 下各自的真实路径：FlashMLA 的
> 稀疏 decode native 就是 bf16 q + fp8 K；trtllm 要求 q 与 KV 同 dtype（fp8 q + fp8 KV）。
> production 正是如此运行。

---

## 7. fp8/fp8 实测结果（阶段 C，B300，已通过派发验证）

speedup = `flashmla_ms / trtllm_ms`，**大于 1 表示 FlashMLA 更慢**。数据取自 `--csv` 精确值。

| B | S_K | flashmla(ms) | trtllm(ms, **fp8**) | speedup |
|---|-----|------|------|------|
| 1 | 4096 | 0.0388 | 0.0144 | 2.70 |
| 1 | 16384 | 0.0389 | 0.0144 | 2.71 |
| 1 | 32768 | 0.0369 | 0.0143 | 2.57 |
| 4 | 4096 | 0.0451 | 0.0164 | 2.75 |
| 4 | 16384 | 0.0451 | 0.0164 | 2.74 |
| 4 | 32768 | 0.0451 | 0.0164 | 2.75 |
| 16 | 4096 | 0.0492 | 0.0205 | 2.40 |
| 16 | 16384 | 0.0492 | 0.0205 | 2.40 |
| 16 | 32768 | 0.0492 | 0.0205 | 2.40 |

### 7.1 解读
- **decode（q_len=1）下 trtllm 快约 2.4~2.75 倍**草（——在 fp8/fp8 同精度下）
- **与阶段 A（trtllm bf16）对比**：flashmla 不变（0.037~0.049，本就 fp8 KV），trtllm 从
  0.016~0.025 → **0.014~0.021**（更快）——fp8 KV 把 KV 读带宽减半的效果。于是 speedup 反而
  **从 ~2.0~2.44 拉大到 ~2.4~2.75**。
- **对 KV 长度不敏感**（4k~32k 几乎不变）——稀疏只 attend topk=2048，与总 KV 长无关。
- **batch 增大 speedup 略收窄**（2.7→2.4）——两端固定开销随 batch 摊薄、绝对时间变大，计算占比上升。

### 7.2 物理解释（为什么 decode 下 trtllm 赢）
q_len=1 时绝对时间极小（十几 µs），由**固定开销**（kernel launch / 元数据 / GEMV 退化导致
tensor core 利用率极低）主导，而非实际计算量。trtllm-gen 的 decode kernel 固定开销更低，故更快。
（这与 `sparse_prefill` 的方向相反——见第 8 节。）

---

## 8. 与 sparse_prefill 的方向一致性（q_len 交叉点）

| 模式 | q_len | 谁快 | 比值 |
|------|-------|------|------|
| sparse_decode | 1 | **trtllm** | ~2.4~2.75× |
| sparse_prefill / extend | 1250（cp=8, 90k+10k）| **flashmla** | ~0.90 |

- q_len 小（decode）：固定开销主导，trtllm 赢。
- q_len 大（prefill/长 extend）：进入 roofline，FlashMLA 的**专用稀疏 prefill kernel**
  （`flash_mla_sparse_fwd`）略胜 trtllm 把 decode kernel 拿来做 prefill。
- **交叉点约在 q_len 64~256**（见 `changelog.md` 3.5 的 q_len 扫描）。

> 注：sparse_decode 与 sparse_prefill 的 fp8 处理不同——decode 是 fp8 KV（两端对齐 fp8），
> prefill 的 FlashMLA kernel `flash_mla_sparse_fwd` 要求 **bf16 KV**，故 sparse_prefill 两端
> 都用 bf16，本就一致，无需对齐。

---

## 9. 仍存在的限制
1. **未对每个时间点逐一做 cos_diff 数值校验**：sparse_decode 两端独立生成随机输入、且 q dtype
   不同（bf16 vs fp8），不便直接 cosine diff；目前只比时间。（`--verify` 仅覆盖 sparse_prefill。）
2. **不含前后处理**：只测 attention kernel，不含 RoPE / KV 写入 / topk indexer / fp8 量化本身。
3. **fp8 数值精度**：基准用单位 descale（bmm1_scale=softmax_scale），值与 production 的逐张量
   descale 不同；这影响数值精度，但**不影响 kernel 派发与计时**。

---

## 10. 复现命令（B300）

```bash
cd benchmark/mla_cp_kernel_profile

# fp8 q + fp8 KV 的 sparse decode（阶段 C）
python bench_mla_kernels.py --mode sparse_decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --csv sd_fp8.csv
```
输出表的 speedup = flashmla/trtllm；trtllm 列应在 0.014~0.021ms 量级，不再报 `Missing TRTLLM-GEN kernel`。

---

## 11. 代码与 commit 引用

| 内容 | 位置 |
|------|------|
| 基准 trtllm sparse decode 实现 | `bench_mla_kernels.py: make_trtllm_sparse_decode`（q→fp8 在此） |
| 基准 flashmla sparse decode 实现 | `bench_mla_kernels.py: make_flashmla_sparse_decode`（q 保持 bf16） |
| production DSA decode（fp8 q + fp8 KV） | `dsa_backend.py: _forward_trtllm`（def 2046；`mla_quantize_and_rope_for_fp8` 2079；调用 2152） |
| production FlashMLA decode（bf16 q + fp8 K） | `dsa_backend.py: _forward_flashmla_kv`（def 1776；调用 1816；`quantize_k_cache` 1809） |
| fp8 query 量化 | `python/sglang/srt/layers/attention/utils.py: mla_quantize_and_rope_for_fp8`（def 387） |
| trtllm fp8 KV dim = 576（不 override） | `model_runner_kv_cache_mixin.py: calculate_mla_kv_cache_dim`（def 182；193–201） |
| q dtype 必须 == KV dtype | flashinfer `csrc/trtllm_fmha_kernel_launcher.cu`（`ICHECK_EQ(kv_data_type, q_data_type)`，外部包） |

### 相关 commit
| commit | 作用 |
|--------|------|
| `1b1632d` | sparse_decode：q 保持 bf16，只 K cache fp8（针对 FlashMLA，正确） |
| `d58a91b` | 加入 trtllm-gen sparse decode（初始 trtllm KV 为 bf16） |
| `81a768a` | 把 trtllm KV 对齐到 fp8（但漏改 query → 阶段 B 的 Missing kernel） |
| `3a9567a` | **修复：trtllm query 也改 fp8（q dtype 必须等于 KV dtype）**；并订正文档 |

---

## 12. 结论
1. 原来 trtllm 一端是 **bf16 KV**（不对等）；对齐时只改 KV、漏改 query，触发 `Missing TRTLLM-GEN kernel`。
2. 根因是 **trtllm 要求 query dtype == KV dtype**（派发约束），不是布局不兼容，更不是 "fp8/fp8 不可能"。
3. 修复 = **query 也改 fp8**，与 production（`mla_quantize_and_rope_for_fp8`）一致。
4. fp8/fp8 下 **decode 区间 trtllm 快约 2.4~2.75 倍**，且因 fp8 KV 进一步降低 trtllm 的内存流量，
   speedup 比阶段 A（trtllm bf16）更大；与 sparse_prefill 的 q_len 交叉点结论一致。
