# Changelog — sparse_prefill 修正与测量记录

本文件记录 `mla_cp_kernel_profile` 基准在 sparse prefill / extend 路径上的几次
关键修改：改了哪些函数、测试逻辑如何重写，以及修正后的实测结果。

测试硬件：NVIDIA B300 SXM6（SM 10.3），torch 2.12.0+cu130，sgl_kernel 由 CUDA 13
从源码重新编译，flashinfer 0.6.x（trtllm-gen kernel 来源）。

---

## 1. 改了哪些函数

### 1.1 `make_trtllm_sparse_prefill`（核心修正）

**问题**：早先版本调用了错误的 flashinfer API。原来用的是
`flashinfer.prefill.trtllm_ragged_attention_deepseek`，这是 **V3（普通 MLA）的
dense ragged MHA** 路径——需要先用 `kv_b_proj` 把 latent 展开成 192/128 的普通
K/V。它与 FlashMLA 的 absorbed sparse 路径在三方面都不对等：sparse vs dense、
absorbed vs unfolded、维度（576/512 vs 192/128）不同。

**依据**：核查 sglang 中 V3.2（DSA）的实际代码
`dsa_backend.py:_forward_trtllm`（prefill 时以 `is_prefill=True` 调用）确认，
DSA 的 trtllm 路径并不调用 ragged dense kernel，而是与 decode 用同一个
absorbed-sparse MLA kernel：

```python
# dsa_backend.py:2152
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    query=q,                                 # absorbed, head_dim 576
    kv_cache=kv,                             # latent KV（不展开）
    sparse_mla_top_k=self.dsa_index_topk,    # sparse，只 attend topk
    backend="trtllm-gen",
)
```

注意函数名虽含 `decode`，但 prefill 也用它——`is_prefill` 只改变 page table 的
变换方式，attention kernel 本身相同（即“用 decode kernel 做 prefill”，FlashMLA
与 trtllm 都是这个模式）。

**修改后**：`make_trtllm_sparse_prefill` 改为调用
`trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=topk)`，输入为 absorbed
query `(1, s_q, H, 576)` bf16、latent paged KV `(nblk, 1, 64, 576)`、topk
block_tables `(1, s_q, topk)`。这样与 `flash_mla_sparse_fwd` 对齐：两端都是
absorbed + topk sparse，head_dim 576 → 512。

### 1.2 `make_flashmla_sparse_prefill`

扩展为 **extend** 语义：s_q 个新 query token attend 到 s_kv = cached + new 的总
KV，并按 64/128 对 head 数做 padding（Blackwell 上为 128，与 dsa_backend 一致）。

### 1.3 `run_sparse_prefill`

重写为 extend + context parallel 版本：

- `--cached-len`：已在 KV 中的前缀长度；`--seq-q`：新进入的 token 数；
  总 KV = cached + new。
- `--cp-size N`：context parallel 把 new token 切到 N 个 rank，单 rank 的
  query 长度为 `seq-q // N`，KV 仍为完整长度。测量的是单个 rank 的那次 kernel。
- 输出列：`NEW / CACHED / KV_TOT / q_per_rank / flashmla / trtllm / speedup`。

### 1.4 `_verify_sparse_prefill`（新增，正确性校验）

由于两个 kernel 的 KV 布局不同（flashmla flat vs trtllm paged），且各自独立
生成输入，无法直接判断是否在算“同一件事”。新增 `--verify`：

- 用 **同一份** latent KV（flat 给 flashmla，同一张量 reshape 成 paged 给
  trtllm，page_size=64 时二者等价）、**同一个** q、**同一组**绝对 topk indices，
  分别跑两个 kernel，比较输出的 cosine diff。
- 实测 `cos_diff = 2.666e-06`（[OK]），说明两个 kernel 在共享输入下输出一致，
  时间比较因此有效。

### 1.5 其他

- `--csv`：所有模式可导出统一表格（含每个 backend 的耗时、note、speedup）。
- `run_all.sh`：一次跑完所有模式并合并出 `*_combined.csv`；sparse_prefill 默认
  90k cached + 10k new，可用 `CACHED_LEN / NEW_LEN / CP_SIZE` 覆盖。

## 2. 测试逻辑

- 计时：CUDA events + warmup + 每次迭代 L2 flush，取 50 次中位数。
- 正确性：`--verify` 在共享输入下做 cosine diff（仅 sparse_prefill）。
- sparse prefill / extend 两端均为 absorbed + topk(2048) sparse，输入规格对齐，
  属于“同一运算的 kernel 对比”。

## 3. 实测结果（B300，已通过 --verify）

speedup = flashmla_ms / trtllm_ms。**小于 1 表示 FlashMLA 更快。**

### 3.1 不同 context 长度（new=10000, cp=1, H=128）

| cached | KV_TOT | flashmla(ms) | trtllm(ms) | speedup |
|--------|--------|--------------|------------|---------|
| 30000  | 40000  | 4.661 | 5.377 | 0.87 |
| 90000  | 100000 | 5.367 | 5.983 | 0.90 |
| 200000 | 210000 | 5.749 | 6.370 | 0.90 |
| 500000 | 510000 | 6.004 | 6.560 | 0.92 |

### 3.2 不同 CP size（cached=90000, new=10000, H=128）

| cp_size | q/rank | flashmla(ms) | trtllm(ms) | speedup |
|---------|--------|--------------|------------|---------|
| 1 | 10000 | 5.352 | 5.962 | 0.90 |
| 2 | 5000  | 2.656 | 2.899 | 0.92 |
| 4 | 2500  | 1.186 | 1.356 | 0.87 |
| 8 | 1250  | 0.582 | 0.670 | 0.87 |

### 3.3 不同 new token 数（cached=90000, cp=1, H=128）

| new | KV_TOT | flashmla(ms) | trtllm(ms) | speedup |
|-----|--------|--------------|------------|---------|
| 1000  | 91000  | 0.473  | 0.518  | 0.91 |
| 5000  | 95000  | 2.680  | 2.912  | 0.92 |
| 10000 | 100000 | 5.525  | 5.987  | 0.92 |
| 20000 | 110000 | 11.150 | 12.181 | 0.92 |

### 3.4 sparse_decode（q_len=1, fp8 KV, H=128）

注意：decode 的 q_len=1，与 prefill（q_len 大）方向相反。

| B | S_K | flashmla(ms) | trtllm(ms) | speedup |
|---|-----|--------------|------------|---------|
| 1  | 4096  | 0.039 | 0.016 | 2.37 |
| 1  | 16384 | 0.037 | 0.016 | 2.25 |
| 1  | 32768 | 0.037 | 0.016 | 2.26 |
| 4  | 4096  | 0.045 | 0.018 | 2.44 |
| 4  | 16384 | 0.045 | 0.020 | 2.21 |
| 4  | 32768 | 0.045 | 0.021 | 2.20 |
| 16 | 4096  | 0.049 | 0.023 | 2.18 |
| 16 | 16384 | 0.049 | 0.025 | 1.99 |
| 16 | 32768 | 0.049 | 0.025 | 2.00 |

在 q_len=1 的 decode 下，**trtllm 反而快约 2 倍**。绝对时间都很小
（0.016~0.049ms），此时固定开销（launch / 元数据）占主导，而非实际计算量。

### 3.5 q_len 扫描：交叉点（cached=90000, cp=1, H=128）

固定 KV，仅改变 query 长度，观察 speedup 随 q_len 的变化：

| new (q_len) | flashmla(ms) | trtllm(ms) | speedup |
|-------------|--------------|------------|---------|
| 1    | 0.033 | 0.016 | 2.00 |
| 16   | 0.041 | 0.027 | 1.54 |
| 64   | 0.051 | 0.047 | 1.09 |
| 256  | 0.147 | 0.152 | 0.97 |
| 1024 | 0.476 | 0.583 | 0.82 |
| 4096 | 2.199 | 2.379 | 0.92 |

**交叉点在 q_len ≈ 64~256 之间**：q_len < 64 时 trtllm 更快，q_len ≥ 256 时
FlashMLA 更快，q_len≈64 时基本持平（1.09）。

## 4. 结论与观察

1. **谁更快取决于 q_len（即 decode vs prefill）**，不能一概而论：
   - **q_len 小（decode / 短 extend，< ~64）**：trtllm 更快，q_len=1 时约 2 倍。
   - **q_len 大（prefill / 长 extend，≥ ~256）**：FlashMLA 更快，约 8~18%。
   - 交叉点约在 q_len 64~256。

2. **物理解释**：q_len 小时，绝对时间很小（几十微秒），由固定开销
   （kernel launch / 元数据）主导，trtllm 的固定开销更低；q_len 大时，由实际
   attention 计算量主导，FlashMLA 的 sparse kernel 在 B300 上算得略快。

3. **关于最初 claim**：导师“flashmla 比 trtllm 慢”的说法**在 decode（q_len 小）
   时成立，在 prefill（q_len 大）时相反**。而早先测到的“慢 4.5 倍”是用了错误的
   API（sparse 对 dense ragged）造成的，并非 kernel 本身——见第 1.1 节。

4. **对 KV 长度不敏感**：无论 decode 还是 prefill，KV 从几 k 增到 500k+，时间
   几乎不变（sparse 只 attend topk=2048）。主要成本来自 q_len。

5. **CP 不改变结论**：CP 只是把 new token 切到各 rank，两个 kernel 的 q 同等
   变小，speedup 比值在 cp=1~8 下保持 ~0.9 不变（见 3.2）。即 FlashMLA 在
   prefill 更快是 kernel 本身的特性，与 CP 无关。

6. **正确性**：sparse_prefill 在 `--verify` 下 cos_diff = 2.666e-06，两个 kernel
   共享输入时输出一致，时间对比有效。sparse_decode 的 2 倍差异尚未做同样的
   cos_diff 校验，是后续应补的一项。

## 5. 限制

- 仍未对每个时间数据点逐一做 cos_diff（`--verify` 用较小 q、capped KV 抽样校验）。
- 仅测 attention kernel，不含 RoPE / KV 写入 / topk indexer 等前后处理。
- dense decode 路径在 B300 上 FlashMLA 无 kernel（`n/a`），属设备能力结论，
  与本次 sparse 对比无关。
