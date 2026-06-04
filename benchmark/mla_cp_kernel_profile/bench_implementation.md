# Benchmark 实现说明（bench_implementation.md）

本文档说明 `mla_cp_kernel_profile` 下基准测试的设计逻辑：测量什么、目的是什么、
测试输入如何构造，以及当前已知的问题。面向阅读者为后续维护与复核本基准的人员。

测试目标硬件为 NVIDIA B300（Blackwell，SM 10.3）。被比较的两个对象是
FlashMLA（sgl_kernel）与 trtllm-gen（flashinfer 提供的 kernel）。

---

## 1. 测量对象

基准测试比较的是 **attention kernel 本身的执行时间**，分 MLA 的不同阶段与不同
实现路径。每个被测形状（shape）记录两个后端的耗时与二者的比值（speedup = a / b）。

被测的两个后端：

- FlashMLA：来自 `sgl_kernel`，需用 CUDA 13 从源码重新编译后才在 B300 上有 kernel。
- trtllm-gen：来自 `flashinfer` 包，通过 `flashinfer.decode.*` / `flashinfer.prefill.*`
  调用，运行时 JIT 编译出 SM103 的 cubin。

## 2. 测量方法（timing harness）

为保证 GPU 计时准确，`time_kernel` 使用三项措施：

1. **CUDA events**：用 `torch.cuda.Event` 在 GPU 时间线上打点，测量真实的 kernel
   执行时间，避免异步调用导致只测到“提交时间”。
2. **warmup**：正式测量前先空跑若干次，排除首次调用的一次性开销（kernel 加载、
   JIT 编译、显存池初始化、GPU 时钟升频）。
3. **L2 flush**：每次迭代前用一块大缓冲区清空 L2 cache，避免重复调用时数据驻留
   缓存导致的非真实加速。

取 50 次迭代的中位数（median），以减小偶发抖动的影响。

部分模式还做 **cosine diff 交叉校验**：只有当两个后端算出的结果一致时，那一行的
时间对比才被视为有效。

## 3. 测试模式（运行场景）

当前共 5 个模式，通过 `--mode` 选择：

| 模式 | 测量内容 | FlashMLA 调用 | trtllm 调用 |
|------|----------|---------------|-------------|
| `decode` | q_len=1 的 absorbed MLA decode | `flash_mla_with_kvcache`（dense） | `trtllm_batch_decode_with_kv_cache_mla` |
| `prefill_absorbed` | 用 decode kernel 做 prefill（q_len>1） | `flash_mla_with_kvcache`（dense） | 同上 decode kernel |
| `prefill_ragged` | 纯 ragged prefill 回退路径 | flashinfer ragged wrapper | `trtllm_ragged_attention_deepseek` |
| `sparse_decode` | DSA/V3.2 稀疏 decode（fp8 KV） | `flash_mla_with_kvcache(indices=...)` | `trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=...)` |
| `sparse_prefill` | DSA 稀疏 prefill / EXTEND（q_len>1） | `flash_mla_sparse_fwd` | `trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=...)`（absorbed sparse） |

模式设计的背景：

- 在 B300 上，FlashMLA 的 **dense decode kernel 不存在**（源码里 SM100 没有
  dense decode 实例），因此 `decode` / `prefill_absorbed` 的 FlashMLA 列在
  B300 上为 `n/a`，仅 trtllm 有数。这本身是一个结论。
- `sparse_decode` 与 `sparse_prefill` 是 B300 上 FlashMLA 实际能运行的路径
  （SM100 编译了 sparse decode 与 sparse prefill kernel）。
- `sparse_prefill` 用于复现导师指出的“sglang 用 FlashMLA 的 kernel 做 prefill”
  这一情形。

### 3.1 EXTEND 场景与 context parallel（CP）

按导师反馈，`sparse_prefill` 被扩展为 **extend** 语义，而非纯 prefill：

- `--cached-len`：已经在 KV cache 中的前缀长度（cached prefix）。
- `--seq-q`：本次新进入的 query token 数（new）。
- 总 KV 长度 = cached + new。例如 90k cached + 10k new。

`--cp-size N` 反映 context parallel：CP 把 **new token** 切分到 N 个 rank 上，
因此单个 rank 的 query 长度为 `seq-q // N`，而它仍然 attend 到完整的（all-gather
后的）KV。基准测量的是 **单个 rank 实际执行的那次 kernel 调用**。CP 跨 rank 的
KV all-gather 是模型层的集合通信，不属于 kernel，本基准不计入。

## 4. 测试输入（shape 构造）

DeepSeek-V3 MLA 常量：

```
kv_lora_rank      = 512
qk_nope_head_dim  = 128
qk_rope_head_dim  = 64
head_dim (absorbed q/kv) = 512 + 64 = 576
head_dim_v (输出)        = 512
head_dim (non-absorbed)  = 128 + 64 = 192
page_size = 64
DSA topk  = 2048
```

各模式的输入形状：

- **decode / prefill_absorbed（absorbed 形式）**
  - q: `(B, s_q, H, 576)`，decode 时 s_q=1，prefill_absorbed 时 s_q>1
  - kv（paged latent）: `(num_blocks, page_size, 1, 576)`，head 轴为 1（MQA 共享 latent）
  - 输出: `(B, s_q, H, 512)`
  - dtype 可选 bf16 / fp8

- **prefill_ragged（non-absorbed 形式）**
  - q/k: `(T, H, 192)`，v: `(T, H, 128)`，T 为 batch 内所有 query token 之和
  - 用 `cu_seqlens` 表示 ragged 边界

- **sparse_decode（fp8 KV）**——两端 q dtype 不同，各自的 native fp8 路径：
  - FlashMLA：q `(B, 1, H, 576)` **bf16** + K cache `quantize_k_cache` packed fp8
    （`is_fp8_kvcache=True`；其 native 路径就是 bf16 q + fp8 K）
  - trtllm：q `(B, 1, H, 576)` **fp8** + 平铺 576 fp8 KV。trtllm-gen 强制
    `query dtype == KV dtype`（flashinfer launcher 的 `ICHECK_EQ`），bf16 q + fp8 KV
    会因找不到 cubin 而报 `Missing TRTLLM-GEN kernel`；production 经
    `mla_quantize_and_rope_for_fp8` 把 q 也量化为 fp8，故此处 q 用 fp8。
  - indices: `(B, 1, topk)` int32，指向被选中的 KV 行
  - head 数会按 64/128 的倍数 padding（与 dsa_backend 在 Blackwell 上的处理一致）

- **sparse_prefill / EXTEND**（两端均为 absorbed + sparse，已对齐 DSA 实际路径）
  - 批量 B 个请求，每个请求 s_q 个新 query token（s_q = new // cp_size），attend 到
    s_kv = cached + new 的总 KV；两端共享同一份 latent KV（sparse 每个 query 只读 topk
    行，共享与否读取流量相同，对计时有代表性）。head 数两端都按 64/128 倍数 padding。
  - FlashMLA：`flash_mla_sparse_fwd`，q `(B*s_q, H_pad, 576)` bf16（扁平 query 行），
    kv `(s_kv, 1, 576)` bf16，indices `(B*s_q, 1, topk)`
  - trtllm：`trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=topk)`，
    q `(B, s_q, H_pad, 576)` bf16 absorbed，kv 为 latent paged cache
    `(num_blocks, 1, 64, 576)` bf16，block_tables `(B, s_q, topk)`，seq_lens 长度 B
  - 两端 head_dim 同为 576、输出 512、均只 attend topk 个 KV

## 5. 公平性：一次重要修正

### 5.1 修正的问题（曾经的错误）

早先版本的 `sparse_prefill` 在 trtllm 一侧调用了**错误的 API**：用了
`flashinfer.prefill.trtllm_ragged_attention_deepseek`，这是 **V3（普通 MLA）的
dense ragged MHA 路径**，需要先用 `kv_b_proj` 把 latent 展开成 192/128 的普通
K/V。这与 FlashMLA 的 absorbed sparse 路径在三个方面都不对等：

- sparse vs dense（FlashMLA 只读 topk，trtllm 读整段）
- absorbed vs unfolded（trtllm 多了一次 `kv_b_proj` 展开 GEMM，且本基准没计入）
- 维度不一致（576/512 vs 192/128）

### 5.2 正确路径（V3.2 / DSA）

核查 sglang 中 DSA 的实际代码（`dsa_backend.py:_forward_trtllm`，prefill 时以
`is_prefill=True` 调用）后确认：**DSA 的 trtllm 路径并不调用 ragged dense kernel，
而是与 decode 用同一个 absorbed-sparse MLA kernel**：

```python
# dsa_backend.py:2152
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    query=q,                       # absorbed, head_dim 576
    kv_cache=kv,                   # latent KV（不展开）
    sparse_mla_top_k=self.dsa_index_topk,   # sparse，只 attend topk
    backend="trtllm-gen",
)
```

即在 DSA 中，trtllm 也用 absorbed latent（576）+ topk sparse，且同样是“用 decode
API 做 prefill”（`is_prefill` 只影响 page table 的变换方式）。

### 5.3 修正后的状态

`make_trtllm_sparse_prefill` 已改为调用
`trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=topk)`，与 FlashMLA 对齐：

| 项 | FlashMLA（`flash_mla_sparse_fwd`） | trtllm（`trtllm_batch_decode_..._mla`） |
|----|------------------------------------|------------------------------------------|
| 计算类型 | absorbed + sparse(topk) | absorbed + sparse(topk) |
| q head_dim | 576 | 576 |
| 输出 head_dim | 512 | 512 |
| KV 表示 | latent 576 | latent 576 |
| kv_b_proj 展开 | 无 | 无 |

5.1 中列出的三项不对等因此基本消除，`sparse_prefill` 成为同一运算的 kernel 对比。
`sparse_decode` 模式此前已使用正确的 API，无需改动。

### 5.4 仍存在的次要限制

- **输入数据独立生成**：两端用各自的随机张量，因此 `sparse_prefill` /
  `sparse_decode` 仍无法做 cosine diff 数值校验（只比时间，不验证数值一致）。
- **decode/prefill_absorbed 的 dense 路径**：这两个模式在 B300 上 FlashMLA dense
  decode kernel 不存在，仍为 `n/a`，这是设备能力的结论而非公平性问题。

## 6. 后续可选项

1. **数值校验**：让两端共享同一份 latent KV 与 topk indices，从而可做 cosine diff，
   验证两个 kernel 输出一致（目前只验证时间）。
2. **端到端口径**：如需对比整条注意力路径而非单个 kernel，可把 RoPE、KV 写入、
   topk indexer 等前后处理一并计入。当前仅测 attention kernel。

## 7. 三个 sparse/prefill 场景的数学差异

通用 attention 原型（单个 query `q`）：

```
o = softmax( q · K_S^T / sqrt(d) ) · V_S
```

`sparse_decode` / `prefill_ragged` / `sparse_prefill` 沿三个维度区分：
(A) 集合 `S` 是哪些 KV（**sparse** 只取 topk vs **dense** 取全部）；
(B) head 是否 **absorbed**（latent 576/512 vs 展开后的 192/128）；
(C) query 数 `q_len`。

### 7.1 sparse_decode（DSA/V3.2，q_len=1）
- `S` = top-2048：indexer 从全部 `S_K` 中选出的 2048 个 KV，**sparse**。
- **absorbed latent**：`d_qk=576`（512 latent + 64 rope），`d_v=512`，不展开 per-head KV。
- `q_len=1`；因果性由 indexer 只选过去 token 实现，**无三角 mask**。
- `FLOPs ≈ 2H·(topk·d_qk + topk·d_v) = O(H·topk·d)`，**与 S_K 无关**
  （实测 4k~32k 时间平坦即此因）。

### 7.2 prefill_ragged（dense MHA，q_len=S_q=2k~8k，bf16）
- `S` = 全部过去 key：每个 query 因果地 attend 所有过去 K，**dense + 三角 mask**。
- **unfolded MHA**：latent 经 `kv_b_proj` 展开成 per-head `K(192)`/`V(128)`，
  `d_qk=192`，`d_v=128`，**不 absorb**。ragged 用 `cu_seqlens` 表示变长边界。
- `q_len=S_q`，且此处 `S_kv=S_q`（纯 prefill 无缓存）。
- `FLOPs ≈ 2H·(1/2)·S_q·S_kv·(d_qk+d_v) = O(H·S_q²·d)`，**对 S_q 平方**
  （实测 2k→8k，约 4x 长度对应约 12x 时间）。

### 7.3 sparse_prefill / extend（CP，q/rank=1250）
- `S` = top-2048（与 7.1 相同，**sparse**）。
- **absorbed latent** 576/512（与 7.1 相同）。
- `q_len=1250`：把 7.1 推广到 1250 个 query，每个在 100k KV（cached 90k + new 10k）
  中取自己的 topk 2048。CP 把 10000 个 new token 切到 8 rank → 每 rank `q_len=1250`，
  KV 为 all-gather 后的完整 100k。
- `FLOPs ≈ 2H·q_len·(topk·d_qk + topk·d_v) = O(H·q_len·topk·d)`，
  **对 q_len 线性，与总 KV 长度无关**。

### 7.4 一览表

| | S（attend 哪里） | head 表示 | d_qk/d_v | q_len | KV 范围 | FLOPs |
|---|---|---|---|---|---|---|
| sparse_decode | top-2048（sparse） | absorbed | 576/512 | 1 | topk of S_K | O(H·topk·d) |
| prefill_ragged | 全部（dense, causal） | **unfolded** | **192/128** | S_q | 全 S_kv=S_q | O(H·S_q²·d) |
| sparse_prefill | top-2048（sparse） | absorbed | 576/512 | 1250 | topk of 100k | O(H·q_len·topk·d) |

### 7.5 关键直觉
- **sparse_decode ↔ sparse_prefill 是同一个 sparse-absorbed kernel**，差别只在
  `q_len`（1 vs 1250）；后者就是"把 sparse decode 拉长到 prefill 长度"。两者都与
  KV 总长无关、对 q_len 线性。
- **prefill_ragged 根本不同**：dense（读全 KV 而非 topk）+ unfolded（192/128，需
  `kv_b_proj` 展开 GEMM 与额外显存）+ 对 S_q 平方。即便 q_len 相同也远贵，且随长度
  平方暴涨。
- 因此 sparse_prefill（0.58ms，q_len 1250）比 prefill_ragged（2.1ms，q_len 8192）
  便宜得多：sparse 把 KV 工作量降到 dense 的 `topk/S_kv`（此处约 1/50）。
- 三者 softmax scale 均为 `1/sqrt(192)`（`SOFTMAX_SCALE = HEAD_DIM_QK**-0.5`）；即便
  absorbed（576），scale 仍按原始 192 维（见 bench 常量注释）。

## 8. 各模式的实现方法（详解）

通用结构：每个模式有一个 `build_*` 输入构造 + 两个 `make_*`（每后端一个）返回
`(run_fn, note)`；`time_kernel(run_fn)` 计时；部分模式做 cosine diff 交叉校验。
kernel 不可用时 `make_*` 返回 `(None, 原因)`，该行显示 `n/a` 并把原因写入 note 列，
**不中断 sweep**。

### 8.1 decode（`run_decode` → `_bench_absorbed_pair`）
- 输入 `build_decode_inputs(b, s_q=1, s_k, h, dtype)`：q `(b,1,h,576)`；paged latent
  kv `(b*nblk,64,576)`；block_table `(b,nblk)`；cache_seqlens=`[s_k]*b`。fp8 时 q 与
  kv **一起** cast 到 fp8（dtype 一致）。
- FlashMLA：`flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens,
  head_dim_v=512, tile_md, num_splits, softmax_scale, causal=True, descale_q/k,
  is_fp8_kvcache)`。B300 上 bf16/fp8 dense decode kernel **不存在** → 捕获异常返回
  note（`n/a`）。
- trtllm：`trtllm_batch_decode_with_kv_cache_mla(query, kv_cache=(nblk,1,64,576),
  qk_nope=128, kv_lora=512, qk_rope=64, block_tables, seq_lens, max_seq_len,
  bmm1_scale=softmax_scale)`。
- 交叉校验：仅当两端都计时成功时，各再调用一次取输出做 cos_diff，并用 `_safe_call`
  防止失败 kernel 二次抛错中断 sweep。B300 上 flashmla n/a，故 cos_diff 为空。

### 8.2 prefill_absorbed（`run_prefill_absorbed` → `_bench_absorbed_pair`）
- 与 decode **完全相同**的两个 kernel，仅 `s_q>1` 且 `s_k=s_q`（纯 prefill，causal）。
  对应 sglang 在 dispatch 选 `AttnForwardMethod.MLA` 时"用 decode kernel 做 prefill"。
- FlashMLA 列同样 `n/a`（复用的就是缺失的 dense decode kernel）。

### 8.3 prefill_ragged（`run_prefill_ragged`）
- 输入 `build_prefill_inputs(b, s_q, h, dtype)`：q/k `(b*s_q,h,192)`，v `(b*s_q,h,128)`，
  `cu_seqlens`、`seq_lens`。
- FlashMLA 回退：`BatchPrefillWithRaggedKVCacheWrapper(workspace,"NHD").plan(
  qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk=192,
  head_dim_vo=128, causal=True, sm_scale, q_data_type).run(q,k,v)`。这是 dispatch 不选
  absorbed-MLA 时 flashmla backend 实际回退到的 flashinfer ragged。
- trtllm：`trtllm_ragged_attention_deepseek(query, key, value, workspace, seq_lens,
  max_q_len, max_kv_len, bmm1_scale, bmm2_scale=1, batch_size, cum_seq_lens_q/kv,
  is_causal=True, ...)`。
- 交叉校验：两端都 ok 时做 cos_diff（实测约 7e-7，确认是同一运算）；同样用 `_safe_call`
  与 `r.ok` 守卫，避免失败 kernel 二次抛错。

### 8.4 sparse_decode（`run_sparse_decode`）
- 输入 `build_sparse_inputs(b, s_k, h, topk=2048)`：q `(b,1,h,576)` bf16；kv
  `(b*nblk,64,1,576)` bf16 经 `quantize_k_cache` 量化成 FlashMLA 的 **packed fp8**；
  绝对 topk 索引 `(b,1,topk)`。另外为 trtllm 单独构造 **平铺 fp8** KV `(b*nblk,1,64,576)`。
- FlashMLA：head pad 到 128，`flash_mla_with_kvcache(q_bf16, k_cache=packed_fp8,
  block_table=空(b,0), cache_seqlens, head_dim_v=512, tile_md, num_splits,
  softmax_scale, is_fp8_kvcache=True, indices=topk)`。q 保持 **bf16**（其 native fp8
  路径就是 bf16 q + fp8 K）。
- trtllm：q **cast 到 fp8**（必须与 fp8 KV 的 dtype 一致，否则 `Missing TRTLLM-GEN
  kernel`），`trtllm_batch_decode_with_kv_cache_mla(query_fp8, kv_cache=平铺576 fp8,
  ..., block_tables=topk索引, sparse_mla_top_k=topk, bmm1_scale)`。
- 不做 cos_diff（两端独立随机输入且 q dtype 不同）；只比时间。

### 8.5 sparse_prefill / extend（`run_sparse_prefill`）
- CP 建模：`cp=--cp-size`，`cached=--cached-len`；对每个 `(b, new, h)`：
  `kv_tot=cached+new`，`q_per_rank=max(1, new//cp)`（量纲就是单 rank 的 query 数）。
- FlashMLA：`make_flashmla_sparse_prefill(b, q_per_rank, kv_tot, h, topk)` →
  `flash_mla_sparse_fwd(q=(b*q_per_rank, H_pad, 576), kv=(kv_tot,1,576),
  indices=(b*q_per_rank,1,topk), sm_scale, d_v=512)`。
- trtllm：`make_trtllm_sparse_prefill(b, q_per_rank, kv_tot, h, topk)` →
  `trtllm_batch_decode_with_kv_cache_mla(query=(b,q_per_rank,H_pad,576),
  kv_cache=(nblk,1,64,576), block_tables=(b,q_per_rank,topk), seq_lens=[kv_tot]*b,
  max_seq_len=kv_tot, sparse_mla_top_k=topk, bmm1_scale)`。
- 两端共享同一份 latent KV 模型；head 两端都 pad 到 64/128 倍数（公平）。
- `--verify`：`_verify_sparse_prefill` 用**同一份** latent KV（flat 给 flashmla，
  reshape 成 paged 给 trtllm，page_size=64 时二者等价）、**同一** q、**同一组**绝对
  topk 索引分别跑两端，比较 cos_diff（实测约 1e-6，确认同一运算）。为控制参考成本，
  verify 用较小 q（≤256）与 capped KV（≤8192）。

## 9. 文件说明

- `bench_mla_kernels.py`：核心微基准，包含上述 5 个模式、timing harness、CSV 导出。
- `run_all.sh`：一次性跑完所有模式，按模式输出 log 与 CSV，并合并出
  `<stamp>_combined.csv`。
- `profile_e2e_cp.sh`：启动真实 DeepSeek 服务（开启 `--enable-prefill-context-parallel`）
  并抓取 torch profiler trace，用于验证微基准是否反映真实部署。
- `benchmarkno1.md`：记录整个调查的推理过程与背景结论。
- 本文件 `bench_implementation.md`：基准的设计逻辑、输入构造与当前问题。
