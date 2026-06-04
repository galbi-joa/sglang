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

- **sparse_decode（fp8 KV）**
  - q: `(B, 1, H, 576)` bf16（仅 K cache 量化为 fp8，q 保持 bf16）
  - kv: 经 `quantize_k_cache` 量化后的 fp8 布局
  - indices: `(B, 1, topk)` int32，指向被选中的 KV 行
  - head 数会按 64/128 的倍数 padding（与 dsa_backend 在 Blackwell 上的处理一致）

- **sparse_prefill / EXTEND**（两端均为 absorbed + sparse，已对齐 DSA 实际路径）
  - FlashMLA：`flash_mla_sparse_fwd`，q `(s_q, H, 576)` bf16，kv `(s_kv, 1, 576)` bf16，
    indices `(s_q, 1, topk)`（s_q = new // cp_size，s_kv = cached + new）
  - trtllm：`trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=topk)`，
    q `(1, s_q, H, 576)` bf16 absorbed，kv 为 latent paged cache
    `(num_blocks, 1, 64, 576)` bf16，block_tables `(1, s_q, topk)` 指向被选中的 KV
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

## 7. 文件说明

- `bench_mla_kernels.py`：核心微基准，包含上述 5 个模式、timing harness、CSV 导出。
- `run_all.sh`：一次性跑完所有模式，按模式输出 log 与 CSV，并合并出
  `<stamp>_combined.csv`。
- `profile_e2e_cp.sh`：启动真实 DeepSeek 服务（开启 `--enable-prefill-context-parallel`）
  并抓取 torch profiler trace，用于验证微基准是否反映真实部署。
- `benchmarkno1.md`：记录整个调查的推理过程与背景结论。
- 本文件 `bench_implementation.md`：基准的设计逻辑、输入构造与当前问题。
