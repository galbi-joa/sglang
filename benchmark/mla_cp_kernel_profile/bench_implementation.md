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
| `sparse_prefill` | DSA 稀疏 prefill / EXTEND（q_len>1） | `flash_mla_sparse_fwd` | `trtllm_ragged_attention_deepseek`（dense） |

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

- **sparse_prefill / EXTEND**
  - FlashMLA：q `(s_q, H, 576)` bf16，kv `(s_kv, 1, 576)` bf16，indices `(s_q, 1, topk)`
    （s_q = new // cp_size，s_kv = cached + new）
  - trtllm：q `(s_q, H, 192)`，k `(s_kv, H, 192)`，v `(s_kv, H, 128)`，
    q 与 kv 用各自的 cu_seqlens（q 跨度为 new，kv 跨度为 cached+new，从而构成 extend）

## 5. 当前已知问题（公平性）

这是本基准目前最需要注意的部分。**FlashMLA 与 trtllm 在 `sparse_prefill` 中并不是
在做完全相同的运算**，存在以下差异：

### 5.1 sparse 与 dense 不同

- FlashMLA 走 `flash_mla_sparse_fwd`，是 **sparse**：只 attend topk（2048）个 KV。
- trtllm 走 `trtllm_ragged_attention_deepseek`，是 **dense**：attend 整段 KV。

当 KV 很长（如 100k）而 topk=2048 时，二者的计算量相差可达数十倍。也就是说当前
对比在计算量上对 FlashMLA 有利（读得少），但即便如此，小尺寸下实测 FlashMLA 仍更慢，
这一点本身值得记录。

### 5.2 absorbed 与 unfolded 不同，且 trtllm 漏算了 kv_b_proj

MLA 的 latent 有两种处理方式：

- **absorbed（FlashMLA）**：把权重吸收到 query 一侧，直接在 latent 空间（576）做
  attention，不展开 K/V。kernel 直接吃 latent。
- **unfolded（trtllm）**：在进入 kernel 之前，模型层先用 `kv_b_proj` 把 latent
  （`kv_a`, 512）展开成普通的 K/V（`forward_mha.py` 中
  `kv = self.kv_b_proj(kv_a)` → k_nope/v，再 cat 出 192 维的 K）。kernel 只看到
  普通 MHA（q/k/v = 192/192/128），并不知道这来自 MLA latent。

由此带来一个测量上的偏差：trtllm 路径在实际部署中包含一次 **`kv_b_proj` 的展开
GEMM**，而本基准只测了 attention kernel，没有计入这次 GEMM。也就是说：

- FlashMLA 总开销 ≈ attention kernel
- trtllm 总开销 ≈ kv_b_proj GEMM + attention kernel

当前基准把 trtllm 的展开成本漏掉了，对 trtllm 有利。若要做端到端（部署视角）的公平
对比，需要把 `kv_b_proj` GEMM 计入 trtllm 路径。

### 5.3 输入数据独立生成

两个后端各自用独立的随机张量，因此 `sparse_prefill` 无法做 cosine diff 交叉校验。
这对时间测量影响不大，但无法验证两者是否在算“同一件事”。

### 5.4 维度不一致

| 项 | FlashMLA（absorbed sparse） | trtllm（unfolded dense） |
|----|----------------------------|--------------------------|
| q head_dim | 576 | 192 |
| 输出 head_dim | 512 | 128 |
| KV 读取 | topk × 576 | full × 192 × H |

## 6. 可选的修正方向

针对第 5 节的问题，有三个方向：

1. **两端都用 dense（apples-to-apples）**：FlashMLA 改用其 dense prefill kernel
   （`dense_prefill_fwd`，对应 SM100 的 `fmha_cutlass_fwd_sm100.cu`），与 trtllm
   ragged 同为 full attention，计算量一致，且可做 cosine diff 校验。
2. **两端都用 sparse**：确认 `flashinfer` 的 prefill 接口是否提供 sparse top-k
   参数；若有，则两端都做 topk sparse。
3. **保持现状，但明确标注**：在输出与文档中写明“FlashMLA=sparse、trtllm=dense，
   且 trtllm 未计入 kv_b_proj 展开成本”，把当前对比定位为“sglang 中两条可选路径的
   对比”，而非“同一运算的 kernel 对决”。

## 7. 文件说明

- `bench_mla_kernels.py`：核心微基准，包含上述 5 个模式、timing harness、CSV 导出。
- `run_all.sh`：一次性跑完所有模式，按模式输出 log 与 CSV，并合并出
  `<stamp>_combined.csv`。
- `profile_e2e_cp.sh`：启动真实 DeepSeek 服务（开启 `--enable-prefill-context-parallel`）
  并抓取 torch profiler trace，用于验证微基准是否反映真实部署。
- `benchmarkno1.md`：记录整个调查的推理过程与背景结论。
- 本文件 `bench_implementation.md`：基准的设计逻辑、输入构造与当前问题。
