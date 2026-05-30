# Benchmark Note #1 — MLA Attention Kernel Profiling under the CP Scenario

> English version first, 中文版本在下方.

---

## English

### 1. Where this started

Someone's claim:

> *"On B300, FlashMLA is much slower than trtllm-gen in the CP (Context
> Parallel) scenario. First confirm the claim, then profile to find the
> difference, then optimize FlashMLA."*

Before we can profile anything, we had to answer a chain of "what does this
actually mean in SGLang?" questions. This note records the reasoning so the
benchmark files don't look like they appeared out of nowhere.

### 2. The chain of facts we established (in order)

**(a) FlashMLA and trtllm-gen are two interchangeable attention backends.**
SGLang selects an attention kernel via `--attention-backend <name>`. Both
`flashmla` and `trtllm_mla` are registered in
`python/sglang/srt/layers/attention/attention_registry.py`. They are competing
implementations of the *same* MLA attention, so it is meaningful to compare
them head-to-head.

**(b) trtllm-gen's call site is in flashinfer — confirmed.**
The `trtllm_mla` backend is a thin wrapper. Every trtllm-gen kernel entry point
is a `flashinfer.*` call (e.g.
`trtllm_mla_backend.py:830` →
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen")`).
SGLang contains no definition of these functions; they come from the pinned
dependency `flashinfer_python==0.6.11.post1` (`python/pyproject.toml`).
FlashMLA, by contrast, calls SGLang's own compiled `sgl_kernel.flash_mla`.

**(c) flashinfer is just a kernel-library dependency, not something special.**
Many SGLang backends (`flashinfer`, `flashinfer_mla`, `trtllm_mla`,
`trtllm_mha`) are built on flashinfer. `FlashMLABackend` even *inherits* from
`FlashInferMLAAttnBackend`. So flashinfer showing up everywhere is normal — it
is the shared kernel toolbox.

**(d) CP is a model-level concern, NOT a kernel feature.**
This was the single most important realization. The actual CP work — splitting
the sequence across ranks and re-gathering the latent KV — happens at the model
level:
- `deepseek_v2.py:1819 rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(...)`
- invoked from the absorbed-MLA prepare in `forward_mla.py:384`

Once CP has reassembled the full KV, the attention kernel just receives a
complete KV tensor and computes attention. The kernel does not know — and does
not need to know — that CP happened. Therefore **"trtllm-gen handles CP" is
false**; CP is handled one layer above the kernel.

**(e) SGLang does support CP, but narrowly.**
CLI flags exist (`--enable-prefill-context-parallel`,
`--enable-dsa-prefill-context-parallel`, `--attention-context-parallel-size`),
docs exist (`docs/basic_usage/deepseek_v32.md`), and CI tests exist
(`test/registered/cp/`). But it is **prefill-only**, DeepSeek-family only, and
**verified on Hopper with the fa3 backend** (`server_args.py:1925`).

**(f) Per-backend CP wiring differs.**
In `attention_backend_handler.py`, the shared handler `_handle_attention_backend`
(used by fa3 / flashinfer / flashmla) has a CP branch at the top:
`if mla_use_prefill_cp(...): return MLA(absorbed)`. But
`handle_attention_trtllm_mla` has **no** such branch — it routes extend to
`MHA_CHUNKED_KV`, which skips the `rebuild_cp_kv_cache` gather (that gather lives
only in the absorbed-MLA path). So trtllm-gen's CP-*prefill* path is not wired
today; its decode path is unaffected because CP clears its metadata before
decode anyway.

### 3. The conclusion that defines the benchmark

Putting (d)–(f) together:

- CP is the *stage*, not the thing being measured. The thing the mentor calls
  "slow" is the **attention kernel** that runs inside a CP deployment.
- The fair, decisive comparison is **kernel vs kernel**, split by phase:

| phase | FlashMLA backend runs | trtllm_mla backend runs |
|-------|-----------------------|-------------------------|
| decode  | `flash_mla_with_kvcache` (sgl_kernel, DeepSeek FlashMLA) | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` |
| prefill | flashinfer ragged (falls back via `super().forward_extend`) | `flashinfer.prefill.trtllm_ragged_attention_deepseek` |

- **decode is the real battleground.** prefill is largely a
  flashinfer-vs-flashinfer comparison because FlashMLA does not run its own
  kernel for pure prefill.

### 4. The #1 hypothesis (architecture gating)

Evidence from the build and tests:
- `sgl-kernel/cmake/flashmla.cmake` builds FlashMLA **dense decode only for
  sm90 (Hopper)**; its sm100 source list is **sparse-only**.
- `test_flashmla.py` gates on `is_sm90_supported` (SM90).
- `test_trtllm_mla.py` gates on `_REQUIRED_MAJOR = 12` (Blackwell, B200/B300).

So on **B300 (Blackwell, SM10x)** FlashMLA dense decode may have **no native
fast path**, while trtllm-gen has a Blackwell-optimized kernel in flashinfer.
This is the leading explanation for "FlashMLA is much slower" and is exactly
what the benchmark is designed to confirm or refute.

### 5. The exact kernel I/O shapes (DeepSeek-V3 MLA)

Constants: `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`,
`v_head_dim=128`, page size `64`.

**Decode (absorbed MLA — the kernel sees the latent dim, not 192):**
- q: `(B, 1, H, 576)` where `576 = kv_lora_rank + qk_rope = 512 + 64`
- kv_cache: FlashMLA `(N_blk, 64, 1, 576)` vs trtllm-gen `(N_blk, 1, 64, 576)`
  (note the differing head-axis position / page layout)
- output: `(B, 1, H, 512)`, finally viewed as `(B, H*128)`
- softmax scale uses the original `192**-0.5`, not `576`.

**Prefill (non-absorbed, ragged, causal):**
- q/k: `(T, H, 192)`, v: `(T, H, 128)`, `cu_seqlens` ragged layout.

### 6. What each file does and why

- **`benchmark/mla_cp_kernel_profile/bench_mla_kernels.py`** — the core
  microbenchmark. Splits decode vs prefill; builds a *shared* paged KV so both
  backends attend to identical data; times with CUDA events + per-iteration L2
  flush (cold-cache, kernel-fair, median of 50); **cross-checks outputs via
  cosine diff** so a timing row is only trusted when both kernels computed the
  same thing; optionally emits a Chrome trace for the
  `llm-torch-profiler-analysis` skill. A row printing `n/a` is itself a finding
  (that backend has no path on this device/shape).

- **`benchmark/mla_cp_kernel_profile/profile_e2e_cp.sh`** — end-to-end
  validation. Launches a real DeepSeek server with
  `--enable-prefill-context-parallel` and captures a torch-profiler trace, so we
  can confirm the isolated microbench reflects an actual CP deployment and see
  which kernel each phase lands on.

- **`benchmark/mla_cp_kernel_profile/README.md`** — usage and result-reading
  guide.

### 7. Status / next steps

- Scripts are syntax-checked only; they **must be run on the B300 box** (this
  dev box has no GPU/torch).
- Next: run the decode sweep, read `speedup = flashmla_ms / trtllm_ms` vs
  `seq_k`, validate `cos_diff` is small, and note any `n/a`. Then decide whether
  the fix is a new Blackwell dense-decode kernel (in the external
  `sgl-project/FlashMLA` repo, wired via `flashmla.cmake`) or a backend
  dispatch/wrapper change in `flashmla_backend.py`.

---

## 中文

### 1. 起点

有些人说：

> *"在 B300 上，CP（Context Parallel，上下文并行）场景下 FlashMLA 比 trtllm-gen
> 慢很多。先确认这个 claim，然后 profile 找出差异，再优化 FlashMLA。"*

在做任何 profile 之前，我们必须先回答一连串"这在 SGLang 里到底是什么意思？"的
问题。这份笔记记录了推理过程，这样这些 benchmark 文件就不会看起来像是凭空冒出来的。

### 2. 我们依次确认的事实链

**(a) FlashMLA 和 trtllm-gen 是两个可互换的 attention 后端。**
SGLang 通过 `--attention-backend <名称>` 选择 attention kernel。`flashmla` 和
`trtllm_mla` 都注册在
`python/sglang/srt/layers/attention/attention_registry.py` 里。它们是*同一个*
MLA attention 的相互竞争的实现，所以拿来正面对比是有意义的。

**(b) trtllm-gen 的调用点在 flashinfer —— 已确认。**
`trtllm_mla` 后端是一个薄封装。每一个 trtllm-gen kernel 入口都是 `flashinfer.*`
调用（例如 `trtllm_mla_backend.py:830` →
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen")`）。
SGLang 里没有这些函数的定义；它们来自被 pin 住的依赖
`flashinfer_python==0.6.11.post1`（`python/pyproject.toml`）。相比之下，
FlashMLA 调用的是 SGLang 自己编译的 `sgl_kernel.flash_mla`。

**(c) flashinfer 只是一个 kernel 库依赖，并不特殊。**
SGLang 的许多后端（`flashinfer`、`flashinfer_mla`、`trtllm_mla`、`trtllm_mha`）
都建立在 flashinfer 之上。`FlashMLABackend` 甚至*继承*自
`FlashInferMLAAttnBackend`。所以到处都看到 flashinfer 是正常的 —— 它就是共享的
kernel 工具箱。

**(d) CP 是模型层的事，不是 kernel 的功能。**
这是最重要的一个认知。真正的 CP 工作 —— 把序列切分到各个 rank、再把 latent KV
重新汇总 —— 发生在模型层：
- `deepseek_v2.py:1819 rebuild_cp_kv_cache()` → `cp_all_gather_rerange_output(...)`
- 由 `forward_mla.py:384` 的 absorbed-MLA prepare 调用

一旦 CP 重新拼好完整的 KV，attention kernel 只是接收一个完整的 KV 张量然后计算
attention。kernel 不知道、也不需要知道 CP 发生过。因此 **"trtllm-gen 负责 CP" 是
错的**；CP 是在 kernel 上面一层处理的。

**(e) SGLang 确实支持 CP，但范围很窄。**
有 CLI 参数（`--enable-prefill-context-parallel`、
`--enable-dsa-prefill-context-parallel`、`--attention-context-parallel-size`），
有文档（`docs/basic_usage/deepseek_v32.md`），也有 CI 测试
（`test/registered/cp/`）。但它是**仅 prefill**、仅 DeepSeek 系列、并且**在
Hopper 上用 fa3 后端验证过**（`server_args.py:1925`）。

**(f) 各后端的 CP 接线不同。**
在 `attention_backend_handler.py` 里，共享 handler `_handle_attention_backend`
（fa3 / flashinfer / flashmla 使用）开头有一个 CP 分支：
`if mla_use_prefill_cp(...): return MLA(absorbed)`。但
`handle_attention_trtllm_mla` **没有**这个分支 —— 它把 extend 路由到
`MHA_CHUNKED_KV`，从而跳过了 `rebuild_cp_kv_cache` 的汇总（那个汇总只存在于
absorbed-MLA 路径里）。所以 trtllm-gen 的 CP-*prefill* 路径目前没有接线；它的
decode 路径不受影响，因为 CP 在 decode 之前本来就会清空它的元数据。

### 3. 决定 benchmark 形态的结论

把 (d)–(f) 合起来：

- CP 是*舞台*，不是被测量的对象。导师说"慢"的那个东西是在 CP 部署里运行的
  **attention kernel**。
- 公平、决定性的对比是**按阶段拆分的 kernel 对 kernel**：

| 阶段 | FlashMLA 后端运行 | trtllm_mla 后端运行 |
|------|-------------------|---------------------|
| decode  | `flash_mla_with_kvcache`（sgl_kernel，DeepSeek FlashMLA） | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` |
| prefill | flashinfer ragged（通过 `super().forward_extend` 回退） | `flashinfer.prefill.trtllm_ragged_attention_deepseek` |

- **decode 才是真正的战场。** prefill 在很大程度上是 flashinfer 对 flashinfer 的
  比较，因为 FlashMLA 在纯 prefill 时不运行自己的 kernel。

### 4. 头号假设（架构 gating）

来自构建和测试的证据：
- `sgl-kernel/cmake/flashmla.cmake` 只为 **sm90（Hopper）构建 FlashMLA dense
  decode**；它的 sm100 源文件列表是**只有 sparse**。
- `test_flashmla.py` 用 `is_sm90_supported`（SM90）做 gating。
- `test_trtllm_mla.py` 用 `_REQUIRED_MAJOR = 12`（Blackwell，B200/B300）做 gating。

所以在 **B300（Blackwell，SM10x）** 上，FlashMLA dense decode 可能**没有原生的
快速路径**，而 trtllm-gen 在 flashinfer 里有一个针对 Blackwell 优化的 kernel。
这是"FlashMLA 慢很多"的首要解释，也正是这个 benchmark 要去确认或推翻的东西。

### 5. 精确的 kernel 输入/输出 shape（DeepSeek-V3 MLA）

常量：`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、
`v_head_dim=128`、page size `64`。

**Decode（absorbed MLA —— kernel 看到的是 latent 维度，不是 192）：**
- q: `(B, 1, H, 576)`，其中 `576 = kv_lora_rank + qk_rope = 512 + 64`
- kv_cache：FlashMLA 是 `(N_blk, 64, 1, 576)`，trtllm-gen 是 `(N_blk, 1, 64, 576)`
  （注意 head 轴位置 / page 布局不同）
- 输出：`(B, 1, H, 512)`，最终 view 成 `(B, H*128)`
- softmax scale 用的是原始的 `192**-0.5`，不是 `576`。

**Prefill（非 absorbed、ragged、causal）：**
- q/k：`(T, H, 192)`，v：`(T, H, 128)`，`cu_seqlens` ragged 布局。

### 6. 每个文件做什么、为什么

- **`benchmark/mla_cp_kernel_profile/bench_mla_kernels.py`** —— 核心
  microbenchmark。拆分 decode 与 prefill；构建一份*共享*的 paged KV 让两个后端
  attend 到完全相同的数据；用 CUDA events + 每次迭代 L2 flush 计时（冷缓存、对
  kernel 公平、取 50 次中位数）；**通过 cosine diff 交叉校验输出**，只有当两个
  kernel 算的是同一个东西时，那一行的计时才可信；可选地导出 Chrome trace 供
  `llm-torch-profiler-analysis` skill 使用。某一行打印 `n/a` 本身就是一个发现
  （说明该后端在这个设备/shape 上没有路径）。

- **`benchmark/mla_cp_kernel_profile/profile_e2e_cp.sh`** —— 端到端验证。用
  `--enable-prefill-context-parallel` 启动一个真实的 DeepSeek 服务器并捕获
  torch-profiler trace，这样我们可以确认隔离的 microbench 是否反映真实的 CP
  部署，并看到每个阶段落到哪个 kernel。

- **`benchmark/mla_cp_kernel_profile/README.md`** —— 使用方法与结果解读指南。

### 7. 状态 / 下一步

- 脚本仅做了语法检查；它们**必须在 B300 机器上运行**（这台开发机没有
  GPU/torch）。
- 下一步：跑 decode sweep，看 `speedup = flashmla_ms / trtllm_ms` 随 `seq_k`
  的变化，验证 `cos_diff` 足够小，并记录任何 `n/a`。然后决定修复方向是写一个新的
  Blackwell dense-decode kernel（在外部 `sgl-project/FlashMLA` 仓库里，通过
  `flashmla.cmake` 接线），还是在 `flashmla_backend.py` 里改后端的
  dispatch/wrapper。
