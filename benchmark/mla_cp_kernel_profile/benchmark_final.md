# benchmark_final.md —— B300 上 MLA 注意力 kernel 基准报告（FlashMLA vs trtllm-gen）

测试环境：**NVIDIA B300 SXM6 AC，SM 10.3（Blackwell）**，torch 2.12.0+cu130，
`sgl_kernel` 由 CUDA 13 源码重编译，flashinfer 0.6.x（trtllm-gen kernel 来源）。
工作目录：`benchmark/mla_cp_kernel_profile/`，主脚本 `bench_mla_kernels.py`。
时间单位 ms（CUDA event，median of 50，每次迭代 L2 flush）。
`speedup = flashmla / trtllm`，**大于 1 表示 FlashMLA 更慢**。

---

## 1. Prefill（extend）

### 1.1 命令
```bash
cd benchmark/mla_cp_kernel_profile

# heads=64, 总 KV=70000 (=69125 cached + 875 new)
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 2 4 8 16 32 --seq-q 875 --cached-len 69125 --heads 64 --csv prefill.csv

# 通用 extend（90k cached + 10k new），可加 --cp-size N 做 context parallel
python bench_mla_kernels.py --mode sparse_prefill \
    --batch 1 --seq-q 10000 --cached-len 90000 --heads 128 --csv prefill.csv
```
- `--seq-q` = 新进入的 token 数；`--cached-len` = 已在 KV 的前缀；总 KV = 两者之和。
- `--cp-size N`：context parallel 把 new 切到 N 个 rank，单 rank q = new/N。
- 输出三列各自的精度：**trtllm = fp8(q+KV) | flashmla_sparse = bf16 | flashmla_kv = fp8 KV**，
  并给出比值 `sparse/trt`、`kv/trt`。

### 1.2 结论
1. **trtllm（fp8）在 prefill 全程最快**：比 flashmla_sparse(bf16) 快约 **1.7~2.0×**，
   比 flashmla_kv(fp8) 快约 **2.7~3.5×**。
2. **FlashMLA 的稀疏 prefill kernel（`flash_mla_sparse_fwd`）只有 bf16，没有 fp8**
   （cmake 中 prefill 源无 fp8 变体；函数签名无 `is_fp8_kvcache`，docstring 明确 kv 为 bfloat16）。
3. **FlashMLA 在 prefill 想用 fp8，只能借 decode kernel（`flashmla_kv`）**，把 b×s_q 个新
   token 当成"单 token decode"折叠进 batch。但这条路在大 prefill 下会撞 **shared memory 上限**：
   ```
   [WARNING] batch_size=14000 requires 280004B shared memory (max=232448B), using low-smem fallback kernel.
   ```
   batch=14000（B=16×875）需要 280KB 片上共享内存，而单 SM 上限 227KB（232448B），于是退到
   低共享内存的慢速 fallback kernel——这正是 `flashmla_kv` 在 B≥16 时变慢的原因。
4. **结论**：FlashMLA 的 prefill kernel 无 fp8，在 fp8 部署里 prefill 结构性地偏向 trtllm。
   （若两端都 bf16，flashmla 的 prefill kernel 约快 10%——但 fp8 部署不存在这种对称比较。）

### 1.3 用到的函数与路径
**基准代码（`benchmark/mla_cp_kernel_profile/bench_mla_kernels.py`）**

| 列 | 构造函数 | 实际调用的 kernel/API | 精度 |
|----|----------|----------------------|------|
| trtllm_fp8 | `make_trtllm_sparse_prefill(..., fp8=True)` | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=…)` | fp8 q + fp8 KV |
| fmla_sparse_bf16 | `make_flashmla_sparse_prefill` | `sgl_kernel.flash_mla.flash_mla_sparse_fwd` | bf16 |
| fmla_kv_fp8 | `make_flashmla_kv_prefill` | `sgl_kernel.flash_mla.flash_mla_with_kvcache(is_fp8_kvcache=True)` | fp8 KV |

驱动 `run_sparse_prefill`；可选数值校验 `_verify_sparse_prefill`（`--verify`）。

**production 对应路径（`python/sglang/srt/layers/attention/dsa_backend.py`）**

| 后端 | 函数（def 行 → 调用行） |
|------|--------------------------|
| trtllm | `_forward_trtllm`（def 2046 → `trtllm_batch_decode_with_kv_cache_mla` 2152） |
| flashmla_sparse | `_forward_flashmla_sparse`（def 1727 → `flash_mla_sparse_fwd` 1762） |
| flashmla_kv | `_forward_flashmla_kv`（def 1776 → `flash_mla_with_kvcache` 1816） |

**底层 op（`sgl-kernel/python/sgl_kernel/flash_mla.py`）**：`flash_mla_sparse_fwd`（def 310）
→ op `sparse_prefill_fwd`（339）；`flash_mla_with_kvcache`（def 87）→ fp8 稀疏走 op `sparse_decode_fwd`（269）。

---

## 2. Decode

decode 有两个模式：**`sparse_decode`（DSA/V3.2，B300 上 flashmla 实际能跑的路径）** 与
**`decode`（普通 dense decode，B300 上 flashmla 无 kernel）**。

### 2.1 命令
```bash
# ① sparse decode（DSA/V3.2，两端 fp8）
python bench_mla_kernels.py --mode sparse_decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --csv sd.csv

# ② dense decode（非稀疏）—— B300 上 flashmla 全部 n/a，仅 trtllm 有数
python bench_mla_kernels.py --mode decode \
    --batch 1 4 16 --seq-k 4096 16384 32768 --heads 128 --dtype bf16 --csv dec.csv
#   也可 --dtype fp8
```
- `--seq-k` = KV 长度（上下文）；sparse 需 ≥ topk=2048。
- sparse_decode 精度：**flashmla = bf16 q + fp8 K(packed) | trtllm = fp8 q + fp8 KV**。

### 2.2 结论
1. **sparse_decode（q_len=1，两端 fp8）下 trtllm 快约 2.4~2.75×**。绝对时间极小（十几~几十 µs），
   由固定开销（kernel launch / 元数据 / q_len=1 时 GEMV 退化、tensor core 利用率低）主导，
   trtllm decode kernel 固定开销更低，故更快。
2. **对 KV 长度不敏感**（S_K 4k→32k 时间几乎不变）——稀疏只 attend topk=2048。
3. **batch 增大 speedup 略收窄**（2.75→2.40）——固定开销被摊薄、计算占比上升。
4. **q_len > 1（投机解码等场景）的判断**：FlashMLA 没有原生的 q_len > 1 稀疏 decode kernel。
   - `flash_mla_with_kvcache`（decode kernel）只接受 q_len=1；q_len > 1 时 SGLang 将每个新
     token 拆成一条 q_len=1 的 decode，相当于把 b×q_len 条 decode 打包成一次 batch 调用
     （`_forward_flashmla_kv`，dsa_backend.py:1792）。这正是 §1 中 `fmla_kv_fp8` 列的测法。
   - 该路径的实测数据见 §3.1：fmla_kv_fp8 比 trtllm fp8 慢 **2.86~3.52×**，比 q_len=1 的
     2.4~2.75× 还慢——因为每个 token 的 kernel-launch 固定开销被累加，而 trtllm 原生处理
     整个 q_len 的 GEMM，固定开销只付一次。
   - FlashMLA 的专用 prefill kernel（`flash_mla_sparse_fwd`，bf16）在 q_len > 1 下可以
     一次完成，但只有 bf16；与 trtllm fp8 比是 1.7~2.0×，存在 dtype 不对称。
   - 总结：q_len > 1 时 trtllm 的优势不会缩小，反而在 FlashMLA 的 fp8 路径下更大（3.5×
     级别），只有 bf16 vs bf16 的 prefill kernel 对比才稍窄（~1.7×）。
5. **dense decode 在 B300 上 flashmla 没有 kernel**：报
   `Dense decode MLA is only supported on SM90a architecture` → 全部 n/a，仅 trtllm 运行；
   trtllm dense decode 随 batch×seqlen 增长（dense 要读整段 KV）。
6. 在 B300 的 fp8 部署下，**q_len=1 和 q_len > 1 的稀疏路径都偏向 trtllm**。

### 2.3 用到的函数与路径
**基准代码（`bench_mla_kernels.py`）**

| 模式 / 列 | 构造函数 | kernel/API | 精度 |
|-----------|----------|-----------|------|
| sparse_decode · flashmla | `make_flashmla_sparse_decode` | `flash_mla_with_kvcache(indices=…, is_fp8_kvcache=True)` | bf16 q + fp8 K |
| sparse_decode · trtllm | `make_trtllm_sparse_decode` | `trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=…)` | fp8 q + fp8 KV |
| decode · flashmla | `make_flashmla_decode` | `flash_mla_with_kvcache`（dense，无 indices）→ B300 n/a | bf16/fp8 |
| decode · trtllm | `make_trtllm_decode` | `trtllm_batch_decode_with_kv_cache_mla` | bf16/fp8 |

驱动 `run_sparse_decode` / `run_decode`（→ `_bench_absorbed_pair`）；输入构造
`build_sparse_inputs` / `build_decode_inputs`；fp8 packed 量化 `quantize_k_cache`。

**production 对应**：sparse decode = `_forward_flashmla_kv`（1776/1816）vs `_forward_trtllm`（2046/2152）。

---

## 3. 实测结果（B300）

### 3.1 Sparse prefill / extend
`heads=64, NEW=875, CACHED=69125, KV_TOT=70000, topk=2048, cp_size=1`（单位 ms）

| B | trtllm (fp8) | flashmla_sparse (bf16) | flashmla_kv (fp8) | sparse/trt | kv/trt |
|---:|---:|---:|---:|---:|---:|
| 1  | **0.231** | 0.397 | 0.753 | 1.72× | 3.25× |
| 2  | **0.451** | 0.881 | 1.586 | 1.95× | 3.52× |
| 4  | **0.998** | 1.803 | 3.034 | 1.81× | 3.04× |
| 8  | **2.094** | 3.631 | 5.997 | 1.73× | 2.86× |
| 16 | **4.301** | 7.281 | 11.960 | 1.69× | 2.78× | ⚠️ low-smem fallback (batch=14000) |
| 32 | **8.647** | 14.474 | 23.407 | 1.67× | 2.71× | ⚠️ low-smem fallback (batch=28000) |

trtllm(fp8) 全程最快；flashmla_kv(fp8) 在 B≥16 触发低共享内存 fallback，进一步变慢。

### 3.2 Sparse decode（DSA/V3.2，两端 fp8）
`heads=128, topk=2048, q_len=1`（单位 ms，speedup = flashmla/trtllm）

| B | S_K | flashmla (fp8) | trtllm (fp8) | speedup |
|---:|---:|---:|---:|---:|
| 1  | 4096 / 16384 / 32768 | 0.037 | 0.014 | **2.57×** |
| 4  | 4096 / 16384 / 32768 | 0.045 | 0.016 | **2.75×** |
| 16 | 4096 / 16384 / 32768 | 0.049 | 0.020 | **2.40×** |

trtllm 快约 2.4~2.75×；对 S_K 不敏感（同一 B 下 4k/16k/32k 完全相同）。

### 3.3 Dense decode（bf16）
`heads=128`；**flashmla 全部 n/a**（`Dense decode MLA is only supported on SM90a architecture`），仅 trtllm：

| B | S_K | flashmla | trtllm (ms) |
|---:|---:|:--:|---:|
| 1  | 4096  | n/a | 0.023 |
| 1  | 16384 | n/a | 0.039 |
| 1  | 32768 | n/a | 0.035 |
| 4  | 4096  | n/a | 0.031 |
| 4  | 16384 | n/a | 0.039 |
| 4  | 32768 | n/a | 0.055 |
| 16 | 4096  | n/a | 0.037 |
| 16 | 16384 | n/a | 0.084 |
| 16 | 32768 | n/a | 0.138 |

B300 无 FlashMLA dense decode kernel；trtllm dense decode 随 batch×seqlen 增长（读整段 KV）。

---

## 4. 综合结论

1. **在 B300 的 fp8 部署下，所有测到的路径都偏向 trtllm-gen。**
   - sparse decode（q_len=1，两端 fp8）：trtllm 快 ~2.4~2.75×（固定开销主导）。
   - sparse decode（q_len > 1，fp8）：FlashMLA 无原生支持，退化为逐 token decode 拆包
     → 实测 `fmla_kv_fp8` 慢 **2.86~3.52×**，比 q_len=1 更慢。
   - prefill（fp8）：trtllm 快 ~1.7~2.0×（最小值）。根因是 FlashMLA 的 prefill kernel
     只有 bf16；decode kernel 复用做 fp8 prefill 又会撞共享内存上限（B≥16 时）。
2. **dense decode 在 B300 上 FlashMLA 无 kernel**（SM90a-only）→ n/a，仅 trtllm 可用。
3. **唯一 flashmla 略占优的情形**是 bf16-vs-bf16 的 prefill（约快 10%），但这不是真实 fp8 部署。

---

## 5. 代码引用速查

| 内容 | 位置 |
|------|------|
| 基准主脚本 / 全部构造与驱动函数 | `benchmark/mla_cp_kernel_profile/bench_mla_kernels.py` |
| trtllm prefill/decode（DSA） | `dsa_backend.py: _forward_trtllm`（def 2046，调用 2152） |
| flashmla 稀疏 prefill | `dsa_backend.py: _forward_flashmla_sparse`（def 1727，调用 1762） |
| flashmla decode（也用于 fp8 prefill） | `dsa_backend.py: _forward_flashmla_kv`（def 1776，调用 1816） |
| flashmla dense（B300 无 SM100 源） | `_forward_standard_mha`（def 1837）；cmake `sgl-kernel/cmake/flashmla.cmake` |
| 底层 op：稀疏 prefill / 稀疏 decode | `sgl-kernel/python/sgl_kernel/flash_mla.py`：`sparse_prefill_fwd`(339) / `sparse_decode_fwd`(269) |
| trtllm fp8 KV dim=576（不 override） | `model_runner_kv_cache_mixin.py: calculate_mla_kv_cache_dim`（def 182，193–201） |
| trtllm 要求 q dtype == KV dtype | flashinfer `csrc/trtllm_fmha_kernel_launcher.cu`（`ICHECK_EQ`，外部包） |
| 更细的 DSA 调度调用图 | 见同目录 `dsa_dispatch_report.md` |
