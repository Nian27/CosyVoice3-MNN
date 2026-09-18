# MNN 运行时 hint 单变量 A/B — Gate 记录（2026-09-15）

**净结果：四个杠杆全部闭环，没有任何一个达到"可采纳且已验证收益"的门槛，
产品运行路径一行未改。** 三个杠杆是"根本没有作用对象"（模型/设备层面不存在），
第四个（KV 量化）有作用对象但实测为负收益且数值有损。

真机：Honor BKQ-AN90 / SM8850 / Adreno 830，adb 网络连接（`10.40.128.171:37587`）。
被测模型：`/data/local/tmp/cosy-mnn-stage2/hift-core.mnn`（= 出货的 `hift-core.fp32.mnn`，70,218,952 B）。
基准：`CosyVoiceHiFTBenchmark.out`（`core` 模式，174 帧，`cpu low`，6 线程）。

## 结论汇总

| 杠杆 | Gate | 依据 |
|---|---|---|
| `BackendConfig::Memory_High`（解除 deconv 4 线程上限） | **INERT — 不采纳** | `CPUDeconvolutionCreator::onCreate` 调用 **0 次**；同一进程内 `[conv-diag]` 触发 160 次，证明插桩本身有效 |
| `HintMode::CPU_ENABLE_KLEIDIAI` | **INERT（本机）— 不采纳** | 置 1 后 80 个卷积全部打印 `kleidiai_declined`；设备自报 `sme2: 0`，KleidiAI 的 dense/depthwise 路径无法启用 |
| `HintMode::WINOGRAD_MEMORY_LEVEL` | **INERT — 不采纳** | 0 / 1 / 3 三档下 80 个卷积的选择完全相同：79 个 `dense_tiled_winograd_off` + 1 个 `conv1x1_strassen`。该模型卷积不满足 `canUseWinograd` |
| `HintMode::ATTENTION_OPTION` | **有效杠杆，但四种量化全部不采纳** | LLM 确实使用 fused attention：`CPUAttentionCreator` 被调用 **24 次**（24 层），默认 `attentionOption=8`，经 config `attention_mode` 可配。logits 数值 A/B：int8 KV cosine ≥0.9991、TQ3/TQ4 cosine 0.89–0.97；速度 A/B：int8 KV 反而**慢 6%**（3.960→3.730 tok/s，组内噪声 0.3–0.9%）。结论保持默认 8 |

**四个杠杆中三个无效，且无效的原因是"根本没有作用对象"，不是"收益小于噪声"。**
这比墙钟 A/B 能给的结论强得多；第四个（`attentionOption`）有效，且已用与时钟无关的
logits 数值 Gate 定出 int8 可用 / TQ 不可用。

## ⚠️ 前置结论：本机当前处于**降频钳制**状态，所有墙钟 A/B 作废

在排查 LLM 吞吐异常时发现：

```text
cpu0-5 : cur=2496000  min=787200   max=2496000   (硬件上限 3628800)
cpu6-7 : cur=883200   min=883200   max=1977600   (硬件上限 4608000, min 被钉在 883200)
governor=walt   温度 42.2 / 42.6 °C   省电模式=off
```

**温度只有 42 °C，不是热节流；频率上限是被外部钳制的**，且 cpu6-7 的最低频率被钉在 883 MHz。
设备上存在 `/data/local/tmp/SnapdragonProfiler/`，怀疑是此前的 profiling 会话来钳频后未恢复。

铁证（同一设备、同一二进制参数，连续执行）：

| 二进制 | tokens/s |
|---|---|
| 设备上留存的 2026-07-15 旧二进制 `llm_persistent` | **4.633** |
| 本次重建的 `CosyVoiceLlmPersistentBenchmark.out` | **4.255** |
| 文档基线（`STAGE3_FEASIBILITY.md:73`） | 85 ~ 95 |

**旧二进制同样只有 4.6 tok/s ⇒ 不是本次重建的问题，是设备状态的问题。** 这同时解释了
HiFT 墙钟为何在算子选择完全一致时出现 2.2x 摆动。

**在做任何性能对比之前必须先恢复满频**（通常重启设备即可；若由 profiler 会话钳制，
需结束该会话）。恢复后用 `cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq`
确认为 4608000 再开始测量。本文件中所有墙钟数字都产生于降频状态，**一律不得作为基线**。


## 关键测量：墙钟在本机不可用于 A/B

`winograd_1` 与 `base` 的算子选择**逐条相同**（见上表），但同一次扫描中：

```text
winograd_1  median_ms = 847.960
base        median_ms = 1841.838   (同一二进制、同一形状、同一循环数)
```

**在算子选择完全一致的前提下出现 2.2x 墙钟差异** —— 说明本机墙钟由频率/调度/热态支配，
任何小于该量级的"优化收益"在单机 A/B 下都不可判。前一轮 32 次扫描得到的
"littlecore=100 快 18%"、"mem_high 快 16%" 等排序因此**全部作废**，不作为结论。

后续在本机做性能 A/B 必须先有"该路径是否被选中"级别的插桩证据，再谈计时。

## `attentionOption` 的取数（第 2 轮）

**不需要改代码**：LLM 的 config JSON 直接支持该 hint ——
`transformers/llm/engine/src/llm.cpp:166-170` 读 `attention_mode`（旧键 `quant_qkv` 兼容），
默认 8，并 `rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode)`。
注意 `:171-173`：当 `reuse_kv()` 且模式为 10 时会强制改成 9。

真机实测（`config` 变体 + `MNN_ATTN_DIAG=1` 回执）：

| attention_mode | 解析结果 | tokens | invalid | tok/s |
|---:|---|---:|---:|---:|
| 8（默认）| flash=1 kvQuantMode=0 | 40 | 0 | 3.965 |
| 9 | flash=1 kvQuantMode=1 (K int8) | 40 | 0 | 3.776 |
| 10 | flash=1 kvQuantMode=2 (K+V int8) | 40 | 0 | 3.773 |
| 11 | flash=1 kvQuantMode=3 (K TQ3) | 40 | 0 | 3.975 |
| 13 | flash=1 kvQuantMode=5 (K TQ4) | 40 | 0 | 3.975 |

5 个模式的输出 **token 完全相同（同一 SHA-256）**，`invalid_outputs=0`。

**token 级比对不构成有效 Gate**（两个限制）：

1. **Gate 不敏感**：`sampler_type` 取 `greedy` 才能得到确定性（`temperature: 0.0` 与
   `top_k: 1` 都仍是随机的，实测三次运行结果互不相同）。而 greedy 在本模型上**退化为
   单 token 重复**（`uniq=1`，换更长/不同句子仍是 `29,29,29,...`）。退化序列对扰动不敏感。
2. **速度不可比**：设备处于降频钳制状态（见上节），表中 tok/s 全部落在 3.77–3.98 的噪声带内，
   **不构成"无收益"的结论**。

### 改用 logits 做与时钟无关的数值 Gate（第 3 轮）

token 比对既然不敏感，就直接比对**模型原始 logits**：在 `Llm::sample`
（`transformers/llm/engine/src/llm.cpp:786`，每一步采样的必经点）加了 env 门控落盘
`MNN_LOGITS_DUMP=<dir>`，每个 attention_mode 取前 8 步、每步 158720 floats。

比对结果（参考组 = mode 8 默认；`research/mnn-cosyvoice3/compare-logits.ps1`）：

| run | attention_mode | KV 量化 | maxabs | rms | cosine | argmax |
|---|---:|---|---:|---:|---:|---|
| m8b（对照组）| 8 | 无 | **0.000E+000** | **0.000E+000** | **1.00000000** | SAME |
| m9 | 9 | K int8 | 3.4E-01 ~ 5.0E-01 | 3.9E-02 | 0.99932 | SAME |
| m10 | 10 | K+V int8 | 3.6E-01 ~ 6.5E-01 | 4.0E-02 | 0.99915 | SAME |
| m11 | 11 | K TQ3 | 5.4E+00 ~ 6.8E+00 | 3.9E-01 ~ 4.9E-01 | **0.891 ~ 0.934** | SAME |
| m13 | 13 | K TQ4 | 4.0E+00 ~ 6.4E+00 | 2.7E-01 ~ 3.3E-01 | **0.956 ~ 0.970** | SAME |

**对照组的 maxabs 恰为 0、cosine 恰为 1** ⇒ 工具是确定性的，上表差异**完全归因于
attention 模式**，与设备频率、温度、负载无关。这是本轮唯一不受降频影响的硬结论：

- **int8 KV（模式 9/10）扰动很小**：cosine ≥ 0.9991，rms ≈ 0.04，比无量化仅差约 0.1%。
  这与他们已有的经验一致（int8 attention 可接受，TQ 不可）。
- **TQ3 / TQ4（模式 11/13）扰动大一个数量级以上**：cosine 掉到 0.89 / 0.96，
  rms 0.27–0.49，maxabs 达 4–7。按他们自己的先例门槛（Flow 蒸馏要求 mel cosine 0.9988、
  NPU 要求窗 corr 0.99756），**TQ 系远低于可用线**。
- `argmax` 全部 SAME，但**这一列不可用**：退化序列的 argmax 恒为 151953，
  对扰动不敏感（与 token 比对同一个毛病）。判据应看 cosine / rms。

**仍未取得**：无。速度已在本轮测出，见下节。

真正的结论：**`attention_mode` 可经配置生效；int8 KV 数值扰动小（cosine ≥0.9991）但速度更慢，
TQ3/TQ4 数值上不可用。结论是保持默认 8，不做任何 KV 量化。**

### 速度 A/B：int8 KV 反而慢 6%（第 4 轮）

先算理论天花板。`llm_config.json` 给出真实结构：**Qwen2-0.5B，hidden 896、24 层、
GQA 14 heads / 2 KV heads、head_dim 64**。于是

```text
每 token KV = 2(K,V) × 2(kv heads) × 64(head_dim) × 24(layers) = 6144 值 = 24 KB (fp32)
每 token 权重流量 = llm.mnn.weight = 353 MB（int4 主干 + int8 head）
```

| 上下文 L | KV 读/步 | 占权重流量 |
|---:|---:|---:|
| 100 | 2.4 MB | 0.7% |
| 500（App 的 maxTokens 上限）| 12 MB | 3.4% |

⇒ int8 KV 最多省 0.5%–2.5% 的每步内存流量，**却要为每步加入反量化开销**。

真机实测（交错 4 轮，mode 8/9/10，100 tokens，长 prompt）：

| mode | KV 量化 | median tok/s | 组内跨度 |
|---:|---|---:|---:|
| 8 | 无 | **3.960** | 0.3% |
| 9 | K int8 | 3.730 | 0.9% |
| 10 | K+V int8 | 3.762 | 4.4% |

**组内噪声 0.3–0.9%，组间差 6%，区间几乎不重叠 ⇒ int8 KV 让 decode 慢约 6%，方向与
"省带宽所以更快"的直觉相反**，与上面的带宽预算完全吻合：省下的是 1–3% 的流量，
付出的是每步反量化的固定开销。

注意这与前面 HiFT 墙钟的混乱并不矛盾：LLM benchmark 是长时稳态负载（每次 25 秒），
组内才收敛到 0.3%；HiFT 是短促突发，才出现 2.2x 摆动。**降频钳制影响绝对值，
但同钳制下的交错相对比较依然有效**——这一点由本表的组内跨度证明。

**最终判定：`attention_mode` 保持默认 8，四种 KV 量化（int8 ×2、TQ ×2）全部不采纳。**
默认配置在这一杠杆上已经是最优。


## 复现


```powershell
# 构建（MNN 树见下节）
adb push <build>\CosyVoiceHiFTBenchmark.out /data/local/tmp/cosy-mnn-stage2/
adb shell chmod 755 /data/local/tmp/cosy-mnn-stage2/CosyVoiceHiFTBenchmark.out

# 算子选择判定
adb shell "cd /data/local/tmp/cosy-mnn-stage2 && MNN_CONV_DIAG=1 MNN_DECONV_DIAG=1 \
  ./CosyVoiceHiFTBenchmark.out hift-core.mnn core inputs /data/local/tmp/out.bin 174 cpu low 1 6"
```

> 注意：`adb push` 之后必须 `chmod 755`。漏掉时进程不执行、诊断计数全为 0，
> 会被误读成"该路径未被使用"（本次已踩到一次，已用同时触发的 `[conv-diag]` 计数排除）。

## 为本次实验对 MNN 源码做的改动（均在 `E:\AndroidStudioProjects\MNN-master`）

1. **修复构建**：该树 2026-09-14 被覆盖进一批新算子实现（`GeometryFusedProj` /
   `GeometryGatedRMSNorm` / `ShapeFusedProj` / `ShapeGatedRMSNorm` / `FusedProjBufExecution`），
   但它们依赖的 `OpType_FusedLinear` / `OpType_GatedRMSNorm` **在 `schema/default/MNN.fbs` 中不存在**，
   导致链接前编译失败（9 errors）。已将上述 6 个文件改名为 `*.disabled-schema-missing` 隔离；
   隔离后 `MNNTransform` / `MNN_CL` 均无残留引用，等价于 stock 3.6.1 行为。
2. **`threadpool-fix.patch` 重新实现 → 失败 → 已还原为原版**。仓库 `mnn-patches/` 只有 hexagon
   那一个补丁，文档引用的 `threadpool-fix.patch` 与 `qnn-layout-fix.patch` 均不存在。
   按文档描述重写后，`CosyVoiceLlmPersistentBenchmark.out` 在退出时触发
   `FORTIFY: pthread_mutex_lock called on a destroyed mutex` 并 Aborted。
   **已把 `ThreadPool.cpp` 完整还原为原版**（`mCondition.wait(_l, [this]{...})` 结构可核对），
   两个目标均重新链接通过。
   > **更正（第 2 轮）**：还原后重新推送并复测，**崩溃依旧复现**。因此第 1 轮
   > "崩溃由本次线程池补丁引入"的归因**是错的**，特此撤回。
   > 更可能的原因是链接形态差异：设备上留存的 `llm_persistent` 只有 6.3 MB，
   > 而静态链接 MNN 的产物是 117–143 MB，故旧二进制必然是动态链接 `libMNN.so`——
   > 共享库与静态链接的静态对象析构顺序不同，退出路径行为因此不同。
   > 该崩溃发生在指标产出之后（`request=0` 已打印、token 已落盘），**不影响测量结果**，
   > 但属本树未解释的差异，记录待查。
   > **结论：该补丁仍然缺失，且不能照文档描述简单重写，需要拿到原始补丁或重新设计。**
3. **修复 LLM 构建**（`transformers/llm/engine/CMakeLists.txt:76`）：`MNN_BUILD_LLM=ON` 时
   configure 直接失败 —— `add_custom_command(TARGET llm POST_BUILD ...)` 作用在 OBJECT 库上，
   CMake 3.28+ 已禁止（本机 3.31.6）。该命令只是把 LLM 头文件拷到 native include 目录，
   已改为等价的 `add_custom_target(... ALL)` + `add_dependencies(llm_native_include llm)`。
4. **修复 LLM benchmark 链接**（本仓库 `mnn-jni/CMakeLists.txt`）：静态构建下
   `target_link_libraries(... llm)` 与 `libMNN.a` 的 `--whole-archive` 重复引入 llm 的 object，
   报大量 `duplicate symbol`。因 `MNN_SEP_BUILD` 在静态构建中被强制关闭
   （MNN `CMakeLists.txt:133-136`）时，llm 的 object 已被放进 MNN 归档（`:877-880`），
   故加 `if (MNN_SEP_BUILD)` 守卫，仅在 llm 是独立库时才显式链接。
5. **诊断插桩**（env 门控，不设置时零开销）：
   - `source/backend/cpu/CPUDeconvolution.cpp`：`MNN_DECONV_DIAG=1` 打印 creator 调用、
     deconv 实得线程数、每次执行耗时与累计耗时。
   - `source/backend/cpu/compute/ConvolutionFloatFactory.cpp`：`MNN_CONV_DIAG=1` 打印每个
     卷积最终选择的执行器路径。
   - `source/backend/cpu/CPUAttention.cpp`：`MNN_ATTN_DIAG=1` 打印 fused attention 的创建次数、
     `attentionOption` 解析出的 flash 开关与 KV 量化模式。
   - `transformers/llm/engine/src/llm.cpp`（`Llm::sample`）：`MNN_LOGITS_DUMP=<dir>` 把每步
     采样前的原始 logits 落盘（默认前 8 步）。这是绕开降频设备与退化采样的数值 Gate 入口。

## 新增到本仓库的 A/B 工装

- `mnn-jni/CosyVoiceHintKnobs.hpp`：env 驱动的 hint 开关
  （`MNN_HINT_MEMORY` / `MNN_HINT_KLEIDIAI` / `MNN_HINT_WINOGRAD` /
  `MNN_HINT_LITTLECORE` / `MNN_HINT_ATTENTION` / `MNN_HINT_CORE_IDS`）。
  **全部不设置时行为与改动前逐字节一致**（`Memory_Normal` + 不调用 `setSessionHint`）。
- `mnn-jni/CosyVoiceHiFTBenchmark.cpp`：接入上述开关，并把 `memoryMode` / `sessionHints`
  写入 metrics JSON 与 stdout，使每次运行自带配置回执。
- `research/mnn-cosyvoice3/run-hint-ab-hift.ps1`：交错 + min-of-N 的墙钟扫描脚本
  （注意：在设备降频钳制状态下其输出不可用）。
- `research/mnn-cosyvoice3/compare-logits.ps1`：attention 模式的 logits 数值比对
  （maxabs / rms / cosine / argmax），**与时钟无关**，是当前唯一可用的硬 Gate。

**未改动任何产品运行路径**：`CosyVoiceRuntime.kt` 的 flow/hift/LLM 配置、精度、线程数一律未动，
三个"无效"杠杆因此也没有进入 App。

## 附带实测：长句退化未复现

同一二进制、带 15s 冷却、min-of-4：

| mel 帧 | min (ms) | ms/帧 |
|---:|---:|---:|
| 174 | 1305 | 7.50 |
| 342 | 1907 | 5.58 |
| 864 | 3722 | **4.31** |

单位帧耗时随长度**下降**，与 `PITFALLS_AND_FIXES.md` 记录的 342→864 帧 3.9→8.1 ms/帧相反。
鉴于上文的墙钟不可靠性，此结果**只作为待复核线索**，不足以推翻原结论。

## 遗留

- **先恢复满频再谈性能**（见上文降频钳制一节）。在 cpu6 `scaling_max_freq` 回到 4608000 之前，
  任何吞吐/延迟数字都不可用；`attentionOption` 的 A/B 也因此**尚未取数**。
  已确认 adb 无 root（`adbd cannot run as root in production builds`，uid=2000），
  写 `scaling_max_freq` 返回 `Permission denied`，**因此解除钳制需要人工操作**：
  重启设备，或结束此前把频率钉住的 profiler 会话（设备上存在 `/data/local/tmp/SnapdragonProfiler/`）。
- `attentionOption`：**已闭环**。数值（logits）+ 速度双向取数完成，结论是保持默认 8、
  四种 KV 量化全部不采纳（int8 数值可用但慢 6%；TQ 数值不可用）。
- **设备降频钳制仍未解除**（cpu6 `scaling_max_freq`=1977600，应为 4608000；adb 无 root）。
  本次所有结论都不依赖绝对值，但若要把本文件里的 tok/s、ms 当作**基线**引用，
  必须先重启设备恢复满频并重测。
- `threadpool-fix.patch` 原始补丁仍未找回（本次重写尝试已证伪并还原）。
  另有一个未解释差异：本树静态构建的产物在退出时触发
  `pthread_mutex_lock called on a destroyed mutex`（stock 线程池下同样复现，
  不影响测量结果），旧动态链接产物不复现，待查。
- 长句退化在本机降频状态下未复现，需满频后重测才能定论。

