# CosyVoice3-MNN：NPU / GPU 定向模型改造执行手册 v1.0

**目标设备**：荣耀 Magic8 Pro / SM8850 / Adreno 830 / 12 GB  
**运行框架**：MNN 3.6.1 + OpenCL + Qualcomm QNN/HTP  
**目标**：不是把所有模块强行迁移到同一加速器，而是改造模型与运行链，使每个模块进入最适合的硬件路径，并最终降低 App 内真实端到端 RTF。

---

## 0. 最终目标架构

### 0.1 产品稳定版

```text
文本前端                         CPU
LLM Prefill（满足长度门槛时）   MNN Hexagon NPU
LLM Decode / lm_head / Sampler  MNN CPU C4/W4
Conditioner                     CPU
Flow 两步学生模型               MNN OpenCL GPU
F0 / Source                     CPU
HiFT                            当前 CPU Teacher；逐步替换为 NPU/GPU Student
复数恢复 / ISTFT / PCM          CPU
```

### 0.2 研究完成后的目标版

```text
LLM：W4 block64 + C4
  Prefill：Hexagon
  Decode：CPU

Flow-GPU：FP16 安全、静态 bucket、无 CPU fallback 的两步学生模型

HiFT-Accelerator-Student：
  同一套学生网络
  ├─ QNN/HTP 静态 context（NPU）
  └─ MNN OpenCL FP16 静态 Session（GPU）

运行时根据设备、长度和热状态选择 NPU 或 GPU；CPU Teacher 仅作质量回退与注册验证。
```

---

## 1. 当前事实基线：任何后续改造都必须对齐

### 1.1 模块与规模

```text
LLM          约 370M
Conditioner  约 30M
Flow         约 330M
HiFT         约 85M
```

### 1.2 当前有效路径

| 模块 | 当前最优后端 | 当前结论 |
|---|---|---|
| LLM | CPU | Decode 占 79%–93%，当前主要瓶颈是 Decode，不是 Prefill |
| Flow | OpenCL High / Buffer mode 68 | 两步蒸馏有效；首次 bucket resize/tuning 很慢，热态推理可用 |
| HiFT | CPU High / 6 线程 | 342 帧约 1.35 s；当前 QNN 12 帧窗口约 4.5 s，不可作为最终结构 |
| ISTFT | CPU | periodic Hann + OLA 窗平方和归一化已验收 |

### 1.3 已证明不能继续重复的路线

```text
1. 不再把当前 FP32 HiFT 切成 12 帧窗口强行上 HTP。
2. 不再把异步提交时间当作真实执行时间。
3. 不再使用输出不随输入变化的 cache“假执行”。
4. 不在产品热路径反复 createSession / resizeSession / OpenCL tuning。
5. 不用离线单窗耗时替代 App 模块耗时和端到端 RTF。
```

---

## 2. 总体策略：分成“运行链改造”和“模型改造”两条线

### A 线：不重新训练，先释放现有模型性能

1. Flow Session 池、共享 Runtime、固定 bucket、按需预热。
2. LLM 使用 MNN 3.6.1 重新导出，启用 C4 / W4 block64；先优化 CPU Decode。
3. 统一 App 内模块 benchmark 和端到端 benchmark。
4. HiFT 长句 CPU 超线性退化定位，作为 Student 的对照基线。

### B 线：重新训练，构建真正适合 NPU/GPU 的模型

1. 现有 HiFT 的 FP16 动态范围审计。
2. 构建 HiFT-Accelerator-Student。
3. 将 Flow 进一步改造成 FP16 安全、静态 bucket 的 GPU Student。
4. 同一 Student 分别导出 QNN context 和 MNN OpenCL 模型。

A 线先完成，B 线随后推进。不要等 B 线训练完成才修产品运行链。

---

# 第一部分：统一工程与验收基础

## 3. 冻结基线

建立标签：

```text
baseline-20260806
```

冻结并记录：

```text
代码 commit
MNN commit / 版本
QNN SDK 版本
V81 Stub SHA-256
V81 Skel SHA-256
各 MNN 模型 SHA-256
各 QNN context SHA-256
测试手机系统版本
App 包名与签名
```

固定测试集至少包含：

```text
S1  短句：约 130–150 speech tokens / 约 342 mel 帧
S2  中句：约 400–600 mel 帧
S3  长句：约 850–900 mel 帧
S4  高频女声
S5  低沉男声
S6  爆破音、齿音、数字、英文混合文本
S7  同一角色连续 10 句
```

每次生成保存：

```text
speech token 文件与 hash
Flow mel 文件与 hash
HiFT 复数频谱或原始输出
最终 PCM/WAV
模块 report.json
```

---

## 4. 统一三层测速

### L1：离线数值验证

回答：模型是否能运行、输出是否正确。

### L2：App 内模块基准

回答：模块在真实 App UID、真实 linker namespace、真实 Runtime 中是否更快。

### L3：App 端到端

回答：用户从提交文本到可播放 PCM 是否更快。

所有计时必须满足：

```text
换输入，输出必须改变
后端同步完成后仍然快
连续运行多次稳定
数值或音质验收通过
工作量等价
```

统一 JSON 字段：

```json
{
  "modelHash": "...",
  "backend": "cpu/opencl/hexagon/qnn",
  "inputFrames": 342,
  "inputTokens": 134,
  "warmup": 2,
  "repeat": 10,
  "prepareMs": 0.0,
  "resizeMs": 0.0,
  "executeMs": 0.0,
  "outputMs": 0.0,
  "moduleTotalMs": 0.0,
  "audioSeconds": 0.0,
  "rtf": 0.0,
  "peakRssMb": 0,
  "peakDmaBufMb": 0,
  "thermalStatus": "none",
  "outputHash": "...",
  "finite": true
}
```

---

# 第二部分：Flow 的 GPU 定向改造

## 5. P0：先完成产品级 Session 池

Flow 目前已经是最适合 GPU 的模块。第一目标不是重训，而是彻底消除首次 bucket 调优进入合成热路径的问题。

### 5.1 运行时结构

```text
FlowRuntimeManager
├─ 一个 Interpreter
├─ 一个共享 OpenCL Runtime
├─ 一个串行 GPU 工作线程
├─ Session[512]
├─ Session[1024]
└─ Session[1280]
```

首版只允许最多 2 个 Session 常驻，用 LRU 管理。不要同时初始化 3–6 个 OpenCL Session。

### 5.2 Bucket 选择

保留现有 bucket 集：

```text
512 / 768 / 1024 / 1280 / 1536 / 2048
```

但首版只按真实数据分布预热最常用的 1–2 个。当前长句 1038 sequence 对应 1280 bucket，应优先覆盖 512 和 1280，1024 按需创建。

### 5.3 Session 生命周期

```text
create Runtime：App/模型初始化阶段一次
create Session：该 bucket 第一次需要时一次
resizeSession：该 bucket 创建时一次
warmup inference：一次
updateCacheFile：一次
runSession：重复使用
```

热路径禁止：

```text
create Interpreter
create Runtime
create Session
resize 到真实句长
重复 updateCacheFile
并发创建另一个 OpenCL Session
```

### 5.4 OpenCL 配置

当前已实测优先：

```text
precision = High
mode = 68 = Wide tuning + Buffer
```

Session 池稳定后，再对比：

```text
68  = Wide + Buffer
132 = Wide + Image
```

每个模式必须使用独立 cache 文件：

```text
flow-opencl-high-mode68.cache
flow-opencl-high-mode132.cache
```

### 5.5 Flow P0 验收

```text
同 bucket 第二次 resizeMs < 50 ms
342 帧对应 Flow 总耗时不高于当前热态
864 帧 Flow 总耗时 <= 3.0 s
冷启动 20 次无黑屏
连续合成 10 次无 OpenCL 错误
30 分钟 PSS / DMA-BUF 不线性增长
```

---

## 6. P1：Flow 图结构审计

生成 `flow_op_audit.csv`：

```text
节点名
算子类型
输入输出 shape
数据类型
是否动态 shape
是否产生 Transpose/Raster
是否回退 CPU
OpenCL kernel 名称
单算子耗时
```

重点查：

```text
Transpose / Raster 是否过多
Attention QK / Softmax 是否在 GPU
LayerNorm 是否拆成大量逐元素算子
是否存在 CPU fallback
每一步是否重新建立 mask / position / time embedding
NC4HW4 与 NCHW 是否频繁互转
```

判定门：

```text
任何单步 CPU fallback > 20 ms：必须消除
Transpose/Raster 占推理 > 15%：必须重写图
单个动态 shape 导致重新 tuning：改固定 bucket
```

---

## 7. P2：构建 FP16 安全的 Flow-GPU Student

当前 High 精度是由于 Attention score、Softplus/GELU 等数值风险。目标不是直接把 Session 改成 Low，而是训练一份对 FP16 友好的两步学生模型。

### 7.1 Teacher

```text
现有两步 Flow High-precision 模型
```

### 7.2 Student 第一版不改宏观接口

```text
输入、输出、两步时间点保持不变
保留当前 CFG 单分支与宏轨迹蒸馏结果
仅修改内部数值与算子形式
```

### 7.3 数值稳定化规则

1. Q、K 在 matmul 前显式缩放：

```text
Q_scaled = Q / sqrt(d)
score = Q_scaled @ K^T
score = score - max(score)
score = clamp(score, -20, 0)
```

2. 训练阶段加入 fake-FP16：

```text
关键激活经过 fp16 round-trip 后继续计算
统计每层 Inf/NaN/最大绝对值
```

3. 残差分支加入可训练或固定缩放：

```text
y = x + 0.5 * branch(x)
```

4. 激活优先采用后端可融合的算子：

```text
ReLU / SiLU（只有在 OpenCL profile 证明融合时保留）
```

5. 避免在主干中使用：

```text
大范围 exp
不稳定 Softplus
重复 Transpose
动态 Loop
运行时生成巨大常量 mask
```

### 7.4 Flow Student 蒸馏损失

```text
L = 1.0 * MSE(v_student, v_teacher)
  + 0.5 * MSE(mel_student, mel_teacher)
  + 0.2 * (1 - cosine(mel_student, mel_teacher))
  + 0.1 * feature_distill
```

保持现有门槛：

```text
mel cosine >= 0.9988
SNR >= 26 dB
响度差不超过 1 dB
不能重现“直接降步”产生的 2–5 dB 响度抬高
```

### 7.5 Flow-GPU 导出

```text
PyTorch Student
→ ONNX 静态 bucket
→ MNN 3.6.1
→ OpenCL Low/Normal/High 三档 App 内测试
```

静态导出优先做：

```text
bucket 512
bucket 1280
```

通过后再补其他 bucket。

### 7.6 Flow-GPU 成功门槛

```text
Low 或 Normal 音质通过
864 帧推理 <= 2.0 s
相对 High 热态至少快 20%
无 NaN/Inf
连续 30 分钟稳定
```

若 Student 仍只能 High 精度，保留当前 High 路线，不再为了“使用 FP16”牺牲音质。

---

# 第三部分：LLM 的 NPU / CPU 定向改造

## 8. 正确目标：NPU Prefill + CPU Decode

现有数据中 Decode 占 79%–93%。因此 LLM 改造优先级如下：

```text
第一：CPU Decode C4 / KV Cache / 分配优化
第二：长 prompt 条件下 Hexagon Prefill
第三：不优先做 Hexagon Decode
```

---

## 9. 使用 MNN 3.6.1 重新导出 LLM

不能只替换运行库，旧模型必须重新导出。

目标配置：

```text
Transformer C4 graph
W4 symmetric
block_size = 64
packed RoPE
fused Attention / LayerNorm / Gather
```

### 9.1 拆分模型

将 CosyVoice LLM 拆成：

```text
Transformer trunk
Speech-token embedding
Speech-token lm_head
RAS sampler
```

优先让标准 Transformer trunk 进入 MNN LLM/C4 路径；自定义 speech-token head 与采样器继续在 CPU。

### 9.2 量化白名单

优先 W4：

```text
q_proj / k_proj / v_proj / o_proj
MLP up / gate / down
```

首版保持较高精度：

```text
Embedding
LayerNorm / RMSNorm
最终 lm_head（先 FP16/FP32，再做单独量化）
```

### 9.3 CPU Decode 优化项

```text
启用 C4 图
启用 Flash Attention
KV Cache 常驻，不允许每 token 扩容
position / mask buffer 预分配
lm_head buffer 常驻
Sampler 不创建临时大数组
Prompt voice token 使用 Prefix Cache
不在每 token 做 JSON / JNI / 文件 I/O
```

输出指标：

```text
decodeTotalMs
decodeTokenCount
P50/P90 ms per token
KV Cache bytes
lmHeadMs
samplerMs
```

### 9.4 Hexagon Prefill 策略

建立实际 crossover，而不是固定认为 NPU 必定更快：

```text
64 / 128 / 192 / 256 / 384 / 512 prompt tokens
```

分别测：

```text
CPU C4 Prefill
OpenCL C4 Prefill
Hexagon Prefill
Hexagon→CPU KV 交接
```

运行时策略：

```text
promptTokens < crossover：CPU Prefill
promptTokens >= crossover：Hexagon Prefill
Decode：CPU
```

### 9.5 LLM 验收

```text
CPU Decode 相比当前至少快 15%
Speech token 无空、无全零、无异常重复
相同固定种子下结果可回归
Hexagon Prefill 净收益包含 KV 交接后仍 > 20%
完整 LLM 总耗时下降 >= 8%，否则 Hexagon Prefill 不默认开启
```

---

# 第四部分：HiFT 的 NPU / GPU 定向重构

## 10. 先做一次现有 HiFT 的 FP16 抢救审计

目的不是继续调 12 帧窗口，而是判断旧模型能否低成本转为低精度整图。

### 10.1 逐层数据采集

对以下长度分别采集：

```text
T = 12 / 64 / 342 / 864
```

每层输出：

```text
min / max / mean / RMS
P99 / P99.9
Inf 数 / NaN 数
FP32 vs fake-FP16 corr
FP32 vs fake-FP16 max_abs
```

必须定位：

```text
第一层 Inf
第一层 NaN
第一个误差突然放大的层
```

### 10.2 抢救手段顺序

```text
输入标准化
权重离线重标定
层间 scale / inverse-scale
残差缩放
bias 使用 FP32 累加
敏感输出 clamp
将 exp / sin 移出 NPU 图
替换 LeakyReLU
```

### 10.3 停止条件

满足任一条件就停止抢救旧模型，转 Student：

```text
超过 25% 主干层必须 FP32
低精度后仍需 12 帧窗口
342 帧整图仍慢于 CPU 1.35 s
任何稳定化导致可闻音色变化
```

---

## 11. HiFT-Accelerator-Student：推荐结构

### 11.1 设计原则

```text
静态 shape
单次整句或单次 bucket 执行
FP16 / A16 友好
算子白名单同时覆盖 QNN HTP 与 MNN OpenCL
通道数为 32 的倍数
主干无 exp / sin / LeakyReLU / 动态控制流
尽量不使用 ConvTranspose
布局全程统一
```

### 11.2 输入输出接口

保持上游接口：

```text
mel     [B, 80, T]
source  [B, 18, 120T+1]
```

推荐输出改为：

```text
compressed complex spectrum [B, 18, 120T+1]
前 9 通道：压缩后的实部
后 9 通道：压缩后的虚部
```

CPU 端执行固定反压缩和 ISTFT。

推荐复数压缩：

```text
y = sign(x) * log1p(alpha * abs(x)) / log1p(alpha * xmax)
```

这样训练目标落在近似 `[-1,1]`，避免原图中的大范围 `exp(L)` 与相位 `sin(P)` 在低精度中放大误差。

### 11.3 主干草案

```text
MelStem:
  Conv1D 80 -> 128, k=3
  ReLU

Upsample-8:
  Nearest Resize x8
  Conv1D 128 -> 128, k=5
  ResBlock x2

Upsample-5:
  Nearest Resize x5
  Conv1D 128 -> 128, k=5
  ResBlock x2

Upsample-3:
  Nearest Resize x3
  Conv1D 128 -> 96, k=3
  ResBlock x2

SourceStem:
  Conv1D 18 -> 32, k=3
  ReLU

Fusion:
  Concat [96 + 32] -> 128
  Conv1D 128 -> 128, k=3
  ResBlock x4

Output:
  Conv1D 128 -> 18, k=1
  Tanh 或线性输出后离线标定
```

备选通道：

```text
小型：96 / 32 / 128
标准：128 / 64 / 192
```

首版从小型开始，避免 64 帧模型已出现的大图超线性内存问题。

### 11.4 ResBlock 约束

```text
Conv1D k=3
ReLU
Conv1D k=3
Residual Add
```

不使用：

```text
LeakyReLU 展开
动态 dilation 列表
复杂门控
GroupNorm 的复杂 reshape 链
频繁 Transpose
```

### 11.5 首批静态 bucket

只做两个：

```text
T=384   覆盖当前 342 帧短句
T=1024  覆盖当前 864 帧长句
```

通过后再扩展：

```text
192 / 576 / 768
```

不要一开始生成六个 context。

---

## 12. Teacher 数据集生成

Teacher 使用当前已验收的 CPU HiFT 和正确 ISTFT。

每条样本保存：

```text
mel.npy
source.npy
teacher_complex.npy
teacher_pcm.wav
metadata.json
```

训练切片长度分布：

```text
20%：96–192 帧
40%：193–384 帧
30%：385–768 帧
10%：769–1024 帧
```

数据必须包含：

```text
不同角色和音色
男女声
高低音
中英混合
数字和标点
齿音、爆破音
句尾静音
长句连续韵律
```

先做 Pilot：

```text
约 2,000–5,000 个切片
```

Pilot 达到速度和基本音质门槛后，再扩充正式数据。

---

## 13. HiFT Student 训练

### 13.1 Stage A：FP32 蒸馏

```text
Teacher 冻结
Student FP32
随机长度训练
```

建议损失：

```text
L = 1.0 * L1(compressed_complex)
  + 0.5 * L1(log_magnitude)
  + 1.0 * MultiResolutionSTFT(pcm)
  + 0.1 * L1(pcm)
  + 0.1 * feature_distill
```

首版不引入 GAN，先保证稳定、可部署、可回归。

### 13.2 Stage B：FP16 仿真训练

```text
权重和激活做 fake-FP16
关键累加保持 FP32
每步监控 Inf/NaN
激活采用训练可学习 scale 或固定校准 scale
```

### 13.3 Stage C：硬件在环回归

每个 checkpoint：

```text
导出 ONNX
生成 QNN / MNN 模型
在手机运行固定验证集
将真实后端误差写回报告
```

硬件在环不参与反向传播，但作为模型选择门禁。

---

## 14. 双后端导出

### 14.1 NPU 路径

```text
PyTorch Student
→ ONNX 静态 shape
→ Qualcomm 官方 QNN Converter
→ V81 context binary
→ 原生 QNN runner / App JNI
```

不要再以 MNN QNN Backend 作为最终 HiFT 编译器主线。

### 14.2 GPU 路径

```text
同一 PyTorch Student
→ ONNX 静态 shape
→ MNN 3.6.1
→ OpenCL Low / Normal
→ 固定 bucket Session
```

### 14.3 算子交集白名单

首版只允许：

```text
Conv / DepthwiseConv
Resize Nearest
ReLU
Add / Mul
Concat
Reshape（固定 shape）
Transpose（最多输入输出边界）
Tanh（若 profile 可接受）
```

发现白名单外算子，优先改模型，不优先写自定义 Op。

---

## 15. HiFT Student 验收

### 15.1 数值与音质

```text
输出全部 finite
削波数 = 0
phase circular >= 0.999
harsh <= CPU Teacher
6000 Hz 相干能量不高于 CPU Teacher
句尾无“得”声、无爆音、无周期声
不同音色不串音
```

### 15.2 性能

```text
384 bucket：
  NPU 或 GPU 模块总耗时 <= 1.0 s

1024 bucket：
  模块总耗时 <= 3.5 s

相对对应 CPU Teacher：
  至少快 20%
```

### 15.3 稳定性

```text
连续 100 句
冷启动 20 次
10 分钟 / 30 分钟热态
PSS / DMA-BUF 不线性增长
无 FastRPC / OpenCL / 系统级崩溃
```

达不到 20% 净收益，不替换产品路径。

---

# 第五部分：App 异构调度

## 16. 统一 Backend 接口

```cpp
enum class LlmPrefillBackend {
    Cpu,
    Hexagon
};

enum class FlowBackend {
    OpenClHigh,
    OpenClFp16Student
};

enum class HiftBackend {
    CpuTeacher,
    QnnStudent,
    OpenClStudent
};
```

同一输入输出协议：

```text
LLM 输出 speech tokens
Flow 输出 mel
HiFT 输出 PCM
```

后端切换不能改变业务层文件格式和播放队列。

---

## 17. 运行时选择策略

### LLM

```text
prompt token 少：CPU Prefill
prompt token 多：Hexagon Prefill
Decode：CPU
```

### Flow

```text
默认 OpenCL Session 池
GPU 发生异常：停止后台预热，保留单 Session 产品路径
```

### HiFT Student

```text
NPU context 可用且温度正常：QNN Student
NPU 不可用但 GPU Session ready：OpenCL Student
二者不可用或质量回归失败：CPU Teacher
```

回退不是“默认 CPU”，而是产品安全机制。

---

## 18. 流水线调度

```text
LLM 句 n+1
Flow 句 n
HiFT 句 n-1
AudioTrack 播放句 n-2
```

限制：

```text
HTP 同一时间只运行一个主要任务
OpenCL Flow 与 OpenCL HiFT Student 首版不要并发
Flow GPU 操作全部在同一线程串行
队列容量 1–2
播放侧缓存 1–2 段
```

不要用一个全局 mutex 把 LLM、Flow、HiFT 全部串行化。

---

# 第六部分：代码与目录落地

## 19. 建议目录

```text
research/accelerator_adaptation/
├── common/
│   ├── test_manifest.json
│   ├── metrics.py
│   └── hashes.py
├── llm/
│   ├── export_mnn361_c4.py
│   ├── quant_w4_block64.json
│   ├── validate_speech_tokens.py
│   └── benchmark_prefill_decode.py
├── flow_gpu/
│   ├── collect_activation_stats.py
│   ├── flow_fp16_student.py
│   ├── train_flow_student.py
│   ├── export_bucket_onnx.py
│   └── validate_flow.py
├── hift_student/
│   ├── export_teacher_cache.py
│   ├── collect_fp16_stats.py
│   ├── hift_accelerator_student.py
│   ├── train_fp32.py
│   ├── train_fake_fp16.py
│   ├── export_onnx.py
│   ├── convert_qnn.sh
│   ├── convert_mnn.sh
│   └── validate_audio.py
└── app_benchmark/
    ├── benchmark_cases.json
    ├── schema.json
    └── compare_reports.py
```

App：

```text
mnn-jni/
├── CosyVoiceFlowSessionPool.cpp
├── CosyVoiceLlmHybridRuntime.cpp
├── CosyVoiceHiftStudentQnn.cpp
├── CosyVoiceHiftStudentOpenCl.cpp
├── CosyVoiceBenchmarkNative.cpp
└── BackendManifest.cpp
```

---

## 20. 每个模型的 Manifest

```json
{
  "name": "hift-accelerator-student",
  "version": 1,
  "inputProtocol": "mel80+source18",
  "outputProtocol": "compressed-complex18",
  "buckets": [384, 1024],
  "backends": {
    "qnn": {
      "soc": "SM8850",
      "htpArch": "V81",
      "sdk": "2.37",
      "precision": "fp16-or-a16",
      "contextSha256": "..."
    },
    "opencl": {
      "precision": "low",
      "mode": 68,
      "modelSha256": "..."
    }
  },
  "fallback": "hift-cpu-teacher"
}
```

---

# 第七部分：执行顺序与停止门

## 21. 推荐顺序

### P0：完成 Flow Session 池

交付：

```text
同 bucket resize < 50 ms
Flow 长句总耗时 < 3.0 s
稳定性通过
```

### P1：MNN 3.6.1 重导出 LLM

交付：

```text
C4/W4 block64 CPU Decode 基线
Prefill crossover 曲线
Hexagon Prefill 可选路径
```

### P2：HiFT FP16 审计

交付：

```text
首个 Inf/NaN 节点
逐层动态范围报告
是否值得抢救旧模型的结论
```

### P3：HiFT Student 384 bucket Pilot

交付：

```text
FP32 Student 音质
fake-FP16 稳定
NPU/GPU 双导出
真实 App 速度
```

判定：

```text
384 bucket 仍 >= 1.35 s：立即重审结构，不扩展 1024
音质不通过：不做性能扩展
```

### P4：HiFT Student 1024 bucket

交付：

```text
长句音质
长句性能
VTCM/显存/内存 profile
```

### P5：Flow FP16 Student

仅在 Session 池后 Flow infer 仍是端到端主要瓶颈时实施。

### P6：异构流水线与 30 分钟验收

最终目标：

```text
短句热态 RTF < 0.8
长句 RTF <= 0.8（第一阶段先进入 <1.0）
连续朗读无音色漂移、无系统崩溃
```

---

## 22. 项目停止条件

### 停止旧 HiFT NPU 抢救

```text
需要大量 FP32 岛
仍需小窗口
仍慢于 CPU
```

### 停止 Hexagon Prefill 默认启用

```text
完整 LLM 总耗时下降 < 8%
或端到端下降 < 3%
```

### 停止 Flow FP16 Student

```text
音色或响度明显变化
mel cosine < 0.9988
相对 High 热态加速 < 15%
```

### Student 不能替代 Teacher

```text
质量门未过
性能净收益 < 20%
稳定性未过
```

---

# 23. 最终决策

当前最可行的路线不是把原始四个模型同时强行迁往 NPU/GPU，而是：

```text
1. Flow 保持 GPU，并把 Session 生命周期做成产品级。
2. LLM 用 MNN 3.6.1 重导出 C4/W4，CPU Decode 优先，Hexagon Prefill 条件启用。
3. 当前 HiFT 作为 Teacher；重新训练一个同时满足 QNN HTP 与 OpenCL 算子交集的低精度 Student。
4. Student 使用静态 384/1024 bucket，单次执行，不再使用 12 帧重叠窗口。
5. 所有性能结论只接受 App 内同步模块基准和端到端 RTF。
```

这条路线既保留现有已经跑通的成果，又真正改变模型结构，使它进入 NPU/GPU 的优势区，而不是继续通过部署技巧补偿一个硬件不友好的原模型。
