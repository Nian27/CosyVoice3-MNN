# CosyVoice3-MNN

**CosyVoice3 MNN 手机端本地语音合成 — Android arm64 正式版**

CosyVoice3 本地 TTS App。使用阿里 MNN 推理引擎，全部在手机本地运行，无需联网。

> v1.1.0 使用 MNN 3.6.1。在已验证的 SM8850 上自动启用
> `第 0 层 q_proj NPU + 其余 LLM CPU`；其他设备自动使用 CPU/OpenCL 兼容路径。
> 整模型 NPU 会导致 Token 坍缩和错误声音，本项目不会默认启用。

---

## 目录

- [项目背景](#项目背景)
- [尝试过的路线（历程）](#尝试过的路线历程)
- [整体架构](#整体架构)
- [合成管线详解（4 个阶段）](#合成管线详解4-个阶段)
- [移植过程](#移植过程)
- [问题记录（供后续开发者参考）](#问题记录供后续开发者参考)
- [性能数据（荣耀 Magic8 Pro 真机实测）](#性能数据荣耀-magic8-pro-真机实测)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [模型管理](#模型管理)
- [音色管理](#音色管理)
- [手机创建音色](#手机创建音色)
- [GPU / CPU 后端切换](#gpu--cpu-后端切换)
- [项目结构](#项目结构)
- [从源码构建](#从源码构建)
- [常见问题 / Troubleshooting](#常见问题--troubleshooting)
- [技术栈](#技术栈)
- [模型信息](#模型信息)
- [欢迎参与](#欢迎参与)
- [许可证](#许可证)

---

## 尝试过的路线（历程）

这个项目走过很多条路，有成功的、有失败的、有部分成功的，全部记录如下：

| 路线 | 结果 | 关键文档 |
|------|------|----------|
| **CrispASR/ggml + Vulkan**（第一尝试） | ❌ 失败：Adreno DeviceLost、FP16 噪声、RTF 5.11 | [DEVELOPMENT_STORY.md 第 2 节](docs/DEVELOPMENT_STORY.md#2-第一阶段crispasrggml--vulkan-路线失败) |
| **MNN/OpenCL**（第二尝试，当前主线） | ✅ 成功发布 v1.1.0，但热态 RTF 仅 0.79~1.0、内存 2.25 GB、仅 SM8850 单机验证，**实际效果没有想象中好** | [DEVELOPMENT_STORY.md](docs/DEVELOPMENT_STORY.md)、[性能数据](#性能数据荣耀-magic8-pro-真机实测) |
| **Flow 蒸馏**（关键突破） | ✅ 10 步 → 2 步，seq516 从 18.4 秒降到 2.374 秒 | [STAGE3_FEASIBILITY.md](research/mnn-cosyvoice3/STAGE3_FEASIBILITY.md) |
| **MNN + Hexagon NPU（单算子）** | ⚠️ 部分成功：仅 q_proj 放 NPU（wall time -5.94%），收益有限，Token 崩溃风险 | [NPU_RELEASE_VALIDATION.md](docs/NPU_RELEASE_VALIDATION.md)、[加速器改造计划](docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md) |
| **HiFT 切 12 帧窗口上 HTP**（早期尝试） | ❌ 失败：约 4.5 秒/窗口，假执行/异步提交耗时不能当真 | [加速器改造计划 1.3 节](docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md) |
| **QAIRT QNN 全图 HTP 迁移** | ⏳ 进行中：converter 语义已证明（P4.2 WAV corr=1.0）、A16W8 per-channel 达 24.77ms/12帧，但最终 PCM 未达标（corr 0.835） | [PITFALLS_AND_FIXES.md](docs/PITFALLS_AND_FIXES.md) 第 8 类 |

踩坑红线（已证明不能继续重复）：FP32 HiFT 切 12 帧窗口强行上 HTP；把异步提交时间当真实执行时间；输出不随输入变化的 cache“假执行”；热路径反复 createSession/resizeSession/OpenCL tuning；用离线单窗耗时冒充端到端 RTF。

完整的踩坑与解决办法：**[docs/PITFALLS_AND_FIXES.md](docs/PITFALLS_AND_FIXES.md)**（全历程 12 大类总结：现象 → 根因 → 解决 → 汇总表 → 剩余 3 坑 → 三大教训）、**[docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md](docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md)**（冻结基线 `baseline-20260806`）。

---

## 项目背景

**起因**：想要一个音色好的本地 TTS 模型，不用联网，不用调 API。CosyVoice3 音质不错但官方只有 Python 版本。于是搞了这台移植——PyTorch → ONNX → MNN → Android arm64 .so，中间踩了 Vulkan DeviceLost、FP16 溢出、Flow 蒸馏一堆坑。

**本来没打算发**，就是自己用。后来觉得折腾了这么久，代码放着也是放着，不如开源出来，万一有人也想在手机上跑本地 TTS 能少走点弯路。

### 移植过程

从 PyTorch 到手机上能跑，大概经历了这些：

1. **CrispASR/ggml + Vulkan** → Adreno GPU 报 `ErrorDeviceLost`，废弃
2. **换 MNN/OpenCL** → PyTorch 导出 ONNX → MNN 转换 → 但 10 步 Flow 跑一次 18 秒，没法用
3. **ONNX 稳定性修复** → FP16 精度下 Softplus 溢出、GELU 溢出，重写了激活函数
4. **CFG 单分支蒸馏** → 把双分支 CFG 教师蒸馏成单分支学生，batch=2→1，单步 1.84 秒→0.91 秒
5. **两步宏轨迹蒸馏** → 10 步 Euler solver → 2 步，最终两步仅 2.37 秒，mel 余弦 0.9988
6. **JNI 集成** → Conditioner/LLM/Flow/HiFT 从子进程迁到 App 内常驻 JNI，修复 OpenCL 子进程不能继承的问题
7. **手机调优** → Flow 桶机制避免 GPU kernel 重编译、HiFT CPU 6 线程、Buffer mode 68

详细过程见 [`docs/DEVELOPMENT_STORY.md`](docs/DEVELOPMENT_STORY.md)，所有可复现脚本在 [`research/mnn-cosyvoice3/`](research/mnn-cosyvoice3/)。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     MainActivity (Compose UI)             │
│  模型管理 │ 音色选择 │ 自动硬件调度 │ 试听 │ 创建音色 │ 下载 │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
                ▼                         ▼
       ┌────────────────┐      ┌──────────────────────┐
       │  CosyVoiceStore │      │  CosyVoiceRuntime     │
       │  模型/音色/注册   │      │  合成管线编排 (Mutex)  │
       │  ZIP导入/导出   │      │  LLM→Cond→Flow→HiFT  │
       └────────────────┘      └──────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────┐
          ▼                             ▼                     ▼
┌───────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ CosyVoiceLlmNative│   │ CosyVoiceFlowNative  │   │ CosyVoiceHiFTNative │
│ (libcosy_llm_jni) │   │ (libcosy_flow_jni)   │   │ (libcosy_hift_jni)  │
├───────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ MNN LLM           │   │ MNN Flow            │   │ MNN HiFT            │
│ CPU / q_proj NPU  │   │ CPU/OpenCL 推理      │   │ CPU 推理            │
│ 生成语音 Token     │   │ Token → Mel 频谱     │   │ Mel → WAV 波形      │
└───────────────────┘   └─────────────────────┘   └─────────────────────┘
                                                 ┌──────────────────────┐
                                                 │CosyVoiceEnrollmentNtv│
                                                 │(libcosy_enrollment)  │
                                                 │ 音色注册（可选扩展） │
                                                 └──────────────────────┘
```

原生运行库动态链接 MNN 3.6.1，并包含 Hexagon Stub；DSP Skeleton 位于
`app/src/main/assets/hexagon/`。完整构建与验证边界见
[`docs/NPU_RELEASE_VALIDATION.md`](docs/NPU_RELEASE_VALIDATION.md)。

| 文件 | 大小 | 功能 |
|------|------|------|
| `libMNN.so` / `libllm.so` | 动态运行库 | MNN 3.6.1 与 LLM |
| `libMNN_htpops.so` | Hexagon Stub | 与 DSP Skeleton 配合调用 NPU |
| `libcosy_llm_jni.so` | JNI | LLM CPU / 受限 q_proj NPU |
| `libcosy_flow_jni.so` | JNI | Flow CPU/OpenCL |
| `libcosy_hift_jni.so` | JNI | HiFT CPU |
| `libcosy_enrollment_jni.so` | JNI | 音色注册 |

---

## 合成管线详解（4 个阶段）

每次合成一句文字要走 4 个阶段，顺序执行。故障排查时按这个顺序看日志：

### 阶段 1：LLM — 文字 → 语音 Token

```
输入: "你好，欢迎使用阅读"
        │
        ▼
    [prompt-speech-tokens.csv]  ← 当前选中音色的参考 Token（零样本复刻才有）
        │
        ▼
    LLM (SM8850: q_proj NPU + CPU；其他设备: CPU)
        │
        ▼
输出: speech-tokens-0.csv  ← 一串数字，每个 0~6560 表示一个语音码本索引
```

- 耗时：约 1.5~2.0 秒
- 如果是"指令演绎"（Instruct2）模式，不附加参考 Token，用指令文字替代
- SM8850 仅将 `/layers.0/self_attn/q_proj/Linear` 放到 Hexagon；质量门失败会整句回退 CPU

### 阶段 2：Conditioner — Token 预处理

```
输入: speech-tokens-0.csv + 音色档案（prompt-cond.bin / spks.bin）
        │
        ▼
    Conditioner (libcosy_conditioner_exec.so / 独立子进程 / flow-conditioner.fp32.mnn)
        │
        ▼
输出: flow-input/ 目录下的 Mel 频谱碎片文件
```

- 独立 C++ 可执行文件，通过 ProcessBuilder 启动
- 负责将 Token 和音色信息合并成 Flow 模型的输入格式
- 如果 Conditioner 失败，日志在 run-{timestamp}/conditioner.log

### 阶段 3：Flow — Token → Mel 频谱

```
输入: flow-input/ → Conditioner 输出的特征
        │
        ▼
    Flow (libcosy_flow_jni / CPU 6线程 或 OpenCL GPU / flow.fp16.mnn ~633 MB)
        │
        ▼
输出: student_target_mel_android.bin  ← Mel 频谱数据
```

- **这是最复杂的阶段**：MNN FP16 模型，~633 MB 权重
- **Flow 后端可选**：CPU（`flowBackend="cpu"`）或 OpenCL GPU（`flowBackend="opencl"`）
- CPU 模式：`precision="normal"`, `threads=6`
- OpenCL 模式：`precision="high"`, mode 可选 4(自动)/68(Buffer)/132(Image)
- Flow 输出是 80 通道的 Mel 频谱，帧率 50fps（每帧 20ms）
- GPU 编译缓存存在 `gpu-cache/flow-fp16-{backend}-{precision}-mode{mode}.cache`

### 阶段 4：HiFT — Mel → WAV 波形

```
输入: mel.bin（上一步的 Mel 频谱）+ source-linear-weight.bias
        │
        ▼
    HiFT (libcosy_hift_jni / CPU 6线程 或 OpenCL GPU / hift-core.fp32.mnn ~67 MB + hift-f0.fp32.mnn ~12.6 MB)
        │
        ▼
输出: hift-android.wav ← 24kHz 单声道 16-bit PCM WAV
```

- HiFT 其实就是 HiFi-GAN + F0 预测
- 后端：CPU（High/6 线程）为主；OpenCL 实测比 CPU 慢（2.89s vs 1.7s，动态长度+小算子搬运开销大），已不使用
- 输出文件 > 44 字节才算成功

---

## 问题记录（供后续开发者参考）

> 分两部分：**历史已修复（早期 App 层问题）** 与 **当前已知问题（2026-08 状态）**。
> 完整踩坑历程（12 大类：Vulkan 环境、转换改坏、假加速、音频后处理、量化失败等）见 [docs/PITFALLS_AND_FIXES.md](docs/PITFALLS_AND_FIXES.md)。

### 历史已修复（早期 App 层问题）

1. **非高通 OpenCL 闪退**：Manifest `uses-native-library libOpenCL.so required=true` 只有 Adreno 有 → 独立 App 不加该声明，`CosyVoiceRuntime.detectBestFlowBackend()` 运行时检测，没有就降级 CPU。*非高通真机仍待实测（见下方已知问题 4）。*
2. **Flow 只支持 OpenCL、无 CPU fallback**：backend 硬编码 `"opencl"` → `CosyVoiceSynthesisOptions` 新增 `flowBackend` 字段并下传，CPU 模式自动 `precision="normal"`、`threads=6`。
3. **`extractNativeLibs` 导致 .so 加载失败**（`UnsatisfiedLinkError`）→ cosytest 与独立 App 均加 `jniLibs.useLegacyPackaging = true`。
4. **首次打开 App 空指针崩溃**：模型未就绪时 `selectVoiceProfile()` 抛异常 → 加 `if (modelStatus().ready)` 保护。
5. **导出模型时 ContentProvider URI 残留**：写入失败后 uri 处理不对 → try-catch 包裹，失败时显式删除不完整文件。
6. **GPU 编译缓存不清理**：切换 backend 后首合成 30+ 秒 → 删除模型时一并删除 `gpu-cache/`。

### 当前已知问题（2026-08）

1. **NPU（QAIRT QNN HTP）HiFT 未过最终音频门槛（实验进行中，不进入正式版）**：性能已达标（24.77 ms/12帧，比 CPU ≈36 ms 快约 31%），84 窗 out18 tensor corr 0.99756，但最终 PCM corr 仅 0.835、relative L2 0.563——局部 Tensor 误差经幅度/相位/ISTFT/OLA 被放大。详见 [PITFALLS_AND_FIXES.md 第 8 类](docs/PITFALLS_AND_FIXES.md)。
2. **Flow Session 池产品稳定性**：性能逻辑已明确（共享 Runtime、1-2 常驻 Session、LRU、后台预热），正常合成/冷启动/长句/连续使用的稳定性验收未完成。
3. **长句 CPU HiFT 退化**：342 帧约 3.9ms/帧，864 帧约 8.1ms/帧（单位帧耗时翻倍，非线性），根因未完全定位。
4. **非高通设备未验证**：`libcosy_flow_jni.so` 的 CPU 后端在麒麟/天玑上能否运行**完全没有真机验证**。
5. **冷启动慢**：冷态首次合成 GPU 编译 kernel 等待 10-15 秒是正常的（正式 App 有 GPU 编译缓存）。

---

## 性能数据（荣耀 Magic8 Pro 真机实测）

测试设备：荣耀 Magic8 Pro（第五代骁龙8至尊版 SM8850 / Adreno 830 / 12 GB RAM）

| 指标 | 热态（连续合成） | 冷态（首次合成） |
|------|------------------|------------------|
| **实时系数（RTF）** | **0.79 - 0.96** | **~1.7** |
| LLM 耗时 | ~1.6 秒 | ~1.6 秒 |
| Flow GPU 耗时 | 0.8 - 1.1 秒 | ~3.4 秒（GPU 编译） |
| Flow CPU 耗时 | ~2.5 - 3.5 秒（预估） | ~3.5 - 4.0 秒（预估） |
| HiFT CPU 耗时 | 1.4 - 2.0 秒 | ~1.9 秒 |
| 进程内存（早期版，单模块常驻） | ~947 MB | ~947 MB |
| 进程内存（v1.1.0，全 MNN 三模块常驻） | PSS ~2.25 GB | PSS ~2.25 GB |

> 数据日期：2026-07-20 ~ 07-23 真机实测（荣耀 Magic8 Pro / SM8850）。
> 内存是**版本不同**：早期版本只有部分模块常驻，PSS 约 947 MB；v1.1.0 的 LLM+Flow+HiFT 三模块全 MNN 常驻后 PSS 约 2.25 GB
> （其中 GPU kgsl-3d0 约 1.27 GB），整批合成结束释放 Session 后回落到约 269 MB。

> **非高通手机**：Flow CPU 模式的性能预期比上面 GPU 数据慢 2-3 倍，实测 RTF 可能在 2~3 之间。整体 10 秒文本的合成时间可能在 8~15 秒，仍然可用但不如高通流畅。

---

## 系统要求

| 项目 | 要求 |
|------|------|
| Android | API 26+（Android 8.0+） |
| 架构 | arm64-v8a 处理器 |
| RAM | 推荐 8 GB+（v1.1.0 全 MNN 常驻峰值 PSS ~2.25 GB；早期版本约 1 GB，需留足余量） |
| **存储（模型）** | **至少 1.4 GB**（模型包 ~1.3 GB + 运行时临时文件） |
| **存储（创建音色）** | **额外 ~974 MB**（可选扩展，不安装不影响合成） |
| GPU（可选） | 任何 OpenCL 支持的 GPU（Adreno / Mali / PowerVR） |

---

## 快速开始

### 1. 下载 APK

从 [GitHub Releases](https://github.com/Nian27/CosyVoice3-MNN/releases) 下载最新的
`CosyVoice3-MNN-v1.1.0-arm64.apk`。

```bash
# 或者自己构建（见"从源码构建"章节）
```

### 2. 安装 APK

```bash
adb install -r CosyVoice3-MNN-v1.1.0-arm64.apk
```

### 3. 下载模型

完整模型和可选音色扩展托管在
[Hugging Face：VicenTrent/Cosy-Voice-MNN](https://huggingface.co/VicenTrent/Cosy-Voice-MNN)。
App 可点击“在线安装”断点续传并校验完整 ZIP，也可手动下载后导入：

| 文件 | 大小 | 说明 |
|------|------|------|
| `cosyvoice3-mnn-mobile-fp16-complete.zip` | 1,399,083,563 B | 完整 MNN 模型包（17 个文件），SHA-256 `B1C74DFC...FEB1C84` |
| `cosyvoice3-mnn-enrollment-extension.zip` | ~974 MB | 音色创建扩展（可选，如需手机创建音色） |

### 4. 导入模型

打开 App：

1. **导入 MNN 模型包** → 点击"导入 ZIP" → 选择 `cosyvoice3-mnn-mobile-fp16-complete.zip`
   - 自动校验 17 个文件的 SHA-256
   - 导入完成后状态显示"已安装 · 1.30 GiB"
2. **（可选）导入音色创建扩展** → 点击"导入创建扩展" → 选择 `cosyvoice3-mnn-enrollment-extension.zip`
   - 导入后即可使用"手机创建音色"功能

### 5. 试听

1. 默认已选中"基准音色"，直接输入文字
2. 点击"合成并试听"
3. 等待 3-10 秒即可听到合成的语音

---

## 模型管理

### 文件清单

模型包包含 17 个文件，共 ~1.3 GB。存储在 `app内部存储/files/cosyvoice3-mnn/model/` 下：

| 文件 | 大小 | 作用 |
|------|------|------|
| `llm.mnn.weight` | 336.5 MB | LLM 权重（最大的单个文件） |
| `embeddings_bf16.bin` | 271.2 MB | LLM 嵌入层 |
| `flow.cfg-student-2step.batch1.fp16.mnn.weight` | 632.4 MB | Flow 模型权重 |
| `rand-noise.bin` | 4.58 MB | Flow 共享噪声 |
| `flow-conditioner.fp32.mnn` | 4.20 MB | Conditioner 模型 |
| `prompt-speech-tokens.csv` | ~400 B | 基准音色 Token |
| `prompt-cond.bin` | ~54 KB | 基准音色 Condition |
| `spks.bin` | 320 B | 基准音色 Speaker 向量 |
| `source-linear-weight.bias` | 36 B + 4 B | 线性映射参数 |
| 其余文件 | 几 KB ~ 几十 MB | 配置文件、Tokenizer、F0 模型等 |

### ZIP 导入机制

1. 读取 ZIP 中的每个文件，文件名匹配 `MODEL_FILE_SPECS` 列表
2. 解压到 staging 目录
3. 校验文件大小和 SHA-256
4. 逐个移动到 model 目录（原子操作：先写 `.importing` 再 rename）

### 导出

选择"导出 ZIP" → 选择保存位置 → 生成无压缩的 ZIP 包（~1.3 GB）

---

## 音色管理

### 音色档案结构

每个音色是一个目录，存储在 `.../voices/{profile-id}/`：

```
profile.json              # 元数据：名称、Token数、Hash等
prompt-speech-tokens.csv  # 参考语音 Token（逗号分隔的数字）
prompt-cond.bin           # 参考语音 Conditioner 输出（80ch × frame × 4B）
spks.bin                  # 说话人嵌入向量（80ch × 4B = 320B）
source.wav（可选）        # 创建音色时的原始参考音频
rand-noise.bin（symlink） # 符号链接到模型目录的共享噪声
```

### 内置基准音色

- ID: `builtin-mnn-reference-v1`
- 87 Token，174 帧
- 从模型包中的 `prompt-speech-tokens.csv` + `prompt-cond.bin` + `spks.bin` 构建
- 不可删除

### 导入音色 ZIP

支持批量导入（选多个 ZIP 或一个包含多个音色的 ZIP）：

1. 点击"批量导入 ZIP"
2. 选择音色档案 ZIP（单个或多个）
3. 自动校验兼容性（modelId 必须匹配）
4. 导入后自动选中最后一个导入的音色

### 导出音色

- 单个导出：点击对应音色的"导出"按钮
- 全部导出：点击"导出全部"（仅导出非内置音色）

### 实时音色

如果某个音色的 Token 数超过 125（比如从 MP3 创建的音色），合成时会报错。
点击"生成实时版"会：
1. 截取前 125 个 Token
2. 截取对应的 Conditioner 数据（80ch × 125 帧）
3. 生成一个新的音色档案（自动命名为"原名称（实时）"）

---

## 手机创建音色

需要先导入"音色创建扩展"（~974 MB）。

### 流程

1. **选择参考音频** → 点击"选择 MP3/音频"，选一段包含单人清晰语音的文件
2. **截取片段** → 填入起止秒数（推荐 3-5 秒，必须 3-15 秒）
3. **填写信息** → 音色名称 + 片段对应的文字（必须完全一致）
4. **创建** → 点击"创建并选中音色"
5. **等待** → 约 30-60 秒，后台运行：
   - 解码音频 → MediaCodec 硬解
   - 语音 Tokenizer（speech-tokenizer-v3.fp32.inline.mnn, ~924 MB）→ 提取 Token
   - CAM++ 说话人识别（campplus.fp32.mnn）→ 提取 Speaker 向量
   - 说话人特征投影 → 生成 speaker.bin
6. **完成** → 自动选中新音色，可直接合成试听

### 常见错误码

| 错误码 | 含义 |
|--------|------|
| 10 | 参考 WAV 文件无法读取 |
| 11 | 人声长度不在 3-15 秒 |
| 20-28 | 语音 Tokenizer 模型问题（检查扩展文件是否完整） |
| 30-37 | 说话人模型问题 |
| 40 | Token 超过 125 个限制 |
| 42 | 音色文件写入失败（检查存储空间） |

---

## 自动硬件调度

用户不需要手动选择后端。App 按 SoC、OpenCL 和 Hexagon 初始化结果自动决定：

| 条件 | LLM | Flow | HiFT |
|------|------|------|------|
| 已验证 SM8850 + Hexagon 可用 | 第 0 层 q_proj NPU，其余 CPU | OpenCL | CPU |
| 其他高通 / 其他 SoC | CPU | OpenCL 可用则 GPU，否则 CPU | CPU |
| NPU Token 或报告异常 | 整句自动重跑 CPU | 不变 | 不变 |

“运行库能加载”不等于“NPU 已验证”。未通过声音和 Token 正确性验证的 SoC
不会自动启用 Hexagon。

---

## 项目结构

```
Fun-CosyVoice/
├── mnn-jni/                         # 🔧 MNN JNI C++ 源码（14 个文件 + CMakeLists.txt）
│   ├── CosyVoiceLlmJni.cpp           #    LLM JNI 包装
│   ├── CosyVoiceFlowJni.cpp          #    Flow JNI 包装
│   ├── CosyVoiceHiFTJni.cpp          #    HiFT JNI 包装
│   ├── CosyVoiceEnrollmentJni.cpp    #    音色注册 JNI 包装
│   ├── CosyVoice*Benchmark.cpp       #    各模块真机基准测试
│   └── CMakeLists.txt                #    编译配置
├── app/
│   ├── build.gradle.kts              # Android 应用配置
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/fun/cosyvoice/
│       │   ├── MainActivity.kt                  # Compose UI 主界面（552 行）
│       │   ├── CosyVoiceRuntime.kt              # 合成管线编排（291 行）
│       │   ├── CosyVoiceStore.kt                # 模型/音色/注册管理（619 行）
│       │   ├── CosyVoiceVoiceProfile.kt         # 音色档案数据模型（66 行）
│       │   ├── CosyVoiceInstruction.kt          # Instruct2 指令预设（38 行）
│       │   ├── CosyVoiceAudioDecoder.kt         # MediaCodec 音频解码（215 行）
│       │   ├── CosyVoiceModelDownloader.kt      # HuggingFace 在线下载（160 行）
│       │   ├── CosyVoiceLlmNative.kt            # LLM JNI 接口（18 行）
│       │   ├── CosyVoiceFlowNative.kt           # Flow JNI 接口（19 行）
│       │   ├── CosyVoiceHiFTNative.kt           # HiFT JNI 接口（21 行）
│       │   └── CosyVoiceEnrollmentNative.kt     # 音色注册 JNI 接口（42 行）
│       └── jniLibs/arm64-v8a/                   # MNN 全家桶 + 5 个 JNI 包装 .so + Hexagon HTP stub
│       └── assets/hexagon/                      # DSP Skeleton（libMNN_htpops_skel.so，按 SoC 配套）
├── gradle/
│   └── libs.versions.toml           # 依赖版本目录
├── build.gradle.kts                  # 根构建脚本
├── settings.gradle.kts               # 项目设置
├── gradlew / gradlew.bat             # Gradle Wrapper
├── mnn-patches/
│   └── mnn-3.6.1-hexagon-stage-filter.patch  # MNN Hexagon 算子分阶段过滤补丁
├── research/
│   └── mnn-cosyvoice3/              # 完整模型移植研究脚本（可复现的蒸馏/构建/基准）
│       ├── scripts/*.py              # Python：蒸馏训练、模型导出、数值验证
│       ├── *.ps1                     # PowerShell：构建、运行、基准测试
│       ├── STAGE3_FEASIBILITY.md     # Flow 蒸馏可行性报告
│       └── VOICE_ENROLLMENT*.md      # 音色创建方案
└── docs/
    ├── DEVELOPMENT_STORY.md          # 开发全流程记录（含尝试路线总览）
    ├── RESEARCH_MEMORY.md            # 逐日研究记忆
    ├── PITFALLS_AND_FIXES.md         # 全历程踩坑 12 大类总结
    ├── ACCELERATOR_ADAPTATION_PLAN_v1.0.md  # NPU/GPU 定向改造计划
    └── NPU_RELEASE_VALIDATION.md     # v1.1.0 NPU 验证记录
```

### 核心代码行数统计

| 文件 | 行数 | 职责 |
|------|------|------|
| CosyVoiceStore.kt | 619 | 最核心：模型校验、ZIP 导入导出、音色管理、注册管理 |
| MainActivity.kt | 552 | Compose UI + Activity 生命周期 + 事件处理 |
| CosyVoiceRuntime.kt | 291 | 合成管线编排、Mutex 同步、性能报告 |
| CosyVoiceAudioDecoder.kt | 215 | MediaCodec 音频解码 + WAV 写入 |
| CosyVoiceModelDownloader.kt | 160 | HuggingFace 在线断点续传下载 |
| **合计** | **2,041** | |

---

## 从源码构建

### 环境要求

- JDK 17+
- Android SDK 36（compileSdk）
- Android NDK（仅编译原生库需要，已提供预编译 .so）

### 构建命令

```bash
# 1. 克隆
git clone https://github.com/Nian27/CosyVoice3-MNN.git
cd CosyVoice3-MNN

# 2. 构建 Debug APK
./gradlew :app:assembleDebug

# 3. 构建 Release APK（需配置签名）
./gradlew :app:assembleRelease

# 4. 安装
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 编译原生库

预编译的 .so 已包含在 `app/src/main/jniLibs/arm64-v8a/`。如需自己编译：

1. 使用 MNN 3.6.1、Android NDK r25c 和匹配设备的 Hexagon SDK。
2. 应用 `mnn-patches/mnn-3.6.1-hexagon-stage-filter.patch`。
3. 将 `mnn-jni/` 加入 MNN 构建并交叉编译 arm64-v8a。
4. 同时部署 `libMNN_htpops.so` 和 DSP Skeleton，不能只复制 Stub。

---

## 常见问题 / Troubleshooting

### Q：App 打开后闪退

1. 检查 Logcat：`adb logcat -s "CosyVoice*" "*:E"`
2. 确认 Android 8.0+ / arm64
3. 如果看到 `UnsatisfiedLinkError` → 确认 APK 包含 .so 文件（解压 APK 查看 `lib/arm64-v8a/`）

### Q：合成后没有声音 / 显示错误

1. **"LLM JNI 执行失败"** → 检查模型是否完整（"MNN 模型"卡片应该有绿色的"已安装"）
2. **"Flow JNI 执行失败"** → 检查日志 `run-{timestamp}/flow-report.jsonl`，常见原因：
   - OpenCL 设备不支持 → App 会自动使用 CPU
   - GPU 编译缓存损坏 → 删除模型重新导入
3. **"HiFT JNI 执行失败"** → 类似 Flow 的排查方法

### Q：合成很慢

- **冷态首次合成**：~10 秒正常，因为 GPU 需要编译 kernel
- **持续合成变慢**：检查是否有后台进程占用 CPU/GPU
- **非高通手机 **：Flow CPU 模式预计比 GPU 慢 2-3 倍，整体合成时间 ~8-15 秒

### Q："音色提示过长"错误

当前音色的 Token 数超过 125。解决方法：
- 点击该音色的"生成实时版"按钮
- 或用更短的参考音频重新创建音色

### Q：音色创建失败

1. 确认已导入音色创建扩展（~974 MB）
2. 确保截取 3-5 秒清晰单人语音
3. 确保填写的文字与音频内容一致
4. 查看 Logcat 中的错误码（见上面"常见错误码"表格）

### Q：导入 ZIP 时 SHA-256 校验失败

- ZIP 文件可能损坏 → 重新下载
- 文件被修改过 → 重新下载原始模型包
- 存储空间不足 → 确保有至少 1.4 GB 空闲

### Q：我想在非高通手机上测试

1. 安装 APK
2. 导入模型
3. App 会自动使用 CPU/OpenCL 兼容路径
4. 试听
5. **如果兼容模式也崩溃** → 请在 GitHub Issues 中提交 Logcat

---

## 技术栈

| 技术 | 用途 |
|------|------|
| **Kotlin 2.3.10** | 主语言 |
| **AGP 8.13.2** | Android Gradle 插件 |
| **JDK 17** | 编译 |
| **Jetpack Compose + Material 3** | UI 框架 |
| **Kotlin Coroutines** | 异步管线 / Mutex 同步 |
| **OkHttp 5** | HuggingFace 模型下载 |
| **阿里 MNN** | LLM / Flow / HiFT 推理引擎 |
| **OpenCL** | Flow GPU 加速（HiFT 实测 OpenCL 比 CPU 慢，保持 CPU） |
| **MediaCodec** | 参考音频解码 |
| **Gradle Version Catalog** | 依赖管理 |

---

## 模型信息

| 模型 | 框架 | 大小 | 许可证 | 来源 |
|------|------|------|--------|------|
| CosyVoice3-0.5B (MNN) | MNN | ~1.3 GB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| Speech Tokenizer v3 | MNN | ~924 MB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| CAM++ Speaker | MNN | ~27 MB | Apache 2.0 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |

模型下载链接：**[Hugging Face 模型仓库](https://huggingface.co/VicenTrent/Cosy-Voice-MNN)**

---

## 实验记录与现状说明

本项目是**持续进行的实验项目**。完整的开发历程、失败记录、踩过的坑和解决方案见：

| 文档 | 内容 |
|------|------|
| [docs/DEVELOPMENT_STORY.md](docs/DEVELOPMENT_STORY.md) | 完整开发历程：CrispASR/ggml+Vulkan 路线失败 → 转 MNN/OpenCL → Flow 蒸馏 → 真机调优全过程 |
| [docs/RESEARCH_MEMORY.md](docs/RESEARCH_MEMORY.md) | 各阶段决策、数据、教训与研究记忆 |
| [docs/PITFALLS_AND_FIXES.md](docs/PITFALLS_AND_FIXES.md) | 全历程踩坑与解决 12 大类总结（含汇总表、剩余 3 坑、三大教训） |
| [docs/NPU_RELEASE_VALIDATION.md](docs/NPU_RELEASE_VALIDATION.md) | MNN/Hexagon NPU 验证记录（部分成功） |
| [mnn-patches/mnn-3.6.1-hexagon-stage-filter.patch](mnn-patches/mnn-3.6.1-hexagon-stage-filter.patch) | MNN Hexagon 算子分阶段过滤补丁 |

### 坦诚说明（重要）

1. **MNN 路线是成功的，但实际效果没有想象中好**：全链路在 SM8850（荣耀 Magic8 Pro）真机跑通并发布了 v1.1.0，但热态 RTF 约 0.79~1.0（勉强实时），冷启动 10-15 秒，常驻内存 2.25 GB+，且只在 SM8850 一台设备验证过；非高通设备完全未验证。
2. **NPU（Hexagon）只部分成功**：仅 `q_proj` 单个算子放 NPU 通过验证（LLM wall time -5.94%、decode TPS +5.93%），收益有限；MNN Hexagon 需按 SoC 配套 stub/skel 库，默认不启用，不能作为正式 NPU 方案。
3. **后续 NPU/加速实验（QAIRT QNN 全图）仍在进行中**：目前进展与结论已总结在 [docs/PITFALLS_AND_FIXES.md](docs/PITFALLS_AND_FIXES.md)（A16W16 快但坏音、A16W8 per-channel 24.77ms 但 PCM 未达标等），未完成部分暂不公开实验产物细节。

本项目定位是"实验验证 + 单机可用"，不是开箱即用的通用本地 TTS 方案。

---

## 欢迎参与

**目前这个项目只在我的一台荣耀 Magic8 Pro（SM8850）上测试通过**。非高通设备（麒麟/天玑/Exynos）能不能跑、跑得怎么样，都需要大家帮忙验证。

### 你可以做什么

| 角色 | 能做 |
|------|------|
| **有非高通手机的人** | 安装 APK 试试能不能用，[提 Issue](https://github.com/Nian27/CosyVoice3-MNN/issues/new) 告诉我结果 |
| **Android 开发者** | 修 Bug、优化性能、加功能，直接提 Pull Request |
| **对 TTS 感兴趣的人** | 读代码、提建议、分享你创建的音色档案 |
| **任何用户** | 用用看，不好用就骂，骂完记得告诉我哪里不好 |

### 我知道的问题（还没修）

当前已知问题（非高通未验证、冷启动慢、NPU 未过音频门槛、长句退化等）统一维护在 [问题记录](#问题记录供后续开发者参考) 的"当前已知问题（2026-08）"列表，不再重复列举。

### 贡献方式

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/my-fix`）
3. 提交修改（`git commit -am 'fix: xxx'`）
4. 推送到分支（`git push origin feature/my-fix`）
5. 创建 Pull Request

---

## 许可证

- **应用代码（Kotlin/Java/Gradle）**：Apache 2.0
- **原生库（.so）**：从 CrispASR 和 MNN 构建，适用其各自的许可证
- **CosyVoice3 模型**：Apache 2.0（[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)）
- **MNN**：Apache 2.0（[alibaba/MNN](https://github.com/alibaba/MNN)）

---

## 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 阿里巴巴 CosyVoice 语音合成模型
- [alibaba/MNN](https://github.com/alibaba/MNN) — 阿里巴巴 MNN 推理引擎
- [CrispASR/mnn-cosyvoice-jni](https://github.com/crisp-oss/mnn-cosyvoice-jni) — MNN CosyVoice JNI 绑定
- [gedoor/Legado](https://github.com/gedoor/Legado) — 阅读 Archive，本项目的来源
- 所有在 [Issues](https://github.com/Nian27/CosyVoice3-MNN/issues) 中反馈问题、提交代码的贡献者

---

> **遇到问题？** → [提交 Issue](https://github.com/Nian27/CosyVoice3-MNN/issues/new)
>
> 提交时请附上：
> - 手机型号 / Android 版本
> - Logcat（`adb logcat -s "CosyVoice*" "*:E"`）
> - 如果是合成问题，提供 `run-{timestamp}/` 目录下的所有日志文件
