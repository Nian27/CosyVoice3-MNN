# Fun-CosyVoice 开发全流程记录

> 从 Legado Archive 提取 CosyVoice3 本地 TTS 为独立 App 的完整过程。
>
> 本文档记录了每一步做什么、为什么做、改了什么文件，方便后续开发者理解代码演变和定位问题。

---

## 目录

1. [动机：为什么从 Legado 独立出来](#1-动机为什么从-legado-独立出来)
2. [代码剥离（基础版）](#2-代码剥离基础版)
3. [第一轮编译问题修复](#3-第一轮编译问题修复)
4. [Flow CPU / OpenCL 双后端](#4-flow-cpu--opencl-双后端)
5. [ZIP 导入/导出模型系统](#5-zip-导入导出模型系统)
6. [音色档案管理](#6-音色档案管理)
7. [手机创建音色（Enrollment）](#7-手机创建音色enrollment)
8. [GPU 调度 UI](#8-gpu-调度-ui)
9. [Bug 修复记录](#9-bug-修复记录)
10. [APK 构建和发布](#10-apk-构建和发布)
11. [后续可能的改进方向](#11-后续可能的改进方向)

---

## 1. 动机：为什么从 Legado 独立出来

### 背景

Legado / 阅读 Archive 的 `cosytest` 分支（`app/src/cosytest/`）在 v3.3.26.06071119 中集成了完整的 CosyVoice3 本地 TTS 功能。代码结构：

```
app/src/cosytest/
├── java/io/legado/app/
│   ├── cosy/
│   │   ├── CosyVoiceNative.kt         # 原生接口集合（Llm/Flow/HiFT/Enroll）
│   │   ├── CosyVoiceRuntime.kt        # 合成管线
│   │   ├── CosyVoiceVoiceProfile.kt   # 音色档案
│   │   └── CosyVoiceModelStore.kt     # 模型/音色管理
│   ├── tts/TtsCosyVoice.kt            # 阅读 TTS 引擎适配
│   └── ui/config/
│       └── CosyVoiceManagerActivity.kt # 管理界面
```

但在 Legado 内部存在以下问题：

1. **耦合严重**：与阅读的朗读流程、UI 框架高度绑定，不能独立测试
2. **非高通设备闪退**：Flow 硬编码 OpenCL，Manifest 中 `libOpenCL.so required="true"`
3. **难以调试**：集成在大型项目中，每次改 TTS 都要编译整个 App
4. **难以复现**：用户报的 Bug 需要在特定手机上运行阅读 App，而不是独立测试

### 目标

创建一个最小化的独立 App：
- 只保留 CosyVoice3 合成功能
- 支持所有 arm64 Android 设备（不限于高通）
- Compose Material 3 UI
- 开源，方便协作

---

## 2. 代码剥离（基础版）

### 步骤 1：创建新项目

用 Android Studio 创建新项目 `Fun-CosyVoice`，包名 `com.cosyvoice.app`，Compose 模板。

### 步骤 2：复制核心代码

从 Legado 复制到新项目的文件：

| 来自 Legado | 到 Fun-CosyVoice | 改动 |
|-------------|------------------|------|
| `CosyVoiceNative.kt` | → 拆分 4 个文件（Llm/Flow/HiFT/Enroll） | 每个 JNI 绑定独立 |
| `CosyVoiceRuntime.kt` | → `CosyVoiceRuntime.kt` | 删掉阅读相关的 TTS 适配，只保留 synthesize() |
| `CosyVoiceVoiceProfile.kt` | → `CosyVoiceVoiceProfile.kt` | 基本不变 |
| `CosyVoiceModelStore.kt` | → `CosyVoiceStore.kt` | 包名改动，简化在线下载 |
| `CosyVoiceInstruction.kt` | → `CosyVoiceInstruction.kt` | 基本不变 |
| `CosyVoiceAudioDecoder.kt` | → `CosyVoiceAudioDecoder.kt` | 不变 |
| `CosyVoiceModelDownloader.kt` | → `CosyVoiceModelDownloader.kt` | 不变 |
| `CosyVoiceManagerActivity.kt` | → `MainActivity.kt` | 完全重写为 Compose |

### 步骤 3：复制 .so 文件

Legado 的 jniLibs 目录有 5 个 .so 文件已经编译好，直接复制。

### 步骤 4：包名和 import 修改

Legado 代码的包名是 `io.legado.app.cosy`，新项目改为 `com.cosyvoice.app`。所有 import 和 `R.xx` 引用都需要更新。

### 步骤 5：重写 UI

原来 Legado 用的是传统的 Activity + XML 布局。新项目全部用 Jetpack Compose Material 3。

UI 组件：
- 模型管理卡片（导入/导出/删除）
- GPU 调度卡片（CPU/OpenCL 切换）
- 音色选择和管理卡片
- 创建音色卡片
- 试听卡片

---

## 3. 第一轮编译问题修复

### 问题：Missing `R.id.xxx` / `Context` 相关

因为从 Legado 的 XML 布局改为 Compose，所有 `findViewById()`、`R.layout.xxx`、`R.id.xxx` 都删掉了。Compose 用 `@Composable` 函数替代。

### 问题：`applicationId` 冲突

Legado 使用 `io.legado.app`，新 App 使用 `com.cosyvoice.app`。AndroidManifest.xml 中的 package、Service 注册等都要更新。

### 问题：`System.loadLibrary("OpenCL")` 在非高通手机上抛异常

原来 Legado 的 Manifest 有：
```xml
<uses-native-library android:name="libOpenCL.so" android:required="true"/>
```

修复：改为 `required="false"`，运行时检测。

### 问题：Compose Navigation 依赖

Legado 使用了 Navigation Compose，新 App 简化为一屏展示，不需要导航。

---

## 4. Flow CPU / OpenCL 双后端

### 为什么重要

CosyVoice3 合成管线中最耗时的阶段是 Flow（Token → Mel 频谱）。原来只在 OpenCL GPU 上跑，非高通设备（麒麟/天玑）没有 OpenCL 支持，直接崩溃。

### 实现方案

`CosyVoiceSynthesisOptions` 新增 `flowBackend` 字段：
```kotlin
data class CosyVoiceSynthesisOptions(
    val flowBackend: String = CosyVoiceRuntime.detectBestFlowBackend(),
    // ...
) {
    val flowPrecision: String get() = if (flowBackend == "cpu") "normal" else "high"
    val flowThreads: Int get() = if (flowBackend == "cpu") 6 else flowGpuMode
}
```

自动检测：
```kotlin
val openClAvailable: Boolean by lazy {
    try {
        System.loadLibrary("OpenCL"); true
    } catch (_: Throwable) { false }
}

fun detectBestFlowBackend(): String {
    return if (openClAvailable) "opencl" else "cpu"
}
```

`CosyVoiceFlowNative.run()` 把 `backend` 参数传给 JNI：
```kotlin
val flowExitCode = CosyVoiceFlowNative.run(
    modelPath = ..., backend = options.flowBackend,
    precision = options.flowPrecision, threads = options.flowThreads,
    ...
)
```

### 注意事项

- CPU 模式使用 6 线程，`precision="normal"`（FP16 模型在 CPU 上用 FP32 精度）
- GPU 编译缓存按 `{backend}-{precision}-mode{mode}` 分别存储
- 切换 backend 后首次合成较慢（需要重新编译 GPU kernel 或预热 CPU）

---

## 5. ZIP 导入/导出模型系统

### 设计原则

1. **完整性校验**：每个文件记录预期大小 + SHA-256
2. **原子操作**：先解压到 staging 目录，校验通过后再逐个 move 到目标目录
3. **错误恢复**：导入失败不破坏已有文件
4. **无压缩导出**：导出时用 `Deflater.NO_COMPRESSION`，避免二次压缩

### 模型文件清单

在 `CosyVoiceStore` 中定义，每个文件有 name/bytes/sha256 三个字段：

```kotlin
val MODEL_FILE_SPECS = listOf(
    CosyVoiceModelFileSpec("llm.mnn", 419_640L, "939C3646..."),
    CosyVoiceModelFileSpec("llm.mnn.weight", 352_844_666L, "325948C3..."),
    // ... 共 17 个文件
)
```

### 导入流程

```
用户选择 ZIP
    │
    ▼
openInputStream(uri) → ZipInputStream
    │
    ▼
遍历 ZIP 条目 → 匹配 MODEL_FILE_SPECS → 解压到 staging/
    │
    ▼
校验每个文件：大小 + SHA-256
    │
    ▼
逐个 .importing 写入 → rename → 覆盖原有文件
    │
    ▼
完成，删除 staging/
```

### 导出流程

```
用户选择输出 URI
    │
    ▼
openOutputStream(uri) → ZipOutputStream(NO_COMPRESSION)
    │
    ▼
遍历 MODEL_FILE_SPECS → 逐个写入 ZIP 条目
    │
    ▼
完成
```

---

## 6. 音色档案管理

### 什么是音色档案

音色档案是一组文件，包含：
1. **参考语音 Token**（prompt-speech-tokens.csv）：一串数字，表示说话人的语音特征
2. **Conditioner 输出**（prompt-cond.bin）：80ch × frame × 4B 的 Mel 频谱
3. **说话人嵌入**（spks.bin）：80ch × 4B = 320B 的 Speaker 向量
4. **元数据**（profile.json）：名称、Token 数、哈希值等

### 内置音色

从模型包中的文件自动构建。ID: `builtin-mnn-reference-v1`，87 Token。不可删除。

### 导入/导出

支持单个和批量导入。导入时校验：
- `modelId` 必须匹配当前模型
- Token 数必须在 1-375 范围内
- 所有必需的二进制文件必须存在

### 实时音色

对于 Token 数 > 125 的音色，合成时会报错。需要生成"实时版"：
1. 截取前 125 个 Token
2. 截取对应的 Conditioner 数据（80ch × 125 帧）
3. Speaker 向量不变
4. 自动命名"原名称（实时）"

---

## 7. 手机创建音色（Enrollment）

### 需要的额外模型

| 文件 | 大小 | 作用 |
|------|------|------|
| `speech-tokenizer-v3.fp32.inline.mnn` | ~924 MB | 语音 Tokenizer |
| `campplus.fp32.mnn` + `.weight` | ~27 MB | 说话人识别 |
| `flow-speaker-affine-weight.bias` | ~60 KB | Speaker 投影 |

### 流程详解

```
用户选择音频文件 + 填写信息
    │
    ▼
1. MediaCodec 解码 → PCM WAV
    │
    ▼
2. 释放合成模型（Runtime.close()）→ 节省内存
    │
    ▼
3. 语音 Tokenizer → 提取 Token
    │
    ▼
4. CAM++ Speaker → 提取 Speaker 向量
    │
    ▼
5. 说话人特征投影 → spks.bin
    │
    ▼
6. 写入音色档案目录 → 导入到系统
    │
    ▼
完成
```

### 关键限制

- 参考音频必须 3-15 秒
- 生成的 Token 不能超过 125 个（约 5 秒语音）
- 文字必须与参考音频内容完全一致（LLM 训练时要求）
- 需要约 1 GB 额外内存（Tokenizer 模型 ~924 MB）

---

## 8. GPU 调度 UI

### 设计思路

用户可能需要在不同设备上调试 GPU 后端，所以 UI 提供完整的控制：

| 控件 | 控制目标 | 可选值 |
|------|----------|--------|
| Flow 后端 | Flow 阶段使用 CPU 还是 OpenCL | `cpu` / `opencl` |
| Flow Mode | OpenCL 内存模式 | 自动(4) / Buffer(68) / Image(132) |
| HiFT 后端 | HiFT 阶段核心网络使用 CPU 还是 OpenCL | `cpu` / `opencl` |
| HiFT Mode | HiFT OpenCL 内存模式 | 自动(4) / Buffer(68) / Image(132) |

UI 使用 `ChoiceButton` 组件（选中的是填充按钮，未选中的是描边按钮），通过 Compose `mutableStateOf` 保持状态。

### 默认值

```kotlin
detectBestFlowBackend()  // 有 OpenCL 返回 "opencl"，否则 "cpu"
```

---

## 9. Bug 修复记录

按时间顺序排列：

### #1: `libOpenCL.so required="true"` → 非高通设备崩溃

- **文件**: `AndroidManifest.xml`
- **改动**: `required="true"` → `required="false"`
- **影响**: 所有非高通设备现在可以安装 App

### #2: Flow 后端硬编码 OpenCL

- **文件**: `CosyVoiceRuntime.kt`（Legado 版本）
- **改动**: 添加 `CosyVoiceSynthesisOptions.flowBackend` + 运行时检测
- **影响**: 非高通设备可以使用 CPU 合成

### #3: `extractNativeLibs` 未配置

- **文件**: `app/build.gradle.kts`
- **改动**: 添加 `jniLibs.useLegacyPackaging = true`
- **影响**: 修复 Android 12+ 上 .so 加载失败

### #4: 首次打开 App 空指针

- **文件**: `CosyVoiceStore.kt` 的 `selectVoiceProfile()`
- **改动**: 添加 `if (modelStatus().ready)` 保护
- **影响**: 未导入模型时打开 App 不再崩溃

### #5: Flow 多线程安全

- **文件**: `CosyVoiceRuntime.kt`
- **改动**: 整个 synthesize() 用 `Mutex.withLock` 保护，不允许并发合成
- **影响**: 防止多个合成请求互相干扰

### #6: GPU 编译缓存无清理

- **文件**: `CosyVoiceStore.kt`
- **改动**: `deleteModel()` 增加 `gpuCacheDir.listFiles()?.forEach(File::delete)`
- **影响**: 删除模型时 GPU 缓存也被清理

---

## 10. APK 构建和发布

### Debug APK

```bash
./gradlew :app:assembleDebug
# 输出：app/build/outputs/apk/debug/app-debug.apk (~44 MB)
```

### 构建产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `app-debug.apk` | ~44 MB | Debug 版本，含 .so（32.7 MB）+ Kotlin 代码 + Compose 依赖 |
| （不含模型） | — | 模型需通过 ZIP 导入 |

### 安装

```bash
adb install -r app-debug.apk
```

---

## 11. 后续可能的改进方向

### 高优先级

1. **非高通手机实测**：目前 CPU 模式的性能数据是预估的，需要在麒麟/天玑上跑一遍
2. **Flow CPU 优化**：MNN CPU 后端对 FP16 模型的推理优化可能不如 FP32 充分
3. **更多 JNI 错误码文本**：目前 LLM/Flow/HiFT 的错误码只返回数字，用户看不懂

### 中优先级

4. **模型分片下载**：1.3 GB 的 ZIP 下载体验不好，可以分片 + 断点续传
5. **English 界面**：目前只有中文界面
6. **批量合成**：批量处理多个文本

### 低优先级

7. **更多推理引擎**：支持 SNPE / CoreML / NNAPI 等
8. **TTS 缓存**：相同文本直接返回缓存结果
9. **音色市场**：用户共享音色档案的平台

---

> **最后更新**：2026-07-21
>
> **作者**：Nian27
>
> **反馈**：[GitHub Issues](https://github.com/Nian27/Fun-CosyVoice/issues)
