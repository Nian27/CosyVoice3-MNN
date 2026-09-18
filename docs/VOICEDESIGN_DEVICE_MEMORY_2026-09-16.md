# VoiceDesign 真机内存与 Decoder 配置实测结论

日期：2026-09-16
设备：SM8850 / BKQ-AN90 / Android 17 / arm64-v8a（Honor HyperHold 生效）
相关代码：`ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp`、`app/src/main/java/com/fun/cosyvoice/`

## 1. 结论

在当前设备与 MNN RuntimeManager 生命周期下，VoiceDesign 的分阶段「释放 Talker → 加载内嵌 Decoder」方案未能有效回收已加载模块占用的匿名内存，因此无法通过 HyperHold 内存约束。

当前唯一经过完整真机验证、能够稳定连续运行的方案为：

**Talker 与 Tokenizer Decoder 均采用外置权重模式。**

该配置的代价是 Decoder 加载与执行速度显著下降，但可将峰值 HWM 控制在约 0.90 GB，避免设备进入 HyperHold 换出或直接被系统终止。

## 2. 分阶段释放实验

计划在 VoiceDesign 自回归生成结束后释放 GraphB、codec head、frame embedding、code predictor 等 Talker 模块，再加载内嵌 Tokenizer Decoder，以避免两阶段模型内存重叠。

真机实测：

| 打点 | anon |
|---|---:|
| tokenizer + text table | 575,448 kB |
| Talker Module reset 后 | 518,196 kB |
| 内嵌 Decoder 加载后 | 970,164 kB |

Talker 模块释放后 anon 仅下降约 **56 MB**，远低于预设的 **≥400 MB** Gate。

因此：

**G1 释放有效性 FAIL，按预设回退条件终止分阶段方案。**

**在当前共享 `RuntimeManager/Executor` 生命周期下，释放 `Module` 后相关 anon 未被有效归还，实测仅下降 56 MB，因此分阶段释放无法满足本设备内存 Gate。**

该结论仅针对当前实现和真机环境，不泛化为 MNN 所有 Runtime/Session 使用方式。

## 3. 五种配置真机对比

| 配置 | anon | 峰值 HWM | Decoder | 结论 |
|---|---:|---:|---:|---|
| 全内嵌基线 | 1.11 GB | 2.17 GB | 约 20 s | FAIL，内存过高 |
| **全外置（当前正式配置）** | **0.61 GB** | **0.90 GB** | **约 459 s** | **PASS，已完整跑通 2 次** |
| Decoder 内嵌 | 0.76 GB | 1.86 GB | 约 20 s | FAIL，HyperHold |
| Decoder FP16 | — | ~1.13 GB | 约 20 s | FAIL，PCM 最大误差 2693 LSB |
| 分阶段释放 + 内嵌 Decoder | 0.97 GB | 1.93 GB | 约 20 s | FAIL，释放无效且峰值过高 |

本设备长期可用内存较紧张。实测当 HWM 超过约 **1.2 GB** 后，HyperHold 换出及系统内存压力显著增加。

因此当前产品优先级为：

**稳定运行 > Decoder 加载速度。**

### 3.1 关于外置 Decoder 慢 23 倍的机制

MNN 源码依据（`source/core/OpCommonUtils.cpp`）：

```cpp
bool usemmap = false;
    usemmap = (backend->getRuntime()->hint().useCachedMmap > 1);
if ((!usemmap)) {
    bool res = _RebuildExternalOp(externalFile, op, builder);   // 把外部权重读进内存
}
```

```cpp
// source/core/Session.cpp
case Interpreter::HintMode::USE_CACHED_MMAP: runtimeHint.useCachedMmap = value; break;
case MNN::Interpreter::EXTERNAL_WEIGHT_DIR:  runtimeHint.weightMemoryPath = path; break;

// source/backend/cpu/CPUBackend.cpp
if (hint().useCachedMmap) {
    syncValid = MNNFileExist("...sync.static");
    hint().useCachedMmap += syncValid;      // 1 -> 2（生成 sync 后才走 mmap）
}
```

据此尝试过 `USE_CACHED_MMAP + EXTERNAL_WEIGHT_DIR` 组合（这是外置权重不慢的官方路径）。**实测失败**：MNN 会把权重复制进 `<prefix>.static`（默认 `mmapFileSize=1024MB`），实测写到 1,073,741,824 B 即停止推进、进程卡在设计阶段；且该机制本质是「用磁盘换速度」，与省内存目标方向相反。已撤回。

同样尝试过 `RuntimeManager::setCache`（OpenCL kernel 缓存，CosyVoice 侧 Flow/HiFT 一直这么做）用于降低 prefill 冷启动开销（实测冷 64.1 s / 热 35.6 s）。**实测失败**：设了之后进程会卡在 decoder 之后不再推进（frame 97 之后 10 分钟零输出，RSS 停在 1.59 GB，且始终没有 .cache 文件生成），收益未证实而风险明确，已撤回。

### 3.2 Decoder T 缩减（300 → 96）实测无效

为降低 Decoder 会话缓冲，导出过 T=96 的静态图。数值 Gate 通过（同一 codes 下 `cos=1.000000000`、int16 后 max 1 LSB、858/130560 = 0.66% 样本差 1 LSB）。

但**内存与磁盘均无收益**：`t300.mnn` 457,023,384 B vs `t96.mnn` 456,941,592 B（差 0.018%），anon 也未下降 —— 权重占 435 MB 且与 T 无关，T 只影响会话缓冲。该改动不保留。

### 3.3 FP16 Decoder 实测 FAIL

`--fp16` 转换可把 Decoder 从 457,023,384 B 降到 228,797,140 B（正好减半），但数值不可接受：

```
cos        = 0.999920309      （整体波形相似）
maxdiff    = 8.220e-02
int16 不等  = 176400 / 188160  (93.75%)
int16 最大差 = 2693 LSB        ← 完全不可接受
```

vocoder 对权重相位极度敏感，FP16 的微小扰动会造成波形级重构差异。**不使用 FP16 Decoder。**

## 4. 关于 VoiceDesign 与 CosyVoice 的直接结合

当前 VoiceDesign 与 CosyVoice 使用不同的语音 tokenizer / codec 表示。

因此，在**不训练额外跨模型映射网络、保持现有 CosyVoice enrollment 契约**的前提下，不能直接把 Qwen3-TTS VoiceDesign 的 codec token 作为 CosyVoice VoiceProfile。

当前可靠链路仍为：

`文字描述 → VoiceDesign codec → Tokenizer Decoder → reference WAV/PCM → CosyVoice enroll → VoiceProfile`

其中「WAV 落盘再读取」本身不是主要瓶颈。即使改成 Decoder PCM 直接传给 enrollment，收益预计很小（中间 WAV 仅 370 KB，写+读 <100 ms），无法解决 Tokenizer Decoder 本身的主要耗时。

未来可独立研究：

`VoiceDesign representation → CosyVoice speaker embedding / VoiceProfile`

但该方向属于跨模型音色空间对齐研究，不属于当前工程闭环。

## 5. 当前性能结论

当前全外置配置能够稳定完成文字设计音色，但 Tokenizer Decoder 为主要耗时来源。

因此，在当前设备、当前 MNN 外置权重实现以及已验证模型配置下：

**一次 VoiceDesign 音色创建约为分钟级一次性成本；音色创建完成后可缓存 VoiceProfile，后续小说朗读无需再次执行 VoiceDesign。**

该时间不是模型或系统的理论下限。后续仍可通过以下方向优化：

- Tokenizer Decoder 的 HTP/NPU 部署；
- MNN external-weight 加载路径优化；
- Decoder 生命周期与独立 Runtime 隔离；
- 跨模型 VoiceIdentity bridge。

## 6. 当前正式配置

现阶段冻结：

- VoiceDesign Talker：外置权重；
- Tokenizer Decoder：外置权重；
- 不使用 FP16 Decoder；
- 不使用同 RuntimeManager 内的分阶段 Module 释放；
- 不改变现有 CosyVoice enrollment 接口；
- VoiceDesign 完成后生成参考音频并自动 enrollment 为统一 `VoiceProfile`。

该配置作为下一阶段 UI、VoiceProfile 管理和完整 ReaderVoice 集成的稳定基线。

## 7. 后续优化纪律

不要在本基线上随机试内存方案。若继续优化 Decoder，应单独开 **Decoder-HTP / 独立 Runtime** 分支，不再扰动上述已冻结的产品基线。

## 附：本轮同时交付的 UI 进度反馈

一次设计耗时分钟级，此前 UI 上只有一句「正在生成」，用户无法判断是否卡死。现在 native 每帧写 `<模型目录>/progress.txt`，UI 每秒轮询显示：

```
正在准备 prompt
prefill 完成（54 token，32 秒），开始逐帧解码
解码中 23 帧（约 1.84 秒音频）· 已用 43 秒 · 约 0.57 秒/帧 · 预计还需 0 分 49 秒
解码结束：共 98 帧（7.84 秒音频），EOS 自然停止。正在生成波形…
波形已生成（188160 samples / 7.84 秒），正在写 WAV
```

ETA 用**最近 20 帧的滑动窗口**速率，而非全程平均 —— 前几帧含 OpenCL kernel 编译，用全程平均会把 ETA 高估十几倍（实测第 5 帧算出「还需 13 分 11 秒」，实际 45 秒）。

## 8. 关键通用修复：enrollment 前的 dither（两条路径都要）

初次验收「参考音频克隆」路径时直接失败：

```
enroll exitCode=37 (说话人特征无效)
```

**根因**：VoiceDesign 生成的音频里，静音是**精确 0 样本**（实测某条 7.76 s 样本中 19.9% 为 0，fbank 774 帧里 12 帧全 0）。enrollment 内 CAMPPlus 的前置 fbank 会取 `log(能量)`，`log(0) = -inf` → NaN 传播 → campplus 输出非有限 → `embedding.size()==192 && finiteVector()` 判定失败 → 返回 37。真实录音有本底噪声，永不精确为 0，所以不会踩到。

**修法**：在 **enrollment 之前无条件**给 WAV 叠加 ±1 LSB 的确定性 dither（LCG，可复现，约 -90 dB 听不见）。两处都要：

| 路径 | 位置 | 说明 |
|---|---|---|
| 文字设计音色 | `CosyVoiceVoiceDesigner.createFromDesign` | 设计产物 → enroll 之前 |
| 参考音频克隆 | `MainActivity.createVoiceProfile` | 解码截取后 → enroll 之前 |

**这不是设计路径的补丁，而是 enrollment 的前置条件**：用户完全可能把合成音频（含 VoiceDesign 产物）当参考音频导入克隆，不加就必然失败。真实录音加 dither 也无害。

**验证结果**：
```
已补 dither -> clone-dither.wav
enroll exitCode=0 (成功)
音色已注册: voice-1789551292376 / 克隆音色B / tokens=125 / frames=250
=== 克隆成功，耗时 4882 ms ===
```
随后用该克隆音色合成一句话通过：`20.00 秒音频 · peak 0.979 · rms 0.120 · finite=True`（PC 侧逐项复核一致）。

**教训**：dither 属"合成音频作为 enrollment 输入"的通用前置处理，不得只加在产生合成音频的那条链上。

## 9. 两条音色创建路径的端到端验收（2026-09-16）

| 路径 | VoiceProfile | 结果 |
|---|---|---|
| 文字设计音色 | `voice-1789540901382`（HOT2） | ✅ 设计(98帧 EOS) → enroll(125 Token) → 注册 → 合成 5.44 s |
| 参考音频克隆 | `voice-1789551292376`（克隆音色B） | ✅ enroll(125 Token, 4.88 s) → 注册 → 合成 20.00 s |

两者共用同一个 `installEnrolledVoiceProfile()`，产出同一种 `VoiceProfile` —— 下游合成侧零改动。

