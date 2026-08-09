# 踩坑与解决办法汇总

> 整合自全历程回顾总结：从最开始迁移到 P4.5，遇到的坑归成 12 大类。
> 很多当时看起来像"NPU 不行"，最后发现根因完全不同。
> 大部分条目附细节（现象 → 根因 → 解决 → 结果）。

---

## 一、GPU / Android 后端坑：不是模型错，是运行环境先炸

最早走 ggml/CrispASR + Vulkan，LLM、Flow、HiFT 都出现过：

```text
vk::Queue::submit: ErrorDeviceLost
```

关闭 Vulkan FP16 能绕过 DeviceLost 但声音刺耳失真；纯 CPU 慢到约 37.6 s / 33字，因此整体迁到 MNN/OpenCL。

MNN/OpenCL 又遇到：

- **子进程不能继承 Android App 的 OpenCL/linker namespace** → 全部迁到 App 进程内 JNI 常驻（细节见 DEVELOPMENT_STORY Bug #3）。
- **Flow 动态 Shape 首次触发 12-17 秒 GPU kernel 编译** → 固定 bucket（Flow 桶表 256/384/512/768/1024/1280/1536/2048；HiFT mel 帧桶 128~2048）。256/384 小桶对 sequence 254/294 分别省 51%/27%，目标区域逐值一致（`max_abs=0`）。
- **OpenCL Buffer/Image/Auto 模式差距极大**：SM8850 上 Buffer(68) 热态约 0.786 秒 vs 自动(4) 约 2.5 秒 vs Image(132) 约 3.75 秒；但 Buffer(68) GPU PSS 约 1,267.8 MB，自动(4) 仅约 7.7 MB。不能默认自动模式最快，按设备实测选择。
- **HiFT CPU 线程**：High/6 约 1.35-1.39 秒，8 线程更慢且波动；Low/6 虽快但增益 +2.267 dB、SNR 8.24 dB、最大误差 0.761，**禁止用于产品**。
- **App bug**：`threads=6` 只传给了 F0，CPU HiFT core 误把 `hiftGpuMode=4` 当线程数 → 修正后热态 RTF 从 ~1.0 降到 0.89-0.90。
- **全 MNN 常驻内存 2.25 GB** → 整批预合成结束后 `CosyVoiceRuntime.close()` 释放 Session（队列代次保护，禁止逐句释放）；退出页必须在独立 IO scope 中关闭（先取消 scope 会导致释放任务永不执行，PSS 946.7 MB → 268.9 MB）。
- **MNN CPU 线程池缺陷**（threadpool-fix.patch）：worker 只检查 `mActiveCount`，栅栏等待时丢任务/忙等 → 改为扫描全部任务位，无 pending 才条件等待。

### 最终经验

```text
Android GPU 后端 ≠ PC 上模型能跑就能直接搬过去
```

必须同时考虑：linker namespace、Runtime 生命周期、Session 复用、动态 Shape、GPU kernel tuning、cache、驱动稳定性。

---

## 二、模型迁移最大的坑：转换成功 ≠ 函数没被改坏

旧链路 `PyTorch → ONNX → MNN → QNN` 里出现过：

1. **Transpose rank 错误**：4 维 Tensor 生成错误长度 perm `{0,2,1}`，破坏图验证。
2. **Resize/Resample 语义错误**：nearest 每个值应重复 8 次，QNN 一度首尾 0、中间没正确重复，卷积结果大幅偏离。
3. **LeakyReLU 转换错误**：`y=max(x,sx)` 实现有问题，手工改成 `y = sx + (1-s)·ReLU(x)` 后某层 corr 从 0.727 → 0.999999。
4. **WeightNorm/Constant/Snake 核查**：重新核查 80/80 Conv 权重、`conv_post`、WeightNorm 折叠、Snake α Constant、f0 predictor 是否被错误包含、`m_source` 是否越界进入 NPU 图，最终确认 17.4M 参数边界完整。
5. **MNN 外部权重缺失时"全零成功"**：缺 `.weight` 文件时打印 `Can't open file` 后仍返回全零输出 → 所有基准必须同时检查 finite、RMS 和 ONNX 数值误差。
6. **MNN QNN backend 三维卷积布局错误**（qnn-layout-fix.patch）：1D conv 的 3 维 NCHW `{n,c,h}` 送 QNN 需转 4 维 NHWC `{n,h,w=1,c}`（插入宽度轴并交换 c/h），改 `QNNBackend.cpp`/`QNNUtils.cpp`。

### 最终经验

任何转换必须建立四层验证链：

```text
PyTorch → ONNX golden → QNN CPU golden → HTP
```

不能直接 `converter成功 → context成功 → 宣布模型正确`。

---

## 三、App 里用 NPU 的坑：硬件能用，但 Android 权限链很折腾

最初误以为"荣耀系统是不是根本不允许第三方 App 访问 NPU"，后来证明不是。真正的问题：

```text
V81 Stub 版本必须匹配 / libcdsprpc 依赖 / libhidlbase namespace / Manifest uses-native-library
HTP skel 位置 / ADSP_LIBRARY_PATH / filesDir 权限 / unsigned PD
```

最终工作路径：

```text
App UID → libQnnHtp → libcdsprpc → FastRPC → unsigned PD → V81 skel → HTP
```

skel 不能随便放，要在 App 可访问位置配合环境变量。解决后 App 内 QNN 真实可运行。

### 经验

"App 不能用 NPU" 和 "模型在 NPU 上性能不好" 必须分开。当前已明确：NPU 访问权限 ✅、App QNN 集成 ✅、V81 HTP 执行 ✅——问题早已不是"荣耀不让用 NPU"。

---

## 四、整个项目最坑的一类：假加速、假执行、错误测速

1. **20 ms 的"超快 NPU"**：`MNN blocks ≈ 15–21 ms` 看似巨大加速，强制同步后发现 `runSession 真正完成 ≈ 105–108 ms/12帧`——异步"任务提交 ≠ 任务完成"。
2. **0.4 ms cache "神速"**：换输入后输出不变——根本没真实重新计算，属于**假执行**。
3. **单窗口快不代表全句快**：12 帧窗口 stride=8，342 帧要 43 窗（43×12=516，多算 51%）——必须乘完整调用次数。
4. **Batch 并没有并行**：Batch4 = 389 ms ≈ 4 × ~97 ms，无并行收益。
5. **长窗口更糟**：64 帧 = 1405 ms 严重超线性——窗口变大 ≠ 摊薄启动开销，可能跨过 VTCM/内存/kernel 阈值。

旧路线最终权威结果：`MNN→QNN FP32：105–108 ms/12帧` vs `CPU：≈36 ms/12帧`——**旧 NPU 失败本质是性能失败**。

---

## 五、音频后处理坑：Tensor 很接近，声音仍然能完全坏掉

1. **小窗口直接拼接**：8 帧窗口 → 50 Hz 嘀嘀声、边界跳变 → 改 12 帧 + 重叠。
2. **直接通道加权**：输出通道直接加权融合人为制造 6 kHz coherent carrier（-13.6 dB）→ 改复数域融合后恢复 -30.8 dB ≈ CPU。
3. **ISTFT 窗函数**：必须 `periodic Hann, n/16`，不是 `symmetric Hann, n/15`；还要 `window² overlap-add normalization`。
4. **torch.istft 行为**（P4.2）：把 1441 误认为采样点数，实际是 STFT 帧数；真实时域长度 `(T-1)×hop + n_fft`；`torch.istft` 逐点除 window envelope、`center=True` 裁剪，补齐后才与 Torch 逐采样点对齐。

### 经验

HiFT 验证至少三层：`Tensor指标 → 频谱指标 → 最终PCM/WAV`，不能只看 `corr(out18)`。

---

## 六、P3 Student 最大的坑：看起来是"数据不够"，最后发现函数本身难以表示

46 切片 → val corr 0.005；118 切片 → 0.29，很容易得出"多加数据就行"。证伪过程：

- `src_stft` 基本是纯 F0 正弦（高次谐波接近 0），但 Teacher 目标有复杂谐波和相位结构——Source 本身信息不足；
- 假设 `φk ≈ k·φsrc` 被数据否掉——相位不锁定于 source；
- **单样本过拟合测试**：20.8M 参数 Student 300 步连训练切片自己（corr≈0.007）都拟合不上；多个回归变体全部趋向静态谱/条件均值；4 个 demo 除 Teacher 外全是噪音。

结论：`P3 普通回归 Student ❌`。

### 教训

> 不能看到验证集提高就直接判断"继续扩大数据一定解决"。必须加**单样本过拟合测试**：模型连一个样本都记不住时，优先查结构表达能力、输出表示、对齐、损失函数，而不是继续堆数据。

---

## 七、P4 官方 QAIRT 路线的坑：这次终于把"快"和"对"分开了

彻底绕开 `MNN → QNN`，改成 `PyTorch → ONNX → qnn-onnx-converter → QAIRT/QNN`。环境坑：

```text
两个僵尸 converter 进程 / Python pyd 实际绑定 3.10 不是 3.12 / onnx 1.22 移除 onnx.version
```

解决：专用 `qnn310` env + onnx 1.21 + 杀掉僵尸进程，转换只需约 20 秒。

P4.2 关键 golden：

```text
QNN CPU vs ONNX:  PCM corr = 1.00000000  max_abs = 5.832e-6  RMSE = 4.37e-7
```

把"converter 语义错误"基本排除。

---

## 八、P4 量化阶段：局部 Tensor 很好，最终 PCM 仍然不够好（当前核心问题）

### 第一轮 A16W16

性能成功：`25.97 ms/12帧`（比 CPU ≈36 ms 快约 28%）；但 `corr ≈ 0.180` 音质完全不行；84 窗口重校准后 `corr ≈ 0.199` 几乎没救回来。P4.5 深入定位修正了早期怀疑：

> **第一个真正崩坏点不是 Snake，而是 `/ups.0/Conv` 的 W16 高深度累加。**

### 已验证失败的路线（不应重复）

```text
扩大 calibration ❌ / W16 headroom ❌ / 拆卷积 ❌ / W8/W16 混合整数图 ❌ / 盲目增加 FP16 节点 ❌/无明确收益
```

### 当前最佳图（A16W8 per-channel + conv_post FP16）

```text
性能 24.77 ms/12帧 ✅   84 窗平均 corr = 0.99756 ✅   最差窗口 47 corr = 0.96541
但最终波形：PCM corr = 0.83507 ❌   relative L2 = 0.56347 ❌
```

> **局部频谱 Tensor corr=0.9976 不代表最终时域波形接近。** 幅度/相位/重叠窗口/ISTFT/OLA 会把很小但系统性的误差放大（尤其 window 47 这类局部异常，经重叠和相位传播后对 PCM 的影响远大于它在 Tensor corr 里的权重）。

以后 `out18 corr` 只能作为中间指标，最终 Gate 必须是：`PCM corr / relative L2 / 高频异常 / clipping / 试听`。

---

## 九、Flow GPU 独立的坑

长句 `Flow 总时间 ≈ 7.0s`，拆开：`resize ≈ 4.5s`、`infer ≈ 2.5s`。F1 证明同一 Session 同 bucket：`1269.8 ms → 0 → 0`——核心是**首次 bucket 调优 / Session 生命周期**，不是模型本身每次需要 4.5 秒。

方案：`共享 OpenCL Runtime / 最多 1–2 个常驻 Session / LRU / 串行 / 按需创建 / 后台预热`。

另外重复 OpenCL benchmark 出现过黑屏、Bluetooth HAL died——只能记录为"高度怀疑 vendor GPU/OpenCL 链路"，没有 KGSL/Adreno fault 完成归因，不能写成"已证明是驱动 bug"。

---

## 十、LLM 认知坑：官方"NPU 快 7.9 倍"不等于我们的 TTS 快 7.9 倍

MNN 3.6.1 官方 `Qwen3-0.6B Prefill: Hexagon ≈ CPU 的 7.9×`，但 App 实测：

```text
短句：Prefill ≈300ms，Decode ≈1.2–1.3s
长句：Prefill ≈408ms，Decode ≈5.4s（Decode 占 80%～93%）
```

即便 Prefill 变 40ms，端到端只改善几个百分点——必须看 **Amdahl's Law**。LLM 优化重点：`CPU Decode / C4 / KV Cache / W4 / packed RoPE`，而不是先折腾 Hexagon Prefill。

相关：LLM q_proj 单算子放 NPU 验证成功但收益很小（wall time -5.94%、TPS +5.93%），且逐算子出现过 Token 崩溃/非法 Token，未通过正确性验证的算子不得启用；MNN Hexagon 需按 SoC 配套 stub/skel 库。

---

## 十一、长句"长度退化坑"（尚未彻底解决）

```text
短句 HiFT：342帧 1.35s ≈ 3.9ms/帧
长句 HiFT：864帧 ≈7.0s ≈ 8.1ms/帧（单位帧耗时翻倍，非线性）
```

长 Shape 下存在 cache/memory、kernel 算法切换、线程调度、临时 buffer、大 Tensor 带宽之一。若 NPU 最终不可用，CPU fallback 必须处理该退化。

---

## 十二、汇总表

| 类别 | 当时看到的现象 | 真正根因 | 状态 |
|------|--------------|---------|------|
| Vulkan | DeviceLost | Adreno/运行环境兼容性 | 放弃路线 |
| OpenCL FP16 | 声音坏 | Softplus/GELU/Attention 数值范围 | 改 High |
| 动态 Flow | 12–17s | 首次 GPU 编译/resize | bucket 解决 |
| 子进程 GPU | OpenCL 不可用 | Android namespace | JNI 解决 |
| QNN App | NPU 加载失败 | Stub/skel/FastRPC/native lib | 已解决 |
| MNN→QNN | Tensor 错 | Transpose/Resize/LeakyReLU 转换 | 已解决 |
| 20ms NPU | 看似超快 | 异步提交 | 假结果 |
| 0.4ms cache | 看似神速 | 未真实执行 | 假结果 |
| 8帧 HiFT | 嘀嘀/接缝 | 上下文不足 | 12帧解决 |
| 6kHz 刺耳 | 高频载波 | 错误直接通道融合 | 复数融合解决 |
| ISTFT | 波形不对 | Hann/OLA/center 语义 | 已解决 |
| 旧 NPU | 正确但 105ms | FP32 + MNN/QNN 图效率差 | 放弃旧链 |
| P3 Student | 泛化差 | Student 连单样本都表达不了 | 关闭 |
| QAIRT 转换慢 | 卡很久 | 僵尸进程 + Python/ONNX 环境 | 已解决 |
| QNN Converter | 是否语义正确 | P4.2 WAV corr=1 | 已证明 |
| A16W16 | 26ms 但坏音 | W16 高深度累加先崩 | 已定位 |
| 增大 calib | 无改善 | 非简单 range 估计问题 | 停止 |
| A16W8 PC | 24.77ms、tensor 很好 | 时域误差仍被放大 | **当前问题** |
| Flow 重复 GPU 测试 | 黑屏 | vendor 链路高度怀疑 | 避免压力测 |
| LLM NPU | 官方 7.9× | Prefill 占比太小 | 优先级降低 |
| 长句 HiFT | ms/frame 翻倍 | 长 Shape 资源退化 | 未完全解决 |

---

## 当前真正还剩下的坑只有三个

1. **NPU HiFT 最终 PCM 保真**（当前 P4 主问题）：24.77ms ✅、84 窗 corr 0.99756 ✅，但 PCM corr 0.835 ❌、relative L2 0.563 ❌。下一步：window 47 → 18 通道 → 幅度/相位误差 → 哪个 stage 开始系统漂移，再决定是否建立连续 late-stage FP16 island，而不是继续随机加 FP16 节点。
2. **Flow Session 池产品稳定性**：正常合成、冷启动、长句、连续使用稳定性验收。
3. **长句 CPU HiFT 退化**：864 帧为什么从 3.9ms/frame 恶化到 8.1ms/frame。

---

## 整个项目最大的三个教训

1. **"能跑"不等于"算对"，"算对"不等于"快"，"局部快"不等于"端到端快"。**
2. **音频模型不能只看中间 Tensor 相关系数。最终 PCM、相位、高频和试听必须是最后 Gate。**
3. **NPU 不是把原模型扔进去就会自动快。真正成功需要模型精度、算子形态、编译器、内存布局和硬件数据路径同时匹配。**

当前与最开始最大的不同：**已经真实证明 HiFT 在 SM8850 HTP 上可以做到 24.77ms/12帧。NPU 性能问题基本解决；剩下的是如何在这个速度下把最终 PCM 从 0.835 拉回接近 Teacher。**

---

> 相关文档：`docs/DEVELOPMENT_STORY.md`（开发全流程 + 尝试路线总览）、`docs/RESEARCH_MEMORY.md`（逐日记忆）、`docs/NPU_RELEASE_VALIDATION.md`（NPU A/B 验证）、`docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md`（加速器改造计划，冻结基线 `baseline-20260806`）。
