# 踩坑与解决办法汇总（补充记录）

> 本文件补充 `DEVELOPMENT_STORY.md` 与 `RESEARCH_MEMORY.md` 未覆盖的后期阶段坑：
> 2026-07-23 真机调优阶段、以及 MNN/Hexagon QNN NPU 实验阶段。
> 每条均给出：现象 → 根因 → 解决方案 → 结果。

---

## 一、真机调优阶段（2026-07-23，Honor Magic8 Pro / SM8850）

### 1. LLM 线程数：6 线程反而比 4 线程慢

- **现象**：真机交错测试 LLM 4/6 线程，4 线程为 65.13 / 67.37 / 68.86 tok/s，6 线程为 45.37 / 57.23 / 54.22 tok/s。
- **根因**：LLM 是内存带宽瓶颈，多线程无收益；6 线程还会与 HiFT CPU/6 争抢核心。
- **解决**：LLM 固定 4 线程，不改 6。

### 2. HiFT CPU 线程数：8 线程更慢，Low 精度直接禁用

- **现象**：HiFT CPU High/8 不稳定；Low/6 虽然最快（约 0.77-0.79 秒），但相对 High 增益约 +2.267 dB、SNR 仅 8.24 dB、最大样本误差约 0.761。
- **根因**：Low 精度改变了波形和响度；8 线程有调度抖动。
- **解决**：HiFT CPU 固定 High/6 线程；Low 精度禁止用于产品。

### 3. App 代码 bug：`threads=6` 只传给了 F0，CPU HiFT core 误用 GPU mode 当线程数

- **现象**：改完线程配置后 HiFT core 没有任何收益。
- **根因**：`CosyVoiceNative.kt` 里 CPU HiFT core 把 `hiftGpuMode=4`（GPU mode 数字）当成 MNN CPU 线程数传入。
- **解决**：CPU core 与 F0 都传 6；只有 OpenCL core 才传 4/68/132 GPU mode。
- **结果**：修正后热态 RTF 从 ~1.0 降到 0.89-0.90。

### 4. Flow 动态形状：每句不同长度触发 12-17 秒 GPU kernel 编译

- **现象**：首次未见长度编译 12-17 秒。
- **解决**：固定序列桶。原桶表 `512/768/1024/1280/1536/2048`，后扩展 `256/384`。
- **结果**：sequence 254 用 256 桶约 353 ms（512 桶 727 ms，快约 51%）；sequence 294 用 384 桶约 527 ms（快约 27%）；目标区域与精确长度逐元素一致（`max_abs=0`），是减少无效 padding，不是降精度。

### 5. Flow OpenCL 内存模式：快和省内存不可兼得

- **现象**：SM8850 上 Buffer(68) 热态约 0.786 秒但 GPU PSS 约 1,267.8 MB；自动(4) 约 2.39-2.52 秒但 GPU PSS 仅约 7.7 MB；Image(132) 约 3.75 秒。
- **解决**：SM8850 保留 mode 68（否则全 MNN 持续朗读没有实时余量），其他厂商按运行时探测结果选择；GPU 失败自动回退 CPU。
- **注意**：OpenCL 3.0 只把 1.2 作为强制基线，2.x/3.0 其余是可选能力，不能仅凭版本字符串复用高通的 mode 68。

### 6. 内存峰值：全 MNN 常驻 2.25 GB

- **现象**：加载后 App PSS 约 2.25 GB，`/dev/kgsl-3d0` 约 1,268.6 MB，scudo secondary 约 607.3 MB。
- **解决**：整批预合成结束后自动 `CosyVoiceRuntime.close()` 释放 LLM/Flow/HiFT Session，已生成 WAV 继续播放；队列代次保护阻止旧章节收尾误释放新章节正在复用的运行时；禁止逐句释放。退出管理页必须在未被 UI scope 取消的独立 IO scope 中关闭（否则先取消 scope 导致释放任务永远不执行，真机 PSS 从约 946.7 MB 降到 268.9 MB）。

### 7. 硬件能力显示误导

- **现象**：界面称"NPU 已启用"，但 `qnnBackendBundled=false`、实际合成仍是 CPU。
- **根因**：APK 没有 QNN/HTP backend 或配套库，手机有 NPU ≠ App 有可用执行路径；MNN QNN 要求按高通 SoC/Hexagon 版本配套 QNN stub/skel 库。
- **解决**：NPU 只能作为独立 PoC 验证，不能显示为已启用；OpenCL 3.0 只用于 GPU 路线，不能激活 NPU。

### 8. 本地音色分配不到

- **现象**：默认"其他音色"档案始终分配不到角色。
- **根因**：主角/重要角色匹配性别/年龄标签时排除了未分类音色，但未分类音色应作为兜底候选。
- **解决**：没有明确标签匹配时可选择未分类本地零样本音色，但排除明确相反性别和不兼容年龄标签；本地音色目录 App 启动时同步，不再依赖先打开管理页。

---

## 二、MNN/Hexagon QNN NPU 实验阶段（WSL Ubuntu + MNN 3.6.1 + Hexagon V81）

### 9. MNN QNN backend 三维卷积布局错误（qnn-layout-fix.patch）

- **现象**：1D 卷积/时间序列张量（MNN 3 维）送 QNN 后 shape 错误、输出错乱。
- **根因**：MNN 的 3 维 NCHW 张量是 `{n, c, h}`（1D conv，宽隐含），QNN 的 Conv2d 需要 4 维 NHWC `{n, h, w=1, c}`；TENSORFLOW 视图的临时张量也缺宽度轴。
- **解决**：修改 `source/backend/qnn/backend/QNNBackend.cpp` 与 `QNNUtils.cpp`：
  - NC4HW4/3 维：插入 `w=1`；
  - NCHW/3 维：交换 c/h 并插入 `w=1`；
  - `getNHWCShape()` 同样处理 3 维 NCHW 常量。
- **补丁**：`qnn-layout-fix.patch`。

### 10. MNN CPU 线程池忙等/丢失任务（threadpool-fix.patch）

- **现象**：多任务场景线程池 worker 可能丢任务或忙等，耗时不稳定。
- **根因**：worker 循环只检查 `mActiveCount`，任务栅栏已在等待、active 计数已清零时不会重新扫描任务位；存在 pending 任务时忙等不做正确让步。
- **解决**：worker 循环改为扫描全部任务位 `mTasks[i].second[threadIndex]`，发现 pending 继续执行；无 pending 再进条件等待。
- **补丁**：`threadpool-fix.patch`。

### 11. LLM q_proj 单算子放 NPU：成功但收益很小

- **现象**：仅 `layers.0/self_attn/q_proj/Linear` 放 Hexagon NPU，LLM wall time P50 从 1598.23 ms 降到 1503.35 ms（-5.94%），decode TPS 从 95.57 到 101.23（+5.93%）；NPU 推理 3/3 成功，PCM 无异常。
- **根因**：LLM 其余部分（Attention、KV Cache、lm_head）仍跑 CPU；NPU 只分到一小块算子，收益被 CPU 瓶颈掩盖。
- **解决/边界**：验证必须逐算子确认输出 Token 合法；出现过 Token 崩溃/非法 Token 的算子在通过正确性验证前不能启用；NPU 只作验证，默认禁用，所有设备仍走可靠的 CPU/OpenCL 路线。
- **坑**：MNN Hexagon 后端需要逐 SoC 配套 stub/skel 库；不同 SoC 不能共用同一套 NPU 包。
- **文件**：`mnn-patches/mnn-3.6.1-hexagon-stage-filter.patch`（算子分阶段过滤）、`mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp`（NPU A/B 基准入口，支持 `hexagon` backend 选择）、`docs/NPU_RELEASE_VALIDATION.md`（完整验证记录）。

### 12. MNN 外部权重缺失时"全零成功"

- **现象**：缺失 `.weight` 外部文件时 MNN 打印 `Can't open file` 后仍返回全零"成功"输出。
- **解决**：所有基准必须同时检查 finite、RMS 和 ONNX 数值误差，不能只看"返回成功"。

---

## 三、尚未公开的路线

QAIRT QNN 全图 HTP 迁移（HiFT 上 NPU）等后续实验仍在进行中、未完成，暂不在本仓库公开细节。完成并达到验收门槛后另行补充。

---

> 相关文档：`docs/DEVELOPMENT_STORY.md`（07-21 前全流程）、`docs/RESEARCH_MEMORY.md`（07-21 前逐日记忆）、`docs/NPU_RELEASE_VALIDATION.md`（NPU A/B 验证）。
