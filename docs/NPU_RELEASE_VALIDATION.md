# v1.1.0 NPU 正式版验证记录

## 结论

本版本只在已验证的 Qualcomm SM8850 上自动启用 Hexagon，并且只把
`/layers.0/self_attn/q_proj/Linear` 放到 NPU。LLM 的 Attention、KV Cache、
`lm_head` 和其余算子继续走 CPU；Flow 使用 OpenCL，HiFT 使用 CPU。

这不是“整模型 NPU”。整层、全部卷积和连续 Hexagon 解码已经在真机上出现
Token 坍缩与错误声音，因此不得作为正式默认策略。

## 已验证效果

同一设备、同一输入进行 CPU/NPU A/B 三轮：

| 指标 | CPU P50 | q_proj NPU P50 | 变化 |
| --- | ---: | ---: | ---: |
| LLM wall time | 1598.23 ms | 1503.35 ms | -5.94% |
| Decode throughput | 95.57 token/s | 101.23 token/s | +5.93% |

端到端验证总耗时 11.32 秒，NPU 方案 3/3 次命中；输出 PCM 有限、非静音，
Token 合法且音色链路正常。

## 正确性保护

- App 启动时检查 SoC 和 Hexagon Stub/Skeleton 是否可用。
- 未验证 SoC 即使运行库能加载，也不会自动启用 NPU。
- 每次 NPU LLM 后检查运行报告、非法输出、Token 种类和连续重复。
- 任一检查失败，废弃整句 NPU 结果并自动用 CPU 重跑。

## 原生构建

- MNN：3.6.1
- Android NDK：r25c
- Hexagon DSP：V81
- MNN 源码需先应用
  `mnn-patches/mnn-3.6.1-hexagon-stage-filter.patch`。
- `mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp` 读取三个分段文件：
  `hexagon-stage-layers.txt`、`hexagon-stage-ops.txt`、
  `hexagon-stage-name.txt`。

## 发布边界

当前量化结论仅适用于 Honor BKQ-AN90 / SM8850 实测设备。其他 Snapdragon、
MediaTek、Kirin 或 Exynos 设备会使用 CPU/OpenCL 兼容路径，不能宣传为已验证
NPU 加速。
