# CosyVoice3-MNN v1.1.0

这是首个带正式签名和受限 NPU 加速的独立版本。

## 主要更新

- SM8850 自动启用 LLM 第 0 层 `q_proj` Hexagon NPU，其余 LLM 算子继续使用 CPU。
- NPU 运行报告或 Token 质量异常时，废弃当前结果并自动整句回退 CPU。
- Flow 自动选择 OpenCL/CPU，HiFT 默认使用经过验证的 CPU 路径。
- App 不再要求用户手动选择 CPU、GPU、NPU。
- 新增 Hugging Face 在线模型安装，支持断点续传、完整 ZIP SHA-256 和 17 个模型文件二次校验。
- 原生运行库升级为 MNN 3.6.1 动态库，并正确打包 Hexagon Stub 与 DSP Skeleton。
- APK 使用 VicenTrent 正式证书签名。

## 下载

- APK：`CosyVoice3-MNN-v1.1.0-arm64.apk`
- 完整模型：
  [VicenTrent/Cosy-Voice-MNN](https://huggingface.co/VicenTrent/Cosy-Voice-MNN)
- 音色创建扩展：同一 Hugging Face 模型仓库，按需下载。

## 校验

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `CosyVoice3-MNN-v1.1.0-arm64.apk` | 27,795,217 | `A25D6E0822F000724E6E8599B6D28EC791ECCCEE9162CBF76FF7EF2E088EC996` |
| `cosyvoice3-mnn-mobile-fp16-complete.zip` | 1,399,083,563 | `B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84` |
| `cosyvoice3-mnn-enrollment-extension.zip` | 997,807,778 | `59EF5C8810D3CEAD01FF21A64379CF0A819F0A72651A6BA2B2343E9DA5A72231` |

## NPU 性能与边界

Honor BKQ-AN90 / SM8850 同输入 A/B 三轮中，LLM wall time P50 从
1598.23 ms 降至 1503.35 ms，约加速 5.94%；decode throughput 从
95.57 提升至 101.23 token/s，约提升 5.93%。

整模型、整层或连续 Hexagon 解码会导致 Token 坍缩和错误声音，因此本版本
不会启用这些路径。其他 SoC 即使能加载 Hexagon 库，也继续使用 CPU/OpenCL，
直到完成独立的正确性和性能验证。

## v1.0.0 模型附件纠错

v1.0.0 Release 中 1.3 GB 模型附件误命名为
`cosyvoice3-mnn-3.6.1.zip`。GitHub 记录的 SHA-256 与 Hugging Face 的
`cosyvoice3-mnn-mobile-fp16-complete.zip` 完全一致：
`B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84`。

因此纠错操作是把 v1.0.0 附件恢复为正式文件名，并在旧版说明中标注版本关系；
不是用一个未经校验的新 ZIP 覆盖相同内容。
