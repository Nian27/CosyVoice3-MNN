---
license: apache-2.0
library_name: mnn
tags:
- text-to-speech
- cosyvoice
- android
- mnn
- on-device
- chinese
---

# CosyVoice3 MNN Android 模型包

供 [Nian27/CosyVoice3-MNN](https://github.com/Nian27/CosyVoice3-MNN)
Android arm64 App 使用。模型全部在手机本地推理；联网只用于首次下载模型。

## 文件与校验

| 文件 | 字节 | SHA-256 | 用途 |
| --- | ---: | --- | --- |
| [cosyvoice3-mnn-mobile-fp16-complete.zip](./cosyvoice3-mnn-mobile-fp16-complete.zip) | 1,399,083,563 | `B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84` | 必需，17 个合成模型文件 |
| [cosyvoice3-mnn-enrollment-extension.zip](./cosyvoice3-mnn-enrollment-extension.zip) | 997,807,778 | `59EF5C8810D3CEAD01FF21A64379CF0A819F0A72651A6BA2B2343E9DA5A72231` | 可选，仅手机创建音色需要 |

## 使用

推荐安装 CosyVoice3-MNN v1.1.0 或更新版本：

1. 在 App 的“MNN 模型”卡片点击“在线安装”。
2. App 从本仓库断点续传完整 ZIP。
3. 下载完成后先校验 ZIP SHA-256，再校验内部 17 个文件。
4. 安装成功后自动删除临时 ZIP，避免长期多占约 1.3 GB。

也可以在本页面手动下载完整 ZIP，然后在 App 中点击“导入 ZIP”。

## 版本说明

- 模型包文件名不等于 MNN 运行库版本。
- CosyVoice3-MNN v1.1.0 App 使用 MNN 3.6.1 运行库。
- GitHub v1.0.0 曾把完整模型附件误命名为
  `cosyvoice3-mnn-3.6.1.zip`；其 SHA-256 与本页完整模型包一致，现按正式文件名纠正。

## NPU 边界

SM8850 上只验证了 LLM 第 0 层 `q_proj` 的受限 Hexagon 加速。模型本身并非
“全 NPU 模型”；整模型连续 Hexagon 解码会产生错误 Token 和错误声音。
