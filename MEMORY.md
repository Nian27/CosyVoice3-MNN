# CosyVoice3-MNN 项目长期记忆

## 项目位置

- 正式发布工作树：`E:\AndroidStudioProjects\CosyVoice3-MNN-formal`
- 上游仓库：`https://github.com/Nian27/CosyVoice3-MNN`
- 默认分支：`main`
- 模型权重不提交到 Git 仓库，通过 Release 附件独立分发。

## 已验证运行边界

- Qualcomm SM8850 上，LLM 仅把 `layers.0/self_attn/q_proj/Linear` 放到 Hexagon NPU，其余算子留在 CPU，能保持有效 Token 和音频。
- 连续 Hexagon 解码、整层 Hexagon 和全部卷积 Hexagon 已出现 Token 坍缩或音频错误，不得作为正式默认方案。
- q_proj 单算子 A/B 三轮验证中，LLM wall time P50 从 1598.23 ms 降到 1503.35 ms，约提升 5.94%；decode TPS 从 95.57 提升到 101.23，约提升 5.93%。
- NPU 只对已验证机型自动启用；其他设备继续使用可靠的 CPU/OpenCL 路径。

## 构建环境

- MNN/Hexagon 原生构建位于 WSL Ubuntu，使用 MNN 3.6.1、Android NDK r25c、Hexagon V81。
- 大型构建缓存和模型不得写入 C 盘。

## 发布待办

- v1.0.0 Release 的 1.3 GB 模型附件误命名为 `cosyvoice3-mnn-3.6.1.zip`。
  GitHub 附件与 Hugging Face 正式完整包的 SHA-256 均为
  `B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84`，
  内容一致，应纠正文件名和说明，不重复上传未经验证的 ZIP。
- v1.1.0：versionCode 2；正式 APK 名称
  `CosyVoice3-MNN-v1.1.0-arm64.apk`，大小 27,795,217 字节，SHA-256
  `A25D6E0822F000724E6E8599B6D28EC791ECCCEE9162CBF76FF7EF2E088EC996`。
- 正式 APK 使用 `E:\AndroidStudioProjects\legado-signing\vicentrent-release.jks`
  签名；密码只保存在被 Git 忽略的 `signing.properties`，不得写入仓库。
- Hugging Face 模型仓库：`VicenTrent/Cosy-Voice-MNN`。v1.1.0 App 在线下载
  完整 ZIP，支持断点续传、整包 SHA-256 和内部 17 文件二次校验；安装成功后
  删除临时 ZIP。
- 构建缓存固定在 `E:\AndroidBuildCache\CosyVoice3-MNN`，避免占用 C 盘。

## 实验记录文档

后续开发前先读 `docs/` 下的记录，避免重复踩坑：

- `docs/DEVELOPMENT_STORY.md`：开发全流程 + 尝试路线总览（CrispASR/Vulkan 失败 → MNN/OpenCL → 蒸馏 → NPU 实验）
- `docs/PITFALLS_AND_FIXES.md`：全历程踩坑 12 大类总结（现象 → 根因 → 解决 → 汇总表 → 剩余 3 坑 → 三大教训）
- `docs/RESEARCH_MEMORY.md`：逐日研究记忆（2026-07-14 ~ 08-06）
- `docs/ACCELERATOR_ADAPTATION_PLAN_v1.0.md`：NPU/GPU 定向改造计划（冻结基线 `baseline-20260806`，含"已证明不能重复的路线"红线）
- `docs/NPU_RELEASE_VALIDATION.md`：v1.1.0 NPU（q_proj 单算子）验证记录

当前未完成实验（QAIRT QNN 全图 HiFT：A16W8 per-channel 24.77ms/12帧 但 PCM corr 0.835 未达标）不进入正式版，状态见 PITFALLS_AND_FIXES.md 第 8 类。
