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
- 【2026-09-18 更正】上条「整层/全部卷积 Hexagon 会 Token 坍缩」的**判据不可靠**：当时的
  报告字段 `continuousHexagon` / `npuFirstTokenValid` 只是配置回显（不是运行期测量），而质量门
  用的 `longestRun >= 8` 会把正常终止的输出也判成坍缩。同一句 63 字（`inputTokens=196`）
  在 `layers.0 q_proj` 这条线上实测 **4/4 次首试通过**：prefill 0.62~0.73 s、decode 9.1~11.7 s、
  37.6~38.8 tok/s、wall 10.3~13.1 s 且都正常出 EOS；同句 OpenCL 6 次里 3 次撞满 `max_new_tokens`
  无 EOS。因此「长文本不能用 NPU」不成立，App 侧的按长度分流已取消。
  详见 ReaderVoiceMobile `docs/DECISIONS.md` 的 ADR-053 增补 NPU-R8 / R9 / R10；
  构建机上的 MNN fork 与 JNI 源码已固化到本仓库 `mnn-patches/mnn-cosyvoice-npu-fork/`。

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
- `docs/VOICEDESIGN_DEVICE_MEMORY_2026-09-16.md`：**VoiceDesign 真机内存与 Decoder 配置实测结论**（5 配置对比 + 分阶段释放/FP16/T 缩减/OpenCL 缓存/权重 mmap 缓存 5 条失败路线的实测数据）

### VoiceDesign 音色创建（2026-09-16 冻结基线）

- **正式配置：Talker 与 Tokenizer Decoder 均外置权重** → anon 0.61 GB、峰值 HWM 0.90 GB，已完整跑通 2 次
- Decoder 外置的代价：约 459 s（内嵌只要 20 s），但内嵌 HWM 1.86 GB 会触发 Honor HyperHold 换出
- **优先级：稳定运行 > Decoder 速度**。本设备 HWM 超过约 1.2 GB 后系统内存压力显著增加
- 已实测失败、**不要重复尝试**的路线：① 同 RuntimeManager 内分阶段 Module 释放（anon 只降 56 MB）② FP16 Decoder（PCM 最大差 2693 LSB）③ T 300→96（内存磁盘都无收益）④ OpenCL kernel 缓存 setCache（卡死在 decoder 后）⑤ USE_CACHED_MMAP 权重 mmap 缓存（写到 1 GB 卡死）
- 后续优化 Decoder 必须**单独开 Decoder-HTP / 独立 Runtime 分支**，不再扰动该基线

当前未完成实验（QAIRT QNN 全图 HiFT：A16W8 per-channel 24.77ms/12帧 但 PCM corr 0.835 未达标）不进入正式版，状态见 PITFALLS_AND_FIXES.md 第 8 类。
