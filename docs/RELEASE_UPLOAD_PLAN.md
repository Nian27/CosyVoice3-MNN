# 发布/上传清单（2026-09-18，已更新）

> 回答一个问题：**哪些东西能进 Git 仓库，哪些不能，各自走哪条渠道。**
> 结论：源码与 native `.so` 进 Git（**已推送**）；模型权重不进 Git（GitHub 单文件硬上限 100 MB，
> 本项目单个权重最大 2.63 GiB），走 GitHub Releases 或 Hugging Face。

## 0. 已完成

```text
代理          : http://127.0.0.1:7897（本机可直连 GitHub；未设代理时 curl 会 8 秒超时）
代码推送      : 5bd13bc..43fb40d  main -> main   → https://github.com/Nian27/CosyVoice3-MNN
远端 main     : 43fb40d7c952ffa219ab1365a98c3277969be399（已核实）
本次提交      : 52 个文件 / +6,490 行，最大单文件 259,680 B
提交作者      : 占位 VicenTrent <VicenTrent@users.noreply.github.com>
                （原 git config user.name/email 为空）要改：git commit --amend --reset-author
```

## 1. 进 Git 仓库（全部 < 100 MB）

| 类别 | 内容 |
|---|---|
| App 源码 | `app/src/main/java/com/fun/cosyvoice/*.kt`、`app/src/main/java/io/legado/app/cosy/CosyVoiceJniBridge.kt` |
| 单测 | `app/src/test/java/com/cosyvoice/app/CosyVoiceLlmAttemptPlanTest.kt` + `app/src/test/resources/llm-tokens/*`（6 个真机 token 向量） |
| Native `.so` | `app/src/main/jniLibs/arm64-v8a/*.so`（最大 `libMNN.so` 48,769,280 B） |
| JNI 源码 | `mnn-jni/*.cpp` / `*.h` / `CMakeLists.txt` |
| MNN fork 补丁 | `mnn-patches/mnn-cosyvoice-npu-fork/`（README + `llm-engine.patch` + `backend-hexagon-qnn.patch`） |
| 文档 | `README.md`、`MEMORY.md`、`WORKSPACE.md`、`docs/*.md` |

`.gitignore` 挡住 `models/`、`*.apk`、`build/`、`signing.properties`；提交前扫过一遍，仓库内没有
API key / token / 口令（`signing.properties` 不在工作树里）。

## 2. 模型权重：不进 Git

### 2.1 已有两个包 —— 已经在 GitHub Releases 上，不需要重复上传

Release **v1.0.0** 的附件（`curl https://api.github.com/repos/Nian27/CosyVoice3-MNN/releases/tags/v1.0.0` 核实）：

```text
cosyvoice3-mnn-mobile-fp16-complete.zip     1,399,083,563 B
cosyvoice3-mnn-enrollment-extension.zip       997,807,778 B
app-debug.apk                                  25,618,788 B
```

### 2.2 VoiceDesign 包（新增）—— 暂时**不能**上传，两个前置问题

| 项 | 值 |
|---|---|
| 内容 | 设备 `files/cosyvoice3-mnn/voicedesign/` 里 25 个模型文件（不含 `*.log` / `*-preview.wav` / `*-tokens.csv` / `last-*`） |
| 合计 | 4,812,979,079 B（4.48 GiB） |
| 单个最大 | `graphb28_v6_fp16.mnn.weight` = 2,821,857,280 B（2.63 GiB） |

1. **超出 GitHub Release 的单附件 2 GB 上限** → 只能拆成 <2 GB 的分片，或走 Hugging Face（推荐）。
2. **许可证未确认**。仓库内 `research/mnn-cosyvoice3/deploy-voicedesign-models.sh` 显示这些权重来自
   `/data/local/tmp/qwen3tts/vdfull` 与 `/data/local/tmp/vdtok`（**qwen3tts** 目录），出处不明。
   按项目纪律「LICENSE_UNKNOWN 的第三方资产不进入公开仓库」，**确认来源与许可证之前不公开上传**。

（另注：该脚本里写的是 `tokenizer_decoder_static_t300.mnn`，设备上实际是
`tokenizer_decoder_static_t96.mnn` —— 两边清单已经不一致，重新打包时要先对齐。）

## 3. 上游 CosyVoice 版本核对（2026-09-18，已联网核实）

```text
GitHub FunAudioLLM/CosyVoice
  refs/heads/main = 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
  tags            = v2.0 / flow_cache（没有 v3 tag）

HuggingFace FunAudioLLM/Fun-CosyVoice3-0.5B-2512
  sha          = 29e01c4e8d000f4bcd70751be16fa94bf3d85a18
  lastModified = 2026-02-03
本项目固定   = 29e01c4e8d000f4bcd70751be16fa94bf3d85a18（stage3-manifest）
               + 7067950070211f149d9bf25642cc961cad06fa02（stage2-manifest）
```

**结论：模型侧已经是最新 revision**（HF 当前的 sha 与仓库固定值一致）。
代码侧本项目没有 vendor 上游 Python 实现（是自己 PyTorch → ONNX → MNN 移植的），
所以"更新 CosyVoice 代码"对当前仓库没有作用对象。
