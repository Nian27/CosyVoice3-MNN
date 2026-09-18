# 发布/上传清单（2026-09-18）

> 这份文件回答一个问题：**哪些东西能进 Git 仓库，哪些不能，各自走哪条渠道。**
> 结论先行：源码与 native `.so` 进 Git；**模型权重一律不进 Git**（GitHub 单文件硬上限 100 MB，
> 而本项目的权重单个就有 2.63 GiB），走 GitHub Releases（单附件 < 2 GB）或 Hugging Face。

## 1. 进 Git 仓库（已全部 `<100 MB`，可安全提交）

本次提交共 51 个文件，最大的一个是 259,680 B 的 `libcosy_voicedesign_jni.so`。

| 类别 | 内容 |
|---|---|
| App 源码 | `app/src/main/java/com/fun/cosyvoice/*.kt`、`app/src/main/java/io/legado/app/cosy/CosyVoiceJniBridge.kt` |
| 单测 | `app/src/test/java/com/cosyvoice/app/CosyVoiceLlmAttemptPlanTest.kt` + `app/src/test/resources/llm-tokens/*`（6 个真机 token 向量） |
| Native `.so` | `app/src/main/jniLibs/arm64-v8a/*.so`（最大 `libMNN.so` 48,769,280 B） |
| JNI 源码 | `mnn-jni/*.cpp` / `*.h` / `CMakeLists.txt` |
| MNN fork 补丁 | `mnn-patches/mnn-cosyvoice-npu-fork/`（README + `llm-engine.patch` + `backend-hexagon-qnn.patch`） |
| 文档 | `README.md`、`MEMORY.md`、`WORKSPACE.md`、`docs/*.md` |
| 基线二进制 | `research/mnn-cosyvoice3/baseline-so/libcosy_enrollment_jni.LEGACY.so`（44,040 B） |

`.gitignore` 已经挡住 `models/`、`*.apk`、`build/`、`signing.properties`，提交前扫过一遍：
仓库内**没有** API key / token / 口令（`signing.properties` 不在工作树里）。

## 2. 不进 Git：模型权重

| 包 | 字节 | 单文件最大 | 能进 Git？ | 能进 GitHub Release？ |
|---|---:|---:|---|---|
| `cosyvoice3-mnn-mobile-fp16-complete.zip` | 1,399,083,563 (1.30 GiB) | 663,253,312 | ✗ | ✓（< 2 GB） |
| `cosyvoice3-mnn-enrollment-extension.zip` | 997,807,778 (951 MiB) | — | ✗ | ✓ |
| **VoiceDesign 包（新增，尚未打包）** | **4,812,979,079 (4.48 GiB)** | **2,821,857,280 (2.63 GiB)** | ✗ | **✗（2.63 GiB > 2 GB 上限）** |

VoiceDesign 包 = 设备上 `files/cosyvoice3-mnn/voicedesign/` 里那 25 个模型文件（不含 `*.log` /
`*-preview.wav` / `*-tokens.csv` / `last-*` 这些探针产物）。清单见文末。

### 渠道 A：GitHub Releases（两个已有 zip）

```bash
# 在**有网**的机器上执行；本机（DSH 环境）无外网，curl github.com 超时。
gh release create v1.2.0 \
  --title "CosyVoice3-MNN v1.2.0" \
  --notes-file RELEASE_NOTES.md \
  models/cosyvoice3-mnn-mobile-fp16-complete.zip \
  models/cosyvoice3-mnn-enrollment-extension.zip
```

### 渠道 B：Hugging Face（VoiceDesign 包，唯一可行）

沿用已有仓库 `VicenTrent/Cosy-Voice-MNN`（`release-manifest.json` 里两个 zip 的下载地址就是它）：

```bash
# 先打包（在设备上或从设备拉下来的 voicedesign 模型目录里）
zip -0 cosyvoice3-mnn-voicedesign-complete.zip <25 个模型文件>
sha256sum cosyvoice3-mnn-voicedesign-complete.zip

# 再上传（需要 HF token；本机无外网）
huggingface-cli upload VicenTrent/Cosy-Voice-MNN \
  cosyvoice3-mnn-voicedesign-complete.zip cosyvoice3-mnn-voicedesign-complete.zip
```

上传后要把新包补进 `release-manifest.json` 的 `models[]`（name / bytes / sha256 / url / required）。

## 3. 还没做、需要你决定的事

1. **推到哪个仓库**：本仓库现在的 `origin` 是上游 `https://github.com/Nian27/CosyVoice3-MNN.git`，
   你没有它的写权限。需要你给出目标仓库 URL（fork 或新建），并确认外部环境能 `git push`。
2. **提交作者身份**：`git config user.name/user.email` 原本是空的，本次提交用了一个占位身份。
   要改：`git commit --amend --reset-author`（或先 `git config user.name/user.email`）。
3. **VoiceDesign 包要打进发布物吗**：它 4.48 GiB，且必须走 HF（见上）。

## 4. VoiceDesign 包文件清单（25 个，合计 4,812,979,079 B）

```text
codec_emb_fp16.mnn                     12,583,868
codec_emb_weight.f32                  251,658,240
codec_head_fp16.mnn                         2,360
codec_head_fp16.mnn.weight             12,595,200
codepred_prefill_fp16.mnn                 123,768
codepred_prefill_fp16.mnn.weight      161,830,912
codepred_rope_cos.f32                       8,192
codepred_rope_sin.f32                       8,192
codepred_step_fp16.mnn                     94,648
codepred_step_fp16.mnn.weight         161,830,912
frame_emb_fp16.mnn                    138,430,844
graphb28_v6_fp16.mnn                      503,448
graphb28_v6_fp16.mnn.weight         2,821,857,280   ← 超过 GitHub Release 的 2 GB 单附件上限
lm_head_weight.f32                    125,829,120
prompt_emb.f32                            507,904
talker_codec_emb.f32                   25,165,824
talker_rope_cos.f32                       196,608
talker_rope_sin.f32                       196,608
text_embedding.fp16                   622,329,856
text_proj_fc1.fp16                      8,388,608
text_proj_fc1_bias.f32                      8,192
text_proj_fc2.fp16                      8,388,608
text_proj_fc2_bias.f32                      8,192
tokenizer.bin                           3,490,103
tokenizer_decoder_static_t96.mnn      456,941,592
```

## 5. 本机（DSH 环境）实测：外网不可达

```text
$ curl -sS -m 8 -o /dev/null -w "%{http_code}" https://github.com
curl: (28) Operation timed out after 8001 milliseconds with 0 bytes received
```

也就是 `git push` / `gh release create` / `huggingface-cli upload` 都必须换一台有网的机器执行。
本文档的作用就是把那一步压缩成「复制粘贴一条命令」。
