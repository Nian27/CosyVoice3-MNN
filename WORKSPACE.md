# CosyVoice3-MNN-Plus 工作区说明

ReaderVoice 系列的最新 CosyVoice3-MNN 整合工作区（2026-08-25 建立）。本文件不入 Git 上游，仅本工作区使用。

## 内容

- `app/` `mnn-jni/` `docs/` `research/`：上游 [Nian27/CosyVoice3-MNN](https://github.com/Nian27/CosyVoice3-MNN) 最新代码
  （HEAD = `5bd13bc`，含 2026-08-25 HiFT FP16 增补：`corePrecision="low"`，HiFT core 2766→863 ms，RTF 0.82）。
- `models/`：模型资产（来自 HF `VicenTrent/Cosy-Voice-MNN`，未入 Git）：
  - `cosyvoice3-mnn-mobile-fp16-complete.zip`（1334 MB，17 文件合成模型包）
  - `cosyvoice3-mnn-enrollment-extension.zip`（952 MB，0 样本音色创建扩展：speech-tokenizer-v3 + campplus + flow-speaker-affine）
  - `model.sha256` / `enrollment.sha256`（整包校验）

## 下一步主线：Qwen3-TTS VoiceDesign → CosyVoice 音色

目标：用 Qwen3-TTS-12Hz-1.7B-VoiceDesign（或 0.6B-CustomVoice）按自然语言描述生成全新音色，让 CosyVoice 使用该音色合成任意文本。

路径（两段都已就绪，缺的是衔接）：

1. **设计**：Qwen3-TTS VoiceDesign 按描述生成设计音色的参考音频 WAV
   （工程在 `E:\AndroidStudioProjects\qwen3tts-mnn`，ONNX→MNN 已到 M3，M4 待真机）。
2. **建档**：把该 WAV 送入本工程已发货的 0 样本音色创建扩展
   （`CosyVoiceEnrollmentJni.cpp`：speech-tokenizer-v3 → speech tokens + prompt mel；campplus → speaker embedding；affine → spks），
   产出 CosyVoice 音色档案（`voices/<id>/`：prompt-speech-tokens.csv + prompt-cond.bin + spks.bin + rand-noise 软链）。
3. **合成**：CosyVoice3-MNN 用该档案零样本复刻设计音色。

衔接要点（RESEARCH_MEMORY 既有结论）：参考音频取最稳定 3-5 秒、≤125 speech tokens/250 mel 帧；
设计音频的文字必须与 Qwen3-TTS 输入完全一致（作为 transcript）。

## 相关路径

- 上游仓库：https://github.com/Nian27/CosyVoice3-MNN
- HF 模型仓库：https://huggingface.co/VicenTrent/Cosy-Voice-MNN
- 音色设计工程：`E:\AndroidStudioProjects\qwen3tts-mnn`
- ReaderVoice 集成侧：`E:\AndroidStudioProjects\ReaderVoiceMobile\app-android\src\main\java\io\legado\app\cosy\`
