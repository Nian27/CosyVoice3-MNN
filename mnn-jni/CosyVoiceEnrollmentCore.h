#pragma once

// CosyVoiceEnrollmentCore —— VoiceProfile 注册的唯一算法实现。
//
// 设计意图（ADR-053 后续，2026-09-16）：
//   「参考音频克隆」与「文字设计音色」是两种不同的 VoiceIdentity Source，
//   但从 enrollment 往下必须走完全同一条链。此前算法只存在于
//   CosyVoiceEnrollmentJni.cpp 内部、且只接受 WAV 路径，
//   导致 VoiceDesign 必须先把 PCM 落成 WAV 再让 enrollment 读回来。
//
//   现在把算法抽到这里，提供两个入口：
//     enrollFromWavFile()  —— 旧行为，保持逐位不变（参考音频克隆路径用）
//     enrollFromPcm()      —— 新入口，直接吃内存里的 float PCM（VoiceDesign 路径用）
//
// 数值纪律（重要）：
//   VoiceDesign 路径**不**直接把 float PCM 交给 enrollFromPcm，
//   而是先经 quantizeInt16InPlace() 做一次「int16 编码 -> 解码」的内存内 round-trip，
//   使 enrollment 看到的数值与「writeWav16 落盘 -> MNN::AUDIO::load 读回」完全一致。
//   理由：已有 enrollment golden 绑定的是那条路径；先保 golden，等整链稳定后
//   再单独评估「直接用 float PCM 是否更好」。
//   两侧的确切语义：
//     编码  float v -> clamp[-1,1] -> (int16_t)lrintf(v * 32767.0f)
//     解码  int16_t q -> q / 32768.0f
//   （32767 / 32768 的不对称是既有实现的一部分，不得"修正"。）

#include <cstddef>
#include <string>
#include <vector>

namespace cosy {

struct EnrollmentModels {
    std::string speechTokenizer;   // speech-tokenizer-v3.fp32.inline.mnn
    std::string campPlus;          // campplus.fp32.mnn
    std::string affineWeight;      // flow-speaker-affine-weight.bin
    std::string affineBias;        // flow-speaker-affine-bias.bin
};

struct EnrollResult {
    int code = -1;                 // 0 = 成功；非 0 见 CosyVoiceEnrollmentJni 的错误码表
    int promptTokens = 0;
    int promptFrames = 0;
    int inputSampleRate = 0;
    double inputSeconds = 0.0;
    int whisperFrames = 0;
    int campFrames = 0;
    double tokenizerMs = 0.0;
    double campMs = 0.0;
    double totalMs = 0.0;
};

// 内存内做「int16 编码 -> 解码」round-trip，语义见文件头注释。
void quantizeInt16InPlace(std::vector<float>& pcm);

// 在「int16 域」（float 表示，量化步长 1/32768）上叠加 ±1 LSB 的确定性 dither。
//
// 为什么需要：合成音频（含 VoiceDesign 产物）的静音是「精确 0 样本」，而 enrollment 里
// CAMPPlus 的前置 fbank 会取 log(能量) -> log(0) = -inf -> NaN 传播 -> 返回 37。
// 真实录音有本底噪声不会踩到，但用户完全可能把合成音频当参考导入克隆。
//
// 与 Kotlin 侧 ditherInPlace(WAV 的 int16 字节 ±1) 数学等价；放在这里是为了让
// VoiceDesign 路径在「不落 WAV」的前提下也能得到同样的输入。
// 注意：enrollFromPcm / enrollFromWavFile **不会**自动调用它 —— 保持 Core 是纯算法，
// 由调用方决定（这样 Gate E1 的「新旧库逐位一致」才可判定）。
void ditherInt16InPlace(std::vector<float>& pcm);

// 从已在内存的 float PCM 产出三个注册产物：
//   prompt-speech-tokens.csv / prompt-cond.bin / spks.bin
// 约束与旧入口一致：时长须落在 3~15 秒（按 sampleRate 折算）。
EnrollResult enrollFromPcm(const float* pcm, std::size_t sampleCount, int sampleRate,
                           const EnrollmentModels& models,
                           const std::string& outputDirectory, int threads);

// 旧入口：读 WAV 后走同一条链。
EnrollResult enrollFromWavFile(const std::string& wavPath,
                               const EnrollmentModels& models,
                               const std::string& outputDirectory, int threads);

}  // namespace cosy
