#include "CosyVoiceEnrollmentCore.h"

#include <android/log.h>

#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Module.hpp>
#include <audio/audio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace cosy {
namespace {

constexpr const char* kLogTag = "CosyVoiceEnrollment";
constexpr float kPi = 3.14159265358979323846f;
constexpr int kDefaultRealtimePromptTokens = 125;

// 【2026-09-18】实时零样本前缀的长度上限。
//
// 原来这里是编译期常量 125。但真机实测发现它会决定 NPU 能否工作：
//   prompt 87 token（内置参考音色）→ LLM inputTokens 135 → Hexagon 连续解码启用 ✓
//                                    decodeMs=3306   tokensPerSecond=35.99  （NPU 速度）
//   prompt 125 token（VoiceDesign）→ LLM inputTokens 183 → 连续解码【不启用】✗
//                                    decodeMs=56493  tokensPerSecond=4.09   （CPU 速度）
// 质量门据此整句回退 CPU，LLM 从 ~4 秒变成 ~47 秒。
//
// 为了让阈值可以不用重编就二分定位，改成可被环境变量 VDS_RT_LIMIT 覆盖
// （由 JNI 调用方在调到 native 之前用 Os.setenv 设置）。
int maxRealtimePromptTokens() {
    if (const char* env = getenv("VDS_RT_LIMIT")) {
        const int value = atoi(env);
        if (value >= 20 && value <= 200) return value;
    }
    return kDefaultRealtimePromptTokens;
}

using MNN::Express::NHWC;
using MNN::Express::REFLECT;
using MNN::Express::VARP;
using MNN::Express::_Const;
using MNN::Express::_Log;
using MNN::Express::_Maximum;
using MNN::Express::_Pad;
using MNN::Express::_Scalar;

void logError(const char* stage, int code) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s failed code=%d", stage, code);
}

std::string joinPath(const std::string& directory, const std::string& name) {
    if (!directory.empty() && directory.back() != '/' && directory.back() != '\\') {
        return directory + "/" + name;
    }
    return directory + name;
}

std::string parentDirectory(const std::string& path) {
    const auto position = path.find_last_of("/\\");
    return position == std::string::npos ? "." : path.substr(0, position);
}

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

bool finiteVector(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool readFloats(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) return false;
    input.seekg(0);
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(input);
}

bool writeFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

// 与 CosyVoiceEnrollmentJni.cpp 原实现逐字一致
std::vector<float> sincResample(const float* input, size_t inputCount, int inputRate, int outputRate) {
    if (inputRate == outputRate) return std::vector<float>(input, input + inputCount);
    const double ratio = static_cast<double>(outputRate) / inputRate;
    const size_t outputCount = static_cast<size_t>(std::floor(inputCount * ratio));
    const int radius = 24;
    const double cutoff = std::min(1.0, ratio) * 0.94;
    std::vector<float> output(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
        const double position = static_cast<double>(i) / ratio;
        const int center = static_cast<int>(std::floor(position));
        double sum = 0.0;
        double weightSum = 0.0;
        for (int sample = center - radius + 1; sample <= center + radius; ++sample) {
            if (sample < 0 || sample >= static_cast<int>(inputCount)) continue;
            const double distance = position - sample;
            const double normalized = distance / radius;
            if (std::abs(normalized) >= 1.0) continue;
            const double sincArg = kPi * cutoff * distance;
            const double sinc = std::abs(sincArg) < 1e-9 ? 1.0 : std::sin(sincArg) / sincArg;
            const double window = 0.5 + 0.5 * std::cos(kPi * normalized);
            const double weight = cutoff * sinc * window;
            sum += input[sample] * weight;
            weightSum += weight;
        }
        output[i] = static_cast<float>(weightSum == 0.0 ? 0.0 : sum / weightSum);
    }
    return output;
}

MNN::ScheduleConfig cpuSchedule(int threads, MNN::BackendConfig& backend) {
    backend.precision = MNN::BackendConfig::Precision_High;
    backend.memory = MNN::BackendConfig::Memory_Normal;
    backend.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.backupType = MNN_FORWARD_CPU;
    schedule.numThread = threads;
    schedule.backendConfig = &backend;
    return schedule;
}
int runSpeechTokenizer(const std::string& modelPath, const std::vector<float>& features,
                       int frames, int threads, std::vector<int32_t>& tokens, double& inferenceMs) {
    MNN::BackendConfig backend;
    auto schedule = cpuSchedule(threads, backend);
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backend, threads);
    MNN::Express::ExecutorScope executorScope(executor);
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule));
    if (!runtime) { logError("speech-tokenizer/create-runtime", 20); return 20; }
    runtime->setExternalPath(parentDirectory(modelPath), 3);
    if (std::ifstream(modelPath + ".weight").good()) {
        runtime->setExternalFile(modelPath + ".weight");
    }
    MNN::Express::Module::Config moduleConfig;
    moduleConfig.shapeMutable = true;
    moduleConfig.dynamic = true;
    const std::vector<std::string> inputNames = {"feats", "feats_length"};
    const std::vector<std::string> outputNames = {"indices"};
    std::shared_ptr<MNN::Express::Module> module(
        MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), runtime, &moduleConfig));
    if (!module) { logError("speech-tokenizer/load-module", 21); return 21; }
    const auto moduleInfo = module->getInfo();
    if (!moduleInfo || moduleInfo->inputs.size() != 2) return 22;
    auto featureInput = MNN::Express::_Input(
        {1, 128, frames}, moduleInfo->inputs[0].order, moduleInfo->inputs[0].type);
    auto lengthInput = MNN::Express::_Input(
        {1}, moduleInfo->inputs[1].order, moduleInfo->inputs[1].type);
    const auto featureInfo = featureInput->getInfo();
    if (!featureInfo || featureInfo->size != features.size() ||
        featureInfo->type != halide_type_of<float>()) return 24;
    std::copy(features.begin(), features.end(), featureInput->writeMap<float>());
    const auto lengthInfo = lengthInput->getInfo();
    if (!lengthInfo || lengthInfo->size != 1 || lengthInfo->type != halide_type_of<int32_t>()) return 22;
    lengthInput->writeMap<int32_t>()[0] = frames;
    const auto start = std::chrono::steady_clock::now();
    auto outputs = module->onForward({featureInput, lengthInput});
    if (outputs.size() != 1) return 25;
    const auto outputInfo = outputs[0]->getInfo();
    const int32_t* outputData = outputs[0]->readMap<int32_t>();
    inferenceMs = elapsedMs(start);
    if (!outputInfo || !outputData || outputInfo->type != halide_type_of<int32_t>()) return 26;
    tokens.assign(outputData, outputData + outputInfo->size);
    if (tokens.empty()) return 27;
    const auto minMax = std::minmax_element(tokens.begin(), tokens.end());
    const std::set<int32_t> unique(tokens.begin(), tokens.end());
    if (*minMax.first < 0 || *minMax.second > 6560 || unique.size() < 4) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "speech-tokenizer invalid count=%zu min=%d max=%d unique=%zu",
                            tokens.size(), *minMax.first, *minMax.second, unique.size());
        return 28;
    }
    return 0;
}

int runCampPlus(const std::string& modelPath, const std::vector<float>& features,
                int frames, int threads, std::vector<float>& embedding, double& inferenceMs) {
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!interpreter) { logError("campplus/create-interpreter", 30); return 30; }
    interpreter->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig backend;
    auto schedule = cpuSchedule(threads, backend);
    MNN::Session* session = interpreter->createSession(schedule);
    if (!session) { logError("campplus/create-session", 31); return 31; }
    MNN::Tensor* input = interpreter->getSessionInput(session, "input");
    if (!input) return 32;
    interpreter->resizeTensor(input, {1, frames, 80});
    interpreter->resizeSession(session);
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    if (hostInput.elementSize() != features.size()) return 34;
    std::copy(features.begin(), features.end(), hostInput.host<float>());
    input->copyFromHostTensor(&hostInput);
    const auto start = std::chrono::steady_clock::now();
    if (interpreter->runSession(session) != MNN::NO_ERROR) return 35;
    inferenceMs = elapsedMs(start);
    MNN::Tensor* output = interpreter->getSessionOutput(session, "output");
    if (!output) return 36;
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&hostOutput);
    embedding.assign(hostOutput.host<float>(), hostOutput.host<float>() + hostOutput.elementSize());
    return embedding.size() == 192 && finiteVector(embedding) ? 0 : 37;
}
std::vector<float> extractWhisperFeatures(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    auto features = MNN::AUDIO::whisper_fbank(waveform, 16000, 128, 400, 160, 0);
    auto info = features->getInfo();
    if (!info || info->size == 0) return {};
    frames = info->dim.back();
    const float* data = features->readMap<float>();
    return std::vector<float>(data, data + info->size);
}

std::vector<float> extractCampFeatures(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    auto features = MNN::AUDIO::fbank(waveform, 16000, 80, 400, 160, 0.0f, 0.97f);
    if (features == nullptr || !features->getInfo()) return {};
    auto info = features->getInfo();
    frames = info->dim[0];
    const float* source = features->readMap<float>();
    std::vector<float> result(source, source + info->size);
    for (int bin = 0; bin < 80; ++bin) {
        double sum = 0.0;
        for (int frame = 0; frame < frames; ++frame) sum += result[frame * 80 + bin];
        const float mean = static_cast<float>(sum / frames);
        for (int frame = 0; frame < frames; ++frame) result[frame * 80 + bin] -= mean;
    }
    return result;
}

std::vector<float> extractPromptMel(const std::vector<float>& audio, int& frames) {
    auto waveform = _Const(audio.data(), {static_cast<int>(audio.size())}, NHWC, halide_type_of<float>());
    const int padding[] = {720, 720};
    waveform = _Pad(waveform, _Const(padding, {2}, NHWC, halide_type_of<int>()), REFLECT);
    MNN::AUDIO::MelscaleParams mel;
    mel.n_mels = 80;
    mel.n_fft = 1920;
    mel.sample_rate = 24000;
    mel.htk = false;
    mel.norm = true;
    mel.f_min = 0.0f;
    mel.f_max = 12000.0f;
    MNN::AUDIO::SpectrogramParams spectrum;
    spectrum.n_fft = 1920;
    spectrum.hop_length = 480;
    spectrum.win_length = 1920;
    spectrum.window_type = MNN::AUDIO::HANNING;
    spectrum.center = false;
    spectrum.normalized = false;
    spectrum.power = 1.0f;
    auto features = MNN::AUDIO::mel_spectrogram(waveform, &mel, &spectrum);
    features = _Log(_Maximum(features, _Scalar<float>(1e-5f)));
    auto info = features->getInfo();
    if (!info || info->dim.size() != 2 || info->dim[1] != 80) return {};
    frames = info->dim[0];
    const float* source = features->readMap<float>();
    std::vector<float> channelMajor(static_cast<size_t>(80) * frames);
    for (int frame = 0; frame < frames; ++frame) {
        for (int bin = 0; bin < 80; ++bin) {
            channelMajor[static_cast<size_t>(bin) * frames + frame] = source[frame * 80 + bin];
        }
    }
    return channelMajor;
}

bool projectSpeaker(const std::vector<float>& embedding, const std::string& weightPath,
                    const std::string& biasPath, std::vector<float>& speaker) {
    std::vector<float> weights;
    std::vector<float> bias;
    if (!readFloats(weightPath, 80 * 192, weights) || !readFloats(biasPath, 80, bias)) return false;
    double norm = 0.0;
    for (float value : embedding) norm += static_cast<double>(value) * value;
    norm = std::sqrt(std::max(norm, 1e-12));
    speaker.resize(80);
    for (int output = 0; output < 80; ++output) {
        double value = bias[output];
        for (int input = 0; input < 192; ++input) {
            value += weights[output * 192 + input] * (embedding[input] / norm);
        }
        speaker[output] = static_cast<float>(value);
    }
    return finiteVector(speaker);
}

}  // namespace
void quantizeInt16InPlace(std::vector<float>& pcm) {
    // 语义与 writeWav16(编码) + MNN::AUDIO::load(解码) 完全一致，见头文件说明。
    for (float& value : pcm) {
        float clamped = value;
        if (clamped > 1.0f) clamped = 1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        const int16_t quantized = static_cast<int16_t>(lrintf(clamped * 32767.0f));
        value = static_cast<float>(quantized) / 32768.0f;
    }
}

void ditherInt16InPlace(std::vector<float>& pcm) {
    // 确定性 LCG（与 Kotlin 侧同参数），保证可复现。
    // 在 int16 域做 ±1 LSB：float 表示下即 ±(1/32768)。
    constexpr float kStep = 1.0f / 32768.0f;
    uint64_t seed = 0x5DEECE66DULL;
    for (float& value : pcm) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const int delta = static_cast<int>((seed >> 33) & 1ULL) * 2 - 1;   // -1 或 +1
        value += static_cast<float>(delta) * kStep;
    }
}

EnrollResult enrollFromPcm(const float* pcm, std::size_t sampleCount, int sampleRate,
                           const EnrollmentModels& models,
                           const std::string& outputDirectory, int threads) {
    EnrollResult result;
    const auto totalStart = std::chrono::steady_clock::now();
    result.inputSampleRate = sampleRate;
    result.inputSeconds = sampleRate > 0 ? static_cast<double>(sampleCount) / sampleRate : 0.0;
    if (pcm == nullptr || sampleRate <= 0) { result.code = 10; return result; }
    // 与旧入口一致的两道门槛：至少 16000 样本，且时长落在 3~15 秒
    if (sampleCount < 16000) { result.code = 10; return result; }
    if (sampleCount < static_cast<std::size_t>(sampleRate) * 3 ||
        sampleCount > static_cast<std::size_t>(sampleRate) * 15) { result.code = 11; return result; }

    const std::vector<float> source(pcm, pcm + sampleCount);
    std::vector<float> audio16 = sincResample(source.data(), source.size(), sampleRate, 16000);
    std::vector<float> audio24 = sincResample(source.data(), source.size(), sampleRate, 24000);

    int whisperFrames = 0;
    int campFrames = 0;
    int promptFrames = 0;
    auto whisperFeatures = extractWhisperFeatures(audio16, whisperFrames);
    auto campFeatures = extractCampFeatures(audio16, campFrames);
    auto promptMel = extractPromptMel(audio24, promptFrames);
    result.whisperFrames = whisperFrames;
    result.campFrames = campFrames;
    if (whisperFeatures.empty() || campFeatures.empty() || promptMel.empty()) { result.code = 12; return result; }

    std::vector<int32_t> tokens;
    std::vector<float> embedding;
    int code = runSpeechTokenizer(models.speechTokenizer, whisperFeatures, whisperFrames, threads, tokens, result.tokenizerMs);
    if (code != 0) { result.code = code; return result; }
    code = runCampPlus(models.campPlus, campFeatures, campFrames, threads, embedding, result.campMs);
    if (code != 0) { result.code = code; return result; }

    // 每句携带的零样本前缀保持有界；CAMPPlus 仍看完整参考片段。
    const int alignedTokens = std::min(
        {static_cast<int>(tokens.size()), promptFrames / 2, maxRealtimePromptTokens()});
    if (alignedTokens <= 0) { result.code = 40; return result; }
    tokens.resize(alignedTokens);
    const int alignedFrames = alignedTokens * 2;
    if (alignedFrames != promptFrames) {
        std::vector<float> aligned(static_cast<size_t>(80) * alignedFrames);
        for (int channel = 0; channel < 80; ++channel) {
            std::copy(promptMel.begin() + static_cast<size_t>(channel) * promptFrames,
                      promptMel.begin() + static_cast<size_t>(channel) * promptFrames + alignedFrames,
                      aligned.begin() + static_cast<size_t>(channel) * alignedFrames);
        }
        promptMel.swap(aligned);
        promptFrames = alignedFrames;
    }
    std::vector<float> speaker;
    if (!projectSpeaker(embedding, models.affineWeight, models.affineBias, speaker)) { result.code = 41; return result; }

    std::ofstream tokenOutput(joinPath(outputDirectory, "prompt-speech-tokens.csv"));
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) tokenOutput << ',';
        tokenOutput << tokens[i];
    }
    if (!tokenOutput || !writeFloats(joinPath(outputDirectory, "prompt-cond.bin"), promptMel) ||
        !writeFloats(joinPath(outputDirectory, "spks.bin"), speaker)) { result.code = 42; return result; }

    result.promptTokens = static_cast<int>(tokens.size());
    result.promptFrames = promptFrames;
    result.totalMs = elapsedMs(totalStart);

    std::ofstream report(joinPath(outputDirectory, "enrollment-report.json"));
    report << std::fixed << std::setprecision(3)
           << "{\n"
           << "  \"inputSampleRate\": " << result.inputSampleRate << ",\n"
           << "  \"inputSeconds\": " << result.inputSeconds << ",\n"
           << "  \"whisperFrames\": " << result.whisperFrames << ",\n"
           << "  \"campFrames\": " << result.campFrames << ",\n"
           << "  \"promptTokens\": " << result.promptTokens << ",\n"
           << "  \"promptFrames\": " << result.promptFrames << ",\n"
           << "  \"tokenizerMs\": " << result.tokenizerMs << ",\n"
           << "  \"campMs\": " << result.campMs << ",\n"
           << "  \"totalMs\": " << result.totalMs << "\n"
           << "}\n";
    result.code = report ? 0 : 43;
    return result;
}

EnrollResult enrollFromWavFile(const std::string& wavPath, const EnrollmentModels& models,
                               const std::string& outputDirectory, int threads) {
    EnrollResult result;
    auto loaded = MNN::AUDIO::load(wavPath, 0);
    if (loaded.first == nullptr || loaded.second <= 0) { result.code = 10; return result; }
    auto info = loaded.first->getInfo();
    if (!info || info->size == 0) { result.code = 10; return result; }
    const float* source = loaded.first->readMap<float>();
    if (source == nullptr) { result.code = 10; return result; }
    return enrollFromPcm(source, static_cast<std::size_t>(info->size), loaded.second,
                         models, outputDirectory, threads);
}

}  // namespace cosy