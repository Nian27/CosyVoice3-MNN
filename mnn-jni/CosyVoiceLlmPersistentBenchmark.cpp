#include <llm/llm.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using MNN::Transformer::Llm;

namespace {

constexpr int kSpeechOffset = 151924;
constexpr int kSpeechTokenMax = kSpeechOffset + 6560;
constexpr int kSpeechEos = kSpeechOffset + 6562;

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<int> readPromptSpeechTokens(const std::string& path) {
    std::ifstream input(path);
    std::vector<int> tokens;
    int token = 0;
    while (input >> token) {
        tokens.push_back(token);
        if (input.peek() == ',') {
            input.ignore();
        }
    }
    return tokens;
}

bool isSpeechOutput(int token) {
    return (token >= kSpeechOffset && token <= kSpeechTokenMax) ||
           token == kSpeechEos;
}

int utf8CodePointCount(const std::string& text) {
    int count = 0;
    for (unsigned char value : text) {
        if ((value & 0xC0) != 0x80) {
            ++count;
        }
    }
    return count;
}

bool writeSpeechTokens(const std::string& path, const std::vector<int>& outputIds) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    bool first = true;
    for (int token : outputIds) {
        if (token < kSpeechOffset || token > kSpeechTokenMax) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        output << token - kSpeechOffset;
        first = false;
    }
    output << '\n';
    return static_cast<bool>(output);
}

bool writeAllTokens(const std::string& path, const std::vector<int>& outputIds) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    for (size_t index = 0; index < outputIds.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << outputIds[index];
    }
    output << '\n';
    return static_cast<bool>(output);
}

std::string joinPath(const std::string& directory, const std::string& file) {
    if (directory.empty()) {
        return file;
    }
    const char last = directory.back();
    if (last == '/' || last == '\\') {
        return directory + file;
    }
#ifdef _WIN32
    return directory + "\\" + file;
#else
    return directory + "/" + file;
#endif
}

std::string currentRss() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return line.substr(6);
        }
    }
    return {};
}

void signalReadyAndWait(const std::string& readyPath, const std::string& startPath) {
    if (readyPath.empty() || startPath.empty()) {
        return;
    }
    std::ofstream(readyPath) << "ready\n";
    while (!std::ifstream(startPath).good()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

struct LlmRuntimeState {
    std::string configPath;
    std::unique_ptr<Llm> llm;
};

LlmRuntimeState& llmRuntimeState() {
    static LlmRuntimeState state;
    return state;
}

LlmRuntimeState& cpuContinuationRuntimeState() {
    static LlmRuntimeState state;
    return state;
}

void resetLlmRuntime() {
    llmRuntimeState() = LlmRuntimeState{};
    cpuContinuationRuntimeState() = LlmRuntimeState{};
}

bool ensureRuntimeLoaded(
    LlmRuntimeState& runtime,
    const std::string& configPath,
    double& loadMs) {
    if (runtime.llm && runtime.configPath == configPath) {
        loadMs = 0.0;
        return true;
    }
    runtime = LlmRuntimeState{};
    const auto loadStart = std::chrono::steady_clock::now();
    runtime.llm.reset(Llm::createLLM(configPath));
    if (!runtime.llm || !runtime.llm->load()) {
        runtime = LlmRuntimeState{};
        return false;
    }
    runtime.configPath = configPath;
    loadMs = elapsedMs(loadStart);
    return true;
}

std::string cpuConfigForHexagon(const std::string& configPath) {
    std::string cpuPath = configPath;
    const std::string hexagonMarker = "config-hexagon-";
    const size_t position = cpuPath.rfind(hexagonMarker);
    if (position == std::string::npos) {
        return {};
    }
    cpuPath.replace(position, hexagonMarker.size(), "config-cpu-");
    return cpuPath;
}

}  // namespace

int cosyVoiceLlmMain(int argc, const char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " config.json prompts.txt max_tokens prompt_speech_tokens.csv output_dir"
                  << std::endl;
        return 2;
    }

    const std::string configPath = argv[1];
    const std::string promptsPath = argv[2];
    const int maxTokens = std::max(1, std::atoi(argv[3]));
    const std::string promptTokenPath = argv[4];
    const std::string outputDirectory = argv[5];
    const std::string readyPath = argc >= 7 ? argv[6] : "";
    const std::string startPath = argc >= 8 ? argv[7] : "";
    const bool appendPromptSpeechTokens = argc < 9 || std::string(argv[8]) != "0";
    const std::vector<std::string> prompts = readLines(promptsPath);
    const std::vector<int> promptSpeechTokens = readPromptSpeechTokens(promptTokenPath);
    if (prompts.empty() || (appendPromptSpeechTokens && promptSpeechTokens.empty())) {
        std::cerr << "prompts_or_prompt_speech_tokens_empty=true" << std::endl;
        return 3;
    }

    const bool hexagonRuntime = configPath.find("hexagon") != std::string::npos;
    const size_t configSlash = configPath.find_last_of("/\\");
    const std::string configDirectory =
        configSlash == std::string::npos ? "." : configPath.substr(0, configSlash);
    const std::vector<std::string> stagedLayerLines =
        readLines(joinPath(configDirectory, "hexagon-stage-layers.txt"));
    const std::string stagedHexagonLayers =
        stagedLayerLines.empty() ? "" : stagedLayerLines.front();
    const std::vector<std::string> stagedOpLines =
        readLines(joinPath(configDirectory, "hexagon-stage-ops.txt"));
    const std::string stagedHexagonOps =
        stagedOpLines.empty() ? "" : stagedOpLines.front();
    const std::vector<std::string> stagedNameLines =
        readLines(joinPath(configDirectory, "hexagon-stage-name.txt"));
    const std::string stagedHexagonName =
        stagedNameLines.empty() ? "" : stagedNameLines.front();
    // Production hard gate: continuous full-graph Hexagon Attention/KV is known to
    // collapse CosyVoice speech tokens (typically 158484 / 151948). Hexagon
    // is allowed continuously only for an explicit transformer-layer allow-list.
    const bool continuousHexagon = hexagonRuntime && !stagedHexagonLayers.empty();
    if (continuousHexagon) {
        setenv("MNN_HEXAGON_LAYERS", stagedHexagonLayers.c_str(), 1);
        if (stagedHexagonOps.empty()) {
            unsetenv("MNN_HEXAGON_OPS");
        } else {
            setenv("MNN_HEXAGON_OPS", stagedHexagonOps.c_str(), 1);
        }
        if (stagedHexagonName.empty()) {
            unsetenv("MNN_HEXAGON_NAME");
        } else {
            setenv("MNN_HEXAGON_NAME", stagedHexagonName.c_str(), 1);
        }
        unsetenv("MNN_HEXAGON_ATTENTION");
        unsetenv("MNN_HEXAGON_ATTENTION_LAYER");
    } else {
        unsetenv("MNN_HEXAGON_LAYERS");
        unsetenv("MNN_HEXAGON_OPS");
        unsetenv("MNN_HEXAGON_NAME");
        unsetenv("MNN_HEXAGON_ATTENTION");
        unsetenv("MNN_HEXAGON_ATTENTION_LAYER");
    }
    const bool hybridNpuPrefill = hexagonRuntime && !continuousHexagon;
    LlmRuntimeState& runtime = llmRuntimeState();
    double primaryLoadMs = 0.0;
    if (!ensureRuntimeLoaded(runtime, configPath, primaryLoadMs)) {
        resetLlmRuntime();
        std::cerr << "load=false config=" << configPath << std::endl;
        return 4;
    }
    LlmRuntimeState* cpuRuntime = nullptr;
    double cpuLoadMs = 0.0;
    if (hybridNpuPrefill) {
        const std::string cpuConfigPath = cpuConfigForHexagon(configPath);
        if (cpuConfigPath.empty()) {
            std::cerr << "cpu_continuation_config=false config=" << configPath << std::endl;
            return 4;
        }
        cpuRuntime = &cpuContinuationRuntimeState();
        if (!ensureRuntimeLoaded(*cpuRuntime, cpuConfigPath, cpuLoadMs)) {
            resetLlmRuntime();
            std::cerr << "cpu_continuation_load=false config=" << cpuConfigPath << std::endl;
            return 4;
        }
    }
    Llm* llm = runtime.llm.get();
    const double loadMs = primaryLoadMs + cpuLoadMs;
    std::cout << std::fixed << std::setprecision(3)
              << "load_ms=" << loadMs
              << " hybrid_npu_prefill=" << (hybridNpuPrefill ? "true" : "false")
              << " continuous_hexagon=" << (continuousHexagon ? "true" : "false")
              << " vm_rss=" << currentRss() << std::endl;
    signalReadyAndWait(readyPath, startPath);

    std::ofstream metrics(joinPath(outputDirectory, "llm-persistent.jsonl"));
    if (!metrics) {
        std::cerr << "cannot_create_metrics=true" << std::endl;
        return 5;
    }

    for (size_t request = 0; request < prompts.size(); ++request) {
        llm->reset();
        std::string prompt = llm->apply_chat_template(prompts[request]);
        if (prompt.empty()) {
            prompt = prompts[request];
        }
        const std::string marker = "<|endofprompt|>";
        const size_t markerPosition = prompts[request].rfind(marker);
        const std::string targetText = markerPosition == std::string::npos
                                           ? prompts[request]
                                           : prompts[request].substr(markerPosition + marker.size());
        const int minSpeechTokens = std::max(2, utf8CodePointCount(targetText) * 2);
        llm->setCosyVoiceMinTokens(minSpeechTokens);
        std::vector<int> inputIds = llm->tokenizer_encode(prompt);
        if (appendPromptSpeechTokens) {
            for (int token : promptSpeechTokens) {
                inputIds.push_back(token + kSpeechOffset);
            }
        }

        const auto requestStart = std::chrono::steady_clock::now();
        std::vector<int> outputIds;
        double npuPrefillMs = 0.0;
        double cpuPrefillMs = 0.0;
        double decodeMs = 0.0;
        int inputTokenCount = 0;
        int generatedTokenCount = 0;
        if (hybridNpuPrefill) {
            const auto* npuBeforeContext = llm->getContext();
            const int64_t npuPrefillBefore = npuBeforeContext->prefill_us;
            const std::vector<int> firstIds = llm->generate(inputIds, 1);
            const auto* npuContext = llm->getContext();
            const int64_t npuPrefillDelta = npuContext->prefill_us >= npuPrefillBefore
                                                ? npuContext->prefill_us - npuPrefillBefore
                                                : npuContext->prefill_us;
            npuPrefillMs = npuPrefillDelta / 1000.0;
            inputTokenCount = npuContext->prompt_len;
            if (firstIds.size() != 1 || firstIds.front() == kSpeechEos ||
                !isSpeechOutput(firstIds.front())) {
                std::cerr << "npu_first_token_invalid=true";
                if (!firstIds.empty()) {
                    std::cerr << " token=" << firstIds.front();
                }
                std::cerr << std::endl;
                return 7;
            }

            Llm* cpuLlm = cpuRuntime->llm.get();
            cpuLlm->reset();
            cpuLlm->setCosyVoiceMinTokens(std::max(1, minSpeechTokens - 1));
            const auto* cpuBeforeContext = cpuLlm->getContext();
            const int64_t cpuPrefillBefore = cpuBeforeContext->prefill_us;
            const int64_t cpuDecodeBefore = cpuBeforeContext->decode_us;
            std::vector<int> cpuInputIds = inputIds;
            cpuInputIds.push_back(firstIds.front());
            const std::vector<int> continuationIds =
                cpuLlm->generate(cpuInputIds, std::max(1, maxTokens - 1));
            const auto* cpuContext = cpuLlm->getContext();
            const int64_t cpuPrefillDelta = cpuContext->prefill_us >= cpuPrefillBefore
                                                ? cpuContext->prefill_us - cpuPrefillBefore
                                                : cpuContext->prefill_us;
            const int64_t cpuDecodeDelta = cpuContext->decode_us >= cpuDecodeBefore
                                               ? cpuContext->decode_us - cpuDecodeBefore
                                               : cpuContext->decode_us;
            cpuPrefillMs = cpuPrefillDelta / 1000.0;
            decodeMs = cpuDecodeDelta / 1000.0;
            outputIds.reserve(1 + continuationIds.size());
            outputIds.push_back(firstIds.front());
            outputIds.insert(outputIds.end(), continuationIds.begin(), continuationIds.end());
            generatedTokenCount = static_cast<int>(outputIds.size());
        } else {
            const auto* beforeContext = llm->getContext();
            const int64_t prefillBefore = beforeContext->prefill_us;
            const int64_t decodeBefore = beforeContext->decode_us;
            outputIds = llm->generate(inputIds, maxTokens);
            const auto* context = llm->getContext();
            const int64_t prefillDelta = context->prefill_us >= prefillBefore
                                             ? context->prefill_us - prefillBefore
                                             : context->prefill_us;
            const int64_t decodeDelta = context->decode_us >= decodeBefore
                                            ? context->decode_us - decodeBefore
                                            : context->decode_us;
            npuPrefillMs = prefillDelta / 1000.0;
            decodeMs = decodeDelta / 1000.0;
            inputTokenCount = context->prompt_len;
            generatedTokenCount = context->gen_seq_len;
        }
        const double wallMs = elapsedMs(requestStart);

        int validSpeech = 0;
        int invalid = 0;
        int speechTokens = 0;
        for (int token : outputIds) {
            if (isSpeechOutput(token)) {
                ++validSpeech;
                if (token != kSpeechEos) {
                    ++speechTokens;
                }
            } else {
                ++invalid;
            }
        }
        const std::string tokenFile =
            joinPath(outputDirectory, "speech-tokens-" + std::to_string(request) + ".csv");
        if (!writeSpeechTokens(tokenFile, outputIds)) {
            std::cerr << "cannot_write_tokens=" << tokenFile << std::endl;
            return 6;
        }
        const std::string rawTokenFile =
            joinPath(outputDirectory, "raw-output-ids-" + std::to_string(request) + ".csv");
        if (!writeAllTokens(rawTokenFile, outputIds)) {
            std::cerr << "cannot_write_raw_tokens=" << rawTokenFile << std::endl;
            return 6;
        }

        const double prefillMs = npuPrefillMs + cpuPrefillMs;
        const double tokensPerSecond = decodeMs > 0.0
                                           ? generatedTokenCount * 1000.0 / decodeMs
                                           : 0.0;
        metrics << std::fixed << std::setprecision(6)
                << "{\"request\":" << request
                << ",\"inputTokens\":" << inputTokenCount
                << ",\"generatedTokens\":" << generatedTokenCount
                << ",\"speechTokens\":" << speechTokens
                << ",\"validSpeechOutputs\":" << validSpeech
                << ",\"invalidOutputs\":" << invalid
                 << ",\"promptSpeechTokensAppended\":"
                 << (appendPromptSpeechTokens ? "true" : "false")
                 << ",\"hybridNpuPrefill\":" << (hybridNpuPrefill ? "true" : "false")
                 << ",\"continuousHexagon\":" << (continuousHexagon ? "true" : "false")
                 << ",\"stagedHexagonLayers\":\"" << stagedHexagonLayers << "\""
                 << ",\"stagedHexagonOps\":\"" << stagedHexagonOps << "\""
                 << ",\"stagedHexagonName\":\"" << stagedHexagonName << "\""
                 << ",\"npuFirstTokenValid\":" << (hybridNpuPrefill ? "true" : "false")
                << ",\"loadMs\":" << loadMs
                << ",\"prefillMs\":" << prefillMs
                << ",\"npuPrefillMs\":" << npuPrefillMs
                << ",\"cpuPrefillMs\":" << cpuPrefillMs
                << ",\"decodeMs\":" << decodeMs
                << ",\"wallMs\":" << wallMs
                << ",\"tokensPerSecond\":" << tokensPerSecond << "}\n";
        metrics.flush();

        std::cout << std::fixed << std::setprecision(3)
                  << "request=" << request
                  << " speech_tokens=" << speechTokens
                  << " hybrid_npu_prefill=" << (hybridNpuPrefill ? "true" : "false")
                  << " npu_prefill_ms=" << npuPrefillMs
                  << " cpu_prefill_ms=" << cpuPrefillMs
                  << " prefill_ms=" << prefillMs
                  << " decode_ms=" << decodeMs
                  << " wall_ms=" << wallMs
                  << " tokens_per_second=" << tokensPerSecond
                  << " invalid_outputs=" << invalid
                  << " vm_rss=" << currentRss() << std::endl;
        if (speechTokens == 0 || invalid != 0) {
            return 7;
        }
    }

    return 0;
}

#ifndef COSYVOICE_LLM_LIBRARY
int main(int argc, const char* argv[]) {
    return cosyVoiceLlmMain(argc, argv);
}
#endif
