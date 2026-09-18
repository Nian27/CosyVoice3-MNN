//
//  CosyVoiceHintKnobs.hpp
//  CosyVoice3-MNN benchmark-only A/B knobs for MNN runtime hints.
//
//  Why this exists: the CosyVoice3 benchmarks previously hard-coded
//  BackendConfig::Memory_Normal and never called Interpreter::setSessionHint,
//  so a set of MNN features that are already compiled into the shipped
//  libMNN.so stayed dormant. This header exposes each of them through an
//  environment variable so that ONE build can be swept across variants with
//  a single changed variable per run (project rule: one variable per
//  experiment).
//
//  Default behaviour is byte-identical to the previous code: every knob
//  unset => Memory_Normal and no setSessionHint call at all.
//
//  Knobs (all optional):
//    MNN_HINT_MEMORY      normal|high|low  -> BackendConfig::MemoryMode
//                                           (high lifts the 4-thread cap in
//                                            CPUDeconvolution for ConvTranspose)
//    MNN_HINT_KLEIDIAI    0|1              -> HintMode::CPU_ENABLE_KLEIDIAI
//    MNN_HINT_WINOGRAD    0..3             -> HintMode::WINOGRAD_MEMORY_LEVEL
//    MNN_HINT_LITTLECORE  0..100           -> HintMode::CPU_LITTLECORE_DECREASE_RATE
//    MNN_HINT_ATTENTION   int              -> HintMode::ATTENTION_OPTION
//    MNN_HINT_CORE_IDS    "6,7,0,1,2,3"    -> HintMode::CPU_CORE_IDS (int* overload)
//
//  Usage on device:
//    adb shell "MNN_HINT_MEMORY=high /data/local/tmp/CosyVoiceHiFTBenchmark.out ..."
//
#ifndef COSYVOICE_HINT_KNOBS_HPP
#define COSYVOICE_HINT_KNOBS_HPP

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace cosyvoice_hints {

inline bool envPresent(const char* name) {
    const char* raw = std::getenv(name);
    return raw != nullptr && raw[0] != '\0';
}

inline int envInt(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    return std::atoi(raw);
}

inline std::string envText(const char* name, const char* fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return std::string(fallback);
    }
    return std::string(raw);
}

// BackendConfig::MemoryMode from MNN_HINT_MEMORY; default Memory_Normal.
inline MNN::BackendConfig::MemoryMode memoryModeFromEnv() {
    const std::string value = envText("MNN_HINT_MEMORY", "normal");
    if (value == "high") {
        return MNN::BackendConfig::Memory_High;
    }
    if (value == "low") {
        return MNN::BackendConfig::Memory_Low;
    }
    return MNN::BackendConfig::Memory_Normal;
}

inline const char* memoryModeName(MNN::BackendConfig::MemoryMode mode) {
    switch (mode) {
        case MNN::BackendConfig::Memory_High:
            return "high";
        case MNN::BackendConfig::Memory_Low:
            return "low";
        default:
            return "normal";
    }
}

// Apply every hint that is actually set. Returns a human-readable summary for
// the benchmark report; "none" means MNN ran with its own defaults.
inline std::string applySessionHints(MNN::Interpreter* net) {
    if (net == nullptr) {
        return "none";
    }
    std::string summary;
    auto note = [&summary](const std::string& item) {
        if (!summary.empty()) {
            summary += ",";
        }
        summary += item;
    };

    if (envPresent("MNN_HINT_KLEIDIAI")) {
        const int value = envInt("MNN_HINT_KLEIDIAI", 0);
        net->setSessionHint(MNN::Interpreter::HintMode::CPU_ENABLE_KLEIDIAI, value);
        note("kleidiai=" + std::to_string(value));
    }
    if (envPresent("MNN_HINT_WINOGRAD")) {
        const int value = envInt("MNN_HINT_WINOGRAD", 3);
        net->setSessionHint(MNN::Interpreter::HintMode::WINOGRAD_MEMORY_LEVEL, value);
        note("winograd=" + std::to_string(value));
    }
    if (envPresent("MNN_HINT_LITTLECORE")) {
        const int value = envInt("MNN_HINT_LITTLECORE", 50);
        net->setSessionHint(MNN::Interpreter::HintMode::CPU_LITTLECORE_DECREASE_RATE, value);
        note("littlecore=" + std::to_string(value));
    }
    if (envPresent("MNN_HINT_ATTENTION")) {
        const int value = envInt("MNN_HINT_ATTENTION", 8);
        net->setSessionHint(MNN::Interpreter::HintMode::ATTENTION_OPTION, value);
        note("attention=" + std::to_string(value));
    }
    if (envPresent("MNN_HINT_CORE_IDS")) {
        // Comma-separated CPU ids, e.g. "6,7,0,1,2,3". On SM8850 the two
        // 4.6GHz cores are cpu6/cpu7 while cpu0-5 are 3.63GHz, so the default
        // thread->core mapping can leave the fastest cores idle.
        const std::string raw = envText("MNN_HINT_CORE_IDS", "");
        std::vector<int> ids;
        size_t cursor = 0;
        while (cursor < raw.size()) {
            const size_t comma = raw.find(',', cursor);
            const std::string token = raw.substr(
                cursor, comma == std::string::npos ? std::string::npos : comma - cursor);
            if (!token.empty()) {
                ids.push_back(std::atoi(token.c_str()));
            }
            if (comma == std::string::npos) {
                break;
            }
            cursor = comma + 1;
        }
        if (!ids.empty()) {
            net->setSessionHint(MNN::Interpreter::HintMode::CPU_CORE_IDS, ids.data(), ids.size());
            note("coreids=" + raw);
        }
    }
    if (summary.empty()) {
        return "none";
    }
    return summary;
}

} // namespace cosyvoice_hints

#endif // COSYVOICE_HINT_KNOBS_HPP
