#include <jni.h>

// 薄包装：解析 jstring -> EnrollmentModels -> CosyVoiceEnrollmentCore。
//
// 算法全部在 CosyVoiceEnrollmentCore.cpp（参考音频克隆与文字设计音色共用同一份）。
// 本文件只负责 JNI 边界，不再持有任何算法实现 —— 这样"修一次两边都生效"。

#include "CosyVoiceEnrollmentCore.h"

#include <algorithm>
#include <string>

namespace {

class UtfChars {
public:
    UtfChars(JNIEnv* env, jstring value)
        : env_(env), value_(value), chars_(value ? env->GetStringUTFChars(value, nullptr) : nullptr) {}

    ~UtfChars() {
        if (chars_) env_->ReleaseStringUTFChars(value_, chars_);
    }

    const char* get() const { return chars_ ? chars_ : ""; }

private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_;
};

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_io_legado_app_cosy_CosyVoiceEnrollmentNative_enroll(
    JNIEnv* env, jobject, jstring tokenizerPath, jstring campPath,
    jstring affineWeightPath, jstring affineBiasPath, jstring wavPath,
    jstring outputDirectory, jint threads) {
    UtfChars tokenizer(env, tokenizerPath);
    UtfChars camp(env, campPath);
    UtfChars weights(env, affineWeightPath);
    UtfChars bias(env, affineBiasPath);
    UtfChars wav(env, wavPath);
    UtfChars output(env, outputDirectory);
    if (env->ExceptionCheck()) return 1;
    cosy::EnrollmentModels models;
    models.speechTokenizer = tokenizer.get();
    models.campPlus = camp.get();
    models.affineWeight = weights.get();
    models.affineBias = bias.get();
    const auto result = cosy::enrollFromWavFile(
        wav.get(), models, output.get(), std::max(1, static_cast<int>(threads)));
    return result.code;
}
