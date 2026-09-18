package com.cosyvoice.app

/**
 * VoiceDesign 原生封装（第 5 个 JNI 库 libcosy_voicedesign_jni.so）。
 *
 * 与另外 4 个库不同，这个库是新编的，导出符号就是正确的 com_cosyvoice_app_*，
 * 因此不需要 io.legado.app.cosy 桥接层。
 *
 * 链：文字描述 -> tokenize -> prompt 拼装 -> GraphB prefill -> 自驱 16 码
 *     -> EOS -> tokenizer decoder -> 24kHz WAV
 */
internal object CosyVoiceVoiceDesignNative {
    init { System.loadLibrary("cosy_voicedesign_jni") }

    external fun nativeCreate(): Long

    /** GB-SPLIT-P0：4L embedded vs external 的 A/B 探针（读 <dir>/vd_gb4l.txt 决定 A/B） */
    external fun nativeGb4lProbe(modelDir: String, reps: Int): String
    external fun nativeRelease(ptr: Long)
    external fun nativeLastError(ptr: Long): String
    external fun nativeLoad(ptr: Long, modelDir: String, backend: String, maxPos: Int, nativeLibDir: String): Boolean
    /**
     * @param enrollModelsDir 非空时，decoder 出 PCM 后**直接**调用 CosyVoiceEnrollmentCore
     *                        产出注册产物（prompt-speech-tokens.csv / prompt-cond.bin / spks.bin）
     *                        写进 enrollOutDir —— 不再经过 WAV 往返。
     *                        reference.wav 仍然会写（它是 VoiceIdentity 的长期资产，不是传输介质）。
     * @param enrollOutDir    注册产物输出目录（须已存在）
     */
    external fun nativeRun(
        ptr: Long, modelDir: String, outWavPath: String, maxFrames: Int,
        instruct: String, text: String, lang: Int,
        enrollModelsDir: String, enrollOutDir: String
    ): String
}
