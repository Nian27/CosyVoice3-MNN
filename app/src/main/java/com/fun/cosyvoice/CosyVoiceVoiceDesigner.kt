package com.cosyvoice.app

import android.util.Log
import java.io.File

/**
 * 统一音色创建 —— 「文字设计音色」模式。
 *
 * 流水线（与「参考音频克隆」共用同一条下游）：
 *
 *   音色描述 ──VoiceDesign──▶ reference.wav ──CosyVoiceEnrollmentNative.enroll──▶ VoiceProfile
 *                                                                                    │
 *                                                              保存角色库 ◀──────────┘
 *                                                                  │
 *                                                           CosyVoice 朗读（不变）
 *
 * 关键点：
 *  - 参考文本是固定的（用户不必填），因此 promptText 天然已知 —— enroll 后可直接注册；
 *  - VoiceDesign 的 WAV 会被 installEnrolledVoiceProfile 存成 source.wav 保留下来，
 *    将来换 enrollment 版本 / 迁移后端都不必再付一次生成成本；
 *  - 合成侧零改动（产出的是同一种 VoiceProfile）。
 */
internal object CosyVoiceVoiceDesigner {

    private const val TAG = "CosyVoiceDesign"

    /**
     * 固定的参考文本（设计出来的音色会念这句话，念出来的音频就是注册用的参考音频）。
     *
     * 【2026-09-18 NPU-R14】长度按「实时版上限」定，不再是 3~6 秒的泛泛区间。
     *
     * 注册时音频会被 `MAX_REALTIME_PROMPT_TOKENS = 125`（= 5.0 秒 / 250 帧）截断，
     * 多出来的部分直接丢掉；而 `promptPrefix` 里存的仍然是**完整文本**。
     * 于是文本与音频提示不匹配 —— 实测旧默认（26 字）设计出 7.68 秒，被截到 5.0 秒，
     * 前 5 秒大约只覆盖前 17 个字，却告诉模型这句话有 26 个字。
     *
     * 内置基准音色 `builtin-mnn-reference-v1` 用的是 15 个字 / 87 token（3.48 秒），
     * 文本与音频是对齐的。这里对齐到同一量级：15 个字。
     *
     * **时长是估算，不是实测**：按旧默认的实测速率（26 字 → 7.68 秒，约 0.30 秒/字）推算
     * 15 字约 4.4 秒 —— 在 5 秒上限之内（不会被截断），也高于 `minSeconds = 3.0`。
     * 真实时长仍受这句文本的停顿影响（旧默认的 7.68 秒里含句末长停顿），
     * 需要跑一次 design 才能确认；若确认低于 3 秒，把这句话再加长几个字即可。
     */
    const val DEFAULT_REFERENCE_TEXT =
        "风从北边吹过来，带着一点凉意。"

    /** 默认音色描述（UI 首次打开时的占位）。 */
    const val DEFAULT_DESCRIPTION = "青年女性，声音清冷柔和，语速偏慢，带一点疏离感"

    /**
     * 一次完整的「文字设计音色」创建。
     *
     * @param designModelsDir VoiceDesign 模型目录（21 个文件）
     * @param backend "opencl" / "cpu" / "htp"
     * @param maxFrames 自驱解码上限（300 = decoder 的静态上限）
     * @param minSeconds/maxSeconds design 模式的时长区间（原生 enroll 支持 3~15 秒）
     */
    fun createFromDesign(
        store: CosyVoiceStore,
        nativeLibDir: String,
        designModelsDir: File,
        description: String,
        displayName: String,
        backend: String = "opencl",
        maxFrames: Int = 300,
        referenceText: String = DEFAULT_REFERENCE_TEXT,
        minSeconds: Double = 3.0,
        maxSeconds: Double = 10.0,
        onProgress: (String) -> Unit = {}
    ): CosyVoiceVoiceProfile {
        require(description.isNotBlank()) { "请填写音色描述" }
        require(displayName.isNotBlank()) { "请填写音色名称" }
        require(store.modelStatus().ready) { "MNN 朗读模型不完整" }
        require(store.enrollmentStatus().ready) { "请先导入音色创建扩展" }
        require(designModelsDir.isDirectory) { "VoiceDesign 模型目录不存在：$designModelsDir" }

        CosyVoiceDesignService.rssLine("before-any-native")
        val runDir = File(store.workDir, "design-" + android.os.SystemClock.elapsedRealtime())
            .apply { deleteRecursively(); mkdirs() }
        try {
            val referenceWav = File(runDir, "reference.wav")
            // 注册产物直接由 native 在 decoder 之后产出（不再经 WAV 往返）。
            val profileOut = File(runDir, "cosyvoice").apply { mkdirs() }

            // ---- 1. 文字设计音色 ----
            onProgress("正在用描述生成音色（首次会编译图，约 5-8 分钟）…")
            val t0 = System.currentTimeMillis()
            val engine = CosyVoiceVoiceDesignNative.nativeCreate()
            check(engine != 0L) { "VoiceDesign 引擎创建失败" }
            CosyVoiceDesignService.rssLine("after-loadLibrary+nativeCreate")
            val designStatus: String
            try {
                val dir = designModelsDir.absolutePath.trimEnd('/') + "/"
                check(CosyVoiceVoiceDesignNative.nativeLoad(engine, dir, backend, 384, nativeLibDir)) {
                    "VoiceDesign 模型加载失败：" + CosyVoiceVoiceDesignNative.nativeLastError(engine)
                }
                // 关键：把 enrollment 模型目录与产物目录一并交给 native，
                // decoder 出 PCM 后**直接**产出注册产物（不再经 WAV 往返）。
                designStatus = CosyVoiceVoiceDesignNative.nativeRun(
                    engine, dir, referenceWav.absolutePath, maxFrames, description, referenceText, 2055,
                    store.enrollmentDir.absolutePath, profileOut.absolutePath)
            } finally {
                CosyVoiceVoiceDesignNative.nativeRelease(engine)
            }
            Log.i(TAG, "design status=$designStatus")
            onProgress("设计完成（${(System.currentTimeMillis() - t0) / 1000}s）：$designStatus")
            check(designStatus.startsWith("OK")) { designStatus }
            check(referenceWav.isFile && referenceWav.length() > 44L) { "VoiceDesign 未产出 WAV" }

            // design 模式的时长检查（原生 enroll 支持 3~15 秒，这里用 3~10 秒）
            val seconds = (referenceWav.length() - 44L) / 2.0 / 24000.0
            check(seconds in minSeconds..maxSeconds) {
                "设计出的参考音频 %.2f 秒，不在 %.0f~%.0f 秒范围内".format(seconds, minSeconds, maxSeconds)
            }

            // dither 现在由 native 在「int16 round-trip 之后」的 PCM 上做
            // （见 voicedesign_jni.cpp 里 cosy::ditherInt16InPlace 的调用）。
            // reference.wav 因此保持**未加 dither 的原始设计产物** —— 它是资产，
            // 不该被预处理污染。

            // 诊断用：把 design 产物留一份，便于排查 enroll 失败（也方便人工试听）
            runCatching {
                referenceWav.copyTo(File(store.voiceDesignDir, "last-reference.wav"), overwrite = true)
                File(store.voiceDesignDir, "last-reference.txt").writeText(
                    "description=" + description + "\n" +
                    "referenceText=" + referenceText + "\n" +
                    "seconds=" + "%.3f".format(seconds) + "\n" +
                    "bytes=" + referenceWav.length() + "\n", Charsets.UTF_8)
            }

            // ---- 2. enrollment 已在 native 内完成（与「参考音频克隆」共用 CosyVoiceEnrollmentCore）----
            // nativeRun 的返回串会带 "ENROLL_OK tokens=.. frames=.. ms=.."；失败时返回 ERR ENROLL_FAIL。
            check(profileOut.isDirectory && File(profileOut, "prompt-cond.bin").isFile) {
                "native 未产出注册产物（profileOut=${profileOut.absolutePath}）"
            }

            // ---- 3. 注册进角色库（source.wav 会被保留） ----
            onProgress("正在注册音色档案…")
            val profile = store.installEnrolledVoiceProfile(
                displayName = displayName,
                promptText = referenceText,      // 固定文本 -> 天然已知，用户不用填
                nativeOutputDirectory = profileOut,
                sourceWav = referenceWav)

            // ---- 4. 记录来源（便于将来重 enrollment / 迁移，不必再生成一次） ----
            runCatching {
                val designMeta = File(profile.directory, "design.json")
                designMeta.writeText(
                    buildString {
                        append("{\n")
                        append("  \"source\": \"VOICE_DESIGN\",\n")
                        append("  \"description\": ").append(json(description)).append(",\n")
                        append("  \"referenceText\": ").append(json(referenceText)).append(",\n")
                        append("  \"backend\": ").append(json(backend)).append(",\n")
                        append("  \"seconds\": ").append("%.3f".format(seconds)).append(",\n")
                        append("  \"designStatus\": ").append(json(designStatus)).append(",\n")
                        append("  \"promptTokenCount\": ").append(profile.promptTokenCount).append(",\n")
                        append("  \"promptFrameCount\": ").append(profile.promptFrameCount).append(",\n")
                        append("  \"promptTruncatedByRealtimeLimit\": ")
                            .append(profile.promptTokenCount >= store.realtimePromptTokenLimit())
                            .append(",\n")
                        append("  \"createdAt\": ").append(System.currentTimeMillis()).append("\n")
                        append("}\n")
                    }, Charsets.UTF_8)
            }.onFailure { Log.w(TAG, "写入 design.json 失败", it) }

            onProgress("音色创建完成：${profile.displayName}（${profile.promptTokenCount} Token）")
            return profile
        } finally {
            runDir.deleteRecursively()
        }
    }

    /**
     * 就地把 16-bit PCM WAV 的每个样本叠加 ±1 LSB 的确定性 dither。
     *
     * 只改样本数据，不动 44 字节头。用确定性 LCG（不用随机源）保证可复现。
     */
    /** 供诊断入口复用（enrollOnly）。 */
    fun applyDitherForTest(wav: File) = ditherInPlace(wav)

    private fun ditherInPlace(wav: File) {
        val bytes = wav.readBytes()
        require(bytes.size > 44) { "WAV 过小" }
        // 校验是 16-bit PCM 且 data 块从 44 开始（VoiceDesign 固定这么写）
        require(bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte()) { "不是 RIFF" }
        var seed = 0x5DEECE66DL
        var changed = 0
        var i = 44
        while (i + 1 < bytes.size) {
            seed = (seed * 6364136223846793005L + 1442695040888963407L)
            val d = ((seed ushr 33).toInt() and 1) * 2 - 1   // -1 或 +1
            val lo = bytes[i].toInt() and 0xFF
            val hi = bytes[i + 1].toInt()                  // 有符号高字节
            var v = (hi shl 8) or lo                        // int16
            if (v > 32767) v -= 65536
            var nv = v + d
            if (nv > 32767) nv = 32767
            if (nv < -32768) nv = -32768
            bytes[i] = (nv and 0xFF).toByte()
            bytes[i + 1] = ((nv shr 8) and 0xFF).toByte()
            if (nv != v) changed++
            i += 2
        }
        wav.writeBytes(bytes)
        Log.i(TAG, "dither 完成：$changed 个样本被修改（±1 LSB）")
    }

    private fun json(s: String): String {
        val sb = StringBuilder("\"")
        for (c in s) when (c) {
            '\"' -> sb.append("\\\"")
            '\\' -> sb.append("\\\\")
            '\n' -> sb.append("\\n")
            '\r' -> sb.append("\\r")
            '\t' -> sb.append("\\t")
            else -> if (c < ' ') sb.append("\\u%04x".format(c.code)) else sb.append(c)
        }
        return sb.append("\"").toString()
    }
}
