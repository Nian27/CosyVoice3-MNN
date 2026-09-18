package com.cosyvoice.app

import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import org.json.JSONObject

private data class VoiceEnrollmentRequest(
    val displayName: String,
    val promptText: String,
    val startSeconds: Double,
    val endSeconds: Double
)

class MainActivity : ComponentActivity() {

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val runtimeReleaseScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val store by lazy { CosyVoiceStore(this) }
    private var modelStatus by mutableStateOf(CosyVoiceModelStatus(false, 0L, emptyList()))
    private var enrollmentStatus by mutableStateOf(CosyVoiceModelStatus(false, 0L, emptyList()))
    private var voiceProfiles by mutableStateOf<List<CosyVoiceVoiceProfile>>(emptyList())
    private var selectedVoiceProfileId by mutableStateOf(CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID)
    private var busyMessage by mutableStateOf("")
    private var designProgress by mutableStateOf("")
    private var resultMessage by mutableStateOf("")
    private var selectedAudioUri by mutableStateOf<Uri?>(null)
    private var selectedAudioName by mutableStateOf("")
    private var mediaPlayer: MediaPlayer? = null
    private var pendingVoiceExportIds: List<String> = emptyList()

    private val modelZipLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::importModel)
    }
    private val modelExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let(::exportModel)
    }
    private val voiceProfileZipLauncher = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) importVoiceProfiles(uris)
    }
    private val voiceProfileExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        val ids = pendingVoiceExportIds
        pendingVoiceExportIds = emptyList()
        if (uri != null && ids.isNotEmpty()) exportVoiceProfiles(uri, ids)
    }
    private val enrollmentZipLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::importEnrollmentExtension)
    }
    private val enrollmentExportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let(::exportEnrollmentExtension)
    }
    private val referenceAudioLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching { contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            selectedAudioUri = uri
            selectedAudioName = displayName(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 设计音色路径不需要朗读模型；提前跳过 initialize 可以省下约 1.45 GB 常驻内存，
        // 否则「朗读模型 + VoiceDesign 4.7G」一起驻留会被 lmkd 杀掉（实测 MemFree 仅 ~200MB）。
        val designOnly = intent?.getBooleanExtra("autorunDesign", false) == true
        if (!designOnly) {
            CosyVoiceRuntime.initialize(this)
        } else {
            Log.i("CosyVoice", "autorunDesign：跳过 CosyVoiceRuntime.initialize 以省内存")
        }
        selectedVoiceProfileId = store.selectedVoiceProfileId()
        refresh()
        // 无 UI 闭环入口（adb 触发）：
        //   am start --ez autorunDesign true --es designDesc "…" --es designName "…"
        // 只跑 enrollment 的快速诊断入口：
        //   am start --ez enrollOnly true --es enrollWav "/path/to.wav"
        if (intent?.getBooleanExtra("enrollOnly", false) == true) {
            runEnrollOnly(intent?.getStringExtra("enrollWav") ?: "")
        }
        // 无 UI 合成触发（adb）：
        //   am start --ez autorunPreview true --es previewText "…" [--es previewVoice <profileId>]
        // 内存探针：合成一次 → 测 anon → close() → 再测 anon
        //   am start --ez closeProbe true
        // 参考音频克隆路径的无 UI 验收入口：
        //   am start --ez autorunClone true --es cloneWav /path/to.wav --es cloneText "对应文字" --es cloneName "音色名"
        // 走【和 UI 按钮完全相同】的路径（startDesignService → :design 独立进程），
        // 用于自动化验证真实用户路径，而不是 Activity 内的 runDesignClosedLoop。
        // 实验：把某个音色裁剪到实时长度（用于定位 Hexagon 连续解码能接受的 prompt 长度）
        // am start --ez autorunTrimVoice true --es trimVoiceId voice-xxx
        // 裁剪长度读 <模型目录>/rt_limit.txt
        if (intent?.getBooleanExtra("autorunTrimVoice", false) == true) {
            val vid = intent?.getStringExtra("trimVoiceId").orEmpty()
            Thread {
                val r = runCatching { store.createRealtimeVoiceProfile(vid) }
                    .fold({ "OK id=${it.id} tokens=${it.promptTokenCount} frames=${it.promptFrameCount}" },
                        { "ERR ${it.javaClass.simpleName}: ${it.message}" })
                android.util.Log.e("VD_BUILD", "TRIM_RESULT $r")
                runCatching { File(store.voiceDesignDir, "trim-result.txt").writeText(r + "\n") }
            }.start()
        }
        // GB-SPLIT-P0 探针：am start --ez autorunGb4l true --ei gb4lReps 10
        if (intent?.getBooleanExtra("autorunGb4l", false) == true) {
            val reps = intent?.getIntExtra("gb4lReps", 10) ?: 10
            Thread {
                val f = File(store.voiceDesignDir, "gb4l-result.txt")
                val r = runCatching { CosyVoiceVoiceDesignNative.nativeGb4lProbe(store.voiceDesignDir.absolutePath, reps) }
                    .getOrElse { "ERR " + it.javaClass.simpleName + ": " + it.message }
                android.util.Log.e("VD_BUILD", "GB4L_RESULT " + r)
                runCatching { f.writeText(r + "\n") }
            }.start()
        }
        // 【2026-09-18】串行化验证入口：模拟用户连点 N 次（走的是和生产按钮完全相同的
        // startDesignService 路径，而不是 am start 新 Activity —— 后者到已运行的 Activity
        // 不会重跑 onCreate，根本到不了 startDesignService）。
        // am start --ez autorunDesignBurst true --ei burstCount 3
        if (intent?.getBooleanExtra("autorunDesignBurst", false) == true) {
            val count = intent?.getIntExtra("burstCount", 3) ?: 3
            val burstDesc = intent?.getStringExtra("designDesc")
                ?: "青年女性，声音清冷柔和，语速偏慢，带一点疏离感"
            scope.launch {
                repeat(count) { i ->
                    startDesignService("BURST" + (i + 1), burstDesc,
                        CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT, "opencl")
                    if (i < count - 1) delay(1500)
                }
            }
        }
        if (intent?.getBooleanExtra("autorunDesignService", false) == true) {
            startDesignService(
                name = intent?.getStringExtra("designName") ?: "服务路径测试",
                description = intent?.getStringExtra("designDesc")
                    ?: "青年女性，声音清冷柔和，语速偏慢，带一点疏离感",
                referenceText = CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT,
                backend = intent?.getStringExtra("designBackend") ?: "opencl")
        }
        if (intent?.getBooleanExtra("autorunClone", false) == true) {
            runCloneClosedLoop(
                wavPath = intent?.getStringExtra("cloneWav") ?: "",
                promptText = intent?.getStringExtra("cloneText") ?: "风从北边吹过来，带着一点凉意，远处那盏灯忽明忽暗。",
                displayName = intent?.getStringExtra("cloneName") ?: "克隆测试音色")
        }
        if (intent?.getBooleanExtra("closeProbe", false) == true) {
            val lf = File(store.voiceDesignDir, "close-probe.log")
            fun plog(s: String) { Log.i("CosyVoiceDesign", s); runCatching { lf.appendText(s + "\n") } }
            runCatching { lf.writeText("") }
            fun anonKb(): Long = try {
                File("/proc/self/status").readLines().firstOrNull { it.startsWith("RssAnon:") }
                    ?.filter { it.isDigit() }?.toLong() ?: -1L
            } catch (_: Throwable) { -1L }
            fun rssKb(): Long = try {
                File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmRSS:") }
                    ?.filter { it.isDigit() }?.toLong() ?: -1L
            } catch (_: Throwable) { -1L }
            Thread {
                try {
                    plog("=== close 探针 ===")
                    plog("A 启动时     rss=" + rssKb() + " anon=" + anonKb())
                    CosyVoiceRuntime.ensureInitialized(this)
                    plog("B initialize后 rss=" + rssKb() + " anon=" + anonKb())
                    val out = File(cacheDir, "probe/p.wav"); out.parentFile?.mkdirs()
                    kotlinx.coroutines.runBlocking {
                        CosyVoiceRuntime.synthesize(store, "测试。", out,
                            CosyVoiceSynthesisOptions(
                                hardwarePlan = CosyVoiceRuntime.recommendedHardwarePlan(),
                                voiceProfileId = selectedVoiceProfileId)) { }
                    }
                    plog("C 合成后     rss=" + rssKb() + " anon=" + anonKb())
                    kotlinx.coroutines.runBlocking { CosyVoiceRuntime.close() }
                    System.gc(); Thread.sleep(1500); System.gc(); Thread.sleep(3000)
                    plog("D close后    rss=" + rssKb() + " anon=" + anonKb())
                    plog("=== 结论: close 释放了 " + ((anonKb() - 0)) + " kB (看 C→D 差) ===")
                } catch (t: Throwable) {
                    plog("探针失败: " + t.javaClass.simpleName + ": " + t.message)
                }
            }.start()
        }

        if (intent?.getBooleanExtra("autorunPreview", false) == true) {
            val text = intent?.getStringExtra("previewText") ?: "你好，欢迎使用阅读。"
            val vid = intent?.getStringExtra("previewVoice")
            if (!vid.isNullOrBlank()) { selectedVoiceProfileId = vid; store.selectVoiceProfile(vid) }
            runPreviewClosedLoop(text)
        }
        if (intent?.getBooleanExtra("autorunDesign", false) == true) {
            startDesignService(
                name = intent?.getStringExtra("designName") ?: "文字设计音色",
                description = intent?.getStringExtra("designDesc")
                    ?: CosyVoiceVoiceDesigner.DEFAULT_DESCRIPTION,
                referenceText = intent?.getStringExtra("designRefText")
                    ?: CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT,
                backend = intent?.getStringExtra("designBackend") ?: "opencl")
        }
        setContentView(ComposeView(this).apply {
            setContent {
                CosyVoiceManagerScreen(
                    modelStatus, enrollmentStatus, voiceProfiles, selectedVoiceProfileId,
                    selectedAudioName, busyMessage, resultMessage, designProgress,
                    onBack = { finish() },
                    onDownloadModel = { downloadModel() },
                    onImportModel = { modelZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportModel = { modelExportLauncher.launch("cosyvoice3-mnn-complete.zip") },
                    onDeleteModel = { deleteModel() },
                    onImportVoiceProfile = { voiceProfileZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportVoiceProfile = { profile -> startVoiceExport(listOf(profile.id), "${profile.displayName}.zip") },
                    onExportAllVoiceProfiles = {
                        val ids = voiceProfiles.filterNot { it.builtIn }.map { it.id }
                        startVoiceExport(ids, "cosyvoice3-voices.zip")
                    },
                    onVoiceProfileSelected = { selectVoiceProfile(it) },
                    onCreateVoiceByDesign = { name, desc, refText -> createVoiceByDesign(name, desc, refText) },
                    onCreateRealtimeVoiceProfile = { createRealtimeVoiceProfile(it) },
                    onDeleteVoiceProfile = { deleteVoiceProfile(it) },
                    onImportEnrollment = { enrollmentZipLauncher.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onExportEnrollment = { enrollmentExportLauncher.launch("cosyvoice3-mnn-enrollment-extension.zip") },
                    onDeleteEnrollment = { deleteEnrollmentExtension() },
                    onSelectReferenceAudio = { referenceAudioLauncher.launch(arrayOf("audio/*")) },
                    onCreateVoice = { createVoiceProfile(it) },
                    onPreview = { text, options -> preview(text, options) }
                )
            }
        })
    }

    override fun onDestroy() {
        mediaPlayer?.release()
        scope.cancel()
        runtimeReleaseScope.launch {
            try { CosyVoiceRuntime.close() } finally { runtimeReleaseScope.cancel() }
        }
        super.onDestroy()
    }

    private fun refresh() {
        modelStatus = store.modelStatus()
        enrollmentStatus = store.enrollmentStatus()
        voiceProfiles = if (modelStatus.ready) store.voiceProfiles() else emptyList()
        if (modelStatus.ready && voiceProfiles.none { it.id == selectedVoiceProfileId }) {
            selectedVoiceProfileId = CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID
            store.selectVoiceProfile(selectedVoiceProfileId)
        }
    }

    private fun selectVoiceProfile(id: String) { store.selectVoiceProfile(id); selectedVoiceProfileId = id }

    private fun importEnrollmentExtension(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查音色创建扩展"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.importEnrollmentZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { refresh(); resultMessage = "音色创建扩展导入完成 · ${formatBytes(enrollmentStatus.installedBytes)}" }
                .onFailure { resultMessage = "音色创建扩展导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "enrollment import failed", it) }
            busyMessage = ""
        }
    }

    /**
     * 界面入口：文字设计音色 -> reference.wav -> enroll -> VoiceProfile。
     * 与无 UI 闭环走的是同一个编排器，只是带进度回调。
     */
    private fun createVoiceByDesign(name: String, description: String, referenceText: String) {
        if (busyMessage.isNotBlank()) return
        startDesignService(name, description, referenceText, "opencl")
    }

    /**
     * 设计进度轮询。
     *
     * 一次设计要 4-9 分钟（decode ~1-3 分钟、decoder ~7 分钟），之前 UI 上只有
     * 「正在用描述生成音色…」这一句，用户看不到任何进展，会以为卡死。
     * native 侧每帧写 <模型目录>/progress.txt，这里每秒读一次显示出来。
     */
    private var designPollJob: kotlinx.coroutines.Job? = null

    private fun startDesignProgressPolling() {
        designPollJob?.cancel()
        val f = File(store.voiceDesignDir, "progress.txt")
        val state = File(store.voiceDesignDir, "design-state.txt")
        val seqFile = File(store.voiceDesignDir, "done-seq.txt")
        fun readSeq(): Int = runCatching {
            seqFile.takeIf(File::isFile)?.readText()?.trim()?.split(' ')?.firstOrNull()?.toIntOrNull() ?: 0
        }.getOrDefault(0)
        val seq0 = readSeq()   // 本轮开始前的完成序号
        designPollJob = scope.launch {
            val deadline = System.currentTimeMillis() + 15 * 60 * 1000L
            while (isActive) {
                val txt = runCatching { if (f.isFile) f.readText().trim() else "" }.getOrDefault("")
                if (txt.isNotBlank()) designProgress = txt
                // 【2026-09-18】权威判据：完成序号变了 = 本轮已结束（成功或失败）。
                // 不能只看 progress.txt 的文本：下一个排队任务一开头就 writeProgress("")，
                // UI 每秒轮询很可能错过「完成」，结果新音色一直不出现。
                if (readSeq() != seq0) {
                    refresh()
                    delay(1500)
                    designProgress = ""
                    break
                }
                // native / Service 写的终态文本（正常情况下序号判据会先命中，这是快速路径）
                if (txt.startsWith("完成")) {
                    refresh()
                    delay(3000)
                    designProgress = ""
                    break
                }
                if (txt.startsWith("失败")) {
                    resultMessage = txt
                    designProgress = ""
                    refresh()
                    break
                }
                // 【2026-09-18】进程消失兜底：:design 已 idle 却始终没有终态 → 判定异常。
                // 不要求 txt 非空 —— progress.txt 可能已被下一轮清空。
                val st = runCatching { if (state.isFile) state.readText().trim() else "" }.getOrDefault("")
                if (st.startsWith("idle")) {
                    if (txt.isNotBlank()) {
                        resultMessage = "设计进程已退出但未报告完成，请查看 design-loop.log（最后进度：$txt）"
                    }
                    designProgress = ""
                    refresh()
                    break
                }
                // 【2026-09-18】超时兜底：任何情况下都不允许永久轮询。
                if (System.currentTimeMillis() > deadline) {
                    resultMessage = "设计超过 15 分钟未结束，请查看 design-loop.log（最后进度：$txt）"
                    designProgress = ""
                    refresh()
                    break
                }
                delay(1000)
            }
        }
    }

    /** 启动前台服务跑设计（切后台也不会被冻结）。 */
    private fun startDesignService(name: String, description: String, referenceText: String, backend: String) {
        val i = Intent(this, CosyVoiceDesignService::class.java)
            .putExtra(CosyVoiceDesignService.EXTRA_NAME, name)
            .putExtra(CosyVoiceDesignService.EXTRA_DESC, description)
            .putExtra(CosyVoiceDesignService.EXTRA_TEXT, referenceText)
            .putExtra(CosyVoiceDesignService.EXTRA_BACKEND, backend)
        // 【2026-09-18】串行化的权威在 Service（单槽排队）；这里只做提示，不阻断 ——
        // 用户连点时的意图通常是"用最新参数重来"，交给 Service 覆盖排队即可。
        val alreadyRunning = designProgress.isNotBlank()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(i) else startService(i)
        designProgress = "已提交，正在启动…"
        startDesignProgressPolling()
        resultMessage = if (alreadyRunning) {
            "已有设计在运行，本次请求已排队（会覆盖之前排队的请求）。"
        } else {
            "已开始设计音色：约 4-9 分钟（解码 1-3 分钟 + 波形 7 分钟）。可切到其它 App，进度见下方。"
        }
    }

    @Suppress("unused")
    private fun createVoiceByDesignInline(name: String, description: String, referenceText: String) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在用描述生成音色（首次需编译图，约 4-6 分钟）"
            resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    CosyVoiceRuntime.close()
                    withContext(Dispatchers.Main.immediate) { busyMessage = "正在用描述生成音色…" }
                    CosyVoiceVoiceDesigner.createFromDesign(
                        store = store,
                        nativeLibDir = applicationInfo.nativeLibraryDir,
                        designModelsDir = store.voiceDesignDir,
                        description = description,
                        displayName = name,
                        backend = "opencl",
                        referenceText = referenceText
                    ) { stage ->
                        // 编排器在主线程外回调，这里直接更新（Compose 会重组）
                        busyMessage = stage
                    }
                }
            }.onSuccess { profile ->
                busyMessage = ""
                refresh()
                selectVoiceProfile(profile.id)
                resultMessage = buildString {
                    append("音色设计完成：").append(profile.displayName)
                    append(" · ").append(profile.promptTokenCount).append(" Token")
                    append("\n已自动选中，可直接在下方「合成并试听」。")
                    append("\n参考音频与描述已保留在音色目录，将来重新注册不必再生成。")
                }
            }.onFailure {
                busyMessage = ""
                resultMessage = "音色设计失败：" + it.localizedMessage.orEmpty()
                Log.e("CosyVoice", "voice design failed", it)
            }
        }
    }

    /** 无 UI 合成闭环：用选中音色合成一句话，结果写 synthesis-loop.log。 */
    private fun runPreviewClosedLoop(text: String) {
        val logFile = File(store.voiceDesignDir, "synthesis-loop.log")
        fun log(s: String) { Log.i("CosyVoiceDesign", s); runCatching { logFile.appendText(s + "\n") } }
        runCatching { logFile.writeText("") }
        if (!modelStatus.ready) { log("朗读模型未就绪: " + modelStatus.missingFiles.joinToString(",")); return }
        CosyVoiceRuntime.ensureInitialized(this)
        Thread {
            val t0 = android.os.SystemClock.elapsedRealtime()
            try {
                log("=== 无 UI 合成开始 ===")
                log("音色: " + selectedVoiceProfileId)
                log("文本: " + text)
                val out = File(cacheDir, "cosyvoice-preview/autorun-" + t0 + ".wav")
                out.parentFile?.mkdirs()
                val report = kotlinx.coroutines.runBlocking {
                    CosyVoiceRuntime.synthesize(store, text, out,
                        CosyVoiceSynthesisOptions(
                            hardwarePlan = CosyVoiceRuntime.recommendedHardwarePlan(),
                            voiceProfileId = selectedVoiceProfileId)) { stage -> log("[进度] " + stage) }
                }
                log("合成完成: " + report.displayText())
                log("输出: " + report.output.absolutePath + " bytes=" + report.output.length())
                runCatching {
                    val keep = File(store.voiceDesignDir, "last-preview.wav")
                    report.output.copyTo(keep, overwrite = true)
                    log("已保存: " + keep.absolutePath + " bytes=" + keep.length())
                }
                log("=== 合成成功，耗时 " + (android.os.SystemClock.elapsedRealtime() - t0) + " ms ===")
            } catch (t: Throwable) {
                log("=== 合成失败: " + t.javaClass.simpleName + ": " + t.message + " ===")
                Log.e("CosyVoiceDesign", "preview failed", t)
            }
        }.start()
    }

    /**
     * 参考音频克隆路径的无 UI 闭环 —— 与「文字设计音色」共用同一个 installEnrolledVoiceProfile()。
     * 这条路径此前从未跑通过（更早被 JNI 符号问题挡住），需要独立验收。
     */
    private fun runCloneClosedLoop(wavPath: String, promptText: String, displayName: String) {
        val logFile = File(store.voiceDesignDir, "clone-loop.log")
        fun log(s: String) { Log.i("CosyVoiceDesign", s); runCatching { logFile.appendText(s + "\n") } }
        runCatching { logFile.writeText("") }
        CosyVoiceRuntime.ensureInitialized(this)
        Thread {
            val t0 = android.os.SystemClock.elapsedRealtime()
            try {
                log("=== 参考音频克隆闭环开始 ===")
                log("音色名: " + displayName)
                log("参考文字: " + promptText)
                val wav = File(wavPath)
                log("参考音频: " + wav.absolutePath + " exists=" + wav.isFile + " bytes=" + wav.length())
                if (!wav.isFile) { log("参考音频不存在"); return@Thread }
                if (!store.modelStatus().ready) { log("朗读模型未就绪"); return@Thread }
                if (!store.enrollmentStatus().ready) { log("音色创建扩展未就绪"); return@Thread }
                // 与 UI 克隆路径一致：先补 dither，避免合成音频的「精确 0 静音」把 campplus 打成 NaN
                val dithered = File(store.workDir, "clone-dither.wav")
                wav.copyTo(dithered, overwrite = true)
                CosyVoiceVoiceDesigner.applyDitherForTest(dithered)
                log("已补 dither -> " + dithered.absolutePath)
                val out = File(store.workDir, "clone-out").apply { deleteRecursively(); mkdirs() }
                val code = CosyVoiceEnrollmentNative.enroll(
                    tokenizerModelPath = store.enrollmentFile("speech-tokenizer-v3.fp32.inline.mnn").absolutePath,
                    campPlusModelPath = store.enrollmentFile("campplus.fp32.mnn").absolutePath,
                    affineWeightPath = store.enrollmentFile("flow-speaker-affine-weight.bin").absolutePath,
                    affineBiasPath = store.enrollmentFile("flow-speaker-affine-bias.bin").absolutePath,
                    sourceWavPath = dithered.absolutePath,
                    outputDirectory = out.absolutePath,
                    threads = 6)
                log("enroll exitCode=" + code + " (" + CosyVoiceEnrollmentNative.errorMessage(code) + ")")
                if (code != 0) { log("=== 克隆失败 ==="); return@Thread }
                val profile = store.installEnrolledVoiceProfile(
                    displayName = displayName,
                    promptText = promptText,
                    nativeOutputDirectory = out,
                    sourceWav = wav)
                log("音色已注册: " + profile.id + " / " + profile.displayName +
                    " / tokens=" + profile.promptTokenCount + " / frames=" + profile.promptFrameCount)
                log("=== 克隆成功，耗时 " + (android.os.SystemClock.elapsedRealtime() - t0) + " ms ===")
            } catch (t: Throwable) {
                log("=== 克隆失败: " + t.javaClass.simpleName + ": " + t.message + " ===")
                Log.e("CosyVoiceDesign", "clone loop failed", t)
            }
        }.start()
    }

    /** 只跑 enrollment 的诊断入口（不跑 VoiceDesign），用于隔离问题。 */
    private fun runEnrollOnly(wavPath: String) {
        val logFile = File(store.voiceDesignDir, "enroll-only.log")
        fun log(s: String) {
            Log.i("CosyVoiceDesign", s)
            runCatching { logFile.appendText(s + "\n") }
        }
        runCatching { logFile.writeText("") }
        Thread {
            try {
                val wav = File(wavPath)
                log("=== enroll-only 开始 ===")
                log("wav=" + wav.absolutePath + " exists=" + wav.isFile + " bytes=" + wav.length())
                if (!wav.isFile) { log("WAV 不存在"); return@Thread }
                // 打印 WAV 头，确认格式
                val head = ByteArray(44)
                wav.inputStream().use { it.read(head) }
                log("header=" + head.joinToString("") { "%02x".format(it) })
                // 复制一份并加 dither（与 design 路径一致），避免污染原文件
                val dithered = File(store.workDir, "enroll-only-dither.wav")
                dithered.writeBytes(wav.readBytes())
                CosyVoiceVoiceDesigner.applyDitherForTest(dithered)
                log("已加 dither -> " + dithered.absolutePath + " bytes=" + dithered.length())
                val out = File(store.workDir, "enroll-only-out").apply { deleteRecursively(); mkdirs() }
                val code = CosyVoiceEnrollmentNative.enroll(
                    tokenizerModelPath = store.enrollmentFile("speech-tokenizer-v3.fp32.inline.mnn").absolutePath,
                    campPlusModelPath = store.enrollmentFile("campplus.fp32.mnn").absolutePath,
                    affineWeightPath = store.enrollmentFile("flow-speaker-affine-weight.bin").absolutePath,
                    affineBiasPath = store.enrollmentFile("flow-speaker-affine-bias.bin").absolutePath,
                    sourceWavPath = dithered.absolutePath,
                    outputDirectory = out.absolutePath,
                    threads = 6)
                log("exitCode=" + code + " (" + CosyVoiceEnrollmentNative.errorMessage(code) + ")")
                log("输出文件: " + out.list()?.joinToString(", "))
                out.listFiles()?.forEach { log("  " + it.name + " = " + it.length() + " bytes") }
                log("=== enroll-only 结束 ===")
            } catch (t: Throwable) {
                log("=== enroll-only 异常: " + t.javaClass.simpleName + ": " + t.message + " ===")
            }
        }.start()
    }

    /**
     * 无 UI 闭环：文字设计音色 -> reference.wav -> enroll -> VoiceProfile -> 用新音色合成一句。
     * 进度写 files/cosyvoice3-mnn/design-loop.log，方便 adb 取回。
     */
    private fun runDesignClosedLoop(description: String, displayName: String, backend: String) {
        val logFile = File(store.voiceDesignDir, "design-loop.log")
        fun log(s: String) {
            Log.i("CosyVoiceDesign", s)
            runCatching { logFile.appendText(s + "\n") }
        }
        runCatching { logFile.writeText("") }
        Thread {
            val t0 = System.currentTimeMillis()
            try {
                log("=== 无 UI 闭环开始 ===")
                log("描述: " + description)
                log("试听文本: " + CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT)
                log("后端: " + backend)
                val profile = CosyVoiceVoiceDesigner.createFromDesign(
                    store = store,
                    nativeLibDir = applicationInfo.nativeLibraryDir,
                    designModelsDir = store.voiceDesignDir,
                    description = description,
                    displayName = displayName,
                    backend = backend
                ) { log("[进度] " + it) }
                log("音色已注册: " + profile.id + " / " + profile.displayName +
                    " / tokens=" + profile.promptTokenCount + " / frames=" + profile.promptFrameCount)
                log("音色目录: " + profile.directory.absolutePath)
                log("目录内容: " + profile.directory.list()?.joinToString(", "))
                log("=== 闭环成功，总耗时 " + (System.currentTimeMillis() - t0) + " ms ===")
            } catch (t: Throwable) {
                log("=== 闭环失败: " + t.javaClass.simpleName + ": " + t.message + " ===")
                Log.e("CosyVoiceDesign", "design closed loop failed", t)
            }
        }.start()
    }

    private fun createVoiceProfile(request: VoiceEnrollmentRequest) {
        if (busyMessage.isNotBlank()) return
        val duration = request.endSeconds - request.startSeconds
        if (duration !in 3.0..5.0) { resultMessage = "当前片段 %.1f 秒，请调整为 3-5 秒后再创建".format(duration); return }
        val uri = selectedAudioUri
        if (uri == null) { resultMessage = "请先选择 MP3 或其他音频文件"; return }
        if (!modelStatus.ready || !enrollmentStatus.ready) { resultMessage = "请先导入 MNN 朗读模型和音色创建扩展"; return }
        scope.launch {
            busyMessage = "正在释放朗读模型，为音色注册腾出内存"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    CosyVoiceRuntime.close()
                    val runDirectory = File(store.workDir, "enroll-${System.currentTimeMillis()}").apply { deleteRecursively(); mkdirs() }
                    try {
                        val sourceWav = File(runDirectory, "source.wav")
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在解码并截取参考音频" }
                        val decoded = CosyVoiceAudioDecoder.decodeSegmentToWav(this@MainActivity, uri, request.startSeconds, request.endSeconds, sourceWav)
                        // 关键：enrollment 的 CAMPPlus 前置 fbank 会取 log(能量)，
                        // 而合成音频（含 VoiceDesign 产物）的静音是「精确 0 样本」-> log(0) = -inf
                        // -> NaN 传播 -> campplus 输出非有限 -> enroll 返回 37「说话人特征无效」。
                        // 真实录音有本底噪声不会踩到；但用户完全可能把合成音频当参考导入，
                        // 所以这里无条件叠加 ±1 LSB（约 -90 dB，听不见）的确定性 dither。
                        CosyVoiceVoiceDesigner.applyDitherForTest(sourceWav)
                        val nativeOutput = File(runDirectory, "profile").apply { mkdirs() }
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在提取语音 Token 和说话人特征" }
                        val exitCode = CosyVoiceEnrollmentNative.enroll(
                            tokenizerModelPath = store.enrollmentFile("speech-tokenizer-v3.fp32.inline.mnn").absolutePath,
                            campPlusModelPath = store.enrollmentFile("campplus.fp32.mnn").absolutePath,
                            affineWeightPath = store.enrollmentFile("flow-speaker-affine-weight.bin").absolutePath,
                            affineBiasPath = store.enrollmentFile("flow-speaker-affine-bias.bin").absolutePath,
                            sourceWavPath = sourceWav.absolutePath,
                            outputDirectory = nativeOutput.absolutePath, threads = 6)
                        check(exitCode == 0) { "${CosyVoiceEnrollmentNative.errorMessage(exitCode)}($exitCode)" }
                        val report = JSONObject(File(nativeOutput, "enrollment-report.json").readText())
                        val profile = store.installEnrolledVoiceProfile(request.displayName, request.promptText, nativeOutput, sourceWav)
                        Triple(profile, decoded, report)
                    } finally { runDirectory.deleteRecursively() }
                }
            }.onSuccess { (profile, decoded, report) ->
                refresh(); selectVoiceProfile(profile.id)
                resultMessage = buildString {
                    append("音色创建完成：${profile.displayName}")
                    append(" · %.2f 秒 · %d Token".format(decoded.durationSeconds, profile.promptTokenCount))
                    append("\n注册耗时 %.2f 秒；已自动选中，可直接合成试听。".format(report.optDouble("totalMs") / 1000.0))
                }
            }.onFailure { resultMessage = "音色创建失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice enrollment failed", it) }
            busyMessage = ""
        }
    }

    private fun deleteEnrollmentExtension() {
        if (busyMessage.isNotBlank()) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除音色创建扩展？").setMessage("将删除约 490 MB 的注册模型。已创建的音色和朗读模型都会保留。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ -> store.deleteEnrollment(); refresh(); resultMessage = "音色创建扩展已删除，现有音色仍可继续使用" }.show()
    }

    private fun exportEnrollmentExtension(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在准备导出音色创建扩展"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportEnrollmentZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "音色创建扩展已导出，可在本页重新导入" }
                .onFailure { resultMessage = "创建扩展导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "enrollment export failed", it) }
            busyMessage = ""
        }
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) return cursor.getString(0).orEmpty()
        }
        return uri.lastPathSegment.orEmpty().ifBlank { "已选择音频" }
    }

    private fun importVoiceProfiles(uris: List<Uri>) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查零样本音色档案"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    uris.flatMapIndexed { index, uri ->
                        withContext(Dispatchers.Main.immediate) { busyMessage = "正在导入音色包 ${index + 1}/${uris.size}" }
                        store.importVoiceProfilesZip(uri)
                    }
                }
            }.onSuccess { profiles ->
                refresh(); profiles.lastOrNull()?.let { selectVoiceProfile(it.id) }
                resultMessage = "已导入 ${profiles.size} 个音色" + profiles.lastOrNull()?.let { "，当前：${it.displayName}" }.orEmpty()
            }.onFailure { resultMessage = "音色导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice profile import failed", it) }
            busyMessage = ""
        }
    }

    private fun startVoiceExport(ids: List<String>, fileName: String) {
        if (busyMessage.isNotBlank()) return
        if (ids.isEmpty()) { resultMessage = "没有可导出的自定义音色"; return }
        pendingVoiceExportIds = ids; voiceProfileExportLauncher.launch(fileName.replace(Regex("[\\/:*?\"<>|]"), "_"))
    }

    private fun exportVoiceProfiles(uri: Uri, ids: List<String>) {
        scope.launch {
            busyMessage = "正在准备导出音色"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportVoiceProfilesZip(uri, ids) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "已导出 ${ids.size} 个音色，可在本页批量导入" }
                .onFailure { resultMessage = "音色导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "voice profile export failed", it) }
            busyMessage = ""
        }
    }

    private fun createRealtimeVoiceProfile(profile: CosyVoiceVoiceProfile) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在生成 125 Token 实时音色"; resultMessage = ""
            runCatching { withContext(Dispatchers.IO) { store.createRealtimeVoiceProfile(profile.id) } }
                .onSuccess { created -> refresh(); selectVoiceProfile(created.id); resultMessage = "实时音色已生成并选中：${created.displayName} · ${created.promptTokenCount} Token" }
                .onFailure { resultMessage = "实时音色生成失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "realtime voice profile failed", it) }
            busyMessage = ""
        }
    }

    private fun importModel(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在检查 MNN 模型包"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.importModelZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { refresh(); resultMessage = "MNN 模型导入完成" }
                .onFailure { resultMessage = "模型导入失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "model import failed", it) }
            busyMessage = ""
        }
    }

    private fun downloadModel() {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在准备 Hugging Face 模型下载"
            resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    CosyVoiceModelDownloader(this@MainActivity, store).download(
                        onProgress = { progress ->
                            scope.launch {
                                busyMessage =
                                    "正在下载模型 ${progress.percent}% · ${formatBytes(progress.totalBytes)} / ${formatBytes(progress.totalExpectedBytes)}"
                            }
                        },
                        onStage = { stage -> scope.launch { busyMessage = stage } }
                    )
                }
            }.onSuccess {
                refresh()
                resultMessage = "MNN 模型在线安装完成"
            }.onFailure {
                resultMessage = "模型下载失败：${it.localizedMessage.orEmpty()}"
                Log.e("CosyVoice", "model download failed", it)
            }
            busyMessage = ""
        }
    }

    private fun preview(text: String, options: CosyVoiceSynthesisOptions) {
        if (busyMessage.isNotBlank()) return
        if (!modelStatus.ready) { resultMessage = "请先导入完整 MNN 模型包"; return }
        val cleanText = text.trim()
        if (cleanText.isBlank()) { resultMessage = "请输入试听文字"; return }
        // 设计路径会跳过 onCreate 的 initialize 以省内存，所以这里在真正合成前补上（幂等）
        CosyVoiceRuntime.ensureInitialized(this)
        scope.launch {
            mediaPlayer?.release(); mediaPlayer = null
            busyMessage = "正在启动 MNN 单句链路"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) {
                    val output = File(cacheDir, "cosyvoice-preview/${cleanText.hashCode()}-${options.hashCode()}.wav")
                    CosyVoiceRuntime.synthesize(store, cleanText, output, options) { stage ->
                        withContext(Dispatchers.Main.immediate) { busyMessage = stage }
                    }
                }
            }.onSuccess { report ->
                resultMessage = report.displayText()
                mediaPlayer = MediaPlayer().apply {
                    setDataSource(report.output.absolutePath)
                    setOnPreparedListener { it.start() }
                    setOnCompletionListener { it.release(); if (mediaPlayer === it) mediaPlayer = null }
                    setOnErrorListener { player, what, extra -> resultMessage = "播放失败：$what/$extra"; player.release(); if (mediaPlayer === player) mediaPlayer = null; true }
                    prepareAsync()
                }
            }.onFailure { resultMessage = "试听失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "preview failed", it) }
            busyMessage = ""
        }
    }

    private fun exportModel(uri: Uri) {
        if (busyMessage.isNotBlank()) return
        scope.launch {
            busyMessage = "正在准备导出 MNN 模型"; resultMessage = ""
            runCatching {
                withContext(Dispatchers.IO) { store.exportModelZip(uri) { message -> scope.launch { busyMessage = message } } }
            }.onSuccess { resultMessage = "MNN 模型已导出，可用同一页面的导入 ZIP 恢复" }
                .onFailure { resultMessage = "模型导出失败：${it.localizedMessage.orEmpty()}"; Log.e("CosyVoice", "model export failed", it) }
            busyMessage = ""
        }
    }

    private fun deleteModel() {
        if (busyMessage.isNotBlank()) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除本地模型？").setMessage("将删除约 1.3 GB 的 CosyVoice3 模型和 GPU 编译缓存。删除后必须重新导入完整 ZIP 才能试听。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ -> performDeleteModel() }.show()
    }

    private fun performDeleteModel() {
        scope.launch {
            busyMessage = "正在删除 MNN 模型"
            withContext(Dispatchers.IO) { CosyVoiceRuntime.close(); store.deleteModel() }
            refresh(); busyMessage = ""; resultMessage = "模型与 GPU 编译缓存已删除"
        }
    }

    private fun deleteVoiceProfile(profile: CosyVoiceVoiceProfile) {
        if (busyMessage.isNotBlank() || profile.builtIn) return
        android.app.AlertDialog.Builder(this)
            .setTitle("删除音色？").setMessage("将删除\"${profile.displayName}\"的本地音色档案。MNN 模型不会被删除。")
            .setNegativeButton("取消", null).setPositiveButton("确认删除") { _, _ ->
                scope.launch {
                    busyMessage = "正在删除音色"
                    runCatching { withContext(Dispatchers.IO) { store.deleteVoiceProfile(profile.id) } }
                        .onSuccess { selectVoiceProfile(CosyVoiceStore.DEFAULT_VOICE_PROFILE_ID); refresh(); resultMessage = "音色已删除：${profile.displayName}" }
                        .onFailure { resultMessage = "音色删除失败：${it.localizedMessage.orEmpty()}" }
                    busyMessage = ""
                }
            }.show()
    }
}

@Composable
private fun CosyVoiceManagerScreen(
    modelStatus: CosyVoiceModelStatus, enrollmentStatus: CosyVoiceModelStatus,
    voiceProfiles: List<CosyVoiceVoiceProfile>, selectedVoiceProfileId: String,
    selectedAudioName: String, busyMessage: String, resultMessage: String, designProgress: String,
    onBack: () -> Unit, onDownloadModel: () -> Unit, onImportModel: () -> Unit,
    onExportModel: () -> Unit, onDeleteModel: () -> Unit,
    onImportVoiceProfile: () -> Unit, onExportVoiceProfile: (CosyVoiceVoiceProfile) -> Unit,
    onExportAllVoiceProfiles: () -> Unit, onVoiceProfileSelected: (String) -> Unit,
    onCreateRealtimeVoiceProfile: (CosyVoiceVoiceProfile) -> Unit, onDeleteVoiceProfile: (CosyVoiceVoiceProfile) -> Unit,
    onImportEnrollment: () -> Unit, onExportEnrollment: () -> Unit, onDeleteEnrollment: () -> Unit,
    onSelectReferenceAudio: () -> Unit, onCreateVoice: (VoiceEnrollmentRequest) -> Unit,
    onCreateVoiceByDesign: (String, String, String) -> Unit,
    onPreview: (String, CosyVoiceSynthesisOptions) -> Unit
) {
    var previewText by androidx.compose.runtime.remember { mutableStateOf("你好，欢迎使用阅读，这是手机 MNN 本地合成测试。") }
    val hardwarePlan = CosyVoiceRuntime.recommendedHardwarePlan()
    var inferenceMode by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceInferenceMode.ZERO_SHOT) }
    var instruction by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceInstruction.DEFAULT_INSTRUCTION) }
    var voiceName by androidx.compose.runtime.remember { mutableStateOf("") }
    var promptText by androidx.compose.runtime.remember { mutableStateOf("") }
    var segmentStart by androidx.compose.runtime.remember { mutableStateOf("0") }
    var segmentEnd by androidx.compose.runtime.remember { mutableStateOf("5") }
    val segmentDuration = segmentStart.toDoubleOrNull()?.let { start -> segmentEnd.toDoubleOrNull()?.minus(start) }
    // 音色创建方式：0 = 参考音频克隆，1 = 文字设计音色
    var voiceCreateMode by androidx.compose.runtime.remember { mutableStateOf(0) }
    var designDescription by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceVoiceDesigner.DEFAULT_DESCRIPTION) }
    var designAdvanced by androidx.compose.runtime.remember { mutableStateOf(false) }
    var designReferenceText by androidx.compose.runtime.remember { mutableStateOf(CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT) }

    Column(Modifier.fillMaxSize().statusBarsPadding().background(MaterialTheme.colorScheme.background)) {
        Surface(shadowElevation = 2.dp) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onBack) { Text("返回") }
                Text("CosyVoice3-MNN", fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {

            // --- MNN Model Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("MNN 模型", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text(
                        if (modelStatus.ready) "已安装 · ${formatBytes(modelStatus.installedBytes)} · 24 kHz 单声道" else "未就绪 · 缺少 ${modelStatus.missingFiles.size} 个文件",
                        color = if (modelStatus.ready) Color(0xFF228B22) else MaterialTheme.colorScheme.error)
                    if (modelStatus.missingFiles.isNotEmpty()) Text(modelStatus.missingFiles.joinToString("\n"), fontSize = 12.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = onDownloadModel, enabled = busyMessage.isBlank()) {
                            Text("在线安装")
                        }
                        Button(onClick = onImportModel, enabled = busyMessage.isBlank()) { Text("导入 ZIP") }
                        OutlinedButton(onClick = onExportModel, enabled = busyMessage.isBlank() && modelStatus.ready) { Text("导出 ZIP") }
                    }
                    if (modelStatus.installedBytes > 0L) OutlinedButton(onClick = onDeleteModel, enabled = busyMessage.isBlank()) { Text("删除模型") }
                }
            }

            // --- Automatic Hardware Scheduling Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("自动硬件调度", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text(hardwarePlan.summary(), fontSize = 13.sp)
                    Text(hardwarePlan.npuStatus, fontSize = 12.sp)
                    Text(
                        "后端由机型验证结果自动选择；NPU 输出异常会整句回退 CPU。",
                        fontSize = 12.sp
                    )
                }
            }

            // --- Voice & Synthesis Card ---
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("音色与演绎", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text("零样本音色", fontSize = 13.sp)
                    voiceProfiles.forEach { profile ->
                        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                ChoiceButton(profile.displayName, selectedVoiceProfileId == profile.id, busyMessage.isBlank()) { onVoiceProfileSelected(profile.id) }
                                Text("${profile.promptTokenCount} Token", Modifier.weight(1f), fontSize = 12.sp,
                                    color = if (profile.promptTokenCount > CosyVoiceStore.MAX_REALTIME_PROMPT_TOKENS) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            if (!profile.builtIn) {
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                                    if (profile.promptTokenCount > CosyVoiceStore.MAX_REALTIME_PROMPT_TOKENS) TextButton(onClick = { onCreateRealtimeVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("生成实时版") }
                                    TextButton(onClick = { onExportVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("导出") }
                                    TextButton(onClick = { onDeleteVoiceProfile(profile) }, enabled = busyMessage.isBlank()) { Text("删除") }
                                }
                            }
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onImportVoiceProfile, enabled = busyMessage.isBlank() && modelStatus.ready) { Text("批量导入 ZIP") }
                        OutlinedButton(onClick = onExportAllVoiceProfiles, enabled = busyMessage.isBlank() && voiceProfiles.any { !it.builtIn }) { Text("导出全部") }
                    }

                    Text("手机创建音色", fontWeight = FontWeight.Bold, fontSize = 15.sp)
                    Text(
                        if (enrollmentStatus.ready) "创建扩展已安装 · ${formatBytes(enrollmentStatus.installedBytes)}" else "创建扩展未安装 · 约 490 MB",
                        color = if (enrollmentStatus.ready) Color(0xFF228B22) else MaterialTheme.colorScheme.error, fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onImportEnrollment, enabled = busyMessage.isBlank()) { Text("导入创建扩展") }
                        OutlinedButton(onClick = onExportEnrollment, enabled = busyMessage.isBlank() && enrollmentStatus.ready) { Text("导出扩展") }
                    }
                    if (enrollmentStatus.installedBytes > 0L) TextButton(onClick = onDeleteEnrollment, enabled = busyMessage.isBlank()) { Text("删除扩展") }

                    Text("音色创建方式", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        ChoiceButton("参考音频克隆", voiceCreateMode == 0, busyMessage.isBlank()) { voiceCreateMode = 0 }
                        ChoiceButton("文字设计音色", voiceCreateMode == 1, busyMessage.isBlank()) { voiceCreateMode = 1 }
                    }

                    if (voiceCreateMode == 1) {
                        OutlinedTextField(voiceName, { voiceName = it },
                            label = { Text("音色名称（必填）") },
                            isError = voiceName.isBlank(),
                            supportingText = { if (voiceName.isBlank()) Text("必填：新音色在角色库里的显示名") },
                            modifier = Modifier.fillMaxWidth(), singleLine = true)
                        OutlinedTextField(designDescription, { designDescription = it },
                            label = { Text("音色描述") },
                            supportingText = { Text("用自然语言描述声音，例如「青年女性，声音清冷柔和，语速偏慢，带一点疏离感」。不需要任何参考音频。") },
                            modifier = Modifier.fillMaxWidth(), minLines = 3)
                        TextButton(onClick = { designAdvanced = !designAdvanced }, enabled = busyMessage.isBlank()) {
                            Text(if (designAdvanced) "收起高级设置" else "高级设置")
                        }
                        if (designAdvanced) {
                            OutlinedTextField(designReferenceText, { designReferenceText = it },
                                label = { Text("试听文本") },
                                supportingText = { Text("设计时会用这段文字生成参考音频；它同时作为 CosyVoice 的参考文字自动登记，所以默认固定、无需填写。") },
                                modifier = Modifier.fillMaxWidth(), minLines = 2)
                        }
                        Text("试听文本：%s".format(designReferenceText), fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        // 按钮禁用时必须说清缺什么，否则用户只会看到「点不动」。
                        val designBlocker = when {
                            busyMessage.isNotBlank() -> "正在忙：$busyMessage"
                            !modelStatus.ready -> "MNN 朗读模型未就绪"
                            !enrollmentStatus.ready -> "音色创建扩展未安装"
                            voiceName.isBlank() -> "请先填「音色名称」"
                            designDescription.isBlank() -> "请先填「音色描述」"
                            else -> null
                        }
                        Button(
                            onClick = { onCreateVoiceByDesign(voiceName, designDescription, designReferenceText) },
                            enabled = designBlocker == null,
                            modifier = Modifier.fillMaxWidth()) { Text("设计并创建音色") }
                        if (designBlocker != null) {
                            Text("⚠ 还不能创建：$designBlocker", fontSize = 13.sp,
                                color = MaterialTheme.colorScheme.error)
                        }
                        // 设计进行中的实时进度（native 每帧写 progress.txt）
                        if (designProgress.isNotBlank()) {
                            Card(Modifier.fillMaxWidth()) {
                                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    Text("设计进行中", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                                    Text(designProgress, fontSize = 13.sp)
                                    Text("可切到其它 App；前台服务会继续跑。", fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                            }
                        }
                        Text("首次设计需要编译图，约 4-6 分钟。", fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    } else {
                    OutlinedButton(onClick = onSelectReferenceAudio, enabled = busyMessage.isBlank()) {
                        Text(if (selectedAudioName.isBlank()) "选择 MP3/音频" else selectedAudioName)
                    }
                    OutlinedTextField(voiceName, { voiceName = it }, label = { Text("音色名称") }, modifier = Modifier.fillMaxWidth(), singleLine = true)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(segmentStart, { segmentStart = it }, label = { Text("起始秒") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), modifier = Modifier.weight(1f), singleLine = true)
                        OutlinedTextField(segmentEnd, { segmentEnd = it }, label = { Text("结束秒") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), modifier = Modifier.weight(1f), singleLine = true)
                    }
                    if (segmentDuration != null) Text("当前片段：%.1f 秒%s".format(segmentDuration, if (segmentDuration in 3.0..5.0) "" else "（需要 3-5 秒）"),
                        color = if (segmentDuration in 3.0..5.0) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error, fontSize = 12.sp)
                    OutlinedTextField(promptText, { promptText = it }, label = { Text("截取片段对应文字") },
                        supportingText = { Text("实时朗读请选择 3-5 秒清晰单人语音；文字必须与片段完全一致。") },
                        modifier = Modifier.fillMaxWidth(), minLines = 2)
                    Button(
                        onClick = {
                            val start = segmentStart.toDoubleOrNull(); val end = segmentEnd.toDoubleOrNull()
                            if (start != null && end != null) onCreateVoice(VoiceEnrollmentRequest(voiceName, promptText, start, end))
                        },
                        enabled = busyMessage.isBlank() && modelStatus.ready && enrollmentStatus.ready && selectedAudioName.isNotBlank() && voiceName.isNotBlank() && promptText.isNotBlank() && segmentStart.toDoubleOrNull() != null && segmentEnd.toDoubleOrNull() != null,
                        modifier = Modifier.fillMaxWidth()) { Text("创建并选中音色") }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        ChoiceButton("普通复刻", inferenceMode == CosyVoiceInferenceMode.ZERO_SHOT, busyMessage.isBlank()) { inferenceMode = CosyVoiceInferenceMode.ZERO_SHOT }
                        ChoiceButton("指令演绎", inferenceMode == CosyVoiceInferenceMode.INSTRUCT2, busyMessage.isBlank()) { inferenceMode = CosyVoiceInferenceMode.INSTRUCT2 }
                    }
                    if (inferenceMode == CosyVoiceInferenceMode.INSTRUCT2) {
                        Text("指令预设", fontSize = 13.sp)
                        CosyVoiceInstruction.presets.chunked(3).forEach { rowPresets ->
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                rowPresets.forEach { preset ->
                                    if (instruction == preset.instruction) Button(onClick = { instruction = preset.instruction }, enabled = busyMessage.isBlank(), modifier = Modifier.weight(1f)) { Text(preset.label) }
                                    else OutlinedButton(onClick = { instruction = preset.instruction }, enabled = busyMessage.isBlank(), modifier = Modifier.weight(1f)) { Text(preset.label) }
                                }
                                repeat(3 - rowPresets.size) { Spacer(Modifier.weight(1f)) }
                            }
                        }
                        OutlinedTextField(instruction, { instruction = it }, label = { Text("演绎指令") },
                            supportingText = { Text("会作为 Instruct2 指令发送给 LLM，不会附加参考语音 Token。") }, modifier = Modifier.fillMaxWidth(), minLines = 2)
                    }

                    OutlinedTextField(previewText, { previewText = it }, label = { Text("试听文字") }, modifier = Modifier.fillMaxWidth(), minLines = 2)
                    Button(
                        onClick = {
                            onPreview(
                                previewText,
                                CosyVoiceSynthesisOptions(
                                    hardwarePlan = hardwarePlan,
                                    voiceProfileId = selectedVoiceProfileId,
                                    inferenceMode = inferenceMode,
                                    instruction = instruction
                                )
                            )
                        },
                        enabled = busyMessage.isBlank() && modelStatus.ready && (inferenceMode != CosyVoiceInferenceMode.INSTRUCT2 || instruction.isNotBlank()),
                        modifier = Modifier.fillMaxWidth()) { Text("合成并试听") }
                }
            }

            if (busyMessage.isNotBlank()) { Row(verticalAlignment = Alignment.CenterVertically) { CircularProgressIndicator(Modifier.padding(8.dp)); Text(busyMessage, modifier = Modifier.weight(1f)) } }
            if (resultMessage.isNotBlank()) { Card(Modifier.fillMaxWidth()) { Text(resultMessage, Modifier.padding(14.dp), fontSize = 14.sp) } }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModeButtons(selected: Int, enabled: Boolean, onSelected: (Int) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(4 to "自动", 68 to "Buffer", 132 to "Image").forEach { (mode, label) -> ChoiceButton(label, selected == mode, enabled) { onSelected(mode) } }
    }
}

@Composable
private fun ChoiceButton(label: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    if (selected) Button(onClick = onClick, enabled = enabled) { Text(label) } else OutlinedButton(onClick = onClick, enabled = enabled) { Text(label) }
}

private fun formatBytes(bytes: Long): String {
    val gib = bytes / 1024.0 / 1024.0 / 1024.0
    return if (gib >= 1.0) "%.2f GiB".format(gib) else "%.1f MiB".format(bytes / 1024.0 / 1024.0)
}
