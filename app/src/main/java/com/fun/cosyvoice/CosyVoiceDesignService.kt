package com.cosyvoice.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import java.io.File

/**
 * 文字设计音色的前台服务。
 *
 * 为什么必须前台服务：一次设计要 4-5 分钟、峰值内存可达 7 GB（VoiceDesign 4.7G
 * + 朗读模型 1.45G + 音色创建扩展 1G）。实测把这段放在 Activity 协程里，用户一切走
 * （或按 HOME）就会被厂商的「应用冻结」挂起 —— 日志里能看到 ActivityManager: freezing。
 * 前台服务能把进程优先级提到 visible 以上，切后台也能跑完。
 */
class CosyVoiceDesignService : Service() {

    companion object {
        const val CH_ID = "cosy_design"
        const val NOTI_ID = 4721
        const val EXTRA_NAME = "name"
        const val EXTRA_DESC = "description"
        const val EXTRA_TEXT = "referenceText"
        const val EXTRA_BACKEND = "backend"
        private val logLock = Any()

        // 【2026-09-18】设计任务串行化。
        // 原来 onStartCommand 每次都 Thread{}.start()，连点就会在同一 :design 进程里并发跑多个设计：
        //   - 每个设计 anon 峰值 3.5~4 GB，N 个并发直接压垮设备；
        //   - 共享 MNN 全局 Executor，互相争用；
        //   - 都写同一组 progress.txt / design-loop.log / rss.log，输出互相覆盖。
        // 现在：同一时刻只有一个在跑，多余的进【单槽】等待（后来者覆盖待跑者）。
        // 选排队而非取消：MNN 的 in-flight forward 不可中断，唯一"真取消"是 kill 进程，
        // 代价是丢弃 3.5~4 分钟计算并可能留 staging 半成品。
        private val pendingLock = Any()
        @Volatile private var runningJob: Thread? = null
        @Volatile private var pendingIntent: Intent? = null
        @Volatile private var pendingStartId: Int = 0

        fun designDir(ctx: Context): File {
            val d = File(ctx.filesDir, "cosyvoice3-mnn/voicedesign")
            if (!d.exists()) d.mkdirs()
            return d
        }

        /** native 的 g_progressPath 也指向这个文件；UI 轮询它 */
        fun progressFile(ctx: Context): File = File(designDir(ctx), "progress.txt")

        /** 跨进程可见的运行态：running <pid> <startedAtMillis> / idle */
        fun stateFile(ctx: Context): File = File(designDir(ctx), "design-state.txt")

        fun writeProgress(ctx: Context, text: String) {
            runCatching { progressFile(ctx).writeText(text) }
        }

        fun writeState(ctx: Context, text: String) {
            runCatching { stateFile(ctx).writeText(text) }
        }

        // 【2026-09-18】设计期间必须持 Wakelock。
        // 实测：屏幕熄灭 / App 切后台时，同一份设计从 0.2~0.3 s/帧 明显变慢 ——
        // 因为 :design 进程里没有任何东西阻止 CPU 降频休眠，每帧 forward 之间被调度延迟拉长。
        @Volatile private var wakeLock: android.os.PowerManager.WakeLock? = null

        fun acquireWakeLock(ctx: Context) {
            runCatching {
                if (wakeLock?.isHeld == true) return
                val pm = ctx.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
                val wl = pm.newWakeLock(android.os.PowerManager.PARTIAL_WAKE_LOCK, "cosyvoice:design")
                wl.setReferenceCounted(false)
                wl.acquire(30 * 60 * 1000L)   // 兜底上限，防泄漏
                wakeLock = wl
            }
        }

        fun releaseWakeLock() {
            runCatching {
                if (wakeLock?.isHeld == true) wakeLock?.release()
                wakeLock = null
            }
        }

        // 【2026-09-18】权威终态序号（每次成功 +1）。
        // 为什么不用 progress.txt 的文本判断完成：下一个 startJob 一开头就 writeProgress("")，
        // UI 每秒轮询很可能错过「完成」→ 新音色不刷新。序号单调递增，永不丢失。
        fun doneSeqFile(ctx: Context): File = File(designDir(ctx), "done-seq.txt")

        fun bumpDoneSeq(ctx: Context, detail: String) {
            runCatching {
                val cur = readDoneSeq(ctx)
                doneSeqFile(ctx).writeText((cur + 1).toString() + " " + detail + " " + System.currentTimeMillis())
            }
        }

        fun readDoneSeq(ctx: Context): Int = runCatching {
            doneSeqFile(ctx).takeIf(File::isFile)
                ?.readText()?.trim()?.split(' ')?.firstOrNull()?.toIntOrNull() ?: 0
        }.getOrDefault(0)

        /** BUILD-GATE 的 Kotlin 侧 fingerprint —— 与 native 的 VD_BUILD_ID 配对 */
        const val APP_BUILD_ID = "APP_BUILD_ID=20260917_150246_BG1"

        /** 读本进程 RssAnon/VmRSS（BUILD-GATE 与内存归因用） */
        fun rssLine(tag: String): String {
            var anon = -1L; var rss = -1L; var hwm = -1L
            try {
                File("/proc/self/status").readLines().forEach { l ->
                    val v = l.filter { it.isDigit() }.toLongOrNull() ?: -1L
                    when {
                        l.startsWith("RssAnon:") -> anon = v
                        l.startsWith("VmRSS:")  -> rss = v
                        l.startsWith("VmHWM:")  -> hwm = v
                    }
                }
            } catch (_: Throwable) {}
            val s = "RSSK[" + tag + "] rss=" + rss + "kB anon=" + anon + "kB hwm=" + hwm + "kB"
            Log.e("VD_BUILD", s)
            return s
        }

        fun logFile(ctx: Context): File = File(designDir(ctx), "design-loop.log")

        fun log(ctx: Context, s: String) {
            Log.i("CosyVoiceDesign", s)
            synchronized(logLock) { try { logFile(ctx).appendText(s + "\n") } catch (_: Throwable) {} }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.e("VD_BUILD", APP_BUILD_ID + " (Service.onCreate)")
        rssLine("Service.onCreate-entry")
    }
    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)
            if (nm.getNotificationChannel(CH_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(CH_ID, "音色设计", NotificationManager.IMPORTANCE_LOW))
            }
        }
    }

    private fun notify(text: String) {
        ensureChannel()
        val b = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) Notification.Builder(this, CH_ID)
                else Notification.Builder(this)
        startForeground(NOTI_ID, b.setContentTitle("文字设计音色")
            .setContentText(text).setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true).build())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        notify("准备中…")
        // 【2026-09-18】串行化：已有任务在跑时，本次请求进【单槽】等待（覆盖上一个待跑请求），
        // 绝不起第二个线程。理由与取舍见 companion object 里的说明。
        if (runningJob?.isAlive == true) {
            synchronized(pendingLock) {
                pendingIntent = intent
                pendingStartId = startId
            }
            log(this, "=== 已有设计在运行，本次请求已排队（startId=$startId） ===")
            notify("已有设计在运行，本次已排队")
            return START_NOT_STICKY
        }
        startJob(intent, startId)
        return START_NOT_STICKY
    }

    /** 真正跑一次设计。调用方必须保证同一时刻只有一个在跑。 */
    private fun startJob(intent: Intent?, startId: Int) {
        val name = intent?.getStringExtra(EXTRA_NAME).orEmpty()
        val desc = intent?.getStringExtra(EXTRA_DESC).orEmpty()
        val refText = intent?.getStringExtra(EXTRA_TEXT) ?: CosyVoiceVoiceDesigner.DEFAULT_REFERENCE_TEXT
        val backend = intent?.getStringExtra(EXTRA_BACKEND) ?: "opencl"
        val job = Thread {
            val store = CosyVoiceStore(this)
            // design-loop.log 与 progress.txt 都必须清：只清 log 时 UI 轮询会立刻读到
            // 上一次残留的「完成」→ 秒结束轮询、新音色不刷新。
            runCatching { logFile(this).writeText("") }
            writeProgress(this, "")
            writeState(this, "running " + android.os.Process.myPid() + " " + System.currentTimeMillis())
            // 防止屏幕熄灭/切后台时 CPU 降频休眠（实测会让每帧明显变慢）。
            acquireWakeLock(this)
            val t0 = System.currentTimeMillis()
            log(this, "=== 前台服务：文字设计音色开始（startId=$startId） ===")
            log(this, "描述: " + desc)
            log(this, "试听文本: " + refText)
            log(this, "后端: " + backend)
            try {
                // 【不要在这里调 CosyVoiceRuntime.close()】
                //
                // 本 Service 跑在 android:process=":design" 的独立进程里，
                // 该进程只会加载 libcosy_voicedesign_jni.so（+ libMNN/libMNNAudio），
                // 不加载朗读链（libcosy_llm_jni / flow / hift / libllm.so）。
                //
                // 而 close() 会调用 CosyVoiceLlmNative.reset() / FlowNative.reset() / HiFTNative.reset()，
                // 这些 object 的 init{} 里有 System.loadLibrary —— 一调就会把朗读链的 .so
                // 拉进本进程，隔离立刻失效，重新出现「MNN 全局 Executor 争用 → enroll 卡死」。
                //
                // 独立进程里本来就没有朗读链资源需要释放（进程退出时内核全额回收）。
                val profile = CosyVoiceVoiceDesigner.createFromDesign(
                    store = store,
                    nativeLibDir = applicationInfo.nativeLibraryDir,
                    designModelsDir = store.voiceDesignDir,
                    description = desc,
                    displayName = name,
                    backend = backend,
                    referenceText = refText
                ) { stage ->
                    log(this, "[进度] " + stage)
                    notify(stage)
                }
                log(this, "音色已注册: " + profile.id + " / " + profile.displayName +
                    " / tokens=" + profile.promptTokenCount + " / frames=" + profile.promptFrameCount)
                log(this, "音色目录: " + profile.directory.absolutePath)
                log(this, "目录内容: " + profile.directory.list()?.joinToString(", "))
                log(this, "=== 闭环成功，总耗时 " + (System.currentTimeMillis() - t0) + " ms ===")
                // 显式落文件 + bump 序号：不依赖 native 写的「完成」被 UI 恰好看到
                // （下一轮 startJob 会清 progress.txt）。
                writeProgress(this, "完成: " + profile.id + " (" + profile.promptTokenCount + " token)")
                bumpDoneSeq(this, "ok " + profile.id)
                notify("完成")
            } catch (t: Throwable) {
                log(this, "=== 闭环失败: " + t.javaClass.simpleName + ": " + t.message + " ===")
                Log.e("CosyVoiceDesign", "design failed", t)
                // 必须落文件：否则 UI 轮询永远等不到终止条件（原来只 notify → 一直转圈）。
                writeProgress(this, "失败: " + (t.message ?: t.javaClass.simpleName))
                bumpDoneSeq(this, "fail " + (t.message ?: t.javaClass.simpleName))
                notify("失败")
            } finally {
                stopForeground(STOP_FOREGROUND_DETACH)
                // 带 startId：只停本次启动，不影响排队中的下一次。
                stopSelf(startId)
                writeState(this, "idle")
                releaseWakeLock()
                // 有待跑请求 → 立刻接着跑，并且【不能 kill 进程】（否则把新任务一起杀掉）。
                val next: Intent?
                val nextStartId: Int
                synchronized(pendingLock) {
                    next = pendingIntent
                    nextStartId = pendingStartId
                    pendingIntent = null
                    pendingStartId = 0
                }
                if (next != null) {
                    log(this, "=== 发现排队请求，立刻开始下一次设计 ===")
                    startJob(next, nextStartId)
                } else {
                    // 无待跑任务 → 结束进程，把 Hexagon DSP 交还主进程的合成链。
                    // 本进程经 CosyVoiceHexagonBootstrap 加载过 libMNN_htpops.so 并设过
                    // ADSP_LIBRARY_PATH，而 Hexagon/DSP 是独占资源 —— 只调 stopSelf() 时
                    // Android 仍会把进程留在 cached 状态，DSP 侧不会释放。
                    // 实测后果（创建音色 → 再合成）：主进程连续 Hexagon 解码无法启用，
                    //   continuousHexagon=false、npuPrefillMs=1078、decodeMs=56493、
                    //   tokensPerSecond=4.09；质量门只能整句回退 CPU（LLM 46.7 秒）。
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        android.os.Process.killProcess(android.os.Process.myPid())
                    }, 400L)
                }
            }
        }
        runningJob = job
        job.start()
    }
}
