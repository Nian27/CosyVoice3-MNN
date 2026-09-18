// voicedesign_jni.cpp - ReaderVoice VoiceDesign HTP full chain (in-app)
// text(离线 tokenize) -> prompt_emb -> 62x GraphB(prefill) -> self-driven AR -> 16 codes/frame
//   -> EOS -> decoder -> reference.wav
#include <jni.h>
#include <android/log.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/Interpreter.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include "vd_text.h"
// VoiceProfile 注册算法（在 Plus 工程的 mnn-jni/ 下，编译时用 -Imnn-jni 找到）。
// 这样「文字设计音色」与「参考音频克隆」走的是同一份 enrollment 实现。
#include "CosyVoiceEnrollmentCore.h"

using namespace MNN;
using namespace MNN::Express;
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VDS", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VDS", __VA_ARGS__)

namespace {

const int CAP = 16;      // codepred KV capacity
const int HD = 128;
const int NL = 28;       // talker layers
const int NKV = 8;       // talker kv heads
const int TKV = 768;     // talker KV capacity
const int CPNL = 5;      // codepred layers
const int CPKV = 8;

struct VdSession {
    std::shared_ptr<Module> mGB, mCH, mCE, mFE, mCPF, mCPS, mDEC;
    const float* lmW = nullptr;
    const float* cpW = nullptr;
    void* lmMap = nullptr; void* cpMap = nullptr;
    std::vector<float> tRopeC, tRopeS;    // (MAXPOS,128) talker merged mrope
    std::vector<float> cRopeC, cRopeS;    // (CAP,128) codepred rope
    vdt::Tokenizer tok;                   // 设备端 BPE 分词器 (与 HF 逐 id 相同)
    vdt::TextTables txtTab;               // text_embedding + text_projection
    // A2: talker codec_embedding (3072,2048) fp32，改为只读映射（原来 readVec 读进匿名内存 25MB）
    void* codecEmbMap = nullptr;
    void* codecEmbMapH = nullptr;
    const float* codecEmbPtr = nullptr;
    int maxPos = 0;
    std::string backend = "cpu";
    // 分阶段加载用：nativeLoad 里创建的 RuntimeManager 需要留给 nativeRun 延迟加载 decoder
    std::shared_ptr<Executor::RuntimeManager> rtm;
    std::shared_ptr<Executor::RuntimeManager> decRtm;   // 实验：decoder 的独立 Runtime
    // 注：曾尝试 OpenCL kernel 缓存与权重 mmap 缓存（见 nativeLoad 里的说明），均已撤回。
    std::mutex mu;
    unsigned long long seed = 123456789ull;
    std::string lastError;
};
static double nextU(unsigned long long& s) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((s >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
// repetition_penalty 与 HF RepetitionPenaltyLogitsProcessor 同语义:
//   score<0 ? score*p : score/p, 只作用于历史中出现过的 id
static int sampleTopK(const float* lg, int vocab, double temp, int topk, bool suppress, double u,
                      double repPenalty = 1.0, const std::vector<int>* hist = nullptr) {
    std::vector<float> lg2;
    const float* src = lg;
    if (repPenalty != 1.0 && hist && !hist->empty()) {
        lg2.assign(lg, lg + vocab);
        for (int id : *hist) {
            if (id < 0 || id >= vocab) continue;
            lg2[id] = (lg2[id] < 0) ? (float)(lg2[id] * repPenalty) : (float)(lg2[id] / repPenalty);
        }
        src = lg2.data();
    }
    std::vector<std::pair<double,int>> v; v.reserve(vocab);
    for (int i = 0; i < vocab; i++) { if (suppress && i >= 2048 && i != 2150) continue; v.push_back({(double)src[i]/temp, i}); }
    std::sort(v.begin(), v.end(), [](const std::pair<double,int>&a, const std::pair<double,int>&b){return a.first>b.first;});
    if ((int)v.size() > topk) v.resize(topk);
    double mx = v[0].first, s = 0; std::vector<double> pr(v.size());
    for (size_t i=0;i<v.size();i++){ pr[i]=exp(v[i].first-mx); s+=pr[i]; }
    for (size_t i=0;i<v.size();i++) pr[i]/=s;
    double acc=0;
    for (size_t i=0;i<v.size();i++){ acc+=pr[i]; if (u<=acc) return v[i].second; }
    return v.back().second;
}
static bool readF(const std::string& p, void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) return false;
    size_t r = fread(d, 1, n, f); fclose(f); return r == n;
}
// 内存打点：读 /proc/self/status，区分匿名(不可回收)与文件页(可回收)
// 同时写文件 —— logcat 环形缓冲会被系统日志挤掉，不可靠。
static std::string g_rssLogPath;
// 进度文件：每帧覆写一次，供 App 侧轮询显示「到第几帧」。
// 背景：一次设计要 4-9 分钟，之前 UI 上除了「正在生成」没有任何反馈，用户会以为卡死。
static std::string g_progressPath;
static void writeProgress(const char* fmt, ...) {
    if (g_progressPath.empty()) return;
    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    FILE* f = fopen(g_progressPath.c_str(), "w");
    if (f) { fprintf(f, "%s\n", buf); fclose(f); }
}
// 读 CPU 当前频率（kHz）。用于验证「前台服务被当成后台 → 限频」这个假设。
static void readCpuFreqKHz(long out[8], int* count) {
    *count = 0;
    for (int i = 0; i < 8; ++i) {
        char p[128];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        FILE* f = fopen(p, "r");
        if (!f) continue;
        long v = -1;
        if (fscanf(f, "%ld", &v) == 1 && v > 0) out[(*count)++] = v;
        fclose(f);
    }
}

static void logRss(const char* tag) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    long rss = -1, anon = -1, file = -1, hwm = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line + 6, "%ld", &rss);
        else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line + 8, "%ld", &anon);
        else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line + 8, "%ld", &file);
        else if (strncmp(line, "VmHWM:", 6) == 0) sscanf(line + 6, "%ld", &hwm);
    }
    fclose(f);
    LOGI("RSS[%s] rss=%ldkB anon=%ldkB file=%ldkB hwm=%ldkB", tag, rss, anon, file, hwm);
    if (!g_rssLogPath.empty()) {
        FILE* o = fopen(g_rssLogPath.c_str(), "a");
        if (o) {
            fprintf(o, "RSS[%s] rss=%ldkB anon=%ldkB file=%ldkB hwm=%ldkB\n", tag, rss, anon, file, hwm);
            fclose(o);
        }
    }
}

static std::vector<float> readVec(const std::string& p, size_t n) {
    std::vector<float> v(n); if (!readF(p, v.data(), n*4)) { v.clear(); } return v;
}
static void writeWav16(const std::string& path, const std::vector<float>& x, int sr) {
    FILE* f = fopen(path.c_str(), "wb"); if (!f) return;
    int n = (int)x.size(); int db = n*2;
    unsigned char h[44] = {0};
    auto p32=[&](int o,unsigned v){h[o]=v&0xff;h[o+1]=(v>>8)&0xff;h[o+2]=(v>>16)&0xff;h[o+3]=(v>>24)&0xff;};
    auto p16=[&](int o,unsigned v){h[o]=v&0xff;h[o+1]=(v>>8)&0xff;};
    memcpy(h,"RIFF",4); p32(4,36+db); memcpy(h+8,"WAVE",4);
    memcpy(h+12,"fmt ",4); p32(16,16); p16(20,1); p16(22,1);
    p32(24,sr); p32(28,sr*2); p16(32,2); p16(34,16);
    memcpy(h+36,"data",4); p32(40,db);
    fwrite(h,1,44,f);
    std::vector<short> pcm(n);
    for (int i=0;i<n;i++){ float v=x[i]; if(v>1)v=1; if(v<-1)v=-1; pcm[i]=(short)lrintf(v*32767.0f); }
    fwrite(pcm.data(),2,n,f); fclose(f);
}
static void* mmapFile(const std::string& p, size_t bytes, void** out) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    void* q = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (q == MAP_FAILED) return nullptr;
    *out = q;
    return q;
}
// 后端可能 stop-execute: 此时 onForward 返回空 vector 或含空 VARP。
// 必须检查后再解引用，否则 VARP::operator->() 直接 SIGSEGV(fault addr 0x0)。
static bool outOk(const std::vector<VARP>& o, size_t need, const char* tag) {
    if (o.size() < need) { LOGI("BACKEND_STOP_EXECUTE at %s (out size=%zu need=%zu)", tag, o.size(), need); return false; }
    for (size_t i = 0; i < need; i++) { if (o[i].get() == nullptr) { LOGI("NULL_VARP at %s[%zu]", tag, i); return false; } }
    return true;
}
static std::shared_ptr<Module> loadM(const std::string& path, std::vector<std::string> i, std::vector<std::string> o,
                                     std::shared_ptr<Executor::RuntimeManager> rtm, std::string& err) {
    Module::Config c; c.shapeMutable = true;
    std::shared_ptr<Module> m(Module::load(i, o, path.c_str(), rtm, &c), Module::destroy);
    if (!m) err = "load failed: " + path;
    return m;
}
} // namespace

// 脚本会把下面这行替换成真实时间戳（BUILD-GATE 的 runtime fingerprint）
#define VD_BUILD_ID "VD_BUILD_ID=20260917_150246_BG1"

// JNI_OnLoad：System.loadLibrary("cosy_voicedesign_jni") 时的最早 native 点。
// 所有 native static initializer 都在这里之前完成，所以这一行的内存读数
// 直接回答「.so 加载本身吃了多少 anon」。
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    __android_log_print(ANDROID_LOG_ERROR, "VD_BUILD", "%s", VD_BUILD_ID);
    logRss("JNI_OnLoad");          // ← 与 Kotlin 侧 before/after loadLibrary 配对
    return JNI_VERSION_1_6;
}

// ============ GB-SPLIT-P0：4L embedded vs external 的 A/B 探针 ============
// 由文件开关 vd_gb4l.txt 触发（内容 "A" = external， "B" = embedded）。
// 只回答一个问题：内嵌改变的是【执行速度】还是【运行时权重驻留】？
extern "C" JNIEXPORT jstring JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeGb4lProbe(
        JNIEnv* env, jobject, jstring jdir, jint reps) {
    const char* c = env->GetStringUTFChars(jdir, nullptr);
    std::string d(c ? c : "");
    env->ReleaseStringUTFChars(jdir, c);
    if (!d.empty() && d.back() != '/') d += "/";

    // 读开关
    std::string which = "A";
    FILE* sf = fopen((d + "vd_gb4l.txt").c_str(), "r");
    if (sf) { char b[8] = {0}; if (fscanf(sf, "%7s", b) == 1) which = b; fclose(sf); }
    const bool useEmbedded = (which == "B");
    const std::string model = d + (useEmbedded ? "graphb4l_embedded.mnn" : "graphb4l_external.mnn");

    auto t0 = std::chrono::steady_clock::now();
    logRss("gb4l-before-load");

    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_High;
    bc.memory = MNN::BackendConfig::Memory_Normal;
    bc.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig sc; sc.numThread = 4; sc.backendConfig = &bc;
    sc.type = MNN_FORWARD_CPU;
    sc.backupType = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtm(
        Executor::RuntimeManager::createRuntimeManager(sc), Executor::RuntimeManager::destroy);
    if (!rtm) return env->NewStringUTF("ERR rtm");
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);

    std::string err;
    auto m = loadM(model, {"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"},
                   {"hidden_states","cur_k","cur_v"}, rtm, err);
    const double loadMs = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (!m) { LOGI("GB4L load failed: %s", err.c_str()); return env->NewStringUTF("ERR load"); }
    logRss("gb4l-after-load");

    // 固定输入（与 28L 导出脚本一致的全零）
    const int NL = 4, MAXKV = 768;
    auto vE = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    auto vK = _Input({NL,1,8,MAXKV,128}, NCHW, halide_type_of<float>());
    auto vV = _Input({NL,1,8,MAXKV,128}, NCHW, halide_type_of<float>());
    auto vC = _Input({1,1,128}, NCHW, halide_type_of<float>());
    auto vS = _Input({1,1,128}, NCHW, halide_type_of<float>());
    auto vA = _Input({1,1,1,MAXKV}, NCHW, halide_type_of<float>());
    auto vM = _Input({1,1,MAXKV,1}, NCHW, halide_type_of<float>());
    memset(vE->writeMap<float>(), 0, 2048*4);
    memset(vK->writeMap<float>(), 0, (size_t)NL*8*MAXKV*128*4);
    memset(vV->writeMap<float>(), 0, (size_t)NL*8*MAXKV*128*4);
    memset(vC->writeMap<float>(), 0, 128*4);
    memset(vS->writeMap<float>(), 0, 128*4);
    memset(vA->writeMap<float>(), 0, MAXKV*4);
    memset(vM->writeMap<float>(), 0, MAXKV*4);

    // warmup 2 次 + 计时 reps 次
    double firstMs = -1, totalMs = 0, hiddenSum = 0;
    for (int i = 0; i < reps + 2; ++i) {
        auto t = std::chrono::steady_clock::now();
        auto o = m->onForward({vE, vK, vV, vC, vS, vA, vM});
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t).count();
        if (o.size() < 1 || o[0]->getInfo() == nullptr) return env->NewStringUTF("ERR forward");
        if (i == 0) { firstMs = ms; logRss("gb4l-after-first-forward"); }
        else if (i >= 2) {
            totalMs += ms;
            const float* hp = o[0]->readMap<float>();
            for (int k = 0; k < 16; ++k) hiddenSum += hp[k];   // 用于核对数值一致
        }
    }
    const double steadyMs = reps > 0 ? totalMs / reps : 0;
    logRss("gb4l-after-reps");
    char buf[320];
    snprintf(buf, sizeof(buf), "OK gb4l=%s loadMs=%.0f firstMs=%.1f steadyMs=%.2f reps=%d hiddenSum=%.6f",
             useEmbedded ? "EMBEDDED" : "EXTERNAL", loadMs, firstMs, steadyMs, reps, hiddenSum);
    LOGI("%s", buf);
    return env->NewStringUTF(buf);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeCreate(JNIEnv*, jobject) {
    __android_log_print(ANDROID_LOG_ERROR, "VD_BUILD", "%s (nativeCreate)", VD_BUILD_ID);
    logRss("nativeCreate-entry");
    auto* sess = new VdSession();
    logRss("nativeCreate-done");
    return (jlong) sess;
}
extern "C" JNIEXPORT void JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeRelease(JNIEnv*, jobject, jlong p) {
    if (p) delete reinterpret_cast<VdSession*>(p);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeLastError(JNIEnv* e, jobject, jlong p) {
    VdSession* s = reinterpret_cast<VdSession*>(p); return e->NewStringUTF(s ? s->lastError.c_str() : "no session");
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeLoad(
        JNIEnv* env, jobject, jlong p, jstring jdir, jstring jbackend, jint maxPos, jstring jnativedir) {
    VdSession* s = reinterpret_cast<VdSession*>(p); if (!s) return JNI_FALSE;
    logRss("nativeLoad-entry");
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    const char* be  = env->GetStringUTFChars(jbackend, nullptr);
    std::string d(dir); s->backend = be; s->maxPos = maxPos;
    std::string nd;
    if (jnativedir) { const char* ndc = env->GetStringUTFChars(jnativedir, nullptr); nd = ndc; env->ReleaseStringUTFChars(jnativedir, ndc); }
    env->ReleaseStringUTFChars(jdir, dir); env->ReleaseStringUTFChars(jbackend, be);
    // DSP skel 必须能被 adsp 找到；命令行为此显式设 ADSP_LIBRARY_PATH，App 里同样需要
    if (!nd.empty()) { std::string ap = nd + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"; setenv("ADSP_LIBRARY_PATH", ap.c_str(), 1); LOGI("ADSP_LIBRARY_PATH=%s", ap.c_str()); }
    if (!d.empty() && d.back() != '/') d += "/";
    BackendConfig bc;
    // 【默认 low，不是 high】—— 这是内存基线的必要组成，不是实验偏好。
    //
    // 实测（2026-09-17，真机 BUILD-GATE 验证过的干净进程）：
    //   Precision_Low  -> GraphB 的 fp16 权重保持低精度 -> anon 增量 2.81 GB（约等于 .weight 的 2.82 GB）OK
    //   Precision_High -> MNN 把 fp16 展开成 fp32      -> anon 增量 4.95 GB（约等于 2.82 x 2）FAIL
    //     实测 after-prefill anon 5,951,364 kB，frame-0 6,636,064 kB，VmHWM 6.84 GB，decode 卡死
    //
    // 所有已跑通的设计（含 206 秒那次）都在 Precision_Low 下完成，它是事实基线。
    // 曾经把它当「实验残留」删掉过一次，anon 立刻从 3.53 GB 涨到 5.95 GB 并卡死 ——
    // 教训：清理前必须先确认该配置对基线是否必要；而且这类配置应显式写进代码，不藏在文件里。
    //
    // 覆盖优先级：环境变量 VDS_PREC > <模型目录>/vd_prec.txt（仅实验用）> 默认 low
    std::string prec = "low";
    if (const char* precEnv = getenv("VDS_PREC")) prec = precEnv;
    else {
        FILE* pf = fopen((d + "vd_prec.txt").c_str(), "r");
        if (pf) { char buf[32] = {0}; if (fscanf(pf, "%31s", buf) == 1) prec = buf; fclose(pf); }
    }
    if (prec == "low") bc.precision = BackendConfig::Precision_Low;
    else if (prec == "low_bf16") bc.precision = BackendConfig::Precision_Low_BF16;
    else bc.precision = BackendConfig::Precision_High;
    LOGI("VDS_PREC=%s", prec.c_str());
    bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sc; sc.numThread = 4; sc.backendConfig = &bc;
    if (s->backend == "htp") sc.type = MNN_FORWARD_HEXAGON;
    else if (s->backend == "opencl") sc.type = MNN_FORWARD_OPENCL;
    else sc.type = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sc), Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);

    // ---- 让 external weight 走 mmap，而不是 _RebuildExternalOp 读进 anon ----
    //
    // 实测（2026-09-17）：graphb28 是 external 导出（.mnn 503KB + .weight 2.82GB）。
    // 默认 useCachedMmap=0，于是 prefill 第一次 forward 时 _RebuildExternalOp 把
    // 2.82 GB 权重读进【匿名内存】：
    //     after-prompt-build anon = 0.72 GB
    //     after-prefill      anon = 3.53 GB      ← +2.81 GB ≈ .weight 大小，完全吻合
    // 之后 decode 稳定在 1.86 秒/帧（autorun 干净进程是 0.5 秒/帧）。
    // 且与 backend 无关（cpu / opencl 的 after-prefill 只差 76 kB）。
    //
    // MNN 源码（OpCommonUtils.cpp:691）：
    //     usemmap = (hint().useCachedMmap > 1);
    // 而（CPUBackend.cpp:293）useCachedMmap 从 1 变 2 需要 <weightMemoryPath>/<prefix>sync.static
    // 已存在；且 static allocator 上限 mmapFileSize 默认只有 1024 MB —— 2.82 GB 必然超限。
    //
    // 这两个正是此前「USE_CACHED_MMAP 实验卡死」的根因（ADR-053 记录的失败）。
    // 现在两处都修：直接设 useCachedMmap=2（跳过 sync.static 依赖），
    // 并把 mmapFileSize 提到 4096 MB。
    // 【已实测失败，撤回 2026-09-17】让 external weight 走 mmap。
    //
    // 动机：graphb28 是 external 导出（.mnn 503KB + .weight 2.82GB）。默认 useCachedMmap=0，
    // prefill 第一次 forward 时 _RebuildExternalOp 把 2.82 GB 读进【匿名内存】：
    //     after-prompt-build anon = 0.72 GB → after-prefill anon = 3.53 GB（+2.81 GB）
    // 之后 decode 稳定在 1.86 秒/帧（干净进程 0.5 秒/帧）。且与 backend 无关
    // （cpu / opencl 的 after-prefill 只差 76 kB，已 A/B 验证）。
    //
    // 试过的配置（两处 bug 都修了）：
    //     rtm->setExternalPath(d, 2);                                  // PathType::EXTERNAL_WEIGHT_DIR
    //     rtm->setHint(Interpreter::HintMode::USE_CACHED_MMAP, 2);     // 直接 2，跳过 sync.static 依赖
    //     rtm->setHint(Interpreter::HintMode::MMAP_FILE_SIZE, 4096);   // 默认 1024 MB < 2.82 GB
    //
    // 实测结果：**anon 确实降下来了**（RssAnon 3.83 GB → 0.27 GB，降 3.56 GB ✓），
    // 但 MNN 会预分配一个正好 mmapFileSize 大小的 .static 文件
    //     <dir>/0_0_0_1_0.static      = 4,294,967,296 B（正好 4 GB）
    //     <dir>/0_0_0_1_sync.static   = 0 B
    // 然后 prefill 死等（20 秒 CPU 增量 0 jiffies，30 秒后进度不动）。
    //
    // 结论：MNN 3.6.1 的 external-weight mmap 路径对 2.82 GB 量级的权重不可用。
    // 这与 ADR-053 记录的「USE_CACHED_MMAP 实验卡死」一致，现在知道了确切机制。
    // 保持默认（rebuild 进 anon），把 2.82 GB 的内存代价记录为已知问题。
    s->rtm = rtm;

    s->rtm = rtm;   // 分阶段加载：nativeRun 要用它延迟加载 decoder
    // OpenCL kernel 编译缓存。
    // 之前没设，每次冷启动都要重编 kernel —— 实测同一份 prompt 的 prefill 首次 64.1s、热跑 35.6s，
    // 那 ~28s 基本就是编译开销。CosyVoice 侧的 Flow/HiFT 一直是设了缓存的（gpu-cache/*.cache），
    // 这里补上，缓存放在模型目录下（调用方保证可写）。
    // 【已撤回】OpenCL kernel 缓存实验（setCache/updateCache）。
    // 动机：CosyVoice 侧 Flow/HiFT 一直设了 gpu-cache，而这里没设，
    //       同一 prompt 的 prefill 冷 64.1s / 热 35.6s，怀疑那 ~28s 是 kernel 编译。
    // 实测结论（2026-09-16）：设了之后进程会卡在 decoder 之后不再推进
    //       （frame 97 之后 10 分钟零输出，RSS 停在 1.59GB，且目录里始终没有 .cache 文件），
    //       收益未证实而风险明确，因此撤回，保持与已验证版本一致。
    // 【已撤回】USE_CACHED_MMAP 实验。
    //
    // 机制（OpCommonUtils.cpp）：usemmap = hint().useCachedMmap > 1；不满足就走
    // _RebuildExternalOp 把外部权重读进内存。而 useCachedMmap 从 1 变 2 需要
    // weightMemoryPath + 已存在的 sync.static 标记。
    //
    // 实测结论（2026-09-16）：这条路走不通 ——
    //   1) MNN 会把权重复制进 "<prefix>.static"（默认 mmapFileSize=1024MB），
    //      实测写到 1,073,741,824 B 就停住，进程卡在设计阶段不再推进（RSS 掉到 453MB）。
    //   2) 它本质是「用磁盘换速度」，与本次「省内存」目标方向相反。
    // 因此撤回，decoder 回到内嵌（方案A）。
    auto t0 = std::chrono::steady_clock::now();
    g_rssLogPath = d + "rss.log";
    { FILE* o = fopen(g_rssLogPath.c_str(), "w"); if (o) fclose(o); }
    logRss("before");
    // GraphB 的权重是这套链里最大的一块（fp16 2.82 GB）。它用 --saveExternalData 导出，
    // 而 MNN 的 useCachedMmap 默认 0，于是 prefill 第一次 forward 时 _RebuildExternalOp
    // 会把整份权重读进【匿名内存】（实测 anon 0.72 GB → 3.53 GB，+2.81 GB ≈ .weight 大小），
    // 之后 decode 从 0.5 秒/帧 掉到 2.07 秒/帧。
    //
    // int4 block-wise 量化把权重降到 884 MB（减小 3.19×），直接减少要读进 anon 的字节数。
    // 若 w4 版不存在则自动回退 fp16，便于 A/B 与回退。
    // 【显式配置，禁止"文件存在就自动切换"】
    //
    // 教训（2026-09-17）：曾经写成 "graphb28_v6_w4.mnn 存在就用 int4"，结果一次实验遗留的
    // 权重文件悄悄改变了产品行为 —— 而 int4 在 CPU 路径上对运行时 anon 零收益（实测
    // 334,064 kB vs 334,020 kB），数值又没做过 Gate。实验模型绝不能因为"碰巧在目录里"
    // 就进入正式路径。
    //
    // 现在：默认永远是 fp16；只有显式写入 vd_graphb_variant=w4 才切换到实验变体。
    {
        std::string variant = "fp16";
        FILE* vf = fopen((d + "vd_graphb_variant.txt").c_str(), "r");
        if (vf) { char b[16] = {0}; if (fscanf(vf, "%15s", b) == 1) variant = b; fclose(vf); }
        if (variant != "w4") variant = "fp16";
        const std::string gbFile = (variant == "w4") ? "graphb28_v6_w4.mnn" : "graphb28_v6_fp16.mnn";
        LOGI("VDS_GRAPHB file=%s variant=%s", gbFile.c_str(), variant.c_str());
        s->mGB = loadM(d + gbFile, {"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"}, {"hidden_states","cur_k","cur_v"}, rtm, s->lastError);
    }
    logRss("graphb28");
    if (!s->mGB) return JNI_FALSE;
    s->mCH  = loadM(d+"codec_head_fp16.mnn", {"hidden_states"}, {"logits"}, rtm, s->lastError);
    // 【已回退】曾用 host fp32 查表替掉下面两个模块，省 ~468MB。但那使数值从 fp16 变成 fp32
    // （实测 diff 4.739e-04），改变了自回归动力学：旧版 68 帧 EOS → host 版 93+ 帧不 EOS。
    // 当前系统的 golden 隐含绑定了 fp16 embedding table，故回退。
    // 若将来要做，需做成 bit-compatible：fp16 存储 + 与原图一致的累加顺序 + 逐级 round-to-fp16。
    s->mCE  = loadM(d+"codec_emb_fp16.mnn", {"codes"}, {"emb"}, rtm, s->lastError);
    s->mFE  = loadM(d+"frame_emb_fp16.mnn", {"codes"}, {"emb"}, rtm, s->lastError);
    logRss("ch+ce+fe");
    s->mCPF = loadM(d+"codepred_prefill_fp16.mnn", {"past_hidden","code0_emb","rope_cos","rope_sin","slot_mask0","slot_mask1","attn_mask"}, {"hidden_last","kv_k_out","kv_v_out"}, rtm, s->lastError);
    s->mCPS = loadM(d+"codepred_step_fp16.mnn", {"token_emb","kv_k","kv_v","rope_cos","rope_sin","slot_mask","attn_mask"}, {"hidden_last","kv_k_out","kv_v_out"}, rtm, s->lastError);
    logRss("codepred");
    // 【最终决策 2026-09-16】decoder 保持外置（全外置配置）。
    //
    // 实测对比（真机 SM8850，MemFree 长期只有 100-600MB）：
    //   全外置      anon 0.61GB  HWM 0.90GB  decoder 459s  -> 实测成功 2 次闭环 ✓
    //   decoder内嵌 anon 0.76GB  HWM 1.86GB  decoder  20s  -> HyperHold 换出，卡在 frame 97 ✗
    //   decoder fp16 anon  -      HWM ~1.13GB decoder 20s  -> 数值 FAIL（int16 最大差 2693 LSB）✗
    //
    // 结论：这台设备上「能不能跑通」由 HWM 决定，HWM > ~1.2GB 就会触发 HyperHold 换出。
    // 459s 虽慢，但 VoiceDesign 是**一次性**成本（设计一次音色，之后朗读很快），
    // 而跑不通不可接受。故取全外置。
    //
    // decoder 慢的机制（MNN 源码级）：OpCommonUtils.cpp 用
    //   usemmap = (hint().useCachedMmap > 1)
    // 判断是否走 mmap；不满足就走 _RebuildExternalOp 把外部权重读进内存。
    // 试过用 USE_CACHED_MMAP + EXTERNAL_WEIGHT_DIR 修，但那机制是把权重复制进
    // "<prefix>.static"（默认 mmapFileSize=1024MB），实测写到 1GB 就卡住，
    // 且本质是「用磁盘换速度」，与省内存目标相反，已撤回。
    // 实验：Decoder 独立 Runtime 隔离 —— decoder 不在这里加载（改到 nativeRun 里用独立 rtm 加载内嵌版）。
    // 这样两套 decoder 不会并存（第一次实验就是因为这里还在加载外置版，导致内存叠加、rss 里出现两行）。
    // 注意：本行暂时停用后，下面的 guard 也不再要求 mDEC。
    (void)rtm;
    logRss("decoder_t300");
    if (!s->mCH || !s->mCE || !s->mFE || !s->mCPF || !s->mCPS) return JNI_FALSE;
    s->lmW = (const float*)mmapFile(d+"lm_head_weight.f32", (size_t)15*2048*1024*sizeof(float), &s->lmMap);
    s->cpW = (const float*)mmapFile(d+"codec_emb_weight.f32", (size_t)15*2048*2048*sizeof(float), &s->cpMap);
    s->tRopeC = readVec(d+"talker_rope_cos.f32", (size_t)maxPos*HD);
    s->tRopeS = readVec(d+"talker_rope_sin.f32", (size_t)maxPos*HD);
    s->cRopeC = readVec(d+"codepred_rope_cos.f32", (size_t)CAP*HD);
    s->cRopeS = readVec(d+"codepred_rope_sin.f32", (size_t)CAP*HD);
    if (!s->tok.load(d+"tokenizer.bin")) { s->lastError = "tokenizer.bin load failed"; LOGI("TOK_LOAD_FAIL"); return JNI_FALSE; }
    if (!s->txtTab.load(d)) { s->lastError = "text tables load failed"; LOGI("TEXTTAB_LOAD_FAIL"); return JNI_FALSE; }
    // A2: 25MB 的表没必要读进匿名内存，改只读映射（数值完全相同，这里只换加载方式）。
    s->codecEmbMap = mmapFile(d+"talker_codec_emb.f32", (size_t)3072*2048*sizeof(float), &s->codecEmbMapH);
    s->codecEmbPtr = (const float*)s->codecEmbMap;
    if (!s->codecEmbPtr) { s->lastError = "talker_codec_emb.f32 missing"; return JNI_FALSE; }
    logRss("codecEmb(mmap)");
    logRss("tokenizer+texttab");
    LOGI("tokenizer + text tables loaded");
    if (s->lmW == nullptr || s->cpW == nullptr || s->tRopeC.empty() || s->tRopeS.empty() || s->cRopeC.empty() || s->cRopeS.empty()) {
        s->lastError = "weight table missing in " + d; return JNI_FALSE;
    }
    double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    LOGI("loaded backend=%s in %.0fms", s->backend.c_str(), ms);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_cosyvoice_app_CosyVoiceVoiceDesignNative_nativeRun(
        JNIEnv* env, jobject, jlong p, jstring jdir, jstring jout, jint maxFrames,
        jstring jinstruct, jstring jtext, jint jlang,
        jstring jenrollModelsDir, jstring jenrollOutDir) {
    VdSession* s = reinterpret_cast<VdSession*>(p);
    if (!s) return env->NewStringUTF("ERR no session");
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    const char* out = env->GetStringUTFChars(jout, nullptr);
    std::string enrollModelsDir, enrollOutDir;
    if (jenrollModelsDir) {
        const char* e = env->GetStringUTFChars(jenrollModelsDir, nullptr);
        enrollModelsDir = e; env->ReleaseStringUTFChars(jenrollModelsDir, e);
    }
    if (jenrollOutDir) {
        const char* e = env->GetStringUTFChars(jenrollOutDir, nullptr);
        enrollOutDir = e; env->ReleaseStringUTFChars(jenrollOutDir, e);
    }
    const bool doEnroll = !enrollModelsDir.empty() && !enrollOutDir.empty();
    std::string d(dir), o(out);
    env->ReleaseStringUTFChars(jdir, dir); env->ReleaseStringUTFChars(jout, out);
    if (!d.empty() && d.back() != '/') d += "/";
    std::lock_guard<std::mutex> lk(s->mu);
    // 进度文件：UI 侧轮询显示。一次设计要 4-9 分钟，没有反馈用户会以为卡死。
    g_progressPath = d + "progress.txt";
    char enrollInfo[128] = {0};
    logRss("nativeRun-entry");
    writeProgress("正在准备 prompt");
    LOGI("M1 strings ok");
    char buf[512];
    // ---- 设备端构造 prompt: tokenize -> text_embedding+projection -> 拼装 ----
    std::string instruct, text;
    if (jinstruct) { const char* c = env->GetStringUTFChars(jinstruct, nullptr); instruct = c; env->ReleaseStringUTFChars(jinstruct, c); }
    if (jtext) { const char* c = env->GetStringUTFChars(jtext, nullptr); text = c; env->ReleaseStringUTFChars(jtext, c); }
    std::vector<float> pe;
    int L = 0;
    if (!instruct.empty() || !text.empty()) {
        auto ii = s->tok.encode("<|im_start|>user\n" + instruct + "<|im_end|>\n");
        auto ai = s->tok.encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
        LOGI("M2 tokenized instruct=%zu assistant=%zu", ii.size(), ai.size());
        if (ai.size() < 9) return env->NewStringUTF("ERR assistant text too short");
        LOGI("M3 before pe.assign");
        pe.assign((size_t)512 * 2048, 0.f);
        LOGI("M4 pe assigned, calling buildPrompt");
        L = vdt::buildPrompt(s->tok, s->txtTab, ii, ai, (int)jlang, s->codecEmbPtr,
                             3072, pe.data(), 512);
        LOGI("M5 buildPrompt returned L=%d", L);
        if (L <= 0) return env->NewStringUTF("ERR prompt build failed");
        LOGI("prompt tokens=%d", L);
        { FILE* vf = fopen((d+"prompt_built.f32").c_str(), "wb");
          if (vf) { fwrite(pe.data(), 4, (size_t)L*2048, vf); fclose(vf); LOGI("M6 prompt dumped"); } else LOGI("M6 dump open failed"); }
    } else {
        FILE* f = fopen((d+"prompt_emb.f32").c_str(), "rb");
        if (!f) return env->NewStringUTF("ERR prompt_emb.f32 missing (and no text given)");
        fseek(f,0,SEEK_END); long nb = ftell(f); fseek(f,0,SEEK_SET);
        L = (int)(nb / (2048*4));
        pe.resize((size_t)L*2048);
        if (fread(pe.data(), 4, pe.size(), f) != pe.size()) { fclose(f); return env->NewStringUTF("ERR prompt_emb short"); }
        fclose(f);
    }
    LOGI("M7 allocating KV buffers");
    const size_t TKVONE = (size_t)NL*NKV*TKV*HD;
    const size_t CPKVONE = (size_t)CPNL*CPKV*CAP*HD;
    std::vector<float> tK(TKVONE,0.f), tV(TKVONE,0.f), sm(TKV,0.f), am(TKV,0.f);
    LOGI("M8 KV buffers ok");
    std::vector<float> cpK(CPKVONE,0.f), cpV(CPKVONE,0.f), csm(CAP,0.f), cam(CAP,0.f);
    VARP vEmb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vK=_Input({NL,1,NKV,TKV,HD},NCHW,halide_type_of<float>());
    VARP vV=_Input({NL,1,NKV,TKV,HD},NCHW,halide_type_of<float>());
    VARP vCos=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSin=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vMask=_Input({1,1,1,TKV},NCHW,halide_type_of<float>());
    VARP vSlot=_Input({1,1,TKV,1},NCHW,halide_type_of<float>());
    auto gbStep = [&](const float* emb, int cp) -> std::vector<float> {
        memcpy(vEmb->writeMap<float>(), emb, 2048*4);
        memcpy(vK->writeMap<float>(), tK.data(), TKVONE*4);
        memcpy(vV->writeMap<float>(), tV.data(), TKVONE*4);
        memcpy(vCos->writeMap<float>(), &s->tRopeC[(size_t)cp*HD], HD*4);
        memcpy(vSin->writeMap<float>(), &s->tRopeS[(size_t)cp*HD], HD*4);
        for (int i=0;i<TKV;i++) am[i]=(i<=cp)?0.f:-1e4f;
        memcpy(vMask->writeMap<float>(), am.data(), TKV*4);
        sm.assign(TKV,0.f); sm[cp]=1.f; memcpy(vSlot->writeMap<float>(), sm.data(), TKV*4);
        std::vector<VARP> o = s->mGB->onForward({vEmb,vK,vV,vCos,vSin,vMask,vSlot});
        if (!outOk(o, 3, "graphb")) return std::vector<float>();
        const float* h = o[0]->readMap<float>();
        std::vector<float> hb(h,h+2048);
        const float* ck = o[1]->readMap<float>(); const float* cv = o[2]->readMap<float>();
        for (int l=0;l<NL;l++) for (int hh=0;hh<NKV;hh++) {
            size_t off = ((size_t)(l*NKV+hh)*TKV + cp)*HD;
            memcpy(&tK[off], ck + ((size_t)(l*NKV+hh))*HD, HD*4);
            memcpy(&tV[off], cv + ((size_t)(l*NKV+hh))*HD, HD*4);
        }
        return hb;
    };
    LOGI("PREFILL_BEGIN prompt_tokens=%d backend=%s", L, s->backend.c_str());
    logRss("after-prompt-build");
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> hidden;
    for (int cp=0; cp<L; cp++) {
        hidden = gbStep(&pe[(size_t)cp*2048], cp);
        if (hidden.size() != 2048) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: graphb_prefill");
        if (cp % 5 == 0 || cp == L-1) LOGI("prefill %d/%d", cp+1, L);
    }
    double prefillMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count();
    logRss("after-prefill");
    writeProgress("prefill 完成（%d token，%.0f 秒），开始逐帧解码", L, prefillMs / 1000.0);
    LOGI("prefill %d tokens in %.0fms", L, prefillMs);
    // self-driven decode
    std::vector<float> lg(3072), lg2(2048), c0e(2048), cpH(1024), femb(2048);
    std::vector<int32_t> allCodes; int T=0, stopReason=-1; double decMs=0;
    std::vector<double> frameTimes;   // 每帧结束时刻，用于滑动窗口估速率
    std::vector<int> code0Hist;   // 供 repetition_penalty(1.05) 使用
    VARP vCH=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCE=_Input({1,1},NCHW,halide_type_of<int32_t>());
    VARP vPFa=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vPFb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vPFc=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vPFs=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vPFm0=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vPFm1=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vPFam=_Input({1,1,2,CAP},NCHW,halide_type_of<float>());
    VARP vSTk=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vSTk_=_Input({CPNL,1,CPKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vSTv=_Input({CPNL,1,CPKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vSTc=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSTs=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSTm=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vSTam=_Input({1,1,1,CAP},NCHW,halide_type_of<float>());
    VARP vFEi=_Input({1,16},NCHW,halide_type_of<int32_t>());
    // 逐阶段计时（单帧 20 次 forward，必须测出时间花在哪一段）
    double tCH=0, tCE=0, tCPF=0, tCPS=0, tFE=0, tGB=0, tMM=0, tSample=0;
    auto tick = [](){ return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count(); };
    for (int n=0;n<maxFrames;n++) {
        int cp = L + n;
        auto td0 = std::chrono::steady_clock::now();
        memcpy(vCH->writeMap<float>(), hidden.data(), 2048*4);
        auto _tCH = tick();
        std::vector<VARP> oc = s->mCH->onForward({vCH}); tCH += ms(_tCH);
        if (!outOk(oc, 1, "codec_head")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: codec_head");
        memcpy(lg.data(), oc[0]->readMap<float>(), 3072*4);
        auto _tS = tick();
        int code0 = sampleTopK(lg.data(), 3072, 0.9, 50, true, nextU(s->seed), 1.05, &code0Hist); tSample += ms(_tS);
        if (code0 == 2150) { stopReason = 0; break; }
        code0Hist.push_back(code0);
        { int32_t c=code0; memcpy(vCE->writeMap<int32_t>(), &c, 4); }
        auto _tCE = tick();
        std::vector<VARP> oe = s->mCE->onForward({vCE}); tCE += ms(_tCE);
        if (!outOk(oe, 1, "codec_emb")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: codec_emb");
        memcpy(c0e.data(), oe[0]->readMap<float>(), 2048*4);
        // codepred prefill
        memcpy(vPFa->writeMap<float>(), hidden.data(), 2048*4);
        memcpy(vPFb->writeMap<float>(), c0e.data(), 2048*4);
        memcpy(vPFc->writeMap<float>(), &s->cRopeC[0], 2*HD*4);
        memcpy(vPFs->writeMap<float>(), &s->cRopeS[0], 2*HD*4);
        csm.assign(CAP,0.f); csm[0]=1.f; memcpy(vPFm0->writeMap<float>(), csm.data(), CAP*4);
        csm.assign(CAP,0.f); csm[1]=1.f; memcpy(vPFm1->writeMap<float>(), csm.data(), CAP*4);
        { std::vector<float> a2((size_t)2*CAP,0.f);
          for (int r=0;r<2;r++) for (int i=0;i<CAP;i++) a2[(size_t)r*CAP+i]=(i<=r)?0.f:-1e4f;
          memcpy(vPFam->writeMap<float>(), a2.data(), (size_t)2*CAP*4); }
        auto _tCPF = tick();
        std::vector<VARP> op = s->mCPF->onForward({vPFa,vPFb,vPFc,vPFs,vPFm0,vPFm1,vPFam}); tCPF += ms(_tCPF);
        if (!outOk(op, 3, "codepred_prefill")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: codepred_prefill");
        memcpy(cpH.data(), op[0]->readMap<float>(), 1024*4);
        memcpy(cpK.data(), op[1]->readMap<float>(), CPKVONE*4);
        memcpy(cpV.data(), op[2]->readMap<float>(), CPKVONE*4);
        int32_t codes[16]; codes[0]=code0;
        auto _tMM0 = tick();
        for (int o2=0;o2<2048;o2++){ double acc=0; const float* w=&s->lmW[(size_t)o2*1024]; for(int i=0;i<1024;i++) acc+=(double)w[i]*cpH[i]; lg2[o2]=(float)acc; } tMM += ms(_tMM0);
        for (int g=0;g<15;g++) {
            auto _tS2 = tick(); codes[g+1] = sampleTopK(lg2.data(), 2048, 0.9, 50, false, nextU(s->seed)); tSample += ms(_tS2);
            if (g==14) break;
            const float* e = &s->cpW[((size_t)g*2048 + codes[g+1])*2048];
            int ccp = 2+g;
            memcpy(vSTk->writeMap<float>(), e, 2048*4);
            memcpy(vSTk_->writeMap<float>(), cpK.data(), CPKVONE*4);
            memcpy(vSTv->writeMap<float>(), cpV.data(), CPKVONE*4);
            memcpy(vSTc->writeMap<float>(), &s->cRopeC[(size_t)ccp*HD], HD*4);
            memcpy(vSTs->writeMap<float>(), &s->cRopeS[(size_t)ccp*HD], HD*4);
            csm.assign(CAP,0.f); csm[ccp]=1.f; memcpy(vSTm->writeMap<float>(), csm.data(), CAP*4);
            for (int i=0;i<CAP;i++) cam[i]=(i<=ccp)?0.f:-1e4f;
            memcpy(vSTam->writeMap<float>(), cam.data(), CAP*4);
            auto _tCPS = tick();
            std::vector<VARP> os_ = s->mCPS->onForward({vSTk,vSTk_,vSTv,vSTc,vSTs,vSTm,vSTam}); tCPS += ms(_tCPS);
        if (!outOk(os_, 3, "codepred_step")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: codepred_step");
            memcpy(cpH.data(), os_[0]->readMap<float>(), 1024*4);
            memcpy(cpK.data(), os_[1]->readMap<float>(), CPKVONE*4);
            memcpy(cpV.data(), os_[2]->readMap<float>(), CPKVONE*4);
            auto _tMM = tick();
            for (int o2=0;o2<2048;o2++){ double acc=0; const float* w=&s->lmW[(size_t)(g+1)*2048*1024 + (size_t)o2*1024]; for(int i=0;i<1024;i++) acc+=(double)w[i]*cpH[i]; lg2[o2]=(float)acc; } tMM += ms(_tMM);
        }
        memcpy(vFEi->writeMap<int32_t>(), codes, 16*4);
        auto _tFE = tick();
        std::vector<VARP> of = s->mFE->onForward({vFEi}); tFE += ms(_tFE);
        if (!outOk(of, 1, "frame_emb")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: frame_emb");
        memcpy(femb.data(), of[0]->readMap<float>(), 2048*4);
        auto _tGB = tick(); hidden = gbStep(femb.data(), cp); tGB += ms(_tGB);
        if (hidden.size() != 2048) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: graphb_decode");
        decMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-td0).count();
        if (n == 0) logRss("frame-0-done");
        else if (n == 4) logRss("frame-4-done");
        else if (n == 9) logRss("frame-9-done");
        allCodes.insert(allCodes.end(), codes, codes+16);
        T++;
        LOGI("frame %d code0=%d T=%d", n, code0, T+1);
        // 进度：解码阶段是整个流程最长的一段（占 60%+），必须让用户看到在动。
        // ETA 用「最近若干帧的速率」而不是全程平均 —— 前几帧含 OpenCL kernel 编译，
        // 用全程平均会把 ETA 高估十几倍（实测第 5 帧算出「还需 13 分 11 秒」，实际 45 秒）。
        {
            double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            frameTimes.push_back(now);
            int W = 20;
            double perFrame = 0.0;
            if ((int)frameTimes.size() >= 2) {
                int lo = std::max(0, (int)frameTimes.size() - 1 - W);
                double span = frameTimes.back() - frameTimes[lo];
                int cnt = (int)frameTimes.size() - 1 - lo;
                perFrame = cnt > 0 ? span / cnt : 0.0;
            }
            if (perFrame > 0.01) {
                int remain = std::max(1, 110 - (n + 1));   // 经验值：EOS 通常落在 ~100 帧附近
                int etaSec = (int)(perFrame * remain);
                writeProgress("解码中 %d 帧（约 %.2f 秒音频）· 已用 %.0f 秒 · 约 %.2f 秒/帧 · 预计还需 %d 分 %d 秒",
                             n + 1, (n + 1) * 0.08, now, perFrame, etaSec / 60, etaSec % 60);
            } else {
                writeProgress("解码中 %d 帧（约 %.2f 秒音频）· 已用 %.0f 秒 · 正在测速…",
                             n + 1, (n + 1) * 0.08, now);
            }
        }
    }
    if (T == 0) return env->NewStringUTF("ERR no frames generated");
    {
        const double tot = tCH+tCE+tCPF+tCPS+tFE+tGB+tMM+tSample;
        LOGI("PERF frames=%d total=%.0fms | codec_head=%.0f codec_emb=%.0f codepred_prefill=%.0f codepred_step=%.0f frame_emb=%.0f graphb=%.0f manualGEMM=%.0f sample=%.0f",
             T, tot, tCH, tCE, tCPF, tCPS, tFE, tGB, tMM, tSample);
        LOGI("PERF per-frame ms | CH=%.1f CE=%.1f CPF=%.1f CPS=%.1f FE=%.1f GB=%.1f MM=%.1f S=%.1f",
             T?tCH/T:0, T?tCE/T:0, T?tCPF/T:0, T?tCPS/T:0, T?tFE/T:0, T?tGB/T:0, T?tMM/T:0, T?tSample/T:0);
        {
            long fq[8]; int nf = 0; readCpuFreqKHz(fq, &nf);
            std::string fs;
            for (int i = 0; i < nf; ++i) fs += std::to_string(fq[i] / 1000) + "MHz,";
            long thr = sysconf(_SC_NPROCESSORS_ONLN);
            LOGI("PERF cpu | online=%ld freqs=[%s]", thr, fs.c_str());
        }
    }
    writeProgress("解码结束：共 %d 帧（%.2f 秒音频），%s。正在切换解码器…",
                  T, T * 0.08, stopReason == 0 ? "EOS 自然停止" : "达到上限");
    // ---- 实验：Decoder 独立 Runtime 隔离 ----
    // 上次「分阶段释放」失败（anon 只降 56MB）的根因是：主 rtm 仍被 s->rtm 持有，
    // Executor 内部的 buffer pool / 权重缓存不归还。这次**连 rtm 一起销毁**：
    //   ① 释放 6 个 talker Module  ② reset 主 rtm  ③ 用独立 rtm 加载【内嵌】decoder
    // 内嵌 decoder 的 forward 只要约 20s（外置因 _RebuildExternalOp 要 459-633s）。
    const bool kIsolateDecoder = true;
    if (kIsolateDecoder) {
        s->mGB.reset(); s->mCH.reset(); s->mCE.reset();
        s->mFE.reset(); s->mCPF.reset(); s->mCPS.reset();
        logRss("talker-modules-reset");
        s->rtm.reset();                       // ← 关键：连 RuntimeManager 一起销毁
        logRss("rtm-reset");

        MNN::BackendConfig decBackend;
        decBackend.precision = MNN::BackendConfig::Precision_High;
        decBackend.memory = MNN::BackendConfig::Memory_Normal;
        decBackend.power = MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig decSchedule;
        decSchedule.type = MNN_FORWARD_CPU;
        decSchedule.backupType = MNN_FORWARD_CPU;
        decSchedule.numThread = 4;
        decSchedule.backendConfig = &decBackend;
        std::shared_ptr<Executor::RuntimeManager> decRtm(
            Executor::RuntimeManager::createRuntimeManager(decSchedule),
            Executor::RuntimeManager::destroy);
        if (!decRtm) return env->NewStringUTF("ERR decoder runtime");
        const int NFR_ISO = 96;   // 必须与图一致（t96）
        FILE* decDiag = nullptr;
        writeProgress("正在加载内嵌波形解码器（约 20 秒的 forward，对比外置 7-10 分钟）…");
        std::string loadErr;
        // T=96（7.68 秒）而不是 T=300（24 秒）：decoder 的内部激活按 T 分配，
        // 实测 T=300 时 forward 峰值 anon +1.52 GB；T=96 应降到约 1/3。
        // 数值 Gate 早已通过（同一 codes 下 cos=1.000000000、int16 max 1 LSB）。
        s->mDEC = loadM(d + "tokenizer_decoder_static_t96.mnn",
                        {"codes"}, {"waveform"}, decRtm, loadErr);
        if (!s->mDEC) return env->NewStringUTF("ERR decoder load failed");
        s->decRtm = decRtm;                   // 保持到解码结束
        logRss("decoder-internal-loaded");
        // 运行时把 decoder 的真实签名打出来（不猜 ONNX/MNNConvert 的结果）
        {
            // 诊断写文件（logcat 环缓冲会把长行冲掉）
            FILE* df = fopen((d + "dec-diag.txt").c_str(), "w");
            auto emit = [&](const char* fmt, ...) {
                char lb[512]; va_list ap; va_start(ap, fmt);
                vsnprintf(lb, sizeof(lb), fmt, ap); va_end(ap);
                LOGI("%s", lb);
                if (df) { fprintf(df, "%s\n", lb); fflush(df); }
            };
            auto mi = s->mDEC->getInfo();
            if (mi) {
                std::string inN, outN;
                for (auto& n : mi->inputNames)  inN  += n + ",";
                for (auto& n : mi->outputNames) outN += n + ",";
                emit("DEC inputs=%zu inNames=[%s] outNames=[%s] defaultFormat=%d version=%s",
                     mi->inputs.size(), inN.c_str(), outN.c_str(), (int)mi->defaultFormat, mi->version.c_str());
                for (size_t i = 0; i < mi->inputs.size(); ++i) {
                    const auto& t = mi->inputs[i];
                    std::string ds;
                    for (int dv : t.dim) ds += std::to_string(dv) + ",";
                    emit("DEC IN[%zu] order=%d code=%d bits=%d dims=[%s] size=%zu",
                         i, (int)t.order, (int)t.type.code, (int)t.type.bits, ds.c_str(), t.size);
                }
                emit("DEC we-feed dims=[1,16,%d] int32 NCHW", NFR_ISO);
            } else {
                emit("DEC info NULL");
            }
            decDiag = df;   // 保持到 forward 之后
        }
        writeProgress("解码器就绪，正在生成波形…");
        const int SPF = 1920, SR = 24000;
        std::vector<int32_t> packedIso((size_t)16 * NFR_ISO, 0);
        for (int t = 0; t < NFR_ISO; t++) { int src = t < T ? t : T - 1; for (int g = 0; g < 16; g++) packedIso[(size_t)g * NFR_ISO + t] = allCodes[(size_t)src * 16 + g]; }
        VARP vDecIso = _Input({1, 16, NFR_ISO}, NCHW, halide_type_of<int32_t>());
        memcpy(vDecIso->writeMap<int32_t>(), packedIso.data(), packedIso.size() * 4);
        if (decDiag) { fprintf(decDiag, "BEFORE_FORWARD" "\n"); fflush(decDiag); }
        logRss("decoder-pre-forward");
        auto tdecIso = std::chrono::steady_clock::now();
        std::vector<VARP> odIso = s->mDEC->onForward({vDecIso});
        logRss("decoder-post-forward");
        if (decDiag) {
            fprintf(decDiag, "AFTER_FORWARD outputs=%zu\n", odIso.size());
            for (size_t i = 0; i < odIso.size(); ++i) {
                auto oi = odIso[i]->getInfo();
                if (oi) {
                    std::string ds; for (int dv : oi->dim) ds += std::to_string(dv) + ",";
                    fprintf(decDiag, "  OUT[%zu] code=%d bits=%d dims=[%s] size=%zu\n",
                            i, (int)oi->type.code, (int)oi->type.bits, ds.c_str(), oi->size);
                } else {
                    fprintf(decDiag, "  OUT[%zu] info=NULL\n", i);
                }
            }
            fflush(decDiag);
            fclose(decDiag); decDiag = nullptr;
        }
        if (!outOk(odIso, 1, "decoder-iso")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: decoder-iso");
        double decdMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tdecIso).count();
        logRss("decoder-forward-done");
        const float* wv = odIso[0]->readMap<float>();
        size_t wl = (size_t)(T > NFR_ISO ? NFR_ISO : T) * SPF;
        std::vector<float> wave(wv, wv + wl);
        // 解码器用完立刻回收（内嵌 457MB）
        s->mDEC.reset();
        s->decRtm.reset();
        logRss("decoder-released");
        writeProgress("波形已生成（%zu samples / %.2f 秒），正在写 reference.wav", wl, (double)wl / SR);
        writeWav16(o, wave, SR);
        int enrollCode = -1;
        if (doEnroll) {
            std::vector<float> enrollPcm(wave);
            cosy::quantizeInt16InPlace(enrollPcm);
            cosy::ditherInt16InPlace(enrollPcm);
            cosy::EnrollmentModels models;
            models.speechTokenizer = enrollModelsDir + "/speech-tokenizer-v3.fp32.inline.mnn";
            models.campPlus        = enrollModelsDir + "/campplus.fp32.mnn";
            models.affineWeight    = enrollModelsDir + "/flow-speaker-affine-weight.bin";
            models.affineBias      = enrollModelsDir + "/flow-speaker-affine-bias.bin";
            writeProgress("正在直接产出注册产物（speech tokens / prompt-cond / speaker）…");
            const auto eStart = std::chrono::steady_clock::now();
            const auto er = cosy::enrollFromPcm(enrollPcm.data(), enrollPcm.size(), SR, models, enrollOutDir, 6);
            enrollCode = er.code;
            const double eMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - eStart).count();
            if (er.code != 0) { LOGI("ENROLL_FAIL code=%d", er.code); snprintf(buf, sizeof(buf), "ERR ENROLL_FAIL code=%d", er.code); return env->NewStringUTF(buf); }
            LOGI("ENROLL_OK tokens=%d frames=%d totalMs=%.0f wallMs=%.0f", er.promptTokens, er.promptFrames, er.totalMs, eMs);
            snprintf(enrollInfo, sizeof(enrollInfo), "ENROLL_OK tokens=%d frames=%d ms=%.0f", er.promptTokens, er.promptFrames, eMs);
        }
        double mn = 1e9, mx = -1e9; size_t nan = 0;
        for (float v : wave) { if (!std::isfinite(v)) { nan++; continue; } if (v < mn) mn = v; if (v > mx) mx = v; }
        writeProgress("完成");
        snprintf(buf, sizeof(buf), "OK steps=%d stop=%s prefill=%.0fms decode=%.0fms decoder=%.0fms wav=%zu samples (%.2fs) finite=%s min=%.4f max=%.4f %s [decoder=INTERNAL+ISOLATED-RTM]",
                 T, stopReason == 0 ? "EOS" : "MAXFRAMES", prefillMs, decMs, decdMs, wl, (double)wl / SR,
                 nan == 0 ? "yes" : "NO", mn, mx, enrollInfo);
        LOGI("%s", buf);
        // 终态进度：否则 UI 会一直停在最后一条中间态上，看起来像卡死（实际已完成）。
        writeProgress("完成");
        return env->NewStringUTF(buf);
    }
    // 【分阶段加载已实测失败，撤回 2026-09-16】
    // 设想：mDEC 只在末尾用一次，与 talker 系生命周期不重叠 -> decode 后先释放 talker 系
    //       （预期 -450MB）再加载内嵌 decoder（+457MB），峰值不叠加，从而拿到内嵌的 20s 速度。
    // 实测：释放 mGB/mCH/mCE/mFE/mCPF/mCPS 六个模块后 anon 只降 56MB（575,448 -> 518,196），
    //       而 frame_emb 一步就占了 ~430MB。峰值 HWM 反而升到 1.93GB。
    // 原因：MNN 的 Module 析构不归还 session 权重内存（RuntimeManager 仍被持有，
    //       Executor 内部有 buffer pool / 权重缓存）。
    // 结论：此路不通，回到已验证可用的全外置配置。
    const int NFR = 96, SPF = 1920, SR = 24000;
    std::vector<int32_t> packed((size_t)16*NFR, 0);
    for (int t=0;t<NFR;t++){ int src = t<T?t:T-1; for(int g=0;g<16;g++) packed[(size_t)g*NFR+t]=allCodes[(size_t)src*16+g]; }
    VARP vDec=_Input({1,16,NFR},NCHW,halide_type_of<int32_t>());
    memcpy(vDec->writeMap<int32_t>(), packed.data(), packed.size()*4);
    auto tdec = std::chrono::steady_clock::now();
    std::vector<VARP> od = s->mDEC->onForward({vDec});
        if (!outOk(od, 1, "decoder")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: decoder");
    double decdMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tdec).count();
    const float* wv = od[0]->readMap<float>();
    size_t wl = (size_t)(T > NFR ? NFR : T) * SPF; if (wl > (size_t)NFR*SPF) wl = (size_t)NFR*SPF;
    std::vector<float> wave(wv, wv+wl);
    writeProgress("波形已生成（%zu samples / %.2f 秒），正在写 reference.wav", wl, (double)wl / SR);
    // reference.wav 始终写：它不是 enrollment 的传输介质，而是 VoiceIdentity 的长期资产
    // （将来换 enrollment 版本 / 语音 tokenizer / CAMPPlus / 迁移 Audio8 都不必再花一次
    //  数分钟的 VoiceDesign 生成成本）。
    writeWav16(o, wave, SR);

    // ---- 直接产出 VoiceProfile 注册产物（不经过 WAV 往返）----
    int enrollCode = -1;
    if (doEnroll) {
        // 数值纪律：先做一次内存内的「int16 编码 -> 解码」round-trip，
        // 使 enrollment 看到的数值与「writeWav16 落盘 -> MNN::AUDIO::load 读回」完全一致。
        // 已有 golden 绑定的是那条路径；先保 golden，等整链稳定后再单独评估直接用 float PCM。
        std::vector<float> enrollPcm(wave);
        cosy::quantizeInt16InPlace(enrollPcm);
        // dither 必须在 int16 round-trip 之后加：语义与「WAV 的 int16 字节 ±1 LSB」一致，
        // 保证不落 WAV 时 enrollment 看到的输入与落 WAV 路径相同（保住已有 golden）。
        cosy::ditherInt16InPlace(enrollPcm);
        cosy::EnrollmentModels models;
        models.speechTokenizer = enrollModelsDir + "/speech-tokenizer-v3.fp32.inline.mnn";
        models.campPlus        = enrollModelsDir + "/campplus.fp32.mnn";
        models.affineWeight    = enrollModelsDir + "/flow-speaker-affine-weight.bin";
        models.affineBias      = enrollModelsDir + "/flow-speaker-affine-bias.bin";
        writeProgress("正在直接产出注册产物（speech tokens / prompt-cond / speaker）…");
        const auto enrollStart = std::chrono::steady_clock::now();
        const auto er = cosy::enrollFromPcm(enrollPcm.data(), enrollPcm.size(), SR,
                                            models, enrollOutDir, 6);
        enrollCode = er.code;
        const double enrollMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - enrollStart).count();
        if (er.code != 0) {
            LOGI("ENROLL_FAIL code=%d", er.code);
            snprintf(buf, sizeof(buf), "ERR ENROLL_FAIL code=%d", er.code);
            return env->NewStringUTF(buf);
        }
        LOGI("ENROLL_OK tokens=%d frames=%d totalMs=%.0f wallMs=%.0f",
             er.promptTokens, er.promptFrames, er.totalMs, enrollMs);
        // 供 Kotlin 侧登记 profile 用
        snprintf(enrollInfo, sizeof(enrollInfo),
                 "ENROLL_OK tokens=%d frames=%d ms=%.0f", er.promptTokens, er.promptFrames, enrollMs);
    }
    double mn=1e9,mx=-1e9; size_t nan=0;
    for (float v : wave){ if(!std::isfinite(v)){nan++;continue;} if(v<mn)mn=v; if(v>mx)mx=v; }
    snprintf(buf,sizeof(buf),"OK steps=%d stop=%s prefill=%.0fms decode=%.0fms decoder=%.0fms wav=%zu samples (%.2fs) finite=%s min=%.4f max=%.4f %s",
             T, stopReason==0?"EOS":"MAXFRAMES", prefillMs, decMs, decdMs, wl, (double)wl/SR,
             nan==0?"yes":"NO", mn, mx, enrollInfo);
    LOGI("%s", buf);
    return env->NewStringUTF(buf);
}
