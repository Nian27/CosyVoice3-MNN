# MNN fork 补丁（cosyvoice-npu 构建机）

## 这是什么

`app/src/main/jniLibs/arm64-v8a/libllm.so`（18,358,600 B，2026-08-25）与
`libcosy_llm_jni.so`（473,424 B，2026-08-25）**不是**上游 MNN 3.6.1 能编出来的：
它们依赖构建机上一份改过的 MNN 与一份更新过的 JNI 源码。本目录把这两处改动固化下来。

## 来源（2026-09-18 从构建机取回）

```text
WSL Ubuntu:  /home/vicentrent/work/cosyvoice-npu/MNN-3.6.1-build          ← MNN fork
             /home/vicentrent/work/cosyvoice-npu/CosyVoice3-MNN/mnn-jni   ← App 侧 JNI 源码
上游基线:     MNN d407447 "[Infra:Release] Update PyMNN packaging for 3.6.1 release (#4657)"
```

Windows 侧可经 `\\wsl$\Ubuntu\home\vicentrent\...` 直接读取（无需启动 WSL）。

## 文件

| 文件 | 内容 |
|---|---|
| `llm-engine.patch` | MNN LLM 引擎的全部改动：`llm.hpp` / `llm.cpp` / `llmconfig.hpp` / `sampler.cpp` / `sampler.hpp` |
| `backend-hexagon-qnn.patch` | Hexagon（含 `htp-ops-lib` DSP attention）与 QNN 后端的改动 |
| `CHANGED-FILES.txt` | 上述范围的 `git diff --stat` |

## LLM 引擎改了什么（`llm-engine.patch` 摘要）

1. **新增采样器 `cosyvoice_ras`**（`sampler.cpp` / `sampler.hpp` / `llmconfig.hpp`）：
   - `stepCosyVoiceRange`：把候选集限制在 `[151924, 151924+6561)` 这 6561 个语音 token；
     **EOS(`158486`) 只有在 `output_tokens.size() >= cosyVoiceMinTokens` 时才被放进候选集**。
   - 管线 = `stepCosyVoiceRange` → `stepTopK` → `stepTopP` → `stepSelect`。
   - **RAS 防重复**（`Sampler::sample` 尾部）：若刚选中的 token 出现在最近 `window`（=10）个输出里
     的次数 ≥ `window * tau`（= `10 * 0.1` = 1），就把该 token 置 `-inf` 重抽一次。
2. **`Llm::setCosyVoiceMinTokens`**：JNI 侧用它把「最少语音 token 数」传进采样器。
3. **Hexagon 也走 cpu 的 attention-mask 分支**（`llm.cpp` `Llm::load()`）：
   `backend_type == "cpu"` 放宽为 `cpu || hexagon` 且 `mValidBlockSize` 为空。

## 与 App 的接口

`mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp`（465 行）是 `libcosy_llm_jni.so` 的真源码；
`mnn-jni/CosyVoiceLlmJni.cpp` 只是把它包成 JNI。其中两处容易误读：

```cpp
// 两者都是【配置回显】，不是运行期测量：
const bool continuousHexagon = hexagonRuntime && !stagedHexagonLayers.empty();
const bool hybridNpuPrefill  = hexagonRuntime && !continuousHexagon;
// 于是报告里的这两个字段也恒等于上面的布尔：
//   "continuousHexagon"  : continuousHexagon
//   "npuFirstTokenValid" : hybridNpuPrefill
```

`hybridNpuPrefill == true` 时走的是**混合执行**：先用 NPU 配置的 LLM 生成 1 个 token
（拿它的 prefill 时间），再用 `config-cpu-*` 另起一个 LLM 实例、把 `input_ids + 第一个 token`
整句重新 prefill 后继续解码。也就是说 hybrid 的 decode 全在 CPU 上。

## 未完成

- 本补丁取自构建机的**工作副本**（未提交）。它是 2026-08-25 那批 `.so` 的最接近来源，
  但无法证明「工作副本 = 当时构建的确切版本」。
- `libcosy_voicedesign_jni.so` / `libcosy_enrollment_jni.so` 等的对应源码在 App 仓库侧，
  不在本目录。
