# CosyVoice3-MNN 项目协作规则

## 项目定位

- 项目：独立 Android CosyVoice3 MNN 客户端。
- 技术栈：Kotlin、Jetpack Compose、Android/Gradle、C++/JNI、MNN。
- 主目标：在不破坏音色和音频正确性的前提下，自动选择已验证的 CPU、GPU、NPU 后端。

## 工作约束

- 默认使用 Windows PowerShell；原生 Hexagon 构建可在现有 WSL Ubuntu 环境中完成。
- 构建缓存、临时产物和大模型放在 E 盘或 WSL 工作区，不占用 C 盘。
- 不提交模型权重、签名口令、API key 或其他凭据。
- 不把“能构建”“库能加载”当作 NPU 可用；必须分别验证后端命中、Token 合法、PCM 有限且非静音、听感和耗时。
- 音频错误属于发布阻断问题。连续 Hexagon 解码在未通过正确性验证前不得启用。
- 修改保持最小范围，不删除研究数据、模型、构建产物或用户未提交改动。

## 发布要求

- 正式版 APK、模型包和运行库必须来自同一版本清单。
- 发布前核对版本号、文件名、SHA-256、文件大小和真机结果。
- README 必须区分 App 版本、MNN 版本和模型包版本。
- 发布附件发现错误时，保留可追溯的纠错说明，并用经过校验的附件替换。
