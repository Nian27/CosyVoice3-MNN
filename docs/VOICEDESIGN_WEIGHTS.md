# VoiceDesign（文字设计音色）权重：现状与缺口

> 结论先行：**已经公开发布的 5 个权重分片是「独立样例工程 VoiceDesign-MNN」那一套，
> 和本 App（CosyVoice3-MNN 整合版）实际使用的那一套不是同一批文件。**
> 直接拿那 5 个分片喂本 App 不保证跑得起来。差异清单见下。

## 1. 已经公开的部分

`Nian27/VoiceDesign-MNN` Release **v1.0.0** 有 5 个权重分片（GraphB 权重 2.82 GB 超过 2 GB 上限，
拆成 part0/part1）+ 一个独立样例 APK。这 5 个分片承载 **21 个文件 / 4,812,549,603 B**，
模型是 `Qwen3-TTS-12Hz-1.7B-VoiceDesign`，逐个文件带 sha256（见该仓库 `release-manifest.json`）。

```text
vd-weights-1-graphb-mnn.zip            503,734 B
vd-weights-2-graphb-w-part0.zip    1,500,000,312 B
vd-weights-3-graphb-w-part1.zip    1,321,857,592 B
vd-weights-4-decoder.zip             944,514,162 B
vd-weights-5-hosttables.zip        1,045,679,777 B
```

本机 `E:/AndroidStudioProjects/vd-weights-staging/` 的 sha256 与该清单**逐条一致**（已核对 7 个关键文件），
`E:/AndroidStudioProjects/vd-weights-out/` 就是那 5 个 zip。**也就是说这一套是可复现的。**

## 2. 本 App 实际用的那一套（设备实测，25 个文件 / 4,812,979,079 B）

与上面那套的差异集中在 6 个文件：

| 设备上的文件 | 字节 | 公开那套里的对应物 | 差异 |
|---|---:|---|---|
| `tokenizer_decoder_static_t96.mnn` | 456,941,592 | `tokenizer_decoder_static_t300.mnn` 457,023,384 | **帧数不同（96 vs 300）**，且本机 `vd-weights-ext/` 里那个 t96 是 287,544 + 456,656,900 两段，和设备的单文件 456,941,592 **也不等** |
| `codepred_prefill_fp16.mnn` | 123,768 | `codepred_prefill_fp16.mnn` 161,953,004 | 设备是 `.mnn` + `.weight` 拆开 |
| `codepred_prefill_fp16.mnn.weight` | 161,830,912 | （并进上一行） | 合并后 161,954,680 ≠ 161,953,004 |
| `codepred_step_fp16.mnn` | 94,648 | `codepred_step_fp16.mnn` 161,923,900 | 同上 |
| `codepred_step_fp16.mnn.weight` | 161,830,912 | （并进上一行） | 合并后 161,925,560 ≠ 161,923,900 |
| `codec_head_fp16.mnn.weight` | 12,595,200 | （公开那套没有单独 weight） | 设备多一个 weight 文件 |

其余 19 个文件（`graphb28_v6_fp16.mnn` + `.weight`、`lm_head_weight.f32`、`codec_emb_weight.f32`、
`text_embedding.fp16`、`text_proj_fc1/fc2`（含 bias）、`talker_codec_emb.f32`、`talker_rope_cos/sin.f32`、
`codepred_rope_cos/sin.f32`、`prompt_emb.f32`、`tokenizer.bin`、`codec_emb_fp16.mnn`、`frame_emb_fp16.mnn`）
与公开那套**同名同大小**；其中 `graphb28_v6_fp16.mnn.weight`（2,821,857,280 B）本机 sha256 =
`f23dbc06646349f3fd90aece8aaf78cdb8f73693d069a6b8d4f0da1031750528`，与该仓库清单声明的**完全一致**
—— 也就是**那 2.82 GB 不需要重复上传**，直接用 v1.0.0 的 part0/part1 拼回来即可：

```bash
cat vd-weights-2-graphb-w-part0.zip 解出的 graphb28_v6_fp16.mnn.weight.part0 \
    vd-weights-3-graphb-w-part1.zip 解出的 graphb28_v6_fp16.mnn.weight.part1 \
  > graphb28_v6_fp16.mnn.weight
sha256sum graphb28_v6_fp16.mnn.weight   # 应为 f23dbc06…7528
```

本机已有的整合版文件：`vd-weights-ext/`（6 个差异文件，已逐个算出 sha256）、
`vd-weights-staging/`（其余 19 个）。**唯一本机没有的**是设备上那个单文件
`tokenizer_decoder_static_t96.mnn`（456,941,592 B），只能从设备拉。

## 3. 要发布给本 App 的话，只差一个 ~1.67 GB 的增量包

```text
增量包内容 = 19 个同名文件（除 graphb28_v6_fp16.mnn.weight 外的全部）
          + 6 个差异文件（vd-weights-ext/）
          + 设备上的 tokenizer_decoder_static_t96.mnn（需从设备拉 456,941,592 B）
共约 1.67 GB → 单个 <2 GB 附件，可以直接放本仓库 Release
graphb28_v6_fp16.mnn.weight（2.82 GB）复用 VoiceDesign-MNN v1.0.0 的 part0/part1
```

## 4. 未决

- **要不要发这个增量包**（约 1.67 GB 上传 + 从设备拉 457 MB）：等确认。
- **t96 还是 t300**：本 App 现在用 t96（96 帧静态解码），公开那套是 t300。两者不能混用；
  要么把本 App 固定回 t300（对得上已发布的包），要么把 t96 一起发出来。这是产品选择。
