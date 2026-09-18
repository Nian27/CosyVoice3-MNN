#!/system/bin/sh
PKG=com.cosyvoice.app.debug
S=/data/local/tmp/cosymodels
M=files/cosyvoice3-mnn/model
E=files/cosyvoice3-mnn/enrollment
run-as $PKG mkdir -p $M $E
put() { cat "$1" | run-as $PKG sh -c "cat > $2/$3" && echo "ok $3" || echo "FAIL $3"; }
echo "--- 朗读模型 (17) ---"
for f in config-cpu-cosyvoice-ras.json llm_config.json llm.mnn llm.mnn.weight embeddings_bf16.bin tokenizer.mtok flow.cfg-student-2step.batch1.fp16.mnn flow.cfg-student-2step.batch1.fp16.mnn.weight flow-conditioner.fp32.mnn prompt-speech-tokens.csv prompt-cond.bin spks.bin rand-noise.bin hift-f0.fp32.mnn hift-core.fp32.mnn source-linear-weight.bin source-linear-bias.bin; do
  put "$S/model/$f" "$M" "$f"
done
echo "--- 音色创建扩展 (5) ---"
for f in speech-tokenizer-v3.fp32.inline.mnn campplus.fp32.mnn campplus.fp32.mnn.weight flow-speaker-affine-weight.bin flow-speaker-affine-bias.bin; do
  put "$S/enrollment/$f" "$E" "$f"
done
echo "--- 完成 ---"
run-as $PKG sh -c "echo model=\$(ls files/cosyvoice3-mnn/model | wc -l) enrollment=\$(ls files/cosyvoice3-mnn/enrollment | wc -l) voicedesign=\$(ls files/cosyvoice3-mnn/voicedesign | wc -l)"
