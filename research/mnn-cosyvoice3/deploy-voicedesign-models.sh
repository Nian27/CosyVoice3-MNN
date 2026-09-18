#!/system/bin/sh
PKG=com.cosyvoice.app.debug
DST=files/cosyvoice3-mnn/voicedesign
V=/data/local/tmp/qwen3tts/vdfull
T=/data/local/tmp/vdtok
run-as $PKG mkdir -p $DST
put() { cat "$1" | run-as $PKG sh -c "cat > $DST/$2" && echo "ok $2" || echo "FAIL $2"; }
put $T/tokenizer.bin          tokenizer.bin
put $T/text_proj_fc1.fp16     text_proj_fc1.fp16
put $T/text_proj_fc2.fp16     text_proj_fc2.fp16
put $T/text_proj_fc1_bias.f32 text_proj_fc1_bias.f32
put $T/text_proj_fc2_bias.f32 text_proj_fc2_bias.f32
put $T/talker_codec_emb.f32   talker_codec_emb.f32
put $V/codec_head_fp16.mnn    codec_head_fp16.mnn
put $V/codec_emb_fp16.mnn     codec_emb_fp16.mnn
put $V/graphb28_v6_fp16.mnn   graphb28_v6_fp16.mnn
put $V/lm_head_weight.f32     lm_head_weight.f32
put $V/codec_emb_weight.f32   codec_emb_weight.f32
put $V/frame_emb_fp16.mnn     frame_emb_fp16.mnn
put $V/prompt_emb.f32         prompt_emb.f32
put $V/talker_rope_cos.f32    talker_rope_cos.f32
put $V/talker_rope_sin.f32    talker_rope_sin.f32
put $V/codepred_rope_cos.f32  codepred_rope_cos.f32
put $V/codepred_rope_sin.f32  codepred_rope_sin.f32
put $V/codepred_prefill_fp16.mnn codepred_prefill_fp16.mnn
put $V/codepred_step_fp16.mnn    codepred_step_fp16.mnn
put $V/tokenizer_decoder_static_t300.mnn tokenizer_decoder_static_t300.mnn
put $T/text_embedding.fp16    text_embedding.fp16
put $V/graphb28_v6_fp16.mnn.weight graphb28_v6_fp16.mnn.weight
echo "--- 部署完成 ---"
run-as $PKG ls $DST | wc -l
