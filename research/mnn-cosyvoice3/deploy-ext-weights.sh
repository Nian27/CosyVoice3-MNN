#!/system/bin/sh
PKG=com.cosyvoice.app.debug
DST=files/cosyvoice3-mnn/voicedesign
E=/data/local/tmp/vdext
run-as $PKG mkdir -p $DST
put() { cat "$1" | run-as $PKG sh -c "cat > $DST/$2" && echo "ok $2" || echo "FAIL $2"; }
put $E/codec_head_fp16.mnn            codec_head_fp16.mnn
put $E/codec_head_fp16.mnn.weight     codec_head_fp16.mnn.weight
put $E/codepred_prefill_fp16.mnn      codepred_prefill_fp16.mnn
put $E/codepred_prefill_fp16.mnn.weight codepred_prefill_fp16.mnn.weight
put $E/codepred_step_fp16.mnn         codepred_step_fp16.mnn
put $E/codepred_step_fp16.mnn.weight  codepred_step_fp16.mnn.weight
put $E/tokenizer_decoder_static_t96.mnn        tokenizer_decoder_static_t96.mnn
put $E/tokenizer_decoder_static_t96.mnn.weight tokenizer_decoder_static_t96.mnn.weight
echo "--- done ---"
run-as $PKG sh -c "cd $DST && ls -la codec_head* codepred_prefill* codepred_step* tokenizer_decoder_static_t96* | awk '{print \$5, \$9}'"
