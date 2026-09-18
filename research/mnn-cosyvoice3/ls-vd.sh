#!/system/bin/sh
PKG=com.cosyvoice.app.debug
D=files/cosyvoice3-mnn/voicedesign
echo "=== voicedesign/ 内容（按修改时间） ==="
run-as $PKG sh -c "cd $D && ls -lat" 2>&1 | head -34
echo
echo "=== design-loop.log 末尾 ==="
run-as $PKG sh -c "tail -12 $D/design-loop.log" 2>&1
echo
echo "=== 哪个文件是 cache ==="
run-as $PKG sh -c "cd $D && ls -la | grep -i cache" 2>&1
echo END
