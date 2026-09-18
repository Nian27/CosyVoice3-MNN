#!/system/bin/sh
PKG=com.cosyvoice.app.debug
P=$(ps -A -o PID,NAME 2>/dev/null | grep $PKG | awk '{print $1}' | head -1)
echo "PID=$P"
if [ -n "$P" ]; then
  awk '{print "state="$3" utime="$14}' /proc/$P/stat
  grep -E "VmRSS|Threads" /proc/$P/status
  echo "--- 关键线程栈 ---"
  kill -3 $P
  sleep 4
  logcat -d 2>/dev/null | grep -aE "CosyVoiceDesign|CosyVoiceRuntime|runBlocking|BlockingCoroutine|cosyvoice.*prio|main.*prio" | tail -5
  echo "--- 主线程栈片段 ---"
  logcat -d 2>/dev/null | grep -a -A14 '"main" prio' | tail -18
fi
echo END
