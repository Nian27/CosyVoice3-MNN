#!/system/bin/sh
echo "=== 进程 ==="
ps -A -o PID,NAME 2>/dev/null | grep cosyvoice
P=$(ps -A -o PID,NAME 2>/dev/null | grep com.cosyvoice.app.debug | awk '{print $1}' | head -1)
echo "PID=$P"
if [ -n "$P" ]; then
  awk '{print "T0 state="$3" utime="$14" stime="$15}' /proc/$P/stat
  sleep 12
  awk '{print "T1 state="$3" utime="$14" stime="$15}' /proc/$P/stat
  grep -E "VmRSS|VmHWM" /proc/$P/status
fi
echo "=== logcat VDS (最近 12 条) ==="
logcat -d -s VDS 2>/dev/null | tail -12
echo "=== logcat 该 app 任何日志 ==="
logcat -d 2>/dev/null | grep -a "cosyvoice" | tail -6
echo END
