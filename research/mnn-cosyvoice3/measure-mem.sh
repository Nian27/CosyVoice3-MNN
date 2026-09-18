#!/system/bin/sh
PKG=com.cosyvoice.app.debug
echo "=== 启动前基线 ==="
grep -E "MemFree|MemAvailable|^Cached" /proc/meminfo | head -3
echo
echo "=== 启动设计（前台服务） ==="
am force-stop $PKG
sleep 2
am start --user 0 -n $PKG/com.cosyvoice.app.MainActivity --ez autorunDesign true --es designName 测量 --es designBackend opencl > /dev/null 2>&1
i=0
while [ $i -lt 25 ]; do
  i=$((i+1))
  sleep 6
  P=$(ps -A -o PID,NAME 2>/dev/null | grep $PKG | awk '{print $1}' | head -1)
  if [ -z "$P" ]; then echo "t=$((i*6))s  进程已消失"; break; fi
  RSS=$(awk '/VmRSS/{print $2}' /proc/$P/status)
  ANON=$(awk '/RssAnon/{print $2}' /proc/$P/status)
  RFILE=$(awk '/RssFile/{print $2}' /proc/$P/status)
  SHR=$(awk '/RssShmem/{print $2}' /proc/$P/status)
  HWM=$(awk '/VmHWM/{print $2}' /proc/$P/status)
  MF=$(awk '/MemFree/{print $2}' /proc/meminfo)
  echo "t=$((i*6))s RSS=$RSS anon=$ANON file=$RFILE shmem=$SHR HWM=$HWM MemFree=$MF (kB)"
done
echo
echo "=== 死因 ==="
logcat -d -b events 2>/dev/null | grep -E "am_proc_died|am_kill" | grep cosyvoice | tail -3
echo END
