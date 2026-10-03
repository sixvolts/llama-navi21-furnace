#!/bin/bash
# sample GPU power, sclk, junction temp every 0.5 s until the file $1.stop exists
H=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon* | head -1)
while [ ! -f "$1.stop" ]; do
  p=$(cat $H/power1_average 2>/dev/null || cat $H/power1_input); s=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | tr -dc '0-9'); t=$(cat $H/temp2_input 2>/dev/null)
  echo "$(date +%s.%N) $((p/1000000)) $s $((t/1000))"; sleep 0.5
done
