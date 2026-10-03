#!/bin/bash
# Item 2: power/thermal envelope. Needs passwordless sudo for the cap writes (checked at start).
# 1) cap curve: tg128, pp2048, DFlash math (greedy) at 300/275/250/225/200 W, with hwmon sampling
# 2) sustained prefill: 120k-token prompt through llama-server at the chosen cap, sampling junction/mem/sclk/power
OUT=${OUT:-$PWD}; export OUT
H=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon* | head -1)
cd "${LLAMA_DIR:-$HOME/llama-navi21-furnace}"; export ROCM_PATH=/usr HIP_PATH=/usr
M=$HOME/models/Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf
setcap() { echo $(( $1 * 1000000 )) | sudo -n tee $H/power1_cap > /dev/null && echo "cap set to $1 W ($(( $(cat $H/power1_cap) / 1000000 )))"; }
sample() { # $1 = tag ; runs until $OUT/env_$1.stop exists
  rm -f $OUT/env_$1.stop; ( while [ ! -f $OUT/env_$1.stop ]; do echo "$(date +%s.%N) $(( $(cat $H/power1_average) / 1000000 )) $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | sed 's/.*: *\([0-9]*\)Mhz.*/\1/') $(( $(cat $H/temp2_input) / 1000 )) $(( $(cat $H/temp3_input) / 1000 )) $(( $(cat $H/temp1_input) / 1000 ))"; sleep 0.5; done ) > $OUT/env_$1.txt & echo $!; }
summarize() { python3 - $OUT/env_$1.txt <<'PY'
import sys, statistics as st
r=[l.split() for l in open(sys.argv[1]) if len(l.split())>=5]; r=[x for x in r if int(x[1])>80]
if r: print(f"   power mean {st.mean(int(x[1]) for x in r):.0f} W max {max(int(x[1]) for x in r)}, sclk median {st.median(int(x[2]) for x in r):.0f} min {min(int(x[2]) for x in r)}, junction max {max(int(x[3]) for x in r)} C, mem max {max(int(x[4]) for x in r)} C, edge max {max(int(x[5]) for x in r if len(x)>5) if any(len(x)>5 for x in r) else '?'} C, {len(r)} samples")
PY
}
MODE=${1:-curve}
if [ "$MODE" = curve ]; then
  for cap in 300 275 250 225 200; do
    setcap $cap || exit 1
    echo "== cap $cap W"
    p=$(sample tg$cap); ./build-dev/bin/llama-bench -m $M -fa 1 -ngl 99 -t 8 -p 0 -n 128 -r 2 2>/dev/null | grep -E "\| *tg128 *\|" | sed -E 's/.*\| *(tg128) *\| *([0-9.]+) ±.*/   \1 = \2 t\/s/'; touch $OUT/env_tg$cap.stop; wait $p; summarize tg$cap
    p=$(sample pp$cap); ./build-dev/bin/llama-bench -m $M -fa 1 -ngl 99 -t 8 -p 2048 -n 0 -r 3 2>/dev/null | grep -E "\| *pp2048 *\|" | sed -E 's/.*\| *(pp2048) *\| *([0-9.]+) ±.*/   \1 = \2 t\/s/'; touch $OUT/env_pp$cap.stop; wait $p; summarize pp$cap
    p=$(sample df$cap); echo -n "   dflash math: "; BIN=./build-dev/bin/llama-cli $HOME/specbench/run.sh dflash base 7 p_math -n 256 -md $HOME/models/dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf -m $M 2>&1 | grep -oE "Generation: [0-9.]+ t/s"; touch $OUT/env_df$cap.stop; wait $p; summarize df$cap
  done
  setcap 300
else
  # sustained: server at 128k, one 120k-token prompt (532000 bytes of wikitext), sample throughout
  cap=${2:-300}; setcap $cap || exit 1
  ./build-dev/bin/llama-server -m $M -ngl 99 -fa on -c 131072 -np 1 -t 8 --host 127.0.0.1 --port 8081 > $OUT/env_srv.log 2>&1 & SRV=$!
  for i in $(seq 1 200); do curl -s -m 2 localhost:8081/health 2>/dev/null | grep -q '"ok"' && break; sleep 2; done
  p=$(sample sus$cap)
  printf '%s' "$(head -c 532000 $OUT/long.txt)

Write a 400-word summary of the text above." > $OUT/prompt.txt
  python3 - > $OUT/req.json <<'PY'
import json, os; p=open(os.environ['OUT']+'/prompt.txt').read()
print(json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":64,"temperature":0}))
PY
  t0=$(date +%s); curl -s -m 3600 localhost:8081/v1/chat/completions -H 'Content-Type: application/json' --data-binary @$OUT/req.json | python3 -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f'   prompt_n={t[\"prompt_n\"]} prompt {t[\"prompt_per_second\"]:.1f} t/s, gen {t[\"predicted_per_second\"]:.1f} t/s')"; echo "   wall $(( $(date +%s) - t0 )) s"
  touch $OUT/env_sus$cap.stop; wait $p; summarize sus$cap
  echo "   junction over time (every 30 s):"; awk 'NR%60==1{printf "     %ds: %sW sclk %s junction %sC mem %sC edge %sC\n", NR/2, $2, $3, $4, $5, $6}' $OUT/env_sus$cap.txt
  grep -E "prompt processing, n_tokens" $OUT/env_srv.log | awk 'NR%40==0{print "     " $0}' | sed 's/.*n_tokens = */     n_tokens=/' | cut -c1-90
  kill $SRV; wait $SRV 2>/dev/null
fi
