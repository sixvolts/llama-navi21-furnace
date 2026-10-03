#!/bin/bash
# usage: srvbench.sh <none|mtp|dflash>  -> decode t/s at several context depths via llama-server
OUT=${OUT:-$PWD}; export OUT
MODE=$1; PORT=8081
cd "${LLAMA_DIR:-$HOME/llama-navi21-furnace}"; export ROCM_PATH=/usr HIP_PATH=/usr
M=${MODELS:-$HOME/models}
case $MODE in
  mtp)    SPEC=(-md $M/mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-temp 1.0 -ngld 99);;
  dflash) SPEC=(-md $M/dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-temp 1.0 -ngld 99);;
  none)   SPEC=();;
esac
./build/bin/llama-server -m $M/Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf "${SPEC[@]}" -ngl 99 -fa on -c 131072 -np 1 -t 8 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port $PORT > $OUT/srv_$MODE.log 2>&1 &
SRV=$!
for i in $(seq 1 200); do curl -s -m 2 localhost:$PORT/health 2>/dev/null | grep -q '"ok"' && break; sleep 2; done
echo "# mode=$MODE"
for depth in ${DEPTHS:-0 4096 16384 32768 65536 122880}; do
  if [ $depth -eq 0 ]; then
    prompt="Write a 400-word essay about the history of Wikipedia and how it is edited."
  else
    prompt="$(head -c ${BYTES:-$((depth*535/100))} $OUT/long.txt)

Write a 400-word summary of the text above."
  fi
  printf '%s' "$prompt" > $OUT/prompt.txt
  python3 - > $OUT/req.json <<'PY'
import json, os
p=open(os.environ['OUT']+'/prompt.txt').read()
print(json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":256,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"seed":42}))
PY
  for rep in 1 2; do
    curl -s -m 3600 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' --data-binary @$OUT/req.json > $OUT/resp.json
    python3 - $depth $rep <<'PY'
import json, sys, os
d=json.load(open(os.environ['OUT']+'/resp.json'))
t=d.get('timings',{})
print(f"depth={sys.argv[1]} rep={sys.argv[2]} prompt_n={t.get('prompt_n')} prompt_tps={t.get('prompt_per_second',0):.1f} gen_n={t.get('predicted_n')} gen_tps={t.get('predicted_per_second',0):.2f}", end=' ')
for k in ('draft_n','draft_n_accepted'):
    if k in t: print(f"{k}={t[k]}", end=' ')
print()
PY
  done
done
kill $SRV; wait $SRV 2>/dev/null
