#!/bin/bash
# llama-server for one Radeon PRO V620: Swift-Qwen3.8-27B Q4_K_XL with a speculative drafter,
# the web tools, the system prompt and the web UI defaults from llama-webui-tools.
#
#   deploy/serve-v620.sh [extra llama-server args...]
#
# Environment:
#   LLAMA_DIR   llama.cpp fork checkout      (default: the one containing this script)
#   TOOLS       llama-webui-tools checkout   (default: ~/llama-webui-tools)
#   MODELS      model directory              (default: ~/models)
#   SPEC        dflash | mtp | none          (default: dflash)
#                 dflash: DFlash2 block drafter, adaptive draft length up to 7.
#                         Fastest on mixed work (math, lists, code).
#                 mtp:    the model's multi-token-prediction head, 3 tokens per step.
#                         About 4% faster than dflash on research-style chat.
#   CTX         context length               (default: 131072; both drafters fit at full context)
#   THREADS     CPU threads                  (default: half of nproc, i.e. physical cores with SMT)
#   HOST        bind address                 (default: 127.0.0.1; 0.0.0.0 for the LAN)
#   PORT        port                         (default: 8080)
#   CORS        allowed web UI origins       (default: http://localhost:PORT,http://127.0.0.1:PORT)
#
# Run the tools repository's scripts/make-mcp-config.sh once first to create its config/mcp-servers.json.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}
TOOLS=${TOOLS:-$HOME/llama-webui-tools}
MODELS=${MODELS:-$HOME/models}
SPEC=${SPEC:-dflash}
CTX=${CTX:-131072}
THREADS=${THREADS:-$(( $(nproc) > 1 ? $(nproc) / 2 : 1 ))}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CORS=${CORS:-http://localhost:$PORT,http://127.0.0.1:$PORT}

MODEL=$MODELS/Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf

# --spec-draft-temp 1.0: when a request samples (temperature > 0), drafts are sampled from the
# drafter's distribution and verified with speculative sampling. Output is distributed exactly
# as without a drafter; about 15% faster than exact-match drafts at temperature 1.0.
case $SPEC in
  dflash) SPEC_ARGS=(-md "$MODELS/dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf" -ngld 99
                     --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-temp 1.0) ;;
  mtp)    SPEC_ARGS=(-md "$MODELS/mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf" -ngld 99
                     --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-temp 1.0) ;;
  none)   SPEC_ARGS=() ;;
  *)      echo "SPEC must be dflash, mtp or none" >&2; exit 1 ;;
esac

[ -f "$TOOLS/config/mcp-servers.json" ] || { echo "run $TOOLS/scripts/make-mcp-config.sh first (clone https://github.com/sixvolts/llama-webui-tools to $TOOLS)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL (see deploy/scripts/download-models.sh)" >&2; exit 1; }

# Ubuntu's ROCm lives under /usr
export ROCM_PATH=/usr HIP_PATH=/usr

# sampling defaults are the model card's; clients can override them per request
exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  -ngl 99 -fa on \
  -c "$CTX" -np 1 -t "$THREADS" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --host "$HOST" --port "$PORT" \
  --mcp-servers-config "$TOOLS/config/mcp-servers.json" \
  --ui-config-file "$TOOLS/config/webui-config.json" \
  --cors-origins "$CORS" \
  "$@"
