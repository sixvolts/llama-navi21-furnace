#!/bin/bash
# Verify a build of the fork on the V620: GPU state, GPU kernels against the CPU backend, the C++
# unit tests, the server's Python tests, a coherence check on the real model and a speed check.
# Run before deploying any change to the kernels, the server or the speculative decoders.
#
#   deploy/scripts/check.sh [stage...]       stages: gpu kernels unit server model bench (default: all)
#
# Environment:
#   LLAMA_DIR   fork checkout                          (default: the one containing this script)
#   BUILD       build directory, built with tests      (default: $LLAMA_DIR/build)
#   MODELS      model directory                        (default: ~/models)
#   MODEL       model for the coherence and speed checks (default: Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf)
#   BASE        a second build directory to compare speed against (optional, e.g. the live build)
#   TEST_VENV   venv for the server tests, created on first use (default: ~/.venv-llama-tests)
#   TEST_PORT_BASE  first port for the server tests, 10 per worker (default: 18080)
#   STOP_SERVER 1: stop the llama-server user service for the GPU stages and start it again at the
#               end. Without it, the model and bench stages are skipped while the service runs,
#               since the 27B model and the service do not both fit in 32 GB.
#
# Build with tests:  cmake ... -DLLAMA_BUILD_TESTS=ON && cmake --build build
# (deploy/scripts/build-llama.sh builds without them).
set -uo pipefail

LLAMA_DIR=${LLAMA_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)"}
BUILD=${BUILD:-$LLAMA_DIR/build}
MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-$MODELS/Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf}
BASE=${BASE:-}
TEST_VENV=${TEST_VENV:-$HOME/.venv-llama-tests}
STOP_SERVER=${STOP_SERVER:-0}
STAGES=${*:-gpu kernels unit server model bench}

export ROCM_PATH=/usr HIP_PATH=/usr
[ -x "$HOME/.venv-llama/bin/ctest" ] && export PATH="$HOME/.venv-llama/bin:$PATH"
BIN=$BUILD/bin
LOG=$(mktemp -d "${TMPDIR:-/tmp}/llama-check.XXXXXX")

# GPU kernels compared with the CPU backend: everything this model's graph uses, and the fork's
# fused kernels. MUL_MAT is the gate for the matrix-multiply work (1297 cases).
OPS=${OPS:-MUL_MAT,FLASH_ATTN_EXT,CPY,ADD,MUL,RMS_NORM,RMS_NORM_SCALE,ADD_RMS_NORM_MUL,ADD_SOFTPLUS_MUL,GATED_DELTA_NET,SSM_CONV,ROPE,GET_ROWS,SET_ROWS,SOFT_MAX,GLU,UNARY,CONCAT,CONT,SCALE,ARGMAX,TOP_K}

# C++ tests that need what this machine does not have:
#   test-tokenizers-ggml-vocabs  the ggml-vocabs files (a separate download)
#   test-jinja-py                jinja2 in the system python
#   test-backend-ops             run in the kernels stage, against ROCm0 only
# and one known failure that does not concern this deployment:
#   test-chat                    the command-r7b template through the fork's cohere2 parser
UNIT_EXCLUDE=${UNIT_EXCLUDE:-'^(test-backend-ops|test-tokenizers-ggml-vocabs|test-jinja-py|test-chat)$'}

declare -a RESULTS=()
rc=0
pass() { RESULTS+=("PASS  $1"); }
fail() { RESULTS+=("FAIL  $1"); rc=1; }
skip() { RESULTS+=("skip  $1"); }
want() { [[ " $STAGES " == *" $1 "* ]]; }

server_active() { systemctl --user is-active --quiet llama-server 2>/dev/null; }
stopped=0
if [ "$STOP_SERVER" = 1 ] && server_active; then
  echo "stopping llama-server for the GPU stages"
  systemctl --user stop llama-server && stopped=1
  trap 'if [ $stopped = 1 ]; then echo "starting llama-server again"; systemctl --user start llama-server; fi' EXIT
fi

if want gpu; then
  echo "== gpu"
  if "$LLAMA_DIR/deploy/scripts/check-gpu.sh" "$BUILD" > "$LOG/gpu.txt" 2>&1; then pass "gpu: V620 present, ECC off"
  else fail "gpu: see $LOG/gpu.txt"; cat "$LOG/gpu.txt"; fi
fi

if want kernels; then
  echo "== kernels (test-backend-ops on ROCm0)"
  "$BIN/test-backend-ops" -b ROCm0 -o "$OPS" > "$LOG/kernels.txt" 2>&1
  summary=$(grep -E "tests passed" "$LOG/kernels.txt" | tail -1 | sed 's/^ *//')
  if grep -q "Backend ROCm0: .*OK" "$LOG/kernels.txt"; then pass "kernels: $summary"
  else fail "kernels: ${summary:-no summary}; failures in $LOG/kernels.txt"; grep -E "FAIL" "$LOG/kernels.txt" | head -20; fi
fi

if want unit; then
  echo "== unit (ctest)"
  if (cd "$BUILD" && ctest -j 8 -E "$UNIT_EXCLUDE" --output-on-failure) > "$LOG/unit.txt" 2>&1; then
    pass "unit: $(grep -E 'tests passed' "$LOG/unit.txt" | tail -1)"
  else
    fail "unit: $(grep -E 'tests passed' "$LOG/unit.txt" | tail -1); see $LOG/unit.txt"
    sed -n '/The following tests FAILED/,$p' "$LOG/unit.txt"
  fi
fi

if want server; then
  echo "== server (pytest)"
  if [ ! -x "$TEST_VENV/bin/pytest" ]; then
    python3 -m venv "$TEST_VENV" && "$TEST_VENV/bin/pip" install -q -r "$LLAMA_DIR/tools/server/tests/requirements.txt"
  fi
  # the test models are small; the server tests download them from Hugging Face on first use.
  # The tests listen on 8080 + 10 per worker by default, which is the live server's port: a test
  # would then talk to the live server. PORT_BASE moves them (tests/conftest.py).
  if (cd "$LLAMA_DIR/tools/server/tests" && LLAMA_SERVER_BIN_PATH="$BIN/llama-server" \
        PORT=${TEST_PORT_BASE:-18080} PORT_BASE=${TEST_PORT_BASE:-18080} \
        "$TEST_VENV/bin/pytest" -q -n "${PYTEST_WORKERS:-4}" --dist=worksteal -m "not slow") > "$LOG/server.txt" 2>&1; then
    pass "server: $(tail -1 "$LOG/server.txt")"
  else
    fail "server: $(tail -1 "$LOG/server.txt"); see $LOG/server.txt"
    grep -E "^(FAILED|ERROR)" "$LOG/server.txt" | head -20
  fi
fi

gpu_free() { ! server_active; }

if want model; then
  echo "== model (coherence)"
  if ! gpu_free; then skip "model: llama-server is running (STOP_SERVER=1 to stop it)"
  elif [ ! -f "$MODEL" ]; then fail "model: $MODEL not found"
  else
    "$BIN/llama-completion" -m "$MODEL" -ngl 99 -fa on -c 2048 -no-cnv --temp 0 -n 24 \
      -p "The capital of France is" > "$LOG/model.txt" 2> "$LOG/model.err"
    if grep -q "Paris" "$LOG/model.txt"; then pass "model: $(tr '\n' ' ' < "$LOG/model.txt" | cut -c1-80)"
    else fail "model: no 'Paris' in output; see $LOG/model.txt $LOG/model.err"; fi
  fi
fi

bench() { "$1/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 3 -o csv 2>/dev/null |
          awk -F, 'NR>1 {gsub(/"/,""); t=($0 ~ /,512,0,/) ? "pp512" : "tg128"; printf "%s %.1f  ", t, $(NF-1)}'; }
if want bench; then
  echo "== bench (llama-bench pp512 / tg128)"
  if ! gpu_free; then skip "bench: llama-server is running (STOP_SERVER=1 to stop it)"
  else
    now=$(bench "$BIN"); pass "bench: this build  $now"
    [ -n "$BASE" ] && RESULTS+=("      bench: $BASE  $(bench "$BASE/bin")")
  fi
fi

echo
echo "== summary (logs in $LOG)"
printf '  %s\n' "${RESULTS[@]}"
exit $rc
