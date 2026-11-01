#!/bin/bash
# Rebuild the model and both drafters from their upstream sources, instead of downloading the
# finished files (download-models.sh). Produces byte-identical files to the published ones with
# the same llama.cpp build and converter. Needs about 135 GB free: 56 GB of safetensors, a 54 GB
# BF16 conversion, 6 GB of drafter inputs and 19 GB of outputs.
# The conversion takes about 8 minutes and the quantization about 9 on a 20-thread CPU;
# everything else is download time.
#
#   deploy/scripts/make-models.sh
#
# Environment:
#   MODELS     working and output directory     (default: ~/models)
#   LLAMA_DIR  llama.cpp fork checkout, built   (default: the one containing this script)
#   PYTHON     python with numpy and pyyaml     (default: ~/.venv-llama/bin/python)
#   CONVERT_VENV  venv for convert_hf_to_gguf.py (needs PyTorch); created on first use with the
#              CPU build of PyTorch, about 1 GB  (default: ~/.venv-llama-convert)
#   HF_TOKEN   Hugging Face token (optional; the sources are public as of 2026-09-27)
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"

MODELS=${MODELS:-$HOME/models}
LLAMA_DIR=${LLAMA_DIR:-$repo}
PYTHON=${PYTHON:-$HOME/.venv-llama/bin/python}
CONVERT_VENV=${CONVERT_VENV:-$HOME/.venv-llama-convert}
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
BIN=$LLAMA_DIR/build/bin
THREADS=$(nproc)

mkdir -p "$MODELS"
cd "$MODELS"

get() { # <repo> <path in repo> [local name]
  local out=${3:-$(basename "$2")}
  [ -f "$out" ] && { echo "have $out"; return; }
  echo "== $out"
  curl -fL -C - ${auth[@]+"${auth[@]}"} -o "$out.part" "https://huggingface.co/$1/resolve/main/$2"
  mv "$out.part" "$out"
}

# 1. inputs. Swift 1.5 is published as safetensors only (its GGUF repository stops at Q8_0)
SRC=Swift-1.5-Qwen3.8-27b
mkdir -p "$SRC"
for f in config.json generation_config.json chat_template.jinja tokenizer.json tokenizer_config.json \
         vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json \
         model.safetensors.index.json; do
  get ukisai/$SRC "$f" "$SRC/$f"
done
for i in $(seq -w 1 18); do
  get ukisai/$SRC model-000$i-of-00018.safetensors "$SRC/model-000$i-of-00018.safetensors"   # the fine-tune, BF16
done
get unsloth/Qwen3.8-27B-GGUF imatrix_unsloth.gguf                              # Unsloth's imatrix (base Qwen3.8-27B)
get z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-BF16.gguf dflash-Qwen3.8-27B-BF16.gguf
get unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf                     # Unsloth's MTP drafter

export ROCM_PATH=/usr HIP_PATH=/usr

# 2. safetensors to a BF16 GGUF. --no-mtp leaves out the MTP head: it is the base model's,
#    unchanged by the fine-tune, and is used through Unsloth's separate MTP drafter (step 5)
if [ ! -f Swift-1.5-Qwen3.8-27B-BF16.gguf ]; then
  if [ ! -x "$CONVERT_VENV/bin/python" ]; then
    python3 -m venv "$CONVERT_VENV"
    "$CONVERT_VENV/bin/pip" install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
  fi
  PYTHONPATH="$LLAMA_DIR/gguf-py" "$CONVERT_VENV/bin/python" "$LLAMA_DIR/convert_hf_to_gguf.py" "$SRC" \
    --outtype bf16 --no-mtp --outfile Swift-1.5-Qwen3.8-27B-BF16.gguf.part
  mv Swift-1.5-Qwen3.8-27B-BF16.gguf.part Swift-1.5-Qwen3.8-27B-BF16.gguf
fi

# 3. the model: Unsloth's UD-Q4_K_XL per-tensor types with the IQ tensors as Q4_K, using
#    Unsloth's importance matrix for the base model. "did not find weights for output.weight"
#    is expected - the imatrix has no entry for it and it is kept at Q6_K.
if [ ! -f Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf ]; then
  "$BIN/llama-quantize" --imatrix imatrix_unsloth.gguf \
    --tensor-type-file "$repo/deploy/recipes/swift-q4_k_xl.txt" \
    --token-embedding-type q4_k --output-tensor-type q6_k \
    Swift-1.5-Qwen3.8-27B-BF16.gguf Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf.part Q4_K_M "$THREADS"
  mv Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf.part Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf
fi

# 4. DFlash2 drafter at Q4_0 (the fastest of the drafter formats tested on gfx1030; the drafter's precision
#    does not change output, only acceptance)
[ -f dflash-Qwen3.8-27B-Q4_0.gguf ] ||
  "$BIN/llama-quantize" dflash-Qwen3.8-27B-BF16.gguf dflash-Qwen3.8-27B-Q4_0.gguf Q4_0 "$THREADS"

# 5. reduced draft vocabulary: each drafter's LM head becomes the 65536 most likely rows of the
#    model's own output projection plus a d2t map. Several times cheaper per step, same output.
VOCAB="$PYTHON $LLAMA_DIR/scripts/draft-vocab/build-draft-vocab.py"
[ -f dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf ] ||
  $VOCAB dflash-Qwen3.8-27B-Q4_0.gguf Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf
[ -f mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf ] ||
  $VOCAB mtp-Qwen3.8-27B-Q4_0.gguf Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf

sha256sum Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf \
  mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf
echo
echo "done. the inputs ($SRC/, the BF16 conversion, BF16 drafter, imatrix) can be deleted to recover about 115 GB."
