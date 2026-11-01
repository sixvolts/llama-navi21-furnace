#!/bin/bash
# Download the model and both drafters (about 20 GB) into $MODELS.
#
#   deploy/scripts/download-models.sh
#
# Environment:
#   MODELS    target directory                (default: ~/models)
#   HF_REPO   Hugging Face repo with the files (default: SixVolts/Swift-1.5-Qwen3.8-27B-GGUF)
#   HF_TOKEN  access token, if the repo is gated
set -euo pipefail

MODELS=${MODELS:-$HOME/models}
HF_REPO=${HF_REPO:-SixVolts/Swift-1.5-Qwen3.8-27B-GGUF}
FILES=(
  Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf         # the model, 16.2 GiB
  dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf   # DFlash2 drafter, 1.3 GiB
  mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf           # MTP drafter, 1.0 GiB
  SHA256SUMS
)

mkdir -p "$MODELS"
cd "$MODELS"
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
for f in "${FILES[@]}"; do
  echo "== $f"
  # -C - resumes an interrupted download
  curl -fL -C - ${auth[@]+"${auth[@]}"} -o "$f" "https://huggingface.co/$HF_REPO/resolve/main/$f"
done
sha256sum -c SHA256SUMS
