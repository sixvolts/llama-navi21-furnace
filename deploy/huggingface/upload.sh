#!/bin/bash
# Upload the model, the drafters and this card to Hugging Face. Run it yourself: it publishes.
#
#   HF_REPO=SixVolts/Swift-1.5-Qwen3.8-27B-GGUF MODELS=~/models deploy/huggingface/upload.sh
#
# Needs the Hugging Face CLI `hf` (installed into ~/.venv-llama by scripts/install-packages.sh)
# and a token with write access to the repo, either saved by `hf auth login` or passed as HF_TOKEN.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

HF=$(command -v hf || true); [ -x "$HOME/.venv-llama/bin/hf" ] && HF=$HOME/.venv-llama/bin/hf
[ -n "$HF" ] || { echo "hf CLI not found: run scripts/install-packages.sh or pip install huggingface_hub" >&2; exit 1; }
"$HF" auth whoami > /dev/null 2>&1 || { echo "not logged in: run '$HF auth login' (or export HF_TOKEN=...)" >&2; exit 1; }

HF_REPO=${HF_REPO:-SixVolts/Swift-1.5-Qwen3.8-27B-GGUF}
MODELS=${MODELS:-$HOME/models}
FILES=(Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf)

# the published files must be exactly the ones the card and SHA256SUMS describe
(cd "$MODELS" && sha256sum -c "$here/SHA256SUMS")

"$HF" repos create "$HF_REPO" --type model --exist-ok
# the license files and the card (Swift Open License section 4: recipients get the license,
# the notices, and a statement of what was changed - the card's "License and changes")
for f in README.md LICENSE LICENSE-APACHE-2.0 NOTICE SHA256SUMS; do
  "$HF" upload "$HF_REPO" "$here/$f" "$f"
done
for f in "${FILES[@]}"; do
  "$HF" upload "$HF_REPO" "$MODELS/$f" "$f"
done
