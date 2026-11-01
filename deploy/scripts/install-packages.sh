#!/bin/bash
# Ubuntu 26.04 packages for building and running llama.cpp on a Radeon PRO V620 (gfx1030).
# Uses Ubuntu's own ROCm 7.1 packages (installed under /usr, not /opt/rocm); no AMD repository
# is needed. Needs sudo. Log out and back in afterwards so the new groups take effect.
set -euo pipefail

sudo apt update
# rocm         Ubuntu's ROCm metapackage: HIP runtime and compiler, rocBLAS/hipBLAS, rocminfo
#              (rocm-dev alone is enough to build; the tested machine has the full metapackage)
# clang-21...  the HIP compiler llama.cpp is built with, and its device libraries
# rocm-smi     GPU clocks, temperature and VRAM use
# libssl-dev   HTTPS in llama-server; the server tests in scripts/check.sh download their models
sudo apt install -y \
  rocm rocm-smi rocm-device-libs-21 \
  clang-21 lld-21 libclang-rt-21-dev \
  build-essential git curl pciutils libssl-dev \
  python3-venv python3-pip

# GPU access for your user (render, video), and kernel log access for check-gpu.sh (adm)
sudo usermod -aG render,video,adm "$USER"

# the build was tested with cmake 3.31.6; Ubuntu 26.04's apt cmake (4.2) has not been tried.
# a private venv keeps the tested version without touching the system.
python3 -m venv "$HOME/.venv-llama"
"$HOME/.venv-llama/bin/pip" install --upgrade pip
"$HOME/.venv-llama/bin/pip" install cmake==3.31.6 ninja numpy pyyaml huggingface_hub

echo
echo "done. log out and back in (group changes), then run deploy/scripts/check-gpu.sh"
