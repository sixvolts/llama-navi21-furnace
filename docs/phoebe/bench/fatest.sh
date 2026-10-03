#!/bin/bash
# usage: fatest.sh <tag> [extra CXX defines...]  -- builds build-dev ggml-hip with the current fattn-tile.cuh and compares facmp output to the old kernel
OUT=${OUT:-$PWD}; export OUT
tag=$1
cd "${LLAMA_DIR:-$HOME/llama-navi21-furnace}"/build-dev && ninja ggml-hip 2>&1 | grep -E "error|warning: unused" | head -20
for s in "8 4096" "512 4096"; do set -- $s
  LD_LIBRARY_PATH=${LLAMA_DIR:-$HOME/llama-navi21-furnace}/build-dev/bin $OUT/facmp $OUT/fa_${tag}_$1_$2.bin $1 $2 2>&1 | grep -v "^ggml_\|^load\|^$" | head -5
  python3 $OUT/cmpbin.py $OUT/fa_build_$1_$2.bin $OUT/fa_${tag}_$1_$2.bin
done
