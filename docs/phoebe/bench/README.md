# Benchmark and verification harnesses used for the V620 work

The tools behind the numbers in `PROGRESS-phoebe.md`. They assume the fork is built in
`build/` (or `build-dev/` where noted) and that `ggml` libraries are in `build/bin`. Scripts
take their paths from the environment: `LLAMA_DIR` (default `~/llama-navi21-furnace`),
`MODELS` (default `~/models`) and `OUT` (working directory for logs, prompts and dumps; default
the current directory).

## Server and system

| file | what it does |
|---|---|
| `srvbench.sh <none\|mtp\|dflash>` | decode t/s at several context depths through llama-server on port 8081, 256 sampled tokens, 2 runs per depth; prints prompt_n, prompt t/s, gen t/s, drafted/accepted. `DEPTHS` and `BYTES` select the depths and the prompt size. Needs `$OUT/long.txt`, a plain-text file of about 800 KB (we used wikitext). Source of the decode tables in sections 19 and 25. |
| `envelope.sh [curve\|sustained [cap]]` | power-cap curve (tg128, pp2048 and a DFlash run at 300 to 200 W) or a sustained 120k-token prefill, with the junction, memory and edge temperatures, clock and power sampled every 0.5 s. Writes the cap through hwmon with `sudo -n`; the DFlash step calls a `~/specbench/run.sh` of ours and can be removed. Sections 20 and 22. |
| `powersample.sh <tag>` | the sampler alone: power, clock and junction every 0.5 s until `<tag>.stop` exists. |
| `fatest.sh <tag>` | rebuilds `ggml-hip` in `build-dev`, runs `facmp` on two attention shapes and compares the dumps with a `build` baseline (`cmpbin.py`). The bit-exactness check for the attention work in section 18. |

## Kernel-level

| file | what it does |
|---|---|
| `facmp.cpp` | one FLASH_ATTN_EXT of this model's shape (D=256, GQA 24:4) on the GPU backend; dumps the output bytes. `FACMP_REPS` repeats for timing. Compare runs with `GGML_CUDA_FA_PARALLEL_BLOCKS` pinned, since the output is only bit-identical at the same KV split. |
| `mmcmp.cpp` | one MUL_MAT (quantized weights by N f32 columns); dumps the f32 output for bitwise comparison. `mmcmp <out.bin> <q4_K\|q5_K\|q6_K\|q8_0\|q4_0> <K> <M> <N> [seed]`. |
| `mmbench2.cpp` | NMAT independent matmuls with distinct weights (larger than the infinity cache) in one graph; time per matmul. `mmbench2 <type> <K> <M> <N> [NMAT=16] [REPS=10]`. Sections 23 and 24. |
| `repack_bench.hip` | the standalone version of the repacked-layout kernel (section 24) against a double-precision reference, on the model's shapes. |
| `cmpbin.py a.bin b.bin` / `reldiff.py a.bin b.bin` | identical-or-not with the count of differing elements, and the relative size of the differences. |
| `ktrace.py` / `perstep.py` | summarize a `rocprofv3 --kernel-trace` CSV by kernel class (section 20) and per decode step. |
| `isacount.py` | instruction census of a kernel from the `--save-temps` ISA (section 23). |
| `PLAN-limits-2026-09-29.md` | the plan reviewed in section 21. |

Build, from the fork's root:

```
g++ -O2 -std=c++17 -I ggml/include docs/phoebe/bench/mmcmp.cpp -L build/bin -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o mmcmp
# same for mmbench2.cpp and facmp.cpp
hipcc -O2 -std=c++17 --offload-arch=gfx1030 -I ggml/include -I ggml/src docs/phoebe/bench/repack_bench.hip \
  -L build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/bin -o repack_bench
```
