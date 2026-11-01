# llama-navi21-furnace

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for **AMD Navi 21 / RDNA2
(gfx1030)**, in particular the **Radeon PRO V620**: a 32 GB datacenter card that sells cheaply
second-hand and that upstream's HIP path treats as an afterthought. The goal is the best
single-card experience for a 27B-class model at long context, with everything measured on
real hardware and every kernel change bit-exact against upstream's arithmetic.

The reference setup is **Swift 1.5 Qwen3.8-27B** (UkisAI's fine-tune of Qwen3.8-27B) at Q4_K_XL
with a speculative drafter, 128k context, on one V620. A from-scratch deployment guide (BIOS,
kernel, ROCm, build, models, server, web tools) is in [deploy/README.md](deploy/README.md);
the quantized model and drafters are on Hugging Face at
[SixVolts/Swift-1.5-Qwen3.8-27B-GGUF](https://huggingface.co/SixVolts/Swift-1.5-Qwen3.8-27B-GGUF).

## Performance on one V620

Swift-Qwen3.8-27B Q4_K_XL (Swift 1.0, 16.6 GiB), flash attention, 128k context, card at 300 W.
Swift 1.5 has the same architecture and quantization and measures the same: with each drafter
its decode speed is within 1% of 1.0's (no drafter 23.5 against 23.9 t/s, MTP 48.5 against
47.9, DFlash2 48.0 against 48.4; 5 prompts at temperature 1.0).

Prefill and plain decode (`llama-bench`; depth = tokens already in the context):

| depth | pp512 t/s | tg32 t/s |
|---|---|---|
| 0 | 492 | 24.6 |
| 4k | 470 | 24.2 |
| 16k | 416 | 23.2 |
| 32k | 358 | 22.0 |
| 64k | 278 | 20.0 |
| 128k | 172 | 16.9 |

Decode with a drafter (`llama-server`, temperature 1.0 with speculative sampling, so the output
distribution is the model's own; prompt is real long text plus a summary request):

| depth | no drafter | MTP (3 tokens) | DFlash2 (adaptive, up to 7) |
|---|---|---|---|
| ~0 | 24.5 | 48.0 | 46.2 |
| 4k | 24.1 | 55.3 | 50.0 |
| 16k | 23.0 | 53.0 | 55.3 |
| 40k | 21.4 | 48.0 | 41.6 |
| 80k | 19.3 | 47.2 | 39.2 |
| 120k | 17.4 | 44.1 | 32.0 |

DFlash2 is the better drafter for short mixed work (math, code, lists: up to 76 t/s); MTP holds
its speed at depth because its cost does not grow with context. Details, methodology and the
history of every number are in [PROGRESS-phoebe.md](PROGRESS-phoebe.md).

## What is changed relative to upstream

The fork tracks upstream `master` (rebased, not merged; see the commit list for the exact base).
Everything below is scoped to HIP / RDNA2 or to the speculative-decoding code and leaves other
backends and GPUs byte-for-byte unchanged.

**Flash attention (tile kernel, `ggml/src/ggml-cuda/fattn-*.cuh`)**
- Occupancy is computed from the kernel's registers and LDS against the WGP instead of HIP's
  occupancy query, which assumes a quarter of RDNA2's register file and made upstream abort
  (`GGML_ASSERT(max_blocks_per_sm > 0)`) for head sizes 256 and 512.
- RDNA2 tile configurations for D=128/256/512/576 (Qwen, Gemma, DeepSeek MLA shapes) and a
  retune of the D=256 tiles for 3 to 8 token verify batches.
- The wide (prefill and verify) kernels read Q from a pre-converted global buffer instead of
  shared memory, which the K reads already saturate: +26 to +30% attention throughput at 16k to 64k
  context.

**Quantized matmul (MMQ, MMVQ)**
- The K-quant sub-block scale is applied as a float multiply on AMD (the integer multiply runs at
  quarter rate there): +8.5% prefill, bit-identical results.
- RDNA2 entries in the MMVQ tables and a 2-wave, wider block for the 2 to 8 column verify shape.
- The q8_1 quantization of the activation is reused across matmuls in one graph, with the cache
  bounded so it cannot overflow the pool under CUDA graphs.

**Graph fusions (CUDA/HIP)**
- Residual add fused into the following RMS norm and weight multiply; RMS norm fused with scale;
  `ADD -> SOFTPLUS -> MUL` fused; a run of same-shaped state-snapshot copies issued as one
  launch; CUDA graphs keyed by shape so batch-size changes do not evict each other.
- Qwen3.5 / gated delta-net: single-token graph fixes that remove a kernel per layer, the
  linear-attention state read from the cache instead of a gathered copy, q and k normalized in
  one op.

**Speculative decoding (`common/speculative.*`, `common/sampling.*`, server)**
- Speculative sampling (accept with probability min(1, p/q), resample the residual) for DFlash2
  and MTP drafts when the request samples at temperature > 0: about +15% at temperature 1.0 with
  no change to the output distribution. Enabled with `--spec-draft-temp 1.0`.
- Drafts are reproducible for a seeded request (per-request drafter RNG).
- Adaptive draft length from observed acceptance, reset per request.
- Reduced draft vocabulary (`d2t`) for the MTP head and DFlash drafters:
  `scripts/draft-vocab/build-draft-vocab.py` replaces a drafter's output head with the most
  likely rows of the target's own head, several times cheaper per draft step.

**Chat formats**
- Cohere Command A (`cohere2`) tool-call and thinking parsing (`common/parsers/cohere2.cpp`).

## Build (gfx1030)

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=OFF
cmake --build build -j
```

Ubuntu's ROCm packages install under `/usr` and hide the HIP cmake package; 
`deploy/scripts/build-llama.sh` carries the extra compiler and library paths that layout
needs, and is the exact build behind the numbers above.

## Running the reference setup

```sh
llama-server -m Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf -ngl 99 -fa on -c 131072 \
  -md dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-temp 1.0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

For long-context sessions use the MTP drafter instead:
`-md mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf --spec-type draft-mtp --spec-draft-n-max 3`.

## Documents

- [PROGRESS-phoebe.md](PROGRESS-phoebe.md): the working log of the single-V620 tuning, with
  every measurement, the negative results, and the reasoning behind each change.
- [NAVI21.md](NAVI21.md): the original gfx1030 flash-attention write-up (the abort, its root cause,
  the first fix and its validation across Qwen, Gemma and MoE models).
- [DEPLOY-NOTES.md](DEPLOY-NOTES.md): serving Qwen3.5-122B-A10B on a 4x V620 box.
- [COMMAND-A-PLUS-TUNING.md](COMMAND-A-PLUS-TUNING.md): tuning findings for Command A Plus on
  4x V620.

Upstream's README, build guide and tool documentation apply unchanged:
[docs/build.md](docs/build.md), [tools/server](tools/server/README.md),
[tools/cli](tools/cli/README.md).

## License

MIT, as upstream llama.cpp.
