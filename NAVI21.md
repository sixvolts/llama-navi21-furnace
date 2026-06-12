# llama-navi21-furnace

Mainline llama.cpp (`ggml-org/llama.cpp` @ `1593d5684`) **plus a flash-attention
fix for AMD Navi 21 / RDNA2 (gfx1030, e.g. Radeon Pro V620)**.

Everything else is stock upstream. The one change is in
`ggml/src/ggml-cuda/fattn-tile.cuh`.

## The bug

With `-fa on`, stock llama.cpp **aborts** on gfx1030 for models with head_dim
256 or 512 (Qwen3.5/Gemma-class attention):

```
fattn-common.cuh:1110: GGML_ASSERT(max_blocks_per_sm > 0) failed
  in launch_fattn<256, ...>  ->  ggml_cuda_flash_attn_ext_tile_case<256, 256>
```

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns 0 — the FA *tile*
kernel can't place even one block on a CU.

### Root cause

The FA tile kernel picks launch parameters (threads / occupancy / tile sizes)
from per-architecture config tables. RDNA3/RDNA4 have WMMA and use a *different*
FA kernel, so **gfx1030 (RDNA2) is the only RDNA card that ever reaches the tile
path** for these head dims — and its config rows were never validated there:

- **D=256:** the RDNA table demanded occupancy 3–8. RDNA2 is wave32; those
  `__launch_bounds__` min-block counts can't be met for DV=256, so the
  occupancy query returns 0. (GCN's wave64 values don't transfer.)
- **D=512** (Gemma global-attention layers, `key_length=512`): the lighter
  `ncols2<=2` kernels aren't instantiated for DV=512, and the `ncols2 in {4,8}`
  ones carry too many DV=512 accumulators for the wave32 VGPR budget.

## The fix (`fattn-tile.cuh`, RDNA-guarded)

1. **D=256 RDNA config:** keep upstream's tile geometry but set occupancy to 1
   and shrink the `ncols=32` `nbatch_fa` (64 → 32).
2. **D=256 dispatch:** cap `ncols2<=2` on RDNA so the GQA group runs in 2-head
   chunks (correct, lighter kernel) — the `ncols2>=4` variants overflow wave32.
3. **D=512 dispatch:** force `cols_per_block<=8` on RDNA so the DV=512
   accumulator set fits.

All three are gated on `GGML_CUDA_CC_IS_RDNA(cc)` / `__gfx906__` / `RDNA`, so
**GCN (gfx906/908) and RDNA3/4 are byte-for-byte unchanged.**

## Validation (Radeon Pro V620 / gfx1030)

Tested on Qwen3.5-27B, Qwen3.6-35B-A3B (MoE), Gemma-4 E4B and 31B:

- **No aborts** with `-fa on` (prefill + decode, f16 and Q8_0 KV).
- **Numerically correct vs `-fa off`** — perplexity (c=256) is within error bars:
  - Qwen3.5-27B: `3.9872 ± 0.24` (on) vs `3.9785 ± 0.24` (off) — 0.2%
  - Gemma-4 E4B: `18.66 ± 2.3` (on) vs `18.86 ± 2.3` (off)
  - Short greedy decode is token-identical to `-fa off` until normal
    floating-point drift (FA and non-FA accumulate in different orders).
- **Prefill speedup** from enabling FA: Qwen +2–7% (more at long context),
  Gemma +8–12%; decode unchanged (bandwidth-bound).

Note: `-fa on` only crashes the *tile* kernel path; this fix is config/dispatch
only, so the attention math is unchanged — hence the perplexity match.

## Build

```
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=OFF
cmake --build build -j
```
