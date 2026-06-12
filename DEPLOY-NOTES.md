# Qwen3.5-122B Serving — Deployment Overview (Navi-21 / V620)

This repo's one code change is the gfx1030 flash-attention **tile-kernel config fix**
in `ggml/src/ggml-cuda/fattn-tile.cuh` (see `NAVI21.md`). This file records how the
122B is actually served on the 4× V620 box and the tuning behind it.

## Hardware
- 4× AMD Radeon Pro V620, 32 GB each (30704 MiB usable), gfx1030 / RDNA2, PCIe-only,
  ~512 GB/s GDDR6, 2 NUMA nodes. ROCm/HIP 7.1.

## Model
- Qwen3.5-122B-A10B: MoE, 122B total / ~10B active, 256 experts (8 used), 48 layers,
  head_dim 256, GQA 32:2.
- Hybrid attention: 36 gated-delta-net linear-attention layers + 12 full softmax-attention
  layers (`full_attention_interval=4`). KV grows only on the 12 full layers, so KV memory
  is ~1/4 of a naive estimate and long context is tractable.
- Production uses the **MTP variant** (`qwen3.5-122b-mtp`) for self-speculative decoding.

## The FA fix (why `-fa on` works here)
Stock llama.cpp **aborts** with `-fa on` on gfx1030 for head_dim 256/512 models: the FA
tile kernel's RDNA config rows were tuned for RDNA3/4 (WMMA), so on RDNA2 wave32 the
occupancy query returns 0 → `GGML_ASSERT(max_blocks_per_sm > 0)`. This repo repairs the
**tile kernel** (occupancy 3-8→1 for D=256, `ncols2≤2` cap, D=512 `cols_per_block≤8`), all
RDNA-guarded. Output is byte-equal to `-fa off`; perplexity within error bars.

> Superseded approach: an earlier local fork sidestepped the crash with a dispatch-level
> override (`fattn.cu`: force VEC for decode, NONE for prefill on RDNA2 D≥256). Head-to-head
> on the same tree showed that approach is **strictly worse** — it ties on decode
> (bandwidth-bound) but loses **−74% on prefill** (pp4096 519 vs 904 t/s), because the D=256
> VEC kernel is a poor fit for prefill batches. The tile fix is the keeper.

## Production config (systemd `llama-server.service`)
| Setting        | Value |
|----------------|-------|
| Binary         | built from this repo (gfx1030, ROCWMMA off) |
| GPUs           | 4 (`-ngl 999 --split-mode layer`) |
| Context        | `--ctx-size 262144`, `--parallel 4` → **4 slots × 64k** |
| KV cache       | **q8_0** (`-ctk q8_0 -ctv q8_0`) — fits 256k×4-slot; see KV notes |
| FlashAttention | `--flash-attn 1` (the tile fix makes this safe) |
| Batch          | `--batch-size 4096 --ubatch-size 2048` |
| Speculation    | `--spec-type draft-mtp --spec-draft-n-max 3` (MTP self-speculation) |
| Misc           | `--jinja --reasoning auto --threads 14` |

## Performance (verified, llama-bench, Q4_K_XL, 4 GPU, f16 KV, tile fix)
- pp512 ~615 t/s, pp4096 ~904 t/s, tg128 ~34 t/s.
- Real ~6.5k-token llama-cli inference: prefill ~771 t/s, decode ~32 t/s, clean.
- Prefill degrades with depth (D=256 FA kernel is the bottleneck at long context).

## KV cache notes (depth behavior)
- f16 K is graceful with depth; **q4_0 K craters** (27→8 t/s by 16k). Do not use q4_0 K.
- Production runs **q8_0** K+V as the capacity/quality middle ground needed to fit
  256k × 4 slots; f16 is preferred when memory allows.

## Speculation notes
- **MTP self-speculation (`draft-mtp`)** is the production lever — uses the model's own
  multi-token-prediction heads. (A separate 0.8B *draft* model did NOT help — the MoE target
  is already ~33 ms/token, so draft cost negates it.)
- **n-gram / prompt-lookup** (`--spec-type ngram-*`) also available: ~3× decode on
  repetitive/structured output (JSON edits, code), no benefit on free-form text, never hurts.

## Build
```
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=OFF
cmake --build build -j
```

## Operational notes
- `llama-bench`'s `-d` (depth) test aborts on this box in the KV state-restore path
  (`state_seq_set_data`), a **harness artifact** — NOT a kernel failure. Real long-context
  inference is fine. Benchmark depth with llama-cli on a real long prompt instead.
- `llama-bench` cannot auto-fit Q4 on a 3-GPU subset (GPU0 overflows); bench on 4 GPUs.
