# Command A Plus on Furnace (4× V620) — Tuning Findings

Companion to `COMMAND-A-PLUS-FURNACE.md`. Records what we measured and concluded
tuning Command A Plus on the 4× Radeon Pro V620 box (gfx1030 / RDNA2), June 2026.
Supersedes the "What's Broken" and "What Needs to Happen" sections of the briefing.

## TL;DR

1. **The flash-attention crash is already fixed.** The `GGML_ASSERT(max_blocks_per_sm > 0)`
   abort at 128K was on an *older* build. The current `~/llama-command-a` build
   (upstream `581e8eca8`, Jun-15) added a dedicated `get_config_amd_rdna()` FA config
   function that resolves it. **No kernel patch needed.** Verified: full 130,642-token
   (128K) prefill + decode, `-fa on`, F16 KV — clean, no assert.
2. **The real blocker is prefill speed, not the crash.** 128K prefill takes **~5.9 hours**
   (6.2 t/s). This is fundamental to the hardware and is *not* tunable away (see below).
3. **Every "easy" optimization lever is a dead end on this hardware** (tensor-parallel
   blocked, Q8 KV backfires, kernel config already optimal). Best config is
   **F16 KV + `ub=2048` + `-sm layer`**.

## Model

`cohere2moe` (CohereLabs command-a-plus-05-2026): 218B total / 25B active MoE.
- 32 layers, 128 experts (8 active + 4 shared), sigmoid routing
- head_dim **128**, GQA **16:1** (128 heads / 8 KV heads)
- **Sliding-window attention, window 4096, on 24/32 layers** → only **8 full-attention layers**
- ctx_train 200000; quant Q3_K_XL (~108 GB on disk, ~100.8 GiB loaded)

## Build

- `~/llama-command-a/`, upstream base `581e8eca8` + `cohere2-chat-handler.patch`
- `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_PEER_COPY=ON`
- Note: `cohere2moe` support exists only in this newer tree, **not** in the older
  `~/llama-navi21-furnace` (Qwen) tree — they can't be unified without an upstream bump.

## Measured performance (4× V620, F16 KV, `ub=2048`, `-sm layer`, `-fa on`)

| test | t/s | notes |
|---|---|---|
| pp2048 | 122 | prefill |
| pp4096 | 100 | prefill (~41 s for 4096 tokens) |
| pp8192 | 82 | |
| pp16384 | 62 | decaying — O(n²) attention |
| **pp 128K** | **6.2** | **~5.9 HOURS** for a full 130K prefill |
| tg1024 (shallow) | 16.3 | decode from ~empty context |
| pp4096 + tg1024 (end-to-end) | 24.4 | decode degrades with depth; budget ~2–3× shallow |

Prefill-vs-depth ≈ 122 → 100 → 82 → 62 → … → 6 t/s. Decode also degrades materially
with context depth (the shallow 16.3 t/s is best case).

## Optimization levers — results

### 1. Tensor parallelism (`-sm row`) — BLOCKED ❌
`-sm row` **crashes** (ROCm error in the cross-GPU reduction). The V620s have **no
peer-to-peer** (rocm-smi bandwidth matrix is all N/A; the build sets
`GGML_CUDA_NO_PEER_COPY=ON` for this reason). So a single prefill stream **cannot be
spread across the 4 GPUs** — `-sm layer` pipelines them, leaving **~1 GPU active at a
time**. This is *the* prefill bottleneck and it is unfixable on this hardware.

### 2. Q8 KV cache — BACKFIRES for speed ❌
| KV | pp2048 | pp8192 |
|---|---|---|
| F16 | 122 | 82 |
| q8_0 V | 55 | **20** (~4× slower) |

In-kernel dequant cost dominates. Q8 KV is a **memory** tool only (to fit 3×128K), at a
steep prefill penalty. **Keep F16 KV.** (Same pattern as quantized KV on the Qwen side.)

### 3. `ub` (ubatch) — modest win ✅
`ub=2048` vs `512`: **+53% at pp2048, +13% at pp8192** — the gain fades with depth.
Free; keep `ub=2048`.

### 4. D=128 RDNA2 FA tile-config tuning — NEGATIVE ❌
Hot prefill row identified analytically + confirmed by measurement:
`fattn-tile.cuh:283` → `CASE(128, 128, 64, 256, 3, 64, 64)` (cols_per_block=64 is selected
for HIP `DKQ≤128` large-batch prefill). Sweep (baseline pp8192 = 82):

| config (nthreads, occupancy, nbatch_fa) | pp8192 |
|---|---|
| **256, 3, 64 (baseline)** | **82** |
| 256, 2, 128 | 72 (worse) |
| 256, 4, 32 | 81 (same) |
| 512, 3, 64 | 72 (worse) |

**Nothing beats baseline** — the config is already near-optimal. The bottleneck is not
the kernel config; it's the 1-of-4-GPU utilization (lever #1) plus RDNA2 having no
matrix cores.

## Why prefill is fundamentally slow (and can't be fixed here)
1. **No tensor parallelism** (no P2P) → only ~1 of 4 GPUs works on a single prefill stream.
2. **No matrix cores** (RDNA2: no WMMA, no MFMA) → the FA tile kernel is the only path.
3. **O(n²) attention** in the 8 full-attention layers dominates at high context.

None of these is config-addressable.

## Hardware: would 32 GB MI50s (no xGMI) be better?
- **Prefill: barely.** MI50s without xGMI also have **no P2P** → same no-tensor-parallel
  limit. Per-GPU they're ~1.3–1.5× (more FP16 FLOPS + ~2× bandwidth), so ~6 h → ~4 h —
  **still impractical for cold 128K.** Doesn't fix the real problem.
- **Decode: ~2× win.** Decode is bandwidth-bound; MI50 HBM2 (~1 TB/s) ≈ 2× the V620
  GDDR6 (~512 GB/s). The 16 t/s shallow (and worse depth-decode) would roughly double.
- **Cost/risk:** gfx906 is **deprecated in modern ROCm** (you're on 7.1) — a real
  software-support project, and you'd lose the supported V620 stack (Qwen production).

**Verdict:** for the **128K long-context** goal, MI50s don't help (the prefill wall is
P2P/tensor-parallelism, which they also lack). They're only compelling if Command A is
reframed as a **short-prompt / long-output (decode-heavy)** service, where ~2× decode matters.

## Recommended deployment
- **Config:** F16 KV, `ub=2048`, `-b 4096`, `-sm layer`, `-fa on`,
  `--jinja --reasoning-format deepseek`.
- **Interactive context:** cap at **~16–32K** (minutes, not hours).
- **128K:** only viable with **prompt caching** (pay the ~6 h prefill once, reuse the KV)
  or by leaning on **aggregate throughput across the 3 concurrent slots** rather than
  single-prompt latency.

## Tooling / repro notes (gotchas for next time)
- `llama-cli` here is **interactive-only** (no `llama-completion` binary; `-no-cnv`
  unsupported). Drive prefill via `llama-server` `/completion` with a **long-lived client** —
  a short HTTP client timeout will cancel the server-side prefill mid-run.
- `test-backend-ops` reports `FLASH_ATTN_EXT` **"not supported"** on ROCm0 for synthetic
  cases (harness quirk) — it cannot be used to probe FA configs here.
- **Build propagation:** editing `fattn-common.cuh` / `fattn-tile.cuh` (headers) does **not**
  reliably trigger recompilation of the FA template instances. Force it with
  `find build -name 'fattn*.o' -delete` before rebuilding, or the change won't take effect.
- `llama-bench -d` (depth) throws ROCm errors on this box (harness artifact, not a real failure).
