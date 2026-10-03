# llama-navi21-furnace on `phoebe` — progress log

Fork of `ggml-org/llama.cpp` tuned for **2× AMD Radeon PRO V620** (Navi21 / gfx1030 /
RDNA2, 32 GB each) on a single-user R&D box.

| | |
|---|---|
| Cloned | 2026-09-19 20:13 UTC |
| Fork point | `e613ef2c8` — *hexagon: enable I32 GET_ROWS (#29116)*, upstream master 2026-09-19 |
| Branch | `tune/gfx1030-mmq` — 18 commits |
| Host | Gigabyte X570 AORUS PRO WIFI (BIOS F34), Ryzen 7 5800XT, Ubuntu 26.04, kernel 7.0.0-31, ROCm 7.1 |
| Status | Software work complete and committed. **Hardware currently down** — see *Open: boot*. |

Headline numbers, single V620, Qwen3.8-27B class model, `-fa on`:

| metric | at clone | now | change |
|---|---|---|---|
| pp512 (prefill) | 437.1 t/s | 442.3 t/s | +1.2% |
| tg128 (decode) | 23.15 t/s | 23.79 t/s | +2.8% |
| decode + speculation | — | ~50 t/s mean, 67 t/s on math | ~2.1× |
| flash-attn (`-fa on`) | unusable | all model shapes working | — |

---

## 1. Hardware bring-up

The cards were dead on arrival to ROCm: `amdgpu` spun forever on
`trn=2 ACK should not assert! wait again !`, KFD exposed no compute nodes, and each
V620 showed only its 2 MB doorbell and 512 KB MMIO BARs with **BAR0 — the 32 GB VRAM
aperture — entirely unassigned**.

**Fix: disable CSM in BIOS.** With CSM off the kernel places both 32 GB BARs, KFD comes
up with `simd_count 144`, and llama.cpp sees ROCm0/ROCm1 at 32752 MiB each, PCIe Gen4 x16.

Wrong turns worth not repeating:

- It is **not** a VBIOS or SR-IOV firmware problem. An earlier session concluded the cards
  needed reflashing; the decisive counter-evidence was that the same cards ran fine in a
  different machine.
- `trn=2 ACK` is not SR-IOV-specific — it shows up on bare-metal Vega/MI25/Radeon VII too,
  and is a downstream symptom of the failed VRAM aperture, not a cause.
- Do **not** force PCIe Gen3. BAR placement is an address-space problem, not a link-speed
  one, and Gen3 would halve bus bandwidth for nothing.

Also set, belt-and-braces, before the CSM change and never individually verified as
necessary: `GRUB_CMDLINE_LINUX="pci=realloc=off amdgpu.gpu_recovery=1 amdgpu.mcbp=0"`.

`~/v620-check.sh` re-verifies the whole chain in one command.

### Measured hardware characteristics

`rocm-smi` reports mclk "1000 MHz" (memory-*controller* clock) against TechPowerUp's
"2000 MHz" (effective). Same physical GDDR6 at 16 Gbps — **not** a half-speed bug, do not
re-chase it. Confirmed by direct microbenchmark:

- pure read: **505.6 GB/s** (~99% of the 512 GB/s spec)
- streaming copy (read+write): 419 GB/s

---

## 2. Build

ROCm 7.1 lives under `/usr`, not `/opt/rocm`. No `g++`, only `clang++-21`. CMake 3.31
rejects the hipcc wrapper and Ubuntu multiarch hides the HIP cmake package, so:

```bash
export ROCM_PATH=/usr HIP_PATH=/usr
CC=/usr/lib/llvm-21/bin/clang CXX=/usr/lib/llvm-21/bin/clang++ \
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--rocm-path=/usr --rocm-device-lib-path=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode" \
  -DCMAKE_HIP_COMPILER_ROCM_LIB=/usr/lib/x86_64-linux-gnu \
  -DCMAKE_HIP_LIBRARY_ARCHITECTURE=x86_64-linux-gnu \
  -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j16
```

To compile-check one kernel without a full build, call `hipcc` directly with
`--offload-arch=gfx1030`.

---

## 3. Flash attention — from unusable to complete

Five commits took the gfx1030 tile path from aborting on every real shape to fully working.
The bugs were RDNA occupancy falling to 1 in the config tables, plus missing dispatch caps.

| commit | fix |
|---|---|
| `a63536900` | RDNA2 tile config fix (D=256/512) |
| `3ceecec3a` | extend to D=128 |
| `9b1ceb1fc` | D=576 occupancy abort (DeepSeek MLA) |
| `a02ddd408` | D=256 MHA, `ncols2=1` occupancy abort |

Validated with `test-backend-ops -o FLASH_ATTN_EXT`: D=128/256/512/576, KV type
f16/q8_0/q4_0, all batch sizes, KV length a multiple of 256.

**The real inference path fully works.** That 256 constraint is not a limitation in
practice — `llama_kv_cache::get_n_kv()` always pads `n_kv` to a multiple of 256
(`llama-kv-cache.cpp:1255`), so it is what the models actually produce.

Measured value of the tile fix: tg64 @8k 19.87 → 22.33, @16k 17.20 → 21.12,
pp512 @8k 268 → 363.

### Known remaining gaps (deliberately not fixed)

True out-of-bounds memory faults remain on: unaligned `n_kv` (e.g. 113 — never occurs in
real inference), exotic KV dtypes (bf16/iq4_nl/q2_0), ALiBi (`max_bias>0`), and attention
sinks. Several are byte-for-byte upstream bugs, visible at D=40/64/72 which our changes
never touch. These are configurations our models do not generate; fixing them would be
hardening upstream's RDNA2 tile kernel broadly, not fork work.

---

## 4. Kernel work

### Landed

**`0aa20a02b` — MMVQ config tables had no RDNA2 entry.** `calc_nwarps` fell through to
`return 1`, so *every decode dispatch was a single wave32 workgroup*. Added an RDNA2 branch
(nwarps 4 for `ncols_dst<=4`, 2 for `<=8`) plus an RDNA2 case in `calc_rows_per_block`.
**tg128 23.15 → 23.58 (+1.9%).** Do not use nwarps=8 (worse) or nwarps=2 at
`ncols_dst==1` (aborts inside `ggml_cuda_mul_mat_vec_q` — a latent upstream bug worth filing).

**`3bccd8323` — delta-net concat took a scalar path.** A `ggml_transpose` before
`ggml_concat(dim=0)` failed `ggml_is_contiguous_to_3`, so a scalar kernel walked the time
axis with ~16× read amplification. `ggml_cont()` after the transpose routes it to ggml's
dedicated `cpy_scalar_transpose`, 4.4× better: `concat_non_cont` 46.69 ms → 0, replaced by
10.57 + 5.75 ms. **pp512 437.6 → 442.5 (+1.1%).**

**`d660b2491` — three latent correctness hazards** from the kernel audit. 0% perf, all gates
green: a NaN guard for `amax==0` in `quantize_mmq_q8_1`; a `static_assert` at
`fattn-tile.cuh:513/598` that would otherwise silently drop Q columns; and mixing `n_nodes`
into the CUDA-graph key (it previously returned `nodes[0]` alone, collapsing the LRU to one
entry).

**`b5f65d215`, `05512bb1d` — MMVQ tuning for the speculative verify shape.** nwarps 4→2 at
`ncols_dst` 2..4 (pp4 78.4 → 83.4); nwarps 2→1 and `rows_per_block` 2→4 at 5..8
(pp8 112.1 → 120.5, +7.6%; pp6 101.1 → 107.7).

**`1184c3f4a`, `e628efa74`, `30061aa39` — q8_1 activation quantize cache.** Roughly 497
quantize calls per graph eval over only 257 distinct `src1` — ~48% redundant. The cache
lives on `ggml_backend_cuda_context`, MMVQ path only, kill switch
`GGML_CUDA_NO_Q8_1_CACHE=1`. **tg128 23.47 → 23.79.** Two follow-up fixes were needed and
are described under *Bugs found the hard way*.

### Tried, measured, rejected — do not redo

This is the more valuable half of the record.

| attempt | outcome |
|---|---|
| **`K_vram` / `MMQ_ITER_K` as a tuning knob** | **Dangerous.** Raising it looks like +29% pp512 but it *silently skips half the K data* and the model emits garbage. `ITER_K` must equal the 256-element tile capacity; the loop strides by `ITER_K/qk` while the tile covers `MMQ_TILE_NE_K`. It is 256 in every arch config because it is structural, not tuned. Test suite missed it — most shapes have k≤256. **Always end-to-end check a suspicious MMQ speedup with real output.** |
| **MMQ stream-k on the Q8_0 fallback** (`54451989c`, reverted `6022d3e4e`) | Made the M=128 `in_proj_ba` GEMM **3× faster** and gave +0.5% pp512 — but **regressed MTP decode 36.9 → 34.7 t/s (−6%)**, reproducible ×4 at 42 °C. Suspected: stream-k's fixup pool perturbs CUDA-graph reuse. **Lesson: always re-measure the production decode path after an MMQ change; prefill-only benchmarking hid a 6% regression.** |
| **Wiring the MUL_MAT_ID crossovers into the dense MMVQ path** | Looked neutral end-to-end but an A/B of the exact range regressed pp7 90.1 vs 102.4 (−12%). Those crossovers were measured for MoE routing and do not transfer to dense matmul. |
| **Forcing VGPR count down for occupancy** | `amdgpu_num_vgpr(168)` *is* honored (187→168) where `__launch_bounds__`/`waves_per_eu` are inert — but pp512 dropped 27% and occupancy was **unchanged** (23.5% vs 23.4%). Occupancy here is **LDS-bound, not VGPR-bound**. Closes the occupancy line; any future work must cut the ~45 KB dynamic LDS. |
| **LDS-staging the q8_1 activations** | The theory was 3.78× activation-vs-weight traffic amplification. Implemented correctly (142 `ds_read` / 30 `ds_write` emitted, tests passed) and it was **~2× slower on every shape**. The amplification theory was wrong. |
| **`rows_per_block=8`** | Predicted to fail on register pressure. It actually uses **fewer** VGPRs (168 vs 200) and gets **higher** occupancy (25.1% vs 22.0%) — and is still slower. Both of my stated mechanisms were wrong. |
| **Register-caching `x` in rms_norm** | Zero change. The kernel launches one 1024-thread workgroup at decode — 2.8% of the GPU, ~12 GB/s — so it is latency-bound, not traffic-bound. Halving reads cannot help. |
| **`{RMS_NORM,SCALE}` fusion** | Declined. Priced at 0.28–0.42% of decode. Also: `ggml_l2_norm` looks like a drop-in for `build_gdn_l2_norm` but the CUDA kernel computes `rsqrt(fmax(sum, eps*eps))` vs the required `rsqrt(sum + eps)` — **not equivalent**, would silently change numerics. |
| **CUDA-graph fork-point fix** | Audit claimed 112 fork points with 2 invalid discarding all. Instrumented reality: only 16 are built and **all 16** fail `is_valid()` on overlapping `Kcur` view writes. The fold is a real latent bug but fixing it yields 0 streams and 0% here. |

**Where the remaining headroom actually is.** `mul_mat_vec_q` runs at **480 of 505 GB/s =
95% of read roofline during its own execution**. An earlier "406 GB/s = 80%" figure divided
the same bytes by the whole token wall-clock — the decode gap is *not* in the GEMV inner
loop. Prefill is 88.3% `mul_mat_q` and 0.6% dispatch overhead. One decode dispatch costs
2.13 µs in situ, so removing N dispatches/token is worth N × 0.0049% of tg.

The only untried lever is widening the weight loads in `vecdotq.cuh` — the Q4_K
`ncols_dst=8` kernel issues **220 `global_load_dword` + 120 `global_load_ushort` and not a
single `dwordx2`/`dwordx4`**; the memory unit saturates on request count long before bytes.
That file is shared by every backend and quant type, so it is high blast radius and needs
explicit buy-in.

---

## 5. Speculative decoding

Two drafters were evaluated end-to-end.

**MTP** (`--spec-type draft-mtp`): single chained head. Per-iteration anatomy at n_max=3
(~60 ms, 2.40 tokens): target verify decode 52 ms = 85%; drafter 8.2 ms = 14%; acceptance
1.40 per iteration (46.5% of drafted). Both the drafter and the target's MMVQ sit at **98%
of the bandwidth roofline** — there is nothing left there.

The n_max sweep is done and the default is optimal: 1→36.4, 2→38.2, **3→38.5**, 4→37.0,
5→33.1, 6→29.9, 8→18.0. `p_min` at 0.4/0.6/0.8 all lose to n_max=3. Acceptance saturates at
~1.43 no matter how deep you draft.

**DFlash2** (`z-lab/Qwen3.8-27B-DFlash2`): block diffusion drafting, works out of the box in
this fork — no porting needed. Two setup notes: `n_max` defaults to 3 and **must** be raised
to `block_size-1 = 7` or the drafter is crippled to 3 of its 7 slots; and use the **Q4_0**
drafter, not Q8_0 (66.6 vs 63.3 t/s — Q4_0 is smallest and unpacks straight into dp4a).
`--spec-draft-p-min` is a **no-op** for dflash2 — the selector-confidence gate never fires.

Head-to-head, 256 tokens, greedy, Q4_0 drafters:

| prompt | MTP n=3 | DFlash n=3 | DFlash n=7 |
|---|---|---|---|
| GSM8K-ish math | 56.3 | 54.6 | **67.7** |
| code gen | 47.9 | **54.8** | 41.4 |
| boilerplate | 53.8 | 53.7 | **55.0** |
| reasoning essay | 40.0 | **40.4** | 31.9 |
| mean | 49.5 | 50.9 | 49.0 |

DFlash n=7 swings +20%/−20% with prompt entropy. An oracle picking n per prompt averages
54.5 — so adaptive depth was the single biggest remaining win, and since the shipped `p_min`
mechanism is dead it had to be built.

### Adaptive draft length (`d6bc22ebd`, `291449c0b`)

Per-sequence EMA of accepted draft tokens (α=0.2); next draft sized as
`clamp(lround(ema*1.3)+1, 1, cap)`. The cap is published on `dp.n_max`, the pre-existing
per-call override, so DFlash and MTP size the draft **up front** rather than truncating.
Kill switch `LLAMA_SPEC_NO_ADAPTIVE=1`. Gated to `cap >= 4`, so MTP at its optimal cap of 3
is bit-identical to before.

**DFlash cap 7: 53.1 adaptive vs 48.8 fixed n=7 (+8.9%)** over 4 workloads.

`291449c0b` fixes a real bug: the EMA carried over between server requests, so a math prompt
following an essay ran 71.8 vs 77.1 fresh. Now reset to the cap in `common_speculative_begin`.

Three variants were tried and lost — do not re-add: post-hoc truncation (49.8; the drafter
has already paid for discarded positions), censoring correction (52.9), and a hysteresis
deadband (51.5). The plain form won at 53.6.

---

## 6. Quantization

Built custom quants of `Jackrong/Qwopus3.8-27B-Flash-V2`, a fine-tune of Qwen3.8-27B.

Replicating an Unsloth UD recipe: `llama-quantize --tensor-type-file` takes `pattern=type`
lines, lowercased and applied with `std::regex_search`, so anchor and escape them. **Correct
ggml type ids: 20=IQ4_NL, 21=IQ3_S, 23=IQ4_XS** — 21/23 were swapped in the first attempt,
silently producing 70 tensors at 3.4 bpw where the reference uses 4.25. Always re-dump the
built file and diff the type histogram against the reference.

Per the no-IQ-in-the-target rule:

| build | size | pp512 | pp8 | tg128 |
|---|---|---|---|---|
| base Qwen3.8-27B UD-Q4_K_XL | 16.34 | 455 | 120 | 23.92 |
| Qwopus, UD recipe with IQ tensors → Q4_K | 16.57 | 431 | 108.5 | 23.81 |
| Qwopus, above with all Q4_K → Q5_K | 17.84 | 420 | 108.4 | 22.43 |

Dropping IQ costs ~10% on pp8 and nothing on tg128 — at `ncols_dst=8` IQ4_XS is 2.1× faster
than Q4_K, and 70 tensors × ~0.11 ms ≈ 7 ms of a 67 ms batch-8 eval, which matches the gap
almost exactly. A KL-divergence study confirmed the quality cost is negligible: mean KLD
0.00685 (no IQ) vs 0.00707 (with IQ), top-1 96.60% vs 96.51% — within error. The Q5_K
variant is the only real quality step (0.00512, 97.03%) and costs ~6% decode for +1.27 GiB.

**Key finding: DFlash2 does not transfer to a fine-tune; MTP does.**

| target | drafter | math | code |
|---|---|---|---|
| base | DFlash2 | 67.5 | 54.7 |
| base | MTP | 56.1 | 47.9 |
| Qwopus FT | DFlash2 | 54.8 | 43.6 |
| Qwopus FT | MTP | **55.9** | **46.2** |

DFlash2 loses ~19%/20% and its entire advantage. Hypothesis, untested: DFlash2 injects
hidden states from target layers [6,20,34,48,62], coupling it to internal representations
that a fine-tune shifts; MTP only consumes the final hidden state, a far more stable
interface. **So: DFlash2 for the stock model, MTP for fine-tunes.**

Extracting the fine-tune's own MTP head (`~/models/extract_mtp_head.py`) turned out to be a
no-op — this fine-tune never retrained the head. Its `blk.64` F32 norms are **bit-identical**
to the base drafter's, and the quantized ones differ by exactly Q4_K round-trip error. Worth
doing only for a fine-tune that actually trains its head; the correlation check costs seconds.

---

## 7. Bugs found the hard way

**Reference-logits memory fault.** `HSA_STATUS_ERROR_MEMORY_FAULT` during KL-divergence runs.
I first blamed host memory pressure and mmap, and was wrong. Bisection showed it needed the
q8_1 cache **and** CUDA graphs **and** multiple sequences together. Running with `-v` showed
**1988 "ROCm buffer pool full"** messages: the cache was unbounded, the legacy pool has 256
free slots, overflow triggered `cudaFree` on memory a captured graph still referenced →
page fault on replay. Fixed in `30061aa39` by bounding the cache to 8 entries and adding the
stream to the key. Validated: 0 faults, bit-identical perplexity 5.0177, tg128 23.72/23.67
with cache on vs 23.38/23.39 off.

**Cache outliving the pools.** The q8_1 cache tripped `GGML_ASSERT(pool_size == 0)` at exit.
Fixed in `e628efa74` by clearing it in the context destructor.

**Batch 8 → 9 cliff (open).** Decode ms by batch: 8 → 72.9, **9 → 108.2**. `MMVQ_MAX_BATCH_SIZE 8`
hands `ne11>=9` to MMQ, and MMQ is catastrophic on this path — forcing it at `ne11<=3` drops
MTP 38.7 → 25.1 t/s (−35%). Extrapolated MMVQ would not cross MMQ until batch ~13–14, so
batches 9–13 leave ~30 ms/eval on the table. Raising the limit means template instantiation,
not a runtime knob (`GGML_CUDA_MMVQ_MAX=64` aborts). Irrelevant at n_max=3; matters for
llama-server with parallel slots.

---

## 8. Current deployment

`~/models/serve-qwopus.sh` — Qwopus fine-tune (Unsloth UD-Q4_K_XL recipe with IQ tensors
moved to Q4_K, 16.57 GiB) + base MTP drafter at n_max 3, on GPU 1, bound to `0.0.0.0` for
LAN and tailscale, with MCP tools (`brave_web_search`, `brave_read`, `web_fetch`).

Measured: math 55.9 / code 46.1 / boilerplate 49.7 / prose 43.6 t/s.

---

## 9. Open items

**Boot (blocking).** The V620s are currently **out of the machine** — it will not POST
reliably with them installed. Root cause identified from the boot record: firmware is not
advertising an above-4G MMIO window, so the 32 GB BAR cannot be placed. When that happens
`amdgpu` probes anyway, reads all-ones from unmapped MMIO, concludes it is an SR-IOV virtual
function, and deadlocks in `xgpu_nv_mailbox_trans_msg` waiting for a hypervisor that does not
exist — an uninterruptible sleep during module load, which is the unbootable machine.

Confirmed by a clean A/B on 2026-09-19, same kernel and BIOS, four hours apart:

| | boot -4 (hung) | boot -3 (worked) |
|---|---|---|
| root bus above-4G window | none | `0x840000000-0xffffffffff` |
| V620 BAR 0 (32 GiB) | `can't assign; no space` | assigned at `0x8000000000` |
| result | hung in probe | `Detected VRAM RAM=32752M, BAR=32768M` |

Fix: **Above 4G Decoding → Enabled**, **CSM → Disabled**. Note that a hang counts as a failed
POST, and the board restores defaults after repeated failures — so the hang keeps undoing the
fix, which makes the setting look flaky. Optional hardening: disable SR-IOV if the board
exposes it, which drops the bridge window requirement from 416 GiB (32 GiB PF + 384 GiB of VF
BARs for 12 VFs) to 32 GiB. Recovery lever: `modprobe.blacklist=amdgpu` at the GRUB prompt.

BIOS is **F34 (July 2021)** against a **Ryzen 7 5800XT (July 2024)** — the firmware predates
the CPU by three years and Linux is already replacing its microcode at boot
(`0x0a201204 → 0x0a201211`). Latest is **F40c**. Worth updating as hardening, but note F35+
adds capsule protection so rollback may be blocked, and flashing resets all settings.

**Not blocking.**

- Batch 8→9 MMVQ/MMQ cliff (§7) — matters for parallel server slots.
- Widening weight loads in `vecdotq.cuh` (§4) — the only untried kernel lever, high blast radius.
- Upstream RDNA2 tile-kernel faults on ALiBi / attention sinks / exotic KV dtypes (§3).
- `ggml_cuda_mul_mat_vec_q` aborts at nwarps=2, `ncols_dst==1` — latent upstream bug, worth filing.

---

## 10. Verification

```bash
# correctness gates
./build/bin/test-backend-ops -o MUL_MAT          # expect 1297/1297
./build/bin/test-backend-ops -o FLASH_ATTN_EXT

# coherence — never trust a kernel speedup without this
./build/bin/llama-cli -m <model> -p "The capital of France is" -n 20

# perf
./build/bin/llama-bench -m <model> -fa 1 -p 512 -n 128
```

Kill switches: `GGML_CUDA_NO_Q8_1_CACHE=1`, `LLAMA_SPEC_NO_ADAPTIVE=1`,
`GGML_CUDA_DISABLE_GRAPHS=1` (also required for `rocprofv3 --kernel-trace` on the decode
path, which segfaults otherwise).

---

## 11. 2026-09-27 overnight: single-V620 speculative decoding push

Goal: the best experience on **one** V620, since that is what most people running this will
have. All numbers below are from a **warm llama-server** (every prompt run once untimed
first): a fresh process pays one-time costs on its first request (lazy kernel loading, first
CUDA-graph captures), e.g. DFlash math 68.5 t/s cold vs 79.3 warm with identical tokens.
Mean over four prompts (math, code, list, essay), 256 tokens, greedy unless noted.

### Results

| model | mode | start of night | final |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL | plain decode | 23.8 | 24.5 |
| | MTP n=3 (reduced-vocab drafter) | 49.6 | **54.2** |
| | **DFlash2, adaptive cap 7 (reduced-vocab drafter)** | 58.0 | **61.1** |
| Swift-Qwen3.8-27B | plain decode | 23.4 (their Q4_K_M) | 24.4 (our Q4_K_XL) |
| | MTP n=3 (reduced-vocab drafter) | 48.6 | 52.6 |
| | **DFlash2 n=3 (reduced-vocab drafter)** | 51.4 | **54.3** |
| | same, temp 1.0 / top-p 0.95 / top-k 20 (model card) | | 52.0 |

Final numbers are on commit 0b1887762. At ~8k tokens of context Swift + DFlash2 still runs at
45.9 t/s (plain 22.6).

### What landed

- **Reduced draft vocabulary** (`scripts/draft-vocab/build-draft-vocab.py`, MTP support in
  `qwen35.cpp`). The drafter's LM head was half its cost (3.0 of 6.2 ms per DFlash iteration;
  MTP runs it every draft step). The script copies the 64k most likely tokens' rows of the
  *target's own* output projection as raw quantized bytes, plus a `d2t` map; logits for kept
  tokens are bit-identical and output is unchanged. 98.9% coverage of held-out model output.
  +2.9% DFlash, +8.5% MTP (vs the stock drafter).
- **CUDA graph cache keyed by shape.** Batch sizes that differ only in token count shared one
  cache entry, so adaptive draft lengths kept evicting it: captures 226 -> 87 per session, +1.5%.
- **Fewer kernels per delta-net layer**: RMS_NORM+SCALE fusion (q/k L2 norm), the `ssm_out`
  and alpha-gate bias adds moved so they fuse into their matmuls, the batch-1 conv-input copy
  replaced by a reshape, and GATED_DELTA_NET reading its state straight from the cache instead
  of a gathered 3 MB copy (single-sequence batches). 1800 -> ~1520 dispatches per token,
  decode 23.90 -> 24.67; the 4-token verify 82.1 -> 83.2.
- **Flash-attn tile fixes scoped to RDNA2.** RDNA3/4 also use the tile kernel for decode and
  small batches; the shared table is upstream's again.

### Swift-Qwen3.8-27B quantization

Our build from Swift's F16: Unsloth's UD-Q4_K_XL per-tensor recipe with the IQ tensors moved to
Q4_K, Unsloth's imatrix (computed on the base model). KL divergence vs Swift's Q8_0, wikitext,
512 ctx, 95 chunks:

| | size | mean KLD | 99% KLD | same top token |
|---|---|---|---|---|
| Swift's own Q4_K_M | 16.79 GiB | 0.0134 | 0.144 | 95.27% |
| our Q4_K_XL (no IQ) | 16.57 GiB | **0.0092** | **0.093** | **96.41%** |

Swift's MTP head is the stock head (all F32 norms bit-identical to the base model), and unlike
Qwopus, DFlash2 transfers to it.

### Tried and rejected

- **MMVQ fusion for 2-8 columns** (residual add, gate+up GLU in the verify batch): the fused
  kernel variant is 5x slower at 8 columns (pp8 119 -> 23 t/s), even with only a bias.
- **Measured-throughput draft-length controller** (per-length moving averages of tokens/ms,
  periodic neighbour probes): worse than the existing one. A length's first use includes graph
  capture (up to 195 ms), which poisons its estimate; iteration cost is nearly flat for n=1..3
  so the choice rides on noisy, content-dependent token counts.
- Graphs off, fewer CPU threads, GPU-side target sampling: all within 1%.
- DFlash2's block width changes the accuracy of *every* drafted position (Swift code:
  first-token acceptance 0.92 at n=3, 0.84 at n=4), which is why n=3 often beats wider drafts.

### Where the time goes now

A DFlash n=3 iteration on Swift is ~56 ms: ~49 ms target verify of 4 tokens, 4.4 ms drafter,
~2-3 ms host. The verify runs at 88-93% of memory bandwidth; what remains is small kernels
(each <1%) or fewer bytes (a quality trade).

## 12. 2026-09-27 follow-up: the three remaining items

| model | mode | before | after |
|---|---|---|---|
| Qwen3.8-27B | DFlash2 adaptive cap 7 | 61.1 | **62.2** |
| | MTP n=3 | 54.2 | **55.0** |
| Swift-Qwen3.8-27B (our Q4_K_XL) | DFlash2, greedy | 54.3 (fixed n=3) | **58.5** (adaptive cap 7) |
| | DFlash2, temp 1.0 | 52.0 | **52.9** |

Commit 5f0437f39, warm llama-server, mean of four prompts.

- **Kernel merges in the verify batch (landed).** ADD->RMS_NORM->MUL (the residual add and the
  next pre-norm, writing both outputs; the allocator aliases the sum over one input and the
  result over the other, which the kernel handles) and ADD->SOFTPLUS->MUL with row-broadcast
  vectors (delta-net decay gate). The 4-token eval runs no binary-op kernels any more
  (224 -> 0), 2026 -> 1802 dispatches, pp4 83.3 -> 84.9.
- **Draft-length control (no new code needed).** Re-measured on the deployed setup, the
  existing adaptive controller now matches the best fixed length on every prompt for Swift
  too: the reduced-vocab drafter made longer drafts cheap. Launcher switched from fixed n=3 to
  cap 7.
- **Host gap between steps (stopped).** Per iteration: target submit 0.65 ms, drafter submit
  0.6, feature hand-off 0.2 (copy 0.02), sampling 0.5-0.7; the drafter's GPU time is ~4 ms.
  Outside GPU waits the main thread's work is spread over many functions at <0.5% each.
  The drafter context never reuses its graph (it alternates the injection and draft graphs),
  but that shows as ~0.1% of CPU samples. No single fixable cause.

## 13. 2026-09-27: speculative sampling for temperature 1.0 chat

With exact-match verification, a draft token survives only when the target's own random sample
equals it, so at temperature 1.0 the acceptance of a draft is p(draft). Commit 2055e58f9 adds
standard speculative sampling for DFlash2 (`--spec-draft-temp T`, server only, default off):
each draft position is sampled from softmax(selector scores / T) over its 16 candidates, and
verification accepts with probability min(1, p/q), otherwise emits a sample of
norm(max(0, p - q)) and stops. The output distribution is exactly the target's. Greedy
requests keep argmax drafts. The target distribution is taken with the grammar applied first,
so tool-call JSON (the web UI's MCP tools attach a lazy grammar) also goes through p/q
verification; the first version fell back to exact match there, which disabled the feature for
every answer once thinking ended.

Chat benchmark (Swift Q4_K_XL, the web UI system prompt, 4 research questions x 3 seeds,
temp 1.0 / top-p 0.95 / top-k 20, 700 tokens), interleaved:

| drafts | t/s | draft accepted |
|---|---|---|
| argmax (before) | 37.4, 37.4, 37.4 | 43% |
| sampled, T = 1.0 | 42.3, 42.6, 42.7, 43.5 | 49% |

Per verified position, sum min(p, q) = 0.67 against p(argmax q) = 0.58. Scored on the same
positions, drafter temperatures 0.85-1.0 are the flat optimum (0.4: 0.64, 1.3: 0.66), so the
drafter is well calibrated and T = 1.0 is used.

Checks: the accept/residual step against synthetic p and q (1M-4M trials, output frequencies
match p); 3000 short completions plain vs speculative, tokens 2-4 not distinguishable
(chi-square p 0.54-0.94); greedy text and PPL identical to the previous build. Both launchers
now pass `--spec-draft-temp 1.0`; greedy requests to such a server give byte-identical output.
With tools attached: 198 and 121 positions verified by p/q, none by exact match, tool call parsed.
Not exercised: the checkpoint-replay branch (needs a rollback beyond n_rs_seq = n_max, which
cannot happen here). MTP still drafts greedily and is the next candidate.

### 13a. The same for MTP (529c6fb90)

The MTP drafter samples each step from softmax(logits / T) over its top-10 candidates and feeds
the sampled token to the next step. Swift Q4_K_XL, MTP head with the 64k reduced vocabulary,
all at temp 1.0, interleaved:

| config | chat bench (t/s) | math / code / list / essay (t/s, mean of 2) | mean |
|---|---|---|---|
| MTP n=3, argmax drafts | 38.0, 37.9 (37% accepted) | | |
| MTP n=3, sampled | 44.2, 44.2 (48% accepted) | 62.1 / 54.1 / 64.3 / 42.7 | 55.8 |
| MTP n=4, sampled | | 64.5 / 56.0 / 70.9 / 38.4 | 57.4 |
| DFlash2 adaptive cap 7, sampled | 41.8, 42.9 (48-49% accepted) | 76.1 / 54.1 / 68.6 / 39.7 | 59.6 |

Per verified position, MTP: sum min(p,q) 0.66 vs p(argmax q) 0.57. Greedy output with the flag
on is identical; with tools attached no position falls back to exact match. With both
drafters sampling, MTP n=3 leads on research chat by ~4% and DFlash2 leads on the mixed prompts
by ~7% (math most of all).

## 14. 2026-09-27: flash-attention occupancy on RDNA2, D=256 retune (b96c5233d)

HIP reports 32768 registers per multiprocessor on gfx1030, a quarter of a WGP's register file,
so its occupancy query returned 0 for flash-attention tile kernels above 128 VGPRs (every larger
tile config aborted) and too few blocks for the rest (the small-batch kernels split the KV cache
too coarsely). launch_fattn now computes RDNA2 occupancy from the kernel's registers and LDS, and
the D=256 tile uses 64-key batches.

Per attention call, Qwen3.5 shape (24 Q heads over 4 KV heads, D=256):

| | before | after |
|---|---|---|
| prefill, 512 tokens | 12.0 TFLOPS | 15.7 TFLOPS |
| decode, 1 token, 64k context | 1200 us | 580 us |
| verify, 4 tokens, 64k | 1766 us | 678 us |
| verify, 8 tokens, 64k | 2190 us | 1162 us |

Swift-Qwen3.8-27B Q4_K_XL at 64k context (llama-bench): pp512 216.7 -> 245.8, tg32 16.5 -> 19.8.
Empty context unchanged (pp512 425). Web UI server, DFlash, one long document at growing depth:

| depth | prompt t/s before -> after | generation t/s before -> after |
|---|---|---|
| 16k | 366 -> 369 | 45.2 -> 49.3 |
| 32k | 304 -> 323 | 41.7 -> 45.5 |
| 64k | 243 -> 269 | 32.0 -> 37.0 |
| 96k | 191 -> 219 | 29.6 -> 41.9 (acceptance 55% -> 61%) |
| 128k | 158 -> 185 | 25.6 -> 38.4 (acceptance 56% -> 63%) |

Prompt speed is the newest 32k chunk. The 96k/128k generation gains are partly higher draft
acceptance in those runs; the controlled figure is the llama-bench one above.
Flash-attention tests pass for D=64/128/256/512/576; KLD against the flash-attention-off path
improved (mean 0.0063 -> 0.0056, max 1.47 -> 0.28). The server now runs 128k context.

Tried and rejected on the way: forced rocBLAS matmuls for prefill (371-378 vs 434 t/s), flash
attention off (slower at depth), ubatch 1024/2048 (no change), a 64-column tile for D=256 (slower).

## 15. 2026-09-27: follow-ups from an outside review

A read-only review of this work proposed ranked ideas; four were tried.

| idea | result |
|---|---|
| A. Run the 8 conv-state snapshot copies per delta-net layer (384 kernels per verify step at draft cap 7) as one launch (7de0c83d2) | +1.4% spec decode (59.3 -> 60.1 t/s, 4 prompts x 2 rounds, every prompt faster). Greedy output byte-identical; a mutant that writes the wrong slots changes it. |
| B1. Verify-batch attention configs: the 3-4 token entry compiled to 256 VGPRs with spills (abef34ca5) | 118 VGPRs, no spills; 8-token verify at 64k 1161 -> 1062 us per call, 4-token 679 -> 662 us. |
| B2. All 6 query heads of a KV head in one block (ncols2 = 6) | Correct, but every variant spills 23-333 VGPRs to scratch: prefill attention 15.5 -> 3.0 TFLOPS. The tile kernel does not handle a non-power-of-two group size efficiently; would need kernel rework. Not committed. |
| E. Truncate the drafter's distribution with the request's top-p before sampling | Scored on the same 7000 positions: expected acceptance 0.6654 -> 0.6687 at top-p 0.95 (worse at 0.8). About +0.5%; not implemented. |
| D. MMVQ for 5-8 token verify: two warp groups each owning half the columns over the same weight rows | Registers 193 -> 98, occupancy 4 -> 9 waves/SIMD, tests pass - and 40% slower (pp8 111 -> 66). The second group's weight reads do not come from cache, so weight traffic doubles. Not committed. |
| G. Delta-net graph merges | One L2 norm over q and k (5aa13658c): bit-identical, 48 fewer kernels per eval, speed within noise (59.36 -> 59.43). Merging the beta/alpha matmuls would save ~0.6 ms per step (~1%) but needs a combined weight in the GGUF, which upstream llama.cpp could not load; not done. |
| F. Draft length chosen from the measured verify cost curve | Predicted +0-4%; measured 59.55 -> 58.64 on four prompts (code/essay +2%, math -3%, list -6%) and no change on chat (43.75 vs 43.3). Not committed. |
| C. MMQ stream-k with more persistent blocks | The old stream-k test was starved (36 blocks, 1 per WGP): 302 t/s at 1x, 415.5 at 2x, but tiling is still faster (428.9). Closed. |

Also found: at temperature 1.0 the same request with the same seed does not give the same text
from run to run, even on an unchanged build (greedy does). Output comparisons must use greedy.

## 16. 2026-09-28: reproducibility of sampled drafts, p_min, and the draft-tree question

**Seeded requests now repeat (b5032ff9a).** With sampled drafts, the same seeded temp-1.0
request gave different text twice in a row on one server; plain sampling and greedy drafts
reproduced exactly. Three causes, found by instrumenting the accept step and diffing two requests:
1. each drafter sampled from one RNG seeded at construction; it is now per sequence and seeded per
   request (target chain: seed, verification: seed+1, drafter: seed+2);
2. DFlash2's 16 candidates per position come back from the GPU top-k in an order that varies
   between runs; the sampling walk now goes in token-id order (argmax never cared);
3. the MTP drafter pairs the first prompt token with the previous token's hidden state, which for
   a prompt at position 0 was whatever the previous request left behind (drafter logits drifted
   0.03% per request). A sequence starting at position 0 now gets a zero state. With prompt reuse
   the carried state is the last generated token's, correct when the new prompt continues from it;
   after an edited or regenerated turn it is stale for one position (not a correctness issue: the
   target verifies every draft). MTP acceptance on the chat bench with greedy drafts: 36.6% before
   and after.
The output distribution is unchanged; greedy text and PPL identical.

**p_min (correction to section 11's "never fires").** On the d2t drafter the confidence gate does
fire (startup log shows the value; drafts shorten), and loses at every setting on the greedy
4-prompt set: 59.8 t/s at 0 -> 56.5 at 0.25 (the code prompt's 61.8 -> 48.3 there is a single run)
-> 56.4 at 0.5 -> 49.2 at 0.8. Shorter drafts lose more accepted tokens than they save verify rows.

**Draft trees: not worth building.** A tree only pays if the target often accepts the drafter's
second choice. Measured on 7000 chat positions at temp 1.0: the target's probability of the
drafter's 1st / 2nd / 3rd choice is 0.591 / 0.118 / 0.057. At equal verify rows (5), a chain of 4
expects 0.59+0.35+0.21+0.12 = 1.26 accepted tokens; a 2-way tree branching at position 1 expects
(0.591+0.118) x 1.59 = 1.13. Break-even needs a second-choice rate above ~0.21; it is 0.12, and
verify rows 5-8 cost 4-5 ms each. Plumbing would be feasible (branches as extra sequence ids;
common_memory::seq_cp copies both contexts, the recurrent state copy is per-cell) but the ceiling
is below zero at this acceptance profile.

## 17. 2026-09-28: prefill - the K-quant integer multiply, and the 250 W power cap

**Where prefill time goes.** 90% is the quantized matmul (MMQ). At the same shape, Q8_0 runs at
37 TOPS while Q4_K/Q5_K (88% of prefill time) run at 24-26; per weight element the earlier
"Q6_K is faster" reading was a shape effect. Hardware counters: VALU-issue-bound, LDS stalls near
zero. The card sat at exactly 250 W (its power cap) at ~2350 MHz against a 2570 MHz maximum.

**Kernel (f75d2cd02, +8.5%).** The Q4_K/Q5_K/Q6_K MMQ dot helpers multiplied each sub-block's
dp4a sum by its integer scale before converting to float: full rate on NVIDIA, quarter rate on
AMD (`v_mul_lo_u32`), 128 of them per 1024 dot products in the Q4_K loop. The product is exact in
fp32 (7-bit scale x sum below 2^17), so on HIP it is now a float multiply: bit-identical results,
pp512 434 -> 470, pp2048 424 -> 459, decode unchanged. Also tried: CU-mode compilation (-11%),
rocBLAS matmuls (-13%), ubatch 1024/2048 (no change).

**Power (+5.9%).** The VBIOS PowerPlay table sets the SMU limit to 250 W and disables the
overdrive power-limit capability, so the driver reports min = max = 250 W and `ppfeaturemask`
cannot change it. Uploading a modified table via `pp_table` made the driver reset the SMU, which
never came back (GPU wedged, reboot with the reset switch) - do not do that. What works: a 3-line
patch to `sienna_cichlid_get_power_limit()` raising only the reported maximum to 300 W for PCI
1002:73a1, built as an out-of-tree `amdgpu.ko` for 7.0.0-34-generic (~/v620-power/, recipe in
the README there). Setting the cap then uses the normal runtime message and the firmware honors
it: prefill draws 282 W mean (299 peak) at ~2440 MHz, junction 80 C, pp2048 459 -> 486 t/s,
pp512 470 -> 491, pp512 at 64k context 246 -> 269; decode is bandwidth-bound and unchanged.

Prefill today: 434 -> 491 t/s at empty context (+13%).

## 18. 2026-09-28: long-context attention - Q read from a global buffer

**Where the wide attention kernels' time goes.** The KQ loop of the tile kernel is bound by LDS
read instructions on RDNA2, not by the dot products: per step each lane reads K (2 x 16 B) and Q
(4 x 16 B) from shared memory and issues 16 `v_dot2`; the next step's reads only start once this
step's dot products are done. Removing the Q reads in a timing probe gave +25%, the K reads +13%.

**Change (this commit).** For the wide kernels (D <= 256, 16 or more columns per block, i.e.
prefill and verify batches of 5 or more tokens) a small pre-pass writes Q scaled and converted
to `half2` into a contiguous buffer, and the KQ loop reads it from there instead of staging it in
shared memory. The Q row is the same for every lane of a warp, so the loads are warp-uniform
and served from cache; the LDS then carries only K. The narrow decode kernels keep the shared-
memory path (they stream K/V and lose from the extra loads). The 32-column kernel drops from
219 to 108 VGPRs and from 29.7 to 13.3 KB of LDS.

| | before | after |
|---|---|---|
| attention prefill, 512 rows at kv 16k / 64k | 16.5 / 16.4 TFLOPS | 21.5 / 20.7 |
| 8-token verify at kv 16k / 64k | 265 / 1021 us | 228 / 979 |
| pp512 at 64k context (server) | 269 t/s | 278 |
| tg32 at 64k, pp2048 at empty context | unchanged | |

**Bit-exactness, checked properly.** The first comparison showed all outputs differing by ~1%
(same greedy text on 6 prompts, PPL 3.7871 vs 3.7966, both equally close to the fp32 CPU result)
although the Q values in the buffer are bit-identical to the shared-memory copy and the ISA has
the same `v_dot2` chains. The cause is the launch, not the kernel: `launch_fattn` picks the
number of KV-split blocks from the kernel's occupancy (registers and LDS), and the lighter kernel
raises the blocks-per-WGP estimate (3 -> 4 for 16 columns, 2 -> 4 for 32), so the KV range is
combined in a different number of partial softmaxes. With the split pinned to the old value
(9 blocks at 8 queries, 3 at 512) the new kernel's output is byte-identical to the old at kv 4096
and 16384. The kernel arithmetic is exact; only the partition of the reduction moved, which it
already does with context length and GPU.

## 19. 2026-09-28: performance table after the attention change

Swift-Qwen3.8-27B Q4_K_XL, one V620 at 300 W, build 5c913c0d2, 128k context, flash attention.

Prefill and plain decode (llama-bench, no drafter; depth = tokens already in the context):

| depth | pp512 t/s | pp2048 t/s | tg32 t/s |
|---|---|---|---|
| 0 | 492 | 487 | 24.6 |
| 4k | 470 | 463 | 24.2 |
| 16k | 416 | 413 | 23.2 |
| 32k | 358 | | 22.0 |
| 64k | 278 | | 20.0 |
| 128k | 172 | | 16.9 |

Decode with the drafters (llama-server, 256 generated tokens, temperature 1.0 / top-p 0.95 /
top-k 20 with `--spec-draft-temp 1.0`; prompt = the first N tokens of wikitext plus "write a
400-word summary", so the drafters see real long-context work, not repetition; mean of 2 runs):

| depth | none | MTP (n=3) | DFlash2 (adaptive, cap 7) | MTP accepted/drafted | DFlash accepted/drafted |
|---|---|---|---|---|---|
| ~0 (essay prompt) | 24.5 | 48.0 | 46.2 | 0.51 | 0.51 |
| 3.7k | 24.1 | 55.3 | 50.0 | 0.67 | 0.53 |
| 13.4k | 23.0 | 53.0 | 55.3 | 0.68 | 0.58 |
| 40k | 21.4 | 48.0 | 41.6 | 0.66 | 0.51 |
| 79k | 19.3 | 47.2 | 39.2 | 0.77 | 0.52 |
| 120k | 17.4 | 44.1 | 32.0 | 0.83 | 0.48 |

Whole-prompt prefill as the server reports it (the 120k prompt from an empty cache, so it
averages over the whole ramp): 279 t/s without a drafter, 268 with MTP, 275 with DFlash.

Reading: MTP holds 2.5x over plain decode all the way to 120k because its acceptance rises with
context (the summary task becomes more predictable) and its per-step cost does not grow. DFlash2
is the better drafter up to about 16k-30k and then falls off: its drafter runs its own attention
over the context, so each draft step gets more expensive with depth while its acceptance stays
flat at ~0.5. For long-context sessions MTP is the better default; the server currently ships
with DFlash (the better choice for short mixed work: math, code, lists).

## 20. 2026-09-29: profile pass - distance to the hardware limits per area

Method: rocprofv3 kernel traces (CUDA graphs off, so the in-step gaps are larger than in
production; production wall times from llama-bench with graphs on), per-step segmentation on
the vocab-head launch, weight bytes from the GGUF (17.06 GB read per token, 52.1 GFLOP/token in
the matmuls, 16 full-attention layers, 64 KB of KV per token), power and clocks from hwmon at
0.5 s. Limits used: 505.6 GB/s measured read bandwidth; int8 dot4 peak 90 TOPS at 2.44 GHz
(76.7 measured by microbench); fp16 dot2 peak 45 TFLOPS; junction limit 100 C.

| area | physics limit | measured | of limit | where the rest goes |
|---|---|---|---|---|
| decode, batch 1, empty ctx | 33.8 ms/token (weights / bandwidth) = 29.6 t/s | 40.5 ms = 24.7 t/s | 83% | MMVQ 35.5 ms (95% of bandwidth); 1,050 small kernels 3.3 ms; dispatch gaps ~1.7 ms |
| decode at 64k | 42.2 ms (weights + 4.3 GB KV) = 23.7 t/s | 50 ms = 20.0 t/s | 84% | attention 9.6 ms for 16 layers = 89% of KV bandwidth |
| decode at 128k | 50.7 ms = 19.7 t/s | 59 ms = 16.9 t/s | 86% | same |
| prefill, empty ctx (MMQ) | 90 TOPS int8 (76.7 achievable) | 28.8 TOPS in the MMQ kernels, 25.6 end to end (492 t/s) | 32% of peak, 38% of achievable | VALU issue: dequant and scale instructions per dot4; 23% occupancy at 192-240 VGPRs (section 17, rocprof note). MMQ is 89% of prefill, delta-net 5%, attention 1%, rest 5% |
| prefill attention at 16k / 64k | 45 TFLOPS fp16 dot2 | 21.5 / 17.1 TFLOPS | 48% / 38% | after the global-Q change the KQ loop's LDS traffic (K only) sits at the same 45 TFLOPS line as compute; the rest is latency at one block per WGP and the V pass. At 64k attention is 42% of the prefill ubatch (777 of 1856 ms), MMQ 51% |
| verify batch 4 (MTP) | 33.8 ms | 42.7 ms GPU | 79% | MMVQ 39 ms; small kernels 3.5 |
| verify batch 8 (DFlash) | 33.8 ms | 65.8 ms GPU | 51% | MMVQ 60.6 ms: latency-bound at 22% occupancy, four levers already tried and closed (mtp-perf-anatomy) |
| MTP drafter, 4 passes/iteration | ~0.54 GB read per pass = 1.1 ms | 6.1 ms per iteration = 1.5 ms per pass | ~70% | |
| DFlash drafter, 1 block pass | 1.36 GB = 2.7 ms | 5.6 ms | 48% | batch-8 MMVQ, same as the verify |
| speculative iteration (MTP) | 33.8 ms verify at bandwidth, drafter free: 2.5 tok/iter -> 74 t/s | 48.8 ms GPU, ~52 wall -> 43-48 t/s | ~60% | verify batch inefficiency (9 ms), drafter (6 ms), small kernels (3.5), gaps (2-3) |
| dispatch / host | 0 | ~1.7-2 ms per step with graphs on (4-5%) | | 1,500-2,100 launches per step |

**Power and heat are now a limit.** Decode draws 261 W at 2480 MHz although it is bandwidth
bound (the clock is not needed there). Prefill sits at the 300 W cap at 2425-2445 MHz and the
junction climbs 91 -> 98 C within 22 s of a pp2048 loop, against a 100 C limit (105 C
emergency); pp2048 measured 476 t/s in that state against 486-492 when cool. A long prefill
(a 120k prompt is 8 minutes at 300 W) will run into the thermal limit unless the card gets
more airflow. Prefill throughput is therefore bounded by energy per token as much as by
instruction count: fewer instructions per MAC is worth exactly as much as it saves in watts.

**What is left, by size.** (1) Batch-8 verify MMVQ: 27 ms per DFlash iteration above the
bandwidth line, and the same kernel family at batch 4 is 5 ms over; needs a different
algorithm (an MMQ that beats MMVQ below batch 14 on RDNA2), all config levers are closed.
(2) Prefill MMQ at a third of the dot4 peak, power-capped; only instruction reduction helps.
(3) Small kernels and gaps in decode, ~5 ms of 40: op fusion, a few percent each.
(4) Long-context attention at 38-48% of dot2 peak: latency-bound; more columns per block or
register blocking, not obvious on RDNA2 without matrix cores. (5) Decode power: a lower clock
for decode would cut ~100 W with no speed loss (not implemented; needs a DPM policy).

## 21. 2026-09-29: plan against the limits, after an independent review

A first plan (W0-W7 below) was drawn from section 20 and reviewed by a second model with read
access to the kernels, the traces and the earlier negative results. The review disassembled the
gfx1030 code objects in `libggml-hip.so` (no GPU needed) and changed the plan in three places:

- **The verify-batch MMVQ kernels are VALU-issue-bound, not latency-bound.** The cost scales
  linearly with the number of columns (ffn_down: 0.077 / 0.090 / 0.132 / 0.205 ms at 1/2/4/8),
  which a latency-bound kernel with the same weight loads would not do, and the batch-8 Q4_K
  kernel has 128 `v_mul_lo_u32` (quarter rate on gfx10) per K-loop iteration against 160
  `v_dot4`: the same integer scale multiply that f75d2cd02 removed from MMQ is still in the MMVQ
  dot bodies (`vecdotq.cuh` `vec_dot_q4_K_q8_1_impl_vmmq`, `..q5_K..`, `..q6_K_q8_1_impl_mmvq`).
  The float form is exact by the same bound argument. Verified in the source and the
  disassembly. Software prefetch (the plan's W2a) hides latency, not issue, and the split-K
  (W2c) raises occupancy without reducing VALU work: both dropped.
- **q8_0 KV (W1) would slow speculative decoding at depth by 20-30% as dispatched today.** On
  RDNA2 only batch 1-2 goes to the vec kernel, which reads q8_0 natively; every verify batch
  (3-8 rows) goes to the tile kernel, which needs f16 and gets it by converting the layer's whole
  K and V cache to a scratch buffer before each call (`launch_fattn`, `need_f16_K/V`). Plain
  tg32 would show a gain and the shipped path a loss. Needs a dispatch change first.
- **The K-quant MMQ layout idea (W4) is refuted by data already measured.** Q5_K already uses
  the unpacked Q8_0-style tile and is the slowest of the three (24.2 vs Q4_K 25.9 vs Q8_0 37.0
  TOPS). The K-quant excess is the per-sub-block epilogue, already at the bit-exact minimum.
  The instruction-mix ceiling for Q4_K MMQ is ~40 TOPS, not 76.7; it runs at ~65-74% of that.
  Realistic bit-exact gain under 10%; exploratory only.

Other corrections: the 32-column attention kernel is at 128 VGPRs (not 108) and its Q loads
compile to 144 `global_load_dwordx4` per iteration (vector path) instead of scalar loads, a
third co-limiter next to LDS and VALU and a two-line bit-exact experiment; the fused
norm+rope kernel never fires for this model (`rope_multi` is not accepted) so 64 launches per
decode step are available there; the norm-to-q8_1 fusion applies to ~130 of the 257 norms, not
all; the MTP catch-up merge is output-equivalent but not kernel-bit-exact (different
partial-sum order in a wider matmul); a 64-column attention tile needs a code branch, not a
table entry, and lands near 190 VGPRs.

**Revised order**

| # | item | effort | expected | numerics |
|---|---|---|---|---|
| 1 | MMVQ K-quant scale multiply in float (Q4_K/Q5_K/Q6_K, HIP only) | 0.5 d | batch-8 MMVQ -5 to -20% (DFlash +4 to +15%), batch-4 -3 to -8% (MTP +2 to +6%), DFlash drafter head too | bit-exact |
| 2 | thermal and power envelope: sustained 120k prefill with junction and memory temperature, cap curve 300-200 W measured with the drafters on | 0.5 d | reproducibility; shipped cap | none |
| 3 | decode launches: norm->q8_1 for the ~130 eligible norms, `rope_multi` in the fused norm-rope, KV write via rope+set_rows, conv+gdn | 2 d | +2.5 to 3% decode and verify | bit-exact if the reductions are copied exactly |
| 4 | MTP catch-up merged into the first draft pass | 0.5 d | +3% MTP | output-equivalent |
| 5 | attention: V-pass and softmax share probes, then scalar Q loads, then a 64-column tile | 2 d | +5 to 15% on attention at 64k | scalar Q bit-exact; 64-col moves the KV split |
| 6 | decode power policy from the cap curve | 0.5 d | -60 to -100 W | none |
| 7 | q8_0 KV with a vec-kernel dispatch for batch <= 8, A/B with drafters | 1 d | +10 to 20% plain decode at 128k; spec path unknown | changes results |
| 8 | MMQ K-quant loop, exploratory | 3 d | <= +10%, uncertain | mostly not bit-exact |

Gates unchanged: bitwise compare on the model's shapes (mmbench for matmuls, facmp for
attention at a pinned split), test-backend-ops, the check.sh gate, interleaved warm A/B with
power and junction logged. Anything not bit-exact is reverted unless decided otherwise; the
one candidate worth that decision is using the q8_1 block sums for the K-quant min term in MMVQ
(exact-in-quantization-noise, drops 32 dp4a and 64 mul/cvt per iteration), not the split-K.

## 22. 2026-09-29: working the plan - items 1 and 2

**Item 1 (76fa4d1ef): K-quant scale multiply in float in the MMVQ dot bodies.** The batch-8 Q4_K
kernel went from 128 `v_mul_lo_u32` per K-loop iteration to 0 (Q5_K 128 -> 0, Q6_K 64 -> 0,
batch-4 Q4_K 32 -> 0). Bit-identical on every model shape at 1/4/8 columns, test-backend-ops
MUL_MAT 1297/1297, PPL and greedy text identical. pp8 114.5 -> 126.4 t/s (+10%); pp4 and tg64
unchanged; DFlash2 greedy math 64.3 -> 72.0, code 58.4 -> 61.6; MTP unchanged (its batch-4 kernel
had a quarter of the multiplies).

**Item 2: power and thermal envelope.** The driver accepts caps from 250 to 300 W only (the
patched module raises the maximum, the minimum stays at the VBIOS 250).

| cap | tg128 | pp2048 | DFlash math | decode power | prefill power / sclk / junction |
|---|---|---|---|---|---|
| 300 W | 24.7 | 480 | 70.6 | 259 W | 297 W / 2455 MHz / 93 C |
| 275 W | 24.7 | 469 | 71.4 | 248 W | 269 W / 2400 MHz / 90 C |
| 250 W | 24.7 | 456 | 70.4 | 238 W | 245 W / 2325 MHz / 86 C |

Sustained: a 120k-token prompt (7.3 minutes of prefill) at 300 W runs at 276.8 t/s with the
junction at 98 to 102 C from the first minute on (limit 100 C, emergency 105), memory 82 C, and
the clock throttled to 2275 to 2365 MHz; at 275 W the same prompt runs at 274.5 t/s with the
junction at 97 to 101 C and the clock dipping to 2210. So the cap is not what limits long
prefill; the cooling is. The card is passive and needs chassis airflow the current setup does not
give it. Until that changes, long prompts run 5 to 8% below the short-burst numbers whatever the
cap, and the shipped cap stays at 300 W (best for short work, no worse sustained). Decode at
batch 1 draws 240 to 260 W at 2480 MHz regardless of the cap; nothing in this range lowers it.

**Item 3 (decode launches).** Three pieces, each gated by the greedy/PPL compare against the
previous build:
- 3a (80d184f1b): the fused rms_norm+mul+rope kernel rejected this model's rope mode (Qwen3.5
  uses the interleaved multi-section rope, mode 40) so q and k ran norm, multiply and rope_multi
  separately. The kernel now carries the four sections and the interleaved selection copied
  from rope_multi; identical output. 32 launches fewer per step, tg64 24.67 -> 24.77.
- 3b (df346917a): the fused norms write the q8_1 activation for the matmul that consumes them,
  registered in the q8_1 cache under the matmul's key; the quantizer launch disappears for
  every norm-fed matmul (attention qkv, ffn gate/up, the delta-net input projections). The
  epilogue repeats quantize_q8_1's lane mapping and warp reductions, so it is bit-identical.
  Quantize launches ~257 -> ~130 per step, total launches 1823 -> 1687, tg64 24.6 -> 24.9.
  The remaining quantizes sit on matmul outputs (swiglu, gated attention output), which no
  norm kernel can absorb.
- 3c, rejected: also fusing the K-cache write (norm+rope+set_rows in one kernel, writing f16
  directly) removed 16 more launches and measured tg64 24.9, but the greedy text changed on 4
  of 6 prompts (PPL equal to four digits): the fused f16 store does not round the rope result
  the way rope-then-set_rows does. Not bit-exact, dropped.

**Item 5a, rejected: scalar loads for Q in the wide attention kernel.** Marking the Q buffer
parameter restrict makes the compiler use scalar (SMEM) loads for the warp-uniform Q reads
(hot loop: 144 vector loads -> 16 vector + 125 scalar). It is 14% slower (20.8 -> 17.9 TFLOPS
at 16k and 64k): on RDNA2 scalar loads and LDS reads share the lgkmcnt wait counter, so every
wait for K from LDS also waits for the Q loads. Reverted; the reviewer's estimate assumed the
two paths were independent.

**Item 6, dropped.** The driver's cap range is 250 to 300 W and batch-1 decode draws 240 to
260 W at max clock with the cap at any of them (section 22); a phase-dependent cap would save
under 20 W. Not worth the root plumbing.

**Item 5b: where the wide attention kernel's time goes** (512 rows, kv 16k, 9.9 ms per layer,
timing builds with one part removed; the V-pass FMA-only probe was invalid and is omitted):

| removed | time | share |
|---|---|---|
| V pass (V and P reads from LDS + the half2 FMAs; V tile loads kept) | 6.86 ms | 31% |
| KQ dot products (K and Q reads kept) | 9.02 ms | 9% |
| exp in the softmax | 9.73 ms | 2% |

With the K reads at 13% and the Q reads at 25% from the earlier probes, the kernel is about
40% reads for the KQ loop, 31% V pass, 9% KQ math, and the rest tile loads, softmax and
barriers. No single part dominates, which is why the remaining lever is a wider column tile
(more MACs per byte read from LDS and per Q load): a 64-column D=256 tile needs a dispatch
branch, a config entry and about 190 VGPRs, for an estimated +10% on the attention share, i.e.
+4% on pp512 at 64k. Its KV split would move with the register count, so it is exact per
element but not byte-identical unpinned. Left for a decision.

**State after this pass** (deploy build d74bc5922, one V620 at 300 W, cool card):

| | before (05a3f7086) | after |
|---|---|---|
| pp512 / tg32 at empty context | 492 / 24.6 | 492 / 24.8 |
| pp512 / tg32 at 64k | 276 / 20.0 | 276 / 20.2 |
| pp4 / pp8 (verify batches) | 82.6 / 114.5 | 82.6 / 127.0 |
| DFlash2 greedy math / code / essay | 64.3 / 58.4 / - | 70.9 / 62.0 / 40.2 |
| MTP greedy math / code / essay | 63.9 / 54.8 / - | 63.7 / 54.7 / 46.4 |

Every landed change is bit-identical to the previous build (greedy text on 6 prompts and PPL
unchanged). Not done from the plan: item 4 (MTP catch-up merge, +3%, deferred: the drafter's
KV positions across process/draft are more entangled than the review assumed), item 5c (the
64-column attention tile, needs a decision on the moving KV split), items 7 and 8 (dropped as
not bit-exact).

**Thermal follow-up (2026-09-30).** The first sustained runs were taken with the blower drawing
another machine's exhaust. With a cool intake and two 120 mm fans added along the card, the
same 120k prefill at 300 W: 280.7 t/s (was 276.8), power held at 298 W mean (was 290), clock
mostly 2360 to 2440 MHz with brief dips to 2315 (was 2275 to 2365), edge 75 C, memory 76 C
(was 82). The junction still reaches 100 C after about 150 s and stays there, so the firmware
still trims the clock by 2 to 5% for the rest of a long prompt. The junction-to-edge gap is a
steady 22 to 25 C at 300 W, which is normal for the die-to-heatsink interface (no repaste
needed); what remains is the fin stack's own limit at this power. Two-minute burst test with
the cool intake: edge 78 -> 72 C, junction 101 -> 97, clock 2345 -> 2445 MHz at the end, pp2048
473 -> 481 t/s.

**Thermal, final state (2026-09-30).** Shroud leak at the power connectors sealed, both side
fans 12 V: sustained 120k prefill at 300 W 281.5 t/s, power 299 W mean the whole run, clock
median 2435 MHz with brief dips to 2275 to 2345, junction settling at 98 to 99 C (momentary
101), edge 74 to 75 C, memory 74 C. Against the cool-card short-burst rate this is a loss of
about 1%, down from 5 to 8% with the original airflow. The remaining margin is the fin stack at
this power; further fans would be diminishing returns. The earlier statement that long
prompts lose 5 to 8% is retracted for this setup.

**Item 5c, rejected: 64-column D=256 attention tile.** Added an RDNA2 dispatch branch and config
for 64 columns per block (32 per head group). 8 warps with 8 columns each: 163 VGPRs, no spill,
17 KB LDS, but 14.0 TFLOPS against 20.9 for the 32-column tile (pp512 at 64k 279 -> 256 t/s).
16 warps with 4 columns each: 102 VGPRs, 19.5 to 20.5 TFLOPS, still slower. Neither is
byte-identical to the 32-column kernel at a pinned KV split (a handful of elements differ), so
it fails the rule as well as the clock. Reverted. The reviewer's +10% estimate did not survive
contact; the wider tile halves K reuse per warp only on paper, since the V pass and the
softmax scale with columns per warp too.

**Item 4, rejected: MTP catch-up merged into the first draft pass.** Implemented (process()
leaves the catch-up batch pending, draft() keeps the accepted prefix, appends its row and decodes
once; flush and drop paths for prompt chunks and new prompts). Greedy text identical on four
prompts including a chunked long prompt, but MTP fell from 63.8 to 50.5 t/s on math. Bisected:
the same plumbing with a separate catch-up decode is already 12% slower (56.1 t/s), because
upstream issues the catch-up from inside the target's decode callback, where it overlaps with
the target's GPU work; anything done in draft() runs after the host has waited for the target's
logits and is fully exposed. The merged decode on top of that lowers draft acceptance from 0.81
to 0.58 (mean draft length 3.4 -> 2.7), so the batched drafter pass is also not equivalent for
the later draft rows. Reverted. The review's "+3%" assumed the catch-up was on the critical path;
it is not.

With items 4 and 5c closed, every item of the plan is either landed (1, 3a, 3b), measured and
rejected (3c, 4, 5a, 5c, 6), or dropped as not bit-exact (7, 8). The landed set is worth about
+10% on batch-8 verify (DFlash +5 to 12%), +1.5% on decode, and the thermal envelope is
characterized. What remains above the current numbers needs either the bit-exact rule relaxed
(q8_0 KV for long-context decode, the q8_1 block-sum min term in MMVQ) or a different algorithm
for the batch-4-to-8 matmuls.

## 23. 2026-09-30: research - the verify-batch matmul (MMVQ at 4 to 8 columns)

**Terms.** GEMV is a matrix times one vector (decode); GEMM is a matrix times many columns
(prefill). MMVQ (`mul_mat_vec_q`) is llama.cpp's quantized GEMV kernel, extended to up to 8
activation columns; every verify step of speculative decoding (draft tokens plus one) runs on
it. MMQ (`mul_mat_q`) is the quantized GEMM used for prefill; below about 13 columns it is
slower than MMVQ because it dequantizes a 128-row tile into shared memory whatever the column
count.

**How MMVQ works** (Q4_K): a 256-weight block is split across 16 lanes, 16 weights each; a
workgroup of one wave handles 4 rows; each lane, per K iteration, loads 2 ints of weights per
row, the scales, and for every column 4 ints of q8_1 activation plus 2 block scales, then per
(row, column) runs 4 `v_dot4` for the weights and 2 for the activation sums, and applies the
sub-block scale and min in float. Partial sums are reduced across the 16 lanes at the end.

**Cost per column, measured** (16 distinct ffn-sized matrices per graph, so nothing sits in the
infinity cache; ms per matmul, Q4_K 5120 x 17408):

| columns | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ms | 0.139 | 0.140 | 0.142 | 0.143 | 0.147 | 0.149 | 0.169 | 0.208 |
| VGPRs of the instance | 31 | 62 | 62 | 78 | 132 | 155 | 173 | 195 |
| waves per SIMD | 8 | 8 | 8 | 8 | 7 | 6 | 5 | 5 |

Flat to 6 columns (+7%), then +14% at 7 and +40% at 8. The per-column arithmetic is not the
problem: the 8-column instance issues the fewest instructions per dot product of all (5.9,
against 6.7 at 4 columns and 13.6 at 1). The instances lose occupancy as their register count
grows, and at 5 waves per SIMD the memory latency is no longer hidden. Q5_K is worse (199
VGPRs at 8 columns) and is 37% of the weights.

**Instruction census of the 8-column K loop** (939 instructions per iteration, 160 dot4):
float multiply/fma/add 352 (the per-sub-block scale and min epilogue, fixed under bit-exact
rules), address and integer 142, int-to-float conversions 100, global loads 72 (all hoisted to
the top of the iteration), nibble unpacking 68, wait counts 21. The reducible part without
touching the arithmetic order is at most a quarter of the loop.

**Experiments, all bit-identical by construction (same per-output operation order):**
- Compiler barrier splitting the 8 columns' activation loads into two halves (to cut live
  registers): Q4_K 8-column VGPRs 195 -> 184, still 5 waves; Q5_K got worse (220); in-model
  pp8 127 -> 119. The exposed second-half loads cost more than the occupancy gained. Rejected.
- DFlash draft cap 6 instead of 7 (verify batch 7 instead of 8): math 63 vs 69 t/s, code and
  essay equal. The batch-8 cost is worth the acceptance. Cap stays at 7.
- 2 rows per block instead of 4 for the 7/8-column instances: VGPRs 195 -> 111 (Q4_K),
  199 -> 121 (Q5_K), all instances back to 8 waves per SIMD, and slower: microbench N=7 0.169 ->
  0.189 ms, N=8 0.208 -> 0.200; in-model pp7 121 -> 109, pp8 128 -> 124, DFlash math 72.5 ->
  69.4. Each activation load now serves 2 rows instead of 4, and the extra cache traffic costs
  more than the occupancy gains. Rejected.

**Where this leaves the verify matmul.** At 4 columns (MTP) it runs at 86% of the bandwidth
line and there is nothing to take. At 7 and 8 columns (DFlash) it is 40% over the line because
of a register cliff: the instances fit 5 waves per SIMD, and both ways of buying occupancy
(fewer live loads, fewer rows per block) cost more in exposed latency or lost reuse than they
return. The float epilogue that dominates the loop is fixed by the bit-exact rule, since any
refactoring of the per-sub-block scale and min terms (deferring the scale, using the q8_1 block
sums for the min term, splitting K across waves) changes the rounding order. The DFlash cap of 7
is still the right setting despite the cliff, because the eighth row buys acceptance on math.

What a purpose-built kernel would do, and why it is off the table under the rule: hold the
activation columns and a K slice in registers, split K across 2 to 4 waves per row group so the
latency is hidden by parallelism instead of by registers, and apply the K-quant scales once per
sub-block rather than once per column. Every one of those reorders the summation. The prize if
it reached the bandwidth line: batch-8 verify 60 -> 34 ms, DFlash iteration about 48 -> 30 ms,
roughly +40 to +60% on math and code with the drafter; MTP +15%. That is the largest remaining
gain on this card, and it needs "same math, any summation order" to be acceptable.

**Reordering allowed: a purpose-built kernel, three layouts, all slower.** With the
summation-order rule relaxed for this matmul (measured first: perplexity through the 8-column
and 4-column verify kernels vs the prefill GEMM is 3.7865 / 3.7894 / 3.7956 +/- 0.206, a
twentieth of the error bar), a new kernel was written for Q4_K, Q5_K and Q6_K at 2 to 8 columns
(`docs/phoebe/mmvq-wide-rejected.cuh`): a lane owns whole 16- or 32-weight runs read with
16-byte loads, each column's activations come as 16-byte loads, the sub-block scales are applied
once per lane, no activation-sum dot products in the final variant. Numerically correct (max
error 1e-6 of the output rms against the generic kernel on all three types).

| variant | Q4_K, 8 columns | note |
|---|---|---|
| generic MMVQ | 0.208 ms | |
| 32 weights per lane, block sums for the min term | 0.195 ms | min term uses the prefill path's approximation; 4 to 6 columns 10 to 15% slower |
| 16 weights per lane, exact sums, 4 rows per wave | 0.239 ms | |
| same, 2 rows per wave | 0.257 ms | 6 waves per SIMD |
| same, 8 rows per wave | 0.325 ms | spills |

Counters for the last layout vs generic at 8 columns: VALU instructions -37%, memory
instruction cycles -47%, wait cycles -78%, occupancy 27% vs 22%, and 19% more busy cycles,
with the texture-address unit at 91 to 93% in both and 13% busier for the wide kernel. On
RDNA2 that unit is the resource both kernels saturate, and it charges per lane and cache line
per instruction, not per instruction: the generic kernel's pattern of consecutive lanes reading
consecutive dwords is already the cheapest per byte, and wider loads from lane-strided addresses
cost the same or more. The 40% cliff at 8 columns is that unit's cost for the extra activation
re-reads, and no lane mapping over the GGUF layout reduces it: every wave must re-read the
activations for its rows, and a wave cannot cover more rows without either spilling or reading
its weights lane-strided. The only design left is a repacked weight layout (rows interleaved in
16-byte pieces so a wave streams contiguous 512-byte runs with one row per lane, activations
broadcast through the scalar cache), which is a custom buffer type and a project of its own.

Verify matmul, final: the generic MMVQ stays. Batch 4 at 86% of the bandwidth line, batch 8 at
55%, and the gap is the address unit, not arithmetic or occupancy.

## 24. 2026-09-30: repacked weight layout for the verify matmul (experiment, GGML_CUDA_ROWPACK=1)

**Idea.** Section 23 found the generic MMVQ kernel bound by the texture-address unit, mostly on
the activations it re-reads for every 4 rows. If the weights are stored so that 32 consecutive
rows are interleaved in 16-byte pieces, a wave with one row per lane streams contiguous
512-byte runs, the activations become wave-uniform and can be read through the scalar path (an
SGPR operand of `v_dot4`), never touching the vector memory unit, and each lane owns whole rows,
so there is no cross-lane reduction. Small row counts split K across waves and reduce the
partials in a fixed order. The layout is built once per weight tensor into a side buffer
(`ggml/src/ggml-cuda/mmvq-rowpack.cuh`, RDNA2, Q4_K/Q5_K, 2D weights, no fusion; the Q6_K
variant exists but is disabled, see below).

**Numerics.** fp32 summation in a different order and the q8_1 block-sum min term (as MMQ).
Q6_K, which has no min term, matches the generic kernel to 2e-6 of the output rms; with exact
activation sums Q4_K matches to 1e-3 (the fp16 block scale both carry); perplexity through the
path: 3.7927 (Q4_K) / 3.7754 (Q5_K) against 3.7865 +/- 0.206 generic.

**What it took to make it fast.** The standalone kernel was 20% faster than the generic one from
the start; inside ggml it was not, because the compiler did the activation address arithmetic
in 64-bit vector registers and moved every address to the scalar unit (68 `v_readfirstlane` and
1,000 extra VALU instructions per super-block). 32-bit offsets with per-column bases hoisted out
of the loops and explicitly wave-uniform split bounds fixed it. The split target (about 1,000
waves) and the column threshold (6) were swept. Staging the activations through shared memory
with a one-block-ahead prefetch was measured 5x slower: broadcast reads land in vector
registers, both dot operands then need VGPRs, and the 8-column instance spills.

**Microbench, ms per matmul (16 distinct matrices per graph):**

| shape, columns | generic | repacked |
|---|---|---|
| Q4_K 5120x17408, 4 / 6 / 8 | 0.143 / 0.149 / 0.204 | 0.154 / 0.166 / 0.170 |
| Q4_K 17408x5120, 4 / 6 / 8 | 0.150 / 0.155 / 0.197 | 0.149 / 0.152 / 0.156 |
| Q4_K 5120x10240, 4 / 6 / 8 | 0.101 / 0.107 / 0.138 | 0.105 / 0.108 / 0.111 |
| Q5_K 5120x17408, 4 / 6 / 8 | 0.169 / 0.172 / 0.217 | 0.174 / 0.186 / 0.195 |
| Q5_K 17408x5120, 4 / 6 / 8 | 0.186 / 0.179 / 0.220 | 0.169 / 0.173 / 0.178 |
| Q6_K 5120x17408, 5 / 8 | 0.199 / 0.217 | 0.233 / 0.230 (disabled) |

At 8 columns 17 to 21% faster (Q4_K) and 10 to 19% (Q5_K); at 4 to 6 columns even or slightly
slower, so the path is taken from 6 columns up. Counters at 8 columns (Q4_K): VALU instructions
-81%, vector-memory cycles -96%, busy cycles -56%, memory unit 57% busy against 93%; the
remaining time is exposed scalar-load latency (about 8 waits per super-block per wave with ~7
waves per SIMD), which shared-memory staging cannot fix (above) and SGPR prefetching cannot
hold (104 SGPRs against 144 per column chunk). Q6_K's generic kernel is already the most
efficient of the three (fewest loads per byte) and the repacked one does not beat it.

**End to end** (llama-bench and greedy DFlash on build-dev; the side buffers duplicate the
weights, so Q4_K and Q5_K together only fit at 4k context and not with a drafter loaded):

| | generic | Q4_K repacked | Q5_K repacked | both |
|---|---|---|---|---|
| pp8 | 127.8 | 138.3 | 136.3 | 148.5 |
| pp6 (threshold 6) | 120.6 | | | 115.6 |
| pp4 / tg32 | 82.9 / 24.8 | 82.8 / 24.8 | | 82.2 / 24.8 |
| DFlash math / code / essay | 72.3 / 62.2 / 40.4 | 75.0 / 63.0 / 43.1 | 74.2 / 62.0 / 39.4 | (does not fit) |
| MTP math | 63.9 | 63.9 | | 64.0 |
| PPL, 8-token micro-batches | 3.7865 | 3.7927 | 3.7754 | 3.7749 |

So the batch-8 verify step is 17% faster with both types, which the drafters turn into +2 to
+7% by prompt (the verify step is about 60% of a DFlash iteration and only its 8-row batches
qualify); 6 columns is slower, so the threshold is 7; MTP (batch 4) and plain decode do not
change. The block-sum min term and the reordered summation move perplexity by less than a
tenth of its error bar.

**Where it stands.** The layout works and the kernel is where the section 23 analysis said the
gain would be, but the shipping cost is the layout itself: the side buffers cannot coexist with
the model (14 GB), so a real deployment has to store the weights in this layout only, which
means every kernel that reads Q4_K/Q5_K weights on the device (MMQ for prefill, the
dequantizers, get_rows) has to understand it, or the tensors have to be stored twice. The
matmul gain itself is real but bounded: the path is at 57% memory-unit busy and stalls on
scalar-load latency, and the 8-column verify step goes from 60 to about 50 ms per iteration,
not to the 34 ms bandwidth line. Left as an environment-gated experiment (default off);
`GGML_CUDA_ROWPACK=1`, `_TYPES`, `_MIN`, `_WAVES` select it.

## 25. 2026-09-30: final performance table (build 2236e6b42)

Swift-Qwen3.8-27B Q4_K_XL, one V620 at 300 W (sealed shroud, two 12 V 120 mm fans on the card,
section 22), build 2236e6b42, 128k context, flash attention, repacked layout off (its default;
the side buffers do not fit at 128k with a drafter). The server was restarted on this build.

Prefill and plain decode (llama-bench, no drafter; depth = tokens already in the context):

| depth | pp512 t/s | pp2048 t/s | tg32 t/s |
|---|---|---|---|
| 0 | 494 | 488 | 24.8 |
| 4k | 471 | 465 | 24.4 |
| 16k | 418 | 414 | 23.3 |
| 32k | 359 | | 22.1 |
| 64k | 278 | | 20.1 |
| 128k | 172 | | 17.0 |

Verify-sized batches: pp4 82.8, pp8 127.2 (148.5 with `GGML_CUDA_ROWPACK=1` and both types,
section 24).

Decode with the drafters (llama-server, 256 generated tokens, temperature 1.0 / top-p 0.95 /
top-k 20 with `--spec-draft-temp 1.0`; prompt = the first N tokens of wikitext plus "write a
400-word summary"; mean of 2 runs):

| depth | none | MTP (n=3) | DFlash2 (adaptive, cap 7) | MTP accepted/drafted | DFlash accepted/drafted |
|---|---|---|---|---|---|
| ~0 (essay prompt) | 24.7 | 48.2 | 48.3 | 0.52 | 0.51 |
| 3.8k | 24.3 | 51.6 | 51.0 | 0.60 | 0.52 |
| 16.7k | 23.2 | 51.4 | 59.9 | 0.64 | 0.60 |
| 40k | 21.5 | 48.2 | 42.9 | 0.66 | 0.50 |
| 78k | 19.4 | 48.5 | 39.7 | 0.80 | 0.53 |
| 120k | 17.5 | 44.4 | 32.8 | 0.83 | 0.49 |

Whole-prompt prefill of the 120k prompt from an empty cache: 281 t/s without a drafter, 271
with MTP, 276 with DFlash.

Against section 19 (build 5c913c0d2): prefill is unchanged (same kernels; the +1% is the
cooling); plain decode is 0.5 to 1% faster from the norm-to-q8_1 fusion (section 22); the
drafted columns are within run-to-run noise. The noise floor of the drafted columns is about
2 t/s: the two MTP runs at 3.8k gave 54.2 and 49.0, since the drafts are sampled at temperature
1.0 and the acceptance varies with the sample. The 16k row is a longer prompt than section 19's
(16.7k against 13.4k tokens), so its DFlash number is not a like-for-like comparison. The
picture from section 19 stands: MTP holds 2.5x over plain decode to 120k, DFlash2 is ahead up
to about 16k-30k and falls off with depth, and the server ships with DFlash for short mixed
work.

## 26. 2026-10-03: wrap-up

Decisions at the end of the optimization pass:

- The repacked weight layout (section 24) is not pursued further. Shipping it means holding
  the weights on the device in that layout only, so the prefill matmul, the dequantizers, the
  row gather and the 1-to-6-column vector kernel would all have to read it, and the last of
  those measured slower in that layout. The gain is bounded at +2 to 7% on DFlash prompts. It
  stays as the `GGML_CUDA_ROWPACK=1` experiment.
- The server on this machine now runs the MTP drafter by default (`--spec-type draft-mtp`,
  3 tokens, `--spec-draft-temp 1.0`): its use is research chat with long tool results, where
  MTP holds about 48 t/s to 120k of context and DFlash2 falls to 33 (section 25). DFlash2 stays
  the default in `deploy/serve-v620.sh` (`SPEC=dflash`), the better choice for short mixed
  work. The server is a systemd user service with lingering enabled, so it starts at boot.
- The harnesses behind the measurements are in `docs/phoebe/bench/` (see its README).
- Unused models and the quantization sources (the F16 and Q8_0 of Swift, the base Qwen3.8
  quant, the full-vocabulary drafters) were removed locally; `deploy/scripts/make-models.sh`
  rebuilds everything from the public sources.
