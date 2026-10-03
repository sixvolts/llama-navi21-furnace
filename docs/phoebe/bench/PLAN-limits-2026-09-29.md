# Plan: closing the gaps to the hardware limits on one V620

Context: llama.cpp fork `llama-navi21-furnace` (main, commit f6ee61ce5) on a single Radeon PRO
V620 (gfx1030, RDNA2, wave32, 72 CUs / 36 WGPs, 32 GB GDDR6 at 505.6 GB/s measured, 300 W cap
via a patched amdgpu, sclk max 2570 MHz). Model: Swift-Qwen3.8-27B Q4_K_XL (17.06 GB read per
decode token, 52.1 GFLOP per token in the matmuls, 65 layers of which 16 are full attention with
GQA 24:4 and D=256, 48 are gated delta-net linear attention). Drafters: MTP (3 tokens, verify
batch 4) and DFlash2 (adaptive up to 7, verify batch 4 to 8).

Ground rule from the owner: kernel changes must be bit-exact (same arithmetic, same rounding);
anything that changes numerics is abandoned. Reordering a reduction (e.g. a split-K) is exact math
but not bit-identical to the current build; that needs an explicit decision (see gate G1).

## 1. The measured gaps (profile pass, PROGRESS section 20)

| area | limit | measured | gap |
|---|---|---|---|
| decode batch 1 | 33.8 ms/tok (bandwidth) | 40.5 ms | MMVQ +1.7 ms; 1,050 small kernels 3.3 ms; gaps 1.7 ms |
| decode at 64k / 128k | 42.2 / 50.7 ms | 50 / 59 ms | attention at 89% of KV bandwidth; the rest as above |
| prefill MMQ (89% of prefill) | 90 TOPS int8 dot4 (76.7 achievable) | 28.8 TOPS | 3x; VALU-issue-bound, 23% occupancy, power-capped |
| prefill attention 16k / 64k | 45 TFLOPS dot2 | 21.5 / 17.1 | 2 to 2.6x; latency-bound, 42% of the ubatch at 64k |
| verify batch 4 / 8 (MMVQ) | 33.8 ms | 42.7 / 65.8 ms (MMVQ 39 / 60.6) | 5 / 27 ms per iteration |
| MTP drafter (4 passes) | 4.3 ms | 6.1 ms | 1.8 ms |
| DFlash drafter (1 block pass) | 2.7 ms | 5.6 ms | 2.9 ms (batch-8 MMVQ) |
| thermal | 100 C junction | 98 C after 22 s of prefill at 300 W, sclk 2445 -> 2425 | will throttle on long prompts |
| decode power | - | 261 W at 2480 MHz for a bandwidth-bound phase | ~100 W wasted |

## 2. Work items

### W0. Thermal and power envelope (first, half a day, no code)
Why first: every prefill number is taken at the 300 W cap; if the junction reaches 100 C on a long
prompt the card throttles and the numbers are not reproducible.
- Sustained prefill test: a 120k-token prompt (8 min) with hwmon sampling at 0.5 s: junction,
  sclk, power, and the server's per-chunk prefill rate. Pass: junction stays below 100 C and sclk
  does not fall below 2400 MHz. Fail: either.
- Power-cap curve: llama-bench tg128 and pp2048 at caps 300 / 275 / 250 / 225 / 200 W (the cap is
  runtime-settable through hwmon power1_cap). Gives the t/s-per-watt curve for both phases and
  says what a lower cap costs prefill and whether decode loses anything at 200 W.
- Airflow: check the card's cooling (the V620 is passive; it depends on chassis airflow), the
  intake temperature and fan curve. If the sustained test fails, the choices are more airflow, a
  cap chosen from the curve (e.g. 275 W if it costs under 3%), or a cap switched by phase (W6).
Deliverable: the curve and a decision on the shipped cap. Blocks nothing else but reorders W2.

### W1. KV cache in q8_0 for long-context decode (half a day, no kernel change)
Hypothesis: at 128k the KV read (8.6 GB) is 34% of the decode roofline; q8_0 halves it. The tile
kernel dequantizes q8_0 K/V to f16 on load, so prefill may lose and decode should gain.
- A/B llama-bench: tg32 and pp512 at depths 0 / 16k / 64k / 128k, f16 vs q8_0 KV, interleaved.
- Quality gate: PPL at 4k and a needle test at 64k; the greedy text will differ (different
  cache precision), which is expected and is a model-output decision, not a kernel one.
- Also measure VRAM: q8_0 frees ~4 GB at 128k, which is the difference between 128k and 200k+.
Expected: +10 to 20% decode at 128k, some prefill loss at depth. Ship only if prefill at 64k loses
less than the decode gain, or make it a launcher option (`KV=q8_0`).

### W2. Batch 4 to 8 MMVQ: latency hiding without changing accumulation order (2 days)
The largest absolute gap: 27 ms per DFlash iteration, 5 ms per MTP iteration. Prior work
(mtp-perf-anatomy note) closed: nwarps / rows_per_block sweeps, LDS-staged activations (2x slower),
widened weight loads (no change), column split across warp groups (weights re-read, 40% slower),
MMQ below batch 13 (worse). Counters: OccupancyPercent 22%, 200 VGPRs, SQ_WAIT_ANY about 40x
SQ_BUSY_CYCLES, L2 hit 38%, MemUnitBusy 89%. Reading: each wave issues a weight load, waits the
full HBM latency, then computes; with 2 to 3 waves per SIMD nothing hides the wait.
- W2a (bit-exact): explicit software prefetch in the ncols 4 to 8 kernel: load the next K block's
  weight ints (and scales) into registers before computing the current block, i.e. a two-deep
  register pipeline in the `mul_mat_vec_q` loop for RDNA2 only. Accumulation order unchanged.
  Cost: ~40 to 60 more VGPRs; at 200 already, so this may need rows_per_block 4 -> 2 for the
  ncols 8 variant to fit (measure both). Expected: batch 8 MMVQ 60 -> 45 ms if latency is the
  limiter; if it is not, the counters will say so (SQ_WAIT_ANY should drop).
- W2b (bit-exact): interleave the per-row dot chains so that the 4 rows' `v_dot4` chains are
  independent instruction streams (they are today, but the ISA should be checked for
  serialization through shared temporaries).
- W2c (not bit-identical, gate G1): split-K. Each workgroup handles a K slice of its rows; partial
  sums are reduced in a second tiny pass (or with atomics in a fixed order). Weight traffic stays
  1x (unlike the column split), occupancy rises 2 to 4x. Changes the float summation order.
  Only if W2a/b do not deliver, and only with the owner's decision.
Measurement: rebuild the `mmbench` harness (per-shape MUL_MAT timing, bitwise output compare
against the current kernel), rocprofv3 counters, then llama-bench pp4/pp8 and the DFlash/MTP
end-to-end. Gate: bitwise identical outputs for W2a/b; test-backend-ops MUL_MAT.

### W3. Decode: fewer launches (2 days, bit-exact by construction)
1,050 non-matmul launches per decode step cost 3.3 ms of GPU time plus ~1.1 us each in dispatch
(1.7 ms with graphs on). Each fusion saves the fixed cost of a kernel (~3 to 5 us) and a launch.
Candidates from the trace (launches per step / ms):
- norms 257 / 0.87 and q8_1 activation quant 257 / 0.46: fuse the quantization into the norm
  kernel that feeds each matmul (norm output goes straight to q8_1; identical values, since the
  quantizer runs on the same fp32 result). Saves 257 launches and ~0.4 ms.
- copies 112 / 0.32: KV-cache writes and the delta-net state snapshots; part is already fused
  (multi-copy launch). Check what remains and whether the KV write can go into the rope kernel.
- rope 32 / 0.16 and get_rows 49 / 0.17: small; rope into the FA prologue is not bit-exact-safe
  (rope math must stay identical) but is a plain move of the same computation.
- delta-net 96 / 0.62: two launches per layer (conv + gated delta); check whether the conv can be
  fused into the gdn kernel for batch 1.
Expected: 1,050 -> ~600 launches, 40.5 -> ~38.5 ms (+5% decode, similar in verify).

### W4. Prefill MMQ: instruction count per dot4 for the K-quants (3 days, uncertain)
Same shape, Q8_0 runs at 37 TOPS and Q4_K at 26; the K-quant loop carries more instructions per
`v_dot4` (LDS reads per dot, unpacking, per-sub-block scale application). The integer-scale fix
already gave +8.5%. Steps:
- ISA accounting: for `mul_mat_q<Q4_K,64>` and `<Q8_0,64>` count per inner-loop iteration:
  v_dot4, ds_read (and width), other VALU, and attribute every non-dot instruction.
- If the LDS tile layout for Q4_K forces narrower or more LDS reads than Q8_0's, change the
  layout (load_tiles) so the loop body matches Q8_0's; the dequantized 8-bit values and the
  per-sub-block correction term are unchanged, so results stay bit-exact.
- If the difference is in the scale epilogue, it is already at the formula minimum (section 17).
Expected if the loop reaches Q8_0 parity: +20% prefill at empty context, which is also a power
reduction per token (the card is at the cap). Risk: the layout is shared across GPUs; keep the
change under the RDNA2 config.

### W5. Long-context attention (1 to 2 days, uncertain)
- W5a: with Q no longer in LDS (13 KB per block), test a 64-column tile for D=256 prefill
  (`ncols 64`, currently capped at 32 for GQA 6 via ncols2=2). Doubles the K reuse per LDS read.
  Risk: accumulator registers double; the compiler spills at 256 VGPRs. Config-table experiment.
- W5b: probe the V pass share (skip it in a timing build) and the softmax share, to know where the
  remaining 60% goes; the KQ loop was the whole story before the global-Q change, now it is not.
- W5c: decode at depth reads KV at 89% of bandwidth; nothing to do there.
Expected: +10 to 20% on the attention share at 64k (attention is 42% of the ubatch), so +5 to 8% on
pp512 at 64k.

### W6. Power policy for decode (half a day, no kernel change)
Decode is bandwidth-bound and draws 261 W at max clock. From the W0 curve, if decode holds its
speed at 200 W, the server can pin a lower cap during generation and raise it for prefill: a
small helper (sysfs writes need root; a systemd path unit or a setuid helper) called from the
server's slot state changes. Saves ~60 to 100 W in the phase that dominates chat time, and keeps
the junction lower for the next prefill. No speed gain; a thermal and cost gain.

### W7. MTP drafter catch-up merge (half a day)
Per iteration the MTP drafter runs 3 draft passes and 1 catch-up pass (6.1 ms GPU). The catch-up
(ingesting the accepted tokens) can be batched with the first draft pass of the next iteration:
3 passes instead of 4. ~1.5 ms per iteration, +3% on MTP. Bit-exact.

## 3. Order and expected payoff

| order | item | effort | expected | risk |
|---|---|---|---|---|
| 1 | W0 thermal/power envelope | 0.5 d | reproducibility; shipped cap decision | none |
| 2 | W1 q8_0 KV at depth | 0.5 d | +10-20% decode at 128k | quality is a model decision |
| 3 | W2a/b MMVQ prefetch | 2 d | batch-8 verify 66 -> ~50 ms: DFlash +15-25% on math/code, MTP +5-8% | may not be the limiter |
| 4 | W3 decode fusions | 2 d | +5% decode and verify | low |
| 5 | W7 MTP catch-up merge | 0.5 d | +3% MTP | low |
| 6 | W4 MMQ K-quant loop | 3 d | up to +20% prefill | uncertain; shared code |
| 7 | W5 attention tiles | 1-2 d | +5-8% pp at 64k | spills |
| 8 | W6 decode power policy | 0.5 d | -60-100 W in decode | needs root plumbing |
| G1 | W2c split-K | decision | batch-8 to ~40 ms | not bit-identical |

## 4. Gates for every kernel change
- Bitwise output compare against the current kernel on the exact model shapes (mmbench for
  matmuls, facmp for attention with the KV split pinned).
- test-backend-ops for the op; the check.sh gate (greedy text plain/MTP/DFlash, PPL).
- Interleaved A/B with the warm server at empty context and 64k; power and junction logged.
- Any change that is not bit-exact is reverted, except through an explicit decision (G1).
