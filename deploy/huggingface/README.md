---
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model: ukisai/Swift-1.5-Qwen3.8-27b
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - rocm
---

# Swift 1.5 Qwen3.8-27B GGUF for one Radeon PRO V620

A quantization of UkisAI's [Swift 1.5 Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
that fits one 32 GB GPU with 64k context, plus two speculative-decoding drafters for it. Built
and measured on an AMD Radeon PRO V620 (Navi 21, gfx1030), a datacenter card that sells cheaply
second-hand. The full setup (BIOS, kernel, ROCm, llama.cpp build, server, web tools) is in
[llama-navi21-furnace/deploy](https://github.com/sixvolts/llama-navi21-furnace/tree/main/deploy).

| file | size | what it is |
|---|---|---|
| `Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf` | 16.2 GiB | the model |
| `dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf` | 1.3 GiB | DFlash2 block drafter |
| `mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf` | 1.0 GiB | multi-token-prediction (MTP) drafter |

**Use the llama.cpp fork**
[sixvolts/llama-navi21-furnace](https://github.com/sixvolts/llama-navi21-furnace), branch
`main`, for the drafters: the MTP drafter's reduced vocabulary (below) and
`--spec-draft-temp` are not in upstream llama.cpp. The model itself is a standard GGUF.

## The model

Unsloth's UD-Q4_K_XL per-tensor recipe for Qwen3.8-27B, applied to Swift 1.5's BF16 weights with
Unsloth's importance matrix, except that the tensors Unsloth stores in IQ formats are stored as
Q4_K (IQ formats dequantize slowly on gfx1030). Token embeddings Q4_K, output Q6_K. The MTP
layer is left out of the file (see the MTP drafter below).

KL divergence against Swift's own Q8_0, on held-out text, measured with the same recipe on
Swift 1.0 (16.57 GiB there, with its MTP layer):

| file | size | KLD |
|---|---|---|
| this file | 16.57 GiB | 0.0092 |
| Swift's Q4_K_M | 16.79 GiB | 0.0134 |

## The drafters

A drafter proposes tokens and the model checks them in one batch, so the output is the model's
own; the drafter only changes speed.

- **DFlash2**: z-lab's [Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
  drafter, quantized from BF16 to Q4_0 (the fastest of the drafter formats tested on gfx1030).
- **MTP**: Unsloth's MTP drafter for Qwen3.8-27B
  ([`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)). Swift 1.5
  did not retrain the MTP head (its `mtp.*` weights match the base model's to quantization
  rounding), so the base model's works unchanged.
- **Reduced draft vocabulary (d2t64k)**: in both, the full 248k-token output head is replaced by
  the 65,536 most likely rows of **Swift's** output projection (this file's Q6_K rows, copied
  unchanged) and a map from those rows to token ids. The head is most of a drafter's cost, so this
  makes drafting several times cheaper; the kept set covers 98.9% of held-out model output, and
  tokens outside it simply cannot be drafted.

## Speed on one V620

Measured on Swift 1.0, which has the same architecture and quantization. Swift 1.5 decodes at the
same speed with each drafter (5 prompts, temperature 1.0, 2 seeds: no drafter 23.5 against
23.9 t/s, MTP 48.5 against 47.9, DFlash2 48.0 against 48.4), and on the same prompts
(10 answers each, DFlash2) it reached finished answers in 6% fewer tokens.

llama-server, warm, Swift sampling settings (temperature 1.0, top-p 0.95, top-k 20), with
`--spec-draft-temp 1.0` (drafts sampled from the drafter and checked with speculative sampling,
which leaves the output distribution unchanged):

| drafter | research chat | math / code / list / essay |
|---|---|---|
| none | 24.3 t/s | about 24 t/s |
| MTP, 3 tokens | 44.2 t/s | 62 / 54 / 64 / 43 t/s |
| DFlash2, adaptive up to 7 | 42.3 to 43.5 t/s | 76 / 54 / 69 / 40 t/s |

At depth (decode t/s with the given number of tokens already in the context; temperature 1.0,
a long-text summary task):

| depth | none | MTP | DFlash2 |
|---|---|---|---|
| 4k | 24.1 | 55.3 | 50.0 |
| 16k | 23.0 | 53.0 | 55.3 |
| 40k | 21.4 | 48.0 | 41.6 |
| 80k | 19.3 | 47.2 | 39.2 |
| 120k | 17.4 | 44.1 | 32.0 |

DFlash2 is the faster drafter for short mixed work; past about 30k tokens of context MTP is,
because its cost does not grow with depth. Prefill (no drafter): 492 t/s at an empty context,
416 at 16k, 278 at 64k, 172 at 128k.

```
llama-server -m Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf -ngl 99 -fa on -c 65536 \
  -md dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-temp 1.0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

For MTP: `-md mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf --spec-type draft-mtp --spec-draft-n-max 3`.

## Reproducing

[`deploy/scripts/make-models.sh`](https://github.com/sixvolts/llama-navi21-furnace/blob/main/deploy/scripts/make-models.sh)
rebuilds all three files from the upstream sources; with the same llama.cpp build the results are
byte-identical to these (`SHA256SUMS`).

## License and changes

Swift 1.5 Qwen3.8-27B is UkisAI's continuation of Swift 1.0, itself a fine-tune of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Copyright 2026 Alibaba Cloud, Apache
License 2.0). UkisAI's contribution is licensed under the
**Swift Open License v1.0** (`LICENSE`): free for personal, research, educational and evaluation
use, and for commercial use by individuals and organizations with gross annual revenue up to
US$1,000,000; above that, commercial use requires a Swift Enterprise License from
[UkisAI](https://ukisai.com/contact). UkisAI's notices are in `NOTICE`; the Apache License 2.0 is
in `LICENSE-APACHE-2.0`.

All three files contain Swift weights (the drafters contain rows of Swift's output projection),
so all three are under the Swift Open License. The DFlash2 drafter body (z-lab) and the MTP
drafter body (Unsloth) are Apache License 2.0.

Changes made here (Swift Open License section 4(b)):

- `Swift-1.5-Qwen3.8-27B-Q4_K_XL-noIQ.gguf`: converted from UkisAI's `model-*.safetensors` to a
  BF16 GGUF with llama.cpp `convert_hf_to_gguf.py --no-mtp` (the MTP layer and the vision encoder
  are left out), then quantized with `llama-quantize`, per-tensor types
  as described above, importance matrix `imatrix_unsloth.gguf` from
  [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). No retraining.
- `dflash-Qwen3.8-27B-Q4_0-d2t64k-swift15xl.gguf`: z-lab's DFlash2 BF16 drafter quantized to Q4_0;
  output head replaced by 65,536 rows of the output projection of the Q4_K_XL file above, plus a
  `d2t` token map. No retraining.
- `mtp-Qwen3.8-27B-d2t64k-swift15xl.gguf`: Unsloth's MTP drafter; output head replaced the same way.
  No retraining.
