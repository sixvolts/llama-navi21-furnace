#!/usr/bin/env python3
"""Give a speculative-decoding drafter a reduced draft vocabulary (d2t).

A drafter only proposes tokens - the target verifies them - so its LM head does not need the
whole vocabulary. For Qwen3.5-family models (248k tokens) the head is most of the drafter's
cost; keeping the K most likely tokens makes it several times cheaper while the output stays
identical (tokens outside the set just cannot be drafted).

Writes a copy of the drafter with
  <head> [n_embd, K]   K rows of the TARGET's output projection, copied as raw quantized
                       bytes, so every kept token's logit is exactly what the drafter
                       computed before
  d2t    [K] i64       head row -> target token id
where <head> is output.weight for DFlash drafters and blk.<L>.nextn.shared_head_head.weight
for Qwen3.5 MTP drafters (whose full-vocabulary output.weight is dropped). EAGLE-3/4 drafters
(architecture eagle3) have their own trained head: its K rows are kept instead of the target's.

Token set: every non-normal (control / user-defined) token, then the ranking file (most
frequent first), then the lowest remaining ids - BPE ids follow merge order, a reasonable
frequency prior for whatever the ranking does not cover.

The shipped qwen3.5-token-ranking.txt counts Qwen3.8-27B's own chat output (code, math,
prose, lists, several languages, thinking on), wikitext and C/C++/Python source. With
K=65536 it covers 98.9% of held-out model output.

usage: build-draft-vocab.py <drafter.gguf> <target.gguf> <out.gguf> [--k 65536] [--ranking FILE]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('drafter')
ap.add_argument('target')
ap.add_argument('out')
ap.add_argument('--k', type=int, default=65536)
ap.add_argument('--ranking', default=str(Path(__file__).with_name('qwen3.5-token-ranking.txt')))
args = ap.parse_args()

tgt = gguf.GGUFReader(args.target)
head = next(t for t in tgt.tensors if t.name == 'output.weight')
n_vocab = int(head.shape[1])
f = tgt.fields['tokenizer.ggml.token_type']
types = [int(f.parts[i][0]) for i in f.data]
assert len(types) == n_vocab, 'target vocabulary and output projection disagree'

keep = {i for i, t in enumerate(types) if t != gguf.TokenType.NORMAL}
for line in open(args.ranking):
    if len(keep) >= args.k:
        break
    if line.strip():
        keep.add(int(line))
for i in range(n_vocab):
    if len(keep) >= args.k:
        break
    keep.add(i)
ids = np.array(sorted(keep), dtype=np.int64)

src = gguf.GGUFReader(args.drafter)
arch = src.fields['general.architecture'].contents()
if arch == 'eagle3':
    # the drafter's own head; the target only supplies the token types
    head = next(t for t in src.tensors if t.name == 'output.weight')
    assert int(head.shape[1]) == n_vocab, 'EAGLE drafter head must cover the whole target vocabulary'
    assert not any(t.name == 'd2t' for t in src.tensors), 'drafter already has a reduced vocabulary'
rows = np.array(head.data)[ids]            # quantized rows as raw bytes

if arch == 'eagle3':
    head_name, drop = 'output.weight', set()
elif arch == 'dflash':
    head_name, drop = 'output.weight', set()
elif f'{arch}.nextn_predict_layers' in src.fields:
    il = src.fields[f'{arch}.block_count'].contents() - 1
    head_name, drop = f'blk.{il}.nextn.shared_head_head.weight', {'output.weight'}
else:
    sys.exit(f'{args.drafter}: not a DFlash or MTP drafter (architecture {arch})')

w = gguf.GGUFWriter(args.out, arch)
for name, fld in src.fields.items():
    if name.startswith('GGUF.') or name == 'general.architecture':
        continue
    vt = fld.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(name, fld.contents(), vt, sub_type=fld.types[-1])
    else:
        w.add_key_value(name, fld.contents(), vt)
for t in src.tensors:
    if t.name in drop or t.name in (head_name, 'd2t'):
        continue
    data = np.array(t.data)
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        data = data.reshape([int(x) for x in reversed(t.shape)])
    # quantized tensors come back byte-shaped; with raw_dtype the writer derives the shape
    w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
w.add_tensor(head_name, rows, raw_dtype=head.tensor_type)
w.add_tensor('d2t', ids, raw_dtype=gguf.GGMLQuantizationType.I64)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'{args.out}: {head_name} = {len(ids)} rows of the {"drafter" if arch == "eagle3" else "target"} head ({head.tensor_type.name}, {rows.nbytes / 2**20:.1f} MiB), d2t [{len(ids)}]')
