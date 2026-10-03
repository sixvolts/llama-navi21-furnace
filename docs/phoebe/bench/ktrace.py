import csv, sys, re, collections, statistics
def cls(name):
    n=name
    if 'mul_mat_vec_q' in n: return 'MMVQ (weights x 1..8 cols)'
    if 'mul_mat_q' in n: return 'MMQ (quantized GEMM)'
    if 'flash_attn' in n: return 'flash attention'
    if 'quantize_q8_1' in n or 'quantize_mmq' in n: return 'q8_1 activation quant'
    if 'gated_delta' in n or 'delta_net' in n or 'ssm' in n or 'conv' in n.lower(): return 'delta-net (linear attn)'
    if 'rms_norm' in n or 'norm' in n: return 'norms (+fused)'
    if 'rope' in n: return 'rope'
    if 'cpy' in n or 'dup' in n or 'concat' in n: return 'copies'
    if 'softmax' in n or 'soft_max' in n: return 'softmax'
    if 'bin_bcast' in n or 'unary' in n or 'silu' in n or 'swiglu' in n or 'glu' in n or 'add' in n or 'mul' in n: return 'elementwise'
    if 'get_rows' in n: return 'get_rows'
    if 'argsort' in n or 'top_k' in n or 'argmax' in n: return 'argsort/topk'
    return 'other'
def load(f):
    rows=[]
    for r in csv.DictReader(open(f)):
        s=int(r['Start_Timestamp']); e=int(r['End_Timestamp'])
        rows.append((s,e,r['Kernel_Name'],int(r['VGPR_Count']),int(r['LDS_Block_Size']),int(r['Scratch_Size'])))
    rows.sort(); return rows
def segments(rows, gap_ns):
    segs=[]; cur=[rows[0]]
    for r in rows[1:]:
        if r[0]-cur[-1][1] > gap_ns: segs.append(cur); cur=[r]
        else: cur.append(r)
    segs.append(cur); return segs
def report(seg, label, per=1.0):
    busy=sum(e-s for s,e,*_ in seg)/1e6; span=(seg[-1][1]-seg[0][0])/1e6
    by=collections.defaultdict(float); cnt=collections.Counter()
    for s,e,n,*_ in seg: by[cls(n)]+=(e-s)/1e6; cnt[cls(n)]+=1
    print(f"--- {label}: {len(seg)} dispatches, GPU busy {busy/per:.2f} ms, span {span/per:.2f} ms (gaps {100*(1-busy/span):.0f}% of span)")
    for k,v in sorted(by.items(), key=lambda x:-x[1]):
        print(f"      {k:28s} {v/per:8.3f} ms  {100*v/busy:5.1f}%  ({cnt[k]/per:.0f} launches)")
    return busy/per, by
if __name__=='__main__':
    f=sys.argv[1]; mode=sys.argv[2]
    rows=load(f)
    if mode=='steps':   # segment by gaps; classify steps by presence of MMQ (prompt) vs not (decode)
        gap=float(sys.argv[3])*1e6 if len(sys.argv)>3 else 1.5e6
        segs=segments(rows, gap)
        dec=[s for s in segs if not any('mul_mat_q' in r[2] and 'vec' not in r[2] for r in s) and len(s)>200]
        pro=[s for s in segs if any('mul_mat_q' in r[2] and 'vec' not in r[2] for r in s)]
        print(f"{f.split('/')[-1]}: {len(segs)} segments; {len(dec)} decode-like, {len(pro)} prompt-like")
        if dec:
            busies=[sum(e-s for s,e,*_ in sg)/1e6 for sg in dec]
            med=sorted(dec, key=lambda sg: sum(e-s for s,e,*_ in sg))[len(dec)//2]
            print(f"decode steps: busy median {statistics.median(busies):.2f} ms, min {min(busies):.2f}, max {max(busies):.2f}")
            report(med, 'median decode step')
        if pro: report(pro[-1], 'last prompt-like segment')
    elif mode=='all':
        report(rows, f.split('/')[-1]+' (whole trace)')
    # top kernels by time
    by=collections.defaultdict(float); meta={}
    for s,e,n,v,l,sc in rows: by[n]+=(e-s)/1e6; meta[n]=(v,l,sc)
    tot=sum(by.values())
    print("top kernels:")
    for n,t in sorted(by.items(), key=lambda x:-x[1])[:int(sys.argv[4]) if len(sys.argv)>4 else 8]:
        v,l,sc=meta[n]; print(f"   {100*t/tot:5.1f}%  vgpr {v:3d} lds {l:6d} scr {sc:4d}  {n[:110]}")

def perstep(f, label):
    rows=[]
    for r in csv.DictReader(open(f)):
        rows.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'], int(r['Grid_Size_X'])))
    rows.sort()
    # vocab-head matmul: the Q6_K (type 14) MMVQ/MMQ launch with the largest grid; one per decode step
    heads=[r for r in rows if ('mul_mat_vec_q<(ggml_type)14' in r[2] or 'mul_mat_q<(ggml_type)14' in r[2])]
    gmax=max(r[3] for r in heads); marks=[r for r in heads if r[3]==gmax]
    n=len(marks)
    # per-step = everything between consecutive head launches (from just after one head to the end of the next)
    steps=[]
    for i in range(1,n):
        a=marks[i-1][1]; b=marks[i][1]
        steps.append([r for r in rows if a < r[0] <= b])
    steps=[s for s in steps if len(s)>100]
    busies=[sum(e-s for s,e,*_ in st)/1e6 for st in steps]
    spans=[(st[-1][1]-st[0][0])/1e6 for st in steps]
    print(f"=== {label}: {n} vocab-head launches -> {len(steps)} steps; busy median {statistics.median(busies):.2f} ms (min {min(busies):.2f} max {max(busies):.2f}); span median {statistics.median(spans):.2f} ms")
    med=sorted(steps, key=lambda st: sum(e-s for s,e,*_ in st))[len(steps)//2]
    by=collections.defaultdict(float); cnt=collections.Counter()
    for s,e,nm,_ in med: by[cls(nm)]+=(e-s)/1e6; cnt[cls(nm)]+=1
    busy=sum(by.values())
    for k,v in sorted(by.items(), key=lambda x:-x[1]): print(f"      {k:28s} {v:8.3f} ms  {100*v/busy:5.1f}%  ({cnt[k]} launches)")
    return steps
if __name__=='__main__' and sys.argv[2]=='perstep':
    perstep(sys.argv[1], sys.argv[1].split('/')[-1])
