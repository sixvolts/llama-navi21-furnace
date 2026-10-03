import csv,sys,collections,statistics,re
def cls(n):
    if 'mul_mat_vec_q' in n: return 'MMVQ'
    if 'mul_mat_q' in n: return 'MMQ'
    if 'flash_attn' in n: return 'flash_attn'
    if 'quantize' in n: return 'quantize'
    if 'gated_delta' in n or 'ssm' in n or 'conv' in n.lower(): return 'delta-net'
    if 'norm' in n: return 'norm'
    if 'rope' in n: return 'rope'
    if 'cpy' in n or 'dup' in n or 'concat' in n: return 'copy'
    if 'get_rows' in n: return 'get_rows'
    if 'soft_max' in n or 'softmax' in n: return 'softmax'
    if 'bin_bcast' in n or 'unary' in n or 'glu' in n or 'add' in n or 'mul' in n or 'scale' in n or 'silu' in n: return 'elementwise'
    if 'argsort' in n or 'top_k' in n or 'argmax' in n: return 'topk'
    return 'other'
f=sys.argv[1]
rows=[]
for r in csv.DictReader(open(f)):
    rows.append((int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Kernel_Name'],int(r['VGPR_Count']),int(r['LDS_Block_Size']),int(r['Grid_Size_X']),int(r['Workgroup_Size_X']),int(r['Workgroup_Size_Y'])))
rows.sort()
heads=[r for r in rows if ('mul_mat_vec_q<(ggml_type)14' in r[2] or 'mul_mat_q<(ggml_type)14' in r[2])]
gmax=max(r[5] for r in heads); marks=[r for r in heads if r[5]==gmax]
steps=[]
for i in range(1,len(marks)):
    a=marks[i-1][1]; b=marks[i][1]
    steps.append([r for r in rows if a<r[0]<=b])
steps=[s for s in steps if len(s)>100]
busies=[sum(e-s for s,e,*_ in st)/1e6 for st in steps]
print(f"{f.split('/')[-1]}: {len(marks)} head launches, {len(steps)} steps; busy median {statistics.median(busies):.2f} ms min {min(busies):.2f} max {max(busies):.2f}")
# group steps by the max ncols_dst of MMVQ present (verify batch size)
def bsz(st):
    m=0
    for r in st:
        mm=re.search(r'mul_mat_vec_q<\(ggml_type\)\d+, (\d+),',r[2])
        if mm: m=max(m,int(mm.group(1)))
    return m
groups=collections.defaultdict(list)
for st in steps: groups[bsz(st)].append(st)
for b,sts in sorted(groups.items()):
    bus=[sum(e-s for s,e,*_ in st)/1e6 for st in sts]
    med=sorted(sts,key=lambda st: sum(e-s for s,e,*_ in st))[len(sts)//2]
    busy=sum(e-s for s,e,*_ in med)/1e6; span=(med[-1][1]-med[0][0])/1e6
    by=collections.defaultdict(float); cnt=collections.Counter()
    for s,e,n,*_ in med: by[cls(n)]+=(e-s)/1e6; cnt[cls(n)]+=1
    print(f"--- max ncols {b}: {len(sts)} steps, busy median {statistics.median(bus):.2f}; median step {len(med)} launches busy {busy:.2f} span {span:.2f}")
    for k,v in sorted(by.items(),key=lambda x:-x[1]): print(f"     {k:12s} {v:7.3f} ms {cnt[k]:5d} launches  {1000*v/cnt[k]:6.1f} us/launch")
# kernel meta of interest
meta={}
for s,e,n,v,l,g,wx,wy in rows:
    if ('mul_mat_vec_q' in n and (', 8,' in n or ', 4,' in n or ', 1,' in n)) or 'flash_attn' in n:
        k=n[:80]; 
        if k not in meta: meta[k]=[v,l,wx,wy,0,0.0]
        meta[k][4]+=1; meta[k][5]+=(e-s)/1e6
for k,(v,l,wx,wy,c,t) in sorted(meta.items(), key=lambda x:-x[1][5])[:14]:
    print(f"   vgpr {v:3d} lds {l:6d} wg {wx}x{wy} n {c:5d} t {t:8.2f} ms  {k}")
