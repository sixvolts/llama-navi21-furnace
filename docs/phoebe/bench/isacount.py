import re,sys,collections
def census(path, want):
    txt=open(path).read().split('\n')
    # find function ranges
    funcs=[];cur=None
    for i,l in enumerate(txt):
        m=re.match(r'\s*\.globl\s+(\S+)',l)
        if m: cur=[m.group(1),i,None]
        if '-- End function' in l and cur: cur[2]=i; funcs.append(cur); cur=None
    for name,a,b in funcs:
        if want not in name or 'stream_k_fixup' in name: continue
        body=txt[a:b]
        # split into basic blocks by labels
        blocks=[];lab=None;ins=[]
        for l in body:
            m=re.match(r'^(\.LBB\S+):',l)
            if m:
                if lab: blocks.append((lab,ins))
                lab=m.group(1);ins=[]
            elif lab and l.startswith('\t') and not l.startswith('\t.') and not l.startswith('\t;'):
                ins.append(l.strip().split()[0])
        if lab: blocks.append((lab,ins))
        best=max(blocks,key=lambda b: sum(1 for x in b[1] if x.startswith('v_dot')))
        c=collections.Counter(best[1])
        tot=len(best[1])
        vgpr=[l for l in body if 'NumVgprs' in l or 'ScratchSize' in l or '; Occupancy' in l or 'NumSgprs' in l]
        print(f"== {name[:60]}  {vgpr}")
        print(f"   hottest block {best[0]}: {tot} instrs")
        groups={'v_dot':0,'ds_read':0,'ds_write':0,'global_load':0,'s_load':0,'s_waitcnt':0,'v_mul_lo':0,'v_mad_u64':0,'v_cvt':0,'v_fma/v_mac/v_mul_f32/v_add_f32/v_sub_f32/v_pk':0,'v_lsh/v_and/v_or/v_bfe/v_perm/v_alignbit':0,'v_mov/v_readlane/v_writelane':0,'s_*':0,'other_v':0}
        for k,v in c.items():
            if k.startswith('v_dot'): groups['v_dot']+=v
            elif k.startswith('ds_read') or k.startswith('ds_load'): groups['ds_read']+=v
            elif k.startswith('ds_write') or k.startswith('ds_store'): groups['ds_write']+=v
            elif k.startswith('global_load') or k.startswith('buffer_load') or k.startswith('flat_load'): groups['global_load']+=v
            elif k.startswith('s_load') or k.startswith('s_buffer_load'): groups['s_load']+=v
            elif k.startswith('s_waitcnt'): groups['s_waitcnt']+=v
            elif k.startswith('v_mul_lo'): groups['v_mul_lo']+=v
            elif k.startswith('v_mad_u64') or k.startswith('v_mad_i64'): groups['v_mad_u64']+=v
            elif k.startswith('v_cvt'): groups['v_cvt']+=v
            elif re.match(r'v_(fma|mac|mul_f32|add_f32|sub_f32|pk_|fmac|mul_f16|add_f16|max_f32|min_f32)',k): groups['v_fma/v_mac/v_mul_f32/v_add_f32/v_sub_f32/v_pk']+=v
            elif re.match(r'v_(lsh|and|or|bfe|perm|alignbit|xor|bfi|not)',k): groups['v_lsh/v_and/v_or/v_bfe/v_perm/v_alignbit']+=v
            elif re.match(r'v_(mov|readlane|writelane|readfirstlane)',k): groups['v_mov/v_readlane/v_writelane']+=v
            elif k.startswith('s_'): groups['s_*']+=v
            elif k.startswith('v_'): groups['other_v']+=v
        for g,v in groups.items(): print(f"   {g:50s} {v}")
        print("   top mnemonics:", c.most_common(18))
for p,w in [('mmq_q4_k.s','ELi64E'),('mmq_q8_0.s','ELi64E'),('mmq_q6_k.s','ELi64E')]:
    census(p,w)
