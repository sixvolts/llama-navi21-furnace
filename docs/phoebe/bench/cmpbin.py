import sys, array
a=array.array('f'); b=array.array('f')
a.frombytes(open(sys.argv[1],'rb').read()); b.frombytes(open(sys.argv[2],'rb').read())
nd=sum(1 for x,y in zip(a,b) if x!=y); mx=max(abs(x-y) for x,y in zip(a,b))
print(f"{sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: {'IDENTICAL' if nd==0 else f'{nd}/{len(a)} differ, max abs {mx:.3e}'}")
