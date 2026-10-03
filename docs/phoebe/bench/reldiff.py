import sys, array, math
a=array.array('f'); b=array.array('f'); a.frombytes(open(sys.argv[1],'rb').read()); b.frombytes(open(sys.argv[2],'rb').read())
mx=max(abs(x-y) for x,y in zip(a,b)); ref=math.sqrt(sum(x*x for x in a)/len(a)); nd=sum(1 for x,y in zip(a,b) if x!=y)
print(f"max abs {mx:.3e}, rms of output {ref:.3e}, max/rms {mx/ref:.2e}, elements differing {nd}/{len(a)}")
