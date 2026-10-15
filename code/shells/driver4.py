import sys, numpy as np, pickle, shells3
name=sys.argv[1]; kw=eval(sys.argv[2]); T=np.pi
out,n,tt=shells3.run(snaps=list(np.linspace(0.7*T,T,16)),**kw)
pickle.dump(out,open(f'run_{name}.pkl','wb')); print(name,'steps',n,'time %.0fs'%tt,flush=True)
