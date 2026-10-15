import sys, numpy as np, pickle, shells2 as shells
name=sys.argv[1]; kw=dict(N=3000,Mmin=1e-4,Mmax=1.0,soft=2e-4,eta_dt=0.03); kw.update(eval(sys.argv[2]))
T=np.pi; snaps=list(np.linspace(0.7*T,T,16))
out,n,tt=shells.run(Tend=T,snaps=snaps,**kw)
pickle.dump(out,open(f'run_{name}.pkl','wb')); print(name,'steps',n,'time %.0fs'%tt, flush=True)
