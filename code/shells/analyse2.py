import pickle, numpy as np, sys
from scipy.optimize import least_squares
mp=lambda x:(1-(1-x)*np.exp(-x))**2
def prof(name,nb=40):
    out=pickle.load(open(f'run_{name}.pkl','rb'))
    e=np.geomspace(1e-3,1.0,nb+1); acc=np.zeros(nb)
    for t,s in out.items():
        acc+=np.histogram(s['r']/s['t'],bins=e,weights=s['dm']/s['t'])[0]
    mu=acc/len(out)/np.diff(e); x=np.sqrt(e[1:]*e[:-1]); return x,mu
for n in sys.argv[1:]:
    x,mu=prof(n)
    # front: first xi beyond 0.05 where mu falls below 0.4 of its median in [0.01,0.05]
    ref=np.median(mu[(x>0.01)&(x<0.05)])
    fr=x[(x>0.05)&(mu<0.4*ref)][0]
    m=(x<0.8*fr)&(mu>0)
    # slope of rho = dlnmu/dlnxi - 2 over [0.002,0.01] and [0.01,0.05]
    def sl(a,b):
        k=(x>a)&(x<b)&(mu>0); return np.polyfit(np.log(x[k]),np.log(mu[k]),1)[0]-2
    r=lambda p: np.log(np.exp(p[0])*mp(x[m]/np.exp(p[1])))-np.log(mu[m])
    s=least_squares(r,[0,np.log(0.005)],bounds=([-3,np.log(1e-5)],[3,np.log(0.1)]))
    A,xc=np.exp(s.x)
    print(f"{n:8s} front xi_f={fr:.3f}  rho slope [0.002,0.01]: {sl(0.002,0.01):+.2f}  [0.01,0.05]: {sl(0.01,0.05):+.2f}  P18 fit xi_c={xc:.4f} (y={fr/xc:.0f}) rms={np.sqrt(np.mean(s.fun**2)):.2f}")
