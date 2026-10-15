import pickle, numpy as np, sys
from scipy.optimize import least_squares
def prof(name,nb=36,lo=1e-3,hi=1.0):
    out=pickle.load(open(f'run_{name}.pkl','rb')); e=np.geomspace(lo,hi,nb+1); acc=np.zeros(nb)
    for t,s in out.items(): acc+=np.histogram(s['r']/s['t'],bins=e,weights=s['dm']/s['t'])[0]
    return np.sqrt(e[1:]*e[:-1]), acc/len(out)/np.diff(e)
shapes={'P18':lambda x:(1-(1-x)*np.exp(-x))**2,
        'pseudo-iso':lambda x:x*x/(1+x*x),
        'Burkert':lambda x:x*x/((1+x)*(1+x*x))*x}      # dM/dx * (1/x)?? -> mu ~ x^2 rho
# Burkert: rho ~ 1/((1+x)(1+x^2)); mu ~ x^2 rho
shapes['Burkert']=lambda x:x*x/((1+x)*(1+x*x))
for n in sys.argv[1:]:
    x,mu=prof(n); ref=np.median(mu[(x>0.02)&(x<0.06)])
    fr=x[(x>0.06)&(mu<0.4*ref)][0]; m=(x<0.8*fr)&(mu>0)
    print(f"{n}: front xi_f={fr:.3f}")
    for k,S in shapes.items():
        r=lambda p: np.log(np.exp(p[0])*S(x[m]/np.exp(p[1])))-np.log(mu[m])
        s=least_squares(r,[0,np.log(0.01)],bounds=([-5,np.log(1e-5)],[5,np.log(1)]))
        A,xc=np.exp(s.x); print(f"   {k:10s} xi_c={xc:.4f} (xi_f/xi_c={fr/xc:.0f})  A={A:.2f}  rms={np.sqrt(np.mean(s.fun**2)):.3f}")
    k=(x>0.001)&(x<0.004)&(mu>0)
    if k.sum()>2: print('   inner rho slope (xi 0.001-0.004):',round(np.polyfit(np.log(x[k]),np.log(mu[k]),1)[0]-2,2))
    print('   mu:',' '.join(f"{a:.3f}:{b:.2f}" for a,b in zip(x[::2],mu[::2])))
