"""Multi-orbit shells: each seed mass shell is split into K orbits. Dynamical-friction coupling:
radial energy of halted orbits is removed at rate kdf*Omega and deposited locally (window in ln r) as random,
isotropic heating (2/3 tangential, 1/3 radial, random weights), so that a spread of angular momenta develops."""
import numpy as np, time
from scipy.optimize import brentq
def run(age=None, kdf=0.1, win=0.2, K=6, N=1500, Mmin=1e-4, Mmax=1.0, lam=0.035, soft=2e-4, eta_dt=0.03, seed=1, snaps=None, Tend=np.pi):
    rng=np.random.default_rng(seed)
    Ms=np.geomspace(Mmin,Mmax,N); dms=np.diff(np.concatenate([[0.0],Ms]))
    M=np.repeat(Ms,K); dm=np.repeat(dms/K,K); n=len(M)
    tta=np.pi*M; j=np.sqrt(2)*lam*2*M
    t=tta[0]*1.0001
    def eta_of(Mi):
        return brentq(lambda e: Mi*(e-np.sin(e))-t,1e-9,2*np.pi-1e-9) if t<2*np.pi*Mi else 2*np.pi-1e-9
    et=np.array([eta_of(m) for m in Ms]); et=np.repeat(et,K)
    r=M*(1-np.cos(et)); vr=np.sin(et)/(1-np.cos(et)); L=np.where(t>=tta,j,0.0)
    snaps=sorted(snaps); out={}; si=0; nstep=0; t0=time.time()
    def acc(r,L,t):
        o=np.argsort(r); cm=np.cumsum(dm[o]); Me=np.empty_like(cm); Me[o]=cm-0.5*dm[o]
        rs=np.sqrt(r*r+(soft*t)**2); return -Me*r/rs**3+L*L/np.maximum(r,1e-12)**3, Me
    a,Me=acc(r,L,t)
    while t<Tend:
        act=t>=tta-1e-12
        tdyn=np.sqrt((r[act]**2+(soft*t)**2)**1.5/np.maximum(Me[act],1e-12))
        dt=eta_dt*min(tdyn.min(),(r[act]/np.maximum(np.abs(vr[act]),1e-9)).min())
        dt=min(dt,1e-3*t,snaps[si]-t+1e-15)
        vr+=0.5*dt*a; r+=dt*vr; neg=r<0; r[neg]=-r[neg]; vr[neg]=-vr[neg]; t+=dt
        newL=(t>=tta)&(L==0); L[newL]=j[newL]
        a,Me=acc(r,L,t); vr+=0.5*dt*a; nstep+=1
        if kdf>0:
            ia=np.where(t>=tta)[0]; o=ia[np.argsort(r[ia])]; m_=len(o)
            Om=np.sqrt(np.maximum(Me[o],1e-12)/(r[o]**2+(soft*t)**2)**1.5)
            kk=kdf if age is None else kdf*((t-tta[o])<age*tta[o])   # friction only on recently accreted matter (infalling lumps)
            f=np.exp(-kk*Om*dt); dE=0.5*dm[o]*vr[o]**2*(1-f*f); vr[o]*=f
            lr=np.log(np.maximum(r[o],1e-30))
            lo=np.searchsorted(lr,lr-win); hi=np.searchsorted(lr,lr+win,side='right')
            wgt=rng.exponential(1.0,m_)*dm[o]
            cw=np.concatenate([[0.0],np.cumsum(wgt)]); W=cw[hi]-cw[lo]
            s_=dE/W; d=np.zeros(m_+1); np.add.at(d,lo,s_); np.add.at(d,hi,-s_)
            de=np.cumsum(d)[:m_]*wgt/dm[o]          # specific energy per orbit (random weights)
            dv=np.sqrt(2*np.maximum(de,0)); c=rng.uniform(-1,1,m_); ph=rng.uniform(0,2*np.pi,m_)
            dvt=dv*np.sqrt(1-c*c); Lt=L[o]+r[o]*dvt*np.cos(ph)       # random vector kick (isotropic)
            L[o]=np.sqrt(Lt**2+(r[o]*dvt*np.sin(ph))**2); vr[o]+=dv*c
        if nstep%200000==0: print('prog',nstep,round(t,4),round(time.time()-t0),flush=True)
        if t>=snaps[si]-1e-14:
            out[float(snaps[si])]=dict(r=r.copy(),vr=vr.copy(),M=M,dm=dm,L=L.copy(),t=t); si+=1
            if si>=len(snaps): break
    return out,nstep,time.time()-t0
