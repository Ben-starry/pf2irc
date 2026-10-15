"""1D spherical collisionless shells with angular momentum.
Units G = v = 1. Seed: every shell E = -1/2, common bang time -> cycloid R = M(1-cos eta), t = M(eta - sin eta);
turnaround R_ta = 2M at t_ta = pi M. At turnaround a shell receives specific angular momentum j (centrifugal barrier).
Afterwards: r'' = -M_enc(<r)/r^2 + j^2/r^3 (Gauss's law, shells may cross).  Leapfrog, global adaptive dt, small softening."""
import numpy as np, sys, json, time
from scipy.optimize import brentq

def run(kdf=0.0, win=0.2, N=3000, Mmin=2e-3, Mmax=1.0, jmode='seed', lam=0.035, f=None, spread=0.0, Tend=None, soft=2e-4, eta_dt=0.02, seed=1, snaps=None):
    rng=np.random.default_rng(seed)
    M=np.geomspace(Mmin,Mmax,N)                    # mass labels (enclosed seed mass)
    dm=np.diff(np.concatenate([[0.0],M]))          # shell masses (innermost carries everything inside Mmin)
    Rta=2*M; tta=np.pi*M
    if jmode=='seed': j=np.sqrt(2)*lam*Rta         # j = sqrt2 lam v R_ta  (A2), v=1
    else: j=f*1.0*Rta/2                            # fraction f of circular at R_ta/2
    if spread>0: j=j*np.exp(spread*rng.standard_normal(N)-spread**2/2)
    t=tta[0]*1.0001
    # initial cycloid positions
    def eta_of(Mi,t):
        g=lambda e: Mi*(e-np.sin(e))-t
        return brentq(g,1e-9,2*np.pi-1e-9) if t<2*np.pi*Mi else 2*np.pi-1e-9
    eta=np.array([eta_of(Mi,t) for Mi in M])
    r=M*(1-np.cos(eta)); vr=np.sin(eta)/(1-np.cos(eta))   # dR/dt = v sin(eta)/(1-cos eta)
    L=np.where(t>=tta,j,0.0)
    if Tend is None: Tend=np.pi*Mmax*1.0
    snaps=sorted(snaps or [Tend]); out={}; si=0
    tnow=[t]
    def acc(r,L):
        o=np.argsort(r); cm=np.cumsum(dm[o]); Menc=np.empty_like(cm); Menc[o]=cm-0.5*dm[o]
        rs=np.sqrt(r*r+(soft*tnow[0])**2)
        return -Menc*r/rs**3 + L*L/np.maximum(r,1e-12)**3, Menc
    a,Menc=acc(r,L); nstep=0; t0=time.time()
    while t<Tend:
        act=t>=tta-1e-12
        tnow[0]=t; tdyn=np.sqrt((r[act]**2+(soft*t)**2)**1.5/np.maximum(Menc[act],1e-12))
        dt=eta_dt*min(tdyn.min(), (r[act]/np.maximum(np.abs(vr[act]),1e-9)).min())
        dt=min(dt, 1e-3*t, snaps[si]-t+1e-15)
        vr+=0.5*dt*a; r+=dt*vr
        neg=r<0; r[neg]=-r[neg]; vr[neg]=-vr[neg]
        t+=dt; tnow[0]=t
        newL=(t>=tta)&(L==0)&(j>0); L[newL]=j[newL]
        a,Menc=acc(r,L); vr+=0.5*dt*a; nstep+=1
        if kdf>0:   # dynamical-friction coupling: radial energy of every halted shell -> tangential energy of the matter inside it
            act=t>=tta
            ia=np.where(act)[0]; o=ia[np.argsort(r[ia])]
            Om=np.sqrt(np.maximum(Menc[o],1e-12)/np.maximum(r[o]**2+(soft*t)**2,1e-24)**1.5)
            f=np.exp(-kdf*Om*dt); dE=0.5*dm[o]*vr[o]**2*(1-f*f); vr[o]*=f
            lr=np.log(np.maximum(r[o],1e-30)); n=len(o)
            lo=np.searchsorted(lr,lr-win); hi=np.searchsorted(lr,lr+win,side='right')   # local window in ln r
            cm=np.concatenate([[0.0],np.cumsum(dm[o])]); Mw=cm[hi]-cm[lo]
            s_=dE/Mw; diff=np.zeros(n+1); np.add.at(diff,lo,s_); np.add.at(diff,hi,-s_)
            de=np.cumsum(diff)[:n]                  # specific energy deposited locally
            L[o]=np.sqrt(L[o]**2+2*r[o]**2*de)
        if nstep%200000==0: print('prog',nstep,round(t,4),round(time.time()-t0),flush=True)
        if t>=snaps[si]-1e-14:
            out[float(snaps[si])]=dict(r=r.copy(),vr=vr.copy(),M=M,dm=dm,L=L.copy(),t=t)
            si+=1
            if si>=len(snaps): break
    return out,nstep,time.time()-t0
