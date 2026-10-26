# Units G=M=v=1: radial energy E=-1/2, cycloid R=1-cos eta, t=eta-sin eta, R_ta=2, t_ta=pi; front: halt at R=1 on the way back.
import numpy as np
from scipy.integrate import solve_ivp
lam=0.035
jta=np.sqrt(2)*lam*2.0          # j = sqrt2 lam v R_ta
def run(mode, jfac=1.0):
    # mode 'ta': j switched on at turnaround (paper); 'ttt': j grows ~t until t_ta (tidal torque theory); 'const': j from start (energy with j fixed)
    t0=1e-6; eta0=(6*t0)**(1/3); R0=1-np.cos(eta0); V0=np.sin(eta0)/(1-np.cos(eta0))
    j=lambda t: jfac*jta*(0 if (mode=='ta' and t<np.pi) else (min(t/np.pi,1) if mode=='ttt' else 1))
    f=lambda t,y:[y[1], -1/y[0]**2 + j(t)**2/y[0]**3]
    ev=lambda t,y: y[0]-1.0; ev.direction=-1; ev.terminal=True
    s=solve_ivp(f,[t0,20],[R0,V0],events=ev,rtol=1e-11,atol=1e-13,max_step=0.01,dense_output=True)
    th=s.t_events[0][0]; Rmax=s.y[0].max()
    return th, Rmax
t0,_=run('ta',0.0)
print("no spin: t_halt",t0,"(1+3pi/2)=",1+1.5*np.pi)
for m in ['ta','ttt']:
    th,Rm=run(m); print(m, "t_halt rel change %.2e"%(th/t0-1), "R_ta change %.2e"%(Rm/2-1), "xi_b change %.2e"%(t0/th-1))
# pericentre after turnaround with constant j_ta in the potential of a point mass M (first infall, no galaxy inside).
# NOTE: superseded for the paper (Sect. 8.1) by checks_sec6to8.py, part (4), which uses the potential of the settled P18 galaxy.
# radial energy -1/2 at R_ta=2 plus rotation: energy E = -1/2 + j^2/(2*4)
for fac,lbl in [(1,'seed lambda=0.035'),(2,'lambda=0.07'),(1/(2*np.sqrt(2)*lam),'circular at R_ta/2 (f=1)')]:
    j=jta*fac; E=-1/(2.0)+j**2/(2*4)   # value at turnaround, Rdot=0: E = -GM/R_ta + j^2/2R_ta^2
    # solve E = -1/r + j^2/(2 r^2) for small root
    a=E; r=np.roots([2*a, 2, -j**2]); rp=min(r[r>0])
    print(f"{lbl}: j={j:.3f} GM/v, pericentre r_p = {rp:.4f} GM/v^2 = R_b/{1/rp:.0f}")
