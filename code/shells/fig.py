import sys; sys.argv=['x']
exec(open('analyse2.py').read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.4,4.4))
for n,lab,c in [('f10','circular at $R_{ta}/2$ ($f=1$)','C0'),('f06','$f=0.6$','C1'),('f03','$f=0.3$','C2'),('seed07','seed, $\\lambda=0.07$','C3'),('seed035','seed, $\\lambda=0.035$','C4')]:
    x,mu=prof(n,30); ax.loglog(x,mu,'-o',ms=3,color=c,label=lab)
xs=np.geomspace(1e-3,0.174,200)
for y,ls in [(50.5,'--'),(100,':')]:
    xc=0.174/y; ax.loglog(xs,0.75*mp(xs/xc),'k'+ls,lw=1.2,label=f'P18 core, $y={y:g}$ (model)')
ax.axvline(0.174,color='grey',lw=0.8); ax.text(0.18,0.02,'$\\xi_b$',color='grey')
ax.set_xlabel('$\\xi=r/(vt)$'); ax.set_ylabel('$\\mu=dM/d\\xi\\,/\\,(v^3t/G)$  (SIS: const.)')
ax.set_ylim(0.01,5); ax.legend(fontsize=7.5,loc='lower left'); ax.set_title('1D collisionless shells: late-time self-similar profile',fontsize=9)
fig.tight_layout(); fig.savefig('shells_profile.png',dpi=160)
