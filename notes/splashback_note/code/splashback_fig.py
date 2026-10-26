"""Figure and numbers for the companion note on splashback.
Reads run_splash_<N>.pkl written by splashback_run.py (1D collisionless shells of the pf2irc seed,
lambda = 0.035, units G = v = 1, outermost mass 1) and compares with the halted-galaxy model of pf2irc.
python3 splashback_fig.py run_splash_1500.pkl"""
import sys, pickle, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import brentq
out = pickle.load(open(sys.argv[1], "rb"))
XIB = 1/(1+1.5*np.pi)
# ---------------- model: mass per unit xi = r/vt, in units of v^3 t / G
def eta_of_xi(xi):                        # cycloid phase on the infalling/expanding branch for xi > xi_b
    f = lambda e: (1-np.cos(e))/(e-np.sin(e)) - xi
    return brentq(f, 1e-6, 1.5*np.pi)     # outside the front: eta < 3pi/2
def mu_out(xi):
    e = eta_of_xi(xi); D = 1-np.sin(e)*(e-np.sin(e))/(1-np.cos(e))**2
    P = (e-np.sin(e))**2/(4*np.pi*(1-np.cos(e))**3*D)
    return 4*np.pi*xi**2*P
y = 50.5
xs_in = np.logspace(-3, np.log10(XIB), 200); u = 1-(1-y*xs_in/XIB)*np.exp(-y*xs_in/XIB)
mu_in = u**2                              # P18: dM/dR = v^2 u^2 / G
xs_out = np.logspace(np.log10(XIB*1.0005), 0, 200); mu_o = np.array([mu_out(x) for x in xs_out])
# ---------------- simulation: stacked profile and splashback
e = np.geomspace(1e-3, 1.0, 51); acc = np.zeros(50); rsp, forb = [], []
for t, s in out.items():
    r, vr, dm, M = s["r"], s["vr"], s["dm"], s["M"]
    acc += np.histogram(r/t, e, weights=dm/t)[0]
    tta = np.pi*M
    orb = t > 2*tta                           # passed (approximately) the first pericentre: orbiting matter
    o = np.argsort(r[orb]); cm = np.cumsum(dm[orb][o]); cm /= cm[-1]
    rsp.append(r[orb][o][np.searchsorted(cm, 0.99)]/t)   # outer edge of the orbiting matter (99% of its mass)
    forb.append(dm[orb & (r/t > XIB)].sum()/dm[orb].sum())
mu = acc/len(out)/np.diff(e); xm = np.sqrt(e[1:]*e[:-1])
rsp, forb = np.array(rsp), np.array(forb)
xta = 2/np.pi                               # current turnaround radius / vt
sl = np.gradient(np.log(np.maximum(mu, 1e-9)), np.log(xm)); k = (xm > XIB) & (xm < 0.5)
xsteep = xm[k][np.argmin(sl[k])]
print(f"steepest drop of dM/dxi outside the front at xi = {xsteep:.3f} (slope of rho {sl[k].min()-2:.1f}) = {xsteep/xta:.3f} r_ta,now")
print(f"splashback (radius enclosing 99% of the orbiting mass): r_sp = {rsp.mean():.3f} +- {rsp.std():.3f} vt "
      f"= {rsp.mean()/xta:.3f} r_ta,now = {rsp.mean()/XIB:.2f} R_b")
print(f"fraction of orbiting mass (t > 2 t_ta) outside the front xi_b: {forb.mean():.3f} +- {forb.std():.3f}")
# ---------------- figure
tL = max(out); s = out[tL]; r, vr, M = s["r"]/tL, s["vr"], s["M"]; tta = np.pi*M
cat = np.where(tL < tta, 0, 2)
C = ["#8c8c8c", "#c0504d", "#1f4e79"]; L = ["before turnaround", "", "after turnaround"]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for k in (0, 2):
    m_ = cat == k; ax[0].scatter(r[m_], vr[m_], s=3, c=C[k], label=L[k], alpha=0.8, lw=0)
for x_, ls_, lb in [(XIB, "-", r"front $R_b$ (model)"), (rsp.mean(), "--", r"splashback $r_{\rm sp}$"), (xta, ":", r"turnaround $r_{\rm ta}$")]:
    ax[0].axvline(x_, c="k", ls=ls_, lw=1, label=lb)
ax[0].set_xscale("log"); ax[0].set(xlim=(3e-3, 1.0), ylim=(-6, 4), xlabel=r"$r/vt$", ylabel=r"$v_r/v$")
ax[0].legend(frameon=False, fontsize=8, loc="lower left", markerscale=3); ax[0].set_title("(a) phase space, 1D collisionless run", fontsize=10)
ax[1].loglog(xm[mu > 0], mu[mu > 0], "o-", ms=3, c=C[2], lw=1, label="1D collisionless run (stacked)")
ax[1].loglog(xs_in, mu_in, c=C[1], lw=1.8, label="model: settled P18 galaxy ($y=50.5$)")
ax[1].loglog(xs_out, mu_o, c=C[1], lw=1.8, ls="--", label="model: shells on first infall")
for x_, ls_ in [(XIB, "-"), (rsp.mean(), "--"), (xta, ":")]:
    ax[1].axvline(x_, c="k", ls=ls_, lw=1)
ax[1].set(xlim=(1e-3, 1.0), ylim=(0.05, 5), xlabel=r"$\xi=r/vt$", ylabel=r"$\mathrm{d}M/\mathrm{d}\xi\;/\;(v^3t/G)$")
ax[1].legend(frameon=False, fontsize=8, loc="upper left"); ax[1].set_title("(b) mass per unit radius", fontsize=10)
fig.tight_layout(); fig.savefig("fig_splashback.png", dpi=250); print("figure: fig_splashback.png")
