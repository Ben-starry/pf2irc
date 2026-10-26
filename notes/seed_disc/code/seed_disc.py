"""The disc that the seed's own angular momentum would make.

Seed of pf2irc: shells of mass M with one specific energy -v^2/2 and spin lambda.
Each shell rotates rigidly with the specific angular momentum it has at turnaround,
    j_shell(M) = 2 sqrt2 lambda G M / v ,
so a mass element at polar angle theta has  j_z = (3/2) j_shell sin^2(theta).
Detailed angular-momentum conservation: the gas element settles on the circular orbit
    R v_c(R) = f_j j_z
of the P18 rotation curve v_c = v sqrt(m(x)/x), x = R/r_c, with r_c = R_b / y.

Units: G = v = 1, total seed mass M_tot = 1, so R_b(1 + 1/4y) = 1.
Outputs: numbers printed, figure fig_seed_disc.png.
Run: python3 seed_disc.py   (numpy, scipy, matplotlib; ~20 s)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

CE = -0.91737
def y_of(lam, ell=0.0):                     # core ratio of pf2irc, Eq. (lambert)
    A = 3*lam**2 + 2*ell/3; B = 0.25 - CE - 0.75*lam**2
    return lambertw(4*A*np.exp(4*B)).real/(4*A)
def m(x):
    x = np.asarray(x, float)
    full = x*(1-np.exp(-x))**2 + 0.25*(1-(1+2*x+2*x*x)*np.exp(-2*x))
    return np.where(x < 1e-3, 4*x**3/3, full)      # series near the centre avoids round-off
def vc(R, rc): return np.sqrt(m(R/rc)*rc/R)          # v_c / v

def F_flat(S):          # disc mass fraction inside S = R/R_max for a flat curve (analytic)
    S = np.clip(S, 1e-12, 1); u = np.sqrt(1-S)
    return 1 - u + S*np.arctanh(np.minimum(u, 1-1e-15))
def Sigma_flat(S):      # surface density in units of M_d / R_max^2 (analytic)
    S = np.clip(S, 1e-12, 1-1e-15); return np.arctanh(np.sqrt(1-S))/(2*np.pi*S)

def disc(lam=0.035, y=None, ell=0.0, fj=1.0, N=2_000_000, seed=1):
    """Monte Carlo disc radii (units of R_b) for the P18 curve; returns R, R_b, r_c, R_max."""
    y = y_of(lam, ell) if y is None else y
    Rb = 1/(1+1/(4*y)); rc = Rb/y
    rng = np.random.default_rng(seed)
    M = rng.uniform(0, 1, N); mu = rng.uniform(-1, 1, N)
    jz = fj*1.5*2*np.sqrt(2)*lam*M*(1-mu**2)
    Rg = np.logspace(-7, 1, 6000); Jg = Rg*vc(Rg, rc)        # invert R v_c(R) = j_z
    R = np.interp(jz, Jg, Rg)
    Rmax = fj*3*np.sqrt(2)*lam                                # j_max / v
    return R/Rb, Rb, rc/Rb, Rmax/Rb, y

def exp_fit(R, lo=1.0, hi=4.0, Rd0=None):
    """Exponential scale length from the surface density between lo and hi times R_d (iterated)."""
    Rd = np.mean(R)/2 if Rd0 is None else Rd0
    for _ in range(20):
        e = np.linspace(lo*Rd, hi*Rd, 25); h, _ = np.histogram(R, e)
        S = h/(np.pi*(e[1:]**2-e[:-1]**2)); Rm = 0.5*(e[1:]+e[:-1]); ok = S > 0
        Rd = -1/np.polyfit(Rm[ok], np.log(S[ok]), 1)[0]
    return Rd

if __name__ == "__main__":
    lam = 0.035
    R, Rb, rc, Rmax, y = disc(lam)
    Rd = exp_fit(R); Rd_half = np.median(R)/1.678     # two definitions of R_d for a non-exponential disc
    jmean = np.sqrt(2)*lam                         # <j_z> = sqrt2 lam G M_tot / v (per M_tot)
    Rd_mmw = jmean/2*(1+1/(4*y))                   # exponential disc in a flat curve: j = 2 v R_d (units of R_b)
    print(f"lambda = {lam}, y = {y:.1f}, r_c = {rc:.4f} R_b, R_max = {Rmax:.4f} R_b")
    print(f"exponential fit (1-4 R_d): R_d = {Rd:.4f} R_b = {Rd/Rmax:.3f} R_max; R_d/r_c = {Rd/rc:.2f}")
    print(f"Mo-Mao-White (j_d = 2 v R_d): R_d = {Rd_mmw:.4f} R_b; R_d/r_c = {Rd_mmw/rc:.2f}; ratio {Rd/Rd_mmw:.3f}")
    print(f"half-mass radius: {np.median(R)/Rd:.3f} R_d (exponential 1.678); edge R_max = {Rmax/Rd:.2f} R_d")
    print(f"R_d from the half-mass radius (R_half/1.678) = {Rd_half:.4f} R_b; R_d/r_c = {Rd_half/rc:.2f}; ratio to MMW {Rd_half/Rd_mmw:.3f}")
    e = np.array([0, 0.1, 0.25, 0.5])*Rd
    inner = np.histogram(R, e)[0]/len(R)
    from math import exp
    expo = [ (1-(1+b)*np.exp(-b)) - (1-(1+a)*np.exp(-a)) for a, b in zip(e[:-1]/Rd, e[1:]/Rd)]
    print("mass in R/R_d = 0-0.1, 0.1-0.25, 0.25-0.5: seed", inner.round(4), " exponential", np.round(expo, 4))
    # flat-curve analytic check
    S = np.sort(R)/Rmax; Fmc = np.arange(1, len(S)+1)/len(S); idx = np.searchsorted(S, [0.1, 0.3, 0.6])
    print("analytic flat-curve F(S) at S=0.1,0.3,0.6:", F_flat(np.array([0.1, 0.3, 0.6])).round(4),
          " Monte Carlo (P18 curve):", Fmc[idx].round(4))
    # spin dependence of R_d/r_c
    lams = np.array([0.015, 0.02, 0.025, 0.035, 0.05, 0.07, 0.1])
    q_seed, q_half, q_mmw = [], [], []
    for l_ in lams:
        R_, Rb_, rc_, Rm_, y_ = disc(l_, N=600_000)
        q_seed.append(exp_fit(R_)/rc_); q_half.append(np.median(R_)/1.678/rc_)
        q_mmw.append(np.sqrt(2)*l_/2*(1+1/(4*y_))/rc_)
    q_seed, q_half, q_mmw = np.array(q_seed), np.array(q_half), np.array(q_mmw)
    print("R_d/r_c (seed fit 1-4 R_d / seed half-mass / MMW):",
          " ".join(f"{a}:{b:.2f}/{d:.2f}/{c:.2f}" for a, b, d, c in zip(lams, q_seed, q_half, q_mmw)))
    print("d ln(R_d/r_c)/d ln lambda (seed, around 0.035):",
          np.round(np.gradient(np.log(q_seed), np.log(lams))[3], 3))

    # ---------------------------------------------------------------- figure
    C1, C2, C3 = "#1f4e79", "#c0504d", "#4f8a3c"
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    # (a) surface density
    edges = np.concatenate([np.linspace(0, 0.2, 21), np.linspace(0.25, 6.5, 60)])*Rd
    h, _ = np.histogram(R, edges); Rm = 0.5*(edges[1:]+edges[:-1])
    Sig = h/len(R)/(np.pi*(edges[1:]**2-edges[:-1]**2))
    x = Rm/Rd; ok = Sig > 0
    ax[0].semilogy(x[ok], Sig[ok]*Rd**2, "o", ms=3, c=C1, label="seed, P18 curve (Monte Carlo)")
    xs = np.linspace(0.02, Rmax/Rd*0.999, 400)
    ax[0].semilogy(xs, Sigma_flat(xs*Rd/Rmax)*(Rd/Rmax)**2, c=C3, lw=1.3, label="seed, flat curve (analytic)")
    ax[0].semilogy(xs, np.exp(-xs)/(2*np.pi), c=C2, ls="--", lw=1.5, label=r"exponential, same $R_d$")
    ax[0].axvline(rc/Rd, c="0.6", lw=0.8, ls=":"); ax[0].text(rc/Rd*1.05, 2e-4, r"$r_c$", color="0.4")
    ax[0].axvline(Rmax/Rd, c="0.6", lw=0.8, ls=":"); ax[0].text(Rmax/Rd*0.83, 2e-4, r"edge", color="0.4")
    ax[0].set(xlabel=r"$R/R_d$", ylabel=r"$\Sigma\,R_d^2/M_d$", xlim=(0, 6.5), ylim=(1e-4, 3))
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right"); ax[0].set_title("(a) surface density", fontsize=10)
    # (b) cumulative mass
    xx = np.linspace(0, 6.5, 300); Fs = np.searchsorted(np.sort(R), xx*Rd)/len(R)
    ax[1].plot(xx, Fs, c=C1, lw=1.8, label="seed")
    ax[1].plot(xx, 1-(1+xx)*np.exp(-xx), c=C2, ls="--", lw=1.5, label="exponential")
    ax[1].axhline(0.5, c="0.8", lw=0.6)
    ax[1].set(xlabel=r"$R/R_d$", ylabel=r"$M_d(<R)/M_d$", xlim=(0, 6.5), ylim=(0, 1.02))
    ax[1].legend(frameon=False, fontsize=8, loc="lower right"); ax[1].set_title("(b) enclosed disc mass", fontsize=10)
    # (c) R_d / r_c against lambda
    ll = np.linspace(0.012, 0.11, 200); yy = np.array([y_of(l_) for l_ in ll])
    ax[2].loglog(lams, q_seed, "o", c=C1, ms=5, label=r"seed disc, exponential fit at 1-4 $R_d$")
    ax[2].loglog(lams, q_half, "s", c=C1, mfc="none", ms=5, label=r"seed disc, $R_{1/2}/1.678$")
    ax[2].loglog(ll, np.sqrt(2)*ll/2*(1+1/(4*yy))*yy/(1+1/(4*yy)), c=C2, ls="--", lw=1.4,
                 label=r"Mo et al. (1998): $\lambda y/\sqrt{2}$")
    ax[2].axhline(0.67, c=C3, lw=1.2); ax[2].text(0.0125, 0.69, r"SPARC dwarfs: median 0.67, $\pm1\sigma$ (0.30 dex)", color=C3, fontsize=7.5)
    ax[2].axhspan(0.67/10**0.30, 0.67*10**0.30, color=C3, alpha=0.12, lw=0)
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    ax[2].xaxis.set_major_locator(FixedLocator([0.015, 0.02, 0.035, 0.05, 0.07, 0.1])); ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax[2].yaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.7, 1, 1.5, 2, 3])); ax[2].yaxis.set_minor_locator(NullLocator())
    ax[2].yaxis.set_major_formatter(FormatStrFormatter("%g")); ax[2].set_ylim(0.3, 3.2)
    ax[2].set(xlabel=r"spin $\lambda$", ylabel=r"$R_d/r_c$  ($f_j=1$)")
    ax[2].legend(frameon=False, fontsize=7.5, loc="lower right"); ax[2].set_title(r"(c) disc and core ($\Lambda=0$)", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_seed_disc.png", dpi=250)
    print("figure written: fig_seed_disc.png")
