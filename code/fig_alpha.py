"""Distribution of the central density slope alpha = n - 2 of the generalised
IRC for the SPARC dwarfs with constrained n (Appendix B, Fig. B.2).
Run from code/; uses fit_exponent.py."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from fit_exponent import load, fit_free_n, N_MIN, N_MAX

G = load()
fits = np.array([fit_free_n(g)[3:5] for g in G])
n, sn = fits.T
ok = np.isfinite(sn) & (sn < 1) & (n > N_MIN + .01) & (n < N_MAX - .01)
a, s = n[ok] - 2, np.maximum(sn[ok], 0.1)
nll = lambda p: 0.5 * np.sum((a - p[0])**2 / (s**2 + p[1]**2) + np.log(s**2 + p[1]**2))
mu, sig = minimize(nll, [-0.35, 0.4], method="Nelder-Mead").x; sig = abs(sig)
print(f"N = {ok.sum()}, median alpha = {np.median(a):.2f}, ML mean = {mu:.2f}, intrinsic scatter = {sig:.2f}")
print(f"alpha <= -0.9: {(a <= -0.9).sum()}; intrinsic fraction below -0.9: {norm.cdf((-0.9 - mu) / sig):.2f}")

fig, ax = plt.subplots(figsize=(3.6, 2.7))
o = np.argsort(a); k = np.arange(1, len(a) + 1)
ax.axhspan(-0.32 - 0.24, -0.32 + 0.24, color="0.88", lw=0, zorder=0)
ax.axhspan(mu - sig, mu + sig, facecolor="none", edgecolor="#3b6ea5", hatch="////", lw=0, alpha=0.35, zorder=0)
ax.axhline(0, color="k", ls=":", lw=1); ax.axhline(-1, color="k", ls="--", lw=1)
ax.errorbar(k, a[o], yerr=s[o], fmt="o", ms=3, color="#3b6ea5", ecolor="0.6", elinewidth=0.7, zorder=2)
kw = dict(fontsize=7, va="bottom")
ax.text(1, 0.03, "core (P18)", **kw)
ax.text(len(a), -0.97, "NFW", ha="right", **kw)
ax.text(len(a), -0.32, "LITTLE THINGS", ha="right", va="center", fontsize=6.5, color="0.35")
ax.set_xlabel("galaxies, ordered by slope"); ax.set_ylabel(r"central slope $\alpha=n-2$")
ax.set_xlim(0, len(a) + 1); ax.set_ylim(-1.7, 1.9)
fig.savefig("fig_alpha.png", dpi=250, bbox_inches="tight")
