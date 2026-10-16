"""Observed against model rotation velocity for n = 2 (P18) and for the exponent
n fitted per galaxy, with the residuals; companion to Fig. 3 of the letter.
Run from code/ ; writes fig_nfree.png and prints the rms values."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fit_exponent as F
plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight", "figure.dpi": 150})
G = F.load(); rows = []
for g in G:
    c3, v, rc, n, sn = F.fit_free_n(g); c2, v2, rc2 = F.fit_fixed_n(g, 2.0)
    rows.append(dict(n=n,
        A=dict(x=g["R"]/rc2, y=g["V"]/v2, m=F.vc(g["R"], 1.0, rc2, 2.0)),
        B=dict(x=g["R"]/rc, y=g["V"]/v, m=F.vc(g["R"], 1.0, rc, n))))
cm = plt.get_cmap("viridis"); lo, hi = 1.0, 3.0
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.32, wspace=0.08))
bins = np.logspace(-1.3, 1.4, 14)
for j, (key, title) in enumerate([("A", r"$n=2$ (P18) for all galaxies"), ("B", r"$n$ fitted for each galaxy")]):
    X = np.concatenate([r[key]["x"] for r in rows]); D = np.concatenate([r[key]["y"]-r[key]["m"] for r in rows])
    for r in rows:
        col = cm((np.clip(r["n"], lo, hi)-lo)/(hi-lo)); d = r[key]
        ax[0, j].plot(d["m"], d["y"], "o", c=col, ms=2.2, alpha=0.75, mec="none")
        ax[1, j].plot(d["x"], d["y"]-d["m"], "o", c=col, ms=2.0, alpha=0.7, mec="none")
    ax[0, j].plot([0, 1.3], [0, 1.3], "k-", lw=1.0)
    rms = np.sqrt(np.mean(D**2))
    ax[0, j].set_title(title, fontsize=9); ax[0, j].text(0.04, 1.2, f"rms = {rms:.3f}", fontsize=8.5)
    idx = np.digitize(X, bins)
    xb = [np.median(X[idx == i]) for i in range(1, len(bins)) if np.sum(idx == i) > 8]
    sb = [np.sqrt(np.mean(D[idx == i]**2)) for i in range(1, len(bins)) if np.sum(idx == i) > 8]
    ax[1, j].plot(xb, sb, "r-", lw=1.2, label="rms in bins"); ax[1, j].plot(xb, -np.array(sb), "r-", lw=1.2)
    ax[1, j].axhline(0, c="k", lw=1.0)
    ax[0, j].set_xlim(0, 1.3); ax[0, j].set_ylim(0, 1.3); ax[0, j].set_aspect("equal")
    ax[0, j].set_xlabel(r"model $v_n(x)$"); ax[1, j].set_xscale("log"); ax[1, j].set_xlim(0.03, 40)
    ax[1, j].set_ylim(-0.25, 0.25); ax[1, j].set_xlabel(r"$x=R/r_c$")
    print(f"{title}: rms {rms:.3f}")
ax[0, 0].set_ylabel(r"observed $V_{\rm obs}/v$"); ax[1, 0].set_ylabel("residual")
ax[0, 1].set_yticklabels([]); ax[1, 1].set_yticklabels([]); ax[1, 0].legend(frameon=False, loc="lower left", fontsize=7)
sm = plt.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(lo, hi))
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03); cb.set_label(r"fitted exponent $n$")
fig.savefig("fig_nfree.png", dpi=300); plt.close(fig)
