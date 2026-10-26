"""Self-similarity check of the 1D shell runs (Appendix H).

    python3 selfsim_check.py seed035 seed07 ...      (reads run_<name>.pkl from driver2.py)

For each run: the mass profile mu(xi) = dM/dxi / (v^3 t/G) of each of the 16 snapshots, xi = r/vt.
If the run is self-similar, the snapshots coincide. Reported: the front xi_f of each snapshot, the inner
density slope of each snapshot (xi = 0.002-0.01), the rms scatter of log mu among snapshots in three ranges
of xi, and the number of shells inside xi = 1e-3 at the last snapshot.
"""
import pickle, sys, numpy as np

def snap_profiles(name, nb=30, lo=1e-3, hi=1.0):
    out = pickle.load(open(f"run_{name}.pkl", "rb"))
    e = np.geomspace(lo, hi, nb + 1); x = np.sqrt(e[1:] * e[:-1])
    ts, mus, last = [], [], None
    for t, s in sorted(out.items()):
        h = np.histogram(s["r"] / s["t"], bins=e, weights=s["dm"] / s["t"])[0] / np.diff(e)
        ts.append(t); mus.append(h); last = s
    return x, np.array(ts), np.array(mus), last

def front(x, mu):
    ref = np.median(mu[(x > 0.01) & (x < 0.05)])
    k = np.where((x > 0.05) & (mu < 0.4 * ref))[0]
    return x[k[0]] if len(k) else np.nan

def slope(x, mu, a=0.002, b=0.01):
    k = (x > a) & (x < b) & (mu > 0)
    return np.polyfit(np.log(x[k]), np.log(mu[k]), 1)[0] - 2

for name in sys.argv[1:]:
    x, ts, mus, last = snap_profiles(name)
    print(f"== {name}: {len(ts)} snapshots, t/t_ta(outer) = {ts[0]/np.pi:.2f}-{ts[-1]/np.pi:.2f}")
    fr = [front(x, m) for m in mus]; sl = [slope(x, m) for m in mus]
    print("  front xi_f per snapshot: " + " ".join(f"{f:.3f}" for f in fr))
    print("  inner rho slope (xi 0.002-0.01) per snapshot: " + " ".join(f"{s:.2f}" for s in sl))
    lm = np.log10(np.where(mus > 0, mus, np.nan))
    for a, b in [(1e-3, 1e-2), (1e-2, 0.1), (0.1, 0.3)]:
        k = (x > a) & (x < b)
        rms = np.nanmean(np.nanstd(lm[:, k], axis=0))
        drift = np.nanmean(lm[-4:, k]) - np.nanmean(lm[:4, k])
        print(f"  xi {a:g}-{b:g}: rms of log mu among snapshots {rms:.3f} dex; "
              f"last 4 minus first 4 snapshots {drift:+.3f} dex")
    xi = last["r"] / last["t"]
    print(f"  shells inside xi=1e-3 at the last snapshot: {np.sum(xi < 1e-3)}; "
          f"inside xi_b/y=3.5e-3: {np.sum(xi < 3.5e-3)}; mass inside 1e-3: {last['dm'][xi < 1e-3].sum():.4f}")
