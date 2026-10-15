"""Generalised P18 profile with a free exponent n.

    rho(R) = v^2 / (4 pi G R^2) * [1 - (1 - x) exp(-x)]^n ,   x = R / r_c

n = 2 is the P18 profile (a core); n < 2 gives an inner cusp rho ~ R^(n-2);
n = 1 gives no mass excess over the singular isothermal sphere (c = 0).

The script fits the 79 late-type SPARC galaxies of Parodi (2018)
  (1) with n = 2 fixed (v, r_c free),
  (2) with one common n for all galaxies (scan),
  (3) with v, r_c and n free for each galaxy,
and writes fit_exponent_results.csv.

Run from the code/ folder (it reads sparc_data/sparc79_fits.csv for the
galaxy list and sparc_data/rotmod/<name>_rotmod.dat for the curves).
Errors as in the paper: errV, 3% of V and 1 km/s added in quadrature.
Needs numpy and scipy; runs in about a minute.
"""
import csv, os
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.integrate import cumulative_trapezoid
from scipy.stats import spearmanr

SP = "sparc_data"
N_MIN, N_MAX = 0.5, 4.0                      # range allowed for n

# ---------------------------------------------------------------- model
_lx = np.linspace(np.log(1e-5), np.log(1e4), 4000)   # grid in ln x
_x = np.exp(_lx)
_u = 1.0 - (1.0 - _x) * np.exp(-_x)

def m_of_x(n):
    """Dimensionless enclosed mass m_n(x) = int_0^x u^n dx' on the grid."""
    f = _u ** n
    m = cumulative_trapezoid(f * _x, _lx, initial=0.0)   # dx = x dln x
    m += 2.0 ** n * _x[0] ** (n + 1) / (n + 1)           # tail below x_min (u ~ 2x)
    return m

def vc(R, v, rc, n):
    """Rotation curve v_c(R) = v sqrt(m_n(x)/x)."""
    m = m_of_x(n)
    return v * np.sqrt(np.interp(np.log(R / rc), _lx, m / _x))

def mass_excess(n):
    """c(n) = lim (m_n(x) - x) = int_0^inf (u^n - 1) dx: zero for n = 1, positive for n > 1."""
    from scipy.integrate import quad
    f = lambda x: (1.0 - (1.0 - x) * np.exp(-x)) ** n - 1.0
    return quad(f, 0, 50, limit=200)[0] + quad(f, 50, 400, limit=200)[0]

# ---------------------------------------------------------------- data
def load():
    gal = []
    for row in csv.DictReader(open(os.path.join(SP, "sparc79_fits.csv"))):
        d = np.loadtxt(os.path.join(SP, "rotmod", row["name"] + "_rotmod.dat"))
        R, V, eV0 = d.T[:3]
        e = np.sqrt(eV0**2 + (0.03 * V)**2 + 1.0)
        gal.append(dict(name=row["name"], R=R, V=V, e=e))
    return gal

# ---------------------------------------------------------------- fits
def fit_fixed_n(g, n):
    """v and r_c free, n fixed. Returns (chi2, v, r_c)."""
    res = lambda p: (vc(g["R"], np.exp(p[0]), np.exp(p[1]), n) - g["V"]) / g["e"]
    s = min((least_squares(res, [np.log(g["V"].max()), np.log(r0)])
             for r0 in (0.3, 1, 3, 10)), key=lambda s: s.cost)
    return 2 * s.cost, np.exp(s.x[0]), np.exp(s.x[1])

def fit_free_n(g):
    """v, r_c and n free. Returns (chi2, v, r_c, n, sigma_n)."""
    res = lambda p: (vc(g["R"], np.exp(p[0]), np.exp(p[1]), p[2]) - g["V"]) / g["e"]
    lo, hi = [-5, np.log(0.01), N_MIN], [10, np.log(100), N_MAX]
    best = None
    for r0 in (0.3, 1, 3):
        for n0 in (1.0, 2.0, 3.0):
            s = least_squares(res, [np.log(g["V"].max()), np.log(r0), n0], bounds=(lo, hi))
            if best is None or s.cost < best.cost:
                best = s
    chi2, dof = 2 * best.cost, max(len(g["R"]) - 3, 1)
    try:   # 1-sigma error of n, enlarged by sqrt(chi2_r) where chi2_r > 1
        cov = np.linalg.inv(best.jac.T @ best.jac)
        sn = np.sqrt(cov[2, 2]) * np.sqrt(max(1.0, chi2 / dof))
    except np.linalg.LinAlgError:
        sn = np.nan
    return chi2, np.exp(best.x[0]), np.exp(best.x[1]), best.x[2], sn

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    G = load()
    npts = sum(len(g["R"]) for g in G)
    print(f"{len(G)} galaxies, {npts} points")
    print("mass excess c(n):", ", ".join(f"n={n}: {mass_excess(n):+.3f}" for n in (1, 1.5, 1.75, 2)))

    # (1)+(2) one common n
    print("\nOne n for all galaxies (v, r_c free per galaxy):")
    tot = lambda n: sum(fit_fixed_n(g, n)[0] for g in G)
    for n in (1.0, 1.5, 1.75, 2.0, 2.5, 3.0):
        print(f"  n = {n:4.2f}: chi2 = {tot(n):7.1f}")
    r = minimize_scalar(tot, bounds=(1.2, 2.5), method="bounded", options={"xatol": 0.01})
    print(f"  best common n = {r.x:.2f}, chi2 = {r.fun:.1f}")

    # (3) n free per galaxy
    rows = []
    for g in G:
        c2, v2, rc2 = fit_fixed_n(g, 2.0)
        c3, v3, rc3, n3, sn3 = fit_free_n(g)
        rows.append(dict(name=g["name"], N=len(g["R"]), v_n2=v2, rc_n2=rc2, chi2_n2=c2,
                         v=v3, rc=rc3, n=n3, sigma_n=sn3, chi2=c3))
    with open("fit_exponent_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r_.items()})

    A = lambda k: np.array([r_[k] for r_ in rows])
    n, sn, v = A("n"), A("sigma_n"), A("v")
    print(f"\nn free per galaxy: chi2 = {A('chi2').sum():.1f} (n = 2 fixed: {A('chi2_n2').sum():.1f}),"
          f" {len(G)} extra parameters")
    ok = np.isfinite(sn) & (sn < 1.0) & (n > N_MIN + 0.01) & (n < N_MAX - 0.01)
    print(f"  n constrained (sigma_n < 1, not at a bound): {ok.sum()} galaxies;"
          f" at bounds: {(n <= N_MIN + 0.01).sum()} at {N_MIN}, {(n >= N_MAX - 0.01).sum()} at {N_MAX}")
    print(f"  median n = {np.median(n[ok]):.2f}  (16-84%: {np.percentile(n[ok], 16):.2f}-{np.percentile(n[ok], 84):.2f})")
    s = np.maximum(sn[ok], 0.1)          # floor on sigma_n
    nll = lambda p: 0.5 * np.sum((n[ok] - p[0])**2 / (s**2 + p[1]**2) + np.log(s**2 + p[1]**2))
    ml = minimize(nll, [1.7, 0.4], method="Nelder-Mead").x
    print(f"  maximum likelihood: mean n = {ml[0]:.2f}, intrinsic scatter = {abs(ml[1]):.2f}")
    z = (n[ok] - 2) / s
    print(f"  below n = 2 by > 2 sigma: {(z < -2).sum()}, above by > 2 sigma: {(z > 2).sum()}")
    rho, p = spearmanr(n[ok], v[ok])
    print(f"  Spearman n vs v: {rho:+.2f} (p = {p:.2f})")
    print("\nper-galaxy results written to fit_exponent_results.csv")
