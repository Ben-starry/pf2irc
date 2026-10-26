"""Writes tab_sparc.tex (Appendix A): per-galaxy P18 fits, free-n fits and end of growth."""
import csv, os, numpy as np
from scipy.optimize import least_squares
import recipe
SP = "sparc_data"; G = 4.30091e-6
ex = np.exp
def m(x): return x*(1-ex(-x))**2 - 0.5*x*(1+x)*ex(-2*x) + 0.25*(1-ex(-2*x))
vn = lambda x: np.sqrt(m(x)/x)
B1 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B1.csv"))))
B2 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B2.csv"))))
FN = {r["name"]: r for r in csv.DictReader(open("fit_exponent_results.csv"))}
rows = []
for b1, b2 in zip(B1, B2):
    name = b2["name"].replace("UGC0786", "UGC07866")
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"))
    R, V, e0 = d.T[:3]; e = np.sqrt(e0**2+(0.03*V)**2+1.0)
    res = lambda p: (np.exp(p[0])*vn(R/np.exp(p[1]))-V)/e
    s = min((least_squares(res, [np.log(V.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    dof = max(len(R)-2, 1); chi = 2*s.cost/dof
    cov = np.linalg.pinv(s.jac.T@s.jac)*max(chi, 1.0)
    v, rc = np.exp(s.x); ev, erc = v*np.sqrt(cov[0, 0]), rc*np.sqrt(cov[1, 1])
    fn = FN[name]; n, sn = float(fn["n"]), float(fn["sigma_n"])
    nstr = f"${n:.2f}\\pm{sn:.2f}$" if np.isfinite(sn) and sn < 1 and 0.51 < n < 3.99 else "--"
    z = recipe.z_end(v, rc)
    rows.append((name.replace("_", "\\_"), len(R), f"${v:.1f}\\pm{ev:.1f}$", f"${rc:.2f}\\pm{erc:.2f}$",
                 f"{v/rc:.0f}", f"{chi:.2f}", f"{float(b1['Rd']):.2f}", nstr,
                 f"{z:.2f}" if z is not None else "--"))
half = (len(rows)+1)//2
L = []
for i in range(half):
    a = rows[i]; b = rows[i+half] if i+half < len(rows) else ("",)*9
    L.append(" & ".join(map(str, a))+" & "+" & ".join(map(str, b))+" \\\\")
open("../tab_sparc.tex", "w").write("\n".join(L)+"\n")
print(len(rows), "rows; galaxies without solution for z_end:", sum(r[-1] == "--" for r in rows))
