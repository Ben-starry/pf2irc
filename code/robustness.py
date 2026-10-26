"""Robustness tests for the article (Sect. 5.6 energy budget, Sect. 6 statistics).

  python3 robustness.py [nmock]      (run from the code/ folder; default nmock = 100)

(1) The 79 P18 fits are redone exactly as in make_figures.py (errors: errV, 3% of V and
    1 km/s in quadrature; covariance inflated where chi2_r > 1).
(2) Goodness of fit: chi2_nu with and without the error floor.
(3) The r_c - v slope b under several estimators: ML with intrinsic scatter in log r_c
    (the paper's estimator; fit errors and their covariance, with and without distance and
    inclination systematics), ML with intrinsic scatter perpendicular to the line (symmetric),
    OLS(y|x), inverse OLS(x|y), bisector, orthogonal; subsamples that reach the flat part;
    v replaced by V_flat of P18 and of SPARC.
(4) Mock test: curves generated with a true b = 2 (and b = 1), at the observed radii and with
    the observed errors, refitted with the same procedure and passed through the same estimator.
(5) Energy budget: cooling and feedback per galaxy and the v-dependence of the net energy.
(6) The spin test: predicted and observed dispersion of log(R_d/r_c).
Writes robustness_results.txt.
"""
import csv, os, sys
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import lambertw

rng = np.random.default_rng(7)
SP = "sparc_data"
G = 4.30091e-6; GYR = 0.977792; OM, OL, H0 = 0.3, 0.7, 0.07; LAM3 = OL*H0**2
CE = -0.917374; LAMSPIN = 0.035
T0_ = 1+1.5*np.pi; CL_ = (105*np.pi/8+83/6)/T0_
NMOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 100
out = []
def report(s): print(s, flush=True); out.append(s)

ex = np.exp
def m(x):  return x*(1-ex(-x))**2 - 0.5*x*(1+x)*ex(-2*x) + 0.25*(1-ex(-2*x))
def vn(x): return np.sqrt(m(x)/x)
def t_of_z(z): return 2/(3*np.sqrt(LAM3))*np.arcsinh(np.sqrt(OL/OM)*(1/(1+z))**1.5)*GYR
def y_at(t, lam=LAMSPIN, eps=0.0):            # core ratio with Lambda for growth ending at t (Gyr)
    tk = t/GYR; l0 = LAM3*(tk/T0_)**2; Rb1 = tk/(T0_*(1+CL_*l0))   # R_b / v
    ell = LAM3*Rb1**2
    A = 3*lam**2+2/3*ell+eps; B = 0.25-CE-0.75*lam**2-0.25*eps
    if A <= 0:
        if -A > np.exp(-1-4*B)/4: return np.nan
        return (lambertw(4*A*np.exp(4*B), k=0).real)/(4*A)
    return lambertw(4*A*np.exp(4*B)).real/(4*A)

# ------------------------------------------------------------------ (1) data and fits
B2 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B2.csv"))))
B3 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B3.csv"))))
B1 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B1.csv"))))
def fit(R, V, e):
    res = lambda p: (np.exp(p[0])*vn(R/np.exp(p[1]))-V)/e
    s = min((least_squares(res, [np.log(V.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    dof = max(len(R)-2, 1); chi = 2*s.cost/dof
    cov = np.linalg.pinv(s.jac.T@s.jac)*max(chi, 1.0)
    cov[0, 0] = max(cov[0, 0], 1e-8); cov[1, 1] = max(cov[1, 1], 1e-8)
    v, rc = np.exp(s.x)
    return v, rc, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]), 2*s.cost
GAL = []
for b2, b3 in zip(B2, B3):
    name = b2["name"].replace("UGC0786", "UGC07866")
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"))
    R, Vo, eV0 = d.T[:3]
    e = np.sqrt(eV0**2+(0.03*Vo)**2+1.0)
    v, rc, slv, slr, cvr, chi2 = fit(R, Vo, e)
    chi2_0 = np.sum(((v*vn(R/rc)-Vo)/eV0)**2)
    vf = float(b3["vflat"]) if b3["vflat"] else np.nan
    GAL.append(dict(name=name, R=R, V=Vo, e=e, e0=eV0, v=v, rc=rc, slv=slv, slr=slr, cvr=cvr,
                    chi2=chi2, chi2_0=chi2_0, vf=vf))
A_ = lambda k: np.array([g[k] for g in GAL])
lv, lrc = np.log10(A_("v")), np.log10(A_("rc"))
L10 = np.log(10)
sx_fit = np.minimum(A_("slv")/L10, 0.5); sy_fit = np.minimum(A_("slr")/L10, 0.5)
cxy_fit = A_("cvr")*sx_fit*sy_fit
NP = sum(len(g["R"]) for g in GAL)

# systematics as in make_figures.py
BT = {}
for l_ in open(os.path.join(SP, "BTFR_Lelli2016a.mrt")).read().split("\n"):
    p_ = l_.split()
    if len(p_) == 11:
        try: BT[p_[0]] = (float(p_[1]), float(p_[2]), int(p_[3]), float(p_[6]), float(p_[7]))
        except ValueError: pass
Dg = np.array([float(b["D"]) for b in B1])
has = np.array([g["name"] in BT for g in GAL])
fD = np.array([BT[g["name"]][1]/BT[g["name"]][0] if g["name"] in BT else np.nan for g in GAL])
eVi = np.array([BT[g["name"]][4] if g["name"] in BT else np.nan for g in GAL])
for lo_, hi_, f_ in [(0, 5, 0.05), (5, 17, 0.30), (17, 19.5, 0.14), (19.5, 1e3, 0.20)]:
    fD[~has & (Dg >= lo_) & (Dg < hi_)] = f_
eVi[~has] = np.median(eVi[has])
sD = np.log10(1+fD)
sx_sys = np.minimum(np.sqrt((A_("slv")/L10)**2+eVi**2), 0.5)
sy_sys = np.minimum(np.sqrt((A_("slr")/L10)**2+sD**2), 0.5)

# ------------------------------------------------------------------ (2) goodness of fit
chi2 = A_("chi2").sum(); chi2_0 = A_("chi2_0").sum(); dof = NP-2*len(GAL)
report(f"[fits] {len(GAL)} galaxies, {NP} points; chi2 = {chi2:.1f} for {dof} dof (chi2_nu = {chi2/dof:.2f}) "
       f"with the error floor; with SPARC errors only chi2 = {chi2_0:.1f} (chi2_nu = {chi2_0/dof:.2f})")
report(f"  common n: Delta chi2 = 19 -> {19/(chi2/dof):.0f} if the errors are rescaled to chi2_nu = 1; "
       f"free n: F = {((389-168)/79)/(168/(NP-3*79)):.1f} for (79, {NP-3*79}) dof")

# ------------------------------------------------------------------ (3) estimators
def ml_y(x, y, sx, sy, cxy):           # intrinsic scatter in y (the paper's estimator)
    def nl(p):
        a, b, ls = p; s2 = sy**2+b**2*sx**2-2*b*cxy+np.exp(2*ls)
        return 0.5*np.sum((y-a-b*x)**2/s2+np.log(s2))
    return minimize(nl, [np.median(y), 1.0, np.log(0.2)], method="Nelder-Mead",
                    options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000)).x
def ml_perp(x, y, sx, sy, cxy):        # intrinsic scatter perpendicular to the line (symmetric)
    def nl(p):
        a, b, ls = p; s2 = sy**2+b**2*sx**2-2*b*cxy+np.exp(2*ls)*(1+b**2)
        return 0.5*np.sum((y-a-b*x)**2/s2+np.log(s2))
    return minimize(nl, [np.median(y), 1.0, np.log(0.2)], method="Nelder-Mead",
                    options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000)).x
def ols(x, y): return np.polyfit(x, y, 1)[0]
def inv(x, y): return 1/np.polyfit(y, x, 1)[0]
def bis(x, y):
    b1, b2 = ols(x, y), inv(x, y); return (b1*b2-1+np.sqrt((1+b1**2)*(1+b2**2)))/(b1+b2)
def orth(x, y):
    c = np.cov(x, y); l, v = np.linalg.eigh(c); return v[1, 1]/v[0, 1]
def boot(fn, *arrs, nb=400):
    n = len(arrs[0]); out_ = []
    for _ in range(nb):
        i = rng.integers(0, n, n); out_.append(fn(*[a[i] for a in arrs]))
    return np.std(out_, axis=0)
xc = lv-np.median(lv)
res = {}
p = ml_y(xc, lrc, sx_sys, sy_sys, cxy_fit); e = boot(lambda *a: ml_y(*a)[1], xc, lrc, sx_sys, sy_sys, cxy_fit)
res["ML, scatter in r_c, with systematics (paper)"] = (p[1], e, np.exp(p[2]))
p = ml_y(xc, lrc, sx_fit, sy_fit, cxy_fit); e = boot(lambda *a: ml_y(*a)[1], xc, lrc, sx_fit, sy_fit, cxy_fit)
res["ML, scatter in r_c, fit errors only"] = (p[1], e, np.exp(p[2]))
p = ml_perp(xc, lrc, sx_sys, sy_sys, cxy_fit); e = boot(lambda *a: ml_perp(*a)[1], xc, lrc, sx_sys, sy_sys, cxy_fit)
res["ML, scatter perpendicular (symmetric)"] = (p[1], e, np.exp(p[2]))
for lbl, fn in [("OLS(r_c|v)", ols), ("inverse OLS(v|r_c)", inv), ("bisector", bis), ("orthogonal", orth)]:
    res[lbl] = (fn(xc, lrc), boot(fn, xc, lrc), np.nan)
r_ = np.corrcoef(lv, lrc)[0, 1]
report(f"[slope r_c ~ v^b] N = {len(lv)}, correlation r = {r_:.2f}, spread of log v {lv.std():.2f} dex, of log r_c {lrc.std():.2f} dex")
for k, (b, e, s) in res.items():
    report(f"  {k:48s} b = {b:5.2f} +- {e:4.2f}   (b=2 at {(2-b)/e:4.1f} sigma, b=1 at {(b-1)/e:+5.1f} sigma)"
           + (f"; intrinsic scatter {s:.2f} dex" if np.isfinite(s) else ""))
# subsamples and other speeds
xR = np.array([g["R"].max()/g["rc"] for g in GAL])
report(f"  R_max/r_c: median {np.median(xR):.1f}, 16-84% {np.percentile(xR,16):.1f}-{np.percentile(xR,84):.1f}; "
       f"N(>=3) = {(xR>=3).sum()}, N(>=5) = {(xR>=5).sum()}")
for lbl, sel in [("R_max/r_c >= 3", xR >= 3), ("R_max/r_c >= 5", xR >= 5)]:
    x_ = lv[sel]-np.median(lv[sel])
    p = ml_y(x_, lrc[sel], sx_sys[sel], sy_sys[sel], cxy_fit[sel])
    e = boot(lambda *a: ml_y(*a)[1], x_, lrc[sel], sx_sys[sel], sy_sys[sel], cxy_fit[sel])
    report(f"  {lbl:20s} N = {sel.sum():2d}: b = {p[1]:.2f} +- {e:.2f} (b=2 at {(2-p[1])/e:.1f} sigma); bisector {bis(x_, lrc[sel]):.2f}")
vf = A_("vf"); ok = np.isfinite(vf)
x_ = np.log10(vf[ok]); x_ -= np.median(x_)
p = ml_y(x_, lrc[ok], np.full(ok.sum(), 0.03), sy_sys[ok], np.zeros(ok.sum()))
e = boot(lambda *a: ml_y(*a)[1], x_, lrc[ok], np.full(ok.sum(), 0.03), sy_sys[ok], np.zeros(ok.sum()))
report(f"  v -> V_flat of P18 (N = {ok.sum()}): b = {p[1]:.2f} +- {e:.2f} (b=2 at {(2-p[1])/e:.1f} sigma)")
okS = np.array([g["name"] in BT for g in GAL])
lVf = np.array([BT[g["name"]][3] if g["name"] in BT else np.nan for g in GAL])
ok2 = okS & (lVf > 0)
x_ = lVf[ok2]-np.median(lVf[ok2]); sxv = np.array([BT[g["name"]][4] for g in GAL if g["name"] in BT])[lVf[okS] > 0]
p = ml_y(x_, lrc[ok2], sxv, sy_sys[ok2], np.zeros(ok2.sum()))
e = boot(lambda *a: ml_y(*a)[1], x_, lrc[ok2], sxv, sy_sys[ok2], np.zeros(ok2.sum()))
report(f"  v -> V_flat of SPARC (Lelli et al. 2016b; N = {ok2.sum()}): b = {p[1]:.2f} +- {e:.2f} (b=2 at {(2-p[1])/e:.1f} sigma)")

# ------------------------------------------------------------------ (4) mock test
def mock(btrue, sint):
    a0 = np.median(lrc); lrt = a0+btrue*xc+rng.normal(0, sint, len(xc))
    vs, rcs, sxs, sys_, cs = [], [], [], [], []
    for g, lr in zip(GAL, lrt):
        v, rc = g["v"], 10**lr
        Vm = v*vn(g["R"]/rc)+rng.normal(0, 1, len(g["R"]))*g["e"]*np.sqrt(chi2/dof)
        v_, rc_, slv, slr, cvr, _ = fit(g["R"], Vm, g["e"])
        vs.append(v_); rcs.append(rc_); sxs.append(slv/L10); sys_.append(slr/L10); cs.append(cvr)
    x = np.log10(vs); y = np.log10(rcs); sx = np.minimum(sxs, 0.5); sy = np.minimum(sys_, 0.5)
    p = ml_y(x-np.median(x), y, sx, sy, np.array(cs)*sx*sy)
    return p[1], bis(x-np.median(x), y), np.exp(p[2])
if __name__ == "__main__":
    # intrinsic scatter for the null: the observed spread of log r_c minus what b=2 explains (floor 0.05 dex)
    s2 = np.sqrt(max(lrc.var()-4*lv.var(), 0.05**2)); s1 = np.sqrt(max(lrc.var()-lv.var(), 0.05**2))
    report(f"[mock] {NMOCK} samples each; noise = errors x sqrt(chi2_nu) = x{np.sqrt(chi2/dof):.2f}; "
           f"intrinsic scatter of the input: {s2:.2f} dex (b=2), {s1:.2f} dex (b=1)")
    pm = res["ML, scatter in r_c, fit errors only"][0]
    for bt, si in [(2.0, s2), (1.0, s1), (2.0, 0.29)]:
        M = np.array([mock(bt, si) for _ in range(NMOCK)])
        report(f"  true b = {bt:.0f}, scatter {si:.2f}: recovered ML b = {np.median(M[:,0]):.2f} "
               f"(16-84%: {np.percentile(M[:,0],16):.2f}-{np.percentile(M[:,0],84):.2f}; rms {M[:,0].std():.2f}), "
               f"bisector {np.median(M[:,1]):.2f}; fraction with ML b <= {pm:.2f} (data): {(M[:,0] <= pm).mean():.3f}")

    # ------------------------------------------------------------------ (5) energy budget
    z8 = 0.8; y8 = y_at(t_of_z(z8))
    Ms = np.array([0.5*10**(-0.4*(float(b["M36"])-3.24)) for b in B1])
    Mg = []
    for g in GAL:
        d = np.loadtxt(os.path.join(SP, "rotmod", g["name"]+"_rotmod.dat"), ndmin=2)
        Mg.append(max(d[-1, 3]*abs(d[-1, 3]), 0)*d[-1, 0]/G)
    Mb = Ms+np.array(Mg); Rd = np.array([float(b["Rd"]) for b in B1])
    v_, rc_ = A_("v"), A_("rc"); Mt = v_**2*y8*rc_/G*(1+1/(4*y8))
    kap = np.log(y8*rc_/(np.e*Rd))-(1-np.euler_gamma); fb = Mb/Mt
    ed = -2*kap*fb; ef = 2*(710/v_)**2*Ms/Mt
    pc = lambda a: np.percentile(a, [16, 50, 84])
    report(f"[energy] growth ending at z = {z8}: y = {y8:.1f}; spin 3lam^2 = {3*LAMSPIN**2:.4f}; "
           f"Lambda 2/3 ell = {(4*(3*LAMSPIN**2)*0+ (lambda t: 2/3*LAM3*(t/GYR/(T0_*(1+CL_*LAM3*(t/GYR/T0_)**2)))**2)(t_of_z(z8))):.4f}")
    report(f"  cooling eps_diss 16/50/84 = {pc(ed).round(3)}; feedback per unit f_c = {pc(ef).round(3)}; f_c for balance = {pc(-ed/ef).round(3)}")
    report(f"  slopes vs log v: log|eps_diss| {np.polyfit(lv, np.log10(-ed), 1)[0]:+.2f}, log eps_fb {np.polyfit(lv, np.log10(ef), 1)[0]:+.2f}")
    fc_med = np.median(-ed/ef)
    for fc in [fc_med, 0.8*fc_med, 1.2*fc_med]:
        net = ed+fc*ef
        ratio = np.array([(4*y8+1)/(4*y_at(t_of_z(z8), eps=e_)+1) if np.isfinite(y_at(t_of_z(z8), eps=e_)) else np.nan for e_ in net])
        okr = np.isfinite(ratio)
        sl = np.polyfit(lv[okr], np.log10(ratio[okr]), 1)[0]
        report(f"  f_c = {fc:.3f}: net eps 16/50/84 = {pc(net).round(4)}; no P18 solution for {(~okr).sum()} galaxies; "
               f"core factor median {np.nanmedian(ratio):.2f}; its slope d log r_c/d log v = {sl:+.2f}")

    # ------------------------------------------------------------------ (6) spin test
    q = Rd/rc_; lq = np.log10(q)
    erq = A_("slr")/L10
    lam_s = LAMSPIN*np.exp(0.5*rng.normal(size=200000))
    zs = rng.choice([0.5, 0.65, 0.8, 1.0, 1.2], size=200000)
    ys = np.array([y_at(t_of_z(z_), l_) for z_, l_ in zip(zs[:20000], lam_s[:20000])])
    lqm = np.log10(lam_s[:20000]*ys/np.sqrt(2))
    pred = np.sqrt(lqm.var()+np.median(erq)**2+0.1**2)
    obs = lq.std(); nb = [np.std(lq[rng.integers(0, len(lq), len(lq))]) for _ in range(2000)]
    lam_only = 0.5/np.log(10)
    pred0 = np.sqrt(lam_only**2+np.median(erq)**2+0.1**2)
    report(f"[spin] observed sd of log(R_d/r_c) = {obs:.3f} +- {np.std(nb):.3f} dex (median R_d/r_c {np.median(q):.2f})")
    report(f"  spin-dependent core (model; sigma_ln lambda = 0.5, epochs z = 0.5-1.2): intrinsic {lqm.std():.3f} dex, "
           f"with errors (r_c {np.median(erq):.3f}, R_d 0.10) {pred:.3f} dex -> data wider by {(obs-pred)/np.std(nb):.1f} sigma")
    report(f"  spin-independent core (R_d/r_c ~ lambda): {lam_only:.3f} dex, with errors {pred0:.3f} dex -> data wider by {(obs-pred0)/np.std(nb):.1f} sigma")

    # ------------------------------------------------------------------ (7) empirical mass excess from a common n
    sys.argv = sys.argv[:1]
    import fit_exponent as FE
    GF = FE.load()
    ns = np.array([1.60, 1.68, 1.72, 1.76, 1.80, 1.84, 1.92, 2.00])
    c2 = np.array([sum(FE.fit_fixed_n(g, n_)[0] for g in GF) for n_ in ns])
    pp = np.polyfit(ns, c2, 2); nbest = -pp[1]/(2*pp[0]); cmin = np.polyval(pp, nbest)
    dn_raw = np.sqrt(1/pp[0]); dn_res = np.sqrt((chi2/dof)/pp[0])
    report(f"[mass excess] common n: chi2(n) = " + ", ".join(f"{a:.2f}:{b:.1f}" for a, b in zip(ns, c2)))
    report(f"  best n = {nbest:.3f} (chi2 {cmin:.1f}); 1 sigma: +-{dn_raw:.3f} (raw errors), +-{dn_res:.3f} (rescaled to chi2_nu=1)")
    report(f"  c(n): n=2: {FE.mass_excess(2.0):.3f}; best {FE.mass_excess(nbest):.3f}; range {FE.mass_excess(nbest-dn_raw):.3f}-{FE.mass_excess(nbest+dn_raw):.3f}")
    open("robustness_results.txt", "w").write("\n".join(out)+"\n")
