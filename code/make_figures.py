#!/usr/bin/env python3
"""
make_figures.py -- all numbers and figures of
"How a galaxy gets its rotation curve" (undergraduate version).

A cold, slightly rotating seed (Lambda-LTB dust, E = -v^2/2, common bang
time) in a flat LambdaCDM universe; its shells are halted into a static
cluster of circular orbits with the Parodi (2018) profile.

    python3 make_figures.py   ->  fig1_static.png ... fig12_sparc_rho_r.png, tab_sparc79.tex, results.txt

Figures:
  fig1_static.png        Parodi profile: rotation curve and density
  fig2_shells.png        shell trajectories and y = R_b/r_c
  fig3_rc_time.png    rotation curves of the example galaxy at five times (with Lambda)
  fig4_timeline.png      growth of r_c, R_b and y
  fig5_primordial.png    early overdensity of the seed
  fig6_disc_halo.png     thick exponential disc + PITS halo (inside-out disc)
  fig7_seed_density.png  seed as a density perturbation: <delta>(t) and mean density
  fig8_local_H.png       local Hubble rate of an under- or overdense region (larger scales)
  fig9_three_radii.png   accretion front, turnaround radius and edge of the seed
  fig10_feedback.png     Phase III: core growth after accretion, driven by feedback
  fig11_sparc_rc.png     doubly normalised rotation curves of the 79 SPARC late-type galaxies
  fig12_sparc_rho_r.png  central density vs scale radius: total matter and Burkert halo (SPARC)
  fig15_phase1_profile.png  Phase I + II density profile, self-similar (Lambda=0)
  fig14_inner_gradient.png  observed inner gradient vs URC prediction (inner points left out)
  fig13_sparc_scaling.png  r_c and rho(0) against v: SPARC vs model
  fig16_surface_density.png  rho(0) r_c against v: SPARC, model, Donato et al. (2009)
  fig17_rcv_distribution.png  cumulative distribution of r_c/v: SPARC vs model
  tab_sparc79_grad.tex   compact two-column table (graduate paper)
  tab_sparc79.tex        per-galaxy table (appendix); sparc_data/sparc79_fits.csv
Data needed: sparc_data/rotmod/*_rotmod.dat (SPARC) and sparc_data/parodi2018_B2.csv, _B3.csv.

Units: kpc, km/s, Msun; times in Gyr. G = 4.30091e-6. Runtime: a few minutes.
Requires numpy, scipy, matplotlib.
"""
import numpy as np
from scipy.special import lambertw
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.special import erfinv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DPI = 300                            # resolution of the PNG figures

G = 4.30091e-6
GYR = 0.977792                       # 1 kpc/(km/s) in Gyr
OM, OL, H0 = 0.3, 0.7, 0.07          # H0 in km/s/kpc (= 70 km/s/Mpc)
LAM3 = OL*H0**2                      # Lambda/3 in (km/s/kpc)^2
out = []
def report(s): print(s); out.append(s)
plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight", "figure.dpi": 150})
C1, C2, C3, C4, C5 = "#1f4e79", "#c0504d", "#4f8a3c", "#8064a2", "#d08a1e"

# ---------------- flat LambdaCDM = Lambda-LTB with E = 0, t_B = 0 ----------------
def t_of_a(a): return 2/(3*np.sqrt(LAM3))*np.arcsinh(np.sqrt(OL/OM)*a**1.5)*GYR
def t_of_z(z): return t_of_a(1/(1+z))
def a_of_t(t): return (np.sqrt(OM/OL)*np.sinh(1.5*np.sqrt(LAM3)*t/GYR))**(2/3)
def z_of_t(t): return 1/a_of_t(t)-1
def H_of_t(t):
    a = a_of_t(t); return H0*np.sqrt(OM/a**3+OL)
def R_bg(M, t):  # E = 0 shell
    return (2*G*M/LAM3)**(1/3)*np.sinh(1.5*np.sqrt(LAM3)*t/GYR)**(2/3)
T0 = t_of_z(0)
report(f"t0 = {T0:.4f} Gyr")
rho_m0 = OM*3*H0**2/(8*np.pi*G)
chk = R_bg(1e10, T0)**3*4*np.pi/3*rho_m0/1e10
report(f"check E=0 shell: (4pi/3) rho_m0 R_bg(t0)^3 / M = {chk:.10f}")

# ---------------- Parodi profile ----------------
ex = np.exp
def m(x):  return x*(1-ex(-x))**2 - 0.5*x*(1+x)*ex(-2*x) + 0.25*(1-ex(-2*x))
def mp(x): return (1-(1-x)*ex(-x))**2
def vn(x): return np.sqrt(m(x)/x)
def F(x):  return mp(x)/x**2
xg = np.concatenate([np.linspace(1e-6, 1, 2001)[:-1], np.logspace(0, np.log10(400), 8000)])
Ws = CubicSpline(xg, cumulative_trapezoid(m(xg)*mp(xg)/xg, xg, initial=0))
Js = CubicSpline(xg, cumulative_trapezoid(xg**2*mp(xg), xg, initial=0))
CE = float(Ws(300) - 300 - 0.25*np.log(300))
XP = brentq(lambda x: x*mp(x)-m(x), 1, 20)
Y0 = np.exp(4*(0.25-CE))
report(f"C_E = {CE:.7f}; x_p = {XP:.5f}; y0 = {Y0:.2f}")

# ---------------- model parameters ----------------
V = 60.0
LAMSPIN = 0.035
ZACC = 0.5
SIG = 0.5
lam_p = lambda p: LAMSPIN*np.exp(SIG*np.sqrt(2)*erfinv(2*p-1))

# energy matching:  W/2 - (Lambda/3) I_2 = -1/2 v^2 M_h (1 - 3 lam^2)
#   =>  Wint(y) + (2 Lambda r_c^2/(3 v^2)) J(y) = m(y) (1 - 3 lam^2)
def y_of(Rb, lam3=LAM3, lam=LAMSPIN):
    Fn = lambda y: Ws(y) + (2*3*lam3*Rb**2/(3*V**2*y**2))*Js(y) - m(y)*(1-3*lam**2)
    return brentq(Fn, 1.5, Y0*0.9999)
def y_asym(Rb, lam3=LAM3, lam=LAMSPIN):
    ell = lam3*Rb**2/V**2
    return brentq(lambda y: (0.25-CE-0.25*np.log(y)) - 3*lam**2*(y+0.25) - (2/3)*ell*y, 1.5, Y0*0.9999)

# ---------------- Lambda-LTB shell orbits, E = -v^2/2, t_B = 0 ----------------
def fR(R, M, lam3): return 2*G*M/R - V**2 + lam3*R**2
def dfR(R, M, lam3): return -2*G*M/R**2 + 2*lam3*R
def Rta(M, lam3=LAM3):
    return 2*G*M/V**2 if lam3 == 0 else brentq(lambda R: fR(R, M, lam3), G*M/V**2, 3*G*M/V**2)
def Iint(Ra, Rb_, M, lam3, Rt):
    """int_Ra^Rb dR/sqrt(f) in kpc/(km/s), with R = Rt - u^2 near turnaround."""
    Rs = 0.8*Rt; tot = 0.0
    if Ra < Rs:
        tot += quad(lambda R: 1/np.sqrt(fR(R, M, lam3)), Ra, min(Rb_, Rs), limit=200)[0]
    if Rb_ > Rs:
        lo = max(Ra, Rs); ua, ub = np.sqrt(max(Rt-Rb_, 0)), np.sqrt(Rt-lo)
        def g(u):
            if u < 1e-12: return 2/np.sqrt(-dfR(Rt, M, lam3))
            return 2*u/np.sqrt(max(fR(Rt-u*u, M, lam3), 1e-300))
        tot += quad(g, ua, ub, limit=200)[0]
    return tot
def t_ta(M, lam3=LAM3): Rt = Rta(M, lam3); return Iint(0, Rt, M, lam3, Rt)*GYR
def t_out(M, R, lam3=LAM3): Rt = Rta(M, lam3); return Iint(0, R, M, lam3, Rt)*GYR
def t_in(M, R, lam3=LAM3):
    Rt = Rta(M, lam3); return (Iint(0, Rt, M, lam3, Rt) + Iint(R, Rt, M, lam3, Rt))*GYR
def R_shell(M, t, lam3=LAM3):
    """areal radius of shell M at time t (before halting)."""
    Rt = Rta(M, lam3); tt_ = Iint(0, Rt, M, lam3, Rt)*GYR
    if t <= tt_:
        return brentq(lambda R: Iint(0, R, M, lam3, Rt)*GYR - t, 1e-9*Rt, Rt)
    return brentq(lambda R: (tt_ + Iint(R, Rt, M, lam3, Rt)*GYR) - t, 1e-6*Rt, Rt)

# ---------------- halting: accretion front ----------------
def halt(Rb, lam3=LAM3, lam=LAMSPIN):
    y = y_of(Rb, lam3, lam); rc = Rb/y; Mh = V**2*rc*m(y)/G
    return t_in(Mh, Rb, lam3), y, rc, Mh
def front_table(lam3, lam=LAMSPIN, Rmax=140):
    Rbs = np.concatenate([np.logspace(-1, 0, 15)[:-1], np.linspace(1, Rmax, 140)])
    rows = np.array([halt(R, lam3, lam) for R in Rbs])
    return Rbs, rows
TAB = {}
for lam3 in [0.0, LAM3]:
    Rbs, rows = front_table(lam3)
    TAB[lam3] = (Rbs, rows)
def front_at(t, lam3=LAM3):
    Rbs, rows = TAB[lam3]
    ts = rows[:, 0]
    Rb = np.interp(t, ts, Rbs)
    t_, y, rc, Mh = halt(Rb, lam3)
    # one Newton-like refinement
    for _ in range(3):
        dt = 1e-3*Rb
        t2 = halt(Rb+dt, lam3)[0]
        Rb = Rb - (t_-t)*dt/(t2-t_)
        t_, y, rc, Mh = halt(Rb, lam3)
    return Rb, y, rc, Mh

# Lambda = 0 limit check
XIB0 = TAB[0.0][0][-1]/(V*TAB[0.0][1][-1, 0]/GYR)
report(f"Lambda=0 limit: xi_b = {XIB0:.6f}, y = {TAB[0.0][1][-1,1]:.4f} (expected 0.174047, 50.5429)")

# ---------------- fiducial galaxy ----------------
TACC = t_of_z(ZACC)
res = {}
for lam3, lab in [(0.0, "Lambda=0"), (LAM3, "Lambda-LTB")]:
    Rb, y, rc, Mh = front_at(TACC, lam3)
    res[lam3] = (Rb, y, rc, Mh)
    rta_e = Rta(Mh, lam3); tta_e = t_ta(Mh, lam3)
    report(f"{lab}: t_acc={TACC:.3f} Gyr: R_b={Rb:.2f} kpc, y={y:.2f}, r_c={rc:.3f} kpc, M_tot={Mh:.4e}, "
           f"edge turnaround R_ta={rta_e:.1f} kpc at t={tta_e:.3f} Gyr (z={z_of_t(tta_e):.2f}); "
           f"ell_b = Lambda R_b^2/(3 v^2) = {lam3*Rb**2/V**2:.4f}")
RBF, YF, RCF, MTOT = res[LAM3]
RTOT = G*MTOT/V**2
r_L = (3*MTOT/(4*np.pi*rho_m0))**(1/3)
report(f"  Lagrangian radius of outermost shell r_L = {r_L:.1f} kpc; r_L/R_b = {r_L/RBF:.2f}; mean density/rho_m0 = {(r_L/RBF)**3:.0f}")
report(f"  y asymptotic formula at t_acc: {y_asym(RBF):.2f} (exact {YF:.2f})")
# same mass, Lambda=0 end of accretion
t_acc0_sameM = halt(brentq(lambda R: halt(R, 0.0)[3]-MTOT, 10, 200), 0.0)[0]
report(f"  same M_tot with Lambda=0 would finish accretion at t={t_acc0_sameM:.3f} Gyr (z={z_of_t(t_acc0_sameM):.3f})")
# spin percentiles at t_acc (fixed R_b of fiducial, with and without Lambda)
for p in [0.16, 0.5, 0.84]:
    l = lam_p(p)
    report(f"  spin p={p}: lambda={l:.4f}: y(Lambda=0) = {y_of(RBF, 0.0, l):.1f}, y(Lambda-LTB, R_b={RBF:.0f}) = {y_of(RBF, LAM3, l):.1f}")
report(f"  pure-Lambda core (lambda->0): y = {y_of(RBF, LAM3, 1e-6):.1f}")

def rc_t(t, lam3=LAM3):
    if t >= TACC: return res[lam3][2]
    return front_at(t, lam3)[2]
report("time line (Lambda-LTB | Lambda=0):")
TL = []
for z in [8, 6, 4, 2, 1, 0.5, 0]:
    t = t_of_z(z)
    if t < TACC:
        Rb1, y1, rc1, Mh1 = front_at(t, LAM3); Rb0, y0_, rc0, _ = front_at(t, 0.0)
    else:
        Rb1, y1, rc1, Mh1 = res[LAM3]; Rb0, y0_, rc0, _ = res[0.0]
    rho0 = V**2/(np.pi*G*rc1**2)/1e9
    TL.append((z, t, rc1, Rb1, y1, rho0, V*vn(1/rc1), rc0, Rb0, y0_))
    report(f"  z={z}: t={t:.3f}  r_c={rc1:.3f} R_b={Rb1:.2f} y={y1:.2f} rho0={rho0:.4f} v(1kpc)={V*vn(1/rc1):.1f} | "
           f"r_c={rc0:.3f} R_b={Rb0:.2f} y={y0_:.2f}")
report(f"rho(0)/rho_m(t) at z=6: {V**2/(np.pi*G*TL[1][2]**2)/(rho_m0*(1+6)**3):.3e}")
# Lambda in the static cluster today
for r in [10, 50, RBF]:
    report(f"  Lambda correction to v_c^2 today at r={r:.0f} kpc: {LAM3*r**2/(V**2*vn(r/RCF)**2):.4f}")

# ---------------- primordial fluctuation ----------------
report("primordial mean overdensity  delta(M,t) = (R_bg/R)^3 - 1  vs linear (3/5) v^2/(H^2 R^2):")
PRIM = {}
for z in [1090, 100, 30]:
    t = t_of_z(z); H = H_of_t(t)
    Mh_t = front_at(t)[3]
    # shells still expanding at t: t < t_ta(M)
    Mlo = brentq(lambda M: t_ta(M) - t, Mh_t, MTOT) if t_ta(Mh_t) < t else Mh_t
    Ms = np.logspace(np.log10(Mlo*1.05), np.log10(MTOT), 30)
    dex, dlin = [], []
    for M in Ms:
        R = R_shell(M, t)
        dex.append((R_bg(M, t)/R)**3 - 1); dlin.append(0.6*V**2/(H**2*R**2))
    PRIM[z] = (Ms, np.array(dex), np.array(dlin), Mh_t, Mlo)
    report(f"  z={z}: halted mass M_h={Mh_t:.3e}, turned-around up to {Mlo:.3e}; delta(M_tot)={dex[-1]:.4e} "
           f"(linear {dlin[-1]:.4e}); slope dln delta/dln M (outer half) = "
           f"{np.polyfit(np.log(Ms[15:]), np.log(dex[15:]), 1)[0]:.4f}")

# ---------------- composite density at epochs ----------------
def rho_profile(t, lam3=LAM3, n=160):
    """returns (r_in, rho_in, r_inf, rho_inf) in kpc, Msun/pc^3"""
    if t < TACC: Rb, y, rc, Mh = front_at(t, lam3)
    else: Rb, y, rc, Mh = res[lam3]
    r_in = np.logspace(-2, np.log10(Rb), 300)
    rho_in = V**2/(4*np.pi*G*rc**2)*F(r_in/rc)/1e9
    if t >= TACC or Mh >= MTOT*0.999: return r_in, rho_in, None, None
    Mg = np.exp(np.linspace(np.log(Mh*1.0005), np.log(MTOT), n))
    Rg = np.array([R_shell(M, t, lam3) for M in Mg])
    ok = np.diff(Rg) > 0
    dRdM = np.gradient(Rg, Mg)
    rho_inf = 1/(4*np.pi*Rg**2*dRdM)/1e9
    return r_in, rho_in, Rg, rho_inf
PROF = {}
for z in [6, 3, 1.5, ZACC, 0]:
    PROF[z] = rho_profile(t_of_z(z))
    if PROF[z][2] is not None:
        jump = PROF[z][1][-1]/PROF[z][3][0]
        tz = t_of_z(z)
        report(f"  z={z}: front R_b={PROF[z][0][-1]:.2f}, edge R_e={PROF[z][2][-1]:.1f} kpc, density jump {jump:.2f}, "
               f"rho_edge/rho_m = {PROF[z][3][-1]/(rho_m0*(1+z)**3/1e9):.3f}")


# ---------------- present core vs accretion end and spin (Lambda-LTB) ----------------
def front_exact(t, lam3=LAM3, lam=LAMSPIN):
    Rb = brentq(lambda R: halt(R, lam3, lam)[0] - t, 0.05, 135)
    tt_, y, rc, Mh = halt(Rb, lam3, lam)
    return Rb, y, rc, Mh
report("present core r_c/v [kpc per km/s] (Lambda-LTB | Lambda=0), spins 16/50/84%:")
RCTAB = []
for za in [2, 1, 0.5, 0.25]:
    ta = t_of_z(za); row = [za]
    for lam3 in [LAM3, 0.0]:
        for p in [0.16, 0.5, 0.84]:
            Rb, y, rc, Mh = front_exact(ta, lam3, lam_p(p)); row.append((rc/V, y))
    RCTAB.append(row)
    report(f"  z_acc={za}: LTB " + " ".join(f"{q[0]:.4f}(y={q[1]:.1f})" for q in row[1:4]) +
           " | L=0 " + " ".join(f"{q[0]:.4f}(y={q[1]:.1f})" for q in row[4:7]))
# ---- explicit growth law: xi_b -> 1/(1+3pi/2) and analytic first-order Lambda lag c_L = (105pi/8+83/6)/(1+3pi/2)
T0_ = 1+1.5*np.pi; CL_ = (105*np.pi/8+83/6)/T0_
def _tau_h(k):
    F = lambda u: 2/u-1+k*u*u
    um = brentq(F, 1.5, 3.5)
    return quad(lambda u: 1/np.sqrt(F(u)), 0, um, limit=200)[0]+quad(lambda u: 1/np.sqrt(F(u)), 1, um, limit=200)[0]
report(f"explicit growth law: c_Lambda analytic = {CL_:.4f} (tau1 = {105*np.pi/8+83/6:.4f}); numerical d tau_h/d kappa at kappa=1e-4: "
       f"{(_tau_h(1e-4)-_tau_h(0))/1e-4:.3f}")
def _explicit(t, lam):                      # t in Gyr
    t = t/GYR; l0 = LAM3*(t/T0_)**2; Rb_ = V*t/(T0_*(1+CL_*l0)); ell_ = LAM3*Rb_**2/V**2
    A_ = 3*lam**2+2/3*ell_; B_ = 0.25-CE-0.75*lam**2
    y_ = lambertw(4*A_*np.exp(4*B_)).real/(4*A_); return Rb_, Rb_/y_
_eR, _eC = [], []
for lam in [0.021, 0.028, 0.035, 0.045, 0.058]:
    for za in [6, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.35, 0.25, 0.1, 0]:
        ta = t_of_z(za); Rb, y, rc, Mh = front_exact(ta, LAM3, lam); a_, b_ = _explicit(ta, lam)
        _eR.append(a_/Rb-1); _eC.append(b_/rc-1)
report(f"  explicit law vs full Lambda-LTB (lambda 0.021-0.058, z 6-0): max |dR_b| = {np.max(np.abs(_eR)):.4f}, max |dr_c| = {np.max(np.abs(_eC)):.4f}")
# scaling check: the shell equation with E = -v^2/2 is invariant under R -> aR, M -> a^3 M, v -> a v (t fixed),
# also with Lambda. Recompute the full Lambda-LTB front for other v at the same t and spin: r_c/v and y must not change.
_V0 = V; _sc = []
for _vt in [30.0, 60.0, 120.0]:
    V = _vt
    for za in [2, 0.5]:
        ta = t_of_z(za)
        Rb_ = brentq(lambda R: halt(R, LAM3, LAMSPIN)[0]-ta, 0.05*_vt/_V0, 135*_vt/_V0)
        _, y_, rc_, Mh_ = halt(Rb_, LAM3, LAMSPIN); _sc.append((_vt, za, rc_/_vt, y_, Mh_/_vt**3, V**2/(np.pi*G*rc_**2)/1e9))
V = _V0
for za in [2, 0.5]:
    r_ = [q for q in _sc if q[1] == za]
    report(f"scaling check (z_end={za}, lambda={LAMSPIN}): " + "; ".join(f"v={q[0]:.0f}: r_c/v={q[2]:.5f} y={q[3]:.3f} M/v^3={q[4]:.4e} rho0={q[5]:.4f}" for q in r_) +
           f"; max rel. dev. of r_c/v {max(abs(q[2]/r_[1][2]-1) for q in r_):.1e}")

# ================= FIGURES =================
# Fig 1: static profile in areal radius
x = np.logspace(-2, 2, 600)
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax[0].plot(x, vn(x), c=C1, lw=2, label="total")
ax[0].plot(x, 1-ex(-x), c=C2, ls="--", label=r"$1-e^{-x}$ term")
ax[0].plot(x, np.sqrt(0.25*(1-ex(-2*x)*(1+2*x+2*x*x))/x), c=C3, ls=":", lw=1.6, label="exponential sphere")
ax[0].axvline(XP, c="0.6", lw=0.8); ax[0].axhline(1, c="0.6", lw=0.8)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.1); ax[0].set_xlabel(r"$x=R/r_c$"); ax[0].set_ylabel(r"$v_c/v$")
ax[0].legend(frameon=False, fontsize=8)
ax[1].loglog(x, F(x)/4, c=C1, lw=2, label="total")
ax[1].loglog(x, ex(-2*x), c=C3, ls=":", lw=1.6, label="exponential sphere")
ax[1].loglog(x, 0.25/x**2, c="0.6", lw=0.8, label=r"$\propto x^{-2}$")
ax[1].set_ylim(1e-5, 2); ax[1].set_xlabel(r"$x=R/r_c$"); ax[1].set_ylabel(r"$\rho/\rho(0)$")
ax[1].legend(frameon=False, fontsize=8)
fig.savefig("fig1_static.png", dpi=DPI); plt.close(fig)

# Fig 2: shell trajectories
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
tt = np.linspace(0.05, T0, 300)
for M, c in zip([5e9, 2e10, 4e10, MTOT], [C4, C2, C5, C1]):
    for lam3, ls in [(LAM3, "-"), (0.0, "--")]:
        th_ = halt(brentq(lambda R: halt(R, lam3)[3]-M, 0.05, 130), lam3)[0] if M <= res[lam3][3]*1.0001 else np.inf
        tv = tt[tt < th_]
        ax[0].plot(tv, [R_shell(M, t, lam3) for t in tv], c=c, ls=ls, lw=1.6 if ls == "-" else 1.0,
                   label=fr"$M={M:.1e}\,M_\odot$".replace("e+", r"\times10^{").replace("\\times10^{10", r"\times10^{10}").replace("\\times10^{09", r"\times10^{9}") if ls == "-" else None)
tf = np.linspace(0.1, TACC, 120)
ax[0].plot(tf, [front_at(t)[0] for t in tf], c="k", lw=1.2, label=r"accretion front $R_b(t)$")
ax[0].plot(tt, R_bg(MTOT, tt), c="0.6", lw=1, ls=":", label=r"$E=0$ shell, $M_{\rm tot}$")
ax[0].set_xlabel(r"$t$ [Gyr]"); ax[0].set_ylabel(r"$R$ [kpc]"); ax[0].set_ylim(0, 260)
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
for p, c in zip([0.16, 0.5, 0.84], [C3, C1, C2]):
    l = lam_p(p)
    Rbs = np.linspace(2, 140, 60)
    ax[1].plot(Rbs, [y_of(R, LAM3, l) for R in Rbs], c=c, lw=1.8, label=fr"$\lambda={l:.3f}$, $\Lambda$LTB")
    ax[1].axhline(y_of(10, 0.0, l), c=c, ls="--", lw=1)
ax[1].axhline(55, c="0.4", lw=0.8, ls=":"); ax[1].text(100, 57, "SPARC 55", color="0.4", fontsize=8)
ax[1].set_xlabel(r"accretion front $R_b$ [kpc]"); ax[1].set_ylabel(r"$y=R_b/r_c$")
ax[1].legend(frameon=False, fontsize=7, loc="lower left"); ax[1].set_ylim(0, 110)
fig.savefig("fig2_shells.png", dpi=DPI); plt.close(fig)

# Fig 3: rotation curves of the example galaxy at the times of Fig. 1 (with Lambda; growth ends at t_acc)
fig, ax = plt.subplots(figsize=(3.6, 3.25))
for tg, c in zip([0.5, 1, 2, 4, TACC], [C4, C3, C5, C2, C1]):
    rc = front_at(tg)[2] if tg < TACC else res[LAM3][2]
    rr = np.logspace(-2, 2, 400)
    vc2 = V**2*m(rr/rc)/(rr/rc) - LAM3*rr**2
    lab = fr"$t={tg:g}$ Gyr" if tg < TACC else fr"$t={tg:.1f}$ Gyr (end)"
    ax.plot(rr, np.sqrt(vc2), c=c, lw=1.6, label=lab)
    ax.plot(rc, V*np.sqrt(m(1.0)), "o", c=c, ms=3.5)
    report(f"Fig 3 (RC vs t): t = {tg:.2f} Gyr, r_c = {rc:.3f} kpc")
ax.set_xscale("log"); ax.set_xlim(0.01, 100); ax.set_ylim(0, 66)
ax.set_xlabel(r"areal radius $R$ [kpc]"); ax.set_ylabel(r"$v_c$ [km s$^{-1}$]")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("fig3_rc_time.png", dpi=DPI); plt.close(fig)

# Fig 4: time line
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.3))
tl = np.linspace(0.3, T0, 150)
for lam3, ls, lab in [(LAM3, "-", r"$\Lambda$LTB"), (0.0, "--", r"$\Lambda=0$")]:
    rcs = [rc_t(t, lam3) for t in tl]
    Rbs_ = [front_at(t, lam3)[0] if t < TACC else res[lam3][0] for t in tl]
    ax[0].semilogy(tl, rcs, c=C1, ls=ls, lw=1.8, label=r"$r_c$, "+lab)
    ax[0].semilogy(tl, Rbs_, c=C2, ls=ls, lw=1.8, label=r"$R_b$, "+lab)
    ax[1].plot(tl, np.array(Rbs_)/np.array(rcs), c=C1, ls=ls, lw=1.8, label=r"$y=R_b/r_c$, "+lab)
ax[0].axvline(TACC, c="0.6", lw=0.8); ax[1].axvline(TACC, c="0.6", lw=0.8)
ax[0].set_xlabel(r"$t$ [Gyr]"); ax[0].set_ylabel("radius [kpc]"); ax[0].legend(frameon=False, fontsize=7, loc="lower right")
ax[1].set_xlabel(r"$t$ [Gyr]"); ax[1].set_ylabel(r"$y$"); ax[1].set_ylim(0, 60); ax[1].legend(frameon=False, fontsize=8)
fig.savefig("fig4_timeline.png", dpi=DPI); plt.close(fig)

# Fig 5: primordial fluctuation
fig, ax = plt.subplots(figsize=(4.8, 3.6))
for z, c in zip([1090, 100, 30], [C1, C2, C3]):
    Ms_, dex, dlin, Mh_t, Mlo = PRIM[z]
    ax.loglog(Ms_, dex, c=c, lw=1.8, label=fr"$z={z}$")
    ax.loglog(Ms_, dlin, c=c, ls=":", lw=1.2)
    ax.axvline(Mh_t, c=c, lw=0.8, ls="--")
ax.axhline(1, c="0.6", lw=0.8)
ax.set_xlabel(r"enclosed mass $M$ [M$_\odot$]"); ax.set_ylabel(r"mean overdensity $\bar\delta(M)$")
ax.legend(frameon=False, fontsize=8)
fig.savefig("fig5_primordial.png", dpi=DPI); plt.close(fig)


# ================= DISC + PITS HALO =================
import warnings
from scipy.special import j0, j1
from scipy.optimize import least_squares
warnings.simplefilter('ignore')
BRK=[0,2,5,10,20,50,100,200,500,1000,3000]
hank=lambda f: sum(quad(f,a,b,limit=800)[0] for a,b in zip(BRK[:-1],BRK[1:]))
q=0.2; RG=np.logspace(-3,np.log10(1000),220)
I=np.array([hank(lambda s: s*j1(s*u)/((1+s*s)**1.5*(1+q*s))) for u in RG])
K=np.array([hank(lambda s: s*s*j0(s*u)/((1+s*s)**1.5*(1+q*s))) for u in RG])
K0=hank(lambda s: s*s/((1+s*s)**1.5*(1+q*s)))
sI=CubicSpline(np.log(RG),np.log(RG*I)); sK=CubicSpline(np.log(RG),I/RG+K)
vd2=lambda R,Md,Rd: G*Md/Rd*np.exp(sI(np.log(np.maximum(R/Rd,1e-3))))
rhoeq=lambda R,Md,Rd: Md/(4*np.pi*Rd**3)*sK(np.log(np.maximum(R/Rd,1e-3)))
vt2=lambda R,rc: V**2*m(R/rc)/(R/rc)
rhot=lambda R,rc: V**2/(4*np.pi*G*rc**2)*mp(R/rc)/(R/rc)**2
vp2=lambda R,r0,rh: 4*np.pi*G*r0*rh**2*(1-rh/R*np.arctan(R/rh))
rhop=lambda R,r0,rh: r0/(1+(R/rh)**2)
# today
Rb0,y0,rc0,_=front_at(TACC)
Rd0=(rc0-0.213)/0.644/2
Md0=0.51**2*vt2(2.2*Rd0,rc0)/vd2(2.2*Rd0,1,Rd0)
Sig0=Md0/(2*np.pi*Rd0**2)   # Msun/kpc^2
kd=Rd0/rc0
rows=[]
report(f"K0(q=0.2)={K0:.4f}; today: rc={rc0:.3f} Rb={Rb0:.2f} Rd={Rd0:.3f} Rd/rc={kd:.3f} Md={Md0:.3e} Sigma0={Sig0/1e6:.1f} Msun/pc2; BTF 47v^4={47*V**4:.2e}")
for z in [6,3,2,1,0]:
    t=t_of_z(z); Rb,y,rc,Mh=front_at(min(t,TACC))
    Rd=kd*rc; Md=2*np.pi*Sig0*Rd**2
    R=np.logspace(np.log10(0.02*rc),np.log10(Rb),300)
    vh2=vt2(R,rc)-vd2(R,Md,Rd); rhoh=rhot(R,rc)-rhoeq(R,Md,Rd)
    f=lambda p: np.sqrt(np.maximum(vp2(R,*np.exp(p)),0))-np.sqrt(np.maximum(vh2,0))
    s=least_squares(f,[np.log(rhot(rc,rc)),np.log(rc)]); r0,rh=np.exp(s.x)
    tot=np.sqrt(vd2(R,Md,Rd)+vp2(R,r0,rh))-np.sqrt(vt2(R,rc))
    Mtot=V**2/G*(Rb+rc/4)
    fd=np.sqrt(vd2(2.2*Rd,Md,Rd)/vt2(2.2*Rd,rc))
    d=dict(z=z,t=t,rc=rc,Rb=Rb,Rd=Rd,Md=Md,fd=fd,fb=Md/Mtot,rho0t=rhot(1e-4*rc,rc),rhoh0=rhoh[0],minh=np.min(rhoh/rhot(R,rc)),
           r0=r0,rh=rh,rms=np.sqrt(np.mean(tot**2)),mx=np.max(abs(tot)))
    rows.append(d)
    report(f"z={z}: t={t:.2f} rc={rc:.3f} Rd={Rd:.3f} Md={Md:.2e} Md/Mtot={d['fb']:.4f} vd/v(2.2Rd)={fd:.3f} | rho_tot(0)={d['rho0t']/1e9:.4f} rho_h/rho_tot(R=0.02 r_c)={rhoh[0]/rhot(R[0],rc):.3f} exact centre={1-rc/(4*kd*V**2/(3*np.pi*K0*G*Sig0)):.3f} min ratio={d['minh']:.3f} | PITS rho0={r0/1e9:.4f} Msun/pc3 rh={rh:.3f} rh/rc={rh/rc:.3f} rho0*rh={r0*rh/1e6:.1f} Msun/pc2 | rms={d['rms']:.2f} max={d['mx']:.2f}")
fig,ax=plt.subplots(1,2,figsize=(9.2,3.6))
cols=["#7b5ea7","#c44e52","#dd8452","#55a868","#1f4e79"]
for d,c in zip(rows,cols):
    rc,Rd,Md,r0,rh,Rb=d['rc'],d['Rd'],d['Md'],d['r0'],d['rh'],d['Rb']
    R=np.logspace(-2,np.log10(Rb),400)
    lab=(r"$z\leq0.5$" if d['z']==0 else f"$z={d['z']}$")
    ax[0].plot(R,np.sqrt(vt2(R,rc)),c=c,lw=1.8,label=lab)
    ax[0].plot(R,np.sqrt(vd2(R,Md,Rd)),c=c,lw=1.1,ls="-.")
    ax[0].plot(R,np.sqrt(vp2(R,r0,rh)),c=c,lw=1.1,ls="--")
    ax[1].loglog(R,rhot(R,rc)/1e9,c=c,lw=1.8,label=lab)
    ax[1].loglog(R,rhop(R,r0,rh)/1e9,c=c,lw=1.1,ls="--")
ax[0].set_xscale("log"); ax[0].set_xlim(0.01,100); ax[0].set_ylim(0,66)
ax[0].set_xlabel(r"areal radius $R$ [kpc]"); ax[0].set_ylabel(r"$v_c$ [km s$^{-1}$]")
from matplotlib.lines import Line2D
ax[0].legend([Line2D([],[],c="k",lw=1.8),Line2D([],[],c="k",lw=1.1,ls="--"),Line2D([],[],c="k",lw=1.1,ls="-.")],
             ["total (P18)","PITS halo","disc"],frameon=False,fontsize=7.5,loc="upper left")
ax[1].set_ylim(1e-6,20); ax[1].set_xlabel(r"areal radius $R$ [kpc]"); ax[1].set_ylabel(r"$\rho$ [M$_\odot$ pc$^{-3}$]")
ax[1].legend(frameon=False,fontsize=7.5,loc="lower left")
fig.savefig("fig6_disc_halo.png", dpi=DPI); plt.close(fig)

# ================= SEED DENSITY =================
# Seed as a density perturbation: <delta>(t) and mean densities
E_ = lambda a: np.sqrt(OM/a**3+OL)
Dgrow = lambda a: 2.5*OM*E_(a)*quad(lambda x: 1/(x*E_(x))**3, 0, a)[0]   # D(a) = a in the matter era
tta_e = t_ta(MTOT); t0 = t_of_z(0)
ts = np.concatenate([np.logspace(np.log10(t_of_z(1090)), np.log10(TACC*0.999), 260), np.linspace(TACC, t0, 60)])
dseed, dgal, dlin = [], [], []
ti = ts[0]; di = (R_bg(MTOT, ti)/R_shell(MTOT, ti))**3-1; Di = Dgrow(a_of_t(ti))
for t in ts:
    Re = R_shell(MTOT, t) if t < TACC else RBF
    dseed.append((R_bg(MTOT, t)/Re)**3-1)
    dlin.append(di*Dgrow(a_of_t(t))/Di)
    if t >= t_of_z(10):
        Rb_, y_, rc_, Mh_ = front_at(min(t, TACC)); dgal.append((R_bg(Mh_, t)/Rb_)**3-1)
    else:
        dgal.append(np.nan)
dseed, dgal, dlin = map(np.array, (dseed, dgal, dlin))
rbg = np.array([3*MTOT/(4*np.pi*R_bg(MTOT, t)**3)/1e9 for t in ts])     # Msun/pc^3
rseed, rgal = rbg*(1+dseed), rbg*(1+dgal)
report("Seed as a density perturbation (M_tot):")
for lab, t in [("z=6", t_of_z(6)), ("z=2", t_of_z(2)), ("edge turnaround", tta_e), ("t_acc", TACC), ("today", t0)]:
    report(f"  {lab}: t={t:.3f} Gyr z={z_of_t(t):.2f}: <delta>_seed={np.interp(t, ts, dseed):.3f}, "
           f"delta_lin={np.interp(t, ts, dlin):.3f}, <delta>_gal={np.interp(t, ts, dgal):.1f}, "
           f"rho_seed={np.interp(t, ts, rseed):.3e}, rho_bg={np.interp(t, ts, rbg):.3e} Msun/pc^3")
i = np.argmin(rseed); report(f"  minimum of rho_seed at t={ts[i]:.3f} Gyr (edge turnaround {tta_e:.3f} Gyr)")
i = np.argmax(dseed > 1); report(f"  <delta>_seed = 1 at t={ts[i]:.3f} Gyr, z={z_of_t(ts[i]):.2f}")
report(f"  Lambda=0 growing-galaxy overdensity 1+<delta> = 4.5(1+1/(4y))/xi_b^2 = {4.5*(1+1/(4*50.54))/0.174047**2:.1f}")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.35))
ax[0].loglog(ts, dseed, c=C1, lw=2, label=r"seed ($M\leq M_{\rm tot}$)")
ax[0].loglog(ts, dlin, c=C1, lw=1.2, ls="--", label="seed, linear theory")
ax[0].loglog(ts, dgal, c=C2, lw=2, label=r"galaxy (inside $R_b$)")
ax[0].axhline(1, c="0.75", lw=0.6, ls=":")
ax[0].set_xlim(ts[0], t0*1.05); ax[0].set_ylim(1e-3, 3e3)
ax[0].set_xlabel(r"$t$ [Gyr]"); ax[0].set_ylabel(r"mean overdensity $\langle\delta\rangle$")
ax[0].legend(frameon=False, fontsize=7, loc="upper left")
ax[1].semilogy(ts, rseed, c=C1, lw=2, label="seed")
ax[1].semilogy(ts, rgal, c=C2, lw=2, label=r"galaxy (inside $R_b$)")
ax[1].semilogy(ts, rbg, c="0.4", lw=1.3, ls="--", label="average Universe")
ax[1].set_xlim(0, 14); ax[1].set_ylim(1e-8, 1e-3)
ax[1].set_xlabel(r"$t$ [Gyr]"); ax[1].set_ylabel(r"mean density [M$_\odot$ pc$^{-3}$]")
ax[1].legend(frameon=True, framealpha=1, facecolor="white", edgecolor="none", fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.40))
for a in ax:
    for t in (tta_e, TACC): a.axvline(t, c="0.6", lw=0.7)
ax[1].text(tta_e, 2.5e-8, " turnaround", fontsize=8, color="0.35")
ax[1].text(TACC, 2.5e-8, r" $t_{\rm acc}$", fontsize=8, color="0.35")
fig.savefig("fig7_seed_density.png", dpi=DPI); plt.close(fig)

# ================= LARGER SCALES: LOCAL HUBBLE RATE =================
# A homogeneous region (one energy E for all its shells, common bang time) inside a flat
# LambdaCDM universe; Planck values for the background. Units: H_global = 1.
OMP, HGL = 0.315, 67.4
OLP = 1-OMP
t0P = 2/(3*np.sqrt(OLP))*np.arcsinh(np.sqrt(OLP/OMP))
def ageP(R, Ek): return quad(lambda r: 1/np.sqrt(OMP/r+2*Ek+OLP*r*r), 0, R, limit=200)[0]
def R0P(Ek): return brentq(lambda R: ageP(R, Ek)-t0P, 1e-3, 5)          # R in units of R_bg(t0)
def local(Ek):
    R = R0P(Ek); return 1/R**3-1, np.sqrt(OMP/R**3+2*Ek/R**2+OLP)       # delta_0, H_loc/H_global
Eks = np.linspace(-0.2, 0.35, 56)
LH = np.array([local(e) for e in Eks])
E73 = brentq(lambda e: local(e)[1]-73.0/HGL, 0.01, 0.5); d73, h73 = local(E73)
fP = OMP**0.55
report("Larger scales (Planck background, H_global = 67.4 km/s/Mpc):")
for e in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]:
    d, h = local(e); report(f"  2E/(H0 R_bg)^2={2*e:+.2f}: delta_0={d:+.3f}  H_loc={HGL*h:.1f}  (linear {HGL*(1-fP*d/3):.1f})")
report(f"  H_loc = 73.0 needs delta_0 = {d73:.3f} (linear theory {-3*(73/HGL-1)/fP:.3f}); f = Om^0.55 = {fP:.3f}")
fig, ax = plt.subplots(figsize=(5.2, 3.7))
ax.axhspan(72, 74, color=C2, alpha=0.15, lw=0)
ax.axhline(73, c=C2, lw=0.8); ax.axhline(HGL, c="0.4", lw=0.8, ls="--")
dd = np.linspace(-0.55, 1.1, 200)
ax.plot(dd, HGL*(1-fP*dd/3), c=C1, lw=1.1, ls=":", label="linear theory")
ax.plot(LH[:, 0], HGL*LH[:, 1], c=C1, lw=2, label=r"exact ($\Lambda$LTB)")
ax.plot([d73], [73], "o", c=C2, ms=5)
ax.text(0.55, 73.4, "local (supernovae)", fontsize=8, color=C2)
ax.text(0.55, HGL+0.4, "global (CMB)", fontsize=8, color="0.35")
ax.set_xlim(-0.55, 1.1); ax.set_ylim(54, 78)
ax.set_xlabel(r"mean overdensity of the local region today, $\delta_0$"); ax.set_ylabel(r"local $H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig("fig8_local_H.png", dpi=DPI); plt.close(fig)

# ================= THE THREE RADII =================
# R-t diagram: shell trajectories, accretion front R_b(t), turnaround radius R_ta(t), edge of the seed R_e(t)
from matplotlib.patches import Circle
Rbs_T, rows_T = TAB[LAM3]
tf_T, Mh_T = rows_T[:, 0], rows_T[:, 3]                       # front time and halted mass versus R_b
def halt_of(M):                                              # (t_halt, R_halt) of shell M
    return np.interp(M, Mh_T, tf_T), np.interp(M, Mh_T, Rbs_T)
tt3 = np.linspace(0.05, 13.5, 400)
Rb3 = np.array([front_at(min(t, TACC))[0] if t > tf_T[0] else np.nan for t in tt3])
Ms3 = np.linspace(0.05, 1.0, 12)*MTOT
tta_e = t_ta(MTOT)
Mta = np.logspace(np.log10(Mh_T[0]), np.log10(MTOT), 80)
tta3 = np.array([t_ta(M) for M in Mta]); Rta3 = np.array([Rta(M) for M in Mta])
fig, ax = plt.subplots(1, 2, figsize=(7.8, 2.35), gridspec_kw={"width_ratios": [1.45, 1]})
a = ax[0]
for M in Ms3:
    th, Rh = halt_of(M)
    ts_ = np.linspace(0.02, min(th, 13.5), 160)
    Rs_ = [R_shell(M, t) for t in ts_]
    a.plot(ts_, Rs_, c="0.75", lw=0.8)
    a.plot([th, 13.5], [Rh, Rh], c="0.75", lw=0.8)
tsE = np.linspace(0.02, TACC, 300); RE = [R_shell(MTOT, t) for t in tsE]
a.plot(tsE, RE, c=C1, lw=2.2, label=r"edge of the seed $R_e(t)$")
a.plot([TACC, 13.5], [RBF, RBF], c=C1, lw=2.2)
a.plot(tta3, Rta3, c=C5, lw=2, ls="--", label=r"turnaround radius $R_{\rm ta}(t)$")
a.plot(tt3, Rb3, c=C2, lw=2.2, label=r"accretion front $R_b(t)$")
a.plot([tta_e], [Rta(MTOT)], "o", c=C5, ms=5); a.plot([TACC], [RBF], "o", c=C2, ms=5)
a.annotate("edge turns around", (tta_e, Rta(MTOT)), (tta_e+0.6, Rta(MTOT)+16), fontsize=8, color=C5, ha="left",
           arrowprops=dict(arrowstyle="-", color=C5, lw=0.6))
a.annotate(r"edge meets front: $t_{\rm acc}$", (TACC, RBF), (TACC+0.6, RBF-26), fontsize=7.5, color=C2,
           arrowprops=dict(arrowstyle="-", color=C2, lw=0.6))
a.text(2.2, 8, "galaxy", fontsize=9, color=C2)
a.text(6.0, 115, "falling\nshells", fontsize=8.5, color="0.35", ha="center")
a.annotate("expanding shells", (1.5, 100), (0.3, 186), fontsize=8.5, color="0.35", ha="left",
           arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))
a.set_xlim(0, 13.5); a.set_ylim(0, 300); a.set_yticks([0, 50, 100, 150, 200])
a.set_xlabel(r"cosmic time $t$ [Gyr]"); a.set_ylabel(r"areal radius $R$ [kpc]")
a.legend(frameon=False, fontsize=7, loc="upper right", labelspacing=0.3, borderaxespad=0.3)
# snapshot at z = 2
b = ax[1]; tS = t_of_z(2)
RbS = front_at(tS)[0]; RtaS = float(np.interp(tS, tta3, Rta3)); ReS = R_shell(MTOT, tS)
for R, fc, ec, lab in [(ReS*1.25, "#f3f3f3", "none", None), (ReS, "#dbe7f3", C1, r"$R_e$"),
                       (RtaS, "#f7e7cf", C5, r"$R_{\rm ta}$"), (RbS, "#f2c9c7", C2, r"$R_b$")]:
    b.add_patch(Circle((0, 0), R, fc=fc, ec=ec, lw=1.6 if ec != "none" else 0, ls="--" if ec == C5 else "-"))
b.text(0, 0, "galaxy", ha="center", va="center", fontsize=8)
b.text(0, -(RbS+RtaS)/2, "falling", ha="center", va="center", fontsize=8)
_rm = (RtaS+ReS)/2
b.annotate("still expanding", (-_rm*0.707, _rm*0.707), (-ReS*0.95, ReS*1.18), ha="center", va="center", fontsize=8,
           arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
b.text(0, -ReS*1.13, "average Universe", ha="center", va="center", fontsize=8, color="0.4")
for R, lab, c in [(RbS, r"$R_b$", C2), (RtaS, r"$R_{\rm ta}$", C5), (ReS, r"$R_e$", C1)]:
    if c == C5: b.text(-R*0.55, R*0.55, lab, fontsize=9, color=c, ha="center", va="center")
    else: b.annotate(lab, (R*0.707, R*0.707), (R*0.707+8, R*0.707+8), fontsize=9, color=c)
lim = ReS*1.3; b.set_xlim(-lim, lim); b.set_ylim(-lim, lim); b.set_aspect("equal"); b.axis("off")
fig.savefig("fig9_three_radii.png", dpi=DPI); plt.close(fig)
report(f"Three radii at z=2: R_b={RbS:.1f}, R_ta={RtaS:.1f}, R_e={ReS:.1f} kpc; edge turnaround t={tta_e:.2f} Gyr, R_ta={Rta(MTOT):.1f}")

# ================= EARLY, COMPACT CORES (little red dots) =================
# Route A: the galaxy at a given early time (Lambda = 0 formulas, accurate at z >~ 3)
# Route B: early turnaround of a shell of mass M, and how rare such a seed is (peak height nu)
def y_lam(lam):                                            # core equation without Lambda, Lambert W
    A_ = 3*lam**2; B_ = 0.25-CE-0.75*lam**2
    return float(np.real(lambertw(4*A_*np.exp(4*B_))))/(4*A_)
# sigma(M) of linear LambdaCDM today: Eisenstein & Hu (1998) no-wiggle transfer function
OB_, NS_, S8_, HH = 0.045, 0.965, 0.81, 0.7
omh2, obh2, fbar = OM*HH**2, OB_*HH**2, OB_/OM
s_EH = 44.5*np.log(9.83/omh2)/np.sqrt(1+10*obh2**0.75)
aG_EH = 1-0.328*np.log(431*omh2)*fbar+0.38*np.log(22.3*omh2)*fbar**2
def T_EH(k):                                               # k in h/Mpc
    Gam = OM*HH*(aG_EH+(1-aG_EH)/(1+(0.43*k*HH*s_EH)**4))
    q = k*(2.7255/2.7)**2/Gam
    L_ = np.log(2*np.e+1.8*q); C_ = 14.2+731/(1+62.5*q)
    return L_/(L_+C_*q*q)
Wth = lambda x: 3*(np.sin(x)-x*np.cos(x))/x**3
def sig2_R(R, A=1.0):                                      # R in Mpc/h
    f = lambda lk: A*np.exp(lk)**(NS_+3)*T_EH(np.exp(lk))**2*Wth(np.exp(lk)*R)**2/(2*np.pi**2)
    return quad(f, np.log(1e-5), np.log(1e3), limit=400)[0]
A_PS = S8_**2/sig2_R(8.0)
def sigma_M(M):                                            # M in Msun
    R = (3*M*HH/(4*np.pi*2.775e11*OM))**(1/3)
    return np.sqrt(sig2_R(R, A_PS))
t_i = t_of_z(1090)
D_ratio = Dgrow(1.0)/Dgrow(a_of_t(t_i))
def dlin0(M, v):                                           # linear overdensity extrapolated to today
    return 0.6*v**2/(H_of_t(t_i)**2*R_bg(M, t_i)**2)*D_ratio
report("Early compact cores (Lambda = 0 formulas):")
report(f"  fiducial seed: delta_lin,0 = {dlin0(MTOT, V):.2f}, sigma(M_tot) = {sigma_M(MTOT):.2f}, nu = {dlin0(MTOT, V)/sigma_M(MTOT):.2f}")
tz6 = t_of_z(6)/GYR
LRD_A = []
for lam in [0.035, 0.021, 0.010]:
    yl = y_lam(lam); rho0 = yl**2/(np.pi*G*XIB0**2*tz6**2)/1e9
    rcs = [XIB0*v*tz6/yl*1e3 for v in (30, 60, 100, 200)]
    Mg = [v*v*XIB0*v*tz6*(1+1/(4*yl))/G for v in (30, 60, 100, 200)]
    LRD_A.append((lam, yl, rho0, rcs))
    report(f"  z=6, lambda={lam}: y={yl:.1f}, rho(0)={rho0:.1f} Msun/pc^3, r_c[pc] for v=30/60/100/200: "
           + " / ".join(f"{r:.0f}" for r in rcs) + "; M_gal = " + " / ".join(f"{m:.1e}" for m in Mg))
LRD_B = []
for tta_ in [0.3, 0.5, 0.9]:
    row = []
    for M in [1e9, 1e10, 1e11]:
        vv = (np.pi*G*M/(tta_/GYR))**(1/3); nu = dlin0(M, vv)/sigma_M(M)
        row.append((vv, nu))
    LRD_B.append((tta_, z_of_t(tta_), row))
    report(f"  turnaround by t={tta_} Gyr (z={z_of_t(tta_):.1f}): (v, nu) for M=1e9/1e10/1e11: "
           + " / ".join(f"({a:.0f}, {b:.1f})" for a, b in row))

# ================= OTHER PROFILES: core equation for Parodi vs PITS (Lambda = 0) =================
PROFS = {"Parodi": (m, mp), "PITS": (lambda x: x-np.arctan(x), lambda x: x*x/(1+x*x))}
report("Core equation m(y)(1-3 lam^2) = int_0^y m m'/x dx for two profiles (Lambda = 0):")
for name, (mf, mpf) in PROFS.items():
    Fint = lambda Y: quad(lambda x: mf(x)*mpf(x)/x, 1e-8, Y, limit=400)[0]
    out_ = []
    for lam in [0.0, 0.021, 0.035, 0.058]:
        Ys = np.logspace(0.3, 4, 300); gv = [mf(Y)*(1-3*lam**2)-Fint(Y) for Y in Ys]
        roots = [brentq(lambda Y: mf(Y)*(1-3*lam**2)-Fint(Y), Ys[i], Ys[i+1])
                 for i in range(len(Ys)-1) if gv[i]*gv[i+1] < 0]
        out_.append("none" if not roots else f"{roots[0]:.1f}")
    report(f"  {name}: y for lambda = 0 / 0.021 / 0.035 / 0.058: " + " / ".join(out_))

# ================= EVOLVING DARK ENERGY (DESI-like w0 wa): effect on the core =================
W0_, WA_ = -0.75, -0.8
fDE = lambda a: a**(-3*(1+W0_+WA_))*np.exp(-3*WA_*(1-a))
def t_of_a_dyn(a): return quad(lambda x: 1/(x*np.sqrt(OM/x**3+OL*fDE(x))), 1e-8, a)[0]/H0*GYR
def core_estimate(t, L3):                                  # explicit growth law with an effective Lambda/3
    tk = t/GYR; l0 = L3*(XIB0*tk)**2; Rb_ = XIB0*V*tk/(1+10*l0); ell = L3*Rb_**2/V**2
    A_ = 3*LAMSPIN**2+2/3*ell; B_ = 0.25-CE-0.75*LAMSPIN**2
    y_ = float(np.real(lambertw(4*A_*np.exp(4*B_))))/(4*A_); return Rb_, y_, Rb_/y_
a_acc = 1/(1+ZACC)
wz = W0_+WA_*(1-a_acc); L3dyn = OL*fDE(a_acc)*H0**2*(-(1+3*wz)/2)
c0 = core_estimate(t_of_z(ZACC), LAM3); c1 = core_estimate(t_of_a_dyn(a_acc), L3dyn)
report(f"Evolving dark energy (w0={W0_}, wa={WA_}): effective Lambda/3 at z={ZACC} is {L3dyn/LAM3:.3f} x constant; "
       f"r_c = {c1[2]:.3f} kpc vs {c0[2]:.3f} kpc (change {100*(c1[2]/c0[2]-1):+.1f}%), y = {c1[1]:.1f} vs {c0[1]:.1f}")

# ================= PHASE III: AFTER GROWTH, FEEDBACK AND THE RESPONSE OF THE GALAXY =================
# Net energy epsilon(t) (in units of v^2 M_tot / 2) gained by the matter after t_acc; constant star formation:
# epsilon(t) = eps0 (t - t_acc)/(t0 - t_acc). Core equation: A = 3 lam^2 + eps + 2 ell/3, B = 1/4 - C_E - 3 lam^2/4 - eps/4.
def y_phase3(eps, ell, lam=LAMSPIN):
    A_ = 3*lam**2+eps+2/3*ell; B_ = 0.25-CE-0.75*lam**2-eps/4
    return float(np.real(lambertw(4*A_*np.exp(4*B_))))/(4*A_)
def phase3(Mt, eps):
    Rb_ = G*Mt/V**2
    for _ in range(30):
        y_ = y_phase3(eps, LAM3*Rb_**2/V**2); Rb_ = G*Mt/V**2*y_/(y_+0.25)
    return Rb_, y_, Rb_/y_
e_SN = 1e49/1.989e43                                       # 1e51 erg per 100 Msun of stars, in (km/s)^2 per Msun
report(f"Phase III: e_SN/(v^2/2) = {e_SN/(0.5*V**2):.0f}; eps = {2*e_SN/V**2:.0f} f_c dM*/M_tot")
T0 = t_of_z(0)
P3 = {}
cases = [(0.5, 0.0), (0.5, 0.002), (0.5, 0.005), (0.5, 0.01), (1.0, 0.0)]
t1 = t_of_z(1.0); Mt1 = front_at(t1)[3]
eps_match = brentq(lambda e: phase3(Mt1, e)[2]-RCF, 1e-5, 0.05)
cases.append((1.0, eps_match))
for zacc_, eps0 in cases:
    tacc_ = t_of_z(zacc_); Mt = MTOT if zacc_ == 0.5 else Mt1
    tt = np.linspace(0.3, T0, 160); rc_ = []
    for t in tt:
        if t < tacc_: rc_.append(front_at(t)[2])
        else: rc_.append(phase3(Mt, eps0*(t-tacc_)/(T0-tacc_))[2])
    P3[(zacc_, eps0)] = (tt, np.array(rc_))
    Rb_, y_, rc0_ = phase3(Mt, eps0)
    report(f"  z_acc={zacc_}, eps0={eps0:.4f}: today R_b={Rb_:.2f} y={y_:.1f} r_c={rc0_:.3f} kpc rho0={V**2/(np.pi*G*rc0_**2)/1e9:.4f} Msun/pc^3")
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
sty = {(0.5, 0.0): (C1, "-"), (0.5, 0.002): (C3, "-"), (0.5, 0.005): (C5, "-"), (0.5, 0.01): (C2, "-"),
       (1.0, 0.0): ("0.5", "--"), (1.0, eps_match): (C4, "--")}
for key, (tt, rc_) in P3.items():
    c, ls = sty[key]; lab = rf"$z_{{\rm acc}}={key[0]}$, $\epsilon_0={key[1]:.2g}$"
    ax[0].plot(tt, rc_, c=c, ls=ls, lw=1.8, label=lab)
    ax[1].semilogy(tt, V**2/(np.pi*G*rc_**2)/1e9, c=c, ls=ls, lw=1.8)
for a in ax:
    a.axvline(TACC, c="0.6", lw=0.7); a.axvline(t1, c="0.6", lw=0.7, ls=":")
    a.set_xlabel(r"$t$ [Gyr]"); a.set_xlim(0, 13.6)
ax[0].set_ylabel(r"$r_c$ [kpc]"); ax[0].set_ylim(0, 4.2); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1].set_ylabel(r"central density $\rho(0)$ [M$_\odot$ pc$^{-3}$]")
fig.savefig("fig10_feedback.png", dpi=DPI); plt.close(fig)

# ceiling of Phase III set by the disc (core ceiling r_c,max of the disc-halo split)
rcmax_disc = 4*kd*V**2/(3*np.pi*K0*G*Sig0)
eps_ceiling = brentq(lambda e: phase3(MTOT, e)[2]-rcmax_disc, 1e-5, 0.05)
report(f"  disc ceiling r_c,max = {rcmax_disc:.3f} kpc reached for eps0 = {eps_ceiling:.4f} (z_acc=0.5)")
for eps0 in [0.0, 0.002, 0.005, 0.01]:
    rc_ = phase3(MTOT, eps0)[2]; report(f"  eps0={eps0}: halo share of central density = {1-rc_/rcmax_disc:+.3f}")

# ================= SPARC: THE 79 LATE-TYPE GALAXIES OF PARODI (2018) =================
# Data: SPARC rotmod files (Lelli, McGaugh & Schombert 2016) in sparc_data/rotmod/, Burkert fits and
# v_flat of Parodi (2018, Tables B.2 and B.3) in sparc_data/. Each rotation curve is fitted with the
# Parodi profile (parameters v, r_c); the error of each point is errV, 3% of V and 1 km/s added in quadrature.
import csv, os
from scipy.optimize import least_squares
SP = "sparc_data"
B2 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B2.csv"))))
B3 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B3.csv"))))
def vBur2(R, r0, rho0):                                    # Burkert halo, rho0 in Msun/kpc^3
    x = R/r0; return G*np.pi*rho0*r0**3*(np.log(1+x*x)+2*np.log(1+x)-2*np.arctan(x))/R
SPF, RCD = [], []
for b2, b3 in zip(B2, B3):
    name = b2["name"].replace("UGC0786", "UGC07866")       # typo in Table B.2
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"))
    R, Vo, eV0, Vg, Vd = d.T[:5]; SBd = d.T[6]
    eV = np.sqrt(eV0**2+(0.03*Vo)**2+1.0)
    res = lambda p: (np.exp(p[0])*vn(R/np.exp(p[1]))-Vo)/eV
    s = min((least_squares(res, [np.log(Vo.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    dof = max(len(R)-2, 1); chi = 2*s.cost/dof
    cov = np.linalg.inv(s.jac.T@s.jac)*max(chi, 1.0)        # errors, inflated if chi2_r > 1
    vv, rcv = np.exp(s.x); ev, erc = vv*np.sqrt(cov[0, 0]), rcv*np.sqrt(cov[1, 1])
    Y = float(b2["Y"]); vb2 = np.abs(Vg)*Vg+Y*Vd**2
    r0B, rhoB = float(b2["r0B"]), float(b2["rho0B"])
    xs = np.linspace(0.01, 40, 4000)*r0B; vh2 = np.max(vBur2(xs, r0B, rhoB*1e9))
    vf = float(b3["vflat"]) if b3["vflat"] else np.nan
    SPF.append(dict(name=name, n=len(R), v=vv, ev=ev, rc=rcv, erc=erc, chi=chi,
                    rho=vv**2/(np.pi*G*rcv**2)/1e9, r0B=r0B, rhoB=rhoB, Y=Y, vf=vf,
                    hshare=vh2/vv**2, bshare=np.max(vb2)/vv**2, SB0=SBd[0],
                    cvr=cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
    RCD.append((R/rcv, Vo/vv, eV/vv, Vo/vv-vn(R/rcv)))
A_ = lambda k: np.array([g[k] for g in SPF])
lv, lrc, lr0, lrB = np.log10(A_("v")), np.log10(A_("rc")), np.log10(A_("r0B")), np.log10(A_("rhoB"))
rng = np.random.default_rng(1)
def bisector(x, y):
    b1 = np.polyfit(x, y, 1)[0]; b2_ = 1/np.polyfit(y, x, 1)[0]
    b = (b1*b2_-1+np.sqrt((1+b1**2)*(1+b2_**2)))/(b1+b2_); return b, np.mean(y)-b*np.mean(x), b1, b2_
def fit_report(lbl, x, y):
    b, a, b1, b2_ = bisector(x, y); n = len(x)
    bs = np.array([bisector(x[i], y[i])[:2] for i in (rng.integers(0, n, n) for _ in range(2000))])
    report(f"  {lbl}: N={n}, bisector slope {b:+.3f} +- {bs[:,0].std():.3f}, intercept {a:+.3f} +- {bs[:,1].std():.3f} "
           f"(OLS y|x {b1:+.3f}, OLS x|y {b2_:+.3f}), scatter {np.std(y-a-b*x):.3f} dex")
    return b, a
report("SPARC, 79 late-type galaxies of Parodi (2018): log(rho/v^2) = a + b log r   [Msun/pc^3, km/s, kpc]")
fit_report("total mass, Parodi profile rho(0) vs r_c", lrc, np.log10(A_("rho"))-2*lv)
bB, aB = fit_report("Burkert halo (Table B.2) rho_0 vs r_0, v = Parodi v", lr0, lrB-2*lv)
m54 = np.isfinite(A_("vf"))
fit_report("Burkert halo, 54 galaxies with v = v_flat (Table B.3)", lr0[m54], lrB[m54]-2*np.log10(A_("vf")[m54]))
M_ = np.column_stack([lr0[m54], np.log10(A_("vf")[m54]), np.ones(m54.sum())])
c_ = np.linalg.lstsq(M_, lrB[m54], rcond=None)[0]
report(f"  check, joint OLS on the 54 (Parodi 2018/2023 form): log rho0 = {c_[0]:+.3f} log r0 {c_[1]:+.3f} log v_flat {c_[2]:+.3f}")
for lbl, y, x in [("halo share v_h,max^2/v^2 vs r_0", np.log10(A_("hshare")), lr0),
                  ("baryon share max V_bar^2/v^2 vs r_0", np.log10(A_("bshare")), lr0),
                  ("disc central surface brightness vs r_c", np.log10(A_("SB0")), lrc)]:
    c = np.polyfit(x, y, 1); n = len(x)
    bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(2000))]
    report(f"  {lbl}: OLS slope {c[0]:+.3f} +- {np.std(bs):.3f}, corr {np.corrcoef(x, y)[0,1]:+.2f}")
c = np.polyfit(lr0, np.log10(A_("Y")), 1); report(f"  Upsilon (Table B.2) vs r_0: slope {c[0]:+.3f}")
for lbl, y in [("halo share", np.log10(A_("hshare"))), ("baryon share", np.log10(A_("bshare")))]:
    M_ = np.column_stack([lr0, lv, np.ones(len(lv))]); c = np.linalg.lstsq(M_, y, rcond=None)[0]
    bs = np.array([np.linalg.lstsq(M_[i], y[i], rcond=None)[0] for i in (rng.integers(0, 79, 79) for _ in range(2000))]).std(0)
    report(f"  {lbl} at fixed v: log share = {c[0]:+.3f}(+-{bs[0]:.3f}) log r_0 {c[1]:+.3f}(+-{bs[1]:.3f}) log v + const")
y = np.log10(A_("rhoB")/A_("rho")); c = np.polyfit(lrc, y, 1)
report(f"  rho_0,Bu/rho(0) vs r_c: slope {c[0]:+.3f}, corr {np.corrcoef(lrc, y)[0,1]:+.2f}, median {np.median(10**y):.2f}")
report(f"  median v = {np.median(A_('v')):.1f} km/s, median r_c = {np.median(A_('rc')):.2f} kpc, "
       f"median chi2_r = {np.median(A_('chi')):.2f}; median r_0/r_c = {np.median(A_('r0B')/A_('rc')):.2f}")
X_ = np.concatenate([r[0] for r in RCD]); Yo = np.concatenate([r[1] for r in RCD]); D_ = np.concatenate([r[3] for r in RCD])
W_ = np.concatenate([r[2] for r in RCD])
report(f"  doubly normalised RC: {len(X_)} points, rms(V/v - model) = {np.sqrt(np.mean(D_**2)):.3f}, "
       f"median |dev| = {np.median(np.abs(D_)):.3f}, median error = {np.median(W_):.3f}; "
       f"rms for R > r_c: {np.sqrt(np.mean(D_[X_>1]**2)):.3f}")
# ---- figure 11: the doubly normalised rotation curve
fig, ax = plt.subplots(2, 1, figsize=(4.4, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
cm = plt.get_cmap("viridis"); vmin_, vmax_ = np.log10(15), np.log10(130)
for g, (x, y, e, dv) in zip(SPF, RCD):
    col = cm((np.log10(g["v"])-vmin_)/(vmax_-vmin_))
    ax[0].plot(x, y, "-", c=col, lw=0.5, alpha=0.6); ax[0].plot(x, y, "o", c=col, ms=2.0, alpha=0.75, mec="none")
    ax[1].plot(x, dv, "o", c=col, ms=2.0, alpha=0.6, mec="none")
xx = np.logspace(-2, np.log10(40), 400)
ax[0].plot(xx, vn(xx), "k-", lw=2.0, label=r"P18 profile $\sqrt{m(x)/x}$")
ax[1].axhline(0, c="k", lw=1.2)
bins = np.logspace(-1.3, 1.4, 14); idx = np.digitize(X_, bins)
xb = [np.median(X_[idx == i]) for i in range(1, len(bins)) if np.sum(idx == i) > 8]
sb = [np.sqrt(np.mean(D_[idx == i]**2)) for i in range(1, len(bins)) if np.sum(idx == i) > 8]
ax[1].plot(xb, sb, "r-", lw=1.2, label="rms in bins"); ax[1].plot(xb, -np.array(sb), "r-", lw=1.2)
ax[0].set_xscale("log"); ax[0].set_xlim(0.03, 40); ax[0].set_ylim(0, 1.45)
ax[0].set_ylabel(r"$V_{\rm obs}/v$"); ax[1].set_ylabel(r"$V_{\rm obs}/v-\sqrt{m/x}$"); ax[1].set_ylim(-0.32, 0.32)
ax[1].set_xlabel(r"$x=R/r_c$")
sm = plt.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(vmin_, vmax_))
cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04); cb.set_label(r"$\log\,v$ [km/s]")
ax[0].legend(frameon=False, loc="upper left"); ax[1].legend(frameon=False, loc="lower left", fontsize=8)
fig.savefig("fig11_sparc_rc.png", dpi=DPI); plt.close(fig)
# ---- figure 12: central density vs scale radius, total mass and dark halo
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
xx = np.linspace(-0.8, 1.45, 10)
ax[0].plot(lrc, np.log10(A_("rho"))-2*lv+4, "o", c=C1, ms=4, label=r"total matter: $\rho(0)$ vs $r_c$")
ax[0].plot(lr0, lrB-2*lv+4, "s", c=C2, ms=3.5, mfc="none", label=r"Burkert halo: $\rho_0$ vs $r_0$")
ax[0].plot(xx, -2*xx+np.log10(1/(np.pi*4.30091e-3*1e6))+4, c=C1, lw=1.3, label="slope $-2$ (exact)")
ax[0].plot(xx, aB+bB*xx+4, c=C2, lw=1.3, ls="--", label=f"bisector slope ${bB:.2f}$")
ax[0].set_xlabel(r"log scale radius [kpc]"); ax[0].set_ylabel(r"$\log(\rho/v^2)+4$  [M$_\odot$pc$^{-3}$, km/s]")
ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(lr0, np.log10(A_("hshare")), "s", c=C2, ms=3.5, mfc="none", label=r"halo: $v_{h,\max}^2/v^2$")
ax[1].plot(lr0, np.log10(A_("bshare")), "o", c=C3, ms=3.5, label=r"baryons: $\max V_{\rm bar}^2/v^2$")
for y, c in [(np.log10(A_("hshare")), C2), (np.log10(A_("bshare")), C3)]:
    p = np.polyfit(lr0, y, 1); ax[1].plot(xx, np.polyval(p, xx), c=c, lw=1.2, ls="--")
ax[1].set_xlabel(r"$\log r_0$ [kpc]"); ax[1].set_ylabel(r"log share of $v^2$"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig12_sparc_rho_r.png", dpi=DPI); plt.close(fig)
# ---- the v-scaling of the model: r_c proportional to v, rho(0) independent of v (fixed spin and end of growth)
for lbl, y in [("log r_c", lrc), ("log rho(0)", np.log10(A_("rho")))]:
    c = np.polyfit(lv, y, 1); n = len(lv)
    bs = [np.polyfit(lv[i], y[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(2000))]
    report(f"  {lbl} vs log v: OLS slope {c[0]:+.3f} +- {np.std(bs):.3f} (model: {'+1' if 'r_c' in lbl else '0'}), "
           f"corr {np.corrcoef(lv, y)[0,1]:+.2f}, scatter {np.std(y-np.polyval(c, lv)):.2f} dex")
q_ = A_("rc")/A_("v")
report(f"  r_c/v [kpc per km/s]: 16/50/84% = {np.percentile(q_,16):.4f} {np.median(q_):.4f} {np.percentile(q_,84):.4f}; "
       f"rho(0) median {np.median(A_('rho')):.4f} Msun/pc^3")
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
vv = np.logspace(np.log10(12), np.log10(160), 50)
cols = {2: C4, 1: C3, 0.5: C1, 0.25: C2}
for row in RCTAB:
    za = row[0]; q16, q50, q84 = row[1][0], row[2][0], row[3][0]
    ax[0].fill_between(vv, q16*vv, q84*vv, color=cols[za], alpha=0.12, lw=0)
    ax[0].plot(vv, q50*vv, c=cols[za], lw=1.4, label=rf"model, growth ends at $z={za:g}$")
    r50 = 1/(np.pi*4.30091e-6*q50**2)/1e9
    ax[1].axhline(r50, c=cols[za], lw=1.4)
    ax[1].axhspan(1/(np.pi*4.30091e-6*q84**2)/1e9, 1/(np.pi*4.30091e-6*q16**2)/1e9, color=cols[za], alpha=0.12, lw=0)
ax[0].loglog(A_("v"), A_("rc"), "ko", ms=3.5, label="SPARC, 79 galaxies")
ax[1].loglog(A_("v"), A_("rho"), "ko", ms=3.5)
ax[0].set_xlabel(r"$v$ [km s$^{-1}$]"); ax[0].set_ylabel(r"$r_c$ [kpc]"); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].set_xlabel(r"$v$ [km s$^{-1}$]"); ax[1].set_ylabel(r"$\rho(0)$ [M$_\odot$ pc$^{-3}$]")
for a in ax:
    a.set_xlim(12, 160); a.set_xticks([20, 30, 50, 100, 150]); a.set_xticklabels(["20", "30", "50", "100", "150"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); fig.savefig("fig13_sparc_scaling.png", dpi=DPI); plt.close(fig)
# ---- surface density rho(0) r_c: model Sigma = v/(pi G r_c/v) (proportional to v), vs Donato et al. (2009)
GPC = 4.30091e-3                                   # pc (km/s)^2 / Msun
Sig_t = A_("rho")*A_("rc")*1e3; Sig_B = A_("rhoB")*A_("r0B")*1e3   # Msun/pc^2 (total; Burkert halo, Parodi 2018)
for lbl, S_ in [("total rho(0) r_c", Sig_t), ("Burkert halo rho_0 r_0 (Parodi 2018)", Sig_B)]:
    y_ = np.log10(S_); c = np.polyfit(lv, y_, 1); n = len(lv)
    bs = [np.polyfit(lv[i], y_[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(2000))]
    report(f"  surface density {lbl}: median {np.median(S_):.0f} Msun/pc^2 (16/84% {np.percentile(S_,16):.0f}/{np.percentile(S_,84):.0f}), "
           f"<log> {y_.mean():.2f} +- {y_.std():.2f} dex; slope vs log v {c[0]:+.2f} +- {np.std(bs):.2f} (model +1), "
           f"corr {np.corrcoef(lv, y_)[0,1]:+.2f}, scatter about fit {np.std(y_-np.polyval(c, lv)):.2f} dex")
for row in RCTAB:
    report(f"  model Sigma at v=73 km/s, growth ends z={row[0]}: {73/(np.pi*GPC*row[2][0]*1e3):.0f} Msun/pc^2 (median spin)")
fig, ax = plt.subplots(figsize=(3.7, 3.0))
ax.axhspan(110, 220, color="0.85", lw=0); ax.axhline(141, c="0.5", lw=1, ls=":")
ax.text(21, 235, "Donato et al. (2009)", fontsize=7, color="0.35")
for row in RCTAB[1:]:
    za = row[0]; ax.plot(vv, vv/(np.pi*GPC*row[2][0]*1e3), c=cols[za], lw=1.4, label=rf"model $\Sigma\propto v$, end at $z={za:g}$")
ax.loglog(A_("v"), Sig_B, "o", mfc="none", mec="0.55", ms=3.5, label=r"Burkert $\rho_0r_0$ (Parodi 2018)")
ax.loglog(A_("v"), Sig_t, "ko", ms=3.5, label=r"total matter $\rho(0)\,r_c$")
ax.set_xlim(20, 160); ax.set_ylim(15, 3000)
ax.set_xticks([20, 30, 50, 100, 150]); ax.set_xticklabels(["20", "30", "50", "100", "150"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel(r"$v$ [km s$^{-1}$]"); ax.set_ylabel(r"surface density [M$_\odot$ pc$^{-2}$]")
ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8, handlelength=1.5)
fig.tight_layout(); fig.savefig("fig16_surface_density.png", dpi=DPI)
# letter version (Fig. A.2 of the letter): P18 cited as P18
for _t in ax.get_legend().get_texts():
    _t.set_text(_t.get_text().replace("(Parodi 2018)", "(P18)"))
fig.savefig("fig_sigma.png", dpi=DPI); plt.close(fig)
# ---- distribution of r_c/v: model (spin distribution x epoch of end of growth) vs SPARC
from scipy.stats import norm as _norm, kstest as _kst
ZG = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
LG = np.exp(np.linspace(np.log(0.006), np.log(0.2), 15))
QG = np.array([[front_exact(t_of_z(z), LAM3, l)[2]/V for l in LG] for z in ZG])     # r_c/v [kpc/(km/s)]
from scipy.interpolate import RectBivariateSpline as _RBS
_lq = _RBS(ZG, np.log(LG), np.log10(QG), kx=3, ky=3)
lnl = np.linspace(np.log(LAMSPIN)-4*SIG, np.log(LAMSPIN)+4*SIG, 161); wl = _norm.pdf(lnl, np.log(LAMSPIN), SIG); wl /= wl.sum()
lq_obs = np.log10(A_("rc")/A_("v"))
sm = np.sqrt((A_("erc")/A_("rc"))**2+(A_("ev")/A_("v"))**2)/np.log(10)        # measurement error of log(r_c/v), conservative
sm = np.minimum(sm, 0.5)
def _lnL(z, sx):
    mu = _lq(z, lnl)[0]                                                          # log q for each spin
    s2 = sm[:, None]**2+sx**2
    return np.sum(np.log(np.sum(wl[None, :]*np.exp(-0.5*(lq_obs[:, None]-mu[None, :])**2/s2)/np.sqrt(2*np.pi*s2), 1)))
zz_ = np.linspace(0.1, 3.0, 59); sxs = np.linspace(0, 0.5, 51)
LL = np.array([[_lnL(z, sx) for sx in sxs] for z in zz_])
iz, isx = np.unravel_index(np.argmax(LL), LL.shape); zb, sxb = zz_[iz], sxs[isx]
L0 = np.array([_lnL(z, 0.0) for z in zz_]); iz0 = np.argmax(L0)
prof_z = LL.max(1); ok_z = zz_[prof_z > LL.max()-0.5]; prof_s = LL.max(0); ok_s = sxs[prof_s > LL.max()-0.5]
def _cdf_model(x, z, sx):                                                         # predicted CDF incl. median measurement error
    mu = _lq(z, lnl)[0]; s_ = np.sqrt(np.median(sm)**2+sx**2)
    return np.sum(wl[None, :]*_norm.cdf((np.atleast_1d(x)[:, None]-mu[None, :])/s_), 1)
mu_med = lambda z: _lq(z, np.log(LAMSPIN))[0, 0]
slope_t = (mu_med(0.4)-mu_med(0.6))/(np.log10(t_of_z(0.4))-np.log10(t_of_z(0.6)))
w_spin = np.sqrt(np.sum(wl*(_lq(zb, lnl)[0]-np.sum(wl*_lq(zb, lnl)[0]))**2))
report("distribution of log(r_c/v):")
report(f"  observed: mean {lq_obs.mean():.3f}, std {lq_obs.std():.3f} dex; median measurement error {np.median(sm):.3f} dex")
report(f"  spin alone (sigma_ln lambda={SIG}) at z_end={zb:.2f}: width {w_spin:.3f} dex; d log(r_c/v)/d log t at median spin, z=0.5: {slope_t:.2f}")
report(f"  max likelihood, single epoch + spin + errors, no extra scatter: z_end = {zz_[iz0]:.2f}, lnL={L0[iz0]:.1f}, "
       f"KS p = {_kst(lq_obs, lambda x: _cdf_model(x, zz_[iz0], 0.0)).pvalue:.3f}")
report(f"  with extra scatter: z_end = {zb:.2f} (range {ok_z.min():.2f}-{ok_z.max():.2f}), sigma_extra = {sxb:.2f} dex "
       f"(range {ok_s.min():.2f}-{ok_s.max():.2f}), Delta lnL = {LL.max()-L0[iz0]:.1f}, "
       f"KS p = {_kst(lq_obs, lambda x: _cdf_model(x, zb, sxb)).pvalue:.3f}")
report(f"  extra scatter as spread in end time: sigma_log t = {sxb/slope_t:.2f} dex, i.e. t_end within x{10**(sxb/slope_t):.2f}")
# parametric bootstrap of the KS test (parameters refitted to each simulated sample, Lilliefors-type calibration)
def _lnL_d(dat, z, sx):
    mu = _lq(z, lnl)[0]; s2 = sm[:, None]**2+sx**2
    return np.sum(np.log(np.sum(wl[None, :]*np.exp(-0.5*(dat[:, None]-mu[None, :])**2/s2)/np.sqrt(2*np.pi*s2), 1)))
def _cdf_d(x, z, sx):
    mu = _lq(z, lnl)[0]; s_ = np.sqrt(np.median(sm)**2+sx**2)
    return np.sum(wl[None, :]*_norm.cdf((np.atleast_1d(x)[:, None]-mu[None, :])/s_), 1)
_D_obs = _kst(lq_obs, lambda x: _cdf_d(x, zb, sxb)).statistic
_rb = np.random.default_rng(7); _zg = np.linspace(0.3, 1.6, 27); _sg = np.linspace(0, 0.45, 19); _Ds = []
_mu_b = _lq(zb, lnl)[0]
for _k in range(300):
    _li = _rb.choice(len(lnl), size=len(lq_obs), p=wl)
    _dat = _mu_b[_li] + np.sqrt(sm**2+sxb**2)*_rb.standard_normal(len(lq_obs))
    _L = np.array([[_lnL_d(_dat, z, s) for s in _sg] for z in _zg]); _i, _j = np.unravel_index(np.argmax(_L), _L.shape)
    _Ds.append(_kst(_dat, lambda x: _cdf_d(x, _zg[_i], _sg[_j])).statistic)
report(f"  KS with refitted parameters (parametric bootstrap, 300 samples): D_obs = {_D_obs:.3f}, calibrated p = {np.mean(np.array(_Ds) >= _D_obs):.3f} "
       f"(naive p = {_kst(lq_obs, lambda x: _cdf_d(x, zb, sxb)).pvalue:.3f})")
for z_ in [0.25, 0.5, 1.0]:
    report(f"  model 16/50/84% of r_c/v at z_end={z_} (spin + median error): "
           + " ".join(f"{10**brentq(lambda x: _cdf_model(x, z_, 0.0)[0]-p_, -4, 1):.4f}" for p_ in [0.16, 0.5, 0.84]))
fig, ax = plt.subplots(figsize=(3.6, 3.2))
xs_ = np.linspace(0.6, 2.9, 300)                                   # log(v/r_c) = -log(r_c/v)
ax.step(np.sort(-lq_obs), np.arange(1, len(lq_obs)+1)/len(lq_obs), where="post", c="k", lw=1.5, label="SPARC, 79 galaxies")
for z_, c_ in [(1.0, C3), (0.25, C2)]:
    ax.plot(xs_, 1-_cdf_model(-xs_, z_, 0.0), c=c_, lw=1.2, ls="--", label=rf"spin only, $z={z_:g}$")
ax.plot(xs_, 1-_cdf_model(-xs_, zb, sxb), c=C1, lw=1.8, label=rf"best fit: $z={zb:.2f}$, $+{sxb:.2f}$ dex")
ax.set_xlabel(r"$\log\,(v/r_c)$ [km s$^{-1}$ kpc$^{-1}$]"); ax.set_ylabel("cumulative fraction"); ax.set_xlim(0.6, 2.9)
ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.6)
fig.tight_layout(); fig.savefig("fig17_rcv_distribution.png", dpi=DPI); plt.close(fig)
# ---- systematic errors: distance (r_c ~ D) and inclination (v ~ 1/sin i), not contained in the fit errors.
# Per-galaxy D errors and e_log V_f (which includes the inclination uncertainty) from SPARC BTFR table (Lelli et al. 2016b)
# for the galaxies listed there; otherwise the median fractional D error of SPARC galaxies at that distance
# (0-5 Mpc: TRGB, 5%; 5-17: mostly Hubble flow, 30%; 17-19.5: Ursa Major, 14%; >19.5: Hubble flow, 20%) and e_logVf = median.
_BT = {}
for l_ in open(os.path.join(SP, "BTFR_Lelli2016a.mrt")).read().split("\n"):
    p_ = l_.split()
    if len(p_) == 11:
        try: _BT[p_[0]] = (float(p_[1]), float(p_[2]), int(p_[3]), float(p_[7]))
        except ValueError: pass
_Dg = np.array([float(b["D"]) for b in csv.DictReader(open(os.path.join(SP, "parodi2018_B1.csv")))])
_has = np.array([g["name"] in _BT for g in SPF])
_fD = np.array([_BT[g["name"]][1]/_BT[g["name"]][0] if g["name"] in _BT else np.nan for g in SPF])
_eVi = np.array([_BT[g["name"]][3] if g["name"] in _BT else np.nan for g in SPF])
_bins = [(0, 5, 0.05), (5, 17, 0.30), (17, 19.5, 0.14), (19.5, 1e3, 0.20)]
for lo_, hi_, f_ in _bins:
    _fD[~_has & (_Dg >= lo_) & (_Dg < hi_)] = f_
_eVi[~_has] = np.median(_eVi[_has])
s_D = np.log10(1+_fD); s_i = _eVi
s_sys = np.sqrt(s_D**2+s_i**2); s_tot = np.sqrt(sm**2+s_sys**2)
def _lnL2(z, sx, sig, sel=slice(None)):
    mu = _lq(z, lnl)[0]; s2 = sig[sel][:, None]**2+sx**2
    return np.sum(np.log(np.sum(wl[None, :]*np.exp(-0.5*(lq_obs[sel][:, None]-mu[None, :])**2/s2)/np.sqrt(2*np.pi*s2), 1)))
def _mlfit(sig, sel=slice(None)):
    L_ = np.array([[_lnL2(z, sx, sig, sel) for sx in sxs] for z in zz_]); i_, j_ = np.unravel_index(np.argmax(L_), L_.shape)
    pz = L_.max(1); ps = L_.max(0)
    return zz_[i_], sxs[j_], zz_[pz > L_.max()-0.5], sxs[ps > L_.max()-0.5]
_z2, _s2, _zr2, _sr2 = _mlfit(s_tot)
_z3, _s3, _, _sr3 = _mlfit(s_tot, _has); _z4, _s4, _, _ = _mlfit(sm, _has)
report(f"systematics: {_has.sum()} of 79 with SPARC distance errors; median sigma_D {np.median(s_D):.3f} dex "
       f"(16/84 {np.percentile(s_D,16):.3f}/{np.percentile(s_D,84):.3f}), inclination proxy e_logVf median {np.median(s_i):.3f}; "
       f"median total error of log(r_c/v): fit {np.median(sm):.3f} -> with systematics {np.median(s_tot):.3f} dex; "
       f"rms systematic {np.sqrt(np.mean(s_sys**2)):.3f} dex")
report(f"  ML with systematics: z_end = {_z2:.2f} ({_zr2.min():.2f}-{_zr2.max():.2f}), extra scatter {_s2:.2f} dex ({_sr2.min():.2f}-{_sr2.max():.2f}); "
       f"as spread in end time sigma_log t = {_s2/slope_t:.2f} dex (x{10**(_s2/slope_t):.2f}); "
       f"subsample with measured errors (N={_has.sum()}): fit errors only z={_z4:.2f} extra {_s4:.2f}, with systematics z={_z3:.2f} extra {_s3:.2f} ({_sr3.min():.2f}-{_sr3.max():.2f})")
_var = lambda a: np.var(a)
report(f"  share of observed variance of log r_c about the v-fit explained by fit+systematic errors: "
       f"{np.mean(s_tot**2)/np.var(lrc-np.polyval(np.polyfit(lv, lrc, 1), lv)):.2f}; of log rho(0) (sigma = sqrt(4 sD^2+4 si^2+...)): "
       f"{np.mean(4*s_D**2+4*s_i**2+(2*A_('ev')/A_('v')/np.log(10))**2+(2*A_('erc')/A_('rc')/np.log(10))**2)/np.var(np.log10(A_('rho'))-np.polyval(np.polyfit(lv, np.log10(A_('rho')), 1), lv)):.2f}; "
       f"x-error attenuation of the r_c-v slope (median errors): {1-np.median(s_i**2+(A_('ev')/A_('v')/np.log(10))**2)/np.var(lv):.3f}; "
       f"median-based share for log r_c: {np.median(s_tot**2)/np.var(lrc-np.polyval(np.polyfit(lv, lrc, 1), lv)):.2f}")
# ---- regression with errors in both variables, their correlation, systematics and intrinsic scatter (max likelihood)
_sx = np.sqrt((A_("ev")/A_("v")/np.log(10))**2+s_i**2)                  # log v: fit + inclination
_sy = np.sqrt((A_("erc")/A_("rc")/np.log(10))**2+s_D**2)                # log r_c: fit + distance
# fit covariance only; built from the fit errors capped at 0.5 dex like the variances below, so that the
# 2x2 error matrix stays positive (uncapped, three galaxies with huge errors made the likelihood undefined)
_cxy = A_("cvr")*np.minimum(A_("ev")/A_("v")/np.log(10), 0.5)*np.minimum(A_("erc")/A_("rc")/np.log(10), 0.5)
_cl = lambda a_: np.minimum(a_, 0.5)
def _mlreg(x, y, sx, sy, cxy):
    def nl(p):
        a_, b_, ls_ = p; s2 = sy**2+b_**2*sx**2-2*b_*cxy+np.exp(2*ls_)
        return 0.5*np.sum((y-a_-b_*x)**2/s2+np.log(s2))
    from scipy.optimize import minimize as _mn
    r_ = _mn(nl, [0.0, 1.0, np.log(0.2)], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return r_.x
_xr, _yr = lv-np.median(lv), lrc
assert np.all(_cl(_sy)**2+_cl(_sx)**2-2*_cxy > 0), "error matrix not positive"
_p = _mlreg(_xr, _yr, _cl(_sx), _cl(_sy), _cxy)
_bsr = []
for _k in range(400):
    i_ = rng.integers(0, len(lv), len(lv)); _bsr.append(_mlreg(_xr[i_], _yr[i_], _cl(_sx)[i_], _cl(_sy)[i_], _cxy[i_])[1:])
_bsr = np.array(_bsr)
# rho(0) = v^2/(pi G r_c^2): log rho = 2 log v - 2 log r_c + const; errors propagated with the covariance
_yrho = np.log10(A_("rho")); _syr = np.sqrt(np.maximum(4*_cl(_sx)**2+4*_cl(_sy)**2-8*_cxy, 1e-6)); _cxr = 2*_cl(_sx)**2-2*_cxy
_pr = _mlreg(_xr, _yrho, _cl(_sx), _cl(_syr), _cxr)
_bsrho = [ _mlreg(_xr[i_], _yrho[i_], _cl(_sx)[i_], _cl(_syr)[i_], _cxr[i_])[1] for i_ in (rng.integers(0, len(lv), len(lv)) for _ in range(400))]
report(f"ML regression (errors in both, fit covariance, D and i systematics, intrinsic scatter): "
       f"log r_c = {_p[1]:.3f} +- {np.std(_bsr[:,0]):.3f} log v, intrinsic scatter {np.exp(_p[2]):.3f} +- {np.std(np.exp(_bsr[:,1])):.3f} dex; "
       f"log rho(0): slope {_pr[1]:+.3f} +- {np.std(_bsrho):.3f}, intrinsic {np.exp(_pr[2]):.3f} dex; median fit correlation of (ln v, ln r_c) {np.median(A_('cvr')):+.2f}; "
       f"constant-Sigma (slope 2) excluded at {(2-_p[1])/np.std(_bsr[:,0]):.1f} sigma")
# ---- spherical vs flattened stellar disc: replace the thin-disc stellar V^2 by G M_*(<R)/R from the 3.6um profile
def _vnS(x): return np.sqrt((x*(1-np.exp(-x))**2-0.5*x*(1+x)*np.exp(-2*x)+0.25*(1-np.exp(-2*x)))/x)
def _fitS(R, V, e):
    res = lambda p: (np.exp(p[0])*_vnS(R/np.exp(p[1]))-V)/e
    s_ = min((least_squares(res, [np.log(V.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    return np.exp(s_.x[0]), np.exp(s_.x[1]), 0, 0
def _sphcorr(name):
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"), ndmin=2)
    R, V, e0, Vd, SBd_ = d[:, 0], d[:, 1], d[:, 2], d[:, 4], d[:, 6]; e = np.sqrt(e0**2+(0.03*V)**2+1.0)
    Sg = 0.5*SBd_*1e6; Mc = np.zeros_like(R); Mc[0] = np.pi*R[0]**2*Sg[0]
    for i in range(1, len(R)): Mc[i] = Mc[i-1]+np.pi*(R[i]-R[i-1])*(R[i]*Sg[i]+R[i-1]*Sg[i-1])
    Vc = np.sqrt(np.maximum(V**2-0.5*Vd**2+G*Mc/R, 1.0))
    v0, r0, _, _ = _fitS(R, V, e); v1, r1, _, _ = _fitS(R, Vc, e)
    return v1/v0, r1/r0, np.median(0.5*Vd**2/np.maximum(G*Mc/R, 1e-9)), np.sum((Vc/v1-_vnS(R/r1))**2), len(R)
_SC = np.array([_sphcorr(g["name"]) for g in SPF])
report(f"thin vs spherical stellar disc: V_disk^2/(G M_*(<R)/R) median per galaxy {np.median(_SC[:,2]):.2f} "
       f"(16/84 {np.percentile(_SC[:,2],16):.2f}/{np.percentile(_SC[:,2],84):.2f}); refit: v ratio {np.percentile(_SC[:,0],[16,50,84]).round(3)}, "
       f"r_c ratio {np.percentile(_SC[:,1],[16,50,84]).round(3)}; shape rms {np.sqrt(_SC[:,3].sum()/_SC[:,4].sum()):.3f}; "
       f"slope r_c-v {np.polyfit(np.log10(A_('v')*_SC[:,0]), np.log10(A_('rc')*_SC[:,1]), 1)[0]:.3f}; median shift log(r_c/v) {np.median(np.log10(_SC[:,1]/_SC[:,0])):+.3f} dex")
# ---- inner slope of the URC, S0 = dv_c/dR(0) = (2/sqrt3) v/r_c = sqrt(4 pi G rho(0)/3), vs surface brightness (Parodi 2018)
B1 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B1.csv"))))
# central surface brightness mu0 [3.6 um, mag/arcsec^2], face-on: exponential fitted to the SPARC disc profile
# (SBdisk of the rotmod files) inside 2.5 R_d and extrapolated to R = 0; M_sun(3.6) = 3.24.
# (The mu0 listed by Parodi 2018 are about 0.5 mag brighter and scatter more, probably projected values.)
def _mu0_faceon(name, Rd):
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"), ndmin=2); R, SB = d[:, 0], d[:, 6]; ok = SB > 0
    m = ok & (R <= 2.5*Rd)
    if m.sum() < 2: m = ok.copy(); m[4:] = False
    return 3.24 + 21.572 - 2.5*np.polyfit(R[m], np.log(SB[m]), 1)[1]/np.log(10)
mu0_P18 = np.array([float(b["mu0"]) for b in B1])
mu0 = np.array([_mu0_faceon(g["name"], float(b["Rd"])) for g, b in zip(SPF, B1)]); Cc = np.array([float(b["C"]) for b in B1])
report(f"mu0 face-on from SPARC profiles: median {np.median(mu0):.2f}; P18 - this: median {np.median(mu0_P18-mu0):+.2f} mag, "
       f"16/84 {np.percentile(mu0_P18-mu0,16):+.2f}/{np.percentile(mu0_P18-mu0,84):+.2f}")
gvd = np.array([float(b["gradv_Rd"]) for b in B3])
S0 = 2*A_("v")/(np.sqrt(3)*A_("rc"))
for lbl, x, y in [("log S0 vs mu0[3.6]", mu0, np.log10(S0)), ("log S0 vs log grad v(R_d) (Parodi 2018)", np.log10(gvd), np.log10(S0)),
                  ("log rho(0) vs mu0[3.6]", mu0, np.log10(A_("rho")))]:
    c = np.polyfit(x, y, 1); n = len(x)
    bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(2000))]
    report(f"  {lbl}: slope {c[0]:+.3f} +- {np.std(bs):.3f}, corr {np.corrcoef(x, y)[0,1]:+.2f}")
for lbl, mk_ in [("HSB (C>1)", Cc > 1), ("LSB (C<1)", Cc <= 1)]:
    report(f"  {lbl}: N={mk_.sum()}, median S0={np.median(S0[mk_]):.1f} km/s/kpc, rho(0)={np.median(A_('rho')[mk_]):.3f} Msun/pc^3, "
           f"r_c/v={np.median((A_('rc')/A_('v'))[mk_]):.4f}, v={np.median(A_('v')[mk_]):.1f}")
report(f"  S0 percentiles 16/50/84: {np.percentile(S0,16):.1f} {np.median(S0):.1f} {np.percentile(S0,84):.1f} km/s/kpc")
# ---- model-independent inner gradient: slope through the origin of the two innermost points (S_in),
#      predicted by the URC fitted WITHOUT these two points (galaxies with >= 5 points)
def _vn(x):
    return np.sqrt((x*(1-np.exp(-x))**2-0.5*x*(1+x)*np.exp(-2*x)+0.25*(1-np.exp(-2*x)))/x)
S_in, S_md, x_in, keep = [], [], [], []
for k_, g in enumerate(SPF):
    d = np.loadtxt(os.path.join(SP, "rotmod", g["name"]+"_rotmod.dat"))
    R_, V_, e_ = d[:, 0], d[:, 1], np.sqrt(d[:, 2]**2+(0.03*d[:, 1])**2+1.0)
    if len(R_) < 5: continue
    ro, vo, eo = R_[2:], V_[2:], e_[2:]
    res_ = lambda p: (np.exp(p[0])*_vn(ro/np.exp(p[1]))-vo)/eo
    s_ = min((least_squares(res_, [np.log(V_.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    vh, rh = np.exp(s_.x); r2, v2 = R_[:2], V_[:2]
    S_in.append(np.sum(r2*v2)/np.sum(r2*r2)); S_md.append(np.sum(r2*vh*_vn(r2/rh))/np.sum(r2*r2))
    x_in.append(np.mean(r2)/g["rc"]); keep.append(k_)
S_in, S_md, x_in, keep = map(np.array, (S_in, S_md, x_in, keep))
vr_ = (A_("v")/A_("rc"))[keep]
report(f"  inner gradient predicted by the IRC fitted without the two innermost points: N={len(keep)}, "
       f"median S_in = {np.median(S_in):.1f} km/s/kpc, median S_in/S_pred = {np.median(S_in/S_md):.3f}, "
       f"rms log ratio {np.std(np.log10(S_in/S_md)):.3f}, corr(log) {np.corrcoef(np.log10(S_in), np.log10(S_md))[0,1]:.2f}, "
       f"range of S_in {S_in.min():.0f}-{S_in.max():.0f}; median x of the points {np.median(x_in):.2f}; "
       f"S_in/(v/r_c) median {np.median(S_in/vr_):.3f} (URC: 1.155 at x=0)")
Ck = Cc[keep]
fig, ax = plt.subplots(figsize=(3.3, 3.2))
for m_, c_, lb in [(Ck > 1, C2, "HSB ($C>1$)"), (Ck <= 1, C1, r"LSB ($C\leq1$)")]:
    ax.loglog(S_md[m_], S_in[m_], "o", c=c_, ms=4, label=lb)
zz = np.array([5, 500]); ax.plot(zz, zz, "k-", lw=1, label="1:1")
ax.set_xlim(7, 200); ax.set_ylim(7, 200)
ax.set_xlabel(r"predicted inner gradient [km s$^{-1}$ kpc$^{-1}$]")
ax.set_ylabel(r"observed inner gradient $S_{\rm in}$ [km s$^{-1}$ kpc$^{-1}$]")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("fig14_inner_gradient.png", dpi=DPI); plt.close(fig)
from scipy import stats as _st
_p = _st.pearsonr(mu0, np.log10(S0)); _u = _st.mannwhitneyu(S0[Cc > 1], S0[Cc <= 1])
report(f"  significance: Pearson r(mu0, log S0) = {_p[0]:.2f}, p = {_p[1]:.1e}; HSB vs LSB S0 Mann-Whitney p = {_u[1]:.1e}")
# ---- do the shape tests discriminate? same fits and held-out test with other two-parameter profiles
_alt = {"URC": _vn,
        "PITS": lambda x: np.sqrt(1-np.arctan(x)/x),
        "Burkert": lambda x: np.sqrt((np.log(1+x)+0.5*np.log(1+x*x)-np.arctan(x))/x),
        "NFW": lambda x: np.sqrt((np.log(1+x)-x/(1+x))/x)}
def _fit2(f, R, V, e):
    r = lambda p: (np.exp(p[0])*f(R/np.exp(p[1]))-V)/e
    return min((least_squares(r, [np.log(a*V.max()), np.log(s)], bounds=([-5, -7], [12, 9]))
                for s in [0.1, 0.3, 1, 3, 10, 30, 100] for a in [1, 2, 5]), key=lambda q: q.cost)
_chi = {k: [] for k in _alt}; _rat = {k: [] for k in _alt}; _dof = []
for g in SPF:
    d = np.loadtxt(os.path.join(SP, "rotmod", g["name"]+"_rotmod.dat"))
    R_, V_, e_ = d[:, 0], d[:, 1], np.sqrt(d[:, 2]**2+(0.03*d[:, 1])**2+1.0); _dof.append(len(R_)-2)
    for k, f in _alt.items():
        _chi[k].append(2*_fit2(f, R_, V_, e_).cost)
        if len(R_) >= 5:
            a_, s_ = np.exp(_fit2(f, R_[2:], V_[2:], e_[2:]).x); r2 = R_[:2]
            _rat[k].append(np.log10(np.sum(r2*V_[:2])/np.sum(r2*a_*f(r2/s_))))
_C = np.array([_chi[k] for k in _alt]); _dof = np.array(_dof)
report("  profile comparison, all fitted to the TOTAL rotation curve V_obs (2 free parameters each, same errors):")
for i, k in enumerate(_alt):
    rr = np.array(_rat[k])
    report(f"    {k:8s} sum chi2 = {_C[i].sum():6.1f} for {_dof.sum()} dof, median chi2/dof {np.median(_C[i]/_dof):.2f}, "
           f"lowest chi2 in {np.sum(np.argmin(_C, 0) == i)}/79; held-out inner gradient: median obs/pred {10**np.median(rr):.2f}, "
           f"rms {np.std(rr):.3f} dex; URC chi2 lower in {np.sum(_C[0] < _C[i])}/79")
# ---- disc scale length vs core: R_d/r_c = f_j lambda y / sqrt(2) (Fall & Efstathiou 1980; Mo, Mao & White 1998)
Rd_ = np.array([float(b["Rd"]) for b in B1]); qd_ = Rd_/A_("rc"); lqd_ = np.log10(qd_)
_pv = _st.pearsonr(np.log10(A_("v")), lqd_)
_x, _y = np.log10(A_("rc")/A_("v")), np.log10(Rd_/A_("v"))
def _bis(x, y):
    a = np.polyfit(x, y, 1)[0]; b = np.polyfit(y, x, 1)[0]; return np.tan(0.5*(np.arctan(a)+np.arctan(1/b)))
_r = _st.pearsonr(_x, _y); _bs = [_bis(_x[i], _y[i]) for i in (rng.integers(0, len(_x), len(_x)) for _ in range(2000))]
report(f"disc vs core: R_d/r_c median {np.median(qd_):.3f}, 16/84% {np.percentile(qd_,16):.3f}/{np.percentile(qd_,84):.3f}, "
       f"scatter {lqd_.std():.3f} dex; corr with log v r={_pv[0]:+.2f} p={_pv[1]:.2f}")
report(f"  log(R_d/v) vs log(r_c/v): r={_r[0]:.2f} p={_r[1]:.1e}, OLS slope {np.polyfit(_x,_y,1)[0]:.2f}, bisector {_bis(_x,_y):.2f} +- {np.std(_bs):.2f}")
# Monte Carlo: spin distribution, spread of epochs from the r_c/v fit (sigma_log t = sxb/slope_t), measurement errors
_rngm = np.random.default_rng(2); _N = 3000; _tz = t_of_z(zb)
_lam = LAMSPIN*np.exp(SIG*_rngm.standard_normal(_N)); _t = _tz*10**((sxb/slope_t)*_rngm.standard_normal(_N))
_Rd, _rc = [], []
for l_, t_ in zip(_lam, _t):
    Rb_, rc_ = _explicit(t_, l_); _rc.append(rc_); _Rd.append(l_*Rb_/np.sqrt(2))      # explicit growth law (1% accurate)
_rc = np.array(_rc)*10**(np.median(sm)*_rngm.standard_normal(_N)); _Rd = np.array(_Rd)*10**(0.10*_rngm.standard_normal(_N))
_xm, _ym = np.log10(_rc/V), np.log10(_Rd/V); _qm = _Rd/_rc
report(f"  model (f_j=1, z_end={zb:.2f}, epoch spread {sxb/slope_t:.2f} dex, errors {np.median(sm):.2f}/0.10 dex): median R_d/r_c {np.median(_qm):.3f}, "
       f"scatter {np.std(np.log10(_qm)):.3f} dex, r={np.corrcoef(_xm,_ym)[0,1]:.2f}, bisector {_bis(_xm,_ym):.2f}; "
       f"implied f_j = {np.median(qd_)/np.median(_qm):.2f}")
# null model: core a FIXED fraction of the front, r_c = R_b/y0 (y0 = model y at median spin and z_end), same spins,
# epochs and errors; disc as before, R_d = f_j lambda R_b/sqrt2. Shares all v-scalings; differs in the spin dependence.
_y0 = _explicit(_tz, LAMSPIN)[0]/_explicit(_tz, LAMSPIN)[1]
_rcn = np.array([_explicit(t_, LAMSPIN)[0]/_y0 for t_ in _t])*10**(np.median(sm)*_rngm.standard_normal(_N))
_qn = _Rd/_rcn
_fjm, _fjn = np.median(qd_)/np.median(_qm), np.median(qd_)/np.median(_qn)
_lim = 1.27
report(f"  null (r_c = R_b/{_y0:.1f}): median R_d/r_c {np.median(_qn):.3f}, scatter {np.std(np.log10(_qn)):.3f} dex, implied f_j {_fjn:.2f}; "
       f"fraction above {_lim} f_j: data {np.mean(qd_ > _lim*_fjm):.2f} (model f_j) / {np.mean(qd_ > _lim*_fjn):.2f} (null f_j), "
       f"model MC {np.mean(_fjm*_qm > _lim*_fjm):.2f}, null MC {np.mean(_fjn*_qn > _lim*_fjn):.2f}; "
       f"KS data vs model p={_st.ks_2samp(lqd_, np.log10(_fjm*_qm)).pvalue:.2f}, vs null p={_st.ks_2samp(lqd_, np.log10(_fjn*_qn)).pvalue:.2f}")
# ---- compact two-column table of the total-matter fits (graduate paper)
def _cells(g):
    if g is None: return "&&&&&"
    return (f"{g['name']} & {g['n']} & ${g['v']:.1f}\\pm{g['ev']:.1f}$ & ${g['rc']:.2f}\\pm{g['erc']:.2f}$ & "
            f"{g['rho']:.3f} & {g['chi']:.2f}")
_h = (len(SPF)+1)//2; _L, _R = SPF[:_h], SPF[_h:]+[None]*(2*_h-len(SPF))
with open("tab_sparc79_grad.tex", "w") as fh:
    for a, b in zip(_L, _R): fh.write(_cells(a)+" & "+_cells(b)+" \\\\\n")
# ---- compactness C (Parodi 2018): correlation with v/r_c, and R_d/r_c independent of C
_lC = np.log10(Cc); _S = np.log10(A_("v")/A_("rc"))
_p1 = _st.pearsonr(_lC, _S); _p2 = _st.pearsonr(mu0, _S)
_r1 = _st.pearsonr(_lC, _S-np.polyval(np.polyfit(mu0, _S, 1), mu0)); _r2 = _st.pearsonr(mu0, _S-np.polyval(np.polyfit(_lC, _S, 1), _lC))
_sc = _st.spearmanr(Cc, qd_); _cm = _st.spearmanr(Cc, mu0)
report(f"compactness: log(v/r_c) vs log C r={_p1[0]:+.2f} p={_p1[1]:.1e}; vs mu0 r={_p2[0]:+.2f} p={_p2[1]:.1e}; C vs mu0 Spearman {_cm[0]:+.2f}")
report(f"  partial: C after mu0 r={_r1[0]:+.2f} p={_r1[1]:.2f}; mu0 after C r={_r2[0]:+.2f} p={_r2[1]:.2f}; "
       f"median v/r_c HSB {np.median((A_('v')/A_('rc'))[Cc>1]):.1f}, LSB {np.median((A_('v')/A_('rc'))[Cc<=1]):.1f}; R_d/r_c vs C Spearman {_sc[0]:+.2f} p={_sc[1]:.2f}")
# ---- galaxy-by-galaxy inversion of the core and disc relations for (lambda, t_acc) with fixed f_j, eps = 0
def _invert(v_, rc_, Rd_, fj):
    def mod(tg, lam):
        t = tg/GYR; l0 = LAM3*(t/T0_)**2; Rb_ = v_*t/(T0_*(1+CL_*l0)); ell_ = LAM3*Rb_**2/v_**2
        A_ = 3*lam**2+2/3*ell_; B_ = 0.25-CE-0.75*lam**2
        return Rb_, Rb_*4*A_/lambertw(4*A_*np.exp(4*B_)).real
    def f(tg):
        Rb_ = mod(tg, 0.03)[0]; return mod(tg, np.sqrt(2)*Rd_/(fj*Rb_))[1]-rc_
    ts = np.linspace(0.05, t_of_z(0), 400); fv = [f(t) for t in ts]
    for i in reversed(range(len(ts)-1)):                 # latest root = physical (small-spin) branch
        if np.isfinite(fv[i]) and np.isfinite(fv[i+1]) and fv[i]*fv[i+1] < 0:
            tg = brentq(f, ts[i], ts[i+1]); Rb_ = mod(tg, 0.03)[0]; return np.sqrt(2)*Rd_/(fj*Rb_), z_of_t(tg)
    return None
import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore")
    for fj in [np.median(qd_)/np.median(_qm), 0.6]:
        _inv = [_invert(g["v"], g["rc"], r_, fj) for g, r_ in zip(SPF, Rd_)]; _ok = np.array([x for x in _inv if x is not None])
        report(f"inversion with f_j={fj:.2f}, eps=0: solved {len(_ok)}/79; lambda median {np.median(_ok[:,0]):.3f} (16/84 {np.percentile(_ok[:,0],16):.3f}/"
               f"{np.percentile(_ok[:,0],84):.3f}), sigma_ln lambda {np.std(np.log(_ok[:,0])):.2f}; z_acc median {np.median(_ok[:,1]):.2f}")
_lam = np.linspace(0.005, 0.5, 5000)
for ell_ in [0.0, 0.0067]:
    A_ = 3*_lam**2+2/3*ell_; B_ = 0.25-CE-0.75*_lam**2; _ly = _lam*lambertw(4*A_*np.exp(4*B_)).real/(4*A_); k_ = np.argmax(_ly)
    report(f"  bound: ell={ell_}: max lambda*y = {_ly[k_]:.3f} at lambda = {_lam[k_]:.3f} -> R_d/r_c <= {(_ly[k_]/np.sqrt(2)):.2f} f_j")
report("  largest R_d/r_c: " + ", ".join(f"{SPF[i]['name']} {qd_[i]:.2f} (r_c={SPF[i]['rc']:.2f})" for i in np.argsort(qd_)[-4:]))
# ---- A&A appendix table: total-matter fits plus disc scale length R_d and compactness C (Parodi 2018, Table B.1)
def _cells_aa(g, b):
    if g is None: return "&&&&&&&"
    return (f"{g['name']} & {g['n']} & ${g['v']:.1f}\\pm{g['ev']:.1f}$ & ${g['rc']:.2f}\\pm{g['erc']:.2f}$ & "
            f"{g['rho']:.3f} & {g['chi']:.2f} & {float(b['Rd']):.2f} & {float(b['C']):.2f}")
_PB = list(zip(SPF, B1)); _L2, _R2 = _PB[:_h], _PB[_h:]+[(None, None)]*(2*_h-len(_PB))
with open("tab_sparc79_aa.tex", "w") as fh:
    for (a, ab), (b, bb) in zip(_L2, _R2): fh.write(_cells_aa(a, ab)+" & "+_cells_aa(b, bb)+" \\\\\n")
# ---- per-galaxy table for the appendix
with open("tab_sparc79.tex", "w") as fh:
    for g in SPF:
        dB = np.log10(g["rhoB"])-2*np.log10(g["v"])-(aB+bB*np.log10(g["r0B"]))
        nm = g["name"]
        fh.write(f"{nm} & {g['n']} & ${g['v']:.1f}\\pm{g['ev']:.1f}$ & ${g['rc']:.2f}\\pm{g['erc']:.2f}$ & "
                 f"{g['rho']:.4f} & {g['chi']:.2f} & {g['r0B']:.2f} & {g['rhoB']:.4f} & {g['Y']:g} & ${dB:+.2f}$ \\\\\n")
with open(os.path.join(SP, "sparc79_fits.csv"), "w") as fh:
    w = csv.writer(fh); keys = list(SPF[0]); w.writerow(keys)
    for g in SPF: w.writerow([g[k] if isinstance(g[k], str) else f"{g[k]:.5g}" for k in keys])

# ================= PHASE I PROFILE (Eq. 8 of the graduate paper), Lambda = 0, self-similar =================
from scipy.optimize import brentq as _bq
_y0, _xb0 = 50.5429, 0.174047
_etab = 2*np.pi-np.arccos(1/(4*_y0+1))
def _xi(e): return (1-np.cos(e))/(e-np.sin(e))
def _D(e): return 1-np.sin(e)*(e-np.sin(e))/(1-np.cos(e))**2
def _P_out(e): return (e-np.sin(e))**2/(4*np.pi*(1-np.cos(e))**3*_D(e))
def _P_mean(e): return 3*(e-np.sin(e))**2/(4*np.pi*(1-np.cos(e))**3)
def _eta_of_xi(xi): return _bq(lambda e: _xi(e)-xi, 1e-6, _etab)
def _P_in(xi):
    x = _y0*xi/_xb0; return (1-(1-x)*np.exp(-x))**2/(4*np.pi*xi**2)
Mtot0 = 7.57e10                                   # seed mass of the example galaxy for Lambda = 0
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.4))
for tg, c in zip([0.5, 1, 2, 4, 8], [C4, C3, C5, C2, C1]):
    t = tg/GYR; Rb = _xb0*V*t
    eta_e = _bq(lambda e: G*Mtot0/V**3*(e-np.sin(e))-t, 1e-9, 2*np.pi-1e-9)
    Re = G*Mtot0/V**2*(1-np.cos(eta_e)) if eta_e < _etab else Rb
    R = np.logspace(-2.3, np.log10(600), 900); rho = np.empty_like(R)
    for i, r in enumerate(R):
        xi = r/(V*t)
        if r <= Rb: rho[i] = _P_in(xi)/(G*t**2)
        elif r <= Re: rho[i] = _P_out(_eta_of_xi(xi))/(G*t**2)
        else: rho[i] = 1/(6*np.pi*G*t**2)
    ax[0].loglog(R, rho/1e9, c=c, lw=1.5, label=f"$t={tg:g}$ Gyr")
    ax[0].plot([Re], [_P_out(_eta_of_xi(Re/(V*t)))/(G*t**2)/1e9] if Re > Rb else [], "o", c=c, ms=3)
ax[0].set_xlabel(r"areal radius $R$ [kpc]"); ax[0].set_ylabel(r"$\rho$ [M$_\odot$ pc$^{-3}$]")
ax[0].set_xlim(5e-3, 600); ax[0].legend(frameon=False, fontsize=8, loc="lower left")
xs_in = np.logspace(-4, np.log10(_xb0), 300); xs_out = np.logspace(np.log10(_xb0*1.0001), 2, 400)
e_out = np.array([_eta_of_xi(x) for x in xs_out])
ax[1].loglog(xs_in, _P_in(xs_in), c=C1, lw=1.8, label="Phase II: P18 profile")
ax[1].loglog(xs_out, _P_out(e_out), c=C2, lw=1.8, label="Phase I: local density, Eq. (9)")
ax[1].loglog(xs_out, _P_mean(e_out), c=C2, lw=1.2, ls="--", label="Phase I: mean density inside $R$")
ax[1].axhline(1/(6*np.pi), c="0.5", lw=0.8, ls=":", label=r"background $1/(6\pi)$")
ax[1].plot([_xb0, _xb0], [_P_out(_etab), _P_in(_xb0)], c="k", lw=0.8)
ax[1].axvline(2/np.pi, c="0.6", lw=0.7); ax[1].text(2/np.pi*1.12, 0.4, "turnaround", fontsize=7.5, color="0.4", rotation=90, va="bottom")
ax[1].text(_xb0*0.85, 1.2e0, "front", fontsize=7.5, color="0.2", ha="right")
ax[1].set_xlabel(r"$\xi=R/vt$"); ax[1].set_ylabel(r"$P(\xi)=G\,t^2\rho$")
ax[1].set_xlim(1e-3, 100); ax[1].set_ylim(1e-2, 1e5); ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("fig15_phase1_profile.png", dpi=DPI)
# letter version (Fig. A.1 of the letter): labels without the long paper's phases and equation numbers
for _t, _l in zip(ax[1].get_legend().get_texts(), ["inside the front: P18 profile", "outside: local density (A.8)",
                                                   "outside: mean density inside $R$", r"background $1/(6\pi)$"]):
    _t.set_text(_l)
fig.savefig("fig_selfsim.png", dpi=DPI); plt.close(fig)
report(f"Phase I profile: P jump at front = {_P_in(_xb0)/_P_out(_etab):.3f}, P_out(xi->inf)*6pi = {_P_out(1e-4)*6*np.pi:.4f}")

# ================= OUT-OF-SAMPLE TEST: SPARC galaxies NOT used to define the P18 profile =================
# The P18 profile was introduced on the 79 galaxies above. Here the same fits are applied to the 94 SPARC
# galaxies with Hubble type T <= 7 (types from SPARC Table 1, Lelli et al. 2016; file sparc_oos_types.csv),
# rotation curves in sparc_data/rotmod_oos/. Same error model, same held-out inner-gradient test.
from scipy.optimize import brentq as _brq
def _fit18(R, V, e):
    res = lambda p: (np.exp(p[0])*_vn(R/np.exp(p[1]))-V)/e
    s = min((least_squares(res, [np.log(V.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    chi = 2*s.cost/max(len(R)-2, 1); cov = np.linalg.pinv(s.jac.T@s.jac)*max(chi, 1.0)
    vv, rr = np.exp(s.x); return vv, rr, chi, rr*np.sqrt(cov[1, 1])
def _oos_one(path):
    d = np.loadtxt(path, ndmin=2); R, V = d[:, 0], d[:, 1]; e = np.sqrt(d[:, 2]**2+(0.03*V)**2+1.0)
    vv, rr, chi, er = _fit18(R, V, e); g = dict(v=vv, rc=rr, chi=chi, erc=er, bul=np.max(d[:, 5]) > 0,
                                                D=V/vv-_vn(R/rr), eV=e/vv, hr=np.nan)
    if len(R) >= 5:
        vh, rh, _, _ = _fit18(R[2:], V[2:], e[2:]); r2 = R[:2]
        g["hr"] = np.log10(np.sum(r2*V[:2])/np.sum(r2*vh*_vn(r2/rh)))
    return g
_OT = [r for r in csv.DictReader(open(os.path.join(SP, "sparc_oos_types.csv"))) if int(r["T"]) <= 7]
_OO = []
for r in _OT:
    g = _oos_one(os.path.join(SP, "rotmod_oos", r["name"]+"_rotmod.dat")); g["T"] = int(r["T"]); g["name"] = r["name"]; _OO.append(g)
_IN = [_oos_one(os.path.join(SP, "rotmod", g["name"]+"_rotmod.dat")) for g in SPF]
# competing cored/cuspy two-parameter profiles on the out-of-sample galaxies (same fits, same errors; equal numbers
# of parameters, so Delta BIC = Delta chi^2 for each galaxy)
for _lbl, _sel in [("T<=7 no bulge", lambda r, g: not g["bul"]), ("T<=7 with bulge", lambda r, g: g["bul"])]:
    _C2 = {k: [] for k in _alt}; _R2 = {k: [] for k in _alt}; _dof2 = []
    for r, g in zip(_OT, _OO):
        if not _sel(r, g): continue
        d = np.loadtxt(os.path.join(SP, "rotmod_oos", r["name"]+"_rotmod.dat"), ndmin=2)
        R_, V_, e_ = d[:, 0], d[:, 1], np.sqrt(d[:, 2]**2+(0.03*d[:, 1])**2+1.0); _dof2.append(len(R_)-2)
        for k, f in _alt.items():
            _C2[k].append(2*_fit2(f, R_, V_, e_).cost)
            if len(R_) >= 5:
                a_, s_ = np.exp(_fit2(f, R_[2:], V_[2:], e_[2:]).x); r2 = R_[:2]
                _R2[k].append(np.log10(np.sum(r2*V_[:2])/np.sum(r2*a_*f(r2/s_))))
    _CC = np.array([_C2[k] for k in _alt]); _dof2 = np.array(_dof2); _n = _CC.shape[1]
    report(f"  OOS profile comparison ({_lbl}, N={_n}, {_dof2.sum()} dof):")
    for i, k in enumerate(_alt):
        _dl = _CC[i]-_CC[0]
        report(f"    {k:8s} sum chi2 = {_CC[i].sum():7.1f}, median chi2/dof {np.median(_CC[i]/_dof2):.2f}, lowest in {np.sum(np.argmin(_CC, 0) == i)}/{_n}; "
               f"median Delta chi2 vs P18 {np.median(_dl):+.2f}, P18 better by >2 in {np.sum(_dl > 2)}, worse by >2 in {np.sum(_dl < -2)}; by >6: {np.sum(_dl > 6)}/{np.sum(_dl < -6)}; "
               f"held-out gradient obs/pred {10**np.median(_R2[k]):.2f}, rms {np.std(_R2[k]):.3f} dex")
def _zrcv(q):                                            # end of growth (median spin, no feedback) for a given r_c/v
    f = lambda z: _explicit(t_of_z(z), LAMSPIN)[1]/V-q
    return _brq(f, 0.0, 6.0) if f(0.0) > 0 > f(6.0) else np.nan
_rows = []
for lab, S in [("Defining sample ($T\\ge8$)", _IN),
               ("$T\\le7$, all", _OO),
               ("$T\\le7$, no bulge", [g for g in _OO if not g["bul"]]),
               ("$T\\le7$, no bulge, $v\\le130$", [g for g in _OO if not g["bul"] and g["v"] <= 130]),
               ("$T\\le7$, with bulge", [g for g in _OO if g["bul"]])]:
    D = np.concatenate([g["D"] for g in S]); E = np.concatenate([g["eV"] for g in S]); hr = np.array([g["hr"] for g in S])
    hr = hr[np.isfinite(hr)]; q = np.array([g["v"]/g["rc"] for g in S])
    _rows.append((lab, len(S), np.median([g["v"] for g in S]), np.sqrt(np.mean(D**2)), np.median(E),
                  np.mean([g["chi"] > 2 for g in S]), 10**np.median(hr), np.std(hr), np.median(q)))
    report(f"OOS {lab}: N={len(S)}, v_med={_rows[-1][2]:.0f}, rms={_rows[-1][3]:.3f}, err_med={_rows[-1][4]:.3f}, "
           f"frac chi2r>2={_rows[-1][5]:.2f}, held-out S_in/S_pred={_rows[-1][6]:.2f} rms {_rows[-1][7]:.3f} dex, v/r_c={_rows[-1][8]:.1f}")
with open("tab_oos_aa.tex", "w") as fh:                  # rows of Table D.1
    for r in _rows:
        fh.write(f"{r[0]} & {r[1]} & {r[2]:.0f} & {r[3]:.3f} & {r[4]:.3f} & {100*r[5]:.0f}\\% & {r[6]:.2f} & {r[7]:.2f} & {r[8]:.0f}\\\\\n")
# matched velocity range: bulgeless late-type spirals vs the defining sample, 40 <= v <= 130 km/s, r_c to better than 50%
_a = [g for g in _IN if 40 <= g["v"] <= 130 and g["erc"]/g["rc"] < 0.5]
_b = [g for g in _OO if 40 <= g["v"] <= 130 and not g["bul"] and g["erc"]/g["rc"] < 0.5]
_qa, _qb = np.array([g["rc"]/g["v"] for g in _a]), np.array([g["rc"]/g["v"] for g in _b])
_vb, _rb = np.array([g["v"] for g in _b]), np.array([g["rc"] for g in _b])
_bs = [np.polyfit(np.log10(_vb[i]), np.log10(_rb[i]), 1)[0] for i in rng.integers(0, len(_vb), (2000, len(_vb)))]
report(f"OOS matched v 40-130: defining N={len(_a)} r_c/v med {np.median(_qa):.4f} (z_end {_zrcv(np.median(_qa)):.2f}); "
       f"bulgeless T<=7 N={len(_b)} r_c/v med {np.median(_qb):.4f} (z_end {_zrcv(np.median(_qb)):.2f}); ratio {np.median(_qa)/np.median(_qb):.2f}; "
       f"Mann-Whitney p={_st.mannwhitneyu(np.log10(_qa), np.log10(_qb)).pvalue:.1e}; T<=7 slope r_c-v {np.polyfit(np.log10(_vb), np.log10(_rb), 1)[0]:.2f} +- {np.std(_bs):.2f}; "
       f"T of these: {np.bincount([g['T'] for g in _b], minlength=8)[:8].tolist()}")

# inner gradient v/r_c against v over all SPARC galaxies fitted (79 dwarfs + 94 T<=7), r_c to better than 50%
_AL = [g for g in _IN+_OO if g["erc"]/g["rc"] < 0.5]
_va = np.array([g["v"] for g in _AL]); _qa_ = _va/np.array([g["rc"] for g in _AL])
_ind = [g for g in _IN if g["erc"]/g["rc"] < 0.5]; _vd = np.array([g["v"] for g in _ind]); _qd = _vd/np.array([g["rc"] for g in _ind])
report("v/r_c vs v (all fitted SPARC galaxies): " + "; ".join(
    f"{lo}-{hi}: N={np.sum((_va>=lo)&(_va<hi))} median {np.median(_qa_[(_va>=lo)&(_va<hi)]):.0f}" for lo, hi in [(20, 60), (60, 100), (100, 150), (150, 200), (200, 400)])
    + f"; Spearman all {_st.spearmanr(_va, _qa_).statistic:+.2f} (p={_st.spearmanr(_va, _qa_).pvalue:.0e}), "
    f"dwarfs {_st.spearmanr(_vd, _qd).statistic:+.2f}, slope dwarfs {np.polyfit(np.log10(_vd), np.log10(_qd), 1)[0]:+.2f}")

# ================= DISSIPATION BY THE BARYONS vs FEEDBACK (energy budget of the core) =================
# Gas that cools into a disc of scale R_d ends deeper in the potential than the same mass would after
# dissipationless settling (mean radius R_b/e for M ~ R). Radiated energy ~ kappa v^2 per unit baryon mass,
# kappa = ln(R_b/(e R_d)); in units of the binding energy (1/2) v^2 M_tot: eps_diss = -2 kappa M_bar/M_tot.
# M_bar: stars (Upsilon_3.6 = 0.5) + gas enclosed at the last point (from V_gas; a lower bound).
# M_tot = (v^2/G) R_b (1+1/4y), R_b = y r_c with y for median spin and growth ending at z = 0.8.
_Rb8, _rc8 = _explicit(t_of_z(0.8), LAMSPIN); _y8 = _Rb8/_rc8
_Ms, _Mg = [], []
for g, b in zip(SPF, B1):
    _Ms.append(0.5*10**(-0.4*(float(b["M36"])-3.24)))
    d = np.loadtxt(os.path.join(SP, "rotmod", g["name"]+"_rotmod.dat"), ndmin=2)
    _Mg.append(max(d[-1, 3]*abs(d[-1, 3]), 0)*d[-1, 0]/G)
_Ms, _Mg = np.array(_Ms), np.array(_Mg); _Mb = _Ms+_Mg
_vv = np.array([g["v"] for g in SPF]); _rr = np.array([g["rc"] for g in SPF])
_Mt = _vv**2*_y8*_rr/G*(1+1/(4*_y8))
_kap = np.log(_y8*_rr/(np.e*Rd_)) - (1 - np.euler_gamma); _fb = _Mb/_Mt   # exponential disc: <ln R> = ln R_d + 1 - gamma
_ed = -2*_kap*_fb                                   # dissipation
_ef = 2*(710/_vv)**2*_Ms/_Mt                    # feedback per unit coupling f_c (1e51 erg per 100 Msun of stars)
_pc = lambda a: np.percentile(a, [16, 50, 84])
_Alim = np.exp(-1-4*(0.25-CE-0.75*LAMSPIN**2))/4
report(f"Dissipation (y={_y8:.1f}, z_end=0.8): M_bar/M_tot 16/50/84 = {_pc(_fb).round(4)}; kappa = {_pc(_kap).round(2)}; "
       f"eps_diss = {_pc(_ed).round(3)}; eps_diss/(3 lam^2) median = {np.median(-_ed)/(3*LAMSPIN**2):.0f}")
report(f"  feedback per unit f_c: eps_fb = {_pc(_ef).round(3)}; coupling for balance f_c = {_pc(-_ed/_ef).round(3)}; "
       f"slopes vs v: |eps_diss| {np.polyfit(np.log10(_vv), np.log10(-_ed), 1)[0]:+.2f}, eps_fb/f_c {np.polyfit(np.log10(_vv), np.log10(_ef), 1)[0]:+.2f}")
report(f"  core equation solvable for A<0 only if |A| <= e^(-1-4B)/4 = {_Alim:.2e}: net loss <= 3lam^2+{_Alim:.5f} = {3*LAMSPIN**2+_Alim:.4f}; y_max there = {1/(4*_Alim):.0f}")

# ---- "growing" individual galaxies (Fig. 3): uses the functions of recipe.py (Appendix C)
import recipe as _g
def _core(v, lam, tg): return _g.grow(v, lam, z_of_t(tg))[0]   # tg in Gyr
def _vc(v, rc, R): return _g.irc(np.maximum(R, 1e-9), v, rc)
_fits = {r["name"]: r for r in csv.DictReader(open("sparc_data/sparc79_fits.csv"))}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
T0 = t_of_z(0); _tt = np.linspace(t_of_z(15), T0, 300)
for k, (gname, col, zalt) in enumerate([("DDO154", C1, 1.0), ("NGC2915", C2, 2.0)]):
    g = _fits[gname]; v_ = float(g["v"]); rc_ = float(g["rc"]); erc = float(g["erc"])
    zacc = brentq(lambda z: _g.grow(v_, LAMSPIN, z)[0]-rc_, 0, 10); tacc = t_of_z(zacc)
    d = np.loadtxt(f"sparc_data/rotmod/{gname}_rotmod.dat")
    Rm = np.linspace(0, d[:, 0].max()*1.05, 300)
    for zz in [6, 2, 1]:
        if zz > zacc: ax[k].plot(Rm, _vc(v_, _core(v_, LAMSPIN, t_of_z(zz)), Rm), c="0.55", lw=0.8, ls="--")
    ax[k].plot(Rm, _vc(v_, rc_, Rm), c=col, lw=1.8)
    ax[k].errorbar(d[:, 0], d[:, 1], d[:, 2], fmt="o", ms=2.5, c="k", lw=0.7, elinewidth=0.7)
    chi = np.mean(((d[:, 1]-_vc(v_, rc_, d[:, 0]))/np.maximum(d[:, 2], 1))**2)
    ax[k].set_title(f"{gname[:3]} {gname[3:]}: growth ends at $z={zacc:.2f}$", fontsize=8.5)
    ax[k].set_xlabel(r"$R$ [kpc]"); ax[k].set_ylim(0, None); ax[k].set_xlim(0, Rm[-1])
    ax[k].text(0.97, 0.05, "dashed: $z=" + ", ".join(str(z) for z in [6, 2, 1] if z > zacc) + "$", transform=ax[k].transAxes, ha="right", fontsize=7.5, color="0.4")
    tg = _tt[_tt <= tacc]; ax[2].plot(tg, [_core(v_, LAMSPIN, t) for t in tg], c=col, lw=1.8, label=f"{gname[:3]} {gname[3:]}")
    ax[2].plot([tacc, T0], [rc_, rc_], c=col, lw=1.2, ls=":")
    ax[2].axhspan(rc_-erc, rc_+erc, color=col, alpha=0.12, lw=0)
    # degeneracy (Sect. 5): growth ending earlier, at zalt, plus feedback eps rising linearly in time
    # to the value eps_f that gives today the same core (Eq. 9)
    tb = t_of_z(zalt); ef = brentq(lambda e: _g.grow(v_, LAMSPIN, zalt, e)[1]-rc_, 0, 0.2)
    _ta = np.linspace(tb, T0, 200)
    ax[2].plot(_ta, [_g.grow(v_, LAMSPIN, zalt, ef*(t-tb)/(T0-tb))[1] for t in _ta], c=col, lw=1.2, ls="-.")
    report(f"degeneracy: {gname} growth ending at z={zalt} (r_c={_g.grow(v_, LAMSPIN, zalt)[0]:.2f}) plus eps rising linearly to {ef:.4f} gives r_c={rc_:.2f} today")
    report(f"grown galaxies: {gname} v={v_:.1f} rc={rc_:.2f} -> z_acc={zacc:.3f} (t={tacc:.2f} Gyr), chi2/N of IRC={chi:.2f}")
ax[0].set_ylabel(r"$v_c$ [km s$^{-1}$]")
ax[2].set_xlabel(r"cosmic time $t$ [Gyr]"); ax[2].set_ylabel(r"$r_c$ [kpc]"); ax[2].set_xlim(0, T0); ax[2].set_ylim(0, None)
ax[2].legend(frameon=False, fontsize=7.5, loc="lower right"); ax[2].set_title(r"core growth, $\lambda=0.035$", fontsize=8.5)
fig.tight_layout(); fig.savefig("fig18_grow.png", dpi=DPI); plt.close(fig)

# ---- resolution and distance (referee point): does r_c depend on distance or on the innermost radius at fixed v?
import re as _re
from scipy import stats as _sts
_rows = []
for r in csv.DictReader(open("sparc_data/sparc79_fits.csv")):
    f = f"sparc_data/rotmod/{r['name']}_rotmod.dat"
    _D = float(_re.search(r"Distance = ([0-9.]+)", open(f).read()).group(1)); _d = np.loadtxt(f)
    _rows.append((float(r["v"]), float(r["rc"]), _D, _d[0, 0]))
_v, _rc, _Dd, _R1 = map(np.array, zip(*_rows)); _lv, _lr = np.log10(_v), np.log10(_rc); _q = _lv - _lr
_rg = np.random.default_rng(1)
_X = np.column_stack([np.ones_like(_lv), _lv, np.log10(_Dd)]); _c = np.linalg.lstsq(_X, _lr, rcond=None)[0]
_bs = np.array([np.linalg.lstsq(_X[i], _lr[i], rcond=None)[0] for i in (_rg.integers(0, 79, 79) for _ in range(3000))]).std(0)
report(f"resolution/distance: Spearman log(v/rc)-D {_sts.spearmanr(_q, _Dd)[0]:+.2f} (p={_sts.spearmanr(_q, _Dd)[1]:.3f}), "
       f"-R1 {_sts.spearmanr(_q, _R1)[0]:+.2f} (p={_sts.spearmanr(_q, _R1)[1]:.3f}); log rc = a + b log v + c log D: "
       f"b={_c[1]:.2f}+-{_bs[1]:.2f}, c={_c[2]:.2f}+-{_bs[2]:.2f}; fraction rc<2R1 {np.mean(_rc < 2*_R1):.2f}")
for _lbl, _m in [("all", np.ones(79, bool)), ("rc>2R1", _rc > 2*_R1), ("rc>3R1", _rc > 3*_R1), ("D<10Mpc", _Dd < 10), ("rc>2R1&D<10", (_rc > 2*_R1) & (_Dd < 10))]:
    _b = np.polyfit(_lv[_m], _lr[_m], 1)[0]
    _bb = np.std([np.polyfit(_lv[_m][i], _lr[_m][i], 1)[0] for i in (_rg.integers(0, _m.sum(), _m.sum()) for _ in range(2000))])
    report(f"  {_lbl}: N={_m.sum()} r_c ~ v^{_b:.2f}+-{_bb:.2f}, median v/rc {np.median(10**_q[_m]):.1f}, std log(v/rc) {_q[_m].std():.3f}, "
           f"Spearman q-D {_sts.spearmanr(_q[_m], _Dd[_m])[0]:+.2f}")

# ML versions of the subsample slopes (same likelihood as the main fit), so that the systematic range is on the same footing
_names_fit = [r["name"] for r in csv.DictReader(open("sparc_data/sparc79_fits.csv"))]
def _mlreg3(x, y, z, sx, sy, cxy):
    def nl(p):
        a_, b_, c_, ls_ = p; s2 = sy**2+b_**2*sx**2-2*b_*cxy+np.exp(2*ls_)
        return 0.5*np.sum((y-a_-b_*x-c_*z)**2/s2+np.log(s2))
    from scipy.optimize import minimize as _mn
    return _mn(nl, [0.0, 1.0, 0.0, np.log(0.2)], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=8000)).x
_zD = np.log10(_Dd)-np.median(np.log10(_Dd))
_pc = _mlreg3(_xr, _yr, _zD, _cl(_sx), _cl(_sy), _cxy)
_bc = np.array([_mlreg3(_xr[i_], _yr[i_], _zD[i_], _cl(_sx)[i_], _cl(_sy)[i_], _cxy[i_])[1:3] for i_ in (rng.integers(0, 79, 79) for _ in range(200))])
report(f"ML with log D covariate: b={_pc[1]:.2f}+-{_bc[:,0].std():.2f}, c={_pc[2]:.2f}+-{_bc[:,1].std():.2f}")
for _lbl, _m in [("all", np.ones(79, bool)), ("rc>2R1", _rc > 2*_R1), ("rc>3R1", _rc > 3*_R1), ("D<10Mpc", _Dd < 10)]:
    _ix = np.where(_m)[0]
    _pm = _mlreg(_xr[_ix], _yr[_ix], _cl(_sx)[_ix], _cl(_sy)[_ix], _cxy[_ix])
    _bm = np.std([_mlreg(_xr[_ix][i_], _yr[_ix][i_], _cl(_sx)[_ix][i_], _cl(_sy)[_ix][i_], _cxy[_ix][i_])[1] for i_ in (rng.integers(0, len(_ix), len(_ix)) for _ in range(200))])
    report(f"  ML {_lbl}: N={len(_ix)} r_c ~ v^{_pm[1]:.2f}+-{_bm:.2f}, intrinsic {np.exp(_pm[2]):.2f} dex; constant Sigma excluded at {(2-_pm[1])/_bm:.1f} sigma")
# ---- radial acceleration relation (McGaugh et al. 2016) and the asymptotic excess mass of the profile (test iv)
# RAR: g_obs = g_bar/(1-exp(-sqrt(g_bar/g_dag))); for g_bar << g_dag, g_obs = sqrt(g_bar g_dag) + g_bar/2 + ...,
# so with the baryons enclosed M(R) = v^2 R/G + M_b/2 (v^4 = G M_b g_dag): excess c > 0. If M_b/2 = v^2 r_c/4G,
# then r_c = 2 v^2/g_dag and Sigma = v^2/(pi G r_c) = g_dag/(2 pi G).
_AA = lambda k_: np.array([g_[k_] for g_ in SPF])
_gdag = 1.2e-10*3.0857e19/1e6                                   # (km/s)^2/kpc
def _mcfree(x, c): return x*(1-np.exp(-x))**2 + c*(1-(1+2*x+2*x*x)*np.exp(-2*x))
def _fitc(R, V, e):
    best = None
    for rc0 in [0.3, 1, 3, 10]:
        for c0 in [0.25, -0.3, 1.0]:
            res = lambda p: (np.exp(p[0])*np.sqrt(np.maximum(_mcfree(R/np.exp(p[1]), p[2]), 1e-12)/(R/np.exp(p[1])))-V)/e
            s_ = least_squares(res, [np.log(max(V.max(), 5)), np.log(rc0), c0], bounds=([-5, -7, -0.7], [12, 9, 5]))
            if best is None or s_.cost < best.cost: best = s_
    try: return best.x[2], np.sqrt(np.linalg.inv(best.jac.T@best.jac)[2, 2]*max(1, 2*best.cost/max(1, len(R)-3)))
    except Exception: return best.x[2], np.nan
_yl, _xl, _co, _cr = [], [], [], []
for g_ in SPF:
    d_ = np.loadtxt(os.path.join(SP, "rotmod", g_["name"]+"_rotmod.dat"), ndmin=2)
    R_, V_, e0_, Vg_, Vd_, Vb_ = d_[:, 0], d_[:, 1], d_[:, 2], d_[:, 3], d_[:, 4], d_[:, 5]; e_ = np.sqrt(e0_**2+(0.03*V_)**2+1.0)
    gb_ = np.maximum((np.abs(Vg_)*Vg_+0.5*Vd_**2+0.7*Vb_**2)/R_, 1e-6); Vr_ = np.sqrt(gb_/(1-np.exp(-np.sqrt(gb_/_gdag)))*R_)
    _yl.append(gb_[-1]/_gdag); _xl.append(R_[-1]/g_["rc"]); _co.append(_fitc(R_, V_, e_)); _cr.append(_fitc(R_, Vr_, e_))
_co, _cr = np.array(_co), np.array(_cr)
_q = (_AA("v")**2/(2*_gdag))/(_AA("rc")/4)
report(f"RAR: outer g_bar/g_dag median {np.median(_yl):.3f}; R_last/r_c median {np.median(_xl):.2f}; Sigma_M = g_dag/(2 pi G) = {_gdag/(2*np.pi*G)/1e6:.1f} Msun/pc^2; "
       f"RAR excess (M_b = v^4/G g_dag) / (v^2 r_c/4G): median {np.median(_q):.2f} (16/84 {np.percentile(_q,16):.2f}/{np.percentile(_q,84):.2f})")
report(f"  free-c fits: observed curves median c {np.median(_co[:,0]):+.2f}, median error {np.nanmedian(_co[:,1]):.1f}; "
       f"RAR curves median c {np.median(_cr[:,0]):+.2f}, median error {np.nanmedian(_cr[:,1]):.1f}")
_cS = np.polyfit(np.log10(_AA("v")), np.log10(_AA("v")**2/(np.pi*G*_AA("rc"))/1e6), 1)
report(f"  Sigma(v) from LSQ line at v = 40, 73, 130 km/s: " + ", ".join(f"{10**np.polyval(_cS, np.log10(x_)):.0f}" for x_ in (40, 73, 130)))
with open("results.txt", "w") as fh: fh.write("\n".join(out)+"\n")
print("done")
