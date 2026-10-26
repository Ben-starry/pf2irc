"""Companion note "Top hats and the nonlinear equation for delta".
Checks (i) that the cycloid solves the nonlinear top-hat equation in Einstein-de Sitter,
(ii) that the shell equation with Lambda and the delta equation (Herrera et al. eq. 10, Lambda case)
give the same evolution in a flat LCDM background, and (iii) draws the mean overdensity of the
nested top hats of the pf2irc seed.  Units: H0 = 1 (time in 1/H0), G = 1; Lambda in 1/time^2.
python3 tophat.py   -> fig_tophat.png, tophat_output.txt"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

out = open("tophat_output.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

# ---------------------------------------------------------------- (i) Einstein-de Sitter
# cycloid of a top hat: R = A(1-cos eta), t = B(eta - sin eta); 1+delta = 9(eta-sin)^2/(2(1-cos)^3)
def _u(eta):                     # eta - sin(eta), with its series for small eta (no cancellation)
    eta = np.asarray(eta, float)
    return np.where(eta < 1e-2, eta**3/6 - eta**5/120, eta - np.sin(eta))
def _w(eta):                     # 1 - cos(eta) = 2 sin^2(eta/2)
    return 2*np.sin(np.asarray(eta, float)/2)**2
def one_plus_delta(eta):
    return 9*_u(eta)**2/(2*_w(eta)**3)
def ddelta_dt(eta):              # d delta/dt = (d/d eta)(1+delta) / (dt/d eta), dt/d eta = 1 - cos eta (units of B)
    u, w = _u(eta), _w(eta)
    return 4.5*(2*u*w*w**3 - 3*u**2*w**2*np.sin(eta))/w**6/w
# delta equation in EdS (Omega_m = 1, H = 2/3t, 4 pi G rho_bar = 2/(3 t^2)), time in units of B
def rhs_eds(t, y):
    d, dd = y
    return [dd, -(4/(3*t))*dd + (4/3)*dd**2/(1+d) + (2/(3*t**2))*d*(1+d)]
eta0 = 0.05
t0 = float(_u(eta0))
d0 = float(one_plus_delta(eta0)) - 1
dd0 = float(ddelta_dt(eta0))
eta_end = 1.9*np.pi
t_end = eta_end - np.sin(eta_end)
sol = solve_ivp(rhs_eds, (t0, t_end), [d0, dd0], rtol=1e-11, atol=1e-14, dense_output=True)
assert sol.success, sol.message
eta_chk = np.linspace(0.3, eta_end, 400)
t_chk = eta_chk - np.sin(eta_chk)
err = np.max(np.abs((1 + sol.sol(t_chk)[0])/one_plus_delta(eta_chk) - 1))
say(f"EdS: max relative difference cycloid vs delta equation up to eta=1.9 pi: {err:.1e}")
for name, eta in [("turnaround", np.pi), ("halting", 1.5*np.pi)]:
    t = eta - np.sin(eta); dlin = 3/20*(6*t)**(2/3)
    say(f"  {name:10s} eta={eta/np.pi:.2f} pi: 1+delta = {one_plus_delta(eta):8.3f}, delta_lin = {dlin:.3f}")
say(f"  collapse   eta=2 pi:    1+delta -> inf,      delta_lin = {3/20*(12*np.pi)**(2/3):.3f}")
say(f"  halting value (9/2)(1+3pi/2)^2 = {4.5*(1+1.5*np.pi)**2:.2f}; top-hat virial 18 pi^2 = {18*np.pi**2:.1f}")

# ---------------------------------------------------------------- (ii) flat LCDM, Lambda in 1/time^2
Om, OL = 0.3, 0.7
Lam = 3*OL                      # Lambda = 3 H0^2 Omega_Lambda, H0 = 1
def a_of_t(t):
    return (Om/OL)**(1/3)*np.sinh(1.5*np.sqrt(OL)*t)**(2/3)
def H_of_t(t):
    return np.sqrt(OL)/np.tanh(1.5*np.sqrt(OL)*t)
def Om_of_t(t):
    a = a_of_t(t); return Om/a**3/(Om/a**3 + OL)
# shell: comoving radius 1, R_i = a_i (1+d_i)^(-1/3), G M = Om/2 (H0 = 1)
GM = Om/2
def rhs_R(t, y):
    R, V = y; return [V, -GM/R**2 + Lam*R/3]
def rhs_d(t, y):                # Herrera et al. (2019) eq. (10) with delta_de = 0 (Lambda)
    d, dd = y; H = H_of_t(t)
    return [dd, -2*H*dd + (4/3)*dd**2/(1+d) + 1.5*H**2*Om_of_t(t)*d*(1+d)]
ti = 1e-4
ai, Hi = a_of_t(ti), H_of_t(ti)
def run(di, tmax=8.0):
    ddi = Hi*di                              # growing mode, delta ~ a, early matter era
    Ri = ai*(1+di)**(-1/3)
    Vi = Ri*(Hi - ddi/(3*(1+di)))            # from delta-dot/(1+delta) = 3(H - Rdot/R)
    stop = lambda t, y: y[0] - 1e-3*ai; stop.terminal = True
    sR = solve_ivp(rhs_R, (ti, tmax), [Ri, Vi], rtol=1e-11, atol=1e-15, dense_output=True, events=stop)
    big = lambda t, y: y[0] - 1e5; big.terminal = True
    sd = solve_ivp(rhs_d, (ti, tmax), [di, ddi], rtol=1e-11, atol=1e-15, dense_output=True, events=big)
    return sR, sd
t0_LCDM = brentq(lambda t: a_of_t(t) - 1, 0.1, 3)
say(f"LCDM (Om=0.3): age t0 = {t0_LCDM:.4f}/H0, Lambda = {Lam:.2f} H0^2")
# threshold: the shell turns around only if  (Rdot^2/2 - GM/R - Lam R^2/6) <= -(3/2)(GM)^(2/3)(Lam/3)^(1/3)
def energy(di):
    Ri = ai*(1+di)**(-1/3); Vi = Ri*(Hi - Hi*di/(3*(1+di)))
    return 0.5*Vi**2 - GM/Ri - Lam*Ri**2/6
Ecrit = -1.5*GM**(2/3)*(Lam/3)**(1/3)
dthr = brentq(lambda d: energy(d) - Ecrit, 1e-6, 1e-2)
say(f"turnaround threshold: delta_i = {dthr:.4e} at a_i = {ai:.3e}  (delta_i/a_i = {dthr/ai:.4f});"
    f" maximum turnaround radius (3GM/Lambda)^(1/3) = {(3*GM/Lam)**(1/3):.4f} (comoving 1 -> a)")
cases = [("collapses early", 3.0*dthr), ("collapses late", 1.15*dthr), ("never turns around", 0.85*dthr)]
runs = []
for lab, di in cases:
    sR, sd = run(di)
    tt = np.linspace(ti*50, min(sR.t[-1], sd.t[-1])*0.999, 600)
    R = sR.sol(tt)[0]; dR = (a_of_t(tt)/R)**3 - 1
    dD = sd.sol(tt)[0]
    m = dD < 1e4
    rel = np.max(np.abs((1+dR[m])/(1+dD[m]) - 1))
    tta = sR.t[np.argmax(sR.y[0])] if sR.y[0].argmax() < len(sR.t)-1 else np.nan
    say(f"  {lab:20s} delta_i/delta_thr={di/dthr:.2f}: max rel. difference R-picture vs delta-picture {rel:.1e};"
        f" t_ta = {tta:.3f}/H0" + ("" if np.isfinite(tta) else " (none)"))
    runs.append((lab, tt, dR, dD))
# asymptote of the non-turning shell: delta tends to a constant
lab, tt, dR, dD = runs[2]
say(f"  non-turning shell: 1+delta at t = 4, 6, 8 /H0: " +
    ", ".join(f"{1+run(cases[2][1], tmax=T)[1].y[0][-1]:.3f}" for T in (4, 6, 8)))

# ---------------------------------------------------------------- (iii) nested top hats of the seed
# E = -v^2/2 for every shell: A = GM/v^2, B = GM/v^3; units G = v = 1, outermost mass 1
def eta_of(t, M):
    x = t/M
    if x >= 2*np.pi: return 2*np.pi
    return brentq(lambda e: e - np.sin(e) - x, 0, 2*np.pi)
Ms = np.geomspace(1e-3, 1, 400)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
eta = np.linspace(0.05, 1.97*np.pi, 800); tt = _u(eta); tta = np.pi
ax[0].semilogy(tt/tta, one_plus_delta(eta), "-", c="#1f4e79", lw=2.2, label=r"cycloid (exact)")
tn = np.linspace(t0*1.01, t_end, 60)
ax[0].semilogy(tn/tta, 1+sol.sol(tn)[0], "o", ms=3.5, mfc="none", c="#c0504d", label=r"$\delta$ equation, numerical")
ax[0].semilogy(tt/tta, 1+3/20*(6*tt)**(2/3), "--", c="0.45", lw=1.2, label=r"linear: $1+\frac{3}{20}(6t/B)^{2/3}$")
for e, s, y in [(np.pi, "turnaround\n$1+\\delta=5.55$", 14), (1.5*np.pi, "halting\n$1+\\delta=147$", 400)]:
    x = (e - np.sin(e))/tta; ax[0].axvline(x, c="0.7", lw=0.8, ls=":")
    ax[0].text(x - 0.03, y, s, fontsize=8, va="center", ha="right")
ax[0].axvline(2, c="0.7", lw=0.8, ls=":"); ax[0].text(1.93, 3e3, "collapse", fontsize=8, ha="right")
ax[0].set(xlabel=r"$t/t_{\rm ta}$", ylabel=r"$1+\langle\delta\rangle$", ylim=(0.8, 1e4), xlim=(0, 2.05))
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].set_title("(a) one top hat, Einstein-de Sitter", fontsize=10)
cols = ["#1f4e79", "#c0504d", "#4f8a3c"]
for (lab, tt2, dR, dD), c in zip(runs, cols):
    a2 = a_of_t(tt2)
    ax[1].loglog(a2, 1+dD, "-", c=c, lw=2, label=lab)
    k = np.linspace(0, len(tt2)-1, 40).astype(int)
    ax[1].loglog(a2[k], 1+dR[k], "o", ms=3.2, mfc="none", c=c)
ax[1].axvline(1, c="0.7", lw=0.8, ls=":"); ax[1].text(1.05, 1.3, "today", fontsize=8)
ax[1].set(xlabel="scale factor $a$", ylabel=r"$1+\delta$", ylim=(1, 1e4), xlim=(0.02, 30))
ax[1].legend(frameon=False, fontsize=8, loc="upper left",
             title=r"lines: $\delta$ equation; circles: shell equation", title_fontsize=8)
ax[1].set_title(r"(b) flat $\Lambda$CDM: two pictures, one solution", fontsize=10)
for T, c in zip([0.02, 0.1, 0.4, 1.0], ["#9bbbd9", "#5b8fc0", "#2d5f93", "#0b2e59"]):
    e = np.array([eta_of(T, M) for M in Ms])
    od = np.where(e < 1.5*np.pi, one_plus_delta(np.maximum(e, 1e-4)), np.nan)
    ax[2].loglog(Ms, od, "-", c=c, lw=1.8, label=fr"$t={T}\,GM_{{\rm out}}/v^3$")
    Mb = T/(1.5*np.pi + 1)
    if Mb > Ms[0]: ax[2].axvspan(Ms[0], Mb, color=c, alpha=0.08)
Mlin = np.geomspace(1e-3, 1, 10)
ax[2].loglog(Mlin, 1 + 3/20*(6*0.02/Mlin)**(2/3), "--", c="0.45", lw=1, label=r"linear at $t=0.02$, $\propto M^{-2/3}$")
ax[2].set(xlabel=r"shell mass $M/M_{\rm out}$", ylabel=r"$1+\langle\delta\rangle(M,t)$", ylim=(1, 300))
ax[2].legend(frameon=False, fontsize=8, loc="upper left")
ax[2].set_title("(c) the seed: nested top hats (shaded: halted)", fontsize=10)
fig.tight_layout(); fig.savefig("fig_tophat.png", dpi=200)
say("figure: fig_tophat.png"); out.close()
