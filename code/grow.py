import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq
# units: kpc, km/s, Msun; time in kpc/(km/s) = 0.9778 Gyr
G, H0, Om, OL = 4.30091e-6, 0.07, 0.3, 0.7
L3, CE, CL = OL*H0**2, -0.91737, 9.640
K = 1 + 1.5*np.pi
def t_of_z(z):                      # flat LCDM
    a = 1/(1 + z)
    return (2/(3*H0*np.sqrt(OL))
            * np.arcsinh(np.sqrt(OL/Om)*a**1.5))
def y_of(lam, ell, eps=0.0):        # Eqs. (13), (24)
    A = 3*lam**2 + eps + 2*ell/3
    B = 0.25 - CE - 0.75*lam**2 - eps/4
    return lambertw(4*A*np.exp(4*B)).real/(4*A)
def grow(v, lam, z_acc, eps=0.0):
    """r_c [kpc] at the end of growth and today."""
    t = t_of_z(z_acc)
    l0 = L3*t**2/K**2               # step 1: front
    Rb = v*t/(K*(1 + CL*l0))
    ell = L3*Rb**2/v**2             # step 2
    y = y_of(lam, ell)              # step 3
    rc = Rb/y                       # step 4
    rc_now = rc*(4*y + 1)/(4*y_of(lam, ell, eps) + 1)
    return rc, rc_now
def irc(R, v, rc):                  # Eq. (3)
    x = R/rc
    return v*np.sqrt((1 - np.exp(-x))**2
        + (1 - (1 + 2*x + 2*x*x)*np.exp(-2*x))/(4*x))
def z_end(v, rc, lam=0.035, eps=0.0):
    """End of growth for a core rc seen today."""
    f = lambda z: grow(v, lam, z, eps)[1] - rc
    return brentq(f, 0, 20) if f(0) > 0 else None
if __name__ == "__main__":
    print(grow(60, 0.035, 0.5))     # 2.44, 2.44 kpc
    print(grow(60, 0.035, 1.0, 0.009))  # 1.47, 2.43
    print(z_end(81.6, 1.38))        # NGC 2915: 1.48
    print(z_end(82.4, 7.90))        # IC 2574: None
    v, rc = 48.8, 1.89              # DDO 154
    z_acc = z_end(v, rc)
    print(z_acc)                    # 0.54
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    R = np.linspace(1e-3, 10, 300)
    for z in [6, 2, 1, z_acc]:
        rc_z = grow(v, 0.035, z)[0]
        a1.plot(R, irc(R, v, rc_z), label=f"z = {z:.2g}")
    a1.set(xlabel="R [kpc]", ylabel="v_c [km/s]"); a1.legend()
    z = np.linspace(z_acc, 10, 200)
    rc_t = [grow(v, 0.035, zi)[0] for zi in z]
    a2.plot(t_of_z(z)*0.9778, rc_t)
    a2.set(xlabel="t [Gyr]", ylabel="r_c [kpc]")
    fig.tight_layout(); fig.savefig("grow_fig.png", dpi=200)
