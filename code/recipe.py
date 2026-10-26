import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq
# units: kpc, km/s, Msun; time in kpc/(km/s) = 0.9778 Gyr
G, H0, Om, OL = 4.30091e-6, 0.07, 0.3, 0.7
L3, CE, CL = OL*H0**2, -0.91737, 9.640
K = 1 + 1.5*np.pi                   # 1/xi_b
def t_of_z(z):                      # flat LCDM
    a = 1/(1 + z)
    return (2/(3*H0*np.sqrt(OL))
            * np.arcsinh(np.sqrt(OL/Om)*a**1.5))
def y_of(lam, ell, eps=0.0):        # core equation with feedback
    A = 3*lam**2 + eps + 2*ell/3
    B = 0.25 - CE - 0.75*lam**2 - eps/4
    return lambertw(4*A*np.exp(4*B)).real/(4*A)
def grow(v, lam, z_acc, eps=0.0):
    """r_c [kpc] at the end of growth and today."""
    t = t_of_z(z_acc)
    l0 = L3*t**2/K**2               # step 1: front
    Rb = v*t/(K*(1 + CL*l0))
    ell = L3*Rb**2/v**2             # step 2: Lambda term
    y = y_of(lam, ell)              # step 3: core ratio
    rc = Rb/y                       # step 4: core
    rc_now = rc*(4*y + 1)/(4*y_of(lam, ell, eps) + 1)
    return rc, rc_now
def irc(R, v, rc):                  # rotation curve of P18
    x = R/rc
    return v*np.sqrt((1 - np.exp(-x))**2
        + (1 - (1 + 2*x + 2*x*x)*np.exp(-2*x))/(4*x))
def z_end(v, rc, lam=0.035, eps=0.0):
    """End of growth for a core rc seen today."""
    f = lambda z: grow(v, lam, z, eps)[1] - rc
    return brentq(f, 0, 20) if f(0) > 0 else None
if __name__ == "__main__":
    print(grow(60, 0.035, 0.5))         # 2.44, 2.44 kpc
    print(grow(60, 0.035, 1.0, 0.009))  # 1.47, 2.43
    print(z_end(48.8, 1.89))            # DDO 154: 0.54
