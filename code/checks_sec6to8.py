"""Numbers quoted in Sects. 6.4, 6.6, 7.4 and 8.1 that are not printed by make_figures.py.

    python3 checks_sec6to8.py

(1) epoch from the median v/r_c for other median spins and a 0.05 dex calibration shift (Sect. 6.4)
(2) epoch if the spin term is not stored in the core (Sect. 6.4)
(3) core ratio y and epoch for the generalised profile with n = 1.76 (Sects. 7.1, 7.4)
(4) pericentre of a shell falling from turnaround into the settled P18 galaxy (Sect. 8.1)
(5) mass excess and central surface density implied by the radial acceleration relation (Sect. 6.6)
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import lambertw

# set-up of make_figures.py (cosmology, front, core equation) without the figures
_src = open("make_figures.py").read().split("\n")
exec("\n".join(_src[:249]))

u = lambda x: 1 - (1 - x) * np.exp(-x)
VRC_OBS = 28.7                          # median v/r_c of the 79 dwarfs, n = 2 fits

def vrc(z, lam):
    Rb, y, rc, Mh = front_exact(t_of_z(z), LAM3, lam)
    return V / rc, y

print("(1) end of growth from the median v/r_c")
for lam in (0.025, 0.035, 0.05):
    for dex in (0.0, 0.05, -0.05):
        target = VRC_OBS * 10**dex
        z = brentq(lambda z: vrc(z, lam)[0] - target, 0.05, 3)
        print(f"  lambda = {lam}, v/r_c = {target:.1f}: z = {z:.2f}")

print("(2) spin term not stored (lambda -> 0.005)")
z = brentq(lambda z: vrc(z, 0.005)[0] - VRC_OBS, 0.05, 3)
print(f"  z = {z:.2f}")

print("(3) generalised profile, n = 1.76")
def m_n(x, n): return quad(lambda t: u(t)**n, 0, x, limit=200)[0]
def c_and_CE(n, X=60):
    c = quad(lambda x: u(x)**n - 1, 0, 200, limit=400)[0]
    I = quad(lambda x: m_n(x, n) * u(x)**n / x, 1e-9, X, limit=400)[0]
    return c, I - X - c * np.log(X)
def y_n(n, A, lam=LAMSPIN):
    c, ce = c_and_CE(n)
    Bp = c * (1 - 3 * lam**2) - ce
    return (c / A) * lambertw((A / c) * np.exp(Bp / c)).real
lam = LAMSPIN
print(f"  Lambda = 0, median spin: y(n=2) = {y_n(2, 3*lam**2):.1f}, y(n=1.76) = {y_n(1.76, 3*lam**2):.1f}")
B4 = 4 * (0.25 - c_and_CE(2)[1] - 0.75 * lam**2)
VRC_OBS_176 = 25.35                     # median v/r_c of the n = 1.76 fits (fit_exponent.py, fit_fixed_n)
def vrc176(z):
    v2, y2 = vrc(z, lam)
    A = (B4 - np.log(y2)) / (4 * y2)    # A of the n = 2 solution at this front: same spin and Lambda term
    return v2 * y_n(1.76, A) / y2
print(f"  with Lambda at z = 0.5: y(n=2) = {vrc(0.5, lam)[1]:.1f}, "
      f"y(n=1.76) = {vrc(0.5, lam)[1]*vrc176(0.5)/vrc(0.5, lam)[0]:.1f}")
z = brentq(lambda z: vrc176(z) - VRC_OBS_176, 0.05, 2)
print(f"  end of growth for n = 1.76 (median v/r_c = {VRC_OBS_176}): z = {z:.2f}")

print("(4) pericentre in the settled P18 galaxy (units v = r_c = 1)")
m2 = lambda x: quad(lambda t: u(t)**2, 0, x)[0]
for y in (50.5, 35.0):
    M_ = m2(y); Rta_ = 2 * M_; E_ = -M_ / Rta_
    Phi = lambda r: -M_ / y - quad(lambda x: m2(x) / x**2, r, y)[0]
    for l in (0.035, 0.07):
        j = np.sqrt(2) * l * Rta_
        rp = brentq(lambda r: Phi(r) + j**2 / (2 * r**2) - E_, 1e-4, y)
        print(f"  y = {y}, lambda = {l}: r_p = {rp:.2f} r_c = R_b/{y/rp:.0f}")

print("(5) radial acceleration relation")
Gsi, gd, Msun, pc = 6.674e-11, 1.2e-10, 1.989e30, 3.0857e16
v = 73e3; Mb = v**4 / (Gsi * gd)
crc = Gsi * Mb / (2 * v**2) / pc / 1e3
print(f"  v = 73 km/s: M_b = {Mb/Msun:.2e} Msun, c r_c = {crc:.2f} kpc, c = {crc/2.4:.2f} for r_c = 2.4 kpc")
print(f"  Sigma = 2 c g_dagger/(pi G) = {2*0.25*gd/(np.pi*Gsi)/(Msun/pc**2):.0f} Msun/pc^2 for c = 1/4")

print("(6) core growth after t_acc (Hashim et al. 2026, isolated simulation: r_c 0.15 -> 0.34 kpc, 2 -> 12 Gyr, ~t^0.48)")
t_now = t_of_z(0)
vrc_m = lambda z: vrc(z, LAMSPIN)[0]
z = brentq(lambda z: vrc_m(z) - VRC_OBS * 2.4 / (2.4 - 0.02 * (t_now - t_of_z(z))), 0.05, 3)
print(f"  absolute growth 0.2 kpc per 10 Gyr on the median r_c = 2.4 kpc: z = {z:.2f}")
z = brentq(lambda z: vrc_m(z) - VRC_OBS * (t_now / t_of_z(z))**0.48, 0.05, 3)
print(f"  proportional growth r_c ~ t^0.48 after t_acc: z = {z:.2f}")
