"""
CHECK: Fermi momenta and velocities of the ACTUAL lattice model (my_functions blocks, normal state,
E_Z = 0, Delta = 0) against the two continuum formulas:
  with shift (code, Scharf):  k_F = k_mu -+ k_so,                 hbar v_F = (hbar^2/m) k_mu
  without shift:              k_F = sqrt(k_mu^2 + k_so^2) -+ k_so, hbar v_F = (hbar^2/m) sqrt(k_mu^2 + k_so^2)
with k_mu = sqrt(2 m mu)/hbar, k_so = m alpha / hbar^2.  Bulk of the junction material (mu_C), k along
the junction (package x) and across it (package y); a -> 0 should approach the continuum.
"""
import numpy as np
import tools as st
from my_functions.hamiltonians import model_onsite, model_hop, region_alpha

HB2M = 76.1996 / 0.038                         # hbar^2 / m, meV nm^2


def bands(q, ka, direction):
    """Electron bands (meV) of the bulk junction material at momenta ka along 'x' or 'y'."""
    p = q.params(E_Z=0.0, phi=0.0)
    ons = model_onsite(p, 'c')[:2, :2]                                  # electron block (c_up, c_dn)
    V = model_hop(p, direction, p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))[:2, :2]
    Vo = model_hop(p, 'y' if direction == 'x' else 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))[:2, :2]
    ph = np.exp(1j * np.asarray(ka))[:, None, None]
    H = ons + V * ph + V.conj().T / ph + Vo + Vo.conj().T              # (n_k, 2, 2); other momentum = 0
    return np.linalg.eigvalsh(H) * q.t


def fermi_points(q, direction):
    """[(k_F in 1/nm, hbar v_F in meV nm)] of both bands, from the zero crossings (linear interpolation
    between grid points; slope = dE/d(ka) * a)."""
    ka = np.linspace(1e-6, min(np.pi, 0.1 * q.a), 200001)              # k up to 0.1 1/nm (k_F ~ 0.035)
    E = bands(q, ka, direction)
    res = []
    for b in (0, 1):
        i = np.flatnonzero(np.diff(np.sign(E[:, b])))[0]
        kF = ka[i] - E[i, b] * (ka[i + 1] - ka[i]) / (E[i + 1, b] - E[i, b])
        res.append((kF / q.a, (E[i + 1, b] - E[i, b]) / (ka[i + 1] - ka[i]) * q.a))
    return sorted(res)


if __name__ == "__main__":
    for alpha in (16.0, 32.0):
        q0 = st.case(100.0, alpha=alpha)
        mu = q0.mu_centre
        k_mu, k_so = np.sqrt(2 * mu / HB2M), alpha / HB2M
        A = (k_mu - k_so, k_mu + k_so, HB2M * k_mu)
        B = (np.hypot(k_mu, k_so) - k_so, np.hypot(k_mu, k_so) + k_so, HB2M * np.hypot(k_mu, k_so))
        print(f"\nalpha = {alpha:g} meV nm, mu_C = {mu} meV")
        print(f"  formula WITH shift   : k_F = {A[0]:.5f} / {A[1]:.5f} 1/nm,  hbar v_F = {A[2]:.2f} meV nm")
        print(f"  formula WITHOUT shift: k_F = {B[0]:.5f} / {B[1]:.5f} 1/nm,  hbar v_F = {B[2]:.2f} meV nm")
        print("   a [nm]  dir   k_F inner / outer [1/nm]    hbar v_F inner / outer [meV nm]")
        for a in (20.0, 10.0, 5.0, 2.5, 1.0):
            q = st.case(100.0, alpha=alpha, a=a)
            for d in ('x', 'y'):
                (k1, v1), (k2, v2) = fermi_points(q, d)
                print(f"   {a:5.1f}   {d}    {k1:.5f} / {k2:.5f}          {v1:6.2f} / {v2:6.2f}")
