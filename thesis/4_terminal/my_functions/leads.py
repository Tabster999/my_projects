"""
Semi-infinite lead: Sancho-Rubio decimation for the surface Green's function / self-energy.

Accuracy note
-------------
Sancho-Rubio is exact for gapped leads.  For leads with propagating modes at the energy
(e.g. SC ribbons with gapless edge modes at E = 0) its rounding error grows like a high
power of 1/eta, so at eta = 1e-5 it can return a surface GF that is slightly (or badly)
off.  Every result is therefore checked against its own defining equation
    eps = H0 + V_out (z - eps)^-1 V_out^dag      (g_s = (z - eps)^-1)
and, if the relative residual is too large, refined by Newton steps on that equation
(each Newton step solved by the same doubling trick).  Energies that still fail are
counted in N_UNRELIABLE and reported with a warning.
"""

import warnings
import numpy as np
from numpy.linalg import inv, norm

N_UNRELIABLE = 0          # number of lead energies that failed the self-consistency check


def sancho_rubio(H_onsite, V_plus, z_batch, max_iter=100, tol=1e-14):
    """
    Batched Sancho-Rubio decimation of the chain ... - H_onsite - V_plus - H_onsite - ...
    (V_plus = H_{n,n+1}).  One run gives the renormalized surface onsite of BOTH ends:
        eps_plus  : lead growing towards +   (outward hop V_plus)
        eps_minus : lead growing towards -   (outward hop V_plus^dag)
    so that g_s = (z - eps)^-1.  z_batch = z * identity, shape (N_E, M, M).
    """
    M, N_E = H_onsite.shape[0], z_batch.shape[0]
    eps_p = np.broadcast_to(H_onsite, (N_E, M, M)).copy()
    eps_m, eps_b = eps_p.copy(), eps_p.copy()
    alpha = np.broadcast_to(V_plus, (N_E, M, M)).copy()
    beta = np.broadcast_to(V_plus.conj().T, (N_E, M, M)).copy()

    scale = max(
    np.max(np.sum(np.abs(V_plus), axis=1)),
    np.max(np.sum(np.abs(H_onsite), axis=1)),
    1.0)

    for _ in range(max_iter):
        g = inv(z_batch - eps_b)
        alpha_g, beta_g = alpha @ g, beta @ g
        agb, bga = alpha_g @ beta, beta_g @ alpha
        eps_p += agb
        eps_m += bga
        eps_b += agb + bga
        alpha, beta = alpha_g @ alpha, beta_g @ beta
        alpha_norm = np.max(np.sum(np.abs(alpha), axis=2))
        beta_norm  = np.max(np.sum(np.abs(beta), axis=2))

        if (alpha_norm < tol * scale and beta_norm < tol * scale): 
            break
    return eps_p, eps_m


def residual(H_onsite, V_out, z_batch, eps):
    """Relative residual |H0 + V_out g V_out^dag - eps| / |eps| per energy (0 for the exact solution)."""
    with np.errstate(all='ignore'):
        R = H_onsite[None] + V_out[None] @ inv(z_batch - eps) @ V_out.conj().T[None] - eps
        return np.nan_to_num(norm(R, axis=(1, 2)) / norm(eps, axis=(1, 2)), nan=np.inf)


def newton_refine(H_onsite, V_out, z, eps, tol=1e-12, max_steps=8):
    """
    Newton iteration for X = z - H0 - V X^-1 V^dag with X = z - eps = g_s^-1 (one energy).
    Linearized step:  D - A D B = -F,  A = V g, B = g V^dag,  solved by doubling
    (D = sum_k A^k (-F) B^k, like Sancho-Rubio).  Returns the best eps and its residual.
    """
    M = H_onsite.shape[0]
    zI = z * np.eye(M)
    X = zI - eps
    best, best_res = eps, np.inf
    for _ in range(max_steps + 1):
        with np.errstate(all='ignore'):
            g = inv(X)
            F = X - (zI - H_onsite) + V_out @ g @ V_out.conj().T
            res = norm(F) / norm(zI - X)
        if not np.isfinite(res) or res > 10 * best_res:
            break                                   # diverging: keep the best so far
        if res < best_res:
            best, best_res = zI - X, res
        if res < tol:
            break
        A, B = V_out @ g, g @ V_out.conj().T
        D = -F
        with np.errstate(all='ignore'):
            for _ in range(100):
                D = D + A @ D @ B
                A, B = A @ A, B @ B
                if norm(A, 1) * norm(B, 1) < 1e-16 or not np.isfinite(D).all():
                    break
        X = X + D
    return best, best_res


def surface_eps_pair(H_onsite, V_plus, z_batch, p, name="lead", need=(True, True)):
    """
    Sancho-Rubio surface onsite (eps_plus, eps_minus), checked and, if needed, Newton-refined.
    need = which orientations to check (the other one is returned unchecked).
    """
    global N_UNRELIABLE
    pair = list(sancho_rubio(H_onsite, V_plus, z_batch, p.max_iter, p.tol))
    zs = np.einsum('nii->n', z_batch) / H_onsite.shape[0]
    for k, V_out in enumerate((V_plus, V_plus.conj().T)):
        if not need[k]:
            continue
        res = residual(H_onsite, V_out, z_batch, pair[k])
        for n in np.flatnonzero(res > 1e-10):
            if p.lead_refine:
                pair[k][n], res[n] = newton_refine(H_onsite, V_out, zs[n], pair[k][n])
        bad = res > 1e-8
        if bad.any():
            N_UNRELIABLE += int(bad.sum())
            warnings.warn("Sancho-Rubio surface GF not self-consistent at some energies "
                          "(counted in my_functions.leads.N_UNRELIABLE); results there are unreliable.")
    return pair[0], pair[1]


class Lead:
    """
    Semi-infinite lead.  All matrices are given in the GLOBAL +x / +y sense
    (the same convention as the central region, H_{j,j+1} = V), and two flags
    say how the lead sits relative to the centre:

    H_onsite   : H_{n,n}, onsite block of one lead cell (M x M).
    V_out      : H_{n,n+1} along the global +x/+y direction inside the lead (M x M).
    V_coupling : the global-direction hop across the lead<->centre bond (M x Nc).

    dual : False -> lead grows towards +x/+y (right lead, bottom ribbon)
           True  -> lead grows towards -x/-y (left lead, top ribbon)
    br   : False -> H_{s,C} = V_coupling          (left lead, top ribbon)
           True  -> H_{s,C} = V_coupling^dagger   (right lead, bottom ribbon)

    Sigma = H_{C,s} g_s H_{s,C},  g_s = (z - eps_s)^-1 from Sancho-Rubio.
    """

    def __init__(self, name, H_onsite, V_out, V_coupling, p, br=False, dual=False):
        self.name = name
        self.dual = dual
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_out, dtype=complex)
        self.V_coupling = np.asarray(V_coupling, dtype=complex).conj().T if br else np.asarray(V_coupling, dtype=complex)
        self.p = p

    def surface_eps(self, z_batch):
        """Renormalized surface onsite eps_s, shape (N_E, M, M)."""
        need = (False, True) if self.dual else (True, False)
        pair = surface_eps_pair(self.H_onsite, self.V_hop, z_batch, self.p, self.name, need)
        return pair[1] if self.dual else pair[0]

    def surface_gf(self, z_batch):
        """Retarded surface GF g_s = (z - eps_s)^-1."""
        return inv(z_batch - self.surface_eps(z_batch))

    def surface_gf_inv(self, z_batch):
        """Inverse surface GF g_s^-1 = z - eps_s (finite even if g_s has a pole)."""
        return z_batch - self.surface_eps(z_batch)

    def self_energy(self, z_batch):
        """Retarded self-energy Sigma(z) = V_coupling^dagger @ g_s(z) @ V_coupling."""
        Vc = self.V_coupling[None, :, :]
        return Vc.conj().transpose(0, 2, 1) @ self.surface_gf(z_batch) @ Vc
