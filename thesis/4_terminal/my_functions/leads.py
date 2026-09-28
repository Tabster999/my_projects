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

import hashlib
import warnings
from collections import OrderedDict
import numpy as np
from numpy.linalg import inv, norm

N_UNRELIABLE = 0          # number of lead energies that failed the self-consistency check
ACCEPT_TOL = 1e-8         # max. relative Dyson residual for a lead solution to be accepted

# Cache of lead solutions: sweeps of central-region parameters (mu_c, tc, tc_barr, ny, phi, ...)
# leave the leads unchanged.  The key contains everything the solution depends on; cached arrays
# are read-only; a cached failure (NaN) is counted again on every reuse.
CACHE_ENABLED = True
CACHE_MAX_BYTES = 1.5e9   # per process; LRU eviction beyond this
CACHE_STATS = {"hits": 0, "misses": 0}
_CACHE = OrderedDict()


def clear_cache():
    _CACHE.clear()
    CACHE_STATS.update(hits=0, misses=0)


def _cache_key(H_onsite, V_plus, zs, p, need):
    h = hashlib.sha1()
    for a in (H_onsite, V_plus, zs):
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    h.update(repr((p.max_iter, p.tol, p.lead_refine, tuple(need), ACCEPT_TOL)).encode())
    return h.hexdigest()


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


def eigenmode_eps(H_onsite, V_out, z):
    """
    Non-iterative surface onsite eps = H0 + V_out F from the lead's decaying Bloch modes
    ((z - H0 - lam V_out - V_out^dag / lam) u = 0, the M modes with smallest |lam|, F = U diag(lam) U^-1).
    Used as a fallback where Sancho-Rubio + Newton fail (at E ~ 0 when the ribbon edge gap << eta).
    """
    import scipy.linalg as sl
    M = H_onsite.shape[0]
    I, O = np.eye(M), np.zeros((M, M))
    lam, X = sl.eig(np.block([[O, I], [V_out.conj().T, -(z * I - H_onsite)]]), np.block([[I, O], [O, -V_out]]))
    order = np.argsort(np.abs(lam))[:M]
    U = X[:M, order]
    return H_onsite + V_out @ (U @ np.diag(lam[order]) @ np.linalg.inv(U))


def surface_eps_pair(H_onsite, V_plus, z_batch, p, name="lead", need=(True, True)):
    """
    Sancho-Rubio surface onsite (eps_plus, eps_minus), checked against the Dyson equation; failures
    are Newton-refined, then replaced by the eigenmode solution, and set to NaN if still failing.
    need = which orientations to check (the other one is returned unchecked).
    """
    global N_UNRELIABLE
    zs = np.einsum('nii->n', z_batch) / H_onsite.shape[0]
    if CACHE_ENABLED:
        key = _cache_key(H_onsite, V_plus, zs, p, need)
        if key in _CACHE:
            _CACHE.move_to_end(key); CACHE_STATS["hits"] += 1
            hit = _CACHE[key]
            n_bad = sum(int(np.isnan(a).any(axis=(1, 2)).sum()) for a in hit)
            if n_bad:
                N_UNRELIABLE += n_bad
                warnings.warn("cached lead solution contains failed (NaN) energies; counted in N_UNRELIABLE.")
            return hit
        CACHE_STATS["misses"] += 1
    pair = list(sancho_rubio(H_onsite, V_plus, z_batch, p.max_iter, p.tol))
    for k, V_out in enumerate((V_plus, V_plus.conj().T)):
        if not need[k]:
            continue
        res = residual(H_onsite, V_out, z_batch, pair[k])
        for n in np.flatnonzero(res > 1e-10):
            if p.lead_refine:
                pair[k][n], res[n] = newton_refine(H_onsite, V_out, zs[n], pair[k][n])
        for n in np.flatnonzero(res > ACCEPT_TOL):       # still failing: eigenmode solution
            try:
                with np.errstate(all='ignore'):
                    cand = eigenmode_eps(H_onsite, V_out, zs[n])
                    r_c = residual(H_onsite, V_out, z_batch[n:n+1], cand[None])[0]
                if np.isfinite(r_c) and r_c < res[n]:
                    pair[k][n], res[n] = cand, r_c
            except (np.linalg.LinAlgError, ValueError):
                pass
        bad = res > ACCEPT_TOL
        if bad.any():
            N_UNRELIABLE += int(bad.sum())
            pair[k][bad] = np.nan                          # never hand on a wrong lead as data
            warnings.warn("lead surface GF not self-consistent at some energies even after Newton and the "
                          "eigenmode fallback (counted in my_functions.leads.N_UNRELIABLE); set to NaN.")
    if CACHE_ENABLED:
        for a in pair:
            a.setflags(write=False)
        _CACHE[key] = (pair[0], pair[1])
        while len(_CACHE) > 1 and sum(a.nbytes + b.nbytes for a, b in _CACHE.values()) > CACHE_MAX_BYTES:
            _CACHE.popitem(last=False)
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
