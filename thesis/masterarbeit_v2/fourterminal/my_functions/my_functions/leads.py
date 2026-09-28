"""Semi-infinite lead: batched Sancho-Rubio decimation for surface GF / self-energy."""

import warnings
import numpy as np
from numpy.linalg import inv


class Lead:
    """
    Semi-infinite lead, ONE orientation convention (no flags):

    H_onsite   : H_{n,n}, onsite block of one lead cell (M x M).
    V_out      : H_{n,n+1}, where cell n+1 is one step FURTHER AWAY from the
                 central region than cell n (M x M).
                 e.g. right lead (cells x = nx, nx+1, ...):  V_out = H_{x,x+1}   = V
                      left  lead (cells x = -1, -2, ...):    V_out = H_{x,x-1}   = V^dagger
    V_coupling : H_{s,C}, rows = lead surface cell s, cols = the central sites it touches (M x Nc).

    Then  g_s = [z - H_00 - V_out g_s V_out^dagger]^{-1}   (Sancho-Rubio fixed point)
    and   Sigma = H_{C,s} g_s H_{s,C} = V_coupling^dagger g_s V_coupling.
    """

    def __init__(self, name, H_onsite, V_out, V_coupling, p, br=False, dual=False):
        self.name = name
        self.dual = dual
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_out, dtype=complex)        # attribute name kept for sgf_cache
        self.V_coupling = np.asarray(V_coupling, dtype=complex).conj().T if br else np.asarray(V_coupling, dtype=complex)
        self.p = p

    def surface_gf(self, z_batch):
        """Batched Sancho-Rubio decimation. z_batch shape: (N_E, M, M)."""
        M = self.H_onsite.shape[0]
        N_E = z_batch.shape[0]
        eps_s = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        eps_b = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        # dual=False: lead grows towards +x/+y, H[surface, next] = V_hop
        # dual=True : lead grows towards -x/-y, H[surface, next] = V_hop^dag
        V_out = self.V_hop.conj().T if self.dual else self.V_hop
        alpha = np.broadcast_to(V_out, (N_E, M, M)).copy()             # surface -> outward
        beta = np.broadcast_to(V_out.conj().T, (N_E, M, M)).copy()

        for _ in range(self.p.max_iter):
            g = inv(z_batch - eps_b)
            alpha_g, beta_g = alpha @ g, beta @ g
            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha
            alpha, beta = alpha_g @ alpha, beta_g @ beta
            if (np.max(np.sum(np.abs(alpha), axis=2)) < self.p.tol
                    and np.max(np.sum(np.abs(beta), axis=2)) < self.p.tol):
                break
        else:
            warnings.warn(f"Sancho-Rubio for lead '{self.name}' not converged after "
                          f"{self.p.max_iter} iterations (increase eta or max_iter).")

        return inv(z_batch - eps_s)

    def self_energy(self, z_batch):
        """Retarded self-energy Sigma(z) = V_coupling^dagger @ g_surface(z) @ V_coupling."""
        g = self.surface_gf(z_batch)
        Vc = self.V_coupling[None, :, :]
        return Vc.conj().transpose(0, 2, 1) @ g @ Vc