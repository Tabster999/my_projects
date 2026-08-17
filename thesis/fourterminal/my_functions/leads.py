"""Semi-infinite lead: batched Sancho-Rubio decimation for surface GF / self-energy."""

import numpy as np
from numpy.linalg import inv


class Lead:
    """
    A semi-infinite lead described by an onsite block H_onsite and an
    inter-cell hopping block V_hop, coupled to the central region via
    V_coupling.

    Parameters
    ----------
    name : str
        Label for the lead (used for bookkeeping only).
    H_onsite : (M, M) array
        Onsite block of the lead unit cell.
    V_hop : (M, M) array
        Hopping block between successive lead unit cells (cell i -> i+1).
    V_coupling : (Nc, M) or (M, Nc) array
        Coupling block between the lead's surface cell and the central region.
    p : Params
        Numerical parameters (max_iter, tol).
    br : bool
        If True, V_coupling is given as (central -> lead) and gets
        conjugate-transposed to the (lead -> central) convention used
        internally. Use this for leads attached on the "far" side
        (e.g. the right lead, or a bottom ribbon).
    """

    def __init__(self, name, H_onsite, V_hop, V_coupling, p, br=False):
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_hop, dtype=complex)
        Vc = np.asarray(V_coupling, dtype=complex)
        self.V_coupling = Vc.conj().T if br else Vc
        self.p = p

    def surface_gf(self, z_batch):
        """Batched Sancho-Rubio decimation. z_batch shape: (N_E, M, M)."""
        M = self.H_onsite.shape[0]
        N_E = z_batch.shape[0]
        eps_s = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        eps_b = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        alpha = np.broadcast_to(self.V_hop, (N_E, M, M)).copy()
        beta = np.broadcast_to(self.V_hop.conj().T, (N_E, M, M)).copy()

        for _ in range(self.p.max_iter):
            g = inv(z_batch - eps_b)
            alpha_g, beta_g = alpha @ g, beta @ g
            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha
            alpha, beta = alpha_g @ alpha, beta_g @ beta
            if (np.max(np.sum(np.abs(alpha), axis=2)) < self.p.tol
                    and np.max(np.sum(np.abs(beta), axis=2)) < self.p.tol):
                break

        return inv(z_batch - eps_s)

    def self_energy(self, z_batch):
        """Retarded self-energy Sigma(z) = V_coupling @ g_surface(z) @ V_coupling^dagger."""
        g = self.surface_gf(z_batch)
        Vc = self.V_coupling[None, :, :]
        return Vc.conj().transpose(0, 2, 1) @ g @ Vc
