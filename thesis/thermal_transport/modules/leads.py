"""Semi-infinite lead: batched Sancho-Rubio decimation for surface GF / self-energy."""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve

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
    dual : bool
        If True, the lead extends towards NEGATIVE x (or y), i.e. it sits at
        the start of the H[i, i+1] = V_hop chain (left lead, top ribbon).
        Irrelevant when V_hop is Hermitian (no SOC in the lead).
    br : bool
        If True, V_coupling is given as (central -> lead) and gets
        conjugate-transposed to the (lead -> central) convention used
        internally. Use this for leads attached on the "far" side
        (e.g. the right lead, or a bottom ribbon).
    """

    def __init__(self, name, H_onsite, V_hop, V_coupling, p, br=False, dual=False):
        self.name = name
        self.dual = dual
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
        # dual=False: lead grows towards +x/+y, H[surface, next] = V_hop
        # dual=True : lead grows towards -x/-y, H[surface, next] = V_hop^dag
        V_out = self.V_hop.conj().T if self.dual else self.V_hop
        alpha = np.broadcast_to(V_out, (N_E, M, M)).copy()
        beta = np.broadcast_to(V_out.conj().T, (N_E, M, M)).copy()

        for _ in range(self.p.max_iter):
            g = np.linalg.solve(z_batch - eps_b, np.broadcast_to(np.eye(M), z_batch.shape))
            alpha_g, beta_g = alpha @ g, beta @ g
            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha
            alpha, beta = alpha_g @ alpha, beta_g @ beta
            if (np.max(np.sum(np.abs(alpha), axis=2)) < self.p.tol
                    and np.max(np.sum(np.abs(beta), axis=2)) < self.p.tol):
                break

        return inv(z_batch - eps_s)

    def self_energy(self, z_batch):
        """
        Retarded self-energy on the CENTRAL-region side:
            Sigma(z) = V_coupling^dagger @ g_surface(z) @ V_coupling,
        with V_coupling stored in the (lead -> central) convention (br flips it).
        """
        g = self.surface_gf(z_batch)
        Vc = self.V_coupling[None, :, :]
        return Vc.conj().transpose(0, 2, 1) @ g @ Vc