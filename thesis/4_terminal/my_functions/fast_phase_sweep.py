"""
Fast phase-sweep engine for FourTerminalJunction.

Two independent speedups over calling FourTerminalJunction.channels() fresh
for every phi in a sweep:

1. Gauge rotation: the Top ribbon's self-energy at any phase phi is unitarily
   related to its value at a reference phase phi_ref by a *diagonal* unitary
   U(phi) (a pure U(1) gauge transform of the pairing term):

       g_T^-1(phi) = U(phi)^dagger . g_T^-1(phi_ref) . U(phi)   (same for Sigma_T)

   with U(phi) = per-site diag(1, 1, e^{i*phi}, e^{i*phi}) in the (e_up, e_down,
   h_up, h_down) local basis (electron components untouched, hole components
   rotated). This means Sancho-Rubio decimation of the Top ribbon (the
   expensive, iterative part) only has to run ONCE, not once per phi.
   Verified numerically equivalent to direct reconstruction (see project notes).

2. LU/Woodbury reuse: solve the system WITHOUT the top ribbon, A_noT, once
   (one LU factorization for all right-hand sides), then attach the top ribbon
   for any phi through its inverse surface GF:
       G = G_noT + G_noT U [g_T^-1(phi) - U^dag G_noT U]^-1 U^dag G_noT,
       U = P_T W_T^dag  (W_T = H_{s,C} of the top ribbon).
   Using g_T^-1 = z - eps_s instead of Sigma_T keeps this stable when the
   ribbon surface GF has a pole (Majorana end state of a narrow topological
   strip at E = 0): Sigma_T ~ 1/eta would make a Sigma-based Woodbury update
   lose all digits.  Only ONE retarded solve is ever needed.

Both are exact (not approximations) given converged decimation.
"""

import numpy as np
from scipy.linalg import solve as batched_solve


class FastPhaseSweep:
    """
    One-time setup at a fixed E_sweep, then cheap per-phi channel evaluation.

    Usage
    -----
        fps = FastPhaseSweep(junction, E_sweep, phi_ref=0.0)
        for phi in phi_array:
            ch = fps.channels_at_phi(phi)          # dict, both sides
            ch_left = fps.channels_at_phi(phi, side_name='left')
    """

    def __init__(self, junction, E_sweep, phi_ref=0.0):
        p = junction.p
        self.p = p
        self.junction = junction
        nx, ny = p.nx, p.ny
        self.nx, self.ny = nx, ny
        dim = 4 * nx * ny
        self.dim = dim
        N_E = len(E_sweep)
        self.E_sweep = np.asarray(E_sweep)

        z_1D = (self.E_sweep + 1j * p.eta)[:, None, None]
        z_lead_N = z_1D * np.eye(4 * ny, dtype=complex)[None, :, :]
        z_lead_SC = z_1D * np.eye(4 * nx, dtype=complex)[None, :, :]
        z_center = z_1D * np.eye(dim, dtype=complex)[None, :, :]

        # --- phi-independent self-energies (computed once, as usual) ---
        Sigma_l = junction.lead_L.self_energy(z_lead_N)
        Sigma_r = junction.lead_R.self_energy(z_lead_N)
        Sigma_b = junction.ribbon_bot.self_energy(z_lead_SC)

        # --- top ribbon at the reference phase: ONE decimation, kept as g_T^-1 ---
        ribbon_top_ref = junction._make_ribbon(phi_ref, p.tc_top, is_bot=False)   # same builder as the junction
        self.gT_inv_ref = ribbon_top_ref.surface_gf_inv(z_lead_SC)               # (N_E, 4nx, 4nx)
        W_T = ribbon_top_ref.V_coupling                                            # H_{s,C}
        self.phi_ref = phi_ref

        iL, iR = junction.idx_L, junction.idx_R
        self.iL, self.iR = iL, iR
        self.cT = np.arange(4 * nx)

        # --- A_noT: everything except the top ribbon ---
        H_C = junction.H_C
        Sigma_tot_noTop = np.zeros((N_E, dim, dim), dtype=complex)
        Sigma_tot_noTop[:, -4 * nx:, -4 * nx:] += Sigma_b
        Sigma_tot_noTop[:, iL[:, None], iL[None, :]] += Sigma_l
        Sigma_tot_noTop[:, iR[:, None], iR[None, :]] += Sigma_r
        A_noT = z_center - H_C[None, :, :] - Sigma_tot_noTop

        # --- one combined solve: lead columns (B1) + top-row columns (PT) ---
        lead_cols = np.concatenate([iL, iR])
        n_lead_cols = len(lead_cols)
        B1 = np.zeros((N_E, dim, n_lead_cols), dtype=complex)
        B1[:, lead_cols, np.arange(n_lead_cols)] = 1.0
        PT = np.zeros((N_E, dim, 4 * nx), dtype=complex)
        PT[:, self.cT, np.arange(4 * nx)] = 1.0
        X = batched_solve(A_noT, np.concatenate([B1, PT], axis=2))
        X_lead, Y_top = X[:, :, :n_lead_cols], X[:, :, n_lead_cols:]

        # keep only what the per-phi update needs (rows at the lead sites)
        self.n_L, self.n_R = len(iL), len(iR)
        self.X_QQ = X_lead[:, lead_cols, :]                                   # G_noT[Q, Q]
        self.Lft = Y_top[:, lead_cols, :] @ W_T.conj().T                      # G_noT[Q, T] W_T^dag
        self.Rgt = W_T @ X_lead[:, self.cT, :]                                # W_T G_noT[T, Q]
        self.C_TT = W_T @ Y_top[:, self.cT, :] @ W_T.conj().T                 # W_T G_noT[T, T] W_T^dag

        self.Gamma_L = 1j * (Sigma_l - Sigma_l.conj().transpose(0, 2, 1))
        self.Gamma_R = 1j * (Sigma_r - Sigma_r.conj().transpose(0, 2, 1))

        e_idx = np.where(np.tile([True, True, False, False], ny))[0]
        h_idx = np.where(np.tile([False, False, True, True], ny))[0]
        self.e_idx, self.h_idx = e_idx, h_idx

    def _gauge_matrix(self, phi):
        u_site = np.array([1.0, 1.0, np.exp(1j * phi), np.exp(1j * phi)])
        u_full = np.tile(u_site, self.nx)
        return u_full   # diagonal entries only; apply via broadcasting, not full matmul

    def channels_at_phi(self, phi, side_name=None):
        """Cheap per-phi evaluation: gauge-rotate g_T^-1, one (4nx)-dim solve, extract channels."""
        u_rel = self._gauge_matrix(phi) / self._gauge_matrix(self.phi_ref)
        gT_inv = (u_rel.conj()[None, :, None]) * self.gT_inv_ref * (u_rel[None, None, :])
        G = self.X_QQ + self.Lft @ batched_solve(gT_inv - self.C_TT, self.Rgt)   # (N_E, nQ, nQ)

        n_L = self.n_L
        G_LL, G_LR = G[:, :n_L, :n_L], G[:, :n_L, n_L:]
        G_RL, G_RR = G[:, n_L:, :n_L], G[:, n_L:, n_L:]

        e_idx, h_idx = self.e_idx, self.h_idx

        def sub(M, r, c):
            return M[:, r[:, None], c[None, :]]

        def dagger(M):
            return M.conj().transpose(0, 2, 1)

        def T(G1, B1_, G2, B2):
            return np.trace(G1 @ B1_ @ G2 @ B2, axis1=1, axis2=2).real

        Gamma_Les = sub(self.Gamma_L, e_idx, e_idx)
        Gamma_Lhs = sub(self.Gamma_L, h_idx, h_idx)
        Gamma_Res = sub(self.Gamma_R, e_idx, e_idx)
        Gamma_Rhs = sub(self.Gamma_R, h_idx, h_idx)

        def side(name):
            G = G_LR if name == 'left' else G_RL      # G^R_{ij}, i = this lead
            G_self = G_LL if name == 'left' else G_RR
            Gamma_own_e = Gamma_Les if name == 'left' else Gamma_Res
            Gamma_own_h = Gamma_Lhs if name == 'left' else Gamma_Rhs
            Gamma_other_e = Gamma_Res if name == 'left' else Gamma_Les
            Gamma_other_h = Gamma_Rhs if name == 'left' else Gamma_Lhs

            G_ee = sub(G, e_idx, e_idx)
            G_eh = sub(G, e_idx, h_idx)
            G_he = sub(G, h_idx, e_idx)
            G_hh = sub(G, h_idx, h_idx)
            G_self_eh = sub(G_self, e_idx, h_idx)
            G_self_he = sub(G_self, h_idx, e_idx)

            return {
                "ee": T(Gamma_own_e, G_ee, Gamma_other_e, dagger(G_ee)),
                "hh": T(Gamma_own_h, G_hh, Gamma_other_h, dagger(G_hh)),
                "eh_cross": T(Gamma_own_e, G_eh, Gamma_other_h, dagger(G_eh)),
                "he_cross": T(Gamma_own_h, G_he, Gamma_other_e, dagger(G_he)),
                "eh_local": T(Gamma_own_e, G_self_eh, Gamma_own_h, dagger(G_self_eh)),
                "he_local": T(Gamma_own_h, G_self_he, Gamma_own_e, dagger(G_self_he)),
            }

        if side_name is None:
            return {'left': side('left'), 'right': side('right')}
        return side(side_name)
