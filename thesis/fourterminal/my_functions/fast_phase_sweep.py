"""
Fast phase-sweep engine for FourTerminalJunction.

Two independent speedups over calling FourTerminalJunction.channels() fresh
for every phi in a sweep:

1. Gauge rotation: the Top ribbon's self-energy at any phase phi is unitarily
   related to its value at a reference phase phi_ref by a *diagonal* unitary
   U(phi) (a pure U(1) gauge transform of the pairing term):

       Sigma_T(phi) = U(phi)^dagger . Sigma_T(phi_ref) . U(phi)

   with U(phi) = per-site diag(1, 1, e^{i*phi}, e^{i*phi}) in the (e_up, e_down,
   h_up, h_down) local basis (electron components untouched, hole components
   rotated). This means Sancho-Rubio decimation of the Top ribbon (the
   expensive, iterative part) only has to run ONCE, not once per phi.
   Verified numerically equivalent to direct reconstruction (see project notes).

2. LU/Woodbury reuse: solve the reference system A(phi_ref) once (one LU
   factorization, reused for every right-hand side), then update to any other
   phi via the Sherman-Morrison-Woodbury identity, since only the Top lead's
   self-energy (rank <= dim(cT)) changes with phi. No refactorization of the
   full system is needed per phi. The "advanced" Green's function blocks are
   obtained as the conjugate-transpose of the already-computed retarded
   blocks (verified numerically equivalent to a full GA solve), so only ONE
   retarded solve is ever needed.

Both are exact (not approximations) given converged decimation.
"""

import numpy as np
from scipy.linalg import solve as batched_solve
from .hamiltonians import onsite_block, Vx, Vy, make_row_hamiltonian
from scipy.linalg import block_diag


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

        # --- Top self-energy at the reference phase (one decimation) ---
        ribbon_top_ref = self._make_ribbon(phi_ref, p.tc_top)
        Sigma_t_ref = ribbon_top_ref.self_energy(z_lead_SC)   # shape (N_E, 4nx, 4nx)
        self.Sigma_t_ref = Sigma_t_ref
        self.phi_ref = phi_ref

        # --- gauge rotation matrix (per-site diag(1,1,e^{i phi},e^{i phi})) ---
        iL, iR = junction.idx_L, junction.idx_R
        self.iL, self.iR = iL, iR
        self.cT = np.arange(4 * nx)

        # --- A0 excludes Top; Aref = A0 - Sigma_T(phi_ref) fully embedded ---
        H_C = junction.H_C
        Sigma_tot_noTop = np.zeros((N_E, dim, dim), dtype=complex)
        Sigma_tot_noTop[:, -4 * nx:, -4 * nx:] += Sigma_b
        Sigma_tot_noTop[:, iL[:, None], iL[None, :]] += Sigma_l
        Sigma_tot_noTop[:, iR[:, None], iR[None, :]] += Sigma_r

        A0 = z_center - H_C[None, :, :] - Sigma_tot_noTop
        Aref = A0.copy()
        Aref[:, :4 * nx, :4 * nx] -= Sigma_t_ref

        # --- combined right-hand side: lead-coupling columns (B1) + Top-block columns (PT) ---
        lead_cols = np.concatenate([iL, iR])
        n_lead_cols = len(lead_cols)
        B1 = np.zeros((N_E, dim, n_lead_cols), dtype=complex)
        rows = np.arange(n_lead_cols)
        B1[:, lead_cols[rows], rows] = 1.0

        PT = np.zeros((N_E, dim, 4 * nx), dtype=complex)
        PT[:, self.cT, np.arange(4 * nx)] = 1.0

        RHS = np.concatenate([B1, PT], axis=2)   # single combined solve -> one LU factorization per energy
        X_combined = batched_solve(Aref, RHS)    # numpy does one LU factorization per batch slice, reused for all RHS columns

        self.Xref = X_combined[:, :, :n_lead_cols]
        self.Ytop = X_combined[:, :, n_lead_cols:]
        self.n_lead_cols = n_lead_cols
        self.n_L = len(iL)
        self.n_R = len(iR)

        self.Stop = self.Ytop[:, self.cT, :]        # (N_E, 4nx, 4nx)
        self.XrefTop = self.Xref[:, self.cT, :]      # (N_E, 4nx, n_lead_cols)
        self.IdT = np.eye(4 * nx, dtype=complex)[None, :, :]

        self.Gamma_L = 1j * (Sigma_l - Sigma_l.conj().transpose(0, 2, 1))
        self.Gamma_R = 1j * (Sigma_r - Sigma_r.conj().transpose(0, 2, 1))

        e_idx = np.where(np.tile([True, True, False, False], ny))[0]
        h_idx = np.where(np.tile([False, False, True, True], ny))[0]
        self.e_idx, self.h_idx = e_idx, h_idx

    def _make_ribbon(self, phi_lead, tc):
        from .leads import Lead
        p = self.p
        onsite_SC = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, twod=True)
        H_intra = make_row_hamiltonian(p.nx, onsite_SC, Vx(p.t_s))
        H_inter = block_diag(*([Vy(p.t_s)] * p.nx))
        V_coupling = block_diag(*([Vy(tc)] * p.nx))
        return Lead('ribbon', H_intra, H_inter, V_coupling, p, br=False)

    def _gauge_matrix(self, phi):
        u_site = np.array([1.0, 1.0, np.exp(1j * phi), np.exp(1j * phi)])
        u_full = np.tile(u_site, self.nx)
        return u_full   # diagonal entries only; apply via broadcasting, not full matmul

    def channels_at_phi(self, phi, side_name=None):
        """Cheap per-phi evaluation: gauge-rotate Sigma_T, Woodbury update, extract channels."""
        nx, ny = self.nx, self.ny
        u_ref = self._gauge_matrix(self.phi_ref)
        u_phi = self._gauge_matrix(phi)
        # Sigma_T(phi) = U(phi)^dagger . Sigma_T(phi_ref) . U(phi), with U(phi_ref) already
        # baked into Sigma_t_ref -> apply the RELATIVE rotation U(phi)/U(phi_ref)
        u_rel = u_phi / u_ref
        Sigma_t_phi = (u_rel.conj()[None, :, None]) * self.Sigma_t_ref * (u_rel[None, None, :])

        dST = Sigma_t_phi - self.Sigma_t_ref
        IdT = np.broadcast_to(self.IdT, (len(self.E_sweep), 4 * nx, 4 * nx))
        Ka = batched_solve(IdT - dST @ self.Stop, dST @ self.XrefTop)
        X1 = self.Xref + self.Ytop @ Ka   # (N_E, dim, n_lead_cols)

        n_L = self.n_L
        # X1 columns ordered [iL cols (n_L), iR cols (n_R)]; rows are full dim, slice to lead rows
        G_LL = X1[:, self.iL[:, None], np.arange(n_L)[None, :]]
        G_RL = X1[:, self.iR[:, None], np.arange(n_L)[None, :]]
        G_LR = X1[:, self.iL[:, None], (n_L + np.arange(self.n_R))[None, :]]
        G_RR = X1[:, self.iR[:, None], (n_L + np.arange(self.n_R))[None, :]]

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
            G = G_RL if name == 'left' else G_LR
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
