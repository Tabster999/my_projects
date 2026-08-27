"""FourTerminalJunction: assembles H_C + the four leads and computes transport channels."""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag

from .hamiltonians import onsite_block, Vx, Vy, make_row_hamiltonian, get_2d_hamiltonian, flat_site_idx
from .leads import Lead


class FourTerminalJunction:
    """
    Geometry: central nx*ny normal region, with SC ribbons (width nx) attached
    along the full top and bottom edges, and normal leads attached at the
    left/right edges (Sancho-Rubio decimated in x).
    """

    def __init__(self, p):
        self.p = p
        self._build()

    def _build(self):
        p = self.p
        nx, ny = p.nx, p.ny

        onsite_C = onsite_block(p.t_c, p.mu_c, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                                 alpha=p.alpha, beta=p.beta, twod=True)
        self.H_C = get_2d_hamiltonian(nx, ny, onsite_C, Vx(p.t_c, p.alpha, p.beta),
                                       Vy(p.t_c, p.alpha, p.beta))

        onsite_N = onsite_block(p.t_n, p.mu_n, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                                 alpha=p.alpha, beta=p.beta, twod=True)
        H_layer_N = make_row_hamiltonian(ny, onsite_N, Vy(p.t_n, p.alpha, p.beta))
        V_n = block_diag(*[Vx(p.t_n, p.alpha, p.beta)] * ny)
        V_coupling_LR = block_diag(*[Vx(p.tc_barr)] * ny)

        self.lead_L = Lead('L', H_layer_N, V_n, V_coupling_LR, p, br=False)
        self.lead_R = Lead('R', H_layer_N, V_n, V_coupling_LR, p, br=True)
        self.ribbon_top = self._make_ribbon(p.phi, p.tc_top)
        self.ribbon_bot = self._make_ribbon(0, p.tc_bot)
        self.chain_top = self._make_single_chain(p.phi, is_bot=False)
        self.chain_bot = self._make_single_chain(0, is_bot=True)

        left_sites = [ix + nx * iy for iy in range(ny) for ix in [0]]
        right_sites = [ix + nx * iy for iy in range(ny) for ix in [nx - 1]]
        self.idx_L = flat_site_idx(left_sites)
        self.idx_R = flat_site_idx(right_sites)

    def _make_ribbon(self, phi_lead, tc, is_bot=False):
        p = self.p
        onsite_SC = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, twod=True)
        H_intra = make_row_hamiltonian(p.nx, onsite_SC, Vx(p.t_s))
        H_inter = block_diag(*([Vy(p.t_s)] * p.nx))
        V_coupling = block_diag(*([Vy(tc)] * p.nx))
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, H_intra, H_inter, V_coupling, p, br=is_bot)

    def _make_single_chain(self, phi_lead, is_bot=False):
        p = self.p
        onsite_sc = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, Bz=0.0, Bxy=0.0, theta_z=0.0, alpha=0.0, beta=0.0, twod=False)
        hop_y = Vy(p.t_s, alpha=0.0, beta=0.0)
        tc = p.tc_bot if is_bot else p.tc_top
        hop_c = Vy(tc)
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, onsite_sc, hop_y, hop_c, p, br=False)

    def _z_batches(self, E_sweep):
        p = self.p
        dim = 4 * p.nx * p.ny
        z1 = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(4 * p.ny, dtype=complex)[None, :, :]
        zN = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(dim, dtype=complex)[None, :, :]
        return z1, zN

    def channels(self, E_sweep, side_name=None, precompute_sigmas_n=None, is_ribbon=True):
        """
        Compute the EC/CAR/LAR transmission channels for the requested lead(s).

        Parameters
        ----------
        E_sweep : (N_E,) array
        side_name : {'left', 'right', None}
            None returns a dict with both {'left': ..., 'right': ...}.
        precompute_sigmas_n : (Sigma_L, Sigma_R) or None
            Reuse precomputed normal-lead self-energies (they don't depend on
            phi), useful for phase sweeps.
        """
        p = self.p
        nx, ny = p.nx, p.ny
        dim = 4 * nx * ny
        N_E = len(E_sweep)

        z_1D = (E_sweep + 1j * p.eta)[:, None, None]
        z_lead_N = z_1D * np.eye(4 * ny, dtype=complex)[None, :, :]
        z_lead_SC = z_1D * np.eye(4 * nx, dtype=complex)[None, :, :]
        z_center = z_1D * np.eye(dim, dtype=complex)[None, :, :]
        z_single = z_1D * np.eye(4, dtype=complex)[None, :, :]

        if precompute_sigmas_n is None:
            Sigma_l = self.lead_L.self_energy(z_lead_N)
            Sigma_r = self.lead_R.self_energy(z_lead_N)
        else:
            Sigma_l, Sigma_r = precompute_sigmas_n

        Sigma_t = self.ribbon_top.self_energy(z_lead_SC) if is_ribbon else self.chain_top.self_energy(z_single)
        Sigma_b = self.ribbon_bot.self_energy(z_lead_SC) if is_ribbon else self.chain_bot.self_energy(z_single)
        Sigma_tot = np.zeros((N_E, dim, dim), dtype=complex)

        if is_ribbon:
            Sigma_tot[:, :4 * nx, :4 * nx] += Sigma_t
            Sigma_tot[:, -4 * nx:, -4 * nx:] += Sigma_b
        else:
            for ix in range(nx):
                Sigma_tot[:, 4*ix:4*(ix+1), 4*ix:4*(ix+1)] += Sigma_t
                Sigma_tot[:, 4*(ix + nx*(ny-1)):4*(ix + nx*(ny-1)+1), 4*(ix + nx*(ny-1)):4*(ix + nx*(ny-1)+1)] += Sigma_b

        iL, iR = self.idx_L, self.idx_R
        #Advanced indexing to add Sigma_l and Sigma_r to the correct blocks of Sigma_tot. 
        # M[N_E, dim, dim] -> M[:, i[:, None], j[None, :]] extracts M[:, i, j] for all energies. 
        Sigma_tot[:, iL[:, None], iL[None, :]] += Sigma_l 
        Sigma_tot[:, iR[:, None], iR[None, :]] += Sigma_r

        GR = inv(z_center - self.H_C[None, :, :] - Sigma_tot)
        GA = GR.conj().transpose(0, 2, 1)
        Gamma_L = 1j * (Sigma_l - Sigma_l.conj().transpose(0, 2, 1))
        Gamma_R = 1j * (Sigma_r - Sigma_r.conj().transpose(0, 2, 1))

        def block(M, i, j): #M[:, i[:, None], j[None, :]] extracts M[:, i, j] for all energies. 
            return M[:, i[:, None], j[None, :]] 

        G_LR, G_LR_A = block(GR, iL, iR), block(GA, iL, iR)
        G_RL, G_RL_A = block(GR, iR, iL), block(GA, iR, iL)
        G_LL, G_LL_A = block(GR, iL, iL), block(GA, iL, iL)
        G_RR, G_RR_A = block(GR, iR, iR), block(GA, iR, iR)

        e_idx = np.where(np.tile([True, True, False, False], ny))[0]
        h_idx = np.where(np.tile([False, False, True, True], ny))[0]

        e = lambda M: M[:, e_idx[:, None], e_idx[None, :]]
        h = lambda M: M[:, h_idx[:, None], h_idx[None, :]]
        eh = lambda M: M[:, e_idx[:, None], h_idx[None, :]]
        he = lambda M: M[:, h_idx[:, None], e_idx[None, :]]

        def T(G1, B1, G2, B2):
            return np.trace(G1 @ B1 @ G2 @ B2, axis1=1, axis2=2).real

        #GaL is the lead under consideration. GaO is the other lead. So if we are considering the left lead, GaL = Gamma_L and GaO = Gamma_R.
        def side(name):
            if name == 'left':
                GL, GLA, GaL, GaO = G_RL, G_RL_A, Gamma_L, Gamma_R
                GLL, GLLA, GaLL = G_LL, G_LL_A, Gamma_L
            else:
                GL, GLA, GaL, GaO = G_LR, G_LR_A, Gamma_R, Gamma_L
                GLL, GLLA, GaLL = G_RR, G_RR_A, Gamma_R
            return {
                "ee": T(e(GaL), e(GL), e(GaO), e(GLA)),
                "hh": T(h(GaL), h(GL), h(GaO), h(GLA)),
                "eh_cross": T(e(GaL), eh(GL), h(GaO), he(GLA)),
                "he_cross": T(h(GaL), he(GL), e(GaO), eh(GLA)),
                "eh_local": T(e(GaLL), eh(GLL), h(GaLL), he(GLLA)),
                "he_local": T(h(GaLL), he(GLL), e(GaLL), eh(GLLA)),
            }

        if side_name is None:
            return {'left': side('left'), 'right': side('right')}
        if side_name == 'left':
            return side('left')
        if side_name == 'right':
            return side('right')
        raise ValueError(f"Invalid side_name: {side_name}. Must be 'left', 'right', or None.")
