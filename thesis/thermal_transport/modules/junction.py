"""FourTerminalJunction: assembles H_C + the four leads and computes transport channels."""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import block_diag

from .hamiltonians import onsite_block, Vx, Vy, make_row_hamiltonian, get_2d_hamiltonian, flat_site_idx
from .leads import Lead


def normal_lead_blocks(p):
    """
    Blocks of the semi-infinite L/R normal leads: (onsite, intra-layer H, inter-layer V).
    Single source of truth -- anything that needs the lead Hamiltonian (junction assembly,
    channel counting, diagnostics) must call this, so the pieces cannot drift apart.
    """
    soc_n = (p.alpha, p.beta) if p.soc_in_n else (0.0, 0.0)
    onsite_N = onsite_block(p.t_n, p.mu_n, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                            alpha=soc_n[0], beta=soc_n[1], twod=True)
    H_layer = make_row_hamiltonian(p.ny, onsite_N, Vy(p.t_n, *soc_n))
    V_hop = block_diag(*[Vx(p.t_n, *soc_n)] * p.ny)
    return onsite_N, H_layer, V_hop


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

        _, H_layer_N, V_n = normal_lead_blocks(p)
        V_coupling_LR = block_diag(*[self._bond(Vx, p.tc_barr)] * ny)

        self.lead_L = Lead('L', H_layer_N, V_n, V_coupling_LR, p, br=False, dual=True)
        self.lead_R = Lead('R', H_layer_N, V_n, V_coupling_LR, p, br=True)
        self.ribbon_top = self._make_ribbon(p.phi, p.tc_top)
        self.ribbon_bot = self._make_ribbon(0, p.tc_bot, is_bot=True)
        self.chain_top = self._make_single_chain(p.phi, is_bot=False)
        self.chain_bot = self._make_single_chain(0, is_bot=True)

        left_sites = [ix + nx * iy for iy in range(ny) for ix in [0]]
        right_sites = [ix + nx * iy for iy in range(ny) for ix in [nx - 1]]
        self.idx_L = flat_site_idx(left_sites)
        self.idx_R = flat_site_idx(right_sites)

    def _bond(self, V, tc):
        """
        Bond C <-> lead.  soc_bonds=False: pure hopping V(tc) (old behaviour).
        soc_bonds=True : the uniform-lattice bond V(t_c, alpha, beta) scaled as a whole
                         by tc/t_c, so tc = t_c means 'no barrier' and tc = 0 decouples
                         exactly (V(tc, alpha, beta) would keep the full SOC at tc = 0).
        """
        p = self.p
        if not p.soc_bonds:
            return V(tc)
        return (tc / p.t_c) * V(p.t_c, p.alpha, p.beta)

    def _make_ribbon(self, phi_lead, tc, is_bot=False):
        p = self.p
        soc_s = (p.alpha, p.beta) if p.soc_in_sc else (0.0, 0.0)
        onsite_SC = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, Bz=p.Bz_s, alpha=soc_s[0], beta=soc_s[1], twod=True)
        H_intra = make_row_hamiltonian(p.nx, onsite_SC, Vx(p.t_s, *soc_s))
        H_inter = block_diag(*([Vy(p.t_s, *soc_s)] * p.nx))
        V_coupling = block_diag(*([self._bond(Vy, tc)] * p.nx))
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, H_intra, H_inter, V_coupling, p, br=is_bot, dual=not is_bot)

    def _make_single_chain(self, phi_lead, is_bot=False):
        """One independent 1D SC chain per column (used when channels(is_ribbon=False)).
        Same Zeeman / SOC / bond conventions as _make_ribbon; only the intra-lead x-hopping
        is dropped, which is what makes the chains independent."""
        p = self.p
        soc_s = (p.alpha, p.beta) if p.soc_in_sc else (0.0, 0.0)
        onsite_sc = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, Bz=p.Bz_s,
                                 alpha=soc_s[0], beta=soc_s[1], twod=False)
        hop_y = Vy(p.t_s, *soc_s)
        tc = p.tc_bot if is_bot else p.tc_top
        hop_c = self._bond(Vy, tc)
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, onsite_sc, hop_y, hop_c, p, br=is_bot, dual=not is_bot)

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
                GL, GLA, GaL, GaO = G_LR, G_RL_A, Gamma_L, Gamma_R   # Tr[Γ_L G_LR Γ_R (G_LR)^†]
                GLL, GLLA, GaLL = G_LL, G_LL_A, Gamma_L
            else:
                GL, GLA, GaL, GaO = G_RL, G_LR_A, Gamma_R, Gamma_L   # Tr[Γ_R G_RL Γ_L (G_RL)^†]
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

    def thermal_T0(self, E=0.0, precompute_sigmas_n=None):
        """
        T -> 0 thermal / nonlocal electrical transmission at ONE energy (Gresta Eqs. 9-10):
            T_th = T^ee_RL + T^he_RL ,   T_el = T^ee_RL - T^he_RL .
        Same physics as thermal_electrical_T0(self.channels(E, 'right')), but one LU and only
        the 2*ny electron columns of lead L are solved (~3x faster, no dense inverse).

        precompute_sigmas_n : (Sigma_L, Sigma_R), each (4ny, 4ny), at this E -- reuse
                              across scans that only change SC-lead parameters.
        """
        p = self.p
        nx, ny = p.nx, p.ny
        dim = 4 * nx * ny
        z = E + 1j * p.eta

        if precompute_sigmas_n is None:
            zN = z * np.eye(4 * ny, dtype=complex)[None]
            SL, SR = self.lead_L.self_energy(zN)[0], self.lead_R.self_energy(zN)[0]
        else:
            SL, SR = precompute_sigmas_n
        zS = z * np.eye(4 * nx, dtype=complex)[None]
        ST, SB = self.ribbon_top.self_energy(zS)[0], self.ribbon_bot.self_energy(zS)[0]

        iL, iR = self.idx_L, self.idx_R
        A = -self.H_C.astype(complex)
        A[np.diag_indices(dim)] += z
        A[:4 * nx, :4 * nx] -= ST
        A[-4 * nx:, -4 * nx:] -= SB
        A[iL[:, None], iL[None, :]] -= SL
        A[iR[:, None], iR[None, :]] -= SR

        e = np.where(np.tile([True, True, False, False], ny))[0]
        h = np.where(np.tile([False, False, True, True], ny))[0]
        rhs = np.zeros((dim, len(e)), dtype=complex)
        rhs[iL[e], np.arange(len(e))] = 1.0                       # electron columns of L
        X = lu_solve(lu_factor(A, overwrite_a=True, check_finite=False), rhs, check_finite=False)

        Gamma_L = 1j * (SL - SL.conj().T)
        Gamma_R = 1j * (SR - SR.conj().T)
        GLe = Gamma_L[e[:, None], e[None, :]]
        G_ee = X[iR[e]]                                           # G^{ee}_{RL}
        G_he = X[iR[h]]                                           # G^{he}_{RL}
        T_ee = np.trace(Gamma_R[e[:, None], e[None, :]] @ G_ee @ GLe @ G_ee.conj().T).real
        T_he = np.trace(Gamma_R[h[:, None], h[None, :]] @ G_he @ GLe @ G_he.conj().T).real
        return T_ee + T_he, T_ee - T_he