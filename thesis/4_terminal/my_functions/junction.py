"""FourTerminalJunction: assembles H_C + the four leads and computes transport channels."""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag

from .hamiltonians import (model_onsite, model_hop, region_alpha, bond_alpha, bond_m0, make_row_hamiltonian,
                           get_2d_hamiltonian, flat_site_idx)
from .leads import Lead


class FourTerminalJunction:
    """
    Geometry: central nx*ny normal region, with SC ribbons (width nx) attached
    along the full top and bottom edges, and normal leads attached at the
    left/right edges (Sancho-Rubio decimated in x).

    Site index = ix + nx*iy.  "Top" is the row iy = 0 (first 4*nx matrix
    indices), "bottom" is iy = ny-1.  The Hamiltonian model is chosen by
    p.model ('rashba' or 'dirac'); all blocks come from hamiltonians.model_*.
    """

    def __init__(self, p):
        self.p = p
        self._build()

    def _build(self):
        p = self.p
        nx, ny = p.nx, p.ny

        # --- central region: store the blocks; the dense H_C is built only on demand ---
        self.onsite_C = model_onsite(p, 'c')
        self.Vx_C = model_hop(p, 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
        self.Vy_C = model_hop(p, 'y', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
        self._H_C = None

        # --- normal leads: one lead cell = a column of ny sites chained along y ---
        onsite_N = model_onsite(p, 'n')
        H_layer_N = make_row_hamiltonian(ny, onsite_N, model_hop(p, 'y', p.t_n, p.m0_n, alpha=region_alpha(p, 'n')))
        V_n = block_diag(*[model_hop(p, 'x', p.t_n, p.m0_n, alpha=region_alpha(p, 'n'))] * ny)                        # H_{x,x+1} inside the N leads
        V_b = block_diag(*[model_hop(p, 'x', p.tc_barr, bond_m0(p.m0_n, p.m0_c), alpha=bond_alpha(p, 'n', 'c'))] * ny)  # H_{x,x+1} across the barrier bond

        # Left lead, cells x = -1, -2, ...: outward hop H_{-1,-2} = V_n^dag, coupling H_{-1,0} = V_b
        self.lead_L = Lead('L', H_layer_N, V_n, V_b, p, br=False, dual=True)
        # Right lead, cells x = nx, nx+1, ...: outward hop H_{nx,nx+1} = V_n, coupling H_{nx,nx-1} = V_b^dag
        self.lead_R = Lead('R', H_layer_N, V_n, V_b, p, br=True, dual=False)
        self.ribbon_top = self._make_ribbon(p.phi, p.tc_top, is_bot=False)
        self.ribbon_bot = self._make_ribbon(0.0, p.tc_bot, is_bot=True)
        self.chain_top = self._make_single_chain(p.phi, is_bot=False)
        self.chain_bot = self._make_single_chain(0, is_bot=True)

        left_sites = [ix + nx * iy for iy in range(ny) for ix in [0]]
        right_sites = [ix + nx * iy for iy in range(ny) for ix in [nx - 1]]
        self.idx_L = flat_site_idx(left_sites)
        self.idx_R = flat_site_idx(right_sites)

    @property
    def H_C(self):
        """Dense central Hamiltonian (4 nx ny)^2, built on first access (~370 MB at 60 x 20).
        Only the dense solvers (channels, FastPhaseSweep) need it; RGF does not."""
        if self._H_C is None:
            p = self.p
            self._H_C = get_2d_hamiltonian(p.nx, p.ny, self.onsite_C, self.Vx_C, self.Vy_C)
        return self._H_C

    def slice_blocks_x(self):
        """
        Blocks of H_C for slicing along x (slice = column ix, sites iy = 0..ny-1):
        H_slice (4ny x 4ny) and V = H[ix, ix+1] (4ny x 4ny).  Used by the RGF solver.
        """
        ny = self.p.ny
        return (make_row_hamiltonian(ny, self.onsite_C, self.Vy_C),
                block_diag(*[self.Vx_C] * ny))

    def _make_ribbon(self, phi_lead, tc, is_bot=False):
        """
        Semi-infinite SC half-plane of width nx.  Top ribbon (y = -1, -2, ...)
        grows towards -y -> dual=True, br=False.  
        Bottom ribbon (y = ny, ny+1, ...) grows towards +y -> dual=False, br=True.  
        Use this builder for ANY ribbon (e.g. at a reference phase) so the orientation is never duplicated.
        """
        p = self.p
        onsite_SC = model_onsite(p, 's', phi=phi_lead)
        H_intra = make_row_hamiltonian(p.nx, onsite_SC, model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's')))
        V_s = block_diag(*([model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))] * p.nx))                     # H_{y,y+1} inside ribbon
        V_c = block_diag(*([model_hop(p, 'y', tc, bond_m0(p.m0, p.m0_c), alpha=bond_alpha(p, 's', 'c'))] * p.nx))       # H_{y,y+1} ribbon<->centre bond
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, H_intra, V_s, V_c, p, br=is_bot, dual=not is_bot)

    def _make_single_chain(self, phi_lead, is_bot=False):
        p = self.p
        onsite_sc = model_onsite(p, 's', phi=phi_lead, twod=False, Bxy=0.0, theta_z=0.0)
        hop_y = model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))                                   # H_{y,y+1}
        tc = p.tc_bot if is_bot else p.tc_top
        hop_c = model_hop(p, 'y', tc, bond_m0(p.m0, p.m0_c), alpha=bond_alpha(p, 's', 'c'))                     # H_{y,y+1} across the bond
        name = 'sc_bot' if is_bot else 'sc_top'
        return Lead(name, onsite_sc, hop_y, hop_c, p, br=is_bot, dual=not is_bot)

    def _z_batches(self, E_sweep):
        p = self.p
        dim = 4 * p.nx * p.ny
        z1 = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(4 * p.ny, dtype=complex)[None, :, :]
        zN = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(dim, dtype=complex)[None, :, :]
        return z1, zN

    def channels(self, E_sweep, side_name=None, precompute_sigmas_n=None, is_ribbon=True):
        """
        Dense reference solver: EC/CAR/LAR transmission channels for the requested lead(s).
        Returns the same keys as FastPhaseSweep and RGFFourTerminal:
        'ee', 'hh', 'eh_cross', 'he_cross', 'eh_local', 'he_local'.

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

        def side(name):
            # T_{i<-j} = Tr[Gamma_i G^R_{ij} Gamma_j G^A_{ji}]  (i = this lead, j = the other one)
            # Same keys as FastPhaseSweep / RGFFourTerminal.channels_at_phi.
            if name == 'left':
                GL, GLA, GaL, GaO = G_LR, G_RL_A, Gamma_L, Gamma_R
                GLL, GLLA, GaLL = G_LL, G_LL_A, Gamma_L
            else:
                GL, GLA, GaL, GaO = G_RL, G_LR_A, Gamma_R, Gamma_L
                GLL, GLLA, GaLL = G_RR, G_RR_A, Gamma_R
            return {
                "ee":       T(e(GaL), e(GL), e(GaO), e(GLA)),        # e(other) -> e(this)
                "hh":       T(h(GaL), h(GL), h(GaO), h(GLA)),        # h(other) -> h(this)
                "eh_cross": T(e(GaL), eh(GL), h(GaO), he(GLA)),      # h(other) -> e(this)  CAR
                "he_cross": T(h(GaL), he(GL), e(GaO), eh(GLA)),      # e(other) -> h(this)  CAR
                "eh_local": T(e(GaLL), eh(GLL), h(GaLL), he(GLLA)),  # h(this)  -> e(this)  LAR
                "he_local": T(h(GaLL), he(GLL), e(GaLL), eh(GLLA)),  # e(this)  -> h(this)  LAR
            }

        if side_name is None:
            return {'left': side('left'), 'right': side('right')}
        if side_name == 'left':
            return side('left')
        if side_name == 'right':
            return side('right')
        raise ValueError(f"Invalid side_name: {side_name}. Must be 'left', 'right', or None.")
