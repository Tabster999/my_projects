"""
fourterminal_all.py -- single-file build of the whole 4-terminal NEGF package
=============================================================================

Everything in one file so it can be moved / attached in one piece:

    Params                 parameters (dataclass)
    onsite_block, Vx, Vy   Nambu-spin 4x4 building blocks
    make_row_hamiltonian, get_2d_hamiltonian, flat_site_idx
    Lead                   Sancho-Rubio surface GF / self-energy
    FourTerminalJunction   H_C + 4 leads + dense channels()
    FastPhaseSweep         dense phase-sweep engine (reference path)
    RGFFourTerminal        RGF + exact subspace-Dyson solver  (OPT 0-5)
    transport helpers      f_electron/f_hole, dc_current_channels,
                           conductance_matrix, partial_G_vectorized, ...
    plotting helpers       style_axis, print_params
    analysis helpers       compute_phase_bias_maps, scan_grid, ...

Basis: Psi = (c_up, c_down, c_down^dag, -c_up^dag)   (Scharf-Pientka)

NOTE ON `myf.`
--------------
The RGF section keeps the `myf.` prefixes from the multi-file layout so it
stays diff-able against the package version.  `myf` is aliased to THIS module
near the bottom (`myf = sys.modules[__name__]`), so `myf.Params(...)`,
`myf.FourTerminalJunction(...)` etc. keep working -- including in your
existing driver scripts, unedited.

This file defines only; it runs nothing on import.  Driver cells live in your
own script.

--------------------------------------------------------------------------
WHY A PLAIN RGF DOES NOT WORK HERE
--------------------------------------------------------------------------
RGF needs A = z - H_C - Sigma_tot BLOCK TRIDIAGONAL along the sweep
direction.  Here it is not, in *either* direction:

  * sweep along x (slices = columns): Sigma_T / Sigma_B are dense in x,
    because the SC ribbons carry intra-lead hopping Vx(t_s) along x.
  * sweep along y (slices = rows):    Sigma_L / Sigma_R are dense in y,
    because the normal leads carry intra-lead hopping Vy(t_n) along y.

A lead breaks tridiagonality iff it attaches to more than ONE slice along
the sweep axis.  With leads on all four sides of a rectangle there is no
sweep direction where all four are single-slice.

THE FIX (exact, not an approximation)
  A0 = z - H_C - Sigma_B     <- block tridiagonal in y, and phi-independent
  W  = P^dag (Sigma_L + Sigma_R + Sigma_T) P
  S  = {row iy=0} U {col ix=0} U {col ix=nx-1},  dim_S = 4*(nx + 2*ny - 2)

W lives entirely inside S, so the Dyson equation closes on S exactly:
  g = P G P^dag = g0 + g0 Sigma_S g   =>   g = (g0^-1 - Sigma_S)^-1
with g0 = P A0^-1 P^dag from the y-RGF.  G_LL, G_LR, G_RL, G_RR are inside
S, so nothing outside S is ever needed.  Algebraically identical to the
dense solve (validated to ~1e-13 with SOC and Zeeman both off and on, at
five phases, both sides, and under a left/right bias).

PHASE SWEEPS
  h = (g0^-1 - Sigma_L - Sigma_R)^-1              (phi-independent)
  M = I_T - h_TT Sigma_T(phi)
  G_XY = h_XY + h_XT Sigma_T(phi) M^-1 h_TY,   X,Y in {L,R}

OPTIMIZATIONS (all exact)
  OPT 0  shared Sancho decimations (4 -> 2 when lead_L/lead_R and the two
         ribbons coincide; DETECTED, not assumed -- a bias that shifts the
         lead chemical potentials falls back to separate decimations)
  OPT 1  block-structured V products, V = kron(I_nx, Vy)   -> O(nx) saving
  OPT 2  Sigma_S never materialised densely (2 nonzero column blocks)
  OPT 3  Sigma_T_ref pre-split by gauge weight -> O(1) per-phi rotation
  OPT 4  A_diag deduplicated (was ny identical copies; ~134 MB at
         nx=12, ny=30, N_E=121)
  OPT 5  h_TL / h_TR stacked -> one LU per phi instead of two

  chunk is a SPEED knob, not only memory: 8-16 measured fastest.
"""
#%% --- IMPORTS ---
from calendar import c
import sys, os
from dataclasses import dataclass, replace
import time
from turtle import color

import numpy as np
from numpy.linalg import inv, solve
from scipy.linalg import block_diag
from scipy.linalg import solve as batched_solve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    from tqdm import tqdm
except Exception:                      # tqdm optional
    def tqdm(x, *a, **k):
        return x

try:
    from IPython.display import display, Math
except Exception:                      # non-IPython session
    display = print
    Math = str



#%% --- PARAMS ---
@dataclass
class Params:
    # --- geometry ---
    nx: int = 10
    ny: int = 1

    # --- normal region (central, x-hop) ---
    t_n: float = 1.0
    mu_n: float = 0.0

    # --- central 2D region ---
    t_c: float = 1.0
    mu_c: float = 0.0

    # --- SC ribbons (top/bottom) ---
    t_s: float = 1.0
    mu_s: float = 0.0
    delta: float = 0.1
    phi: float = np.pi

    # --- couplings ---
    tc_top: float = 0.6
    tc_bot: float = 0.6
    tc_barr: float = 0.5

    # --- spin-orbit / Zeeman ---
    # CHANGED (v7): region assignment as in Gresta et al. Eqs. (3)-(6)
    #   alpha, beta         -> EVERYWHERE: C, L, R, T, B and all coupling bonds (uniform material)
    #   Bz, Bxy, theta_z    -> SC ribbons T and B ONLY
    alpha: float = 0.0    # Rashba           (everywhere)
    beta: float = 0.0     # Dresselhaus      (everywhere)
    Bz: float = 0.0       # out-of-plane Zeeman   (T, B only)
    Bxy: float = 0.0      # in-plane Zeeman magnitude (T, B only)
    theta_z: float = 0.0  # in-plane Zeeman angle     (T, B only)

    # --- numerics ---
    eta: float = 1e-5
    max_iter: int = 450
    tol: float = 1e-14
    kT: float = 1e-3


#%% --- HAMILTONIANS ---
def onsite_block(t, mu, delta=0.0, phi=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0,
                  alpha=0.0, beta=0.0, twod=False):
    """4x4 onsite block: kinetic energy, SC pairing, and Zeeman terms."""
    ons = np.zeros((4, 4), dtype=np.complex128)
    onsite_val = ((4 * t - mu) if twod else (2 * t - mu)) + (alpha**2 + beta**2) / 4
    pairing = delta * np.exp(1j * phi)
    Bxy_c = Bxy * np.exp(1j * theta_z)

    ons[0, 0] = onsite_val + Bz
    ons[1, 1] = onsite_val - Bz
    ons[2, 2] = -onsite_val + Bz
    ons[3, 3] = -onsite_val - Bz

    ons[0, 1] = Bxy_c
    ons[1, 0] = np.conj(Bxy_c)
    ons[2, 3] = Bxy_c
    ons[3, 2] = np.conj(Bxy_c)

    ons[0, 2] = pairing
    ons[1, 3] = pairing
    ons[2, 0] = pairing.conj()
    ons[3, 1] = pairing.conj()
    return ons


def Vx(t, alpha=0.0, beta=0.0):
    """4x4 hopping block along x (includes Rashba/Dresselhaus SOC)."""
    s = 0.5 * (alpha - beta)
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    hop[0, 1] = -s
    hop[1, 0] = s
    hop[2, 3] = s
    hop[3, 2] = -s
    return hop


def Vy(t, alpha=0.0, beta=0.0):
    """4x4 hopping block along y (includes Rashba/Dresselhaus SOC)."""
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array(
        [[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
        dtype=np.complex128,
    ) * (alpha + beta) / 2
    return hop + soc


def make_row_hamiltonian(n, onsite, hop_x):
    """Build a 1D chain of n sites (dim = 4n) with the given onsite and x-hopping block."""
    dim = 4 * n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = onsite
    for i in range(n - 1):
        H[4 * i:4 * i + 4, 4 * i + 4:4 * i + 8] = hop_x
        H[4 * i + 4:4 * i + 8, 4 * i:4 * i + 4] = hop_x.conj().T
    return H


def get_2d_hamiltonian(nx, ny, onsite, Vx_block, Vy_block, dof=4):
    """Build a finite nx x ny 2D lattice Hamiltonian (dim = dof*nx*ny)."""
    dim = dof * nx * ny
    H = np.zeros((dim, dim), dtype=np.complex128)

    def idx(ix, iy):
        return ix + nx * iy

    for iy in range(ny):
        for ix in range(nx):
            j = idx(ix, iy)
            H[dof * j:dof * (j + 1), dof * j:dof * (j + 1)] = onsite

    for iy in range(ny):
        for ix in range(nx - 1):
            i, j = idx(ix, iy), idx(ix + 1, iy)
            H[dof * i:dof * (i + 1), dof * j:dof * (j + 1)] = Vx_block
            H[dof * j:dof * (j + 1), dof * i:dof * (i + 1)] = Vx_block.conj().T

    for iy in range(ny - 1):
        for ix in range(nx):
            i, j = idx(ix, iy), idx(ix, iy + 1)
            H[dof * i:dof * (i + 1), dof * j:dof * (j + 1)] = Vy_block
            H[dof * j:dof * (j + 1), dof * i:dof * (i + 1)] = Vy_block.conj().T

    return H


def flat_site_idx(sites, dof=4):
    """
    Expand a list of physical site indices (convention: site = ix + nx*iy)
    into the corresponding flat matrix-element indices.
    """
    return np.concatenate([np.arange(dof * s, dof * (s + 1)) for s in sites])


# NEW ---------------------------------------------------------------------
# Region assignment of the model: the ONLY place that decides which region
# carries SOC and which carries Zeeman.  Junction, FastPhaseSweep, RGF and
# the thermal-map code all build their blocks through these three functions.
def central_onsite(p):
    """Onsite block of the central region C: SOC (alpha, beta), no Zeeman."""
    return onsite_block(p.t_c, p.mu_c, alpha=p.alpha, beta=p.beta, twod=True)


def normal_lead_blocks(p):
    """
    L/R lead unit cell (one column of ny sites): SOC, no Zeeman.          # CHANGED (v7): + SOC
    Returns H_layer_N, V_n = H[x, x+1], V_coupling_LR.
    """
    onsite_N = onsite_block(p.t_n, p.mu_n, alpha=p.alpha, beta=p.beta, twod=True)       # CHANGED (v7)
    H_layer_N = make_row_hamiltonian(p.ny, onsite_N, Vy(p.t_n, p.alpha, p.beta))        # CHANGED (v7)
    V_n = block_diag(*[Vx(p.t_n, p.alpha, p.beta)] * p.ny)                              # CHANGED (v7)
    V_coupling_LR = block_diag(*[Vx(p.tc_barr, p.alpha, p.beta)] * p.ny)                # CHANGED (v7): SOC on the C-L/R bonds
    return H_layer_N, V_n, V_coupling_LR


def sc_onsite(p, phi_lead):
    """Onsite block of the SC ribbons T/B: pairing + Zeeman + SOC."""        # CHANGED (v7): + SOC
    return onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead,
                        Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                        alpha=p.alpha, beta=p.beta, twod=True)                          # CHANGED (v7)


def sc_ribbon_blocks(p, phi_lead, tc):                                                   # NEW (v7)
    """
    SC ribbon cell (one row of nx sites) with SOC.  Returns
    H_intra, H_inter = H[y, y+1], V_coupling = H[ribbon-cell adjacent to C, C-row] for the TOP ribbon
    convention (use br=True for the bottom ribbon, exactly as before).
    """
    H_intra = make_row_hamiltonian(p.nx, sc_onsite(p, phi_lead), Vx(p.t_s, p.alpha, p.beta))
    H_inter = block_diag(*([Vy(p.t_s, p.alpha, p.beta)] * p.nx))
    V_coupling = block_diag(*([Vy(tc, p.alpha, p.beta)] * p.nx))                        # SOC on the C-T/B bonds
    return H_intra, H_inter, V_coupling
# END NEW -----------------------------------------------------------------


#%% --- LEADS ---
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
    reverse : bool                                                   # NEW
        V_hop is always H[cell i, cell i+1] in the GLOBAL coordinate.
        surface_gf needs the hopping from the surface cell INTO the bulk,
        i.e. Sigma_surf = H[s, s+1] g H[s+1, s] for a lead whose bulk lies
        at increasing index (R at x > nx-1, B at y > ny-1).  For a lead whose
        bulk lies at decreasing index (L at x < 0, T at y < 0) the inward
        hopping is H[s, s-1] = V_hop^dag, so reverse=True stores V_hop^dag.
        Identical results whenever V_hop is Hermitian (no SOC in the lead).
    """

    def __init__(self, name, H_onsite, V_hop, V_coupling, p, br=False, reverse=False):   # CHANGED
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        Vh = np.asarray(V_hop, dtype=complex)                                            # CHANGED
        self.V_hop = Vh.conj().T if reverse else Vh                                      # CHANGED
        self.reverse = reverse                                                           # NEW
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


#%% --- JUNCTION ---
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

        onsite_C = central_onsite(p)                                          # CHANGED: no Zeeman in C
        self.H_C = get_2d_hamiltonian(nx, ny, onsite_C, Vx(p.t_c, p.alpha, p.beta),
                                       Vy(p.t_c, p.alpha, p.beta))

        H_layer_N, V_n, V_coupling_LR = normal_lead_blocks(p)                 # CHANGED: no SOC/Zeeman in L, R

        self.lead_L = Lead('L', H_layer_N, V_n, V_coupling_LR, p, br=False, reverse=True)   # CHANGED: bulk at x<0
        self.lead_R = Lead('R', H_layer_N, V_n, V_coupling_LR, p, br=True)
        self.ribbon_top = self._make_ribbon(p.phi, p.tc_top)
        self.ribbon_bot = self._make_ribbon(0, p.tc_bot, is_bot=True)       
        left_sites = [ix + nx * iy for iy in range(ny) for ix in [0]]
        right_sites = [ix + nx * iy for iy in range(ny) for ix in [nx - 1]]
        self.idx_L = flat_site_idx(left_sites)
        self.idx_R = flat_site_idx(right_sites)

    def _make_ribbon(self, phi_lead, tc, is_bot=False):
        p = self.p
        H_intra, H_inter, V_coupling = sc_ribbon_blocks(p, phi_lead, tc)      # CHANGED (v7): SOC in T/B
        return Lead('ribbon', H_intra, H_inter, V_coupling, p, br=is_bot, reverse=not is_bot)   # CHANGED

    def _z_batches(self, E_sweep):
        p = self.p
        dim = 4 * p.nx * p.ny
        z1 = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(4 * p.ny, dtype=complex)[None, :, :]
        zN = (E_sweep + 1j * p.eta)[:, None, None] * np.eye(dim, dtype=complex)[None, :, :]
        return z1, zN

    def channels(self, E_sweep, side_name=None, precompute_sigmas_n=None):
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

        if precompute_sigmas_n is None:
            Sigma_l = self.lead_L.self_energy(z_lead_N)
            Sigma_r = self.lead_R.self_energy(z_lead_N)
        else:
            Sigma_l, Sigma_r = precompute_sigmas_n

        Sigma_t = self.ribbon_top.self_energy(z_lead_SC)
        Sigma_b = self.ribbon_bot.self_energy(z_lead_SC)

        Sigma_tot = np.zeros((N_E, dim, dim), dtype=complex)
        Sigma_tot[:, :4 * nx, :4 * nx] += Sigma_t
        Sigma_tot[:, -4 * nx:, -4 * nx:] += Sigma_b
        iL, iR = self.idx_L, self.idx_R
        #Using advanced indexing to add Sigma_l and Sigma_r to the correct blocks of Sigma_tot. M[N_E, dim, dim] --> M[:, i[:, None], j[None, :]] extracts M[:, i, j] for all energies. 
        Sigma_tot[:, iL[:, None], iL[None, :]] += Sigma_l 
        Sigma_tot[:, iR[:, None], iR[None, :]] += Sigma_r

        GR = inv(z_center - self.H_C[None, :, :] - Sigma_tot)
        GA = GR.conj().transpose(0, 2, 1)
        Gamma_L = 1j * (Sigma_l - Sigma_l.conj().transpose(0, 2, 1))          # CHANGED: was -2*Im(Sigma), elementwise
        Gamma_R = 1j * (Sigma_r - Sigma_r.conj().transpose(0, 2, 1))          # CHANGED

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
            return np.einsum('nij,njk,nkl,nli->n', G1, B1, G2, B2).real

        #GaL is the lead under consideration. GaO is the other lead. So if we are considering the left lead, GaL = Gamma_L and GaO = Gamma_R.
        def side(name):
            # CHANGED: side = SOURCE lead.  'left' returns T^{ab}_RL = Tr[Gamma_R^a G^r_RL Gamma_L^b G^a_LR],
            # with (G^r_RL)^dag = G^a_LR.  Before: advanced block G^a_RL and Gamma order swapped.
            if name == 'left':
                GL, GLA, GaL, GaO = G_RL, G_LR_A, Gamma_L, Gamma_R            # CHANGED: G_RL_A -> G_LR_A
                GLL, GLLA, GaLL = G_LL, G_LL_A, Gamma_L
            else:
                GL, GLA, GaL, GaO = G_LR, G_RL_A, Gamma_R, Gamma_L            # CHANGED: G_LR_A -> G_RL_A
                GLL, GLLA, GaLL = G_RR, G_RR_A, Gamma_R
            return {
                "ee": T(e(GaO), e(GL), e(GaL), e(GLA)),                       # CHANGED: GaL <-> GaO
                "hh": T(h(GaO), h(GL), h(GaL), h(GLA)),                       # CHANGED
                "eh_cross": T(e(GaO), eh(GL), h(GaL), he(GLA)),               # CHANGED
                "he_cross": T(h(GaO), he(GL), e(GaL), eh(GLA)),               # CHANGED
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


#%% --- FAST PHASE SWEEP (dense reference path) ---
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
        p = self.p
        H_intra, H_inter, V_coupling = sc_ribbon_blocks(p, phi_lead, tc)      # CHANGED (v7): SOC in T/B
        return Lead('ribbon', H_intra, H_inter, V_coupling, p, br=False, reverse=True)   # CHANGED: top, bulk at y<0

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
                # CHANGED: index-consistent order, side = SOURCE lead ('left' -> T^{ab}_RL)
                "ee": T(Gamma_other_e, G_ee, Gamma_own_e, dagger(G_ee)),
                "hh": T(Gamma_other_h, G_hh, Gamma_own_h, dagger(G_hh)),
                "eh_cross": T(Gamma_other_e, G_eh, Gamma_own_h, dagger(G_eh)),
                "he_cross": T(Gamma_other_h, G_he, Gamma_own_e, dagger(G_he)),
                "eh_local": T(Gamma_own_e, G_self_eh, Gamma_own_h, dagger(G_self_eh)),
                "he_local": T(Gamma_own_h, G_self_he, Gamma_own_e, dagger(G_self_he)),
            }

        if side_name is None:
            return {'left': side('left'), 'right': side('right')}
        return side(side_name)


#%% --- RGF (optimized: OPT 0-5) ---
# ----------------------------------------------------------------------
#  index bookkeeping for the subspace S
# ----------------------------------------------------------------------
class _SubspaceIndex:
    """
    Site-level layout of S = {row 0} U {col 0} U {col nx-1}, ordered by
    slice iy, then by the order the sites appear in that slice.

    Attributes
    ----------
    sel_sites[iy] : list of ix kept in slice iy
    sel_flat[iy]  : local flat indices (into a 4*nx slice) of those sites
    posL, posR    : flat indices into S of column ix=0 / ix=nx-1,
                    ordered iy ascending  -> matches junction.idx_L / idx_R
    posT          : flat indices into S of row iy=0, ordered ix ascending
                    -> matches the Sigma_T / ribbon ordering
    dim_S         : 4 * (nx + 2*ny - 2)
    """

    def __init__(self, nx, ny, dof=4):
        if nx < 2:
            raise ValueError("nx must be >= 2 (left and right leads would coincide)")
        self.nx, self.ny, self.dof = nx, ny, dof

        self.sel_sites = []     # which columns (ix) are kept in each row (iy)
        for iy in range(ny):
            self.sel_sites.append(list(range(nx)) if iy == 0 else [0, nx - 1])

        self.sel_flat = [
            np.concatenate([np.arange(dof * ix, dof * (ix + 1)) for ix in s])
            for s in self.sel_sites
        ]                       # four numbers per ix (i.e. site 3 --> 12,13,14,15)
        self.widths = [len(f) for f in self.sel_flat]   # number of matrix indices in each row
        self.offsets = np.concatenate([[0], np.cumsum(self.widths)]).astype(int) # where each row starts in the flattened S
        self.dim_S = int(self.offsets[-1])  # total size of S

        # site -> position within S
        pos = {}
        k = 0
        for iy in range(ny):
            for ix in self.sel_sites[iy]:
                pos[(iy, ix)] = k
                k += 1
        self._pos = pos

        f = lambda iy, ix: np.arange(dof * pos[(iy, ix)], dof * (pos[(iy, ix)] + 1))
        self.posL = np.concatenate([f(iy, 0) for iy in range(ny)])
        self.posR = np.concatenate([f(iy, nx - 1) for iy in range(ny)])
        self.posT = np.concatenate([f(0, ix) for ix in range(nx)])

    def block_slice(self, iy):
        """Index of row iy's block INSIDE the flattened S.  Use to index into g0 or h."""
        return slice(int(self.offsets[iy]), int(self.offsets[iy + 1]))


# ----------------------------------------------------------------------
#  OPT 1: block-structured products with V = kron(I_nx, Vy)
# ----------------------------------------------------------------------
# The inter-slice hopping V = block_diag([Vy]*nx) is site-local in x, so it
# never mixes different ix.  Treating it as a dense (4nx x 4nx) matrix costs
# O((4nx)^3) per product; using the block structure costs O(nx * 4^3), i.e.
# an O(nx) saving on every V^dag g V, V g V^dag, V^dag C and V C.
#
# Done with reshape + broadcast matmul: numpy broadcasts a (4,4) matrix
# against a (..., nx, 4, W) stack, contracting only the 4-axis.

def _left_mul(M4, X, nx):
    """kron(I_nx, M4) @ X   for X of shape (..., 4*nx, W)."""
    dof = M4.shape[0]
    return np.matmul(M4, X.reshape(*X.shape[:-2], nx, dof, X.shape[-1])).reshape(X.shape)


def _right_mul(X, M4, nx):
    """X @ kron(I_nx, M4)   for X of shape (..., R, 4*nx)."""
    dof = M4.shape[0]
    return np.matmul(X.reshape(*X.shape[:-1], nx, dof), M4).reshape(X.shape)


# ----------------------------------------------------------------------
#  core: g0 = P A0^-1 P^dag  via a y-direction RGF
# ----------------------------------------------------------------------
def _g0_on_subspace(z, H_row, V, Sigma_B, sub, Vy4=None):
    """
    Batched over energy.

    Parameters
    ----------
    z        : (N_E,) complex, already E + i*eta
    H_row    : (d, d) intra-slice Hamiltonian of one y-row (d = 4*nx)
    V        : (d, d) inter-slice hopping, H[iy, iy+1] = V
    Sigma_B  : (N_E, d, d) bottom-ribbon self-energy (sits on slice ny-1)
    sub      : _SubspaceIndex
    Vy4      : (4, 4) generator of V.  If given, all V products use the
               block form (OPT 1).  If None, falls back to dense products
               -- same maths, just slower.  Handy for cross-checking.

    Returns
    -------
    g0 : (N_E, dim_S, dim_S)
    """
    ny = sub.ny
    d = H_row.shape[0]
    nx = d // (Vy4.shape[0] if Vy4 is not None else 4)
    N_E = z.shape[0]
    Vd = V.conj().T
    Id = np.eye(d, dtype=complex)

    # --- OPT 1: block V products (V never mixes different ix) ---
    if Vy4 is not None:
        Vy4d = Vy4.conj().T
        VdgV = lambda g: _right_mul(_left_mul(Vy4d, g, nx), Vy4, nx)   # V^dag g V
        VgVd = lambda g: _right_mul(_left_mul(Vy4, g, nx), Vy4d, nx)   # V g V^dag
        VdC  = lambda C: _left_mul(Vy4d, C, nx)                        # V^dag @ C
        VC   = lambda C: _left_mul(Vy4, C, nx)                         # V     @ C
    else:                                                              # dense fallback
        VdgV = lambda g: Vd @ g @ V
        VgVd = lambda g: V @ g @ Vd
        VdC  = lambda C: Vd @ C
        VC   = lambda C: V @ C

    # --- OPT 4: keep ONE copy of the diagonal block ---
    # A_diag was (ny, N_E, d, d) holding ny identical copies of `base`; only
    # the last slice differs, by -Sigma_B.  That was ~134 MB of pure
    # duplication at nx=12, ny=30, N_E=121.
    base = z[:, None, None] * Id[None] - H_row[None]   # broadcasting over energy
    A_last = base - Sigma_B                            # only slice ny-1 carries Sigma_B
    A_of = lambda iy: (A_last if iy == ny - 1 else base)

    # gL sweeps from top (iy = 0) to bottom (iy = ny - 1), gR sweeps from bottom to top
    gL = np.empty((ny, N_E, d, d), dtype=complex)
    gR = np.empty((ny, N_E, d, d), dtype=complex)

    gL[0] = inv(A_of(0))
    for i in range(1, ny):
        gL[i] = inv(A_of(i) - VdgV(gL[i - 1]))

    gR[ny - 1] = inv(A_of(ny - 1))
    for i in range(ny - 2, -1, -1):
        gR[i] = inv(A_of(i) - VgVd(gR[i + 1]))

    g0 = np.zeros((N_E, sub.dim_S, sub.dim_S), dtype=complex)

    for j in range(ny):
        # build fully dressed diagonal block
        Aj = A_of(j).copy()
        if j > 0:
            Aj -= VdgV(gL[j - 1])
        if j < ny - 1:
            Aj -= VgVd(gR[j + 1])
        G_jj = inv(Aj)

        selj = sub.sel_flat[j]
        cj = sub.block_slice(j)

        C0 = G_jj[:, :, selj]                       # (N_E, d, w_j)
        g0[:, sub.block_slice(j), cj] = C0[:, selj, :]

        # propagate upward (i > j) with gR
        C = C0
        for i in range(j + 1, ny):
            C = gR[i] @ VdC(C)
            g0[:, sub.block_slice(i), cj] = C[:, sub.sel_flat[i], :]

        # propagate downward (i < j) with gL
        C = C0
        for i in range(j - 1, -1, -1):
            C = gL[i] @ VC(C)
            g0[:, sub.block_slice(i), cj] = C[:, sub.sel_flat[i], :]

    return g0


# ----------------------------------------------------------------------
#  main solver
# ----------------------------------------------------------------------
class RGFFourTerminal:
    """
    RGF + exact subspace-Dyson solver for FourTerminalJunction.

    Drop-in replacement for the dense path:

        junction = FourTerminalJunction(p)
        rgf = RGFFourTerminal(junction, E_sweep, phi_ref=0.0)
        ch  = rgf.channels_at_phi(p.phi, side_name='left')

    matches

        FastPhaseSweep(junction, E_sweep, phi_ref=0.0).channels_at_phi(p.phi, 'left')

    to machine precision.

    Parameters
    ----------
    junction   : FourTerminalJunction  (supplies leads, ribbons, params)
    E_sweep    : (N_E,) real energies
    phi_ref    : reference phase for the single Sancho-Rubio decimation of
                 the top ribbon.  phi_ref = 0 additionally lets the top and
                 bottom ribbons share one decimation (see OPT 0).
    chunk      : energies per RGF batch.  Smaller is usually FASTER (cache
                 residency), not just lighter on memory: 8-16 measured best.
                 None picks a size targeting ~`mem_budget_gb`.
    mem_budget_gb : soft target for the RGF working set when chunk is None.
    """

    def __init__(self, junction, E_sweep, phi_ref=0.0, chunk=None,
                 mem_budget_gb=2.0, verbose=False):
        self.junction = junction
        p = junction.p
        self.p = p
        self.nx, self.ny = p.nx, p.ny
        self.E_sweep = np.asarray(E_sweep, dtype=float)
        self.phi_ref = phi_ref
        self.N_E = len(self.E_sweep)

        self.sub = _SubspaceIndex(p.nx, p.ny)
        d = 4 * p.nx

        if chunk is None:
            # working set ~ 2*ny*chunk*d^2 (gL,gR) + chunk*dim_S^2
            per_E = (2 * p.ny * d * d + self.sub.dim_S ** 2) * 16
            chunk = max(1, int(mem_budget_gb * 1e9 / per_E))
        self.chunk = int(min(chunk, self.N_E))
        self.verbose = verbose

        self._build_static()
        self._run_rgf()

    # ---------------- static blocks ----------------
    def _build_static(self):
        p = self.p
        nx, ny = self.nx, self.ny

        onsite_C = myf.central_onsite(p)                                      # CHANGED: no Zeeman in C
        # one y-row of the central region: nx sites chained by Vx
        self.H_row = myf.make_row_hamiltonian(nx, onsite_C, myf.Vx(p.t_c, p.alpha, p.beta))
        # H[iy, iy+1]: site-diagonal Vy across the row
        self.Vy4 = myf.Vy(p.t_c, p.alpha, p.beta)      # OPT 1: 4x4 generator of V
        self.V = block_diag(*([self.Vy4] * nx))

        # sanity: the tridiagonal assembly must reproduce junction.H_C exactly
        self._check_tridiagonal()

        # electron / hole index sets inside a 4*ny lead block (same as junction.py)
        self.e_idx = np.where(np.tile([True, True, False, False], ny))[0]
        self.h_idx = np.where(np.tile([False, False, True, True], ny))[0]

    def _check_tridiagonal(self):
        nx, ny, d = self.nx, self.ny, 4 * self.nx
        H = np.zeros((d * ny, d * ny), dtype=complex)
        for iy in range(ny):
            H[d * iy:d * (iy + 1), d * iy:d * (iy + 1)] = self.H_row
        for iy in range(ny - 1):
            H[d * iy:d * (iy + 1), d * (iy + 1):d * (iy + 2)] = self.V
            H[d * (iy + 1):d * (iy + 2), d * iy:d * (iy + 1)] = self.V.conj().T
        if not np.allclose(H, self.junction.H_C):
            raise RuntimeError(
                "y-slice decomposition does not reproduce junction.H_C -- the "
                "site ordering assumption (site = ix + nx*iy) is violated."
            )

    def _make_top_ribbon(self, phi_lead):
        p = self.p
        H_intra, H_inter, V_coupling = myf.sc_ribbon_blocks(p, phi_lead, p.tc_top)   # CHANGED (v7): SOC in T/B
        return myf.Lead('ribbon_top', H_intra, H_inter, V_coupling, p, br=False, reverse=True)   # CHANGED

    # ---------------- the expensive, phi-independent part ----------------
    def _run_rgf(self):
        p = self.p
        nx, ny = self.nx, self.ny
        sub = self.sub
        N_E = self.N_E
        n_L, n_T = 4 * ny, 4 * nx

        z_all = self.E_sweep + 1j * p.eta

        # phi-independent self-energies
        eyeN = np.eye(4 * ny, dtype=complex)[None]
        eyeS = np.eye(4 * nx, dtype=complex)[None]
        zb = z_all[:, None, None]

        # --- OPT 0: share the Sancho decimations -------------------------
        # surface_gf depends ONLY on (H_onsite, V_hop, z); V_coupling enters
        # afterwards.  lead_L and lead_R are built from the same H_layer_N
        # and V_n, so they share one decimation.  Guarded by an assert so
        # this silently stops applying if junction.py ever changes.
        def _se(lead, g):
            """Same as Lead.self_energy, but on an already-computed surface GF."""
            Vc = lead.V_coupling[None, :, :]
            return Vc.conj().transpose(0, 2, 1) @ g @ Vc

        lead_L, lead_R, rib_b = self.junction.lead_L, self.junction.lead_R, self.junction.ribbon_bot

        # Sharing is CONDITIONAL, not assumed: under a bias that shifts the
        # lead chemical potentials (mu_n +/- V) the two leads no longer share
        # H_onsite, so each gets its own decimation.  Detected, not asserted.
        share_leads = (np.allclose(lead_L.H_onsite, lead_R.H_onsite)
                       and np.allclose(lead_L.V_hop, lead_R.V_hop))
        g_NL = lead_L.surface_gf(zb * eyeN)
        Sigma_L = _se(lead_L, g_NL)
        Sigma_R = _se(lead_R, g_NL if share_leads else lead_R.surface_gf(zb * eyeN))

        g_S = rib_b.surface_gf(zb * eyeS)           # bottom ribbon (built at phase 0)
        Sigma_B = _se(rib_b, g_S)

        # top ribbon at the reference phase: ONE decimation for the whole sweep,
        # and none at all if it coincides with the bottom ribbon (phi_ref = 0).
        top = self._make_top_ribbon(self.phi_ref)
        share_ribbon = (self.phi_ref == 0.0
                        and np.allclose(top.H_onsite, rib_b.H_onsite)
                        and np.allclose(top.V_hop, rib_b.V_hop))
        g_T = g_S if share_ribbon else top.surface_gf(zb * eyeS)
        self.Sigma_T_ref = _se(top, g_T)
        if self.verbose:
            n_dec = (1 if share_leads else 2) + (1 if share_ribbon else 2)
            print(f"  Sancho decimations run: {n_dec} (of 4 naive)")

        self.Gamma_L = 1j * (Sigma_L - Sigma_L.conj().transpose(0, 2, 1))
        self.Gamma_R = 1j * (Sigma_R - Sigma_R.conj().transpose(0, 2, 1))

        # --- OPT 2: Sigma_S is never built densely ------------------------
        # It is nonzero ONLY on posL x posL and posR x posR, so g0 @ Sigma_S
        # touches just 8*ny of the dim_S columns.  Keeping the two dense
        # blocks separate turns a dim_S^3 matmul into 2 * dim_S * (4ny)^2.
        posL, posR, posT = sub.posL, sub.posR, sub.posT

        # cached blocks of h = (g0^-1 - Sigma_LR)^-1
        self.h_LL = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_LR = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_RL = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_RR = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_LT = np.empty((N_E, n_L, n_T), dtype=complex)
        self.h_RT = np.empty((N_E, n_L, n_T), dtype=complex)
        self.h_TT = np.empty((N_E, n_T, n_T), dtype=complex)
        # OPT 5: h_TL and h_TR are always solved against the SAME matrix, so
        # store them side by side and solve once.  h_TL / h_TR stay available
        # as views -- no extra memory.
        self.h_TLR = np.empty((N_E, n_T, 2 * n_L), dtype=complex)

        Id_S = np.eye(sub.dim_S, dtype=complex)[None]
        blk = lambda M, r, c: M[:, r[:, None], c[None, :]]

        for s in range(0, N_E, self.chunk):
            e = min(s + self.chunk, N_E)
            if self.verbose:
                print(f"  RGF chunk {s}:{e}")

            g0 = _g0_on_subspace(z_all[s:e], self.H_row, self.V, Sigma_B[s:e],
                                 sub, Vy4=self.Vy4)

            # OPT 2: build (g0 @ Sigma_S) from its two nonzero column blocks
            GS = np.zeros_like(g0)
            GS[:, :, posL] = g0[:, :, posL] @ Sigma_L[s:e]
            GS[:, :, posR] = g0[:, :, posR] @ Sigma_R[s:e]
            np.subtract(Id_S, GS, out=GS)          # GS <- I - g0 Sigma_S, in place

            # h = (I - g0 Sigma_LR)^-1 g0   (never inverts g0 itself)
            h = solve(GS, g0)

            self.h_LL[s:e] = blk(h, posL, posL)
            self.h_LR[s:e] = blk(h, posL, posR)
            self.h_RL[s:e] = blk(h, posR, posL)
            self.h_RR[s:e] = blk(h, posR, posR)
            self.h_LT[s:e] = blk(h, posL, posT)
            self.h_RT[s:e] = blk(h, posR, posT)
            self.h_TT[s:e] = blk(h, posT, posT)
            self.h_TLR[s:e, :, :n_L] = blk(h, posT, posL)
            self.h_TLR[s:e, :, n_L:] = blk(h, posT, posR)

        self.h_TL = self.h_TLR[:, :, :n_L]         # views, not copies
        self.h_TR = self.h_TLR[:, :, n_L:]
        self._Id_T = np.eye(n_T, dtype=complex)[None]
        self._split_sigma_T()                      # OPT 3

    # ---------------- cheap per-phi part ----------------
    def _split_sigma_T(self):
        """
        OPT 3: pre-split Sigma_T_ref by its gauge weight.

        The gauge rotation multiplies entry (a,b) by exp(i*theta*(h_b - h_a)),
        where h = 1 on hole components and 0 on electron ones, so the exponent
        can only be -1, 0 or +1.  Splitting Sigma_T_ref into those three
        pieces ONCE turns the per-phi rotation into two scalar*array products
        -- no np.tile, no division, no fresh index arrays per call.
        """
        hole = (np.arange(4 * self.nx) % 4) >= 2
        D = hole[None, :].astype(int) - hole[:, None].astype(int)   # in {-1, 0, +1}
        S = self.Sigma_T_ref
        self._ST_0 = S * (D == 0)
        self._ST_p = S * (D == 1)
        self._ST_m = S * (D == -1)

    def _sigma_T(self, phi):
        th = phi - self.phi_ref
        return self._ST_0 + np.exp(1j * th) * self._ST_p + np.exp(-1j * th) * self._ST_m

    def green_blocks(self, phi):
        """Return G_LL, G_LR, G_RL, G_RR (retarded), each (N_E, 4ny, 4ny)."""
        Sig_T = self._sigma_T(phi)
        M = self._Id_T - self.h_TT @ Sig_T

        # OPT 5: one LU for both right-hand sides instead of two
        K = solve(M, self.h_TLR)           # (N_E, 4nx, 8ny)
        n_L = 4 * self.ny
        SL, SR = Sig_T @ K[:, :, :n_L], Sig_T @ K[:, :, n_L:]

        G_LL = self.h_LL + self.h_LT @ SL
        G_LR = self.h_LR + self.h_LT @ SR
        G_RL = self.h_RL + self.h_RT @ SL
        G_RR = self.h_RR + self.h_RT @ SR
        return G_LL, G_LR, G_RL, G_RR

    def channels_at_phi(self, phi, side_name=None):
        """Same output contract as FastPhaseSweep.channels_at_phi."""
        G_LL, G_LR, G_RL, G_RR = self.green_blocks(phi)
        e_idx, h_idx = self.e_idx, self.h_idx

        sub_ = lambda M, r, c: M[:, r[:, None], c[None, :]]
        dag = lambda M: M.conj().transpose(0, 2, 1)
        T = lambda G1, B1, G2, B2: np.trace(G1 @ B1 @ G2 @ B2, axis1=1, axis2=2).real

        Ge_L = sub_(self.Gamma_L, e_idx, e_idx)
        Gh_L = sub_(self.Gamma_L, h_idx, h_idx)
        Ge_R = sub_(self.Gamma_R, e_idx, e_idx)
        Gh_R = sub_(self.Gamma_R, h_idx, h_idx)

        def side(name):
            if name == 'left':
                G, G_self = G_RL, G_LL
                own_e, own_h, oth_e, oth_h = Ge_L, Gh_L, Ge_R, Gh_R
            else:
                G, G_self = G_LR, G_RR
                own_e, own_h, oth_e, oth_h = Ge_R, Gh_R, Ge_L, Gh_L

            G_ee, G_hh = sub_(G, e_idx, e_idx), sub_(G, h_idx, h_idx)
            G_eh, G_he = sub_(G, e_idx, h_idx), sub_(G, h_idx, e_idx)
            S_eh, S_he = sub_(G_self, e_idx, h_idx), sub_(G_self, h_idx, e_idx)

            return {
                # CHANGED: index-consistent order, side = SOURCE lead ('left' -> T^{ab}_RL)
                "ee":       T(oth_e, G_ee, own_e, dag(G_ee)),
                "hh":       T(oth_h, G_hh, own_h, dag(G_hh)),
                "eh_cross": T(oth_e, G_eh, own_h, dag(G_eh)),
                "he_cross": T(oth_h, G_he, own_e, dag(G_he)),
                "eh_local": T(own_e, S_eh, own_h, dag(S_eh)),
                "he_local": T(own_h, S_he, own_e, dag(S_he)),
            }

        if side_name is None:
            return {'left': side('left'), 'right': side('right')}
        if side_name not in ('left', 'right'):
            raise ValueError(f"Invalid side_name: {side_name}")
        return side(side_name)

    def channels(self, side_name=None):
        """Channels at the junction's own phase p.phi."""
        return self.channels_at_phi(self.p.phi, side_name=side_name)


#%% --- TRANSPORT ---
def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -1000, 1000)))


def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E + mu) / kT, -1000, 1000)))


def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    """Integrate the EC/CAR/LAR channel contributions into a DC current for one lead."""
    mu_out, mu_in = bias[out_name], bias[in_name]
    f_out_e, f_out_h = f_electron(E_sweep, mu_out, kT), f_hole(E_sweep, mu_out, kT)
    f_in_e, f_in_h = f_electron(E_sweep, mu_in, kT), f_hole(E_sweep, mu_in, kT)

    I_EC = np.trapezoid(ch['ee'] * (f_out_e - f_in_e) - ch['hh'] * (f_out_h - f_in_h), E_sweep)
    I_CAR = np.trapezoid(ch['eh_cross'] * (f_out_e - f_in_h) - ch['he_cross'] * (f_out_h - f_in_e), E_sweep)
    I_LAR = np.trapezoid(ch['eh_local'] * (f_out_e - f_out_h), E_sweep)
    return {'EC': I_EC, 'CAR': I_CAR, 'LAR': I_LAR, 'total': I_EC + I_CAR + I_LAR}


def other_name(name):
    return 'right' if name == 'left' else 'left'


def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    """Full 2x2 (or NxN) differential conductance matrix G_ij = dI_i/dV_j via central differences."""
    G = {}
    for i in leads:
        ch_i, in_name = channel_sweeps[i], other_name(i)
        for j in leads:
            bp, bm = bias0.copy(), bias0.copy()
            bp[j] += dV
            bm[j] -= dV
            rp = dc_current_channels(ch_i, E_sweep, bp, kT, i, in_name)
            rm = dc_current_channels(ch_i, E_sweep, bm, kT, i, in_name)
            G[(i, j)] = (rp['total'] - rm['total']) / (2 * dV)
    return G


def eval_I_total(ch, E, VL, VR, kT):
    """Vectorized total left-lead current I_L(V_L, V_R) for arrays of bias points."""
    fLe, fLh = f_electron(E[None, :], VL, kT), f_hole(E[None, :], VL, kT)
    fRe, fRh = f_electron(E[None, :], VR, kT), f_hole(E[None, :], VR, kT)
    I_curr = (
        ch["ee"][None, :] * (fLe - fRe)
        - ch["hh"][None, :] * (fLh - fRh)
        + ch["eh_cross"][None, :] * (fLe - fRh)
        - ch["he_cross"][None, :] * (fLh - fRe)
        + (ch["eh_local"] + ch["he_local"])[None, :] * (fLe - fLh)
    )
    return np.trapezoid(I_curr, E, axis=1)


def partial_G_vectorized(ch, E, V, dV, kT, scheme="sym"):
    """
    Partial conductances G_LL = dI_L/dV_L and G_LR = dI_L/dV_R, evaluated
    around a bias baseline set by `scheme` ('sym': V_R = +V, 'anti': V_R = -V).
    """
    V = V[:, None]
    if scheme == "sym":
        VR_base = V
    elif scheme == "anti":
        VR_base = -V
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")

    def I_eval(VL, VR):
        return eval_I_total(ch, E, VL, VR, kT)

    G_LL = (I_eval(V + dV, VR_base) - I_eval(V - dV, VR_base)) / (2 * dV)
    G_LR = (I_eval(V, VR_base + dV) - I_eval(V, VR_base - dV)) / (2 * dV)
    return G_LL, G_LR


def total_dIdV_map(ch, E, V, dV, kT, scheme):
    """
    Direct derivative of I_L along the actual bias line - 'sym': V_R=+V,
    'anti': V_R=-V - evaluated at the correct point for each scheme.
    """
    V = V[:, None]
    if scheme == "sym":
        sign = 1
    elif scheme == "anti":
        sign = -1
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")
    Ip = eval_I_total(ch, E, V + dV, sign * (V + dV), kT)
    Im = eval_I_total(ch, E, V - dV, sign * (V - dV), kT)
    return (Ip - Im) / (2 * dV)


#%% --- THERMAL TRANSPORT (NEW) ---
# ======================================================================
#  Everything in this cell is NEW.
#
#  Setup of Gresta et al., PRB 114, 125405, Fig. 1: only L is biased
#  (V_L, T_L); R, T, B grounded.  Linear response, Eqs. (7)-(11):
#
#     T^{ab}_RL(E) = Tr[ Gamma_R^a  G^{r,ab}_RL  Gamma_L^b  G^{a,ba}_LR ]
#     T_th = T^ee_RL + T^he_RL          T_el = T^ee_RL - T^he_RL
#     kappa/kappa0 = int dE (E/kT)^2 (-df/dE) T_th     G/G0 = int dE (-df/dE) T_el
#     T -> 0:  kappa/kappa0 = T_th(0),  G/G0 = T_el(0)
#
#  In channels_at_phi(..., side_name='left') these are ch['ee'] and
#  ch['he_cross'] (side = source lead).
# ======================================================================
def thermal_electrical_T(ch):
    """T_th, T_el from a source-side channel dict (side_name='left')."""
    return ch['ee'] + ch['he_cross'], ch['ee'] - ch['he_cross']


def kappa_G_finite_T(E, T_th, T_el, kT):
    """
    Eqs. (7), (8) at finite temperature on an energy grid E (last axis).
    Returns kappa/kappa0, G/G0.  The grid must cover several kT around 0.
    """
    x = E / kT
    mdf = 0.25 / kT / np.cosh(0.5 * np.clip(x, -700, 700)) ** 2      # -df/dE
    kap = np.trapezoid((x ** 2) * mdf * T_th, E, axis=-1)
    G = np.trapezoid(mdf * T_el, E, axis=-1)
    return kap, G


def ribbon_self_energy_modes(p, z, phi_lead, tc, is_bot, mu_s=None, Bz=None):
    """
    EXACT SC-ribbon self-energy via transverse-mode decomposition.
    Batched over z: z has any shape; returns z.shape + (4nx, 4nx).

    Why it is exact
    ---------------
    One ribbon cell is H_intra = 1_nx (x) h0 + T (x) Vx4, with T the open-chain
    adjacency (T_{i,i+1} = T_{i+1,i} = 1).  This form requires Vx4 = Vx4^dag,
    i.e. no SOC in the ribbon (true in this model).  The inter-cell hopping
    1_nx (x) Vy4 and the coupling 1_nx (x) Vy4(tc) are site-diagonal.  With the
    real orthogonal sine transform O_{i,n} = sqrt(2/(nx+1)) sin(q_n (i+1)),
    q_n = n pi/(nx+1),  O^T T O = diag(2 cos q_n), every block becomes
    diagonal in n, so the ribbon is nx decoupled 4x4 chains along y:
        h_n = h0 + 2 cos(q_n) Vx4,   hopping Vy4.
    Hence  Sigma = (O (x) 1) [ (+)_n Vc^dag g_n Vc ] (O (x) 1)^T.
    Cost O(nx * 4^3 * n_iter) instead of O((4nx)^3 * n_iter).

    Orientation matches FourTerminalJunction: top (is_bot=False) has its bulk
    at y<0 -> reverse=True, coupling H[lead, C] = Vy(tc);  bottom has its
    bulk at y>ny-1, coupling H[lead, C] = Vy(tc)^dag (br=True).
    """
    p_loc = replace(p, mu_s=p.mu_s if mu_s is None else mu_s, Bz=p.Bz if Bz is None else Bz)
    nx = p.nx
    Vx4 = Vx(p_loc.t_s, p_loc.alpha, p_loc.beta)                                          # CHANGED (v7)
    if not np.allclose(Vx4, Vx4.conj().T):
        # CHANGED (v7): SOC in T/B -> the sine transform no longer decouples the ribbon.
        # Fall back to the full Sancho decimation of the 4nx x 4nx ribbon cell, same orientation.
        H_intra, H_inter, V_c = sc_ribbon_blocks(p_loc, phi_lead, tc)
        lead = Lead('ribbon_bot' if is_bot else 'ribbon_top', H_intra, H_inter, V_c, p_loc,
                    br=is_bot, reverse=not is_bot)
        z = np.asarray(z, dtype=complex)
        zb = z.reshape(-1)[:, None, None] * np.eye(4 * p.nx)[None]
        return lead.self_energy(zb).reshape(*z.shape, 4 * p.nx, 4 * p.nx)
    n = np.arange(1, nx + 1)
    q = np.pi * n / (nx + 1)
    O = np.sqrt(2.0 / (nx + 1)) * np.sin(np.outer(n, q))             # O[i, n]
    h0 = sc_onsite(p_loc, phi_lead)
    h_n = h0[None] + (2.0 * np.cos(q))[:, None, None] * Vx4[None]      # (nx, 4, 4)

    z = np.asarray(z, dtype=complex)
    zshape = z.shape
    zz = z.reshape(-1)[:, None, None, None] * np.eye(4)[None, None] - h_n[None]   # (N, nx, 4, 4)
    # Lead.surface_gf only ever uses (z - eps); eps_0 = 0 with z -> z - h_n is identical
    chain = Lead('chain', np.zeros((4, 4)), Vy(p_loc.t_s, p_loc.alpha, p_loc.beta), np.eye(4), p_loc,   # CHANGED (v7)
                 br=False, reverse=not is_bot)
    g = chain.surface_gf(zz.reshape(-1, 4, 4)).reshape(zz.shape)
    Vc = Vy(tc, p_loc.alpha, p_loc.beta).conj().T if is_bot else Vy(tc, p_loc.alpha, p_loc.beta)   # CHANGED (v7) (== Vy(tc) here)
    sig_n = Vc.conj().T @ g @ Vc                                        # (N, nx, 4, 4)
    Sig = np.einsum('in,jn,Nnab->Niajb', O, O, sig_n, optimize=True)
    return Sig.reshape(*zshape, 4 * nx, 4 * nx)


def central_row_blocks(p):
    """H_row (one y-row of C, 4nx x 4nx) and the 4x4 generator Vy4 of V = H[iy, iy+1]."""
    onsite_C = central_onsite(p)
    H_row = make_row_hamiltonian(p.nx, onsite_C, Vx(p.t_c, p.alpha, p.beta))
    Vy4 = Vy(p.t_c, p.alpha, p.beta)
    return H_row, Vy4


def normal_lead_self_energies(p, z):
    """Sigma_L, Sigma_R for z of shape (N,).  Two decimations unless they coincide."""
    H_layer_N, V_n, V_c = normal_lead_blocks(p)
    lead_L = Lead('L', H_layer_N, V_n, V_c, p, br=False, reverse=True)
    lead_R = Lead('R', H_layer_N, V_n, V_c, p, br=True)
    zb = np.asarray(z, dtype=complex)[:, None, None] * np.eye(4 * p.ny)[None]
    gL = lead_L.surface_gf(zb)
    gR = gL if np.allclose(lead_L.V_hop, lead_R.V_hop) else lead_R.surface_gf(zb)
    se = lambda lead, g: lead.V_coupling.conj().T[None] @ g @ lead.V_coupling[None]
    return se(lead_L, gL), se(lead_R, gR)


def transmissions_RL_batch(p, z, Sigma_L, Sigma_R, Sigma_T, Sigma_B, chunk=8, H_row=None, Vy4=None):
    """
    T^ee_RL and T^he_RL for a BATCH of independent problems that share the
    central region (H_row, Vy4) but may differ in z and in all four
    self-energies.  Batch axis = axis 0 of every array (Sigma_L / Sigma_R may
    have batch size 1 and are then broadcast).

    Same exact subspace-Dyson closure as RGFFourTerminal, except that Sigma_T
    enters the closure directly (phi is fixed per batch element, so no
    Woodbury split is needed):
        g0 = P (z - H_C - Sigma_B)^-1 P^dag          (y-RGF, _g0_on_subspace)
        g  = (1 - g0 Sigma_S)^-1 g0,   Sigma_S = Sigma_L + Sigma_R + Sigma_T on S
    and only the posL columns of g are solved for (G_RL, G_LL).
    Returns dict with 'ee', 'he' (T^ee_RL, T^he_RL) and 'he_LL' (local Andreev at L).
    """
    nx, ny = p.nx, p.ny
    sub = _SubspaceIndex(nx, ny)
    if H_row is None:
        H_row, Vy4 = central_row_blocks(p)
    V = block_diag(*([Vy4] * nx))
    posL, posR, posT = sub.posL, sub.posR, sub.posT
    N = z.shape[0]
    e_idx = np.where(np.tile([True, True, False, False], ny))[0]
    h_idx = np.where(np.tile([False, False, True, True], ny))[0]
    ix = lambda M, r, c: M[:, r[:, None], c[None, :]]
    bL = lambda M, s, e: M if M.shape[0] == 1 else M[s:e]

    out = {k: np.empty(N) for k in ('ee', 'he', 'he_LL')}
    Id_S = np.eye(sub.dim_S, dtype=complex)[None]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        SL, SR = bL(Sigma_L, s, e), bL(Sigma_R, s, e)
        g0 = _g0_on_subspace(z[s:e], H_row, V, Sigma_B[s:e], sub, Vy4=Vy4)
        GS = np.zeros_like(g0)
        GS[:, :, posL] = g0[:, :, posL] @ SL
        GS[:, :, posR] = g0[:, :, posR] @ SR
        GS[:, :, posT] += g0[:, :, posT] @ Sigma_T[s:e]     # += : corner columns carry Sigma_L/R AND Sigma_T
        np.subtract(Id_S, GS, out=GS)
        X = solve(GS, g0[:, :, posL])                        # (b, dim_S, 4ny)
        G_RL, G_LL = X[:, posR, :], X[:, posL, :]

        GamL = 1j * (SL - SL.conj().transpose(0, 2, 1))
        GamR = 1j * (SR - SR.conj().transpose(0, 2, 1))
        GLe, GLh = ix(GamL, e_idx, e_idx), ix(GamL, h_idx, h_idx)
        GRe, GRh = ix(GamR, e_idx, e_idx), ix(GamR, h_idx, h_idx)
        # Tr[A G B G^dag] = sum_ij (A G B)_ij conj(G_ij)
        tr = lambda A, G, B: np.einsum('bij,bij->b', A @ G @ B, G.conj()).real
        Gee, Ghe = ix(G_RL, e_idx, e_idx), ix(G_RL, h_idx, e_idx)
        out['ee'][s:e] = tr(GRe, Gee, GLe)
        out['he'][s:e] = tr(GRh, Ghe, GLe)
        Lhe = ix(G_LL, h_idx, e_idx)
        out['he_LL'][s:e] = tr(GLh, Lhe, GLe)
    return out


# ----------------------------------------------------------------------
#  maps
# ----------------------------------------------------------------------
def _map_mu_s_Z(p, mu_s_vals, Z_vals, E=0.0, chunk=8, verbose=True):
    """T^ee_RL, T^he_RL on the (mu_s, Z) grid at energy E, phase p.phi."""
    z0 = E + 1j * p.eta
    SL, SR = normal_lead_self_energies(p, np.array([z0]))
    H_row, Vy4 = central_row_blocks(p)
    nZ = len(Z_vals)
    res = {k: np.empty((len(mu_s_vals), nZ)) for k in ('ee', 'he', 'he_LL')}
    zb = np.full(nZ, z0)
    t0 = time.perf_counter()
    for i, mu_s in enumerate(mu_s_vals):
        ST = np.stack([ribbon_self_energy_modes(p, np.array([z0]), p.phi, p.tc_top, False, mu_s=mu_s, Bz=Z)[0]
                       for Z in Z_vals])
        SB = np.stack([ribbon_self_energy_modes(p, np.array([z0]), 0.0, p.tc_bot, True, mu_s=mu_s, Bz=Z)[0]
                       for Z in Z_vals])
        o = transmissions_RL_batch(p, zb, SL, SR, ST, SB, chunk=chunk, H_row=H_row, Vy4=Vy4)
        for k in res:
            res[k][i] = o[k]
        if verbose:
            print(f"    mu_s row {i+1}/{len(mu_s_vals)}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    return res


def _map_E_Z(p, E_vals, Z_vals, chunk=8, verbose=True):
    """T^ee_RL, T^he_RL on the (E, Z) grid at mu_s = p.mu_s, phase p.phi."""
    z = E_vals + 1j * p.eta
    SL_all, SR_all = normal_lead_self_energies(p, z)
    H_row, Vy4 = central_row_blocks(p)
    nE = len(E_vals)
    res = {k: np.empty((nE, len(Z_vals))) for k in ('ee', 'he', 'he_LL')}
    t0 = time.perf_counter()
    for j, Z in enumerate(Z_vals):
        ST = ribbon_self_energy_modes(p, z, p.phi, p.tc_top, False, Bz=Z)
        SB = ribbon_self_energy_modes(p, z, 0.0, p.tc_bot, True, Bz=Z)
        o = transmissions_RL_batch(p, z, SL_all, SR_all, ST, SB, chunk=chunk, H_row=H_row, Vy4=Vy4)
        for k in res:
            res[k][:, j] = o[k]
        if verbose:
            print(f"    Z column {j+1}/{len(Z_vals)}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    return res


def _map_mu_c_Z(p, mu_c_vals, Z_vals, E=0.0, chunk=8, verbose=True):
    """T^ee_RL, T^he_RL on the (mu_c, Z) grid at energy E; ribbons reused across mu_c."""
    z0 = E + 1j * p.eta
    SL, SR = normal_lead_self_energies(p, np.array([z0]))
    nZ = len(Z_vals)
    ST = np.stack([ribbon_self_energy_modes(p, np.array([z0]), p.phi, p.tc_top, False, Bz=Z)[0] for Z in Z_vals])
    SB = np.stack([ribbon_self_energy_modes(p, np.array([z0]), 0.0, p.tc_bot, True, Bz=Z)[0] for Z in Z_vals])
    res = {k: np.empty((len(mu_c_vals), nZ)) for k in ('ee', 'he', 'he_LL')}
    zb = np.full(nZ, z0)
    t0 = time.perf_counter()
    for i, mu_c in enumerate(mu_c_vals):
        pc = replace(p, mu_c=mu_c)
        H_row, Vy4 = central_row_blocks(pc)
        o = transmissions_RL_batch(pc, zb, SL, SR, ST, SB, chunk=chunk, H_row=H_row, Vy4=Vy4)
        for k in res:
            res[k][i] = o[k]
        if verbose:
            print(f"    mu_c row {i+1}/{len(mu_c_vals)}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    return res


def thermal_map(p, kind, vals1, vals2, chunk=8, verbose=True):
    """
    kind = 'mu_s-Z' : vals1 = mu_s grid, vals2 = Z grid   (E = 0)
           'E-Z'    : vals1 = E grid,    vals2 = Z grid   (mu_s = p.mu_s)
           'mu_c-Z' : vals1 = mu_c grid, vals2 = Z grid   (E = 0)
    Returns dict: T_th, T_el, T_ee, T_he, T_he_LL with shape (len(vals1), len(vals2)).
    At E = 0 and T -> 0: kappa/kappa0 = T_th, G/G0 = T_el.
    """
    fn = {'mu_s-Z': _map_mu_s_Z, 'E-Z': _map_E_Z, 'mu_c-Z': _map_mu_c_Z}[kind]
    r = fn(p, np.asarray(vals1, float), np.asarray(vals2, float), chunk=chunk, verbose=verbose)
    return {'T_th': r['ee'] + r['he'], 'T_el': r['ee'] - r['he'],
            'T_ee': r['ee'], 'T_he': r['he'], 'T_he_LL': r['he_LL']}


#%% --- PLOTTING ---
def style_axis(ax, title):
    """Apply the shared axis style: no top/right spine, inward ticks."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.set_title(title)


def print_params(p):
    """Render a Params instance as a LaTeX parameter table (Jupyter/IPython display)."""
    latex_str = (
        r"\begin{array}{ll | ll | ll}\hline"
        r"\text{Chemical Potentials} & \text{Value} & \text{Hoppings} & \text{Value} & \text{Other} & \text{Value} \\\hline"
        rf"\mu_n & {p.mu_n} & t_n & {p.t_n} & n_x & {p.nx} \\"
        rf"\mu_c & {p.mu_c} & t_c & {p.t_c} & \Delta & {p.delta} \\"
        rf"\mu_s & {p.mu_s} & t_s & {p.t_s} & \phi & {p.phi:.3g} \\"
        rf"& & t_{{\text{{top}}}} & {p.tc_top} & \eta & {p.eta} \\"
        rf"& & t_{{\text{{bot}}}} & {p.tc_bot} & kT & {p.kT} \\"
        rf"& & t_{{\text{{barr}}}} & {p.tc_barr} & & \\\hline\end{{array}}"
    )
    display(Math(latex_str))


#%% --- ANALYSIS ---
def compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map=1e-5, side='left', scheme='sym'):
    """For one parameter instance pc, sweep phi and return G_LL/G_LR(phi, V) maps."""
    junction0 = FourTerminalJunction(pc)
    G_LL_map = np.empty((len(phi_vals), len(V_bias_map)))
    G_LR_map = np.empty_like(G_LL_map)

    fps = FastPhaseSweep(junction0, E_sweep, phi_ref=0.0)
    for i, phi in enumerate(phi_vals):
        ch = fps.channels_at_phi(phi, side_name=side)
        G_LL_map[i], G_LR_map[i] = partial_G_vectorized(
            ch, E_sweep, V_bias_map, dV_map, pc.kT, scheme=scheme
        )

    return G_LL_map, G_LR_map


def summarize_map(G_LL_map, G_LR_map):
    """
    Summarize the non-local conductance map.
    Positive peak G_LR indicates CAR dominates, negative indicates EC dominates.
    """
    flat_GLR = G_LR_map.flatten()
    max_idx = np.argmax(np.abs(flat_GLR))
    peak_GLR = flat_GLR[max_idx]

    return {
        'mean_GLR': np.mean(G_LR_map),
        'peak_GLR': peak_GLR,
        'peak_GLL': np.max(np.abs(G_LL_map)),
    }


def scan_grid(base_params, name1, vals1, name2, vals2, E_sweep, phi_vals, V_bias_map, dV_map=1e-5, side='left', scheme='sym'):
    """2D parameter scan over (name1, name2); returns an object array of per-cell results."""
    results = np.empty((len(vals1), len(vals2)), dtype=object)
    for i, v1 in enumerate(tqdm(vals1, desc=f'{name1} sweep')):
        for j, v2 in enumerate(vals2):
            pc = replace(base_params, **{name1: v1, name2: v2})
            G_LL_map, G_LR_map = compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map, side, scheme)
            results[i, j] = {
                'G_LL_map': G_LL_map,
                'G_LR_map': G_LR_map,
                'summary': summarize_map(G_LL_map, G_LR_map),
            }
    return results


def plot_grid_thumbnails(results, name1, vals1, name2, vals2, phi_vals, V_bias_map, delta):
    """Grid of small contour plots of the non-local conductance G_LR, one per (v1, v2) cell."""
    n1, n2 = len(vals1), len(vals2)
    fig, axs = plt.subplots(n1, n2, figsize=(2.6 * n2, 2.4 * n1), squeeze=False)

    for i in range(n1):
        for j in range(n2):
            ax = axs[i, j]
            G_LR = results[i, j]['G_LR_map']

            vmax = np.max(np.abs(G_LR))
            if vmax == 0:
                vmax = 1e-10  # avoid a singular norm

            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            ax.contourf(phi_vals / np.pi, V_bias_map / delta, G_LR.T, levels=40,
                        cmap='RdBu_r', norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{name2}={vals2[j]:.2g}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{name1}={vals1[i]:.2g}', fontsize=9)

    fig.suptitle(r'Non-local Conductance $G_{LR}$ across ' + f'{name1} x {name2} \n(Red = CAR, Blue = EC)', y=1.05)
    plt.tight_layout()
    return fig, axs


def plot_summary_trends(results, name1, vals1, name2, vals2):
    """Line plot of peak G_LR vs. name1, tracking the EC <-> CAR transition."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for j, v2 in enumerate(vals2):
        peak_glr = [results[i, j]['summary']['peak_GLR'] for i in range(len(vals1))]
        ax.plot(vals1, peak_glr, marker='o', label=f'{name2}={v2:.2g}')
    ax.set_xlabel(name1)
    ax.set_ylabel(r'Peak Non-local Conductance $G_{LR}$ ($e^2/h$)')
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.7)
    ylim = ax.get_ylim()
    ax.text(vals1[0], 0.15 * abs(ylim[1] - ylim[0]), 'CAR Dominates ($G_{LR} > 0$)',
            color='tab:red', fontsize=10, va='bottom')
    ax.text(vals1[0], -0.15 * abs(ylim[1] - ylim[0]), 'EC Dominates ($G_{LR} < 0$)',
            color='tab:blue', fontsize=10, va='top')
    ax.legend()
    style_axis(ax, r'Mechanism Tuning via $G_{LR}$')
    plt.tight_layout()
    return fig, ax


#%% --- SELF-ALIAS: makes every `myf.` reference above resolve to this module ---
myf = sys.modules[__name__]

p = myf.Params(
    nx=40, ny=10, t_n=1.0, mu_n=0.00, t_c=1.00, mu_c=0.0, t_s=1.0, mu_s=0.875,
    delta=0.35, phi=np.pi, tc_top=1.00, tc_bot=1.00, tc_barr=1.0, alpha = 0.2, beta = 0.0, Bz=-0.4, Bxy=0.0, theta_z=0.0*np.pi,
    eta=2e-5, kT=1e-6,
)

junction = myf.FourTerminalJunction(p)
E_sweep = np.linspace(-1.5*p.delta, 1.5*p.delta, 61)
rgf = RGFFourTerminal(
    junction,
    E_sweep,
    phi_ref=0.0,
    verbose=True
)

ch = rgf.channels_at_phi(p.phi, side_name="left")

T_ee = ch["ee"]
T_eh = ch["eh_cross"]
T_he = ch["he_cross"]
T_hh = ch["hh"]

# phi_test = 0.37
# t0 = time.perf_counter()
# fast = myf.FastPhaseSweep(junction, E_sweep, phi_ref=0.0)
# t_fast = time.perf_counter() - t0
# print(f'System size: nx={p.nx}, ny={p.ny}, N_E={len(E_sweep)}')
# print(f"FastPhaseSweep energy sweep: {t_fast:.4f} s")

#t0 = time.perf_counter()
rgf = RGFFourTerminal(junction, E_sweep, phi_ref=0.0)
#t_rgf = time.perf_counter() - t0


# print(f"RGF enrgy sweep:            {t_rgf:.4f} s")
# print(f"Energy speedup: {t_fast:.1f}s / {t_rgf:.1f}s = {(t_fast)/(t_rgf):.2f}x")
# a = fast.channels_at_phi(phi_test, "left")
# b = rgf.channels_at_phi(phi_test, "left")
# print(f"Max rel. error at phi={phi_test:.3f}:")
# for key in a:
#     print(
#         key,
#         np.max(np.abs(a[key] - b[key])),        #type: ignore
#         np.max(np.abs(a[key] / b[key] - 1))     #type: ignore
#     )
# t0 = time.perf_counter()
# for ph in np.linspace(0, 2*np.pi, 41):
#     fast.channels_at_phi(ph, "left")
# t_fast_phi = time.perf_counter() - t0

# t0 = time.perf_counter()
# for ph in np.linspace(0, 2*np.pi, 41):
#     rgf.channels_at_phi(ph, "left")
# t_rgf_phi = time.perf_counter() - t0

# print(f"per-phi: dense={1000*t_fast_phi/41:.1f} ms   rgf={1000*t_rgf_phi/41:.1f} ms")
# print(r"full $\phi$ sweep:" + f"dense={t_fast+t_fast_phi:.1f}s  rgf={t_rgf+t_rgf_phi:.1f}s")
# print(f"Phase speedup: {t_fast+t_fast_phi:.1f}s / {t_rgf+t_rgf_phi:.1f}s = {(t_fast+t_fast_phi)/(t_rgf+t_rgf_phi):.2f}x")
# %%

phi_vals = np.linspace(0.0, 2 * np.pi, 31)
p = myf.Params(
    nx=40, ny=5, t_n=1.0, mu_n=0.00, t_c=1.00, mu_c=0.0, t_s=1.0, mu_s=0.0,
    delta=0.35, phi=np.pi, tc_top=1.00, tc_bot=1.00, tc_barr=1.0, alpha = 0.2, beta = 0.0, Bz=0.4, Bxy=0.0, theta_z=0.0*np.pi,
    eta=1e-5, kT=1e-6,
)

T_ee, T_eh, T_he, T_lar_eh, T_lar_he = [], [], [], [], []
E_fixed = np.array([0.0])
phi_vals_ext = np.linspace(0.0, 2 * np.pi, 41)

for phi_val in phi_vals_ext:
    p_phi = replace(p, phi=phi_val)
    junction_phi = myf.FourTerminalJunction(p_phi)
    ch_left = junction_phi.channels(E_fixed)['left']

    T_ee.append(ch_left['ee'])
    T_eh.append(ch_left['eh_cross'])
    T_he.append(ch_left['he_cross'])
    T_lar_eh.append(ch_left['eh_local'])
    T_lar_he.append(ch_left['he_local'])

#%% --- PLOTTING TRANSMISSIONS ---
T_he_tot = np.array(T_he) + np.array(T_lar_he)


fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax[0].plot(phi_vals_ext / np.pi, T_eh, label=r"$T_{eh}$ (CAR)", c='tab:blue')
ax[0].set_title(fr'Transmissions at fixed E={E_fixed[0]:.2f} vs. $\phi$')
ax[0].plot(phi_vals_ext / np.pi, T_he, label=r"$T_{he}$ (CAR)", c='tab:orange')
ax[0].set_ylabel("Transmission")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(phi_vals_ext / np.pi, T_lar_eh, label=r"$T_{eh}$ (LAR)", c='tab:blue')
ax[1].plot(phi_vals_ext / np.pi, T_he_tot, label=r"$\kappa (T_{he}^{CAR} + T_{he}^{\rm LAR})$", color='tab:orange')
ax[1].set_xlabel(r"$\phi / \pi$")
ax[1].set_ylabel("Transmission")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
# %%
