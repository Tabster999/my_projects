"""
rgf.py -- Recursive Green's function solver for the 4-terminal junction.
=======================================================================

WHY A PLAIN RGF DOES NOT WORK HERE
----------------------------------
RGF needs the dressed inverse GF A = z - H_C - Sigma_tot to be BLOCK
TRIDIAGONAL along the sweep direction.  In this geometry it is not, in
*either* direction:

  * sweep along x (slices = columns):  Sigma_T and Sigma_B are dense in x,
    because the SC ribbons carry intra-lead hopping Vx(t_s) along x
    (see FourTerminalJunction._make_ribbon: H_intra = make_row_hamiltonian).
  * sweep along y (slices = rows):     Sigma_L and Sigma_R are dense in y,
    because the normal leads carry intra-lead hopping Vy(t_n) along y
    (see FourTerminalJunction._build:  H_layer_N = make_row_hamiltonian).

Both are genuine semi-infinite half-planes, so both surface GFs are full
matrices.  Verified numerically: every site-block of Sigma_T(ix,ix') and
Sigma_L(iy,iy') is nonzero.

THE FIX (exact, not an approximation)
-------------------------------------
Split the self-energies into a part that RGF can absorb and a part that is
supported on a SMALL subspace, then close the Dyson equation on that
subspace.

  A0 = z - H_C - Sigma_B        <- block tridiagonal in y (Sigma_B lives on
                                   the last row only).  Also phi-independent.
  W  = P^dag (Sigma_L + Sigma_R + Sigma_T) P

with P the projector onto

  S = {row iy=0}  U  {column ix=0}  U  {column ix=nx-1}

of site-dimension nx + 2*ny - 2 (the two corners are shared), i.e.
dim_S = 4*(nx + 2*ny - 2).

Because the perturbation W lives entirely inside S, the Dyson equation
closes on S exactly:

  g  = P G P^dag  =  g0 + g0 Sigma_S g   =>   g = (g0^-1 - Sigma_S)^-1

with g0 = P A0^-1 P^dag obtained from the y-RGF.  G_LL, G_LR, G_RL, G_RR
are *inside* S, so nothing outside S is ever needed.  No approximation,
no truncation: this is algebraically identical to the dense solve.

PHASE SWEEPS
------------
Sigma_T is the only phi-dependent piece, so we pre-solve

  h = (g0^-1 - Sigma_L - Sigma_R)^-1        (phi-independent)

once per energy, and update per phi with a Woodbury step on the T block
(dimension 4*nx only):

  M       = I_T - h_TT Sigma_T(phi)
  G_XY    = h_XY + h_XT Sigma_T(phi) M^-1 h_TY ,   X,Y in {L,R}

Sigma_T(phi) itself comes from the same diagonal gauge rotation used in
FastPhaseSweep, so the Sancho-Rubio decimation still runs only once.

COST (per energy, d = 4*nx, dim_S = 4*(nx+2*ny-2))
--------------------------------------------------
  dense reference   O((4 nx ny)^3)      = 64 nx^3 ny^3
  RGF sweeps        O(ny * d^3)         = 64 nx^3 ny
  all-pairs on S    O(ny^2 * d^2 * 8)   = 128 nx^2 ny^2
  subspace solve    O(dim_S^3)
  per extra phi     O((4 nx)^3)

CONVENTIONS (identical to modules/solvers.py get_rgf_sns)
---------------------------------------------------------
  H[iy, iy+1] = V,  H[iy+1, iy] = V^dag,  V = block_diag([Vy]*nx)
  gL[i] = (A_ii - V^dag gL[i-1] V)^-1
  gR[i] = (A_ii - V     gR[i+1] V^dag)^-1
  G[i,j] = gR[i] V^dag G[i-1,j]   (i > j)
  G[i,j] = gL[i] V     G[i+1,j]   (i < j)

This module is purely additive: no existing file needs to be modified.
"""
#%% --- IMPORTS --- 
import sys, os 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
from scipy.linalg import block_diag
os.chdir(r"c:\coding\my_projects\thesis\fourterminal")
import my_functions as myf
from dataclasses import replace
import time 
#%% --- CLASSES AND FUNCTIONS --- 
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
#  core: g0 = P A0^-1 P^dag  via a y-direction RGF
# ----------------------------------------------------------------------
def _g0_on_subspace(z, H_row, V, Sigma_B, sub):
    """
    Batched over energy.

    Parameters
    ----------
    z        : (N_E,) complex, already E + i*eta
    H_row    : (d, d) intra-slice Hamiltonian of one y-row (d = 4*nx)
    V        : (d, d) inter-slice hopping, H[iy, iy+1] = V
    Sigma_B  : (N_E, d, d) bottom-ribbon self-energy (sits on slice ny-1)
    sub      : _SubspaceIndex

    Returns
    -------
    g0 : (N_E, dim_S, dim_S)
    """
    ny = sub.ny
    d = H_row.shape[0]
    N_E = z.shape[0]
    Vd = V.conj().T
    Id = np.eye(d, dtype=complex)

    # A_ii for every slice (only the last one carries Sigma_B)
    A_diag = np.empty((ny, N_E, d, d), dtype=complex)
    base = z[:, None, None] * Id[None] - H_row[None] #broadcasting over energy (all E computations at once, vectorized)
    for iy in range(ny):
        A_diag[iy] = base
    A_diag[ny - 1] -= Sigma_B

    # gL sweeps from top (iy = 0) to bottom (iy = ny - 1), gR sweeps from bottom to top
    gL = np.empty((ny, N_E, d, d), dtype=complex)
    gR = np.empty((ny, N_E, d, d), dtype=complex)

    gL[0] = inv(A_diag[0])
    for i in range(1, ny):
        gL[i] = inv(A_diag[i] - Vd @ gL[i - 1] @ V)

    gR[ny - 1] = inv(A_diag[ny - 1])
    for i in range(ny - 2, -1, -1):
        gR[i] = inv(A_diag[i] - V @ gR[i + 1] @ Vd)

    g0 = np.zeros((N_E, sub.dim_S, sub.dim_S), dtype=complex)

    for j in range(ny):
        # build fully dressed diagonal block
        Aj = A_diag[j].copy()
        if j > 0:
            Aj -= Vd @ gL[j - 1] @ V
        if j < ny - 1:
            Aj -= V @ gR[j + 1] @ Vd
        G_jj = inv(Aj)

        selj = sub.sel_flat[j]
        cj = sub.block_slice(j)

        C0 = G_jj[:, :, selj]                       # (N_E, d, w_j)
        g0[:, sub.block_slice(j), cj] = C0[:, selj, :]

        # propagate upward (i > j) with gR
        C = C0
        for i in range(j + 1, ny):
            C = gR[i] @ (Vd @ C)
            g0[:, sub.block_slice(i), cj] = C[:, sub.sel_flat[i], :]

        # propagate downward (i < j) with gL
        C = C0
        for i in range(j - 1, -1, -1):
            C = gL[i] @ (V @ C)
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
                 the top ribbon
    chunk      : energies per RGF batch.  Controls peak memory; None picks
                 a size targeting ~`mem_budget_gb`.
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

        onsite_C = myf.onsite_block(p.t_c, p.mu_c, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                                alpha=p.alpha, beta=p.beta, twod=True)
        # one y-row of the central region: nx sites chained by Vx
        self.H_row = myf.make_row_hamiltonian(nx, onsite_C, myf.Vx(p.t_c, p.alpha, p.beta))
        # H[iy, iy+1]: site-diagonal Vy across the row
        self.V = block_diag(*([myf.Vy(p.t_c, p.alpha, p.beta)] * nx))

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
        onsite_SC = myf.onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, twod=True)
        H_intra = myf.make_row_hamiltonian(p.nx, onsite_SC, myf.Vx(p.t_s))
        H_inter = block_diag(*([myf.Vy(p.t_s)] * p.nx))
        V_coupling = block_diag(*([myf.Vy(p.tc_top)] * p.nx))
        return myf.Lead('ribbon_top', H_intra, H_inter, V_coupling, p, br=False)

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
        Sigma_L = self.junction.lead_L.self_energy(zb * eyeN)
        Sigma_R = self.junction.lead_R.self_energy(zb * eyeN)
        Sigma_B = self.junction.ribbon_bot.self_energy(zb * eyeS)

        # top ribbon at the reference phase: ONE decimation for the whole sweep
        self.Sigma_T_ref = self._make_top_ribbon(self.phi_ref).self_energy(zb * eyeS)

        self.Gamma_L = 1j * (Sigma_L - Sigma_L.conj().transpose(0, 2, 1))
        self.Gamma_R = 1j * (Sigma_R - Sigma_R.conj().transpose(0, 2, 1))

        # Sigma_LR embedded in S
        posL, posR, posT = sub.posL, sub.posR, sub.posT
        Sigma_S = np.zeros((N_E, sub.dim_S, sub.dim_S), dtype=complex)
        Sigma_S[:, posL[:, None], posL[None, :]] += Sigma_L
        Sigma_S[:, posR[:, None], posR[None, :]] += Sigma_R

        # cached blocks of h = (g0^-1 - Sigma_LR)^-1
        self.h_LL = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_LR = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_RL = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_RR = np.empty((N_E, n_L, n_L), dtype=complex)
        self.h_LT = np.empty((N_E, n_L, n_T), dtype=complex)
        self.h_RT = np.empty((N_E, n_L, n_T), dtype=complex)
        self.h_TL = np.empty((N_E, n_T, n_L), dtype=complex)
        self.h_TR = np.empty((N_E, n_T, n_L), dtype=complex)
        self.h_TT = np.empty((N_E, n_T, n_T), dtype=complex)

        Id_S = np.eye(sub.dim_S, dtype=complex)[None]
        blk = lambda M, r, c: M[:, r[:, None], c[None, :]]

        for s in range(0, N_E, self.chunk):
            e = min(s + self.chunk, N_E)
            if self.verbose:
                print(f"  RGF chunk {s}:{e}")

            g0 = _g0_on_subspace(z_all[s:e], self.H_row, self.V, Sigma_B[s:e], sub)
            # h = (I - g0 Sigma_LR)^-1 g0   (never inverts g0 itself)
            h = solve(Id_S - g0 @ Sigma_S[s:e], g0)

            self.h_LL[s:e] = blk(h, posL, posL)
            self.h_LR[s:e] = blk(h, posL, posR)
            self.h_RL[s:e] = blk(h, posR, posL)
            self.h_RR[s:e] = blk(h, posR, posR)
            self.h_LT[s:e] = blk(h, posL, posT)
            self.h_RT[s:e] = blk(h, posR, posT)
            self.h_TL[s:e] = blk(h, posT, posL)
            self.h_TR[s:e] = blk(h, posT, posR)
            self.h_TT[s:e] = blk(h, posT, posT)

        self._Id_T = np.eye(n_T, dtype=complex)[None]

    # ---------------- cheap per-phi part ----------------
    def _sigma_T(self, phi):
        u = np.tile(np.array([1.0, 1.0, np.exp(1j * phi), np.exp(1j * phi)]), self.nx)
        u_ref = np.tile(np.array([1.0, 1.0, np.exp(1j * self.phi_ref),
                                  np.exp(1j * self.phi_ref)]), self.nx)
        u_rel = u / u_ref
        return (u_rel.conj()[None, :, None]) * self.Sigma_T_ref * (u_rel[None, None, :])

    def green_blocks(self, phi):
        """Return G_LL, G_LR, G_RL, G_RR (retarded), each (N_E, 4ny, 4ny)."""
        Sig_T = self._sigma_T(phi)
        M = self._Id_T - self.h_TT @ Sig_T
        K_L = solve(M, self.h_TL)          # (N_E, 4nx, 4ny)
        K_R = solve(M, self.h_TR)

        SL, SR = Sig_T @ K_L, Sig_T @ K_R
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
                "ee":       T(own_e, G_ee, oth_e, dag(G_ee)),
                "hh":       T(own_h, G_hh, oth_h, dag(G_hh)),
                "eh_cross": T(own_e, G_eh, oth_h, dag(G_eh)),
                "he_cross": T(own_h, G_he, oth_e, dag(G_he)),
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
#%% --- COMPARISON OF FAST_PHASE_SWEEP AND RGF --- 
p = myf.Params(
    nx=12, ny=30, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi / 4, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    eta=2e-5, kT=5e-6,
)

junction = myf.FourTerminalJunction(p)
E_sweep = np.linspace(-1.5*p.delta, 1.5*p.delta, 121)
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

phi_test = 0.37
t0 = time.perf_counter()
fast = myf.FastPhaseSweep(junction, E_sweep, phi_ref=0.0)
t_fast = time.perf_counter() - t0
print(f"FastPhaseSweep setup: {t_fast:.4f} s")

t0 = time.perf_counter()
rgf = RGFFourTerminal(junction, E_sweep, phi_ref=0.0)
t_rgf = time.perf_counter() - t0


print(f"RGF setup:            {t_rgf:.4f} s")

a = fast.channels_at_phi(phi_test, "left")
b = rgf.channels_at_phi(phi_test, "left")
for key in a:
    print(
        key,
        np.max(np.abs(a[key] - b[key])),        #type: ignore
        np.max(np.abs(a[key] / b[key] - 1))     #type: ignore
    )
# %%
phi_vals = np.linspace(0.0, 2 * np.pi, 81)
channels = rgf.channels_at_phi(phi_test)
pT = myf.Params(
    nx=8, ny=16, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=0.0, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    eta=2e-5, kT=2e-5,
)

T_ee, T_eh, T_he, T_lar_eh, T_lar_he = [], [], [], [], []
E_fixed = np.array([0.0])
phi_vals_ext = np.linspace(0.0, 2 * np.pi, 81)
t0 = time.perf_counter()

for phi_val in phi_vals_ext:
    p_phi = replace(pT, phi=phi_val)
    junction_phi = myf.FourTerminalJunction(p_phi)
    ch_left = junction_phi.channels(E_fixed)['right']

    T_ee.append(ch_left['ee'])
    T_eh.append(ch_left['eh_cross'])
    T_he.append(ch_left['he_cross'])
    T_lar_eh.append(ch_left['eh_local'])
    T_lar_he.append(ch_left['he_local'])
t_phase = time.perf_counter() - t0
print(f't_phase = {t_phase:.4f} s')
fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax[0].plot(phi_vals_ext / np.pi, T_eh, label=r"$T_{eh}$ (CAR)")
ax[0].set_title(fr'Transmissions at fixed E={E_fixed[0]:.2f} vs. $\phi$')
ax[0].plot(phi_vals_ext / np.pi, T_he, label=r"$T_{he}$ (CAR)")
ax[0].set_ylabel("Transmission")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(phi_vals_ext / np.pi, T_lar_eh, label=r"$T_{eh}$ (LAR)")
ax[1].set_xlabel(r"$\phi / \pi$")
ax[1].set_ylabel("Transmission")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
# %%
