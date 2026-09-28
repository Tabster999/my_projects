"""
rgf.py -- Recursive Green's function solver for the 4-terminal junction.
=======================================================================

GEOMETRY
--------
Central nx*ny region; normal leads L/R on the columns ix = 0 / nx-1, SC ribbons
T/B on the rows iy = 0 / ny-1.  Site index in junction.H_C: ix + nx*iy.

STRATEGY (exact, no approximation)
----------------------------------
1. Sweep along x (slices = columns, width d = 4*ny).  Sigma_L and Sigma_R live on
   the first/last slice, so

       A0 = z - H_C - Sigma_L - Sigma_R

   is block-tridiagonal and g0 = A0^-1 = G(C+L+R) is obtained by RGF.
2. Both SC ribbons are then attached in ONE Dyson step on the boundary subspace
   S = {col 0} U {col nx-1} U {row 0} U {row ny-1}.  With U = P_TB W^dag,
   W = diag(W_T, W_B) the lead couplings (H_{s,C}) and g_s the ribbon surface GFs,

       G_QQ = g0_QQ + (g0_QP W^dag) [g_s^-1 - W g0_PP W^dag]^-1 (W g0_PQ),
       Q = L U R,  P = T U B.

WHY THIS ORDER (numerical stability)
------------------------------------
A narrow topological SC strip has Majorana end states on the row touching the
junction, so g_s (and Sigma = W^dag g_s W) has a pole ~1/eta at E = 0.  Any
intermediate Green's function built from C plus an SC ribbon but WITHOUT the
normal leads then also scales like 1/eta, and later Dyson/Woodbury updates lose
~all digits by cancellation (observed: errors up to 0.25 at eta = 1e-9).
Here (i) the first intermediate is G(C+L+R), broadened by the metallic normal
leads, and (ii) the ribbons enter through g_s^-1 = z - eps_s, which stays finite.
Verified against dense inversion and Kwant down to eta = 1e-9 (verify_solvers.py).

PHASE SWEEPS
------------
Only g_T^-1 depends on phi:  g_T^-1(phi) = U^dag g_T^-1(phi_ref) U,
U = diag(1, 1, e^{i dphi}, e^{i dphi}) per site (W_T commutes with U), so the
top-ribbon decimation runs once.  Per phi: one (8 nx)-dim solve per energy.

COST per energy (d = 4 ny):   RGF sweeps O(nx d^3); boundary propagation
O(nx^2 d^2 w) with w <= d; per phi O((8 nx)^3).

CONVENTIONS (slices = columns ix)
---------------------------------
  H[ix, ix+1] = V (block-diag of the x-hop),  gL[i] = (A_ii - V^dag gL[i-1] V)^-1,
  gR[i] = (A_ii - V gR[i+1] V^dag)^-1,
  G[i,j] = gR[i] V^dag G[i-1,j] (i > j),  G[i,j] = gL[i] V G[i+1,j] (i < j).
H_slice and V come from junction.slice_blocks_x() (the same model blocks that build
H_C), so the solver is model-agnostic and never needs the dense H_C.
"""

import numpy as np
from numpy.linalg import inv, solve
from scipy.linalg import block_diag

from .leads import surface_eps_pair


# ---------------------------------------------------------------------------
#  bookkeeping for the boundary subspace S
# ---------------------------------------------------------------------------
class _BoundarySubspace:
    """
    S = {col 0} U {col nx-1} U {row 0} U {row ny-1}, stored slice by slice
    (slice = column ix), sites inside a slice ordered by iy.

    posL / posR : S-indices of column 0 / nx-1, iy ascending  (matches junction.idx_L/R)
    posT / posB : S-indices of row 0 / ny-1,    ix ascending  (matches the ribbon ordering)
    """

    def __init__(self, nx, ny, dof=4):
        if nx < 2:
            raise ValueError("nx must be >= 2 (left and right leads would coincide)")
        self.nx, self.ny, self.dof = nx, ny, dof
        edge_rows = sorted({0, ny - 1})
        self.sel_sites = [list(range(ny)) if ix in (0, nx - 1) else edge_rows for ix in range(nx)] # creates a list of lists, keeping all rows for the first and last columns, and only the edge rows for the middle columns
        self.sel_flat = [np.concatenate([np.arange(dof * iy, dof * (iy + 1)) for iy in s])
                         for s in self.sel_sites] # convert the sel_sites list into matrix positions
        widths = [len(f) for f in self.sel_flat]
        self.offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
        self.dim_S = int(self.offsets[-1])

        pos, k = {}, 0
        for ix in range(nx):
            for iy in self.sel_sites[ix]:
                pos[(ix, iy)] = k
                k += 1
        f = lambda ix, iy: np.arange(dof * pos[(ix, iy)], dof * (pos[(ix, iy)] + 1))
        self.posL = np.concatenate([f(0, iy) for iy in range(ny)])
        self.posR = np.concatenate([f(nx - 1, iy) for iy in range(ny)])
        self.posT = np.concatenate([f(ix, 0) for ix in range(nx)])
        self.posB = np.concatenate([f(ix, ny - 1) for ix in range(nx)])

    def block_slice(self, ix):
        return slice(int(self.offsets[ix]), int(self.offsets[ix + 1]))


def _g0_on_subspace(z, H_slice, V, Sigma_ends, sub):
    """
    g0 = P A0^-1 P^dag with A0 = z - H - sum(Sigma_ends), via RGF along x.

    z          : (N_E,) complex (E + i eta)
    H_slice    : (d, d) Hamiltonian of one column (d = 4 ny)
    V          : (d, d) H[ix, ix+1]
    Sigma_ends : {slice_index: (N_E, d, d)} self-energies on single slices
    Returns (N_E, dim_S, dim_S).
    """
    n = sub.nx
    d = H_slice.shape[0]
    Vd = V.conj().T
    base = z[:, None, None] * np.eye(d, dtype=complex)[None] - H_slice[None]
    A = np.broadcast_to(base, (n,) + base.shape).copy()
    for s, Sig in Sigma_ends.items():
        A[s] -= Sig

    gL, gR = np.empty_like(A), np.empty_like(A)
    gL[0] = inv(A[0])
    for i in range(1, n):
        gL[i] = inv(A[i] - Vd @ gL[i - 1] @ V)
    gR[n - 1] = inv(A[n - 1])
    for i in range(n - 2, -1, -1):
        gR[i] = inv(A[i] - V @ gR[i + 1] @ Vd)

    # diagonal blocks, restricted to the kept columns of each slice
    D = []
    for j in range(n):
        Aj = A[j].copy()
        if j > 0:
            Aj -= Vd @ gL[j - 1] @ V
        if j < n - 1:
            Aj -= V @ gR[j + 1] @ Vd
        D.append(inv(Aj)[:, :, sub.sel_flat[j]])

    g0 = np.zeros((z.shape[0], sub.dim_S, sub.dim_S), dtype=complex)
    off = sub.offsets
    # rows i >= j:  B = [G(i,0) | G(i,1) | ... | G(i,i)] (kept columns), G(i,j) = gR[i] V^dag G(i-1,j)
    B = D[0]
    g0[:, sub.block_slice(0), :off[1]] = B[:, sub.sel_flat[0], :]
    for i in range(1, n):
        B = np.concatenate([gR[i] @ (Vd @ B), D[i]], axis=2)
        g0[:, sub.block_slice(i), :off[i + 1]] = B[:, sub.sel_flat[i], :]
    # rows i < j:   B = [G(i,i) | ... | G(i,n-1)],                G(i,j) = gL[i] V G(i+1,j)
    B = D[n - 1]
    for i in range(n - 2, -1, -1):
        B = np.concatenate([D[i], gL[i] @ (V @ B)], axis=2)
        g0[:, sub.block_slice(i), off[i]:] = B[:, sub.sel_flat[i], :]
    return g0


# ---------------------------------------------------------------------------
#  main solver
# ---------------------------------------------------------------------------
class RGFFourTerminal:
    """
    RGF + exact boundary-Dyson solver for FourTerminalJunction.
    Same output contract as FastPhaseSweep.channels_at_phi and junction.channels.

        junction = FourTerminalJunction(p)
        rgf = RGFFourTerminal(junction, E_sweep, phi_ref=0.0)
        ch  = rgf.channels_at_phi(np.pi, side_name='right')

    Parameters
    ----------
    junction      : FourTerminalJunction
    E_sweep       : (N_E,) real energies
    phi_ref       : phase at which the top ribbon is decimated (any value; exact for all phi)
    chunk         : energies per RGF batch (None -> sized to ~mem_budget_gb)
    """

    def __init__(self, junction, E_sweep, phi_ref=0.0, chunk=None,
                 mem_budget_gb=2.0, verbose=False):
        self.junction = junction
        p = junction.p
        self.p = p
        self.nx, self.ny = p.nx, p.ny
        self.E_sweep = np.atleast_1d(np.asarray(E_sweep, dtype=float))
        self.phi_ref = phi_ref
        self.N_E = len(self.E_sweep)
        self.sub = _BoundarySubspace(p.nx, p.ny)

        d = 4 * p.ny
        if chunk is None:
            per_E = (3 * p.nx * d * d + self.sub.dim_S ** 2) * 16
            chunk = max(1, int(mem_budget_gb * 1e9 / per_E))
        self.chunk = int(min(chunk, self.N_E))
        self.verbose = verbose

        self._build_static()
        self._run_rgf()

    # ---------------- static blocks ----------------
    def _build_static(self):
        # Slice blocks straight from the junction's model blocks: no dense H_C needed.
        self.H_slice, self.V = self.junction.slice_blocks_x()
        ny = self.ny
        self.e_idx = np.where(np.tile([True, True, False, False], ny))[0]
        self.h_idx = np.where(np.tile([False, False, True, True], ny))[0]

    def check_against_H_C(self):
        """
        Verify that (H_slice, V) reproduce junction.H_C exactly (builds the dense H_C;
        meant for tests).  Checks that no element couples columns more than one apart
        and that every diagonal / first off-diagonal column block equals H_slice / V.
        """
        nx, ny = self.nx, self.ny
        H_C = self.junction.H_C
        cols = lambda ix: (4 * (ix + nx * np.arange(ny))[:, None] + np.arange(4)[None, :]).ravel()
        r, c = np.nonzero(H_C)
        ix_of = (np.arange(H_C.shape[0]) // 4) % nx
        ok = bool(np.all(np.abs(ix_of[r] - ix_of[c]) <= 1))
        for ix in range(nx):
            ci = cols(ix)
            ok &= np.array_equal(H_C[np.ix_(ci, ci)], self.H_slice)
            if ix + 1 < nx:
                ok &= np.array_equal(H_C[np.ix_(ci, cols(ix + 1))], self.V)
        return ok

    # ---------------- phi-independent part ----------------
    def _run_rgf(self):
        p, J, sub = self.p, self.junction, self.sub
        nx, ny, N_E = self.nx, self.ny, self.N_E
        z = self.E_sweep + 1j * p.eta
        zb = z[:, None, None]
        zN = zb * np.eye(4 * ny, dtype=complex)[None]
        zS = zb * np.eye(4 * nx, dtype=complex)[None]

        Sigma_L = J.lead_L.self_energy(zN)
        Sigma_R = J.lead_R.self_energy(zN)
        self.Gamma_L = 1j * (Sigma_L - Sigma_L.conj().transpose(0, 2, 1))
        self.Gamma_R = 1j * (Sigma_R - Sigma_R.conj().transpose(0, 2, 1))

        top_ref = J._make_ribbon(self.phi_ref, p.tc_top, is_bot=False)
        bot = J.ribbon_bot
        if np.array_equal(top_ref.H_onsite, bot.H_onsite) and np.array_equal(top_ref.V_hop, bot.V_hop):
            # same ribbon, opposite growth direction: one Sancho-Rubio run gives both surfaces
            eps_bot, eps_top = surface_eps_pair(bot.H_onsite, bot.V_hop, zS, p, 'sc ribbons')
        else:
            eps_top, eps_bot = top_ref.surface_eps(zS), bot.surface_eps(zS)
        self.gT_inv_ref = zS - eps_top
        self.gB_inv = zS - eps_bot
        W = block_diag(top_ref.V_coupling, J.ribbon_bot.V_coupling)       # H_{s,C}, (8nx, 8nx)
        Wd = W.conj().T

        Q = np.concatenate([sub.posL, sub.posR])
        P = np.concatenate([sub.posT, sub.posB])
        nQ, nP = len(Q), len(P)
        self.n_L = len(sub.posL)
        self.g0_QQ = np.empty((N_E, nQ, nQ), dtype=complex)
        self.Lft = np.empty((N_E, nQ, nP), dtype=complex)      # g0_QP W^dag
        self.Rgt = np.empty((N_E, nP, nQ), dtype=complex)      # W g0_PQ
        self.Cpp = np.empty((N_E, nP, nP), dtype=complex)      # W g0_PP W^dag
        blk = lambda M, r, c: M[:, r[:, None], c[None, :]]

        for s in range(0, N_E, self.chunk):
            e = min(s + self.chunk, N_E)
            if self.verbose:
                print(f"  RGF chunk {s}:{e}")
            g0 = _g0_on_subspace(z[s:e], self.H_slice, self.V,
                                 {0: Sigma_L[s:e], nx - 1: Sigma_R[s:e]}, sub)
            self.g0_QQ[s:e] = blk(g0, Q, Q)
            self.Lft[s:e] = blk(g0, Q, P) @ Wd
            self.Rgt[s:e] = W @ blk(g0, P, Q)
            self.Cpp[s:e] = W @ blk(g0, P, P) @ Wd

    # ---------------- per-phi part ----------------
    def _gT_inv(self, phi):
        dphi = phi - self.phi_ref
        u = np.tile(np.array([1.0, 1.0, np.exp(1j * dphi), np.exp(1j * dphi)]), self.nx)
        return (u.conj()[None, :, None]) * self.gT_inv_ref * (u[None, None, :])

    def green_blocks(self, phi):
        """Retarded G_LL, G_LR, G_RL, G_RR at phase phi, each (N_E, 4ny, 4ny)."""
        nT = 4 * self.nx
        K = -self.Cpp.copy()
        K[:, :nT, :nT] += self._gT_inv(phi)
        K[:, nT:, nT:] += self.gB_inv
        G = self.g0_QQ + self.Lft @ solve(K, self.Rgt)
        n = self.n_L
        return G[:, :n, :n], G[:, :n, n:], G[:, n:, :n], G[:, n:, n:]

    def channels_at_phi(self, phi, side_name=None):
        """Channels 'ee','hh','eh_cross','he_cross','eh_local','he_local' at phase phi."""
        G_LL, G_LR, G_RL, G_RR = self.green_blocks(phi)
        e_idx, h_idx = self.e_idx, self.h_idx
        sub_ = lambda M, r, c: M[:, r[:, None], c[None, :]]
        dag = lambda M: M.conj().transpose(0, 2, 1)
        T = lambda G1, B1, G2, B2: np.trace(G1 @ B1 @ G2 @ B2, axis1=1, axis2=2).real

        Ge_L, Gh_L = sub_(self.Gamma_L, e_idx, e_idx), sub_(self.Gamma_L, h_idx, h_idx)
        Ge_R, Gh_R = sub_(self.Gamma_R, e_idx, e_idx), sub_(self.Gamma_R, h_idx, h_idx)

        def side(name):
            if name == 'left':
                G, G_self = G_LR, G_LL
                own_e, own_h, oth_e, oth_h = Ge_L, Gh_L, Ge_R, Gh_R
            else:
                G, G_self = G_RL, G_RR
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
