"""Band-structure diagnostics: bulk Chern number of the SC ribbons and the junction spectrum with PBC in x."""

import numpy as np

from .hamiltonians import model_onsite, model_hop, region_alpha, bond_alpha, bond_m0, make_row_hamiltonian
from .leads import Lead


def chern_number(p, N=48):
    """
    Chern number of the bulk SC-ribbon Hamiltonian (occupied = negative-energy BdG bands),
    Fukui-Hatsugai lattice method on an N x N k-grid.  Uses t_s, mu_s, delta, Bz_s, ... of p.
    """
    o = model_onsite(p, 's')
    X, Y = model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's')), model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    k = np.linspace(-np.pi, np.pi, N, endpoint=False)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    ex, ey = np.exp(1j * KX)[..., None, None], np.exp(1j * KY)[..., None, None]
    H = o + X * ex + X.conj().T / ex + Y * ey + Y.conj().T / ey
    U = np.linalg.eigh(H)[1][..., :2]
    link = lambda A, B: np.linalg.det(np.swapaxes(A.conj(), -1, -2) @ B)
    Ux, Uy = np.roll(U, -1, 0), np.roll(U, -1, 1)
    Uxy = np.roll(Ux, -1, 1)
    return np.angle(link(U, Ux) * link(Ux, Uxy) * link(Uxy, Uy) * link(Uy, U)).sum() / (2 * np.pi)


def junction_spectrum_kx(p, kx_vals, E_vals):
    """
    Spectral density A(E, kx) = -Im Tr G_C / pi of the junction with periodic boundary
    conditions in x: central column of ny sites, both SC ribbons attached (no normal leads).
    """
    ny = p.ny
    z = np.asarray(E_vals) + 1j * p.eta
    z4 = z[:, None, None] * np.eye(4)[None]
    zC = z[:, None, None] * np.eye(4 * ny)[None]
    VxC, VyC = model_hop(p, 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c')), model_hop(p, 'y', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
    VxS, VyS = model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's')), model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    VcT = model_hop(p, 'y', p.tc_top, bond_m0(p.m0, p.m0_c), alpha=bond_alpha(p, 's', 'c'))
    VcB = model_hop(p, 'y', p.tc_bot, bond_m0(p.m0, p.m0_c), alpha=bond_alpha(p, 's', 'c'))
    A = np.zeros((len(z), len(kx_vals)))
    for n, k in enumerate(kx_vals):
        bloch = lambda h0, Vx: h0 + Vx * np.exp(1j * k) + Vx.conj().T * np.exp(-1j * k)
        H_C = make_row_hamiltonian(ny, bloch(model_onsite(p, 'c'), VxC), VyC)
        top = Lead('top', bloch(model_onsite(p, 's', phi=p.phi), VxS), VyS, VcT, p, br=False, dual=True)
        bot = Lead('bot', bloch(model_onsite(p, 's', phi=0.0), VxS), VyS, VcB, p, br=True, dual=False)
        Sigma = np.zeros_like(zC)
        Sigma[:, :4, :4] += top.self_energy(z4)
        Sigma[:, -4:, -4:] += bot.self_energy(z4)
        G = np.linalg.inv(zC - H_C[None] - Sigma)
        A[:, n] = -np.trace(G, axis1=1, axis2=2).imag / np.pi
    return A


def ribbon_gap(p, nk=301):
    """
    Quasi-1D gap at E = 0 of one SC ribbon (width nx, periodic along y): min over k of |E(k)|.
    If the ribbon is topological (chern_number != 0) this is the hybridization gap of its two
    edge modes across the width -- they only act as propagating Majorana channels if it is << eta.
    """
    from numpy.linalg import eigvalsh
    from scipy.linalg import block_diag
    H0 = make_row_hamiltonian(p.nx, model_onsite(p, 's'), model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's')))
    V = block_diag(*[model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))] * p.nx)
    return min(np.min(np.abs(eigvalsh(H0 + V * np.exp(1j * k) + V.conj().T * np.exp(-1j * k))))
               for k in np.linspace(0, np.pi, nk))


class _ColumnSubspace:
    """Boundary ring (cols 0, nx-1, rows 0, ny-1) plus extra FULL columns `cols` (slice = column ix)."""

    def __init__(self, nx, ny, cols, dof=4):
        self.nx, self.ny = nx, ny
        full = set(cols) | {0, nx - 1}
        edge = sorted({0, ny - 1})
        self.sel_sites = [list(range(ny)) if ix in full else edge for ix in range(nx)]
        self.sel_flat = [np.concatenate([np.arange(dof * iy, dof * (iy + 1)) for iy in s]) for s in self.sel_sites]
        self.offsets = np.concatenate([[0], np.cumsum([len(f) for f in self.sel_flat])]).astype(int)
        self.dim_S = int(self.offsets[-1])
        pos, k = {}, 0
        for ix in range(nx):
            for iy in self.sel_sites[ix]:
                pos[(ix, iy)] = k
                k += 1
        f = lambda ix, iy: np.arange(dof * pos[(ix, iy)], dof * (pos[(ix, iy)] + 1))
        self.site = f
        self.posT = np.concatenate([f(ix, 0) for ix in range(nx)])
        self.posB = np.concatenate([f(ix, ny - 1) for ix in range(nx)])

    def block_slice(self, ix):
        return slice(int(self.offsets[ix]), int(self.offsets[ix + 1]))


class LocalGreen:
    """
    Local density of states inside the finite junction (all four leads attached), on selected
    full columns.  Same exact construction as RGFFourTerminal (x-RGF with normal leads, SC
    ribbons added by one Dyson step), with the requested columns added to the subspace;
    changing the phase is cheap.

        lg = LocalGreen(junction, E_vals, cols=[nx // 2])      # or cols=range(nx) for a full map
        ldos_e, ldos_h = lg.ldos(np.pi)                         # each (N_E, len(cols), ny)

    ldos_e / ldos_h: electron / hole part of -Im G_ii / pi summed over spin.  A Majorana
    state has equal electron and hole weight, so ldos_e = ldos_h where it lives.
    """

    def __init__(self, junction, E_vals, cols, phi_ref=0.0):
        from scipy.linalg import block_diag
        from .rgf import _g0_on_subspace
        from .leads import surface_eps_pair
        p = junction.p
        nx, ny = p.nx, p.ny
        self.nx, self.ny, self.cols, self.phi_ref = nx, ny, list(cols), phi_ref
        sub = _ColumnSubspace(nx, ny, self.cols)
        z = np.asarray(E_vals, dtype=float) + 1j * p.eta
        zb = z[:, None, None]
        zN, zS = zb * np.eye(4 * ny)[None], zb * np.eye(4 * nx)[None]

        Sigma_L, Sigma_R = junction.lead_L.self_energy(zN), junction.lead_R.self_energy(zN)
        top = junction._make_ribbon(phi_ref, p.tc_top, is_bot=False)
        bot = junction.ribbon_bot
        if np.array_equal(top.H_onsite, bot.H_onsite) and np.array_equal(top.V_hop, bot.V_hop):
            eps_bot, eps_top = surface_eps_pair(bot.H_onsite, bot.V_hop, zS, p, 'sc ribbons')
        else:
            eps_top, eps_bot = top.surface_eps(zS), bot.surface_eps(zS)
        self.gT_inv_ref, self.gB_inv = zS - eps_top, zS - eps_bot
        W = block_diag(top.V_coupling, bot.V_coupling)

        H_slice, V = junction.slice_blocks_x()
        g0 = _g0_on_subspace(z, H_slice, V, {0: Sigma_L, nx - 1: Sigma_R}, sub)
        P = np.concatenate([sub.posT, sub.posB])
        M = np.concatenate([sub.site(ix, iy) for ix in self.cols for iy in range(ny)])
        self.g0_MM = g0[:, M, M]                                          # diagonal only
        self.Lft = g0[:, M[:, None], P[None, :]] @ W.conj().T
        self.Rgt = W @ g0[:, P[:, None], M[None, :]]
        self.Cpp = W @ g0[:, P[:, None], P[None, :]] @ W.conj().T

    def ldos(self, phi):
        dphi = phi - self.phi_ref
        u = np.tile(np.array([1.0, 1.0, np.exp(1j * dphi), np.exp(1j * dphi)]), self.nx)
        nT = 4 * self.nx
        K = -self.Cpp.copy()
        K[:, :nT, :nT] += (u.conj()[None, :, None]) * self.gT_inv_ref * (u[None, None, :])
        K[:, nT:, nT:] += self.gB_inv
        G_diag = self.g0_MM + np.einsum('nij,nji->ni', self.Lft, np.linalg.solve(K, self.Rgt))
        A = (-G_diag.imag / np.pi).reshape(len(G_diag), len(self.cols), self.ny, 4)
        return A[..., :2].sum(-1), A[..., 2:].sum(-1)


def edge_state_profile(p, k_y=0.0, n_states=2):
    """
    Site-resolved weight |psi(x)|^2 of the SC ribbon's n_states lowest states across its width
    (the two Majorana edge states in the topological phase), and their decay length.
    Returns (x, density, xi, energies); xi from |psi|^2 ~ exp(-2x/xi) between the left edge and
    the profile minimum (np.nan if too few points).  Use a wide ribbon, e.g. replace(p, nx=200).
    """
    from numpy.linalg import eigh
    from scipy.linalg import block_diag
    H0 = make_row_hamiltonian(p.nx, model_onsite(p, 's'), model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's')))
    V = block_diag(*[model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))] * p.nx)
    w, v = eigh(H0 + V * np.exp(1j * k_y) + V.conj().T * np.exp(-1j * k_y))
    idx = np.argsort(np.abs(w))[:n_states]
    dens = sum((np.abs(v[:, i])**2).reshape(p.nx, 4).sum(1) for i in idx)
    x = np.arange(p.nx)
    xmin = int(np.argmin(dens[:p.nx // 2 + 1]))
    sel = (x >= 2) & (x <= max(6, xmin - 3)) & (dens > 1e-15)
    xi = -2 / np.polyfit(x[sel], np.log(dens[sel]), 1)[0] if sel.sum() > 3 else np.nan
    return x, dens, xi, w[idx]


def coherence_length(p, k_y=0.0):
    """
    Majorana decay length xi of the SC ribbon material [sites]: the slowest evanescent solution
    of the bulk BdG equation at E = 0 (transfer-matrix eigenvalue lambda, xi = -1/ln|lambda|).
    Sets the edge-overlap gap ribbon_gap ~ exp(-nx/xi); rule of thumb nx >~ 12 xi at T -> 0
    (eta = 1e-6) and >~ 7 xi at kT = 1e-3.  Diverges at topological transitions.
    Not the BCS hbar v_F/(pi Delta), which is several times shorter here.
    """
    import scipy.linalg as sl
    Vy = model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    Vx = model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    H0 = model_onsite(p, 's') + Vy * np.exp(1j * k_y) + Vy.conj().T * np.exp(-1j * k_y)
    M = len(H0)
    I, O = np.eye(M), np.zeros((M, M))
    lam = sl.eig(np.block([[O, I], [-Vx.conj().T, -H0]]), np.block([[I, O], [O, Vx]]), right=False)
    inside = np.abs(lam[np.abs(lam) < 1 - 1e-12])
    return -1 / np.log(inside.max()) if inside.size else np.inf
