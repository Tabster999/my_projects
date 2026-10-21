"""
Andreev spectrum, topological transition and Majorana localisation of the Scharf junction.

THREE STEPS, one cell each (run top to bottom in the interactive window):
  1. E(phi) at E_Z = 0, k = 0: count the Andreev bound states; choose W so there are 3-4.
  2. Switch on E_Z: E(phi) at fixed fields, then the class-D invariant Q(E_Z, phi) together
     with the k = 0 gap and the gap of the WHOLE strip (min over k).
  3. Finite device (L = 2000 nm): LDOS along the junction at a topological point.  This tells
     you whether the zero modes are LOCALISED at the ends or PROPAGATE along the junction.

THE CROSS-SECTION.  Steps 1-2 use the infinitely long junction (translation invariant along the
interfaces, Bloch momentum k along them) cut across: n_S SC sites, the n_across normal sites,
n_S SC sites, hard walls at both outer edges -- exactly Scharf's geometry (W_S = 450 nm, App. A).
Every block comes from my_functions (model_onsite / model_hop), so the cross-section is the same
lattice model as the transport code, except that the transport ribbons are semi-infinite.
Package axes: x = along the interfaces (k lives here), y = across.  See scharf_ldos's AXIS SWAP.

THE INVARIANT.  Class D in 1D: Q = sign[Pf M(0) Pf M(pi)], M(k) = -i W H(k) W^dag the BdG
Hamiltonian in the Majorana basis (real antisymmetric at k = 0, pi).  Q = -1 topological: a
finite junction then carries one Majorana at each end.  Q only changes where the gap at k = 0
(or pi) closes; the gap at OTHER k can close without changing Q, and that is the question for
step 3 -- a strip that is topological but nearly gapless at finite k has Majoranas that leak
along the junction instead of sitting at its ends.
"""
# %% --- IMPORTS ---
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                  # interactive window: run from computations/
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
DATA = HERE / 'plot_npz'
import scharf_ldos as S
from scharf_ldos import myf
from my_functions.hamiltonians import model_onsite, model_hop, region_alpha, bond_m0


# %% --- SETUP ---
# a = 20 nm (Scharf, k_F a ~ 0.5) and L = 2000 nm, whatever scharf_ldos.PP is currently set to:
# the cross-section cost grows ~ 1/a^3, and every number in this file was measured here.
PP = replace(S.PP, a=20.0, L=2000.0)
W_S = 900.0                        # nm, SC width per side.  Scharf uses 450; ABS near Delta are
                                   # converged only from ~900 (0.226 vs 0.206 meV at W = 100 nm)


# --- the cross-section Hamiltonian ---
def _blocks(p, n_S):
    """(onsite blocks per row, Vx per row, Vy per bond) of the cross-section, top SC first."""
    ons_top = model_onsite(p, 's', phi=p.phi)
    ons_bot = model_onsite(p, 's', phi=0.0)
    ons_c = model_onsite(p, 'c')
    VxS = model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    VxC = model_hop(p, 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
    VyS = model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    VyC = model_hop(p, 'y', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
    VcT = model_hop(p, 'y', p.tc_top, bond_m0(p.m0, p.m0_c), bond=True)
    VcB = model_hop(p, 'y', p.tc_bot, bond_m0(p.m0, p.m0_c), bond=True)
    ons = [ons_top] * n_S + [ons_c] * p.ny + [ons_bot] * n_S
    Vx = [VxS] * n_S + [VxC] * p.ny + [VxS] * n_S
    Vy = [VyS] * (n_S - 1) + [VcT] + [VyC] * (p.ny - 1) + [VcB] + [VyS] * (n_S - 1)
    return ons, Vx, Vy


def H_cross(p, k, n_S):
    """Bloch Hamiltonian H(k) of the cross-section, (4 n, 4 n) with n = 2 n_S + ny."""
    ons, Vx, Vy = _blocks(p, n_S)
    n = len(ons)
    H = np.zeros((4 * n, 4 * n), complex)
    for j in range(n):
        H[4*j:4*j+4, 4*j:4*j+4] = ons[j] + Vx[j] * np.exp(1j * k) + Vx[j].conj().T * np.exp(-1j * k)
    for j in range(n - 1):
        H[4*j:4*j+4, 4*j+4:4*j+8] = Vy[j]
        H[4*j+4:4*j+8, 4*j:4*j+4] = Vy[j].conj().T
    return H


def n_S_of(pp, W_S=W_S):
    return max(1, round(W_S / pp.a))


def abs_spectrum(pp, phis, E_Z=0.0, k=0.0, n_levels=12, W_S=W_S):
    """The n_levels eigenvalues closest to zero (meV), shape (len(phis), n_levels), sorted."""
    n_S = n_S_of(pp, W_S)
    out = np.empty((len(phis), n_levels))
    for j, ph in enumerate(phis):
        E = np.linalg.eigvalsh(H_cross(pp.params(E_Z=E_Z, phi=float(ph)), k, n_S)) * pp.t
        out[j] = np.sort(E[np.argsort(np.abs(E))[:n_levels]])
    return out


def count_abs(pp, phis=np.linspace(0, 2 * np.pi, 41), W_S=W_S, below=0.98):
    """
    At E_Z = 0, k = 0: the number of DISTINCT positive subgap levels E < below * Delta at phi = 0
    (Kramers pairs counted once -- time reversal makes every k = 0 level twofold at E_Z = 0),
    and the largest such number over phi.
    """
    E = abs_spectrum(pp, phis, E_Z=0.0, n_levels=60, W_S=W_S)
    def distinct(row):
        pos = np.sort(row[(row > 1e-9) & (row < below * pp.Delta)])
        return int(np.sum(np.diff(np.r_[-1.0, pos]) > 1e-6))
    n = np.array([distinct(r) for r in E])
    return int(n[0]), int(n.max())


# --- the Majorana basis and the invariant ---
_W1 = np.array([[1, 0, 0, -1],             # gamma = W Psi, Psi = (c_up, c_dn, c_dn^dag, -c_up^dag)
                [1j, 0, 0, 1j],
                [0, 1, 1, 0],
                [0, 1j, -1j, 0]]) / np.sqrt(2)


def pfaffian_sign_logabs(A):
    """(sign, log|Pf|) of a REAL antisymmetric matrix (Parlett-Reid with pivoting, as pfapack)."""
    A = np.array(A, float)
    n = len(A)
    if n % 2:
        return 0.0, -np.inf
    sgn, logpf = 1.0, 0.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(A[k + 1:, k])))
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            sgn = -sgn
        if A[k, k + 1] == 0.0:
            return 0.0, -np.inf
        sgn *= np.sign(A[k, k + 1])
        logpf += np.log(abs(A[k, k + 1]))
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], tau)
    return sgn, logpf


def majorana_form(H):
    """M = -i W H W^dag: real antisymmetric at k = 0, pi (the residual is returned to check)."""
    n = len(H) // 4
    W = np.kron(np.eye(n), _W1)
    M = -1j * W @ H @ W.conj().T
    return M.real, max(np.abs(M.imag).max(), np.abs(M + M.T).max())


def invariant(p, n_S):
    """Q = sign Pf M(0) * sign Pf M(pi):  -1 topological, +1 trivial."""
    s0, _ = pfaffian_sign_logabs(majorana_form(H_cross(p, 0.0, n_S))[0])
    s1, _ = pfaffian_sign_logabs(majorana_form(H_cross(p, np.pi, n_S))[0])
    return s0 * s1


def gaps(p, n_S, ks=np.linspace(0, np.pi, 61)):
    """(|E|_min at k = 0, min over k of |E|_min, the k of that minimum), in units of t."""
    g = np.array([np.abs(np.linalg.eigvalsh(H_cross(p, k, n_S))).min() for k in ks])
    return g[0], g.min(), ks[int(np.argmin(g))]


class PhasePoint:
    """Picklable for myf.scan: (E_Z [meV], phi) -> [Q, gap(k=0), gap(all k), k_min] (gaps in meV)."""
    def __init__(self, pp, W_S=W_S, ks=np.linspace(0, np.pi, 41)):
        self.pp, self.n_S, self.ks = pp, n_S_of(pp, W_S), ks

    def __call__(self, E_Z, phi):
        p = self.pp.params(E_Z=float(E_Z), phi=float(phi))
        g0, gmin, kmin = gaps(p, self.n_S, self.ks)
        return np.array([invariant(p, self.n_S), g0 * self.pp.t, gmin * self.pp.t, kmin])


class QPoint:
    """Picklable for myf.scan: (E_Z [meV], phi) -> [Q] only (no k scan, ~10x cheaper than PhasePoint)."""
    def __init__(self, pp, W_S=W_S):
        self.pp, self.n_S = pp, n_S_of(pp, W_S)

    def __call__(self, E_Z, phi):
        return np.array([invariant(self.pp.params(E_Z=float(E_Z), phi=float(phi)), self.n_S)])


def q_map(pp, E_Z, phi, name, n_workers=14):
    """Q on the (E_Z, phi) grid, (len(E_Z), len(phi)), cached in plot_npz/scharf_abs_<name>.npz."""
    import scharf_abs
    res = myf.scan(scharf_abs.QPoint(pp), {'E_Z': np.asarray(E_Z), 'phi': np.asarray(phi)},
                   n_workers=n_workers, save=str(DATA / f'scharf_abs_{name}.npz'))
    return res['values'][..., 0]


def bands(pp, E_Z, phi, ks, n_levels=8, W_S=W_S):
    """E(k) along the interfaces (meV), the n_levels levels closest to zero, (len(ks), n_levels)."""
    p, n_S = pp.params(E_Z=E_Z, phi=phi), n_S_of(pp, W_S)
    out = np.empty((len(ks), n_levels))
    for j, k in enumerate(ks):
        E = np.linalg.eigvalsh(H_cross(p, k, n_S)) * pp.t
        out[j] = np.sort(E[np.argsort(np.abs(E))[:n_levels]])
    return out


K_GRID = np.r_[np.linspace(0, 1.5, 76), np.linspace(1.6, np.pi, 12)]   # k_F a ~ 0.5: dense there


# --- the finite device: n_along columns of the cross-section, hard walls at y = +-L/2 ---
def H_device(pp, E_Z, phi, W_S=W_S):
    """
    Sparse BdG Hamiltonian of the finite junction, (4 N, 4 N) with N = n_along * (2 n_S + n_across):
    the cross-section repeated n_along times along the interfaces, decoupled ends (tc_barr = 0,
    Scharf's hard walls).  Index = 4 * (row + n_rows * column) + dof.
    """
    import scipy.sparse as sp
    p, n_S = pp.params(E_Z=E_Z, phi=phi), n_S_of(pp, W_S)
    H0 = H_cross(p, 0.0, n_S)
    ons, Vx, _ = _blocks(p, n_S)
    T = np.zeros_like(H0)
    for j, V in enumerate(Vx):
        T[4*j:4*j+4, 4*j:4*j+4] = V
    H0 = H0 - T - T.conj().T                                  # the k-independent part
    nx = pp.n_along
    shift = sp.diags([np.ones(nx - 1)], [1], shape=(nx, nx))  # column c -> c + 1 is H_{c, c+1} = T
    H = sp.kron(sp.eye(nx), sp.csr_matrix(H0)) + sp.kron(shift, sp.csr_matrix(T))
    H = H + sp.kron(shift.T, sp.csr_matrix(T.conj().T))
    return H.tocsc(), n_S


def device_states(pp, E_Z, phi, n_states=40, W_S=W_S):
    """
    The n_states eigenstates closest to E = 0 of the finite device (shift-invert).
    Returns (E in meV sorted, rho) with rho[n, column, row] = |psi_n|^2 summed over spin and
    particle-hole -- columns run along the interfaces (paper y), rows across (paper x, SC included).
    """
    import scipy.sparse.linalg as sla
    H, n_S = H_device(pp, E_Z, phi, W_S)
    E, V = sla.eigsh(H, k=n_states, sigma=0.0, which='LM')
    o = np.argsort(E)
    n_rows = 2 * n_S + pp.n_across
    rho = (np.abs(V[:, o]) ** 2).reshape(pp.n_along, n_rows, 4, n_states).sum(2)
    return E[o] * pp.t, np.moveaxis(rho, -1, 0), n_S


def ldos_from_states(E_states, rho, E, eta):
    """LDOS(E, column, row) = sum_n rho_n Lorentzian(E - E_n; eta) -- valid only inside the window
    the states cover (|E| < max |E_states|)."""
    L = (eta / np.pi) / ((np.asarray(E)[:, None] - E_states[None, :]) ** 2 + eta ** 2)
    return np.einsum('en,nij->eij', L, rho)


def _save(fig, name):
    fig.savefig(HERE / f'scharf_abs_{name}.png', dpi=110)


def phase_scan(pp, E_Z, phi, name, n_workers=14):
    """PhasePoint on the (E_Z, phi) grid, cached in plot_npz/scharf_abs_<name>.npz (resumable).
    Returns dict Q, gap0, gap, kmin (gaps in meV), each (len(E_Z), len(phi)), plus the axes."""
    import scharf_abs                                  # workers must find the class in the MODULE
    res = myf.scan(scharf_abs.PhasePoint(pp, ks=K_GRID), {'E_Z': np.asarray(E_Z), 'phi': np.asarray(phi)},
                   n_workers=n_workers, save=str(DATA / f'scharf_abs_{name}.npz'))
    V = res['values']
    return dict(Q=V[..., 0], gap0=V[..., 1], gap=V[..., 2], kmin=V[..., 3],
                E_Z=np.asarray(E_Z), phi=np.asarray(phi))


def xi_fit(prof, a):
    """Decay length (nm) of a profile along y: fit log(prof) over the first half, from its maximum
    to where it has fallen 100x (or the middle)."""
    half = prof[:len(prof) // 2]
    i0 = int(np.argmax(half))
    stop = next((i for i in range(i0, len(half)) if half[i] < half[i0] / 100), len(half))
    sel = np.arange(i0, stop)
    if len(sel) < 4:
        return np.nan
    return -1 / np.polyfit(sel * a, np.log(half[sel]), 1)[0]


# %% --- STEP 1: ABS at E_Z = 0, k = 0 -- how many, and which W gives 3-4 ---
if __name__ == "__main__":
    S.show_summary(PP)
    print("W [nm]   E_T [meV]   Delta/E_T   ABS at phi=0   max over phi")
    for W in (100, 200, 400, 600, 700, 800, 1000):
        q = replace(PP, W=W)
        n0, nmax = count_abs(q)
        print(f"{W:6.0f}   {q.E_T:9.3f}   {q.Delta / q.E_T:9.2f}   {n0:12d}   {nmax:12d}")

    W_CHOSEN = 800.0                                   # 4 ABS (distinct Kramers doublets)
    PA = replace(PP, W=W_CHOSEN)
    PHI = np.linspace(0, 2 * np.pi, 121)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, q in zip(axs, (PP, PA)):
        E = abs_spectrum(q, PHI, E_Z=0.0, n_levels=24)
        ax.plot(PHI / np.pi, E / q.Delta, 'k-', lw=1)
        ax.axhline(1, color='r', lw=0.8, ls='--')
        ax.axhline(-1, color='r', lw=0.8, ls='--')
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$\phi/\pi$')
        ax.set_title(f'W = {q.W:.0f} nm, $E_Z$ = 0, $k$ = 0:  {count_abs(q)[0]} ABS (Kramers pairs)',
                     fontsize=10)
    axs[0].set_ylabel(r'$E/\Delta$')
    plt.tight_layout()
    _save(fig, 'step1_EZ0')
    plt.show()


# %% --- STEP 2: switch on E_Z -- spectrum and the topological transition ---
if __name__ == "__main__":
    EZ_SHOW = (0.0, 0.05, 0.1, 0.2, 0.3)
    fig, axs = plt.subplots(1, len(EZ_SHOW), figsize=(3.2 * len(EZ_SHOW), 3.6), sharey=True)
    for ax, EZ in zip(axs, EZ_SHOW):
        E = abs_spectrum(PA, PHI, E_Z=EZ, n_levels=16)
        ax.plot(PHI / np.pi, E / PA.Delta, 'k-', lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f'$E_Z$ = {EZ} meV', fontsize=10)
        ax.set_xlabel(r'$\phi/\pi$')
    axs[0].set_ylabel(r'$E/\Delta$  ($k$ = 0)')
    fig.suptitle(f'W = {PA.W:.0f} nm', fontsize=9)
    plt.tight_layout()
    _save(fig, 'step2_EZ_spectra')
    plt.show()

    PHI_GRID = np.linspace(0, 2 * np.pi, 33)
    MAPS = {'W800': (PA, np.round(np.linspace(0, 0.4, 41), 4)),
            'W100': (PP, np.round(np.linspace(0, 1.2, 41), 4))}
    PD = {name: phase_scan(q, ez, PHI_GRID, name='phase_' + name) for name, (q, ez) in MAPS.items()}

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for row, (name, R) in zip(axs, PD.items()):
        row[0].pcolormesh(R['E_Z'], R['phi'] / np.pi, R['Q'].T, cmap='coolwarm', vmin=-1, vmax=1,
                          shading='nearest')
        row[0].set_title(f'{name}:  Q  (blue = topological)', fontsize=10)
        for ax, key, ttl in ((row[1], 'gap0', 'gap at k = 0'), (row[2], 'gap', 'gap over ALL k')):
            im = ax.pcolormesh(R['E_Z'], R['phi'] / np.pi, R[key].T * 1e3, cmap='magma',
                               vmin=0, vmax=np.nanmax(R['gap0']) * 1e3, shading='nearest')
            ax.contour(R['E_Z'], R['phi'] / np.pi, R['Q'].T, levels=[0], colors='c', linewidths=1)
            fig.colorbar(im, ax=ax, label=r'gap [$\mu$eV]')
            ax.set_title(f'{name}:  {ttl}', fontsize=10)
        for ax in row:
            ax.set_xlabel(r'$E_Z$ [meV]')
            ax.set_ylabel(r'$\phi/\pi$')
    plt.tight_layout()
    _save(fig, 'step2_phase')
    plt.show()

    for name, R in PD.items():
        j = int(np.argmin(np.abs(R['phi'] - np.pi)))
        topo = R['Q'][:, j] < 0
        i = int(np.argmax(np.where(topo, R['gap'][:, j], -1)))
        print(f"{name}, phi = pi: topological at {topo.sum()}/{len(topo)} fields in "
              f"{R['E_Z'][topo].min():.3f}..{R['E_Z'][topo].max():.3f} meV;"
              f"  largest gap over all k = {R['gap'][i, j] * 1e3:.2f} ueV at E_Z = {R['E_Z'][i]:.3f}"
              f"  (gap at k=0 there {R['gap0'][i, j] * 1e3:.1f} ueV, minimum at k a = {R['kmin'][i, j]:.2f})")


# %% --- STEP 3: are the Majoranas localised or propagating?  E(k) and the finite device ---
if __name__ == "__main__":
    def best_point(R):
        """phi = pi, the topological field with the largest gap over all k."""
        j = int(np.argmin(np.abs(R['phi'] - np.pi)))
        i = int(np.argmax(np.where(R['Q'][:, j] < 0, R['gap'][:, j], -1)))
        return float(R['E_Z'][i]), float(R['phi'][j])

    POINTS = {'W100': (PP, best_point(PD['W100'])), 'W800': (PA, best_point(PD['W800']))}
    ETA_LDOS = 0.5e-3                                  # meV: resolves the MBS splitting, unlike 0.05 Delta
    DEV = {}

    fig, axs = plt.subplots(len(POINTS), 4, figsize=(17, 4.2 * len(POINTS)))
    for row, (name, (q, (EZ, ph))) in zip(axs, POINTS.items()):
        ks = np.linspace(0, 1.5, 151)
        Ek = bands(q, EZ, ph, ks, n_levels=8)
        row[0].plot(ks, Ek * 1e3, 'k-', lw=0.8)
        row[0].set_ylim(-0.6 * q.Delta * 1e3, 0.6 * q.Delta * 1e3)
        row[0].set_xlabel(r'$k a$ (along the interfaces)')
        row[0].set_ylabel(r'$E$ [$\mu$eV]')
        row[0].set_title(f'{name}: infinite strip, $E_Z$={EZ:.3f} meV, $\\phi$={ph / np.pi:.2f}$\\pi$',
                         fontsize=9)

        E, rho, n_S = DEV[name] = device_states(q, EZ, ph, n_states=40)
        y = (np.arange(q.n_along) + 0.5) * q.a - q.L / 2
        x = (np.arange(2 * n_S + q.n_across) + 0.5 - n_S) * q.a - q.W / 2
        row[1].plot(np.arange(len(E)), E * 1e3, 'o', ms=3)
        row[1].axhline(0, color='0.6', lw=0.5)
        row[1].set_xlabel('level index')
        row[1].set_ylabel(r'$E_n$ [$\mu$eV]')
        row[1].set_title(f'finite device L = {q.L:.0f} nm: 40 levels nearest 0', fontsize=9)

        n0 = len(E) // 2                               # the positive state closest to zero
        maj = rho[n0 - 1] + rho[n0]                    # the +-E pair: both Majoranas
        row[2].pcolormesh(x, y, maj, shading='nearest', cmap='cividis')
        row[2].axvline(-q.W / 2, color='w', lw=0.5)
        row[2].axvline(q.W / 2, color='w', lw=0.5)
        row[2].set_xlabel('x [nm] (across; S outside the lines)')
        row[2].set_ylabel('y [nm] (along)')
        row[2].set_title(f'$|\\psi|^2$ of the lowest pair, $E$ = $\\pm${E[n0] * 1e3:.2f} $\\mu$eV', fontsize=9)

        for n, ls in ((n0, '-'), (n0 + 1, '--'), (n0 + 2, ':')):
            row[3].plot(y, rho[n].sum(1), ls, label=f'$E$ = {E[n] * 1e3:.2f} $\\mu$eV')
        row[3].set_xlabel('y [nm]')
        row[3].set_ylabel(r'$\sum_x |\psi|^2$')
        row[3].legend(fontsize=8)
        row[3].set_title('profiles along the junction', fontsize=9)

        prof = maj.sum(1)
        print(f"{name} at E_Z = {EZ:.3f} meV, phi = {ph / np.pi:.2f} pi:"
              f"  MBS splitting delta = {2 * E[n0] * 1e3:.3f} ueV,  next level {E[n0 + 1] * 1e3:.2f} ueV"
              f"  (ratio {E[n0 + 1] / E[n0]:.1f});  end/mid = {prof[0] / prof[len(prof) // 2]:.1f};"
              f"  decay length of the pair ~ {xi_fit(prof, q.a):.0f} nm  (L = {q.L:.0f} nm)")
    plt.tight_layout()
    _save(fig, 'step3_localisation')
    plt.show()

    # LDOS(E, y) summed across the normal strip, from the device eigenstates (energy window = theirs)
    fig, axs = plt.subplots(1, len(POINTS), figsize=(6 * len(POINTS), 4))
    for ax, (name, (q, (EZ, ph))) in zip(axs, POINTS.items()):
        E, rho, n_S = DEV[name]
        Ew = np.linspace(-0.9, 0.9, 241) * np.abs(E).max()
        D = ldos_from_states(E, rho[:, :, n_S:n_S + q.n_across].sum(2, keepdims=True), Ew, ETA_LDOS)[..., 0]
        y = (np.arange(q.n_along) + 0.5) * q.a - q.L / 2
        im = ax.pcolormesh(y, Ew * 1e3, D, shading='nearest', cmap='viridis')
        fig.colorbar(im, ax=ax, label=r'LDOS (normal strip)')
        ax.set_xlabel('y [nm] (along)')
        ax.set_ylabel(r'$E$ [$\mu$eV]')
        ax.set_title(f'{name}: $E_Z$={EZ:.3f} meV, $\\phi$={ph / np.pi:.2f}$\\pi$, '
                     f'$\\eta$ = {ETA_LDOS * 1e3} $\\mu$eV', fontsize=9)
    plt.tight_layout()
    _save(fig, 'step3_ldos')
    plt.show()
