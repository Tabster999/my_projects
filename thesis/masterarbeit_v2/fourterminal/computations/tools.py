"""
tools.py -- everything the Scharf ABS studies share, as plain functions.

The study scripts (study_*.py) import ONLY this file (plus ldos for PhysParams).
Nothing here runs on import; it only defines things.

CONTENTS
  1. Parameters        base_params(), case(), field_grid()
  2. Cross-section     the infinitely long junction cut across: ABS spectrum, invariant Q, gaps
  3. Transport, E = 0  kappa, G, local Andreev, det r for one field value
  4. Finite T          channels on an energy grid, thermal(), peak bookkeeping
  5. LDOS              the transport geometry with the probes decoupled
  6. Scans and files   run_scan() (parallel + cached), save_fig()

HOW THE PARALLEL SCANS WORK (section 6).  myf.scan(func, grid) calls func(**point) for every
point of the grid on several processes.  The function must be defined at module level (here),
and the fixed arguments (the parameter set q, ...) are attached with functools.partial:
    run_scan(partial(phase_point, q), {'E_Z': EZ, 'phi': PHI}, cache='name')
calls phase_point(q, E_Z=..., phi=...) for every (E_Z, phi) and stores the results in
plot_npz/name.npz, so a rerun just loads them.  Cache files do NOT know the parameters:
delete them (or use a new name) when you change parameters.
"""
# %% --- IMPORTS ---
import sys
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                  # interactive window: run from computations/
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
DATA = HERE / 'plot_npz'
from ldos import PhysParams, myf            # noqa: E402
from my_functions.hamiltonians import model_onsite, model_hop, region_alpha, bond_m0   # noqa: E402
from my_functions.transport import _kernels        # noqa: E402




# 1. PARAMETERS
def base_params():
    """
    The parameter set of all ABS studies (pure Rashba).  Every cached result in plot_npz/scharf_abs*
    was computed with exactly these values -- they are written out here on purpose, so that
    editing ldos.PP does not silently change these studies.
    """
    q = PhysParams(a=20.0, m_eff=0.038,
                   Delta=0.25, mu_S=1.0, mu_N=3.0, mu_C=0.7,     # mu_N: probe leads, mu_C: junction
                   alpha=16.0, beta=0.0, soc_axis='100',         # pure Rashba
                   g=10.0, B=0.0, theta_z=None,
                   W=150.0, L=2000.0, tc=1.0, tc_barr=0.0, eta_rel=0.05,
                   tc_barr_transport=0.2, eta_rel_transport=1e-5, kT_transport=0.0)
    q.theta_z = q.theta_z_optimal() + np.pi        # field perpendicular to the SOC field (3 pi / 2)
    return q


def case(W, L=2000.0, Delta=0.25, alpha=16.0, beta=0.0, tc=0.2, a=20.0):
    """
    A junction of width W, length L (nm), gap Delta (meV), Rashba alpha and Dresselhaus beta
    (meV nm), probe coupling tc.  The field is always turned perpendicular to the SOC field
    (theta_z_optimal, which depends on alpha and beta), plus pi as in ldos.PP.
    """
    q = replace(base_params(), W=W, L=L, Delta=Delta, alpha=alpha, beta=beta, tc_barr_transport=tc, a=a)
    q.theta_z = q.theta_z_optimal() + np.pi
    return q


def mixed_soc(soc_amp=16.0, theta_soc=0.15 * np.pi):
    """(alpha, beta) for mixed Rashba + Dresselhaus SOC, as in ldos: amplitude and angle."""
    return soc_amp * np.cos(theta_soc), soc_amp * np.sin(theta_soc)


def field_grid(q, n):
    """n Zeeman energies (meV) from 0 to 1.5 E_T (at least 0.3 meV): covers the first diamond."""
    return np.round(np.linspace(0.0, max(0.3, 1.5 * q.E_T), n), 5)


# =============================================================================
# 2. CROSS-SECTION: the infinitely long junction, cut across
# =============================================================================
# Translation invariant along the interfaces (Bloch momentum k there, in units of 1/a), and across:
# n_S superconducting sites | the normal strip | n_S superconducting sites, hard walls outside.
# W_S = 900 nm per side (Scharf uses 450; the levels near Delta converge only from ~900).
W_S = 900.0
K_GRID = np.r_[np.linspace(0, 1.5, 76), np.linspace(1.6, np.pi, 12)]   # k a; dense where k_F a ~ 0.5


def n_S_of(q, W_S=W_S):
    """Number of SC sites per side."""
    return max(1, round(W_S / q.a))


def H_cross(p, k, n_S):
    """
    Bloch Hamiltonian H(k) of the cross-section, a (4n x 4n) matrix with n = 2 n_S + ny.
    p is a my_functions Params (from q.params(E_Z=..., phi=...)).  The blocks come from the same
    model_onsite / model_hop as the transport code, so it is the same lattice model.
    """
    ons_top, ons_bot, ons_c = model_onsite(p, 's', phi=p.phi), model_onsite(p, 's', phi=0.0), model_onsite(p, 'c')
    VxS = model_hop(p, 'x', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    VxC = model_hop(p, 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
    VyS = model_hop(p, 'y', p.t_s, p.m0, alpha=region_alpha(p, 's'))
    VyC = model_hop(p, 'y', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))
    VcT = model_hop(p, 'y', p.tc_top, bond_m0(p.m0, p.m0_c), bond=True)
    VcB = model_hop(p, 'y', p.tc_bot, bond_m0(p.m0, p.m0_c), bond=True)
    ons = [ons_top] * n_S + [ons_c] * p.ny + [ons_bot] * n_S          # one 4x4 block per row
    Vx = [VxS] * n_S + [VxC] * p.ny + [VxS] * n_S                     # hopping along (gives the k)
    Vy = [VyS] * (n_S - 1) + [VcT] + [VyC] * (p.ny - 1) + [VcB] + [VyS] * (n_S - 1)   # across
    n = len(ons)
    H = np.zeros((4 * n, 4 * n), complex)
    for j in range(n):
        H[4*j:4*j+4, 4*j:4*j+4] = ons[j] + Vx[j] * np.exp(1j * k) + Vx[j].conj().T * np.exp(-1j * k)
    for j in range(n - 1):
        H[4*j:4*j+4, 4*j+4:4*j+8] = Vy[j]
        H[4*j+4:4*j+8, 4*j:4*j+4] = Vy[j].conj().T
    return H


def levels(q, E_Z, phi, k=0.0, n_levels=12):
    """The n_levels eigenvalues closest to zero (meV), sorted."""
    E = np.linalg.eigvalsh(H_cross(q.params(E_Z=E_Z, phi=float(phi)), k, n_S_of(q))) * q.t
    return np.sort(E[np.argsort(np.abs(E))[:n_levels]])


def abs_spectrum(q, phis, E_Z=0.0, k=0.0, n_levels=12):
    """E(phi): levels() for every phase, shape (len(phis), n_levels), meV."""
    return np.array([levels(q, E_Z, ph, k, n_levels) for ph in phis])


def bands(q, E_Z, phi, ks, n_levels=8):
    """E(k) along the interfaces: levels() for every k, shape (len(ks), n_levels), meV."""
    return np.array([levels(q, E_Z, phi, k, n_levels) for k in ks])


def count_abs(q, below=0.9):
    """
    Number of Andreev bound states at E_Z = 0, k = 0, phi = 0: distinct positive levels below
    `below` * Delta (at E_Z = 0 every level is a Kramers pair, counted once).
    """
    row = levels(q, 0.0, 0.0, n_levels=60)
    pos = np.sort(row[(row > 1e-9) & (row < below * q.Delta)])
    return int(np.sum(np.diff(np.r_[-1.0, pos]) > 1e-6))


# --- the topological invariant Q (class D) ---
# Q = sign Pf M(0) * sign Pf M(pi), with M = -i W H W^dag the Hamiltonian in the Majorana basis
# (real and antisymmetric at k = 0 and pi).  Q = -1: topological, one Majorana at each end.
_W1 = np.array([[1, 0, 0, -1],             # Majorana basis for Psi = (c_up, c_dn, c_dn^dag, -c_up^dag)
                [1j, 0, 0, 1j],
                [0, 1, 1, 0],
                [0, 1j, -1j, 0]]) / np.sqrt(2)


def pfaffian_sign(A):
    """Sign of the Pfaffian of a real antisymmetric matrix (Parlett-Reid elimination with pivoting)."""
    A = np.array(A, float)
    n, sgn = len(A), 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(A[k + 1:, k])))
        if kp != k + 1:                                          # swap rows/columns k+1 and kp
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            sgn = -sgn
        if A[k, k + 1] == 0.0:
            return 0.0
        sgn *= np.sign(A[k, k + 1])
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], tau)
    return sgn


def invariant(p, n_S):
    """Q of the cross-section: -1 topological, +1 trivial.  p: my_functions Params."""
    W = np.kron(np.eye(len(H_cross(p, 0.0, n_S)) // 4), _W1)
    Q = 1.0
    for k in (0.0, np.pi):
        M = (-1j * W @ H_cross(p, k, n_S) @ W.conj().T).real
        Q *= pfaffian_sign(M)
    return Q


def gap_at(p, n_S, k):
    """Smallest |E| of H_cross(p, k), units of t.  H_cross is block tridiagonal (bandwidth 7): banded solver."""
    from scipy.linalg import eigvals_banded
    H = H_cross(p, k, n_S)
    ab = np.zeros((8, len(H)), complex)
    for i in range(8):
        ab[i, :len(H) - i] = np.diagonal(H, -i)
    return np.abs(eigvals_banded(ab, lower=True)).min()


def gaps(p, n_S, ks=K_GRID):
    """(gap at k = 0, smallest gap over all k, the k a where it sits), in units of t."""
    g = np.array([gap_at(p, n_S, k) for k in ks])
    return g[0], g.min(), ks[int(np.argmin(g))]


def gap_refined(p, n_S, ks=K_GRID):
    """
    Smallest gap over all k (units of t) and its k a: the three lowest local minima of the grid, each
    refined by a 1D minimisation between its neighbours (the dip is ~1/xi_M wide, narrower than the
    grid, so gaps() overestimates it).
    """
    from scipy.optimize import minimize_scalar
    g = np.array([gap_at(p, n_S, k) for k in ks])
    gp = np.r_[np.inf, g, np.inf]
    loc = np.flatnonzero((gp[1:-1] <= gp[:-2]) & (gp[1:-1] <= gp[2:]))
    best = (g.min(), ks[int(np.argmin(g))])
    for i in loc[np.argsort(g[loc])][:3]:
        lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
        r = minimize_scalar(lambda k: gap_at(p, n_S, k), bounds=(lo, hi), method='bounded',
                            options={'xatol': 1e-7})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def k_grid(a):
    """K_GRID for lattice constant a (nm): the same PHYSICAL k range, in units of 1/a."""
    s = a / 20.0
    return np.r_[np.linspace(0, 1.5 * s, 76), np.linspace(1.5 * s + 0.1, np.pi, 12)]


def phase_point(q, E_Z, phi, ks=K_GRID):
    """One (E_Z, phi): [Q, gap at k = 0 (meV), gap over all k = Scharf's Delta_top (meV), k a of it]."""
    p = q.params(E_Z=float(E_Z), phi=float(phi))
    g0, gmin, kmin = gaps(p, n_S_of(q), ks)
    return np.array([invariant(p, n_S_of(q)), g0 * q.t, gmin * q.t, kmin])


def q_point(q, E_Z, phi):
    """One (E_Z, phi): [Q] only (no k scan, ~10x cheaper than phase_point)."""
    return np.array([invariant(q.params(E_Z=float(E_Z), phi=float(phi)), n_S_of(q))])


def gap0_point(q, E_Z, phi):
    """One (E_Z, phi): [gap at k = 0 (meV)] (no invariant)."""
    p = q.params(E_Z=float(E_Z), phi=float(phi))
    return np.array([np.abs(np.linalg.eigvalsh(H_cross(p, 0.0, n_S_of(q)))).min() * q.t])


def phase_line(q, EZ, cache, phi=np.pi):
    """phase_point along E_Z at fixed phi.  Returns dict E_Z, Q, gap0, gap, kmin (gaps in meV)."""
    V = run_scan(partial(phase_point, q), {'E_Z': np.asarray(EZ), 'phi': np.array([phi])}, cache)[:, 0]
    return dict(E_Z=np.asarray(EZ), Q=V[:, 0], gap0=V[:, 1], gap=V[:, 2], kmin=V[:, 3])


def q_map(q, EZ, phis, cache):
    """Q on the (E_Z, phi) grid, shape (len(EZ), len(phis))."""
    return run_scan(partial(q_point, q), {'E_Z': np.asarray(EZ), 'phi': np.asarray(phis)}, cache)[..., 0]


def best_topological(line):
    """Index of the field with the largest gap over all k inside Q = -1 (on a phase_line)."""
    return int(np.argmax(np.where(line['Q'] < 0, line['gap'], -1)))


# =============================================================================
# 3. TRANSPORT AT E = 0 (full transport geometry: finite centre, semi-infinite ribbons and leads)
# =============================================================================
def det_r(rgf, phi):
    """
    det r at E = 0, r = reflection matrix of the left lead (Fulga et al., PRB 83, 155429):
    r = -1 + i Gamma^1/2 G_LL Gamma^1/2 on the propagating channels.  Real; its sign is the
    topological invariant WITH the leads attached (-1: the lead sees an isolated Majorana).
    """
    G_LL = rgf.green_blocks(phi)[0][0]
    w, U = np.linalg.eigh(rgf.Gamma_L[0])
    keep = w > 1e-6 * w.max()
    s, U = np.sqrt(w[keep]), U[:, keep]
    r = -np.eye(len(s)) + 1j * s[:, None] * (U.conj().T @ G_LL @ U) * s[None, :]
    return np.linalg.det(r).real


def transport_point(q, E_Z, phi=np.pi):
    """
    One field value, E = 0 (T -> 0), probe coupling q.tc_barr_transport:
    [kappa/kappa0, G/G0, local Andreev (e^2/h), det r, number of failed lead solves].
        kappa = T_ee + T_he (crossed)   heat: Majorana -> 1/2
        G     = T_ee - T_he (crossed)   charge: Majorana -> 0
    """
    n0 = myf.leads.N_UNRELIABLE
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=float(E_Z))), np.array([0.0]))
    c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(float(phi), 'left').items()}
    return np.array([c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'],
                     det_r(rgf, float(phi)), myf.leads.N_UNRELIABLE - n0])


def transport_phis(q, E_Z, phis):
    """transport_point for every phase in phis from ONE RGF set-up (the phase is cheap): (len(phis), 5)."""
    n0 = myf.leads.N_UNRELIABLE
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=float(E_Z))), np.array([0.0]))
    rgf.green_blocks = lru_cache(maxsize=1)(rgf.green_blocks)       # channels and det r share one solve
    out = []
    for ph in phis:
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(float(ph), 'left').items()}
        out.append([c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'],
                    det_r(rgf, float(ph)), 0.0])
    out = np.array(out)
    out[:, 4] = myf.leads.N_UNRELIABLE - n0
    return out


def transport_line(q, EZ, cache, n_workers=None):
    """transport_point along E_Z at phi = pi.  Returns dict E_Z, kappa, G, andreev, det.
    n_workers: fewer than N_WORKERS for small a (each worker needs several GB at a = 2.5 nm)."""
    V = run_scan(partial(transport_point, q), {'E_Z': np.asarray(EZ)}, cache, n_workers=n_workers or N_WORKERS)
    if V.shape[1] == 4:                          # older caches: [kappa, G, andreev, n_bad], no det r
        V = np.insert(V, 3, np.nan, axis=1)
    if np.nansum(V[:, 4]):
        print(f"    WARNING: {int(np.nansum(V[:, 4]))} failed lead solves in {cache}", flush=True)
    return dict(E_Z=np.asarray(EZ), kappa=V[:, 0], G=V[:, 1], andreev=V[:, 2], det=V[:, 3])


def kappa_peaks(t, k_min=0.3):
    """Indices of the local maxima of kappa above k_min."""
    k = t['kappa']
    return np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > k_min)) + 1


def is_majorana_like(t, i, tol_G=0.05):
    """A kappa peak counts as Majorana-like if 0.45 <= kappa <= 0.55 and |G|/kappa < tol_G."""
    k, G = t['kappa'][i], t['G'][i]
    return (k >= 0.45) & (k <= 0.55) & (np.abs(G) / k < tol_G)


def peak_stats(t):
    """Peak bookkeeping of a transport line: counts, heights, charge."""
    pk = kappa_peaks(t)
    r = np.abs(t['G'][pk]) / t['kappa'][pk]
    return dict(n=len(pk), good=int(is_majorana_like(t, pk).sum()), over=int((t['kappa'][pk] > 0.55).sum()),
                neutral=int((r < 0.05).sum()), kmax=float(np.nanmax(t['kappa'])),
                rmed=float(np.median(r)) if len(pk) else np.nan,
                det_neg=float(np.mean(t['det'] < 0)) if 'det' in t else np.nan)


def print_peaks(label, t):
    """Print where the kappa peaks are, their heights and |G|/kappa."""
    pk = kappa_peaks(t)
    print(f"  {label}: kappa peaks > 0.3 at E_Z = {np.round(t['E_Z'][pk], 3)}", flush=True)
    print(f"      heights {np.round(t['kappa'][pk], 3)},  |G|/kappa {np.round(np.abs(t['G'][pk]) / t['kappa'][pk], 3)}",
          flush=True)


# =============================================================================
# 4. FINITE TEMPERATURE
# =============================================================================
# The probe averages the transmissions over |E| ~ a few kT.  Strategy: compute the six channels on a
# fixed energy grid E >= 0 once (expensive), then do the thermal integral for any T (cheap).
K_B = 0.0861733                                    # meV / K
CHANNELS = ('ee', 'hh', 'eh_cross', 'he_cross', 'eh_local', 'he_local')


def channel_point(q, E_Z, E, phis=(np.pi,)):
    """The six channels at energy E (meV) for each phase in phis, concatenated (6 per phase)."""
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=float(E_Z))),
                              np.array([float(E) / q.t]))
    out = []
    for ph in phis:
        c = rgf.channels_at_phi(ph, 'left')
        out += [float(np.asarray(c[k])[0]) for k in CHANNELS]
    return np.array(out)


def thermal(ch, kT, E):
    """
    (kappa/kappa0, G/G0, local Andreev) at temperature kT (meV) from channels ch on the grid E >= 0
    (particle-hole symmetry folds E < 0 onto E > 0).  Same kernels as myf.linear_response.
    The grid must reach ~16 kT (35 ueV is enough up to ~50 mK).
    """
    S_th = ch['ee'] + ch['he_cross'] + ch['hh'] + ch['eh_cross']
    S_el = ch['ee'] - ch['he_cross'] + ch['hh'] - ch['eh_cross']
    S_a = ch['eh_local'] + ch['he_local']
    if kT == 0:
        return S_th[0] / 2, S_el[0] / 2, S_a[0]
    x = np.asarray(E) / kT
    m = x <= 16.0
    W_th, W_el = _kernels(x[m])
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return trap(W_th * S_th[m], x[m]), trap(W_el * S_el[m], x[m]), 2 * trap(W_el * S_a[m], x[m])


def thermal_line(V, kT, E, phase=0):
    """thermal() for every field of a channel scan V (n_EZ, n_E, 6 * n_phases) -> (n_EZ, 3)."""
    return np.array([thermal({k: V[i, :, 6 * phase + n] for n, k in enumerate(CHANNELS)}, kT, E)
                     for i in range(len(V))])


def kT_of_mK(T_mK):
    return K_B * T_mK * 1e-3


# =============================================================================
# 5. LDOS (transport geometry, probes decoupled: tc_barr = 0, as in Scharf)
# =============================================================================
def ldos_profile(q, E_Z, eta, every=1, E=0.0, phi=np.pi):
    """LDOS at energy E (meV) summed across the normal strip, vs y (nm, along the junction)."""
    p = replace(q.params(E_Z=E_Z, phi=phi), eta=eta / q.t, tc_barr=0.0)
    cols = list(range(0, q.n_along, every))
    e, h = myf.LocalGreen(myf.FourTerminalJunction(p), np.array([E / q.t]), cols=cols).ldos(phi)
    return (np.array(cols) + 0.5) * q.a - q.L / 2, (e + h)[0].sum(1)


def ldos_spectrum(q, E_Z, E_meV, eta, n_end=3, chunk=12, phi=np.pi):
    """LDOS(E) summed over the first n_end columns (END) and n_end MIDDLE columns -> (end, middle)."""
    p = replace(q.params(E_Z=E_Z, phi=phi), eta=eta / q.t, tc_barr=0.0)
    J = myf.FourTerminalJunction(p)
    mid = q.n_along // 2
    cols = list(range(n_end)) + list(range(mid - n_end // 2, mid - n_end // 2 + n_end))
    end, cen = [], []
    for k in range(0, len(E_meV), chunk):
        e, h = myf.LocalGreen(J, np.asarray(E_meV[k:k + chunk]) / q.t, cols=cols).ldos(phi)
        D = (e + h).sum(2)
        end.append(D[:, :n_end].sum(1))
        cen.append(D[:, n_end:].sum(1))
    return np.concatenate(end), np.concatenate(cen)


def spectrum_point(q, E_Z, E_meV, eta):
    """ldos_spectrum as one scan point: [[end LDOS(E)], [middle LDOS(E)]], shape (2, n_E)."""
    return np.array(ldos_spectrum(q, float(E_Z), E_meV, eta, chunk=6))


def ldos_map_E0(q, E_Z, eta, every=1, phi=np.pi):
    """Real-space LDOS(E = 0): (y_nm, x_nm, A[y, x]) over every `every`-th column."""
    p = replace(q.params(E_Z=float(E_Z), phi=phi), eta=eta / q.t, tc_barr=0.0)
    cols = list(range(0, q.n_along, every))
    e, h = myf.LocalGreen(myf.FourTerminalJunction(p), np.array([0.0]), cols=cols).ldos(phi)
    x_nm, y_nm = q.coords(cols)
    return y_nm, x_nm, (e + h)[0]


class _DoubledEdges:
    """Subspace for _g0_on_subspace on the DOUBLED columns (2 x 4 ny): rows 0 and ny-1 of both copies."""

    def __init__(self, nx, ny):
        d = 4 * ny
        edge = np.r_[0:4, 4 * (ny - 1):4 * ny]                      # top and bottom site of one copy
        self.nx = nx
        self.sel_flat = [np.r_[edge, d + edge]] * nx                # copy 1 (8 dof), then copy 2 (8 dof)
        self.offsets = 16 * np.arange(nx + 1)
        self.dim_S = 16 * nx
        o = self.offsets[:-1, None]
        top, bot = (o + np.arange(4)).ravel(), (o + 4 + np.arange(4)).ravel()
        self.P1 = np.r_[top, bot]                                   # same order as W: all tops, then all bottoms
        self.P2 = self.P1 + 8

    def block_slice(self, ix):
        return slice(int(self.offsets[ix]), int(self.offsets[ix + 1]))


def _trace_g0(z, H, V, Sigma_ends, nx):
    """sum_j Tr g0(j, j) of the column chain (same recursion as _g0_on_subspace, diagonal only)."""
    inv = np.linalg.inv
    d = H.shape[-1]
    I = np.eye(d)
    A = [z * I - (H[j] if H.ndim == 3 else H) - Sigma_ends.get(j, 0) for j in range(nx)]
    Vd = V.conj().T
    gL = [inv(A[0])]
    for i in range(1, nx):
        gL.append(inv(A[i] - Vd @ gL[-1] @ V))
    gR = [None] * nx
    gR[nx - 1] = inv(A[nx - 1])
    for i in range(nx - 2, -1, -1):
        gR[i] = inv(A[i] - V @ gR[i + 1] @ Vd)
    tr = 0.0
    for j in range(nx):
        Aj = A[j] - (Vd @ gL[j - 1] @ V if j > 0 else 0) - (V @ gR[j + 1] @ Vd if j < nx - 1 else 0)
        tr += np.trace(inv(Aj))
    return tr


def dos_total_phis(q, E_Z, phis, eta, E=0.0):
    """
    DOS at energy E (1/meV) summed over ALL sites of the centre, for every phase in phis; probes decoupled
    (tc_barr = 0).  Exact, without the all-site Green's function (16 GB per energy at a = 5):
        Tr G = Tr g0 + Tr(K^-1 W (g0^2)_PP W^dag),   K = g_ribbons^-1 - W g0_PP W^dag,
    P = the edge rows touching the ribbons, W = ribbon coupling (as LocalGreen).  (g0^2)_PP is the
    off-diagonal block of the inverse of [[A, -1], [0, A]] (A = z - H_C - Sigma_leads), still block
    tridiagonal, so _g0_on_subspace gives it on the 16 nx edge dofs.
    """
    from scipy.linalg import block_diag
    from my_functions.rgf import _g0_on_subspace
    from my_functions.leads import surface_eps_pair
    p = replace(q.params(E_Z=float(E_Z), phi=np.pi), eta=eta / q.t, tc_barr=0.0)
    J = myf.FourTerminalJunction(p)
    nx, ny = p.nx, p.ny
    z = complex(E / q.t, p.eta)
    zN, zS = z * np.eye(4 * ny)[None], z * np.eye(4 * nx)[None]
    S_L, S_R = J.lead_L.self_energy(zN)[0], J.lead_R.self_energy(zN)[0]
    top = J._make_ribbon(0.0, p.tc_top, is_bot=False)            # phi_ref = 0: top == bottom
    bot = J.ribbon_bot
    if np.array_equal(top.H_onsite, bot.H_onsite) and np.array_equal(top.V_hop, bot.V_hop):
        eps_bot, eps_top = surface_eps_pair(bot.H_onsite, bot.V_hop, zS, p, 'sc ribbons')
    else:
        eps_top, eps_bot = top.surface_eps(zS), bot.surface_eps(zS)
    gT_inv, gB_inv = (zS - eps_top)[0], (zS - eps_bot)[0]
    W = block_diag(top.V_coupling, bot.V_coupling)
    H, V = J.slice_blocks_x()
    cols = J.column_onsite()
    if cols is not None:
        H = cols
    d = 4 * ny
    Id, Zd = np.eye(d), np.zeros((d, d))
    H2 = (np.array([np.block([[h, Id], [Zd, h]]) for h in H]) if H.ndim == 3
          else np.block([[H, Id], [Zd, H]]))
    V2 = np.block([[V, Zd], [Zd, V]])
    sub = _DoubledEdges(nx, ny)
    g2 = _g0_on_subspace(np.array([z]), H2, V2,
                         {0: block_diag(S_L, S_L)[None], nx - 1: block_diag(S_R, S_R)[None]}, sub)[0]
    import scipy.sparse as sps
    Ws = sps.csr_matrix(W)                                          # 4x4 blocks per site
    WMWd = lambda M: (Ws @ (Ws @ M).conj().T).conj().T
    C_pp = WMWd(g2[np.ix_(sub.P1, sub.P1)])                         # W g0_PP W^dag
    C_sq = WMWd(g2[np.ix_(sub.P1, sub.P2)])                         # W (g0^2)_PP W^dag
    tr0 = _trace_g0(z, H, V, {0: S_L, nx - 1: S_R}, nx)
    # the phase sits only in the top block of K: eliminate the bottom ribbon once (Schur complement
    # S = K_TT - K_TB K_BB^-1 K_BT), then each phase costs one solve of size nT instead of 2 nT:
    #   Tr(K^-1 C) = Tr(K_BB^-1 C_BB) + Tr(S^-1 E),   E = (C_TT - Y C_BT) - (C_TB - Y C_BB) X
    nT = 4 * nx
    t_, b_ = slice(0, nT), slice(nT, None)
    Kbb_inv = np.linalg.inv(gB_inv - C_pp[b_, b_])
    X = Kbb_inv @ -C_pp[b_, t_]                                     # K_BB^-1 K_BT
    Y = -C_pp[t_, b_] @ Kbb_inv                                     # K_TB K_BB^-1
    F = C_pp[t_, t_] - C_pp[t_, b_] @ X                             # S = U^dag gT^-1 U - F
    E = (C_sq[t_, t_] - Y @ C_sq[b_, t_]) - (C_sq[t_, b_] - Y @ C_sq[b_, b_]) @ X
    tr0 = tr0 + np.einsum('ij,ji->', Kbb_inv, C_sq[b_, b_])
    out = []
    for ph in phis:
        w = np.exp(1j * ph)
        u = np.tile(np.array([1.0, 1.0, w, w]), nx)
        S = u.conj()[:, None] * gT_inv * u[None, :] - F
        out.append(-(tr0 + np.trace(np.linalg.solve(S, E))).imag)
    return np.array(out) / (np.pi * q.t)


def peak_energies(E, D):
    """Non-negative energies of the local maxima of D(E), lowest first."""
    i = np.flatnonzero((D[1:-1] > D[:-2]) & (D[1:-1] >= D[2:])) + 1
    return E[i][E[i] >= 0]


def splitting(q, E_Z, half_window=0.02, n=161, eta=0.2e-3):
    """Lowest peak of the end LDOS(E) in ueV (= half the Majorana splitting), plus (E, end LDOS)."""
    E = np.linspace(-half_window, half_window, n)
    end, _ = ldos_spectrum(q, E_Z, E, eta)
    pk = peak_energies(E, end)
    return (pk[0] * 1e3 if len(pk) else np.nan), E, end


# =============================================================================
# 6. SCANS AND FILES
# =============================================================================
N_WORKERS = 14


def run_scan(func, grid, cache=None, n_workers=N_WORKERS):
    """
    func(**point) on every point of grid {name: values} in parallel; returns the values array,
    shape (*grid shape, *result shape).  cache='name' stores/loads plot_npz/name.npz (resumable).
    """
    save = str(DATA / f'{cache}.npz') if cache else None
    return myf.scan(func, grid, n_workers=n_workers, save=save)['values']


PLOTS = HERE / 'plots'


def save_fig(fig, name):
    """Save a figure as computations/plots/<name>.png."""
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f'{name}.png', dpi=110)
