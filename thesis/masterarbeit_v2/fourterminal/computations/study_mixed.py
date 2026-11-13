"""
The MIXED SOC (Rashba + Dresselhaus 16 meV nm, theta 0.15 pi; the thesis case), a = 10, W 100, mu_C 0.7: oscillation
analysis, the best kappa = 1/2 plateau, what sets the amplitude.

PARTS (one per former script; the cells of each part are tagged [part]):
  oscillation  (was study_oscillation_mixed.py)  STUDY: the oscillation analysis of study_oscillation (a 20, pure Rashba 16) redone for the MIXED SOC at a = 10,
  plateau      (was study_plateau_mixed.py)  STUDY: the best kappa = 1/2 plateau of the MIXED SOC, as a line (2026-11-04).
  amplitude    (was study_amplitude_mixed.py)  STUDY: what sets the AMPLITUDE of eps in the mixed SOC?  (2026-11-04)
  deck         (2026-11-11)  the Rashba-only advisor figures of slides_nov.py redone in the mixed SOC (length, T, S/N)
  combine      (2026-11-11)  recheck of the 10-07 Gresta + Scharf combination claims; the same at the thesis parameters
  angle        (2026-11-12)  the SOC ANGLE: why the mixed SOC is gapless inside Q = -1 (band tilt), and protection / plateau vs theta

Run one part:   python study_mixed.py oscillation      (several: list them; no argument: every part, in order)
In the interactive window the cells run as usual.  run_part(part) replaces the old `if __name__ == "__main__":`
guard: the parallel workers re-import this file and must not re-run the cells.
Names that collided between parts were renamed (see each PART header); cache names are unchanged.
"""
# %% --- IMPORTS AND THE PART SELECTOR ---
import sys
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from functools import partial
import tools as st
from study_oscillation import bloch_blocks, dip_point, box_point, zeros_of, fit_A, slowest_mode
from study_tuning import junction, topo_point, tr_name, EZ as EZ_TU, PHI as PHI_TU
import scipy.sparse as sp

PARTS = ('oscillation', 'plateau', 'amplitude', 'deck', 'combine', 'angle')
_SELECTED = {a for a in sys.argv[1:] if a in PARTS}


def run_part(part):
    """True for the cells of `part` when this file runs as a script: every part, or only those named on
    the command line.  False in the parallel workers (they import this file)."""
    return __name__ == "__main__" and (not _SELECTED or part in _SELECTED)


# ====================================================================================================================
# PART 'oscillation'  (was study_oscillation_mixed.py)
# ====================================================================================================================
# STUDY: the oscillation analysis of study_oscillation (a 20, pure Rashba 16) redone for the MIXED SOC at a = 10,
# the junction of the tuning maps: study_tuning.junction('mixed', 0.7) = W 100, L 2 um, t_c 1, mu_C 0.7, mu_S 1,
# Rashba + Dresselhaus amplitude 16 meV nm at theta 0.15 pi ('100'), phi = pi.  Is
#     eps(E_Z) = A Delta_out e^{-L/xi_M} |cos(k_osc L + phi0)|
# what sets the splitting here too, and is that eps what the transport sees?
#   1. infinite cross-section: one complex-k solve per field -> the three slowest E = 0 modes (xi, Re k; the first
#      is xi_M, k_osc), Delta_top (gap_refined), Q, and the outer gap dip k0, Delta_out, hbar v_A (dip_point)
#   2. closed box (study_oscillation.box_point): eps
#   3. tests: k_osc L advances by exactly pi between zeros of eps; A, phi0 fit; per-lobe A
#   4. transport on the phi = pi line (tau 0.3387 / 0.5): on two-Majorana points det r gives x = eps / Gamma, so
#      Gamma_eff = eps_box / x must be smooth through the eps zeros (eps itself varies by decades), scale like tau^p
#      (p ~ 2.6, study_tuning), and agree with Gamma from the local Andreev Lorentzian at an eps zero.

# %% [oscillation] --- PARAMETERS ---

QM = junction('mixed', 0.7)                                         # the probe bond does not enter 1-3
EZ = np.round(np.arange(0.10, 1.3501, 0.005), 3)
EZ_T = EZ[::2]                                                      # transport line: step 0.01
TAUS = (0.3387, 0.5)
N_MODES = 3
BOX_WORKERS = 12                                                    # 45 s, 1.3 GB peak per box at a = 10


def osc_point(q, E_Z, phi=np.pi):
    """
    [xi_1, k_1, xi_2, k_2, xi_3, k_3 (nm, 1/nm), Delta_top (meV), Q, k of the gap minimum (1/nm),
     k0 (1/nm), Delta_out (meV), hbar v_A (meV nm), xi_hyp (nm), xi_poly (nm)].
    Modes: the slowest distinct decaying E = 0 solutions (+-k partners merged, as study_branches.modes_point).
    """
    p = q.params(E_Z=float(E_Z), phi=phi)
    n_S = st.n_S_of(q)
    H0, H1 = bloch_blocks(p, n_S)
    n = len(H0)
    I, Z = np.eye(n), np.zeros((n, n))
    lam = sla.eig(np.block([[Z, I], [-H1.conj().T, -H0]]), np.block([[I, Z], [Z, H1]]), right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1 - 1e-9) & (np.abs(lam) > 1e-12)]
    xi, k = -q.a / np.log(np.abs(lam)), np.abs(np.angle(lam)) / q.a
    modes = []
    for j in np.argsort(-xi):
        if any(abs(xi[j] - a) < 1e-4 * a and abs(k[j] - b) < 1e-7 for a, b in modes):
            continue
        modes.append((xi[j], k[j]))
        if len(modes) == N_MODES:
            break
    g, kg = st.gap_refined(p, n_S, st.k_grid(q.a))
    dip = dip_point(q, E_Z, phi, kc=modes[0][1] * q.a)
    return np.r_[np.ravel(modes), g * q.t, st.invariant(p, n_S), kg / q.a, dip]


def gamma_point(k, cases):
    """[Gamma (ueV), andreev(E=0)/2]: where (eh_local + he_local)/2 falls to half (bisection in ln E), cases[k] =
    (tau, E_Z) at phi = pi (as study_tuning.gamma_job)."""
    t, E_Z = cases[int(k)]
    q = junction('mixed', 0.7, t)
    A = lambda E: (lambda c: (c['eh_local'] + c['he_local']) / 2)(dict(zip(st.CHANNELS, st.channel_point(q, E_Z, E))))
    A0, lo, hi = A(0.0), 1e-5, 1e-1
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if A(mid) > A0 / 2 else (lo, mid)
    return np.array([np.sqrt(lo * hi) * 1e3, A0])


def dip_redo(i, ez, kc):
    """dip_point (with the 2026-11-11 step fix) at ez[i], phi = pi, kc = k_osc a from the cached complex-k solve."""
    return dip_point(QM, ez[int(i)], kc=kc[int(i)])


# %% [oscillation] --- 1. INFINITE CROSS-SECTION ---
if run_part('oscillation'):
    V = st.run_scan(partial(osc_point, QM), {'E_Z': EZ}, 'scharf_absOM_modes_mixed_a10')
    # The dip columns (k0 .. xi_poly) predate the dip_point step fix of 2026-11-11, which matters from E_Z ~ 0.9 up (at
    # 1.15 the old h = 2e-3 was 0.8 / xi_hyp: hbar v_A 2.5 % low).  Redone there (2026-11-12); the rest is unchanged.
    redo = np.flatnonzero(EZ >= 0.9)
    D2 = st.run_scan(partial(dip_redo, ez=tuple(EZ[redo]), kc=tuple(V[redo, 1] * QM.a)), {'i': np.arange(len(redo))},
                     'scharf_absOM_dip_mixed_a10_fix', n_workers=6, chunksize=2)
    ch = D2[:, 3] / V[redo, 12] - 1
    print(f"dip columns redone for E_Z >= 0.9 ({len(redo)} fields): xi_hyp changed by {np.nanmin(ch):+.1%} .. "
          f"{np.nanmax(ch):+.1%}, xi_poly by {np.nanmin(D2[:, 4] / V[redo, 13] - 1):+.1%} .. "
          f"{np.nanmax(D2[:, 4] / V[redo, 13] - 1):+.1%}")
    V = V.copy()
    V[redo, 9:] = D2
    xi_M, k_osc, xi_2, k_2, xi_3, k_3, D_top, Q, k_gap, k0, D_out, hv_A, xi_hyp, xi_poly = V.T
    topo = Q < 0
    print(f"Q = -1 for E_Z {EZ[topo].min():.3f} .. {EZ[topo].max():.3f} meV (phi = pi)")
    print(" E_Z    Delta_top  Delta_out  hbar v_A   xi_M   xi_hyp/xi_M  xi_poly/xi_M   k_osc    k0       k_gap"
          "     2nd: xi  Re k     3rd: xi  Re k")
    for j in range(0, len(EZ), 10):
        print(f" {EZ[j]:.2f}  {D_top[j] * 1e3:6.1f}     {D_out[j] * 1e3:6.1f}     {hv_A[j]:6.1f}   {xi_M[j]:6.0f}"
              f"   {xi_hyp[j] / xi_M[j]:.3f}        {xi_poly[j] / xi_M[j]:.3f}        {k_osc[j]:.4f}   {k0[j]:.4f}"
              f"   {k_gap[j]:.4f}   {xi_2[j]:6.0f}  {k_2[j]:.4f}   {xi_3[j]:6.0f}  {k_3[j]:.4f}   Q {Q[j]:+.0f}")


# %% [oscillation] --- 2. CLOSED BOX: eps ---
if run_part('oscillation'):
    eps = st.run_scan(partial(box_point, QM), {'E_Z': EZ}, 'scharf_absOM_box_mixed_a10_L2',
                      n_workers=BOX_WORKERS)[:, 0]
    print(f"\nbox eps: {np.sum(np.isfinite(eps))}/{len(eps)} points, max inside Q = -1 {np.nanmax(eps[topo]):.2f} ueV")


# %% [oscillation] --- 3. TESTS ---
if run_part('oscillation'):
    L = QM.L
    env = D_out * np.exp(-L / xi_M) * 1e3                           # ueV, no prefactor
    z = zeros_of(EZ, np.where(topo, eps, np.nan))
    Phi = np.interp(z, EZ, k_osc) * L
    print(f"\nTEST 1: zeros of eps at {z}")
    print(f"        k_osc L advances by {np.round(np.diff(Phi) / np.pi, 2)} pi between them (model: 1)")
    m = topo & np.isfinite(eps)
    A, phi0, rms = fit_A(eps[m], env[m], k_osc[m], L)
    print(f"FIT:    A = {A:.2f}, phi0 = {phi0 / np.pi:.2f} pi, rms {rms:.2f} ueV = {rms / np.nanmax(eps[m]):.1%} of max eps")
    print("TEST 2: lobe            E_Z    eps_max [ueV]   env [ueV]   A = eps_max/env   L/xi_M")
    for lo, hi in zip(z[:-1], z[1:]):
        w = (EZ > lo) & (EZ < hi) & np.isfinite(eps)
        j = np.flatnonzero(w)[np.argmax(eps[w])]
        print(f"        {lo:.3f}-{hi:.3f}   {EZ[j]:.3f}   {eps[j]:8.3f}        {env[j]:8.3f}     {eps[j] / env[j]:.2f}"
              f"              {L / xi_M[j]:.1f}")


# %% [oscillation] --- 4. TRANSPORT: Gamma_eff = eps_box / x(det r) on the phi = pi line ---
if run_part('oscillation'):
    TL = {t: st.transport_line(junction('mixed', 0.7, t), EZ_T, f'scharf_absOM_line_mixed_tau{t:g}') for t in TAUS}
    iT = np.searchsorted(EZ, EZ_T)
    eT, topoT = eps[iT], topo[iT]
    GE = {}
    print("\n  tau      two-Majorana points   eps_box range [ueV]   Gamma_eff = eps/x: median, 10-90 % [ueV]")
    for t, tl in TL.items():
        k, G, d = tl['kappa'], tl['G'], np.clip(tl['det'], -1 + 1e-12, 1 - 1e-12)
        ok = topoT & (np.abs(G) < 0.1 * np.maximum(k, 1e-12)) & (np.abs(k - (1 - d**2) / 2) < 0.01) & (k > 0.02)
        x = np.sqrt((1 + d) / (1 - d))
        GE[t] = np.where(ok, eT / x, np.nan)
        print(f"  {t:.4f}   {ok.sum():3d} / {topoT.sum()}             {np.nanmin(eT[ok]):.3g} - {np.nanmax(eT[ok]):.3g}"
              f"          {np.nanmedian(GE[t]):.2f}, {np.nanpercentile(GE[t], 10):.2f}-{np.nanpercentile(GE[t], 90):.2f}")
    both = np.isfinite(GE[TAUS[0]]) & np.isfinite(GE[TAUS[1]])
    pr = np.log(GE[TAUS[1]][both] / GE[TAUS[0]][both]) / np.log(TAUS[1] / TAUS[0])
    print(f"  Gamma_eff(0.5) / Gamma_eff(0.3387) on {both.sum()} common points -> p = {np.median(pr):.2f}"
          f" (10-90 %: {np.percentile(pr, 10):.2f}-{np.percentile(pr, 90):.2f}); study_tuning: 2.60-2.71")
    # direct Gamma at the two deepest eps zeros inside Q = -1 (isolated Majoranas: eps << Gamma)
    zin = [e for e in z if 0.25 < e < 1.1][:2]
    CASES = tuple((t, float(EZ[np.argmin(np.where(np.abs(EZ - e) < 0.02, eps, np.inf))])) for t in TAUS for e in zin)
    GD = st.run_scan(partial(gamma_point, cases=CASES), {'k': np.arange(len(CASES))}, 'scharf_absOM_gamma_mixed',
                     n_workers=len(CASES))
    print("\n  direct Gamma (local Andreev Lorentzian):  tau     E_Z     eps_box [ueV]  Gamma [ueV]   Gamma_eff nearby (+-0.1 meV)")
    for (t, e), (g, a0) in zip(CASES, GD):
        near = np.abs(EZ_T - e) < 0.1
        print(f"                                           {t:.4f}  {e:.3f}   {eps[np.argmin(np.abs(EZ - e))]:.4f}"
              f"         {g:6.2f}        {np.nanmedian(GE[t][near]):6.2f}  (andreev(0)/2 = {a0 / 2:.3f})")


# %% [oscillation] --- FIGURES ---
if run_part('oscillation'):
    tz = lambda v: np.where(topo, v, np.nan)
    fig, axs = plt.subplots(4, 1, figsize=(10, 13), sharex=True, layout='constrained')
    axs[0].plot(EZ, eps, 'o', ms=2.5, label=r'$\varepsilon$ (closed box)')
    axs[0].plot(EZ, tz(A * env * np.abs(np.cos(k_osc * L + phi0))), '-', lw=0.9,
                label=rf'${A:.2f}\,\Delta_{{out}} e^{{-L/\xi_M}} |\cos(k_{{osc}}L + {phi0 / np.pi:.2f}\pi)|$')
    axs[0].plot(EZ, tz(A * env), '--', lw=0.8, color='gray', label='envelope')
    for x in z:
        axs[0].axvline(x, color='k', lw=0.5, alpha=0.3)
    axs[0].set_ylabel(r'$\varepsilon$ [$\mu$eV]'); axs[0].legend(fontsize=8)
    axs[1].plot(EZ, tz(xi_M), label=r'$\xi_M$ (slowest complex-$k$ mode)')
    axs[1].plot(EZ, tz(xi_2), ':', label='2nd mode')
    axs[1].plot(EZ, tz(xi_3), ':', label='3rd mode')
    axs[1].plot(EZ, tz(xi_hyp), '--', label=r'$\hbar v_A/\Delta_{out}$ (hyperbola at the dip)')
    axs[1].plot(EZ, tz(xi_poly), 'k.', ms=2, label='dip continued to complex $k$')
    axs[1].axhline(L, color='0.5', lw=0.6, label='L')
    axs[1].set_yscale('log'); axs[1].set_ylabel('length [nm]'); axs[1].legend(fontsize=8)
    axs[2].plot(EZ, tz(D_out * 1e3), '-', label=r'$\Delta_{out}$ [$\mu$eV]')
    axs[2].plot(EZ, tz(D_top * 1e3), ':', label=r'$\Delta_{top}$ [$\mu$eV]')
    axs[2].plot(EZ, tz(hv_A), '--', label=r'$\hbar v_A = \sqrt{\Delta_{out} E^{\prime\prime}}$ [meV nm]')
    axs[2].legend(fontsize=8)
    axs[3].plot(EZ, tz(k_osc), label=r'$k_{osc}$ (slowest mode)')
    axs[3].plot(EZ, tz(k_2), ':', label=r'Re $k$, 2nd mode')
    axs[3].plot(EZ, tz(k_3), ':', label=r'Re $k$, 3rd mode')
    axs[3].plot(EZ, tz(k0), 'k--', lw=0.8, label=r'$k_0$ of the outer dip')
    axs[3].plot(EZ, tz(k_gap), '.', ms=2, label=r'$k$ of the global gap minimum')
    axs[3].set_ylabel(r'$k$ [1/nm]'); axs[3].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[3].legend(fontsize=8)
    fig.suptitle(r'mixed SOC (16 meV nm, $\theta$ 0.15$\pi$), a 10, W 100, L 2 $\mu$m, $t_c$ 1, $\mu_C$ 0.7', fontsize=10)
    st.save_fig(fig, 'oscillation_mixed'); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout='constrained')
    axs[0].semilogy(EZ, tz(eps), 'k-', lw=0.8, label=r'$\varepsilon$ (closed box)')
    for t, c in zip(TAUS, ('C0', 'C1')):
        axs[0].semilogy(EZ_T, GE[t], 'o', ms=3, color=c, label=rf'$\Gamma_{{eff}} = \varepsilon/x$ (det r), tc_barr {t:g}')
        for (tt, e), (g, _) in zip(CASES, GD):
            if tt == t:
                axs[0].plot(e, g, '*', ms=12, color=c, mec='k')
    axs[0].set_ylabel(r'[$\mu$eV]  (stars: $\Gamma$ from the local Andreev Lorentzian)'); axs[0].legend(fontsize=8)
    for t, c in zip(TAUS, ('C0', 'C1')):
        axs[1].plot(EZ_T, TL[t]['kappa'], '-', color=c, lw=1, label=rf'$\kappa$ (NEGF), tc_barr {t:g}')
        g = np.mean([gg for (tt, _), (gg, _) in zip(CASES, GD) if tt == t])
        xx = eps / g
        axs[1].plot(EZ, tz(2 * xx**2 / (1 + xx**2) ** 2), '--', color=c, lw=0.8,
                    label=rf'$2x^2/(1+x^2)^2$, $x = \varepsilon_{{box}}/\Gamma$, $\Gamma$ = {g:.1f} $\mu$eV (Lorentzian)')
    axs[1].axhline(0.5, color='k', lw=0.4, ls=':')
    axs[1].set_ylabel(r'$\kappa/\kappa_0$'); axs[1].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[1].legend(fontsize=8)
    fig.suptitle('mixed SOC, a 10, L 2 um: does the transport see the box splitting?', fontsize=10)
    st.save_fig(fig, 'oscillation_mixed_transport'); plt.close(fig)


# ====================================================================================================================
# PART 'plateau'  (was study_plateau_mixed.py)
# ====================================================================================================================
# STUDY: the best kappa = 1/2 plateau of the MIXED SOC, as a line (2026-11-04).
# Mixed SOC = study_tuning.junction('mixed', mu_C, tau, L): W 100, t_c 1, a 10, Rashba + Dresselhaus 16 meV nm at
# theta 0.15 pi ('100'); only mu_C, the probe bond tau (tc_barr_transport) and L are tuned.
# On the coarse tuning maps (E_Z step 0.05, 31 phases) a plateau run along E_Z at fixed phi is at most 3 points
# (0.15 meV), but along phi at fixed E_Z it reaches ~1 pi.  One phase line costs ONE RGF set-up (~30 s at a = 10;
# the phases themselves are nearly free), so the search runs along phi:
#   1. fine local grids around the four best coarse runs: E_Z centre +- 0.10 (step 0.01) x 241 phases
#   2. Q (Pfaffians of the infinite cross-section) on the phase lines of the longest runs; re-rank inside Q = -1
#   3. the winner's tau (x0.8, x1.25) and L (+-0.5 um) neighbours, same check
#   4. the best line refined in E_Z (step 0.0025), Q on it, and the cut along E_Z through the plateau centre
# Plateau point: |kappa - 1/2| < 0.03, |G| < 0.1 kappa, Q = -1, and the two-Majorana relation
# kappa = (1 - det r^2)/2 within 0.01 (the point is a two-Majorana point, not just kappa ~ 1/2).
# kappa, G at E = 0 (T -> 0).

# %% [plateau] --- PARAMETERS ---

PHI_F = np.linspace(0.0, 2 * np.pi, 241)                 # step pi/120; the last point repeats the first
DPHI = 2.0 / (len(PHI_F) - 1)                             # one phase step in units of pi
CAND = [(0.3, 0.3387, 2000.0, 1.25), (0.7, 0.5, 2000.0, 0.95), (0.7, 0.7, 1500.0, 1.00), (0.7, 0.5, 1500.0, 0.80)]
N_Q = 6                                                   # lines per stage that get Q on every phase
TOL_K, TOL_G, TOL_2M = 0.03, 0.1, 0.01
# colours: the first three categorical slots of the dataviz reference palette (validated all-pairs)
C_K, C_G, C_D, INK, INK2, TRIV = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'


def tag(mu, t, L):
    return f'mu{mu:g}_tau{t:g}_L{L / 1000:g}'


def fine_grid(mu, t, L, ez, **kw):
    """transport_phis on ez x PHI_F -> (len(ez), len(PHI_F), 5): [kappa, G, andreev, det r, n_bad]; kw to run_scan."""
    return st.run_scan(partial(st.transport_phis, junction('mixed', mu, t, L), phis=PHI_F), {'E_Z': np.asarray(ez)},
                       f'scharf_absPlat_tr_{tag(mu, t, L)}_EZ{ez[0]:g}-{ez[-1]:g}', **kw)


def q_line(mu, E_Z):
    """Q on every phase of PHI_F at one field (the infinite cross-section: independent of tau and L)."""
    return st.run_scan(partial(st.q_point, junction('mixed', mu)), {'E_Z': [float(E_Z)], 'phi': PHI_F},
                       f'scharf_absPlat_Q_mu{mu:g}_EZ{E_Z:g}')[0, :, 0]


def plateau_ok(tr, Q=None):
    k, G, d = tr[..., 0], tr[..., 1], tr[..., 3]
    m = (np.abs(k - 0.5) < TOL_K) & (np.abs(G) < TOL_G * k) & (np.abs(k - (1 - d**2) / 2) < TOL_2M)
    return m if Q is None else m & (Q < 0)


def periodic_run(m):
    """Longest run of True on the periodic grid PHI_F (last point = first): (n points, start index)."""
    m = np.asarray(m, bool)[:-1]
    if m.all():
        return len(m), 0
    best = (0, 0)
    for s in np.flatnonzero(m & ~np.roll(m, 1)):
        best = max(best, (int(np.argmin(np.roll(m, -s))), int(s)))
    return best


def straight_run(m, i):
    """The run of True containing index i on a non-periodic grid: (lo, hi) inclusive, or None."""
    if not m[i]:
        return None
    lo, hi = i, i
    while lo > 0 and m[lo - 1]:
        lo -= 1
    while hi < len(m) - 1 and m[hi + 1]:
        hi += 1
    return lo, hi


def rank_lines(grids):
    """grids {(mu, t, L): (ez, tr)} -> [(run in pi, mu, t, L, E_Z, row)] by the transport-only run, longest first."""
    out = [(periodic_run(plateau_ok(tr[i]))[0] * DPHI, mu, t, L, float(e), i)
           for (mu, t, L), (ez, tr) in grids.items() for i, e in enumerate(ez)]
    return sorted(out, key=lambda r: -r[0])


def q_check(lines, grids, label):
    """Q on the N_Q longest lines; returns them re-ranked inside Q = -1: (run, mu, t, L, E_Z, row, start)."""
    print(f"\n  {label}: the {N_Q} longest transport runs, then inside Q = -1", flush=True)
    out = []
    for run0, mu, t, L, e, i in lines[:N_Q]:
        Q = q_line(mu, e)
        n, s = periodic_run(plateau_ok(grids[(mu, t, L)][1][i], Q))
        out.append((n * DPHI, mu, t, L, e, i, s))
        print(f"    {tag(mu, t, L):24s} E_Z {e:.4f}: run {run0:.3f} pi -> {n * DPHI:.3f} pi with Q = -1 "
              f"(Q = +1 on {np.mean(Q[:-1] > 0) * 100:.0f}% of the phases)", flush=True)
    return sorted(out, key=lambda r: -r[0])


def spans(x, m):
    """Contiguous True regions of m on the grid x as (x_lo, x_hi), edges at the midpoints."""
    out, i, n = [], 0, len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and m[j + 1]:
            j += 1
        out.append(((x[i - 1] + x[i]) / 2 if i > 0 else x[0], (x[j] + x[j + 1]) / 2 if j < n - 1 else x[-1]))
        i = j + 1
    return out


# %% [plateau] --- 1-2. FINE LOCAL GRIDS AROUND THE COARSE CANDIDATES, Q CHECK ---
if run_part('plateau'):
    GRIDS = {}
    for mu, t, L, e0 in CAND:
        ez = np.round(np.arange(e0 - 0.10, e0 + 0.1001, 0.01), 3)
        GRIDS[(mu, t, L)] = (ez, fine_grid(mu, t, L, ez))
        print(f"fine grid {tag(mu, t, L)} done", flush=True)
    BEST = q_check(rank_lines(GRIDS), GRIDS, "stage 1 (coarse candidates)")

# %% [plateau] --- 3. THE WINNER'S tau AND L NEIGHBOURS ---
if run_part('plateau'):
    _, mu_w, t_w, L_w, e_w, _, _ = BEST[0]
    NB = {}
    for t in (round(0.8 * t_w, 4), t_w, round(1.25 * t_w, 4)):
        for L in (L_w - 500.0, L_w, L_w + 500.0):
            if (t, L) == (t_w, L_w):
                continue
            ez = np.round(np.arange(e_w - 0.10, e_w + 0.1001, 0.01), 3)
            NB[(mu_w, t, L)] = (ez, fine_grid(mu_w, t, L, ez))
            print(f"neighbour {tag(mu_w, t, L)} done", flush=True)
    BEST = sorted(BEST + q_check(rank_lines(NB), NB, "stage 2 (tau, L neighbours)"), key=lambda r: -r[0])
    GRIDS.update(NB)

# %% [plateau] --- 4. THE BEST LINE: refined in E_Z, Q, and the cut along E_Z through the plateau centre ---
if run_part('plateau'):
    _, mu_b, t_b, L_b, e_b, _, _ = BEST[0]
    ez_r = np.round(e_b + np.arange(-0.0075, 0.00751, 0.0025), 4)
    REF = {(mu_b, t_b, L_b): (ez_r, fine_grid(mu_b, t_b, L_b, ez_r))}
    FIN = q_check(rank_lines(REF), REF, "stage 3 (E_Z refined)")
    FIN = sorted(FIN + [r for r in BEST if r[1:4] == (mu_b, t_b, L_b) and abs(r[4] - e_b) < 1e-9],
                 key=lambda r: -r[0])
    run, mu_b, t_b, L_b, e_b, _, s_b = FIN[0]
    q_b = junction('mixed', mu_b, t_b, L_b)
    tr_b = fine_grid(mu_b, t_b, L_b, np.array([e_b]))[0]           # the winning line on its own (one solve)
    Q_b = q_line(mu_b, e_b)
    n_b = int(round(run / DPHI))
    idx = (s_b + np.arange(n_b)) % (len(PHI_F) - 1)
    phi_c = PHI_F[(s_b + n_b // 2) % (len(PHI_F) - 1)]
    k_b, G_b, d_b = tr_b[:, 0], tr_b[:, 1], tr_b[:, 3]
    print(f"\n  BEST LINE: {tag(mu_b, t_b, L_b)}, E_Z {e_b:.4f} meV: plateau over {run:.3f} pi of phase, "
          f"phi {PHI_F[s_b] / np.pi:.3f} pi .. {PHI_F[idx[-1]] / np.pi:.3f} pi (periodic), centre {phi_c / np.pi:.3f} pi")
    print(f"    on the run: kappa {k_b[idx].min():.4f} .. {k_b[idx].max():.4f} (mean {k_b[idx].mean():.4f}), "
          f"|G| <= {np.abs(G_b[idx]).max():.4f}, det r {d_b[idx].min():+.3f} .. {d_b[idx].max():+.3f}, "
          f"2M residual <= {np.abs(k_b[idx] - (1 - d_b[idx]**2) / 2).max():.4f}")

    ez_c = np.round(np.arange(max(e_b - 0.40, 0.0), e_b + 0.40001, 0.005), 4)
    cut = st.run_scan(partial(st.transport_point, q_b, phi=float(phi_c)), {'E_Z': ez_c},
                      f'scharf_absPlat_cut_{tag(mu_b, t_b, L_b)}_phi{phi_c / np.pi:.4f}pi')
    Q_c = st.run_scan(partial(st.q_point, junction('mixed', mu_b), phi=float(phi_c)), {'E_Z': ez_c},
                      f'scharf_absPlat_Qcut_mu{mu_b:g}_phi{phi_c / np.pi:.4f}pi')[:, 0]
    m_c = plateau_ok(cut, Q_c)
    j_b = int(np.argmin(np.abs(ez_c - e_b)))
    rr = straight_run(m_c, j_b)
    width = (ez_c[rr[1]] - ez_c[rr[0]] + 0.005) if rr else 0.0
    print(f"    cut along E_Z at phi {phi_c / np.pi:.3f} pi: plateau run through E_Z {e_b:.4f} is "
          + (f"{ez_c[rr[0]]:.3f} .. {ez_c[rr[1]]:.3f} meV ({width * 1e3:.0f} ueV wide)" if rr else "absent at this point")
          + f"; all plateau points on the cut: {m_c.sum()} of {len(ez_c)}; Q = -1 for "
          + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in spans(ez_c, Q_c < 0)) + " meV")

# %% [plateau] --- FIGURE: kappa, |G|, det r along phi (the plateau) and along E_Z (the cross-cut) ---
def plot_best(mu, t, L, e, phi_c, run, tr, Q, ez_c, cut, Q_c):
    """Left: the phase line at E_Z = e, window centred on the plateau (phi_c +- pi, so the run is one piece).
    Right: the cut along E_Z at phi_c.  Saves plots/plateau_mixed_best.png."""
    lo = phi_c / np.pi - 1.0
    x0 = PHI_F[:-1] / np.pi
    o = np.argsort((x0 - lo) % 2)
    xw = np.r_[((x0 - lo) % 2 + lo)[o], ((x0 - lo) % 2 + lo)[o][0] + 2]       # closed periodic window
    rows = np.r_[o, o[0]]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex='col', gridspec_kw=dict(height_ratios=(1.6, 1)))
    for col, (x, k, G, d, QQ, xlab) in enumerate([
            (xw, tr[rows, 0], tr[rows, 1], tr[rows, 3], Q[rows], r'$\phi/\pi$  (mod 2)'),
            (ez_c, cut[:, 0], cut[:, 1], cut[:, 3], Q_c, r'$E_Z$ (meV)')]):
        a1, a2 = axs[0, col], axs[1, col]
        for a in (a1, a2):
            for s0, s1 in spans(x, QQ > 0):
                a.axvspan(s0, s1, color=TRIV, lw=0, zorder=0)
            a.grid(color='#d8d7d2', lw=0.5, alpha=0.6)
            a.tick_params(colors=INK2, labelcolor=INK)
            for sp_ in a.spines.values():
                sp_.set_color('#b5b4ae')
        a1.axhspan(0.5 - TOL_K, 0.5 + TOL_K, color=C_K, alpha=0.10, lw=0, zorder=1)
        a1.axhline(0.5, color=INK2, lw=0.8, ls='--', zorder=1)
        a1.plot(x, k, color=C_K, lw=2, zorder=3)
        a1.plot(x, np.abs(G), color=C_G, lw=2, zorder=3)
        a2.axhline(0.0, color=INK2, lw=0.8, ls='--')
        a2.plot(x, d, color=C_D, lw=2)
        a1.set_ylim(-0.02, 1.0)
        a2.set_ylim(-1.05, 1.05)
        a2.set_xlabel(xlab, color=INK)
        a1.set_ylabel('transmission', color=INK)
        a2.set_ylabel(r'$\det r$', color=INK)
        m = plateau_ok(np.c_[k, G, np.zeros_like(k), d], QQ)
        a1.plot(x[m], np.full(m.sum(), 0.93), ls='none', marker='s', ms=2.2, color=C_K, zorder=3)
        a1.text(x[0], 0.955, 'plateau points', color=INK2, fontsize=8, va='bottom')
    a = axs[1, 0]
    a.set_xlim(xw[0], xw[-1])
    tk = np.arange(np.ceil(xw[0] * 4) / 4, xw[-1] + 1e-9, 0.25)
    a.set_xticks(tk)
    a.set_xticklabels([f'{v % 2:g}' for v in tk])
    axs[0, 0].axvline(phi_c / np.pi, color=INK2, lw=0.8, ls=':')
    axs[0, 1].axvline(e, color=INK2, lw=0.8, ls=':')
    axs[0, 0].set_title(rf'along $\phi$ at $E_Z$ = {e:.3f} meV: plateau over {run:.2f}$\pi$ of phase', color=INK,
                        fontsize=10)
    axs[0, 1].set_title(rf'along $E_Z$ at the plateau centre $\phi$ = {phi_c / np.pi % 2:.2f}$\pi$ (dotted: left line)',
                        color=INK, fontsize=10)
    handles = [plt.Line2D([], [], color=C_K, lw=2), plt.Line2D([], [], color=C_G, lw=2),
               plt.Line2D([], [], color=C_D, lw=2), plt.Rectangle((0, 0), 1, 1, color=TRIV),
               plt.Rectangle((0, 0), 1, 1, color=C_K, alpha=0.15)]
    fig.legend(handles, [r'$\kappa/\kappa_0$', r'$|G|/G_0$', r'$\det r$', 'Q = +1 (bulk trivial)',
                         r'$\kappa = 1/2 \pm 0.03$'], loc='upper center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(rf'Best $\kappa = 1/2$ plateau, mixed SOC: $\mu_C$ = {mu:g} meV, $\tau$ = {t:g}, L = {L / 1000:g} $\mu$m, '
                 r'a = 10 nm, $E$ = 0 ($T \to 0$)', color=INK, fontsize=11, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.save_fig(fig, 'plateau_mixed_best')
    print("figure: plots/plateau_mixed_best.png", flush=True)


if run_part('plateau'):
    plot_best(mu_b, t_b, L_b, e_b, phi_c, run, tr_b, Q_b, ez_c, cut, Q_c)


# %% [plateau] --- 5. L = 3 um, tau 0.5 near phi = pi: the coarse candidate of detr_crossings_mixed (2026-11-10) ---
# The coarse phi = pi column (study_tuning [tuning] cell 8, step 0.05) has three neutral points, kappa 0.473 / 0.499 /
# 0.472 at E_Z 0.85 / 0.90 / 0.95, det r -0.23 -> +0.02 -> +0.23.  L 3 was not in stages 1-3 (L 2 +- 0.5 only).
# Coarse runs have shrunk on refinement before (<= 0.15 meV -> 50 ueV), and L 3 has the densest eps oscillation, so
# a zero of eps (kappa -> 0) can hide between coarse points.  (a) the cut along E_Z at phi = pi, step 0.0025, with Q;
# (b) phase lines on 0.80-1.00 (step 0.01, 241 phases) ranked and Q-checked as in stages 1-2.  Compared with the L 2
# stage-1 grid of the same mu_C, tau.  Figure plots/plateau_mixed_L3.png.
L3C = (0.7, 0.5, 3000.0)
EZ_L3 = np.round(np.arange(0.75, 1.05001, 0.0025), 4)
EZ_L3G = np.round(np.arange(0.80, 1.00001, 0.01), 3)
if run_part('plateau'):
    cut_3 = st.run_scan(partial(st.transport_point, junction('mixed', *L3C), phi=np.pi), {'E_Z': EZ_L3},
                        f'scharf_absPlat_cut_{tag(*L3C)}_phi1.0000pi', chunksize=1)
    Qc_3 = st.run_scan(partial(st.q_point, junction('mixed', L3C[0]), phi=np.pi), {'E_Z': EZ_L3},
                       f'scharf_absPlat_Qcut_mu{L3C[0]:g}_phi1.0000pi', chunksize=1)[:, 0]
    m_3 = plateau_ok(cut_3, Qc_3)
    k_3 = cut_3[:, 0]
    in_c = (EZ_L3 >= 0.85) & (EZ_L3 <= 0.95)
    print(f"\n  L 3 um, tau 0.5, cut along E_Z at phi = pi (step 0.0025):")
    print("    plateau (|kappa-1/2| < 0.03, neutral, 2M, Q = -1): "
          + (", ".join(f"{a:.4f}-{b:.4f} ({(b - a) * 1e3:.0f} ueV)" for a, b in spans(EZ_L3, m_3)) or "none"))
    print("    kappa >= 0.4: " + ", ".join(f"{a:.4f}-{b:.4f} ({(b - a) * 1e3:.0f} ueV)" for a, b in spans(EZ_L3, k_3 >= 0.4)))
    print(f"    on 0.85-0.95: kappa {k_3[in_c].min():.3f} .. {k_3[in_c].max():.3f}, |G| <= {np.abs(cut_3[in_c, 1]).max():.3f}, "
          f"Q = -1 on {np.mean(Qc_3 < 0):.0%} of the cut", flush=True)

    G_3 = {L3C: (EZ_L3G, fine_grid(*L3C, EZ_L3G, chunksize=1))}
    B_3 = q_check(rank_lines(G_3), G_3, "L 3 um candidate (phase lines)")
    print(f"    best L 3 line: E_Z {B_3[0][4]:.3f}, {B_3[0][0]:.3f} pi with Q = -1  (L 2 winner: 1.37 pi at 0.930)", flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex='col', gridspec_kw=dict(height_ratios=(1.6, 1)))
    ez2, tr2 = GRIDS[(0.7, 0.5, 2000.0)]
    jpi = len(PHI_F) // 2                                                  # PHI_F[120] = pi
    for a in axs.ravel():
        a.grid(color='#d8d7d2', lw=0.5, alpha=0.6)
        a.tick_params(colors=INK2, labelcolor=INK)
        for sp_ in a.spines.values():
            sp_.set_color('#b5b4ae')
    a1, a2 = axs[:, 0]
    a1.axhspan(0.5 - TOL_K, 0.5 + TOL_K, color=C_K, alpha=0.10, lw=0)
    a1.axhline(0.5, color=INK2, lw=0.8, ls='--')
    a1.plot(ez2, tr2[:, jpi, 0], color=TRIV, lw=4, solid_capstyle='round', zorder=2)
    a1.plot(EZ_L3, k_3, color=C_K, lw=2, zorder=3)
    a1.plot(EZ_L3, np.abs(cut_3[:, 1]), color=C_G, lw=2, zorder=3)
    a1.plot(EZ_L3[m_3], np.full(m_3.sum(), 0.93), ls='none', marker='s', ms=2.2, color=C_K)
    a1.text(EZ_L3[0], 0.955, 'plateau points (L 3)', color=INK2, fontsize=8, va='bottom')
    a1.set_ylim(-0.02, 1.0); a1.set_ylabel('transmission', color=INK)
    a1.set_title(r'along $E_Z$ at $\phi = \pi$', color=INK, fontsize=10)
    a2.axhline(0.0, color=INK2, lw=0.8, ls='--')
    a2.plot(ez2, tr2[:, jpi, 3], color=TRIV, lw=4, solid_capstyle='round')
    a2.plot(EZ_L3, cut_3[:, 3], color=C_D, lw=2)
    a2.set_ylim(-1.05, 1.05); a2.set_ylabel(r'$\det r$', color=INK); a2.set_xlabel(r'$E_Z$ (meV)', color=INK)
    b1, b2 = axs[:, 1]
    for (ez_, tr_), col, lw, lab in (((ez2, tr2), TRIV, 4, r'L 2 $\mu$m'), (G_3[L3C], C_K, 2, r'L 3 $\mu$m')):
        b1.plot(ez_, [periodic_run(plateau_ok(r))[0] * DPHI for r in tr_], color=col, lw=lw, label=lab,
                solid_capstyle='round')
    b1.plot([r[4] for r in B_3], [r[0] for r in B_3], ls='none', marker='o', ms=5, color=C_K, mec='w',
            label=r'L 3, with Q = $-1$')
    b1.plot([0.930], [1.37], ls='none', marker='*', ms=10, color=INK, label=r'best (L 2, $E_Z$ 0.930)')
    b1.set_ylabel(r'longest plateau run along $\phi$ [$\pi$]', color=INK)
    b1.set_title(r'plateau extent in phase, per field', color=INK, fontsize=10)
    b1.legend(fontsize=8, frameon=False)
    for (ez_, tr_), col, lw in (((ez2, tr2), TRIV, 4), (G_3[L3C], C_K, 2)):
        b2.plot(ez_, [np.mean(plateau_ok(r)[:-1]) for r in tr_], color=col, lw=lw, solid_capstyle='round')
    b2.set_ylabel('plateau share of phases', color=INK); b2.set_xlabel(r'$E_Z$ (meV)', color=INK)
    fig.legend([plt.Line2D([], [], color=C_K, lw=2), plt.Line2D([], [], color=C_G, lw=2),
                plt.Line2D([], [], color=C_D, lw=2), plt.Line2D([], [], color=TRIV, lw=4)],
               [r'$\kappa/\kappa_0$, L 3 $\mu$m', r'$|G|/G_0$, L 3', r'$\det r$, L 3', r'L 2 $\mu$m (step 0.01)'],
               loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(r'Mixed SOC, $\mu_C$ = 0.7 meV, $\tau$ = 0.5, a = 10 nm: the L = 3 $\mu$m candidate vs L = 2 $\mu$m',
                 color=INK, fontsize=11, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.save_fig(fig, 'plateau_mixed_L3')
    plt.close(fig)
    print("figure: plots/plateau_mixed_L3.png", flush=True)


# %% [plateau] --- 6. THE BEST POINT ALONG THE GATE (mu_C) vs ALONG THE FIELD (2026-11-10) ---
# Claim to test: mu_C moves k_osc about as fast as E_Z (dk/dE_Z : dk/dmu_C = E_Z / sqrt(E_so^2 + E_Z^2) ~ 0.8), so
# the kappa = 1/2 window along the gate is about as narrow as along the field.  Replaces the old mu scan (pre-bond-
# fix, one mu for probe leads AND strip): only the strip mu_C moves here, the probe leads keep mu_N.  Same point as
# the stage-4 E_Z cut: E_Z 0.930, phi = PHI_F[209] = 1.7417 pi, tau 0.5, L 2 um.  Figure plots/plateau_mixed_muC.png.
MU_CUT = np.round(np.arange(0.60, 0.80001, 0.0025), 4)
MU_Q = np.round(np.arange(0.60, 0.80001, 0.01), 3)
PH_W, EZ_W = PHI_F[209], 0.930
# The widest run ALONG E_Z on the stage-1 L 2 grid (0.01 step) is not at the winner's phase: 9 points (0.92-1.00) at
# phi 1.408-1.433 pi (and 0.083-0.108 pi), a lobe top with det r up to +0.19 (eps_max/Gamma ~ 1.2).  Refined here.
PH_E = PHI_F[171]                                                          # 1.425 pi
EZ_E = np.round(np.arange(0.85, 1.05001, 0.0025), 4)
EZ_EQ = np.round(np.arange(0.85, 1.05001, 0.01), 3)


def mu_point(mu_C):
    """transport_point of the stage-4 winner with the strip chemical potential mu_C (meV)."""
    return st.transport_point(junction('mixed', float(mu_C), 0.5, 2000.0), EZ_W, PH_W)


def mu_q(mu_C):
    """Q of the cross-section at the winner's (E_Z, phi) with strip chemical potential mu_C."""
    return st.q_point(junction('mixed', float(mu_C)), EZ_W, PH_W)


if run_part('plateau'):
    cut_mu = st.run_scan(mu_point, {'mu_C': MU_CUT}, 'scharf_absPlat_mucut_tau0.5_L2_EZ0.93_phi1.7417pi', chunksize=1)
    Q_mu = st.run_scan(mu_q, {'mu_C': MU_Q}, 'scharf_absPlat_muQ_EZ0.93_phi1.7417pi', chunksize=1)[:, 0]
    Q_muc = Q_mu[np.abs(MU_CUT[:, None] - MU_Q[None]).argmin(1)]           # nearest coarse Q
    assert abs(phi_c - PH_W) < 1e-12 and abs(e_b - EZ_W) < 1e-9 and (mu_b, t_b, L_b) == (0.7, 0.5, 2000.0)
    ez_cut, cut_ez, Q_ez = ez_c, cut, Q_c                                   # the stage-4 cut along E_Z
    cut_e2 = st.run_scan(partial(st.transport_point, junction('mixed', 0.7, 0.5, 2000.0), phi=float(PH_E)),
                         {'E_Z': EZ_E}, 'scharf_absPlat_cut_mu0.7_tau0.5_L2_phi1.4250pi', chunksize=1)
    Q_e2 = st.run_scan(partial(st.q_point, junction('mixed', 0.7), phi=float(PH_E)), {'E_Z': EZ_EQ},
                       'scharf_absPlat_Qcut_mu0.7_phi1.4250pi', chunksize=1)[:, 0]
    Q_e2c = Q_e2[np.abs(EZ_E[:, None] - EZ_EQ[None]).argmin(1)]
    print("\n  L 2, tau 0.5: plateau windows through the point (|kappa-1/2| < 0.03, neutral, 2M, Q = -1)")
    for lab, x, c, QQ, x0 in (('E_Z at phi 1.742 pi (winner)', ez_cut, cut_ez, Q_ez, EZ_W),
                              ('mu_C at phi 1.742 pi, E_Z 0.93', MU_CUT, cut_mu, Q_muc, 0.7),
                              ('E_Z at phi 1.425 pi', EZ_E, cut_e2, Q_e2c, 0.96)):
        m = plateau_ok(c, QQ)
        sp = [s for s in spans(x, m) if s[0] - 1e-9 <= x0 <= s[1] + 1e-9]
        sk = [s for s in spans(x, c[:, 0] >= 0.4) if s[0] - 1e-9 <= x0 <= s[1] + 1e-9]
        print(f"    {lab:26s}: plateau " + (f"{sp[0][0]:.4f}-{sp[0][1]:.4f} ({(sp[0][1] - sp[0][0]) * 1e3:.0f} ueV)" if sp else "none")
              + "; kappa >= 0.4 " + (f"{sk[0][0]:.4f}-{sk[0][1]:.4f} ({(sk[0][1] - sk[0][0]) * 1e3:.0f} ueV)" if sk else "none")
              + f"; all plateau windows on the cut: {len(spans(x, m))}, kappa min..max {c[:, 0].min():.3f}..{c[:, 0].max():.3f}"
              + f"; Q = -1 on {np.mean(QQ < 0):.0%}", flush=True)

    fig, axs = plt.subplots(2, 3, figsize=(15.0, 6.0), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
    for col, (x, c, QQ, x0, xl) in enumerate(((ez_cut, cut_ez, Q_ez, EZ_W, r'$E_Z - 0.930$ (meV)'),
                                             (MU_CUT, cut_mu, Q_muc, 0.7, r'$\mu_C - 0.7$ (meV)'),
                                             (EZ_E, cut_e2, Q_e2c, 0.96, r'$E_Z - 0.96$ (meV)'))):
        a1, a2 = axs[:, col]
        for a in (a1, a2):
            for s0, s1 in spans(x - x0, QQ > 0):
                a.axvspan(s0, s1, color=TRIV, lw=0, zorder=0)
            a.grid(color='#d8d7d2', lw=0.5, alpha=0.6)
            a.tick_params(colors=INK2, labelcolor=INK)
            for sp_ in a.spines.values():
                sp_.set_color('#b5b4ae')
        a1.axhspan(0.5 - TOL_K, 0.5 + TOL_K, color=C_K, alpha=0.10, lw=0)
        a1.axhline(0.5, color=INK2, lw=0.8, ls='--')
        a1.axvline(0.0, color=INK2, lw=0.8, ls=':')
        a1.plot(x - x0, c[:, 0], color=C_K, lw=2)
        a1.plot(x - x0, np.abs(c[:, 1]), color=C_G, lw=2)
        a2.axhline(0.0, color=INK2, lw=0.8, ls='--')
        a2.plot(x - x0, c[:, 3], color=C_D, lw=2)
        a1.set_ylim(-0.02, 1.0); a2.set_ylim(-1.05, 1.05); a2.set_xlabel(xl, color=INK)
        a1.set_title([r'field, $\phi$ = 1.742$\pi$ (best phase run)', r'gate ($\mu_C$), $\phi$ = 1.742$\pi$, $E_Z$ 0.930',
                      r'field, $\phi$ = 1.425$\pi$ (widest field run)'][col], color=INK, fontsize=10)
    axs[0, 0].set_xlim(-0.10, 0.10)
    axs[0, 0].set_ylabel('transmission', color=INK); axs[1, 0].set_ylabel(r'$\det r$', color=INK)
    fig.legend([plt.Line2D([], [], color=C_K, lw=2), plt.Line2D([], [], color=C_G, lw=2),
                plt.Line2D([], [], color=C_D, lw=2), plt.Rectangle((0, 0), 1, 1, color=TRIV),
                plt.Rectangle((0, 0), 1, 1, color=C_K, alpha=0.15)],
               [r'$\kappa/\kappa_0$', r'$|G|/G_0$', r'$\det r$', 'Q = +1 (bulk trivial)', r'$\kappa = 1/2 \pm 0.03$'],
               loc='upper center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(r'Mixed SOC, $\mu_C$ 0.7, $\tau$ 0.5, L 2 $\mu$m, a 10: $\kappa = 1/2$ windows along the field and the '
                 r'gate (same 0.2 meV span)', color=INK, fontsize=11, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.save_fig(fig, 'plateau_mixed_muC')
    plt.close(fig)
    print("figure: plots/plateau_mixed_muC.png", flush=True)


# ====================================================================================================================
# PART 'amplitude'  (was study_amplitude_mixed.py)   renamed: EZ -> EZ_AMP
# ====================================================================================================================
# STUDY: what sets the AMPLITUDE of eps in the mixed SOC?  (2026-11-04)
# study_oscillation_mixed: the closed-box zeros follow k_osc, but the lobe heights do NOT follow
# A Delta_out e^{-L/xi_M}: per-lobe A drifts 1.4 -> 5.9 with E_Z.  Parameter-free test from the full model:
# the semi-infinite end modes of the SAME cross-section, built from ALL complex-k modes (slow and evanescent):
#     psi_L(n) = sum_{|lam|<1} c_j phi_j lam_j^{n+1},   psi_L(-1) = 0   (hard wall; c = null vector)
#     psi_R(n) = sum_{|lam|>1} d_j phi_j lam_j^{n-N},   psi_R(N) = 0
# on the slices n = 0 .. N-1 of the box (study_oscillation.box_point: walls at -1 and N).
#   eps_RR : Rayleigh-Ritz of the box H in span{psi_L, psi_R}       (no fit; valid also at L ~ xi_M)
#   eps_t  : |psi_R(N-1)^dag H1 psi_L(N)| / (|psi_L| |psi_R|)        (boundary term, semi-infinite norms)
#   env_M  : eps_t with psi_L(N) cut to the slowest +-k pair, |a_+| + |a_-| -> the TRUE envelope; its ratio to
#            Delta_out e^{-L/xi_M} is what the per-lobe A should be.
# If eps_RR = eps_box, the lobe heights are the end-mode structure, and env_M says what the single-envelope misses.

# %% [amplitude] --- PARAMETERS ---

QM = junction('mixed', 0.7)
EZ_OM = np.round(np.arange(0.10, 1.3501, 0.005), 3)                # the grid of study_oscillation_mixed
EZ_AMP = EZ_OM[::2]                                                     # every second field (~6 min per field)
# colours: dataviz reference palette, first three categorical slots
C1, C2, C3, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'


def end_modes(p, n_S):
    """Complex-k modes at E = 0 and the hard-wall end states. -> dict (see amp_point)."""
    H0, H1 = bloch_blocks(p, n_S)
    n = len(H0)
    I, Z = np.eye(n), np.zeros((n, n))
    lam, V = sla.eig(np.block([[Z, I], [-H1.conj().T, -H0]]), np.block([[I, Z], [Z, H1]]))
    phi = V[:n] / np.linalg.norm(V[:n], axis=0)
    dec = np.abs(lam) < 1                                           # includes lam = 0
    gro = ~dec                                                      # includes lam = inf
    out = dict(H0=H0, H1=H1, n=n, n_dec=int(dec.sum()), n_gro=int(gro.sum()))
    for side, m in (('L', dec), ('R', gro)):
        U, s, Vh = np.linalg.svd(phi[:, m])
        out[side] = (lam[m], phi[:, m], Vh[-1].conj(), s[-1] / s[-2])    # modes, null vector, s_min / s_2
    return out


def psi_L(em, ns):
    """psi_L on the slices ns (array) -> (len(ns), n)."""
    lam, ph, c, _ = em['L']
    return np.array([ph @ (c * lam ** (k + 1)) for k in ns])


def psi_R(em, ns, N):
    """psi_R on the slices ns of a box with N slices -> (len(ns), n); lam = inf contributes only at slice N."""
    lam, ph, d, _ = em['R']
    with np.errstate(divide='ignore', invalid='ignore'):
        rows = []
        for k in ns:
            w = np.where(np.isfinite(lam), lam ** float(k - N), 1.0 if k == N else 0.0)
            rows.append(ph @ (d * w))
    return np.array(rows)


def norm2(lam, ph, c, side):
    """Semi-infinite |psi|^2: sum_{m>=1} |sum_j c_j phi_j z_j^m|^2 with z = lam (L) or 1/lam (R), analytically."""
    with np.errstate(divide='ignore', invalid='ignore'):
        z = lam if side == 'L' else np.where(np.isfinite(lam), 1 / lam, 0.0)
    w = c[:, None].conj() * c[None, :] * (ph.conj().T @ ph)
    zz = z[:, None].conj() * z[None, :]
    return float(np.real(np.sum(w * zz / (1 - zz))))


def box_H(H0, H1, N):
    off = sp.diags(np.ones(N - 1), 1)
    H1s = sp.csr_matrix(H1)
    return (sp.kron(sp.eye(N), sp.csr_matrix(H0)) + sp.kron(off, H1s) + sp.kron(off.T, H1s.conj().T)).tocsr()


def amp_point(q, E_Z, phi=np.pi):
    """
    [eps_RR, eps_t, env_M (ueV), xi_M (nm), k_osc (1/nm), s_min/s_2 left, right, n_dec, n_gro, RR asymmetry,
     size of the slowest set, W_L, W_R]
    W_L: |slowest +-k pair part of psi_L|^2 / |psi_L|^2 (the modes are not orthogonal: W > 1 means the other modes
    cancel part of it, e.g. at E_Z 0.40 where xi_2 ~ xi_M); W_R the same for psi_R.
    """
    p = q.params(E_Z=float(E_Z), phi=phi)
    em = end_modes(p, st.n_S_of(q))
    H0, H1, n = em['H0'], em['H1'], em['n']
    N = q.n_along
    # Rayleigh-Ritz in the box
    G = np.c_[psi_L(em, np.arange(N)).ravel(), psi_R(em, np.arange(N), N).ravel()]
    S = G.conj().T @ G
    Hm = G.conj().T @ (box_H(H0, H1, N) @ G)
    E = sla.eigh((Hm + Hm.conj().T) / 2, S, eigvals_only=True)
    eps_RR = np.abs(E).mean()
    # boundary term with semi-infinite norms, any L
    lamL, phL, c, rL = em['L']
    lamR, phR, d, rR = em['R']
    nL, nR = norm2(lamL, phL, c, 'L'), norm2(lamR, phR, d, 'R')
    with np.errstate(divide='ignore', invalid='ignore'):
        zR = np.where(np.isfinite(lamR), 1 / lamR, 0.0)
    left_row = (phR @ (d * zR)).conj() @ H1                          # psi_R(N-1)^dag H1, independent of N
    a = c * (left_row @ phL)                                         # eps_t = |sum_j a_j lam_j^{N+1}| / norms
    nrm = np.sqrt(nL * nR)
    eps_t = lambda NN: np.abs(np.sum(a * lamL ** (NN + 1))) / nrm
    slow = np.abs(lamL) > np.abs(lamL).max() * (1 - 1e-6)          # the slowest +-k pair
    lm = lamL[slow][0]
    env_M = np.sum(np.abs(a[slow])) * np.abs(lm) ** (N + 1) / nrm
    xi = -q.a / np.log(np.abs(lm))
    W_L = norm2(lamL[slow], phL[:, slow], c[slow], 'L') / nL
    slowR = np.isfinite(lamR) & (np.abs(lamR) < np.abs(lamR[np.isfinite(lamR)]).min() * (1 + 1e-6))
    W_R = norm2(lamR[slowR], phR[:, slowR], d[slowR], 'R') / nR
    return np.array([eps_RR * q.t * 1e3, eps_t(N) * q.t * 1e3, env_M * q.t * 1e3, xi, abs(np.angle(lm)) / q.a,
                     rL, rR, em['n_dec'], em['n_gro'], abs(E.sum()) / max(abs(E).max(), 1e-300), slow.sum(),
                     W_L, W_R])


# %% [amplitude] --- RUN ---
if run_part('amplitude'):
    V = st.run_scan(partial(amp_point, QM), {'E_Z': EZ_AMP}, 'scharf_absAmp_mixed_a10_L2')
    eps_RR, eps_t, env_M, xi, kosc, rL, rR, nd, ng, asym, nslow, W_L, W_R = V.T
    OM = np.load(st.DATA / 'scharf_absOM_modes_mixed_a10.npz')['values'][::2]
    eps_box = np.load(st.DATA / 'scharf_absOM_box_mixed_a10_L2.npz')['values'][::2, 0]
    D_out, xi_M = OM[:, 10] * 1e3, OM[:, 0]
    L = QM.L
    LX = L / xi_M
    print(f"null-vector quality s_min/s_2: left <= {np.nanmax(rL):.1e}, right <= {np.nanmax(rR):.1e};"
          f" modes |lam|<1: {int(nd.min())}-{int(nd.max())}, >1: {int(ng.min())}-{int(ng.max())};"
          f" slowest set size {int(nslow.min())}-{int(nslow.max())}; RR +-E asymmetry <= {np.nanmax(asym):.1e}")
    print(f"xi_M here vs study_oscillation_mixed: max rel diff {np.nanmax(np.abs(xi / xi_M - 1)):.1e}")
    r = eps_RR / eps_box
    print("\n  E_Z    eps_box   eps_RR   eps_t   (ueV)   RR/box   t/box   L/xi_M   W_L    W_R")
    for j in range(0, len(EZ_AMP), 5):
        print(f"  {EZ_AMP[j]:.3f}  {eps_box[j]:8.4f} {eps_RR[j]:8.4f} {eps_t[j]:8.4f}         {r[j]:6.3f}  "
              f"{eps_t[j] / eps_box[j]:6.3f}   {LX[j]:5.2f}   {W_L[j]:.3f}  {W_R[j]:.3f}")
    off_zero = eps_box > 0.3 * np.maximum.accumulate(eps_box)       # away from the zeros (ratio ill-defined there)
    for lo, hi in ((3.0, 9.0), (2.0, 3.0), (1.5, 2.0), (1.0, 1.5), (0.0, 1.0)):
        m = off_zero & (LX >= lo) & (LX < hi)
        if m.any():
            print(f"  L/xi_M {lo:.1f}-{hi:.1f}: eps_RR/eps_box median {np.median(r[m]):.4f} "
                  f"(min {r[m].min():.4f}, max {r[m].max():.4f}) on {m.sum()} fields")
    A_true = env_M / (D_out * np.exp(-L / xi_M))
    print("\n  true envelope / (Delta_out e^{-L/xi_M}) = the A the lobes should show if two end modes suffice:")
    for e in (0.2, 0.3, 0.4, 0.46, 0.54, 0.64, 0.74, 0.87, 1.01, 1.15, 1.3):
        j = np.argmin(np.abs(EZ_AMP - e))
        print(f"    E_Z {EZ_AMP[j]:.2f}: A_true {A_true[j]:.2f}   env_M {env_M[j]:8.3f} ueV   L/xi_M {LX[j]:.2f}   "
              f"W_L {W_L[j]:.3f}  W_R {W_R[j]:.3f}")

# %% [amplitude] --- FIGURE ---
if run_part('amplitude'):
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True, gridspec_kw=dict(height_ratios=(1.8, 1, 1)))
    a = axs[0]
    a.semilogy(EZ_AMP, eps_box, 'o', ms=3.2, color=INK2, label=r'$\varepsilon$, closed box (exact)', zorder=2)
    a.semilogy(EZ_AMP, eps_RR, color=C1, lw=2, label=r'two end Majoranas, Rayleigh-Ritz (no fit)', zorder=3)
    a.semilogy(EZ_AMP, env_M, color=C2, lw=2, ls='--', label='its slowest-mode envelope')
    a.semilogy(EZ_AMP, D_out * np.exp(-L / xi_M), color=C3, lw=2, ls=':',
               label=r'$\Delta_{out}\,e^{-L/\xi_M}$ (single-envelope model, A = 1)')
    a.set_ylabel(r'$\varepsilon$ ($\mu$eV)')
    a.set_ylim(1e-4, 60)
    a.legend(frameon=False, fontsize=9, loc='lower right')
    a.set_title(r'Mixed SOC, a 10, L 2 $\mu$m, $\mu_C$ 0.7, $\phi = \pi$: what sets the amplitude of $\varepsilon$',
                fontsize=10)
    b = axs[1]
    b.axhline(1.0, color=INK2, lw=0.8, ls='--')
    b.plot(EZ_AMP[off_zero], r[off_zero], 'o', ms=3.2, color=C1, label=r'$\varepsilon_{RR}/\varepsilon_{box}$ (away from zeros)')
    b.set_ylabel('ratio')
    b.set_ylim(0.6, 1.1)
    ax2 = b.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax2.set_xticks(EZ_AMP[::25])
    ax2.set_xticklabels([f'{v:.1f}' for v in LX[::25]], fontsize=8)
    ax2.set_xlabel(r'$L/\xi_M$', fontsize=9)
    b.legend(frameon=False, fontsize=9, loc='lower left')
    c = axs[2]
    c.plot(EZ_AMP, A_true, color=C2, lw=2, label=r'$A$ from the end modes: envelope / ($\Delta_{out} e^{-L/\xi_M}$)')
    c.plot(EZ_AMP, W_L, color=C1, lw=2, label=r'$W_L$: slowest-mode norm / $|\gamma_L|^2$ (> 1: cancelled by other modes)')
    c.plot(EZ_AMP, W_R, color=C3, lw=2, ls='--', label=r'$W_R$: same for $\gamma_R$ (equal to $W_L$)')
    c.set_ylabel('dimensionless')
    c.set_xlabel(r'$E_Z$ (meV)')
    c.legend(frameon=False, fontsize=9, loc='upper left')
    for x in axs:
        x.grid(color='#d8d7d2', lw=0.5, alpha=0.6)
    fig.tight_layout()
    st.save_fig(fig, 'amplitude_mixed')
    print("figure: plots/amplitude_mixed.png", flush=True)


# ====================================================================================================================
# PART 'deck'  (2026-11-11)
# ====================================================================================================================
# STUDY: the advisor/professor deck in the MIXED SOC.  The slides_nov.py figures that existed only for pure Rashba
# (a 20) are redone for the thesis junction junction('mixed', 0.7) (a 10 unless stated); the figures are drawn in
# slides_nov.py (mixed_* functions), this part only computes.
#   1. kappa(E_Z) at phi = pi for L = 2 / 3 / 4 um, probe bond 0.3387 (per channel ~ tc_barr 0.2 at a 20, study_rashba
#      [a]).  Rashba (slides_nov.kappa_L): neutral kappa > 0.45 switches on where L/xi_M(E_Z) drops to ~3.8, for every
#      L.  Does the mixed SOC follow the same rule?  xi_M from [oscillation] (infinite junction, L-independent).
#   2. temperature, a = 20: probe bond 0.7 for the mixed case of study_rashba [T] (0.2 / 0.3 / 0.4 exist) -> the
#      too weak / optimum / too strong panels of slide 10 in the mixed SOC.
#   3. temperature, a = 10, at the best kappa = 1/2 region ([plateau]: tau 0.5, L 2 um, E_Z 0.930): the six channels on
#      the study_rashba [T] energy grid along E_Z at phi 1.742 pi (the phase-run winner), 1.425 pi (the field-run
#      line) and pi, and along phi at E_Z 0.930 -> kappa(T), G(T).  Does the region survive 10-20 mK?
#   4. S/N coupling (slide 9): Q and Delta_top on (mu_C, E_Z) and xi_M along E_Z at mu_C 0.7 for t_SN 0.5 / 0.75 / 1
#      (the Rashba version is study_tuning [topology]).

# %% [deck] --- PARAMETERS ---
DECK_WORKERS = 6                                                    # the laptop throttles (95 C) above ~8; crashed 2026-11-11 under 8 + tests
TAU_D = 0.3387
LS_D = (2000.0, 3000.0, 4000.0)
EZ_D = {2000.0: np.round(np.arange(0.10, 1.35001, 0.002), 4),      # 3.4 / 11 / 24 s per point on one thread
        3000.0: np.round(np.arange(0.40, 1.35001, 0.002), 4),      # lower fields: L/xi_M >> 4, kappa ~ 0
        4000.0: np.round(np.arange(0.50, 1.35001, 0.002), 4)}
TAU_TB, EZ_TB0 = 0.5, 0.930
PHI_TB = np.array([1.7417, 1.425, 1.0]) * np.pi                    # along E_Z: winner, field-run line, pi
EZ_TB = np.round(np.arange(0.85, 1.05001, 0.0025), 4)
TCS_D = (0.5, 0.75, 1.0)


def deck_line_cache(L):
    return f'scharf_absDeck_tr_mixed_tau{TAU_D:g}_L{L / 1000:g}'


def junction_topo_mixed(tc):
    """The cross-section of study_tuning [topology] (a 10, W 100, mu_S 1) in the mixed SOC, S/N bond tc."""
    al, be = st.mixed_soc(16.0, 0.15 * np.pi)
    return st.replace(st.case(100.0, alpha=al, beta=be, a=10.0), tc=tc)


# %% [deck] --- 1. kappa(E_Z) FOR L = 2 / 3 / 4 um: where does neutral kappa ~ 1/2 switch on? ---
if run_part('deck'):
    OMd = st.run_scan(partial(osc_point, QM), {'E_Z': EZ}, 'scharf_absOM_modes_mixed_a10')   # cached
    XI_D = lambda e: np.interp(e, EZ, OMd[:, 0])
    TLD = {}
    for L in LS_D:
        TLD[L] = st.run_scan(partial(st.transport_point, junction('mixed', 0.7, TAU_D, L=L)), {'E_Z': EZ_D[L]},
                             deck_line_cache(L), n_workers=DECK_WORKERS, chunksize=2)
        print(f"deck 1: L {L / 1000:g} um done ({int(np.nansum(TLD[L][:, 4]))} failed lead solves)", flush=True)
    print("\n  L [um]  best neutral kappa (E_Z, L/xi_M)   neutral kappa > 0.45 from E_Z (L/xi_M there)   "
          "L/xi_M = 3.8 at E_Z   charged (|G| > 0.1 kappa, kappa > 0.3) from E_Z")
    for L in LS_D:
        e, V = EZ_D[L], TLD[L]
        k, G = V[:, 0], np.abs(V[:, 1])
        neu = G < 0.1 * np.maximum(k, 1e-12)
        j = int(np.argmax(np.where(neu, k, -1)))
        on = e[neu & (k > 0.45)]
        s = L / XI_D(EZ)
        e38 = EZ[np.flatnonzero(s < 3.8)[0]] if (s < 3.8).any() else np.nan
        ch = e[~neu & (k > 0.3)]
        print(f"  {L / 1000:g}       {k[j]:.3f} ({e[j]:.3f}, {L / XI_D(e[j]):.2f})"
              f"                 {on[0] if on.size else np.nan:.3f} ({L / XI_D(on[0]) if on.size else np.nan:.2f})"
              f"                              {e38:.3f}              {ch[0] if ch.size else np.nan:.3f}")


# %% [deck] --- 2. TEMPERATURE at a = 20: the too-strong probe bond for the mixed case of study_rashba [T] ---
if run_part('deck'):
    from study_rashba import SCAN_CASES, E_SCAN, T_SCAN
    q20, EZ20, TC20 = SCAN_CASES['W100_L2_mixed']
    T20 = {tc: st.run_scan(partial(st.channel_point, st.replace(q20, tc_barr_transport=tc)), {'E_Z': EZ20, 'E': E_SCAN},
                           f'scharf_absTtc_W100_L2_mixed_tc{tc:g}', n_workers=DECK_WORKERS, chunksize=8)
           for tc in TC20}
    print("\n  a 20, mixed: best NEUTRAL kappa (|G| < 0.05 kappa) over E_Z 0.55-1.25 at T = "
          + " / ".join(f"{T:g}" for T in T_SCAN) + " mK")
    for tc, V in T20.items():
        best = []
        for T in T_SCAN:
            r = st.thermal_line(V, st.kT_of_mK(T), E_SCAN)
            best.append(np.max(np.where(np.abs(r[:, 1]) < 0.05 * np.maximum(r[:, 0], 1e-12), r[:, 0], 0.0)))
        print(f"    tc_barr {tc:g}: " + " / ".join(f"{b:.3f}" for b in best), flush=True)


# %% [deck] --- 3. TEMPERATURE at a = 10 at the best kappa = 1/2 region (tau 0.5, L 2 um) ---
if run_part('deck'):
    QTB = junction('mixed', 0.7, TAU_TB)
    VTB = st.run_scan(partial(st.channel_point, QTB, phis=tuple(PHI_TB)), {'E_Z': EZ_TB, 'E': E_SCAN},
                      f'scharf_absDeck_T_mixed_tau{TAU_TB:g}_L2_EZ{EZ_TB[0]:g}-{EZ_TB[-1]:g}',
                      n_workers=DECK_WORKERS, chunksize=8)                # (n_EZ, n_E, 6 x 3 phases)
    VTP = st.run_scan(partial(st.channel_point, QTB, EZ_TB0, phis=tuple(PHI_F)), {'E': E_SCAN},
                      f'scharf_absDeck_Tphi_mixed_tau{TAU_TB:g}_L2_EZ{EZ_TB0:g}',
                      n_workers=DECK_WORKERS, chunksize=1)[None]          # (1, n_E, 6 x 241)
    print(f"\n  a 10, tau {TAU_TB}, L 2 um: best neutral kappa (|G| < 0.1 kappa) along E_Z "
          f"{EZ_TB[0]:g}-{EZ_TB[-1]:g} at T = " + " / ".join(f"{T:g}" for T in T_SCAN) + " mK [field window]")
    for n, ph in enumerate(PHI_TB):
        out = []
        for T in T_SCAN:
            r = st.thermal_line(VTB, st.kT_of_mK(T), E_SCAN, phase=n)
            neu = np.abs(r[:, 1]) < 0.1 * np.maximum(r[:, 0], 1e-12)
            win = (neu & (np.abs(r[:, 0] - 0.5) < 0.03)).sum() * (EZ_TB[1] - EZ_TB[0]) * 1e3
            out.append(f"{np.max(np.where(neu, r[:, 0], 0.0)):.3f} [{win:.0f} ueV]")
        print(f"    phi {ph / np.pi:.4f} pi: " + " / ".join(out))
    print(f"  along phi at E_Z {EZ_TB0}: phase extent with |kappa - 1/2| < 0.03 and |G| < 0.1 kappa, mean kappa there")
    for T in T_SCAN:
        r = np.array([st.thermal_line(VTP, st.kT_of_mK(T), E_SCAN, phase=n)[0] for n in range(len(PHI_F) - 1)])
        ok = (np.abs(r[:, 0] - 0.5) < 0.03) & (np.abs(r[:, 1]) < 0.1 * r[:, 0])
        print(f"    {T:4g} mK: {ok.sum() * DPHI:.2f} pi  (kappa {r[ok, 0].mean() if ok.any() else np.nan:.3f}; "
              f"max kappa {r[:, 0].max():.3f})", flush=True)


# %% [deck] --- 4. S/N COUPLING: Q, Delta_top on (mu_C, E_Z) and xi_M at mu_C 0.7 ---
if run_part('deck'):
    from study_tuning import topo_point, xi_scan, MU_C as MU_CT, EZ_TOPO, MU_C0
    MUC_D, XI0_D = {}, {}
    i0 = int(np.argmin(np.abs(MU_CT - MU_C0)))
    for tc in TCS_D:
        MUC_D[tc] = st.run_scan(partial(topo_point, junction_topo_mixed(tc)), {'mu_C': MU_CT, 'E_Z': EZ_TOPO},
                                f'scharf_absDeck_topo_mixed_tc{tc:g}', n_workers=DECK_WORKERS, chunksize=4)
        line = np.flatnonzero(MUC_D[tc][i0, :, 0] < 0)
        XI0_D[tc] = (EZ_TOPO[line], xi_scan(junction_topo_mixed(tc), np.full(len(line), MU_C0), EZ_TOPO[line],
                                            f'scharf_absDeck_xi0_mixed_tc{tc:g}'))
        print(f"deck 4: t_SN {tc:g} done", flush=True)
    # GAPLESS (2026-11-12): in the mixed SOC much of Q = -1 has Delta_top < 1e-3 ueV (real band crossings; nothing
    # between 1e-3 and 0.08 ueV).  There slowest_mode drops the |lam| = 1 modes and returns a bulk-SC mode at 256.6 nm
    # for every field -- NOT a Majorana: such points are excluded from xi_M below (and in slides_nov.slide9).
    print("\n  t_SN   Q = -1 area   gapped Q = -1 area   max Delta_top [ueV] (mu_C, E_Z)   at mu_C 0.7: gapped E_Z,"
          " max Delta_top, xi_M min..max [um], max 2 um / xi_M")
    for tc in TCS_D:
        Qm, g = MUC_D[tc][..., 0], MUC_D[tc][..., 1] * 1e3
        gapped = (Qm < 0) & (g >= 0.01)
        a, b = np.unravel_index(np.nanargmax(np.where(gapped, g, -1)), g.shape)
        ez, xi = XI0_D[tc][0], XI0_D[tc][1][:, 1]
        ok = gapped[i0, Qm[i0] < 0]
        rng = (f"{ez[ok].min():.1f}-{ez[ok].max():.1f} meV, {np.nanmax(g[i0]):.1f} ueV, {xi[ok].min() / 1e3:.2f}.."
               f"{xi[ok].max() / 1e3:.1f}, {2000 / xi[ok].min():.2f}" if ok.any() else "gapless at every Q = -1 field")
        print(f"  {tc:4g}   {np.mean(Qm < 0):6.1%}        {np.mean(gapped):6.1%}           "
              f"{g[a, b] if gapped.any() else 0:6.1f} ({MU_CT[a]:g}, {EZ_TOPO[b]:g})          {rng}", flush=True)


# ====================================================================================================================
# PART 'combine'  (2026-11-11)
# ====================================================================================================================
# CHECK of the 2026-10-07 notes 'Combining Gresta and Scharf (all tested, all negative)' with today's code (no script
# of the 10-07 runs survives), and the same questions at the thesis parameters.  Raw numbers; the 10-07 values are
# printed next to them where the notes give one.
#   A. Gresta model (dimensionless, t = 1; the working point of archive/early_tests/01_single_point.py):
#      A0 baseline; A1 in-plane field in the ribbons (bulk gap, Chern number); A2 tilt (Bz_s 0.94 + Bxy_s);
#      A3 Bz_s + Scharf's in-plane field in the centre (Bxy_c), also with a populated centre; A4 Dresselhaus in both
#      orientations, and nx at beta 0.8.
#   B. thesis parameters (mixed SOC 16 meV nm at 0.15 pi, '100', mu_S 1, Delta 0.25, m* 0.038):
#      B1 bulk of the proximitized ribbon with an out-of-plane Zeeman Bz_s: Chern number and bulk gap (a 10, a 20;
#         pure Rashba 16 for comparison);
#      B2 the junction (a 20) with Chern ribbons, +- Scharf's in-plane field, depleted / populated centre, two probes;
#      B3 out-of-plane Zeeman in the normal strip only (probes kept at 0) at the best point: Q, Delta_top, xi_M, kappa.

# %% [combine] --- PARAMETERS AND HELPERS (definitions only: the parallel workers import this file) ---
from dataclasses import replace as _replace
PG = st.myf.Params(model='rashba', nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
                   delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
TH_INPLANE = (0.0, np.pi / 2)                                       # two in-plane directions (the 10-07 one is unknown)
VZ_B1 = np.array([0.0, 0.5, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0])          # meV, out-of-plane in the ribbons
BZ_B3 = np.array([0.0, 0.05, 0.1, 0.2, 0.4])                                          # meV, out-of-plane in the strip
PHI_B3 = (np.pi, 1.7417 * np.pi)
COMB_WORKERS = 4


def bulk_H(p, kx, ky, region='s'):
    """Bulk BdG H(kx, ky) of one region (lattice units, phase 0), vectorised; same blocks as myf.chern_number."""
    o = st.model_onsite(p, region)
    t, m0 = {'s': (p.t_s, p.m0), 'c': (p.t_c, p.m0_c), 'n': (p.t_n, p.m0_n)}[region]
    X = st.model_hop(p, 'x', t, m0, alpha=st.region_alpha(p, region))
    Y = st.model_hop(p, 'y', t, m0, alpha=st.region_alpha(p, region))
    ex, ey = np.exp(1j * np.asarray(kx))[..., None, None], np.exp(1j * np.asarray(ky))[..., None, None]
    return o + X * ex + X.conj().T / ex + Y * ey + Y.conj().T / ey


def bulk_gap(p, N=241, kmax=np.pi, region='s'):
    """min over (kx, ky) of the smallest |E| of bulk_H (lattice units): N x N grid on [-kmax, kmax]^2, the 6 lowest
    grid points refined by Nelder-Mead.  -> (gap, kx, ky)."""
    from scipy.optimize import minimize
    k = np.linspace(-kmax, kmax, N)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    E = np.abs(np.linalg.eigvalsh(bulk_H(p, KX, KY, region))).min(-1)
    f = lambda v: np.abs(np.linalg.eigvalsh(bulk_H(p, v[0], v[1], region))).min()
    best = (E.min(), KX.flat[E.argmin()], KY.flat[E.argmin()])
    for i in np.argsort(E, axis=None)[:6]:
        r = minimize(f, [KX.flat[i], KY.flat[i]], method='Nelder-Mead',
                     options=dict(xatol=1e-10, fatol=1e-14, maxiter=6000))
        if r.fun < best[0]:
            best = (r.fun, *r.x)
    return best


def gresta_tr(p, phis=(np.pi, 0.0), kT=0.0):
    """[kappa(phis)..., G(phis)..., failed lead solves, overflow warnings] from myf.linear_response (side 'right', as
    the early Gresta scripts)."""
    import warnings
    n0 = st.myf.leads.N_UNRELIABLE
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        k, G = st.myf.linear_response(p, np.asarray(phis, float), kT)
    return np.r_[np.ravel(k), np.ravel(G), st.myf.leads.N_UNRELIABLE - n0,
                 sum('overflow' in str(x.message) for x in w)]


def thesis_q(a, mu_C=0.7, tau=0.3387, beta_on=True):
    """The thesis junction (W 100, L 2 um, t_SN 1, mixed SOC) at lattice constant a; beta_on=False: pure Rashba 16."""
    al, be = st.mixed_soc(16.0, 0.15 * np.pi) if beta_on else (16.0, 0.0)
    return st.replace(st.case(100.0, alpha=al, beta=be, a=a), tc=1.0, mu_C=mu_C, tc_barr_transport=tau)


def comb_point(i, cases):
    """B2: cases[i] = (Vz meV, mu_C, E_Z meV, tau); a = 20.  -> [kappa(pi), kappa(0), G(pi), G(0), andreev(pi), det r(pi),
    failed lead solves]."""
    Vz, mu_C, E_Z, tau = cases[int(i)]
    q = thesis_q(20.0, mu_C, tau)
    n0 = st.myf.leads.N_UNRELIABLE
    p = _replace(q.params_transport(E_Z=float(E_Z)), Bz_s=Vz / q.t)
    rgf = st.myf.RGFFourTerminal(st.myf.FourTerminalJunction(p), np.array([0.0]))
    out = []
    for ph in (np.pi, 0.0):
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(ph, 'left').items()}
        out.append((c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local']))
    return np.array([out[0][0], out[1][0], out[0][1], out[1][1], out[0][2], st.det_r(rgf, np.pi),
                     st.myf.leads.N_UNRELIABLE - n0])


def strip_bz_point(Bz):
    """B3: out-of-plane Zeeman Bz (meV) in the normal strip only (Bz_n = 0), thesis junction a 10, tau 0.5, E_Z 0.93.
    Per phase of PHI_B3: [Q, Delta_top (meV), xi_M (nm), kappa, G, det r, failed lead solves]."""
    from study_oscillation import slowest_mode
    q = thesis_q(10.0, 0.7, 0.5)
    n_S = st.n_S_of(q)
    out = []
    for ph in PHI_B3:
        p = _replace(q.params(E_Z=0.93, phi=ph), Bz=Bz / q.t, Bz_n=0.0)
        Q = st.invariant(p, n_S)
        g, _ = st.gap_refined(p, n_S, st.k_grid(q.a))
        xi = -q.a / np.log(np.abs(slowest_mode(p, n_S)))
        n0 = st.myf.leads.N_UNRELIABLE
        pt = _replace(q.params_transport(E_Z=0.93, phi=ph), Bz=Bz / q.t, Bz_n=0.0)
        rgf = st.myf.RGFFourTerminal(st.myf.FourTerminalJunction(pt), np.array([0.0]))
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(ph, 'left').items()}
        out.append([Q, g * q.t, xi, c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], st.det_r(rgf, ph),
                    st.myf.leads.N_UNRELIABLE - n0])
    return np.array(out)


# %% [combine] --- A0. GRESTA BASELINE ---
if run_part('combine'):
    print("A0  Gresta working point (nx 80, ny 12, Delta 0.35, alpha 1.2, mu_s 0.15, Bz_s 0.94, mu_c -0.03, tc 1.1)")
    g0 = bulk_gap(PG)
    print(f"    Chern {st.myf.chern_number(PG):+.3f} (N 48), {st.myf.chern_number(PG, N=96):+.3f} (N 96); bulk gap "
          f"{g0[0]:.4f} at k = ({g0[1]:.3f}, {g0[2]:.3f}); xi_ribbon {st.myf.coherence_length(PG):.2f} sites  "
          f"[10-07: gap 0.267, xi 5.5]")
    for eta in (1e-5, 1e-6):
        r = gresta_tr(_replace(PG, eta=eta))
        print(f"    eta {eta:g}, kT 0: kappa(pi) {r[0]:.4f}  kappa(0) {r[1]:.4f}  G(pi) {r[2]:+.4f}  G(0) {r[3]:+.4f}  "
              f"failed {int(r[4])}  overflow {int(r[5])}   [notes: 0.4746 at 1e-5, 0.4997 at 1e-6]")
    r = gresta_tr(PG, kT=1e-3)
    print(f"    eta 1e-5, kT 1e-3: kappa(pi) {r[0]:.4f}  kappa(0) {r[1]:.4f}  G(pi) {r[2]:+.4f}   [notes: 0.497, 0.003]",
          flush=True)


# %% [combine] --- A1-A2. IN-PLANE FIELD IN THE RIBBONS: bulk gap and Chern number ---
if run_part('combine'):
    A12 = []
    print("\nA1  Bz_s = 0, in-plane Bxy_s only   [10-07: gapless, bulk gap ~1e-4 for Bxy_s 0.94-3.0, Chern -1 meaningless]")
    for th in TH_INPLANE:
        for B in (0.94, 1.5, 2.0, 3.0):
            p = _replace(PG, Bz_s=0.0, Bxy_s=B, theta_z=th)
            g = bulk_gap(p)
            print(f"    theta {th / np.pi:.2f} pi  Bxy_s {B:4.2f}: bulk gap {g[0]:.2e} at ({g[1]:+.3f}, {g[2]:+.3f})  "
                  f"Chern(N 48) {st.myf.chern_number(p):+.3f}  Chern(N 96) {st.myf.chern_number(p, N=96):+.3f}")
        p = _replace(PG, Bz_s=0.94, Bxy_s=0.94, theta_z=th)
        g = bulk_gap(p)
        print(f"    theta {th / np.pi:.2f} pi  Bz_s 0.94 + Bxy_s 0.94: bulk gap {g[0]:.2e}  Chern(N 96) "
              f"{st.myf.chern_number(p, N=96):+.3f}")
    print("A2  Bz_s 0.94 + Bxy_s (tilt)   [10-07: gap 0.267 -> 0.039 at Bxy_s 0.3, gone by 0.5]")
    for th in TH_INPLANE:
        for B in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            p = _replace(PG, Bxy_s=B, theta_z=th)
            g = bulk_gap(p)
            C = st.myf.chern_number(p, N=96)
            A12.append([th, B, g[0], C])
            print(f"    theta {th / np.pi:.2f} pi  Bxy_s {B:3.1f}: bulk gap {g[0]:.4f}  Chern(N 96) {C:+.3f}", flush=True)
    A12 = np.array(A12)


# %% [combine] --- A3. GRESTA RIBBONS + SCHARF'S IN-PLANE FIELD IN THE CENTRE ---
if run_part('combine'):
    A3 = []
    print("\nA3  Bz_s 0.94 + Bxy_c in the centre, kT 0   [10-07: Bxy_c 0.1: kappa(pi) 0.474 -> 0.148, kappa(0) 0.003 -> 0.652;"
          " Bxy_c 0.8: kappa 0.447, G 0.411; ny 20 no rescue]")
    for th in TH_INPLANE:
        for B in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
            for ny in ((12, 20) if B in (0.0, 0.1) else (12,)):
                r = gresta_tr(_replace(PG, Bxy_c=B, theta_z=th, ny=ny))
                A3.append([th, B, ny, -0.03, *r])
                print(f"    theta {th / np.pi:.2f} pi  Bxy_c {B:4.2f}  ny {ny}: kappa(pi) {r[0]:.4f}  kappa(0) {r[1]:.4f}  "
                      f"G(pi) {r[2]:+.4f}  G(0) {r[3]:+.4f}  failed {int(r[4])}  overflow {int(r[5])}", flush=True)
    print("    populated centre (mu_c > 0), ny 12:")
    for th in TH_INPLANE:
        for mu in (0.1, 0.3):
            for B in (0.0, 0.1, 0.4):
                r = gresta_tr(_replace(PG, mu_c=mu, Bxy_c=B, theta_z=th))
                A3.append([th, B, 12, mu, *r])
                print(f"    theta {th / np.pi:.2f} pi  mu_c {mu:.2f}  Bxy_c {B:4.2f}: kappa(pi) {r[0]:.4f}  kappa(0) {r[1]:.4f}  "
                      f"G(pi) {r[2]:+.4f}  G(0) {r[3]:+.4f}  failed {int(r[4])}  overflow {int(r[5])}", flush=True)
    A3 = np.array(A3)


# %% [combine] --- A4. DRESSELHAUS IN THE GRESTA DEVICE ---
if run_part('combine'):
    A4 = []
    print("\nA4  Dresselhaus beta (alpha 1.2)   [10-07: '110' xi 5.5 -> 49.7 for beta 0 -> 1.1; '100' xi 5.5 -> 4.5, kappa(pi)"
          " 0.91-0.99 with G(pi) ~ 0.33]")
    for ax in ('110', '100'):
        for b in (0.0, 0.2, 0.4, 0.6, 0.8, 1.1):
            p = _replace(PG, beta=b, soc_axis=ax)
            g, C, xi = bulk_gap(p)[0], st.myf.chern_number(p, N=96), st.myf.coherence_length(p)
            row = [b, g, C, xi]
            txt = f"    {ax}  beta {b:3.1f}: bulk gap {g:.4f}  Chern {C:+.3f}  xi {xi:6.2f}"
            for eta in (1e-5, 1e-6):
                r = gresta_tr(_replace(p, eta=eta))
                row += list(r)
                txt += (f" | eta {eta:g}: kappa(pi) {r[0]:.4f} kappa(0) {r[1]:.4f} G(pi) {r[2]:+.4f}"
                        f" failed {int(r[4])} ovf {int(r[5])}")
            A4.append([0 if ax == '110' else 1] + row)
            print(txt, flush=True)
    al_t, be_t = 1.2 * np.cos(0.15 * np.pi), 1.2 * np.sin(0.15 * np.pi)
    p = _replace(PG, alpha=al_t, beta=be_t, soc_axis='100')
    r = gresta_tr(_replace(p, eta=1e-6))
    print(f"    thesis ratio beta/alpha = tan(0.15 pi), amplitude 1.2, '100': bulk gap {bulk_gap(p)[0]:.4f}  Chern "
          f"{st.myf.chern_number(p, N=96):+.3f}  xi {st.myf.coherence_length(p):.2f} | eta 1e-6: kappa(pi) {r[0]:.4f} "
          f"kappa(0) {r[1]:.4f} G(pi) {r[2]:+.4f} failed {int(r[4])} ovf {int(r[5])}")
    print("    '100', beta 0.8, nx   [10-07 (eta 1e-5): kappa(pi) 0.358 -> 0.192 for nx 80 -> 240; notes: unresolved]")
    for nx in (80, 160, 240):
        txt = f"      nx {nx}:"
        for eta in (1e-5, 1e-6, 1e-7):
            r = gresta_tr(_replace(PG, beta=0.8, soc_axis='100', nx=nx, eta=eta))
            txt += f" | eta {eta:g}: kappa(pi) {r[0]:.4f} G(pi) {r[2]:+.4f} failed {int(r[4])} ovf {int(r[5])}"
        print(txt, flush=True)
    A4 = np.array(A4)


# %% [combine] --- B1. THESIS PARAMETERS: CAN THE PROXIMITIZED RIBBON BECOME A CHERN SUPERCONDUCTOR? ---
if run_part('combine'):
    B1 = {}
    print("\nB1  ribbon bulk (mu_S 1, Delta 0.25, m* 0.038), out-of-plane Vz in the ribbon; gap in ueV")
    for name, a, mixed in (('mixed a10', 10.0, True), ('mixed a20', 20.0, True), ('Rashba16 a10', 10.0, False)):
        q = thesis_q(a, beta_on=mixed)
        p0 = q.params(E_Z=0.0, phi=0.0)
        rows = []
        for Vz in VZ_B1:
            p = _replace(p0, Bz_s=Vz / q.t)
            gc = bulk_gap(p, N=161)
            gf = bulk_gap(p, N=401, kmax=0.9)
            g = min(gc[0], gf[0]) * q.t * 1e3
            C1, C2 = st.myf.chern_number(p, N=200), st.myf.chern_number(p, N=400)
            rows.append([Vz, g, C1, C2])
            print(f"    {name}  Vz {Vz:4.2f} meV: bulk gap {g:9.3f} ueV  Chern {C1:+.3f} (N 200) {C2:+.3f} (N 400)", flush=True)
        B1[name] = np.array(rows)


# %% [combine] --- B2. THESIS JUNCTION (a 20) WITH CHERN RIBBONS ---
if run_part('combine'):
    rows = B1['mixed a20']
    chern = rows[(np.abs(rows[:, 3]) > 0.5) & (np.abs(rows[:, 3] - rows[:, 2]) < 0.1) & (rows[:, 1] > 1.0)]
    VZ_B2 = [0.0] + ([float(chern[0, 0]), float(chern[np.argmax(chern[:, 1]), 0])] if len(chern) else [])
    VZ_B2 = list(dict.fromkeys(VZ_B2))
    print(f"\nB2  junction a 20, ribbons Vz {VZ_B2} meV (Chern phase from B1: "
          f"{chern[:, 0].tolist() if len(chern) else 'none'}), mu_C, in-plane E_Z in the strip, probe bond tau")
    CASES_B2 = tuple((Vz, mu, EZ, tau) for Vz in VZ_B2 for mu in (-0.2, 0.7) for EZ in (0.0, 0.3, 0.93) for tau in (1.0, 0.2))
    VB2 = st.run_scan(partial(comb_point, cases=CASES_B2), {'i': np.arange(len(CASES_B2))},
                      'scharf_absComb_B2_' + '_'.join(f'{v:g}' for v in VZ_B2), n_workers=COMB_WORKERS, chunksize=1)
    print("    Vz    mu_C   E_Z   tau  | kappa(pi) kappa(0)  G(pi)    G(0)   andreev(pi) det r(pi) failed")
    for (Vz, mu, EZ, tau), r in zip(CASES_B2, VB2):
        print(f"    {Vz:4.2f}  {mu:+.1f}  {EZ:4.2f}  {tau:3.1f}  |  {r[0]:.4f}   {r[1]:.4f}  {r[2]:+.4f}  {r[3]:+.4f}  "
              f"{r[4]:.4f}     {r[5]:+.3f}    {int(r[6])}")


# %% [combine] --- B3. OUT-OF-PLANE ZEEMAN IN THE NORMAL STRIP ONLY (thesis junction a 10, best point) ---
if run_part('combine'):
    VB3 = st.run_scan(strip_bz_point, {'Bz': BZ_B3}, 'scharf_absComb_B3_stripBz', n_workers=len(BZ_B3), chunksize=1)
    print("\nB3  Bz in the strip (Bz_n = 0), a 10, tau 0.5, E_Z 0.93 (in-plane), L 2 um")
    print("    Bz [meV]  phi/pi  |  Q   Delta_top [ueV]  xi_M [nm]  |  kappa    G        det r   failed")
    for Bz, R in zip(BZ_B3, VB3):
        for ph, r in zip(PHI_B3, R):
            print(f"    {Bz:5.2f}     {ph / np.pi:.4f} | {r[0]:+.0f}   {r[1] * 1e3:8.2f}       {r[2]:8.0f}   |  {r[3]:.4f}  "
                  f"{r[4]:+.4f}  {r[5]:+.3f}   {int(r[6])}", flush=True)


# %% [combine] --- FIGURES ---
if run_part('combine'):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    a = axs[0, 0]
    for th, c in zip(TH_INPLANE, ('#2a78d6', '#eb6834')):
        m = A12[:, 0] == th
        a.plot(A12[m, 1], A12[m, 2], 'o-', color=c, label=rf'$\theta$ = {th / np.pi:.1f}$\pi$')
    a.set_xlabel(r'$B_{xy,s}$ (t)'); a.set_ylabel('ribbon bulk gap (t)'); a.legend(frameon=False)
    a.set_title(r'A2  Gresta: $B_{z,s}$ 0.94 + in-plane leak', fontsize=10)
    a = axs[0, 1]
    for th, c in zip(TH_INPLANE, ('#2a78d6', '#eb6834')):
        m = (A3[:, 0] == th) & (A3[:, 2] == 12) & (A3[:, 3] == -0.03)
        a.plot(A3[m, 1], A3[m, 4], 'o-', color=c, label=rf'$\kappa(\pi)$, $\theta$ = {th / np.pi:.1f}$\pi$')
        a.plot(A3[m, 1], A3[m, 5], 's--', color=c, label=rf'$\kappa(0)$')
        a.plot(A3[m, 1], np.abs(A3[m, 6]), ':', color=c, label=r'$|G(\pi)|$')
    a.set_xlabel(r'$B_{xy,c}$ (t)'); a.set_ylabel('transport'); a.legend(frameon=False, fontsize=7)
    a.set_title(r'A3  Gresta ribbons + in-plane field in the centre', fontsize=10)
    a = axs[0, 2]
    for code, c, lab in ((0, '#2a78d6', "'110'"), (1, '#eb6834', "'100'")):
        m = A4[:, 0] == code
        a.plot(A4[m, 1], A4[m, 4], 'o-', color=c, label=rf'$\xi_{{ribbon}}$ {lab}')
    a.set_yscale('log'); a.set_xlabel(r'$\beta$ (t a)'); a.set_ylabel(r'$\xi_{ribbon}$ (sites)')
    b2 = a.twinx()
    for code, c in ((0, '#2a78d6'), (1, '#eb6834')):
        m = A4[:, 0] == code
        b2.plot(A4[m, 1], A4[m, 11], 's--', color=c, alpha=0.7)                 # kappa(pi) at eta 1e-6
    b2.set_ylabel(r'$\kappa(\pi)$ at $\eta$ 1e-6 (dashed)')
    a.legend(frameon=False, fontsize=8); a.set_title('A4  Dresselhaus in the Gresta device', fontsize=10)
    a = axs[1, 0]
    for (name, R), c in zip(B1.items(), ('#2a78d6', '#1baf7a', '#eb6834')):
        a.semilogy(R[:, 0], np.maximum(R[:, 1], 1e-4), 'o-', color=c, label=f'{name}')
        for Vz, g, C1, C2 in R:
            if abs(C2) > 0.5:
                a.annotate(f'{C2:+.0f}', (Vz, max(g, 1e-4)), fontsize=7, color=c, xytext=(2, 4), textcoords='offset points')
    a.set_xlabel(r'out-of-plane $V_z$ in the ribbon (meV)'); a.set_ylabel(r'ribbon bulk gap ($\mu$eV)')
    a.legend(frameon=False, fontsize=8); a.set_title('B1  thesis ribbon: bulk gap (labels: Chern N 400)', fontsize=10)
    a = axs[1, 1]
    for j, (ph, c) in enumerate(zip(PHI_B3, ('#2a78d6', '#eb6834'))):
        a.plot(BZ_B3, VB3[:, j, 1] * 1e3, 'o-', color=c, label=rf'$\Delta_{{top}}$, $\phi$ = {ph / np.pi:.2f}$\pi$')
    a.set_xlabel(r'out-of-plane $B_z$ in the strip (meV)'); a.set_ylabel(r'$\Delta_{top}$ ($\mu$eV)')
    a2 = a.twinx()
    for j, c in enumerate(('#2a78d6', '#eb6834')):
        a2.plot(BZ_B3, VB3[:, j, 2], 's--', color=c, alpha=0.7)
    a2.set_ylabel(r'$\xi_M$ (nm, dashed)')
    a.legend(frameon=False, fontsize=8); a.set_title(r'B3  $B_z$ in the strip: gap and $\xi_M$', fontsize=10)
    a = axs[1, 2]
    for j, (ph, c) in enumerate(zip(PHI_B3, ('#2a78d6', '#eb6834'))):
        a.plot(BZ_B3, VB3[:, j, 3], 'o-', color=c, label=rf'$\kappa$, $\phi$ = {ph / np.pi:.2f}$\pi$')
        a.plot(BZ_B3, np.abs(VB3[:, j, 4]), ':', color=c, label=r'$|G|$')
    a.axhline(0.5, color='0.5', lw=0.8, ls='--')
    a.set_xlabel(r'out-of-plane $B_z$ in the strip (meV)'); a.set_ylabel('transport')
    a.legend(frameon=False, fontsize=8); a.set_title(r'B3  $\kappa$, $|G|$ ($\tau$ 0.5, $E_Z$ 0.93)', fontsize=10)
    fig.tight_layout()
    st.save_fig(fig, 'combine_checks')
    print("figure: plots/combine_checks.png", flush=True)


# ====================================================================================================================
# PART 'angle'  (2026-11-12)
# ====================================================================================================================
# STUDY: the SOC ANGLE theta: alpha = 16 cos(theta), beta = 16 sin(theta) meV nm ('100'), the field perpendicular to the
# k_x SOC field (theta_z_optimal + pi, as everywhere).  theta = 0 is pure Rashba, 0.15 pi the thesis case.  Why is so
# much of the mixed-SOC Q = -1 region gapless ([deck] 4, [leads]), and does another angle recover or improve the
# protection and the kappa = 1/2 plateau?  junction_angle: a 10, W 100, L 2 um, mu_C 0.7, mu_S 1, S/N bond tc, probe tau.
# theta and pi/2 - theta are the same junction (a pi spin rotation about (1,1,0) swaps the Rashba and Dresselhaus terms
# and the field follows; cell 1 checks 0.35 pi = 0.15 pi), so theta runs over 0 .. pi/4.
#   1. MECHANISM (cross-section, phi = pi).  The two middle BdG bands e_- <= e_+ (eigenvalues n/2 and n/2 + 1 of n)
#      split into the tilt t(k) = (e_+ + e_-)/2 (odd in k by particle-hole symmetry) and the half-gap g(k) = (e_+ - e_-)/2.
#      min_k (g - |t|) is the gap over all k when positive; negative = the pair overlaps zero (gapless).  vs theta
#      (E_Z 0.6, tc 0.5 / 1), vs a rotation of the field off theta_z_optimal (0.15 pi), vs E_Z (0.15 pi, tc 0.75).
#   1b. the same cases refined in k (gap_refined + 401 k within +-4e-4 /nm of its minimum): a zero crossing can be
#      ~1e-4 /nm wide, below cell 1's step, so cell 1's min (g - |t|) can read gapped where the pair crosses zero.
#   2. BULK vs theta: Q, Delta_top (refined) along E_Z at phi = pi for tc 0.5 / 0.75 / 1; the (phi, E_Z) map at tc 1 on
#      half the [tuning] grid (E_Z 0.1 x 16 phases); xi_M at phi = pi, tc 1, only where gapped (the [leads] TRAP).
#   3. TRANSPORT vs theta (tc 1, tau 0.5): kappa, G, det r on the [tuning] grid (theta 0 and 0.15 pi ARE the [tuning]
#      R16 / mixed maps); plateau points = plateau_ok (|kappa - 1/2| < 0.03, |G| < 0.1 kappa, 2M within 0.01) inside
#      gapped Q = -1 (bulk map, nearest point); then the [plateau] stage-1 fine grid (E_Z +- 0.1 step 0.01 x 241 phases)
#      around each angle's longest coarse phase run, with Q and Delta_top on its two longest lines (every 4th phase).
#   4. ROTATING THE FIELD at theta 0.15 pi (theta is fixed by the material and the gates; the field direction is free):
#      phi = pi lines for rotations ROT x tc 0.5 / 0.75 / 1, the tc 1 (phi, E_Z) map for ROT_MAPS, and the coarse
#      transport map at the rotation with the larger gapped area.

# %% [angle] --- PARAMETERS ---
ANG = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)                           # theta / pi
ANG_T = ANG[:-1]                                                    # maps, xi_M, transport (0.25 pi: see cell 1)
TCS_A = (0.5, 0.75, 1.0)
TAU_A = 0.5
ANGLE_WORKERS = 6                                                   # as DECK_WORKERS
GAPLESS_MEV = 1e-5                                                  # Delta_top < 0.01 ueV = gapless ([deck] 4)
KS_A = np.linspace(0.0, 0.08, 161)                                  # 1/nm (k_F ~ 0.03); t odd, g even: k >= 0 suffices
DTHZ = (-0.1, -0.06, -0.03, -0.02, -0.015, -0.01, -0.005, 0.005, 0.015, 0.03, 0.06, 0.1)   # field rotation / pi
EZ_TILT = np.round(np.arange(0.2, 1.4001, 0.1), 2)
TILT_CASES = (tuple((th, tc, 0.6, 0.0) for tc in (0.5, 1.0) for th in ANG + (0.35,))
              + tuple((0.15, tc, 0.6, d) for tc in (0.5, 1.0) for d in DTHZ)
              + tuple((0.15, 0.75, float(e), 0.0) for e in EZ_TILT))
LINE_CASES = tuple((th, tc) for th in ANG for tc in TCS_A) + ((0.35, 1.0),)
EZ_AQ, PHI_AQ = EZ_TU[::2], PHI_TU[::2]                              # E_Z 0.1 x 16 phases = the [tuning] grid halved
EZ_XI = np.round(np.arange(0.1, 1.3001, 0.1), 2)
PHI_FQ = PHI_F[::4]                                                 # Q, Delta_top on the fine lines: every 4th phase
_IE = np.abs(EZ_TU[:, None] - EZ_AQ[None, :]).argmin(1)             # nearest bulk-map point of each transport point
_IP = np.abs(PHI_TU[:, None] - PHI_AQ[None, :]).argmin(1)
_JF = np.abs(PHI_F[:, None] - PHI_FQ[None, :]).argmin(1)
TH_COL = dict(zip(ANG, ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b')))   # sequential blue ramp
TC_COL = dict(zip(TCS_A, ('#2a78d6', '#eb6834', '#1baf7a')))        # first three categorical slots
GAPLESS_C, TRIV_C = '#f5c9b3', '#e4e3df'                            # as slides_nov.slide9


def junction_angle(theta, tc=1.0, tau=TAU_A, L=2000.0, dthz=0.0):
    """The thesis junction with the SOC at angle theta (units of pi), S/N bond tc, probe bond tau, the field rotated by
    dthz (units of pi) off theta_z_optimal + pi.  theta 0.15, tc 1 = study_tuning.junction('mixed', 0.7, tau, L);
    theta 0 = junction('R16', ...)."""
    al, be = st.mixed_soc(16.0, theta * np.pi)
    q = st.replace(st.case(100.0, L=L, alpha=al, beta=be, a=10.0), tc=tc, mu_C=0.7, tc_barr_transport=tau)
    q.theta_z = q.theta_z + dthz * np.pi
    return q


def middle_pair(p, n_S, k):
    """The two middle eigenvalues (n/2 and n/2 + 1 of n, ascending) of H_cross(p, k), units of t (banded solver)."""
    from scipy.linalg import eigvals_banded
    H = st.H_cross(p, k, n_S)
    n = len(H)
    ab = np.zeros((8, n), complex)
    for i in range(8):
        ab[i, :n - i] = np.diagonal(H, -i)
    return eigvals_banded(ab, lower=True, select='i', select_range=(n // 2 - 1, n // 2))


def tilt_case(i):
    """TILT_CASES[i] = (theta, tc, E_Z, field rotation): [t(k), g(k)] in ueV on KS_A at phi = pi, shape (len(KS_A), 2)."""
    th, tc, E_Z, d = TILT_CASES[int(i)]
    q = junction_angle(th, tc, dthz=d)
    p, n_S = q.params(E_Z=float(E_Z), phi=np.pi), st.n_S_of(q)
    E = np.array([middle_pair(p, n_S, k * q.a) for k in KS_A]) * q.t * 1e3
    return np.c_[E.mean(1), (E[:, 1] - E[:, 0]) / 2]


def line_point(i, E_Z):
    """topo_point at phi = pi for LINE_CASES[i] = (theta, tc): [Q, Delta_top (meV; NaN if Q = +1), k of it (1/nm)]."""
    th, tc = LINE_CASES[int(i)]
    return topo_point(junction_angle(th, tc), E_Z)


def map_point(i, E_Z, phi):
    """topo_point on the (phi, E_Z) map, theta ANG_T[i], tc 1."""
    return topo_point(junction_angle(ANG_T[int(i)]), E_Z, phi=phi)


def xi_angle_point(i, E_Z):
    """[Q, Delta_top (meV), xi_M (nm), k_osc (1/nm)] at phi = pi, tc 1, theta ANG_T[i].  xi_M only where gapped: at
    gapless points slowest_mode skips the propagating modes and returns a bulk-SC mode (the [leads] TRAP)."""
    q = junction_angle(ANG_T[int(i)])
    p, n_S = q.params(E_Z=float(E_Z), phi=np.pi), st.n_S_of(q)
    Q = st.invariant(p, n_S)
    if Q > 0:
        return np.array([Q, np.nan, np.nan, np.nan])
    g = st.gap_refined(p, n_S, st.k_grid(q.a))[0] * q.t
    if g < GAPLESS_MEV:
        return np.array([Q, g, np.nan, np.nan])
    lam = slowest_mode(p, n_S)
    return np.array([Q, g, -q.a / np.log(np.abs(lam)), np.abs(np.angle(lam)) / q.a])


def tr_angle(th):
    """Coarse transport map (len(EZ_TU), len(PHI_TU), 5) at theta th, tc 1, tau TAU_A; 0 and 0.15 are [tuning] caches."""
    name = {0.0: tr_name('R16', 0.7, TAU_A, 2000.0), 0.15: tr_name('mixed', 0.7, TAU_A, 2000.0)}.get(
        th, f'scharf_absAngle_tr_th{th:g}_tau{TAU_A:g}')
    return st.run_scan(partial(st.transport_phis, junction_angle(th), phis=PHI_TU), {'E_Z': EZ_TU}, name,
                       n_workers=ANGLE_WORKERS, chunksize=1)


def fine_angle(th, ez):
    """[plateau] stage-1 fine grid at theta th: transport_phis on ez x PHI_F (theta 0.15: the [plateau] cache names)."""
    if th == 0.15:
        return fine_grid(0.7, TAU_A, 2000.0, ez, n_workers=ANGLE_WORKERS, chunksize=1)
    return st.run_scan(partial(st.transport_phis, junction_angle(th), phis=PHI_F), {'E_Z': np.asarray(ez)},
                       f'scharf_absAngle_fine_th{th:g}_EZ{ez[0]:g}-{ez[-1]:g}', n_workers=ANGLE_WORKERS, chunksize=1)


def runs_of(x, m):
    """Contiguous True runs of m on the grid x as 'lo-hi' (grid values), or 'none'."""
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append(f"{x[i]:.2f}-{x[j]:.2f}")
            i = j + 1
        else:
            i += 1
    return ", ".join(out) or "none"


def gapped_max(V_):
    """Largest Delta_top (meV) on gapped Q = -1 points of a topo_point line, 0 if there are none."""
    gp = (V_[:, 0] < 0) & (V_[:, 1] >= GAPLESS_MEV)
    return float(np.max(V_[gp, 1])) if gp.any() else 0.0


ROT = (-0.005, -0.01, -0.015)                                       # field rotations / pi for the phi = pi lines (cell 4)
ROT_CASES = tuple((d, tc) for d in ROT for tc in TCS_A)
ROT_MAPS = (-0.01, 0.01)                                            # rotations of the tc 1 (phi, E_Z) map (cell 4)
ROT_COL = dict(zip((0.0,) + ROT, ('#0b0b0b', '#86b6ef', '#2a78d6', '#104281')))   # d = 0 ink, then a blue ramp


def tilt_refine(i):
    """TILT_CASES[i]: [Delta_top (ueV, gap_refined), its k (1/nm), overlap (ueV)], overlap = max (|t| - g) on 401 k within
    +-4e-4 /nm of that k (> 0: the pair crosses zero).  Needed because the zero crossing can be ~1e-4 /nm wide, below the
    KS_A step: tc 0.75, E_Z 1.2 overlaps by 0.27 ueV over 1e-4 /nm; alpha = beta has a linear node with t = 0."""
    th, tc, E_Z, d = TILT_CASES[int(i)]
    q = junction_angle(th, tc, dthz=d)
    p, n_S = q.params(E_Z=float(E_Z), phi=np.pi), st.n_S_of(q)
    g, k = st.gap_refined(p, n_S, st.k_grid(q.a))
    E = np.array([middle_pair(p, n_S, kk) for kk in k + np.linspace(-4e-4, 4e-4, 401) * q.a]) * q.t * 1e3
    return np.array([g * q.t * 1e3, k / q.a, np.max(np.abs(E.mean(1)) - (E[:, 1] - E[:, 0]) / 2)])


KZ_A = np.linspace(0.018, 0.034, 161)                              # 1/nm: the outer dip, for figure A (a)
ZOOM_CASES = ((0.0, 0.5), (0.15, 0.5))                              # (theta, tc) at E_Z 0.6, phi = pi


def tilt_zoom(i):
    """e_-, e_+ (ueV) on KZ_A for ZOOM_CASES[i] at E_Z 0.6, phi = pi: (len(KZ_A), 2)."""
    th, tc = ZOOM_CASES[int(i)]
    q = junction_angle(th, tc)
    p, n_S = q.params(E_Z=0.6, phi=np.pi), st.n_S_of(q)
    return np.array([middle_pair(p, n_S, k * q.a) for k in KZ_A]) * q.t * 1e3


def rot_line_point(i, E_Z):
    """topo_point at phi = pi, theta 0.15 pi, for ROT_CASES[i] = (field rotation / pi, tc)."""
    d, tc = ROT_CASES[int(i)]
    return topo_point(junction_angle(0.15, tc, dthz=d), E_Z)


def rot_map_point(i, E_Z, phi):
    """topo_point on the (phi, E_Z) map, theta 0.15 pi, tc 1, the field rotated by ROT_MAPS[i]."""
    return topo_point(junction_angle(0.15, dthz=ROT_MAPS[int(i)]), E_Z, phi=phi)


def map_stats(Q, g):
    """(Q = -1 area, gapped area, gapless share of Q = -1, max Delta_top [ueV], mean over gapped [ueV]) of a (phi, E_Z)
    map on PHI_AQ (the phi = 2 pi column, = phi 0, not double-counted)."""
    gp = (Q < 0) & (g >= GAPLESS_MEV)
    q_, gp_ = Q[:, :-1] < 0, gp[:, :-1]
    return (q_.mean(), gp_.mean(), 1 - gp_.sum() / max(q_.sum(), 1), np.max(np.where(gp, g, 0)) * 1e3,
            np.mean(g[gp]) * 1e3 if gp.any() else 0.0)


# %% [angle] --- 1. MECHANISM: tilt and half-gap of the middle BdG pair ---
if run_part('angle'):
    TILT = st.run_scan(tilt_case, {'i': np.arange(len(TILT_CASES))}, 'scharf_absAngle_tilt',
                       n_workers=ANGLE_WORKERS, chunksize=1)               # (n_cases, len(KS_A), 2), ueV
    TSTAT = np.array([[np.abs(T[:, 0]).max(), T[:, 1].min(), np.min(T[:, 1] - np.abs(T[:, 0]))] for T in TILT])
    TCASE = {c: n for n, c in enumerate(TILT_CASES)}
    tst = lambda th, tc, e=0.6, d=0.0: TSTAT[TCASE[(th, tc, e, d)]]
    print("\n  [angle] 1. phi = pi, E_Z 0.6: max |t|, min g, min (g - |t|) [ueV] on k 0-0.08 /nm (step 0.0005)")
    for tc in (0.5, 1.0):
        for th in ANG + (0.35,):
            m, g, s = tst(th, tc)
            print(f"    tc {tc:g}  theta {th:.2f} pi   {m:6.2f}   {g:6.2f}   {s:+7.2f}     t / t(0.10 pi) "
                  f"{m / tst(0.10, tc)[0]:.3f} vs sin 4 theta / sin 0.4 pi {np.sin(4 * np.pi * th) / np.sin(0.4 * np.pi):+.3f};"
                  f"  g / g(0) {g / tst(0.0, tc)[1]:.3f} vs cos 2 theta {np.cos(2 * np.pi * th):.3f}")
    print("  theta 0.15 pi, E_Z 0.6, field rotated off theta_z_optimal by d [pi]: min (g - |t|) [ueV]")
    for tc in (0.5, 1.0):
        print(f"    tc {tc:g}: " + "  ".join(f"{d:+.3f}: {tst(0.15, tc, 0.6, d)[2]:+.1f}" for d in sorted(DTHZ + (0.0,))))
    print("  theta 0.15 pi, tc 0.75, phi = pi:  E_Z   max |t|   min g   min (g - |t|) [ueV]")
    for e in EZ_TILT:
        m, g, s = tst(0.15, 0.75, float(e))
        print(f"                                      {e:.1f}   {m:6.2f}   {g:6.2f}   {s:+7.2f}")


# %% [angle] --- 1b. the signed gap refined in k ---
if run_part('angle'):
    TREF = st.run_scan(tilt_refine, {'i': np.arange(len(TILT_CASES))}, 'scharf_absAngle_tilt_ref',
                       n_workers=ANGLE_WORKERS, chunksize=1)               # (n_cases, 3): Delta_top, its k, overlap
    TZOOM = st.run_scan(tilt_zoom, {'i': np.arange(len(ZOOM_CASES))}, 'scharf_absAngle_tilt_zoom',
                        n_workers=2, chunksize=1)                           # (2, len(KZ_A), 2): e_-, e_+ in ueV
    # signed gap [ueV]: Delta_top where gapped, else minus the largest overlap |t| - g seen (fine or cell 1 grid); 0 = node
    SG = np.where(TREF[:, 0] >= GAPLESS_MEV * 1e3, TREF[:, 0], -np.maximum(np.maximum(TREF[:, 2], -TSTAT[:, 2]), 0.0))
    sg = lambda th, tc, e=0.6, d=0.0: SG[TCASE[(th, tc, e, d)]]
    print("\n  [angle] 1b. signed gap refined in k [ueV] (Delta_top if >= 0.01 ueV, else -overlap; cell 1 value in brackets)")
    for tc in (0.5, 1.0):
        print(f"    E_Z 0.6, tc {tc:g}:  " + "  ".join(f"{th:.2f}: {sg(th, tc):+.2f} ({tst(th, tc)[2]:+.2f})"
                                                       for th in ANG + (0.35,)))
    for tc in (0.5, 1.0):
        print(f"    theta 0.15, tc {tc:g}, rotation d:  " + "  ".join(
            f"{d:+.3f}: {sg(0.15, tc, 0.6, d):+.2f}" for d in sorted(DTHZ + (0.0,))))
    print("    theta 0.15, tc 0.75 vs E_Z:  " + "  ".join(f"{e:.1f}: {sg(0.15, 0.75, float(e)):+.2f} "
                                                        f"({tst(0.15, 0.75, float(e))[2]:+.2f})" for e in EZ_TILT))
    flip = [(c, TSTAT[n, 2], SG[n], TREF[n, 1]) for n, c in enumerate(TILT_CASES) if (TSTAT[n, 2] > 0) != (SG[n] > 0)]
    print(f"    cell 1 grid and refined sign disagree on {len(flip)} of {len(TILT_CASES)} cases:")
    for c, s0, s1, kk in flip:
        print(f"      (theta, tc, E_Z, d) = {c}: grid {s0:+.2f} -> refined {s1:+.3f} ueV at k {kk:.5f} /nm")


# %% [angle] --- 2. BULK vs theta: phi = pi lines, (phi, E_Z) maps at tc 1, xi_M ---
if run_part('angle'):
    LN = st.run_scan(line_point, {'i': np.arange(len(LINE_CASES)), 'E_Z': EZ_TU}, 'scharf_absAngle_lines',
                     n_workers=ANGLE_WORKERS, chunksize=4)                # (n_lines, len(EZ_TU), 3)
    LA = {c: LN[n] for n, c in enumerate(LINE_CASES)}
    print("\n  [angle] 2. phi = pi:  theta  tc     Q = -1 at E_Z                gapped (>= 0.01 ueV) at E_Z    max Delta_top [ueV] (E_Z)")
    for (th, tc), V_ in LA.items():
        Qa, ga = V_[:, 0], V_[:, 1]
        gp = (Qa < 0) & (ga >= GAPLESS_MEV)
        j = int(np.argmax(np.where(gp, ga, -1)))
        print(f"                       {th:.2f}  {tc:4g}   {runs_of(EZ_TU, Qa < 0):27s}  {runs_of(EZ_TU, gp):29s}  "
              + (f"{ga[j] * 1e3:6.1f} ({EZ_TU[j]:.2f})" if gp.any() else "gapless"), flush=True)
    MP = st.run_scan(map_point, {'i': np.arange(len(ANG_T)), 'E_Z': EZ_AQ, 'phi': PHI_AQ}, 'scharf_absAngle_map_tc1',
                     n_workers=ANGLE_WORKERS, chunksize=4)                # (n_theta, 21, 16, 3)
    MQ, MG = MP[..., 0], MP[..., 1]
    MGP = (MQ < 0) & (MG >= GAPLESS_MEV)
    print("  (phi, E_Z) map, tc 1 (E_Z 0-2 step 0.1, 15 phases; phi = 2 pi = 0 not double-counted):")
    print("    theta   Q = -1 area   gapped area   gapless share of Q = -1   max Delta_top [ueV] (phi, E_Z)   mean over gapped")
    for n, th in enumerate(ANG_T):
        q_, gp_ = MQ[n][:, :-1] < 0, MGP[n][:, :-1]
        a, b = np.unravel_index(np.argmax(np.where(MGP[n], MG[n], -1)), MG[n].shape)
        print(f"    {th:.2f}    {q_.mean():6.1%}        {gp_.mean():6.1%}        {1 - gp_.sum() / max(q_.sum(), 1):6.1%}"
              f"                   {MG[n][a, b] * 1e3:6.1f} ({PHI_AQ[b] / np.pi:.2f} pi, {EZ_AQ[a]:.1f})"
              f"            {np.mean(MG[n][MGP[n]]) * 1e3 if MGP[n].any() else 0:6.1f}", flush=True)
    XA = st.run_scan(xi_angle_point, {'i': np.arange(len(ANG_T)), 'E_Z': EZ_XI}, 'scharf_absAngle_xi',
                     n_workers=ANGLE_WORKERS, chunksize=1)                # (n_theta, len(EZ_XI), 4)
    print("  xi_M at phi = pi, tc 1 (gapped fields only):  theta   xi_M min..max [um]   max L/xi_M (L 2 um)   L/xi_M >= 3 at E_Z")
    for n, th in enumerate(ANG_T):
        xi = XA[n, :, 2]
        ok = np.isfinite(xi)
        print(f"                                               {th:.2f}    "
              + (f"{np.nanmin(xi) / 1e3:.2f}..{np.nanmax(xi) / 1e3:.2f}            {2000 / np.nanmin(xi):5.2f}"
                 f"                 {runs_of(EZ_XI, ok & (2000 / np.where(ok, xi, np.inf) >= 3))}" if ok.any() else "none gapped"))
    OMa = st.run_scan(partial(osc_point, QM), {'E_Z': EZ}, 'scharf_absOM_modes_mixed_a10')
    j15 = ANG_T.index(0.15)
    dd = [abs(XA[j15, m, 2] / np.interp(e, EZ, OMa[:, 0]) - 1) for m, e in enumerate(EZ_XI)
          if np.isfinite(XA[j15, m, 2]) and e >= EZ[0]]
    print(f"  check: theta 0.15 xi_M vs the [oscillation] cache at the same fields: max deviation {max(dd) if dd else np.nan:.1e}")


# %% [angle] --- 3. TRANSPORT vs theta (tc 1, tau 0.5, L 2 um) ---
if run_part('angle'):
    TRA = {th: tr_angle(th) for th in ANG_T}
    DPHI_C = 2.0 / (len(PHI_TU) - 1)
    print("\n  [angle] 3. coarse grid (E_Z 0.05 x 30 phases; bulk Q, Delta_top from the nearest map point):")
    print("    theta   kappa > 0.25 (neutral)   plateau: all / in gapped Q=-1   max neutral kappa: all / in gapped Q=-1   "
          "longest phase run in gapped Q=-1 [pi] (E_Z)   failed solves")
    COARSE = {}
    for n, th in enumerate(ANG_T):
        tr = TRA[th]
        k, G = tr[..., 0], tr[..., 1]
        neu = np.abs(G) < 0.1 * np.maximum(k, 1e-12)
        hot = (k > 0.25)[:, :-1]
        gpd = MGP[n][np.ix_(_IE, _IP)]
        P = plateau_ok(tr)
        runs = np.array([periodic_run(P[i] & gpd[i])[0] * DPHI_C for i in range(len(EZ_TU))])
        i_b = int(np.argmax(runs))
        COARSE[th] = (P & gpd, i_b, runs[i_b])
        print(f"    {th:.2f}    {hot.mean():6.1%} ({np.mean(neu[:, :-1][hot]) if hot.any() else np.nan:.2f})          "
              f"{P[:, :-1].mean():5.1%} / {(P & gpd)[:, :-1].mean():5.1%}            {np.max(np.where(neu, k, 0)):.3f} / "
              f"{np.max(np.where(neu & gpd, k, 0)):.3f}                       {runs[i_b]:.2f} ({EZ_TU[i_b]:.2f})"
              f"                            {int(np.nansum(tr[..., 4]))}",
              flush=True)
    FINE, BESTF = {}, {}
    print("  fine grids (E_Z +- 0.1 step 0.01 around the longest coarse run, 240 phases); Q and Delta_top every 4th phase:")
    for th in ANG_T:
        e0 = EZ_TU[COARSE[th][1]]
        ez = np.round(np.arange(e0 - 0.10, e0 + 0.1001, 0.01), 3)
        ez = ez[ez > 0]
        FINE[th] = (ez, fine_angle(th, ez))
        tr = FINE[th][1]
        order = sorted(range(len(ez)), key=lambda i: -periodic_run(plateau_ok(tr[i]))[0])[:2]
        best = (0.0, order[0], 0, None)
        for i in order:
            B = st.run_scan(partial(topo_point, junction_angle(th), float(ez[i])), {'phi': PHI_FQ},
                            f'scharf_absAngle_Qline_th{th:g}_EZ{ez[i]:g}', n_workers=ANGLE_WORKERS, chunksize=2)
            ok = (B[_JF, 0] < 0) & (B[_JF, 1] >= GAPLESS_MEV)
            nr, s = periodic_run(plateau_ok(tr[i]) & ok)
            print(f"    theta {th:.2f} pi, E_Z {ez[i]:.3f}: run {periodic_run(plateau_ok(tr[i]))[0] * DPHI:.3f} pi -> "
                  f"{nr * DPHI:.3f} pi in gapped Q = -1 (of the phases: Q = +1 {np.mean(B[:-1, 0] > 0):.0%}, gapless "
                  f"Q = -1 {np.mean((B[:-1, 0] < 0) & ~(B[:-1, 1] >= GAPLESS_MEV)):.0%})", flush=True)
            if best[3] is None or nr * DPHI > best[0]:
                best = (nr * DPHI, i, s, B)
        BESTF[th] = best
    print("    theta   best run [pi]   E_Z     kappa on it          max |G|   max 2M residual")
    for th, (run, i, s, B) in BESTF.items():
        ez, tr = FINE[th]
        idx = (s + np.arange(int(round(run / DPHI)))) % (len(PHI_F) - 1)
        if run > 0:
            kk, GG, dr = tr[i, idx, 0], tr[i, idx, 1], tr[i, idx, 3]
            print(f"    {th:.2f}    {run:.3f}           {ez[i]:.3f}   {kk.min():.3f}..{kk.max():.3f} ({kk.mean():.3f})"
                  f"   {np.abs(GG).max():.3f}     {np.abs(kk - (1 - dr**2) / 2).max():.1e}")
        else:
            print(f"    {th:.2f}    0               --")


# %% [angle] --- 3b. each angle's best line refined in E_Z (+-0.0075, step 0.0025, as [plateau] 4) ---
# The run along phi is sensitive to the field on the 0.01 grid ([plateau]: 1.23 / 1.37 / 1.15 pi at 0.9225 / 0.930 /
# 0.935), so the angles are compared at their refined best.  theta 0.15 reuses the [plateau] stage-3 cache.
if run_part('angle'):
    print("\n  [angle] 3b. best fine line refined in E_Z (run without the Q check; a longer line gets Q, Delta_top):")
    BESTR = {}
    for th in ANG_T:
        run0, i0, s0, B0 = BESTF[th]
        e_b = float(FINE[th][0][i0])
        ez_r = np.round(e_b + np.arange(-0.0075, 0.00751, 0.0025), 4)
        tr_r = fine_angle(th, ez_r)
        runs_r = np.array([periodic_run(plateau_ok(tr_r[i]))[0] * DPHI for i in range(len(ez_r))])
        BESTR[th] = (run0, e_b, s0, B0, FINE[th][1][i0])
        for i_r in np.argsort(-runs_r):
            if runs_r[i_r] <= BESTR[th][0] + 1e-9:
                break
            if abs(ez_r[i_r] - e_b) < 1e-9:
                continue
            B = st.run_scan(partial(topo_point, junction_angle(th), float(ez_r[i_r])), {'phi': PHI_FQ},
                            f'scharf_absAngle_Qline_th{th:g}_EZ{ez_r[i_r]:g}', n_workers=ANGLE_WORKERS, chunksize=2)
            ok = (B[_JF, 0] < 0) & (B[_JF, 1] >= GAPLESS_MEV)
            nr, s = periodic_run(plateau_ok(tr_r[i_r]) & ok)
            if nr * DPHI > BESTR[th][0] + 1e-9:
                BESTR[th] = (nr * DPHI, float(ez_r[i_r]), s, B, tr_r[i_r])
        print(f"    theta {th:.2f} pi: " + "  ".join(f"{e:.4f}: {r:.3f}" for e, r in zip(ez_r, runs_r))
              + f"  -> best {BESTR[th][0]:.3f} pi at E_Z {BESTR[th][1]:.4f} (in gapped Q = -1)", flush=True)
    print("    theta   best run [pi]   E_Z      kappa on it            max |G|   max 2M residual   det r on it")
    for th, (run, e_b, s, B, trl) in BESTR.items():
        if run <= 0:
            print(f"    {th:.2f}    0               --")
            continue
        idx = (s + np.arange(int(round(run / DPHI)))) % (len(PHI_F) - 1)
        kk, GG, dr = trl[idx, 0], trl[idx, 1], trl[idx, 3]
        print(f"    {th:.2f}    {run:.3f}           {e_b:.4f}   {kk.min():.3f}..{kk.max():.3f} ({kk.mean():.3f})"
              f"   {np.abs(GG).max():.3f}     {np.abs(kk - (1 - dr**2) / 2).max():.1e}           "
              f"{dr.min():+.2f}..{dr.max():+.2f}")


# %% [angle] --- 4. ROTATING THE FIELD at theta 0.15 pi ---
if run_part('angle'):
    RL = st.run_scan(rot_line_point, {'i': np.arange(len(ROT_CASES)), 'E_Z': EZ_TU}, 'scharf_absAngle_rotlines',
                     n_workers=ANGLE_WORKERS, chunksize=4)                # (len(ROT_CASES), len(EZ_TU), 3)
    RLA = {(0.0, tc): LA[(0.15, tc)] for tc in TCS_A}
    RLA.update({c: RL[n] for n, c in enumerate(ROT_CASES)})
    print("\n  [angle] 4. theta 0.15 pi, phi = pi, field rotated by d [pi] off the optimum:")
    print("    tc     d        Q = -1 at E_Z                gapped (>= 0.01 ueV) at E_Z    max Delta_top [ueV] (E_Z)")
    for tc in TCS_A:
        for d in (0.0,) + ROT:
            V_ = RLA[(d, tc)]
            Qa, ga = V_[:, 0], V_[:, 1]
            gp = (Qa < 0) & (ga >= GAPLESS_MEV)
            j = int(np.argmax(np.where(gp, ga, -1)))
            print(f"    {tc:4g}  {d:+.3f}   {runs_of(EZ_TU, Qa < 0):27s}  {runs_of(EZ_TU, gp):29s}  "
                  + (f"{ga[j] * 1e3:6.1f} ({EZ_TU[j]:.2f})" if gp.any() else "gapless"), flush=True)
    RMP = st.run_scan(rot_map_point, {'i': np.arange(len(ROT_MAPS)), 'E_Z': EZ_AQ, 'phi': PHI_AQ},
                      'scharf_absAngle_rotmap_tc1', n_workers=ANGLE_WORKERS, chunksize=4)   # (len(ROT_MAPS), 21, 16, 3)
    RMAP = {0.0: MP[ANG_T.index(0.15)]}
    RMAP.update({d: RMP[n] for n, d in enumerate(ROT_MAPS)})
    RSTAT = {d: map_stats(M[..., 0], M[..., 1]) for d, M in sorted(RMAP.items())}
    print("  (phi, E_Z) map, tc 1:  d        Q = -1 area   gapped area   gapless share of Q = -1   max Delta_top   "
          "mean over gapped [ueV]")
    for d, (qa, ga, gs, gm, gmean) in RSTAT.items():
        print(f"                         {d:+.3f}   {qa:6.1%}        {ga:6.1%}        {gs:6.1%}                   "
              f"{gm:6.1f}          {gmean:6.1f}")
    D_TR = max(ROT_MAPS, key=lambda d: RSTAT[d][1])
    TR_ROT = st.run_scan(partial(st.transport_phis, junction_angle(0.15, dthz=D_TR), phis=PHI_TU), {'E_Z': EZ_TU},
                         f'scharf_absAngle_tr_th0.15_d{D_TR:g}_tau{TAU_A:g}', n_workers=ANGLE_WORKERS, chunksize=1)
    print(f"  coarse transport (tau {TAU_A:g}) at d = {D_TR:+.3f} (the larger gapped area) vs d = 0:")
    print("    d        kappa > 0.25 (neutral)   plateau: all / in gapped Q=-1   max neutral kappa in gapped Q=-1   "
          "longest run [pi] (E_Z)   failed solves")
    ROT_COARSE = {}
    for d, tr in ((0.0, TRA[0.15]), (D_TR, TR_ROT)):
        M = RMAP[d]
        gpd = ((M[..., 0] < 0) & (M[..., 1] >= GAPLESS_MEV))[np.ix_(_IE, _IP)]
        k, G = tr[..., 0], tr[..., 1]
        neu = np.abs(G) < 0.1 * np.maximum(k, 1e-12)
        hot = (k > 0.25)[:, :-1]
        P = plateau_ok(tr)
        runs = np.array([periodic_run(P[i] & gpd[i])[0] * DPHI_C for i in range(len(EZ_TU))])
        i_b = int(np.argmax(runs))
        ROT_COARSE[d] = (P & gpd, i_b, runs[i_b])
        print(f"    {d:+.3f}   {hot.mean():6.1%} ({np.mean(neu[:, :-1][hot]) if hot.any() else np.nan:.2f})          "
              f"{P[:, :-1].mean():5.1%} / {(P & gpd)[:, :-1].mean():5.1%}            {np.max(np.where(neu & gpd, k, 0)):.3f}"
              f"                              {runs[i_b]:.2f} ({EZ_TU[i_b]:.2f})              "
              f"{int(np.nansum(tr[..., 4]))}", flush=True)


# %% [angle] --- SUMMARY TABLE ---
if run_part('angle'):
    print("\n  [angle] SUMMARY (tc 1, phi = pi unless stated):")
    print("    theta   max Delta_top [ueV]   gapped area (map)   max L/xi_M   plateau % in gapped Q=-1 (coarse)   best run [pi] "
          "(refined E_Z)")
    for n, th in enumerate(ANG_T):
        xi = XA[n, :, 2]
        print(f"    {th:.2f}    {gapped_max(LA[(th, 1.0)]) * 1e3:6.1f}                {MGP[n][:, :-1].mean():6.1%}"
              f"              {2000 / np.nanmin(xi) if np.isfinite(xi).any() else np.nan:5.2f}        "
              f"{COARSE[th][0][:, :-1].mean():5.1%}                               {BESTR[th][0]:.3f} "
              f"({BESTR[th][1]:.4f})")
    print(f"    0.15 with the field rotated: max Delta_top at phi = pi " + ", ".join(
        f"{gapped_max(RLA[(d, 1.0)]) * 1e3:.1f} (d {d:+.3f})" for d in ROT) + f";  d {D_TR:+.3f}: gapped area "
        f"{RSTAT[D_TR][1]:.1%}, plateau in gapped Q=-1 {ROT_COARSE[D_TR][0][:, :-1].mean():.1%}")


# %% [angle] --- FIGURES ---
if run_part('angle'):
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    INK2 = '#52514e'
    # --- A. the mechanism ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    a = axs[0, 0]
    for n, (th, tc) in enumerate(ZOOM_CASES):
        lab, ls = (r'pure Rashba ($\theta$ = 0)', '--') if th == 0 else (rf'mixed ($\theta$ = {th:g}$\pi$)', '-')
        a.plot(KZ_A, TZOOM[n, :, 1], ls, color=TH_COL[th], lw=2, label=lab)
        a.plot(KZ_A, TZOOM[n, :, 0], ls, color=TH_COL[th], lw=2)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlim(KZ_A[0], KZ_A[-1])
    a.set_ylim(-40, 40)
    a.set_xlabel(r'$k$ along the junction [1/nm]  ($k < 0$ by symmetry: $e_\pm(-k) = -e_\mp(k)$)')
    a.set_ylabel(r'middle BdG pair $e_\pm(k)$ [$\mu$eV]')
    a.set_title(r'(a) $t_{SN}$ 0.5, $E_Z$ 0.6 meV, $\phi = \pi$: the mixed pair is tilted across zero', fontsize=10)
    a.legend(frameon=False, fontsize=9)
    a = axs[0, 1]
    x = np.linspace(0, 0.25, 101)
    for tc in (0.5, 1.0):
        S = np.array([tst(th, tc) for th in ANG])
        a.plot(ANG, S[:, 1], 's:', color=TC_COL[tc], ms=7, lw=1.5, label=rf'min half-gap $g$, $t_{{SN}}$ {tc:g}')
        a.plot(ANG, S[:, 0], '^--', color=TC_COL[tc], ms=7, lw=1.5, label=rf'max tilt $|t|$, $t_{{SN}}$ {tc:g}')
        a.plot(ANG, [sg(th, tc) for th in ANG], 'o-', color=TC_COL[tc], ms=8, lw=2,
               label=rf'signed gap (refined), $t_{{SN}}$ {tc:g}')
        a.plot(x, S[0, 1] * np.cos(2 * np.pi * x), color='0.6', lw=0.8)
        a.plot(x, tst(0.10, tc)[0] * np.sin(4 * np.pi * x) / np.sin(0.4 * np.pi), color='0.6', lw=0.8)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlabel(r'SOC angle $\theta/\pi$  ($\alpha = 16\cos\theta$, $\beta = 16\sin\theta$ meV nm)')
    a.set_ylabel(r'[$\mu$eV]')
    a.set_title(r'(b) $E_Z$ 0.6, $\phi = \pi$; grey: $g(0)\cos 2\theta$ and $\propto \sin 4\theta$', fontsize=10)
    a.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    a = axs[1, 0]
    ds = np.array(sorted(DTHZ + (0.0,)))
    for tc in (0.5, 1.0):
        a.plot(ds, [sg(0.15, tc, 0.6, d) for d in ds], 'o-', color=TC_COL[tc], ms=7, lw=2, label=rf'$t_{{SN}}$ {tc:g}')
    a.axhline(0, color=INK2, lw=0.8)
    a.axvline(0, color='0.6', lw=0.8, ls=':')
    a.set_xlabel(r'field rotated off the optimum [$\pi$]')
    a.set_ylabel(r'signed gap, refined [$\mu$eV]  (< 0: gapless)')
    a.set_title(r'(c) $\theta$ = 0.15$\pi$, $E_Z$ 0.6: rotating the field', fontsize=10)
    a.legend(frameon=False, fontsize=9)
    a = axs[1, 1]
    S = np.array([tst(0.15, 0.75, float(e)) for e in EZ_TILT])
    a.plot(EZ_TILT, S[:, 1], 's:', color=TC_COL[0.75], ms=7, lw=1.5, label='min half-gap $g$')
    a.plot(EZ_TILT, S[:, 0], '^--', color=TC_COL[0.75], ms=7, lw=1.5, label='max tilt $|t|$')
    a.plot(EZ_TILT, [sg(0.15, 0.75, float(e)) for e in EZ_TILT], 'o-', color=TC_COL[0.75], ms=8, lw=2,
           label='signed gap (refined)')
    V_ = LA[(0.15, 0.75)]
    a.plot(EZ_TU, np.where(V_[:, 0] < 0, V_[:, 1] * 1e3, np.nan), '-', color=INK2, lw=1,
           label=r'$\Delta_{top}$ (refined, cell 2)')
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlim(0.15, 1.45)
    a.set_xlabel(r'$E_Z$ [meV]')
    a.set_ylabel(r'[$\mu$eV]')
    a.set_title(r'(d) $\theta$ = 0.15$\pi$, $t_{SN}$ 0.75, $\phi = \pi$: the gap closes by tilt', fontsize=10)
    a.legend(frameon=False, fontsize=9)
    fig.suptitle(r'Why the mixed SOC is gapless inside $Q = -1$: the middle BdG pair $e_\pm = t \pm g$ (a 10, W 100, '
                 r'$\mu_C$ 0.7)', fontsize=11)
    st.save_fig(fig, 'angle_mechanism', dpi=150)
    plt.close(fig)

    # --- B. bulk along E_Z at phi = pi, and xi_M ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for a, tc in zip((axs[0, 0], axs[0, 1], axs[1, 0]), (1.0, 0.75, 0.5)):
        for th in ANG:
            V_ = LA[(th, tc)]
            gp = (V_[:, 0] < 0) & (V_[:, 1] >= GAPLESS_MEV)
            gl = (V_[:, 0] < 0) & ~gp
            a.plot(EZ_TU, np.where(gp, V_[:, 1] * 1e3, np.nan), 'o-', color=TH_COL[th], ms=4, lw=2,
                   label=rf'{th:.2f}$\pi$')
            a.plot(EZ_TU[gl], np.zeros(gl.sum()), 'x', color=TH_COL[th], ms=8, mew=2)
        a.set_xlabel(r'$E_Z$ [meV]')
        a.set_ylabel(r'$\Delta_{top}$ [$\mu$eV]')
        a.set_title(rf'$t_{{SN}}$ = {tc:g}, $\phi = \pi$  (x at 0: $Q = -1$ but gapless; no point: trivial)', fontsize=10)
        a.set_ylim(bottom=-3)
    axs[0, 0].legend(frameon=False, fontsize=9, title=r'$\theta$', ncol=2, loc='upper left')
    a = axs[1, 1]
    for n, th in enumerate(ANG_T):
        a.semilogy(EZ_XI, XA[n, :, 2] / 1e3, 'o-', color=TH_COL[th], ms=6, lw=2, label=rf'{th:.2f}$\pi$')
    a.axhline(2.0, color=INK2, lw=0.8, ls='--')
    a.text(0.12, 2.15, r'$L$ = 2 $\mu$m', color=INK2, fontsize=10)
    a.set_xlabel(r'$E_Z$ [meV]')
    a.set_ylabel(r'$\xi_M$ [$\mu$m]')
    a.set_title(r'$\xi_M$ at $\phi = \pi$, $t_{SN}$ 1 (gapped fields only)', fontsize=10)
    a.legend(frameon=False, fontsize=9, title=r'$\theta$')
    fig.suptitle(r'Topological protection vs SOC angle $\theta$ (amplitude 16 meV nm; a 10, W 100, $\mu_C$ 0.7)', fontsize=11)
    st.save_fig(fig, 'angle_bulk', dpi=150)
    plt.close(fig)

    # --- C. (phi, E_Z) maps of Delta_top at tc 1 ---
    fig, axs = plt.subplots(1, len(ANG_T), figsize=(3.3 * len(ANG_T) + 1.2, 4.4), sharey=True, layout='constrained')
    vmax = np.nanmax(np.where(MGP, MG, np.nan)) * 1e3
    for n, (a, th) in enumerate(zip(axs, ANG_T)):
        a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where(MQ[n] > 0, 1.0, np.nan), shading='nearest',
                     cmap=ListedColormap([TRIV_C]), vmin=0, vmax=1)
        a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where((MQ[n] < 0) & ~MGP[n], 1.0, np.nan), shading='nearest',
                     cmap=ListedColormap([GAPLESS_C]), vmin=0, vmax=1)
        im = a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where(MGP[n], MG[n] * 1e3, np.nan), shading='nearest',
                          cmap='Blues', vmin=0, vmax=vmax)
        a.set_title(rf'$\theta$ = {th:.2f}$\pi$' + '\n' + f'gapped {MGP[n][:, :-1].mean():.0%}, '
                    f'Q = -1 {np.mean(MQ[n][:, :-1] < 0):.0%}', fontsize=10)
        a.set_xlabel(r'$\phi/\pi$')
    axs[0].set_ylabel(r'$E_Z$ [meV]')
    fig.colorbar(im, ax=axs, label=r'$\Delta_{top}$ [$\mu$eV]  (grey: trivial; orange: $Q = -1$ but gapless)')
    fig.suptitle(r'$\Delta_{top}$ on the ($\phi$, $E_Z$) plane vs SOC angle ($t_{SN}$ 1)', fontsize=11)
    st.save_fig(fig, 'angle_maps', dpi=150)
    plt.close(fig)

    # --- D. transport: coarse kappa maps and the best fine phase line ---
    fig, axs = plt.subplots(2, len(ANG_T), figsize=(3.3 * len(ANG_T) + 1.2, 8.4), layout='constrained')
    for n, th in enumerate(ANG_T):
        a = axs[0, n]
        im = a.pcolormesh(PHI_TU / np.pi, EZ_TU, TRA[th][..., 0], shading='nearest', cmap='viridis', vmin=0, vmax=1,
                          rasterized=True)
        if MGP[n].any() and not MGP[n].all():
            a.contour(PHI_AQ / np.pi, EZ_AQ, MGP[n].astype(float), levels=[0.5], colors='w', linewidths=1.2,
                      linestyles='--')
        Pg = COARSE[th][0]
        ii, jj = np.nonzero(Pg)
        a.plot(PHI_TU[jj] / np.pi, EZ_TU[ii], 'o', ms=4, mfc='none', mec='#e34948', mew=1.2)
        a.set_title(rf'$\theta$ = {th:.2f}$\pi$: plateau {Pg[:, :-1].mean():.1%}', fontsize=10)
        a.set_xlabel(r'$\phi/\pi$')
        if n == 0:
            a.set_ylabel(r'$E_Z$ [meV]')
        a = axs[1, n]
        run, e_b, s, B, trl = BESTR[th]
        a.plot(PHI_F / np.pi, trl[:, 0], '-', color='#2a78d6', lw=2, label=r'$\kappa/\kappa_0$')
        a.plot(PHI_F / np.pi, np.abs(trl[:, 1]), '-', color='#eb6834', lw=1.5, label=r'$|G|/G_0$')
        bad = ~((B[:, 0] < 0) & (B[:, 1] >= GAPLESS_MEV))
        a.plot(PHI_FQ[bad] / np.pi, np.full(bad.sum(), -0.04), 's', color=GAPLESS_C, ms=4)
        a.axhspan(0.47, 0.53, color='0.88', zorder=0)
        a.set_ylim(-0.08, 1.0)
        a.set_title(rf'best line $E_Z$ {e_b:.4f}: run {run:.2f}$\pi$', fontsize=10)
        a.set_xlabel(r'$\phi/\pi$')
        if n == 0:
            a.set_ylabel('transport (squares: not gapped Q = -1)')
            a.legend(frameon=False, fontsize=9)
    fig.colorbar(im, ax=axs[0], label=r'$\kappa/\kappa_0$')
    fig.suptitle(r'$\kappa$ vs SOC angle ($t_{SN}$ 1, $\tau$ 0.5, L 2 $\mu$m; red: plateau points in gapped $Q = -1$, '
                 r'white dashed: edge of gapped $Q = -1$)', fontsize=11)
    st.save_fig(fig, 'angle_transport', dpi=150)
    plt.close(fig)

    # --- F. rotating the field at theta 0.15 pi ---
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6), layout='constrained')
    for a, tc in zip(axs[0, :3], TCS_A):
        for d in (0.0,) + ROT:
            V_ = RLA[(d, tc)]
            gp = (V_[:, 0] < 0) & (V_[:, 1] >= GAPLESS_MEV)
            gl = (V_[:, 0] < 0) & ~gp
            a.plot(EZ_TU, np.where(gp, V_[:, 1] * 1e3, np.nan), 'o-', color=ROT_COL[d], ms=4, lw=2 if d == 0 else 1.5,
                   label=f'{d:+.3f}' if d else '0 (optimum)')
            a.plot(EZ_TU[gl], np.zeros(gl.sum()), 'x', color=ROT_COL[d], ms=8, mew=2)
        a.set_xlabel(r'$E_Z$ [meV]')
        a.set_ylabel(r'$\Delta_{top}$ at $\phi = \pi$ [$\mu$eV]')
        a.set_title(rf'$t_{{SN}}$ {tc:g}  (x at 0: $Q = -1$ but gapless)', fontsize=10)
        a.set_ylim(bottom=-3)
    axs[0, 0].legend(frameon=False, fontsize=9, title=r'rotation $d$ [$\pi$]')
    vmax = max(np.nanmax(np.where((M[..., 0] < 0) & (M[..., 1] >= GAPLESS_MEV), M[..., 1], np.nan)) for M in RMAP.values()) * 1e3
    for a, d in zip(axs[1, :3], sorted(RMAP)):
        M = RMAP[d]
        gp = (M[..., 0] < 0) & (M[..., 1] >= GAPLESS_MEV)
        a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where(M[..., 0] > 0, 1.0, np.nan), shading='nearest',
                     cmap=ListedColormap([TRIV_C]), vmin=0, vmax=1)
        a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where((M[..., 0] < 0) & ~gp, 1.0, np.nan), shading='nearest',
                     cmap=ListedColormap([GAPLESS_C]), vmin=0, vmax=1)
        imd = a.pcolormesh(PHI_AQ / np.pi, EZ_AQ, np.where(gp, M[..., 1] * 1e3, np.nan), shading='nearest',
                           cmap='Blues', vmin=0, vmax=vmax)
        qa, ga, gs = RSTAT[d][:3]
        a.set_title(rf'$t_{{SN}}$ 1, $d$ = {d:+.3f}$\pi$: gapped {ga:.0%}, gapless share {gs:.0%}', fontsize=10)
        a.set_xlabel(r'$\phi/\pi$')
        a.set_ylabel(r'$E_Z$ [meV]')
        a.set_ylim(-0.05, 1.55)
    fig.colorbar(imd, ax=axs[1, :3], label=r'$\Delta_{top}$ [$\mu$eV]  (grey: trivial; orange: $Q = -1$ but gapless)',
                 shrink=0.9)
    for a, d, tr in ((axs[0, 3], 0.0, TRA[0.15]), (axs[1, 3], D_TR, TR_ROT)):
        imk = a.pcolormesh(PHI_TU / np.pi, EZ_TU, tr[..., 0], shading='nearest', cmap='viridis', vmin=0, vmax=1,
                           rasterized=True)
        ii, jj = np.nonzero(ROT_COARSE[d][0])
        a.plot(PHI_TU[jj] / np.pi, EZ_TU[ii], 'o', ms=4, mfc='none', mec='#e34948', mew=1.2)
        a.set_title(rf'$\kappa/\kappa_0$, $\tau$ {TAU_A:g}, $d$ = {d:+.3f}$\pi$: plateau '
                    f'{ROT_COARSE[d][0][:, :-1].mean():.1%}', fontsize=10)
        a.set_xlabel(r'$\phi/\pi$')
        a.set_ylabel(r'$E_Z$ [meV]')
        fig.colorbar(imk, ax=a, shrink=0.9)
    fig.suptitle(r'Rotating the in-plane field at $\theta$ = 0.15$\pi$ ($d$ = rotation off $\theta_z$ optimal; red: plateau '
                 r'points in gapped $Q = -1$)', fontsize=11)
    st.save_fig(fig, 'angle_rotation', dpi=150)
    plt.close(fig)

    # --- E. summary vs theta ---
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), layout='constrained')
    for tc in TCS_A:
        axs[0].plot(ANG, [gapped_max(LA[(th, tc)]) * 1e3 for th in ANG], 'o-', color=TC_COL[tc], ms=8, lw=2,
                    label=rf'$t_{{SN}}$ {tc:g}')
    axs[0].set_ylabel(r'max $\Delta_{top}$ at $\phi = \pi$ [$\mu$eV]')
    axs[0].legend(frameon=False, fontsize=9)
    axs[1].plot(ANG_T, [np.mean(MQ[n][:, :-1] < 0) * 100 for n in range(len(ANG_T))], 's--', color=INK2, ms=7, lw=1.5,
                label='$Q = -1$')
    axs[1].plot(ANG_T, [MGP[n][:, :-1].mean() * 100 for n in range(len(ANG_T))], 'o-', color='#2a78d6', ms=8, lw=2,
                label='gapped $Q = -1$')
    axs[1].set_ylabel(r'share of the ($\phi$, $E_Z$) map [%]  ($t_{SN}$ 1)')
    axs[1].legend(frameon=False, fontsize=9)
    axs[2].plot(ANG_T, [2000 / np.nanmin(XA[n, :, 2]) if np.isfinite(XA[n, :, 2]).any() else np.nan
                        for n in range(len(ANG_T))], 'o-', color='#2a78d6', ms=8, lw=2)
    axs[2].axhline(3, color=INK2, lw=0.8, ls='--')
    axs[2].set_ylabel(r'max $L/\xi_M$ at $\phi = \pi$ (L 2 $\mu$m, $t_{SN}$ 1)')
    axs[3].plot(ANG_T, [BESTF[th][0] for th in ANG_T], 's:', color=INK2, ms=6, lw=1, label=r'$E_Z$ step 0.01')
    axs[3].plot(ANG_T, [BESTR[th][0] for th in ANG_T], 'o-', color='#2a78d6', ms=8, lw=2, label='refined (0.0025)')
    axs[3].legend(frameon=False, fontsize=9)
    axs[3].set_ylabel(r'longest $\kappa = 1/2$ phase run [$\pi$]  ($\tau$ 0.5)')
    for a in axs:
        a.set_xlabel(r'SOC angle $\theta/\pi$')
        a.set_ylim(bottom=0)
    fig.suptitle(r'SOC angle: protection and the $\kappa = 1/2$ plateau ($\theta$ = 0: pure Rashba; 0.15$\pi$: thesis case)',
                 fontsize=11)
    st.save_fig(fig, 'angle_summary', dpi=150)
    plt.close(fig)
    print("figures: plots/angle_mechanism.png, angle_bulk.png, angle_maps.png, angle_transport.png, angle_rotation.png, "
          "angle_summary.png", flush=True)
