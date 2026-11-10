"""
The MIXED SOC (Rashba + Dresselhaus 16 meV nm, theta 0.15 pi; the thesis case), a = 10, W 100, mu_C 0.7: oscillation
analysis, the best kappa = 1/2 plateau, what sets the amplitude.

PARTS (one per former script; the cells of each part are tagged [part]):
  oscillation  (was study_oscillation_mixed.py)  STUDY: the oscillation analysis of study_oscillation (a 20, pure Rashba 16) redone for the MIXED SOC at a = 10,
  plateau      (was study_plateau_mixed.py)  STUDY: the best kappa = 1/2 plateau of the MIXED SOC, as a line (2026-11-04).
  amplitude    (was study_amplitude_mixed.py)  STUDY: what sets the AMPLITUDE of eps in the mixed SOC?  (2026-11-04)

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
from study_oscillation import bloch_blocks, dip_point, box_point, zeros_of, fit_A
from study_tuning import junction
import scipy.sparse as sp

PARTS = ('oscillation', 'plateau', 'amplitude')
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


# %% [oscillation] --- 1. INFINITE CROSS-SECTION ---
if run_part('oscillation'):
    V = st.run_scan(partial(osc_point, QM), {'E_Z': EZ}, 'scharf_absOM_modes_mixed_a10')
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
