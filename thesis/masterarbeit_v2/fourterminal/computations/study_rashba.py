"""
Pure-Rashba studies of the Scharf junction (a = 20 nm unless stated): width, Delta, probe coupling, checks, finite
T, transition, SOC shift, Fermi velocities, lattice constant.

PARTS (one per former script; the cells of each part are tagged [part]):
  W            (was study_W.py)  STUDY: what the junction width W does to the Andreev spectrum and the topological phase.
  wbands       (2026-11-11)  FIGURE: bands E(k) vs W, mixed SOC -- why a wider junction loses the gap at finite k.
  delta        (was study_delta.py)  STUDY: more ABS WITHOUT widening -- a larger Delta (the ABS count goes with Delta / E_T), and
  tc           (was study_tc.py)  STUDY: can a different probe coupling tc_barr_transport replace a longer junction?
  checks       (was study_checks.py)  STUDY: checks on the Majorana resonances (pure Rashba; results in CLAUDE.md).
  T            (was study_T.py)  STUDY: the Majorana kappa resonance at finite temperature (pure Rashba, a = 20 nm, phi = pi).
  transition   (was study_transition.py)  STUDY (UNFINISHED): the topological transition with 2-3 ABS (W = 300 / 800 nm), seen in the LDOS.
  soc_shift    (was study_soc_shift.py)  STUDY: what changes if the constant SOC shift E_so = m(alpha^2 + beta^2)/2 hbar^2 is left out of the
  fermi        (was check_fermi.py)  CHECK: Fermi momenta and velocities of the ACTUAL lattice model (my_functions blocks, normal state,
  a            (was study_a.py)  STUDY: does the result converge as the lattice constant a shrinks?  (W = 100 nm, L = 2 um, pure Rashba)

Run one part:   python study_rashba.py W      (several: list them; no argument: every part, in order)
In the interactive window the cells run as usual.  run_part(part) replaces the old `if __name__ == "__main__":`
guard: the parallel workers re-import this file and must not re-run the cells.
Names that collided between parts were renamed (see each PART header); cache names are unchanged.
"""
# %% --- IMPORTS AND THE PART SELECTOR ---
import sys
import numpy as np
import matplotlib.pyplot as plt
import tools as st
from functools import partial
from my_functions.hamiltonians import model_onsite, model_hop, region_alpha

PARTS = ('W', 'wbands', 'delta', 'tc', 'checks', 'T', 'transition', 'soc_shift', 'fermi', 'a')
_SELECTED = {a for a in sys.argv[1:] if a in PARTS}


def run_part(part):
    """True for the cells of `part` when this file runs as a script: every part, or only those named on
    the command line.  False in the parallel workers (they import this file)."""
    return __name__ == "__main__" and (not _SELECTED or part in _SELECTED)


# ====================================================================================================================
# PART 'W'  (was study_W.py)
# ====================================================================================================================
# STUDY: what the junction width W does to the Andreev spectrum and the topological phase.
# (pure Rashba, a = 20 nm; results in CLAUDE.md, "SCHARF: more ABS by widening W")
#
#   1. ABS at E_Z = 0: how many, for which W.
#   2. Along phi = pi: Q, the gap at k = 0 and the gap over all k (= Scharf's Delta_top).
#   3. k_x = 0 (normal incidence): E(phi), E(E_Z) and the (E_Z, phi) phase diagram.
#   4. (heavy, optional) the ldos-style map on a wide junction, with the Q = -1 boundary.
#
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [W] --- PARAMETERS ---

W_COUNT = (100.0, 300.0, 500.0, 800.0, 1000.0)     # widths for the ABS count (cell 1)
WIDTHS = (100.0, 300.0, 800.0)                      # widths studied in detail (cells 2-3)
PHI = np.linspace(0, 2 * np.pi, 61)                 # phase grid of the k_x = 0 maps
N_EZ = 61                                           # field points of the k_x = 0 maps


# %% [W] --- 1. ABS at E_Z = 0 ---
if run_part('W'):
    PHI_PLOT = np.linspace(0, 2 * np.pi, 101)
    fig, axs = plt.subplots(1, len(W_COUNT), figsize=(3.4 * len(W_COUNT), 3.6), sharey=True)
    print("W [nm]   E_T [meV]   ABS (< 0.9 Delta)")
    for ax, W in zip(axs, W_COUNT):
        q = st.case(W)
        n = st.count_abs(q)
        print(f"{W:6.0f}   {q.E_T:9.3f}   {n:5d}")
        ax.plot(PHI_PLOT / np.pi, st.abs_spectrum(q, PHI_PLOT, n_levels=20) / q.Delta, 'k-', lw=0.8)
        ax.axhline(0.9, color='r', ls=':', lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$\phi/\pi$')
        ax.set_title(f'W = {W:.0f} nm: {n} ABS', fontsize=10)
    axs[0].set_ylabel(r'$E/\Delta$  ($E_Z$ = 0, $k$ = 0)')
    plt.tight_layout(); st.save_fig(fig, 'W_abs'); plt.show()


# %% [W] --- 2. gaps along phi = pi ---
if run_part('W'):
    fig, axs = plt.subplots(1, len(WIDTHS), figsize=(5.5 * len(WIDTHS), 3.6))
    print("\nW [nm]   Q = -1 for E_Z in [meV]   best gap at k = 0   best gap over all k   [ueV]")
    for ax, W in zip(axs, WIDTHS):
        q = st.case(W)
        line = st.phase_line(q, st.field_grid(q, 41), cache=f'scharf_absR_line_W{W:.0f}')
        topo = line['Q'] < 0
        i = st.best_topological(line)
        print(f"{W:6.0f}   {line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f}"
              f"              {line['gap0'][i] * 1e3:8.1f}            {line['gap'][i] * 1e3:8.1f}")
        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        ax.fill_between(line['E_Z'], 0, top, where=topo, color='0.88', step='mid', label='Q = -1')
        ax.plot(line['E_Z'], line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        ax.plot(line['E_Z'], line['gap'] * 1e3, 'r-', label=r'gap over all k ($\Delta_{top}$)')
        ax.set_ylim(0, top)
        ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'gap [$\mu$eV]')
        ax.set_title(f'W = {W:.0f} nm, $\\phi = \\pi$', fontsize=10)
    axs[0].legend(fontsize=7)
    plt.tight_layout(); st.save_fig(fig, 'W_gaps'); plt.show()


# %% [W] --- 3. k_x = 0: spectra and phase diagram ---
if run_part('W'):
    fig, axs = plt.subplots(len(WIDTHS), 4, figsize=(19, 3.7 * len(WIDTHS)))
    j_pi = int(np.argmin(np.abs(PHI - np.pi)))
    for row, W in zip(axs, WIDTHS):
        q = st.case(W)
        EZ = st.field_grid(q, N_EZ)
        Q = st.q_map(q, EZ, PHI, cache=f'scharf_abs_k0_qmap_W{W:.0f}')
        G0 = st.run_scan(partial(st.gap0_point, q), {'E_Z': EZ, 'phi': PHI}, f'scharf_abs_k0_gap0_W{W:.0f}')[..., 0]
        topo = Q[:, j_pi] < 0
        i_b = int(np.argmax(np.where(topo, G0[:, j_pi], -1)))          # largest k_x = 0 gap inside Q = -1
        print(f"W = {W:.0f} nm ({st.count_abs(q)} ABS): k_x = 0 gap max {G0[i_b, j_pi] * 1e3:.1f} ueV"
              f" at E_Z = {EZ[i_b]:.3f};  topological fraction of the map {np.mean(Q < 0):.2f}")

        for ax, ez, ttl in ((row[0], 0.0, 'ABS'), (row[1], EZ[i_b], 'Q = -1 at $\\phi=\\pi$')):
            ax.plot(PHI / np.pi, st.abs_spectrum(q, PHI, E_Z=ez, n_levels=20) / q.Delta, 'k-', lw=0.8)
            ax.set_ylim(-1.05, 1.05); ax.set_xlabel(r'$\phi/\pi$'); ax.set_ylabel(r'$E/\Delta$')
            ax.set_title(f'W = {W:.0f} nm, $k_x$ = 0, $E_Z$ = {ez:.3f} meV ({ttl})', fontsize=9)
        E_ez = np.array([st.levels(q, ez, np.pi, n_levels=16) for ez in EZ])
        row[2].fill_between(EZ, -1.05, 1.05, where=topo, color='0.9', step='mid')
        row[2].plot(EZ, E_ez / q.Delta, 'k-', lw=0.8)
        row[2].set_ylim(-1.05, 1.05); row[2].set_xlabel(r'$E_Z$ [meV]'); row[2].set_ylabel(r'$E/\Delta$')
        row[2].set_title(r'$k_x$ = 0, $\phi = \pi$ (grey: Q = -1)', fontsize=9)
        im = row[3].pcolormesh(EZ, PHI / np.pi, G0.T * 1e3, shading='nearest', cmap='magma')
        row[3].contour(EZ, PHI / np.pi, Q.T, levels=[0], colors='c', linestyles='--', linewidths=1)
        fig.colorbar(im, ax=row[3], label=r'gap at $k_x$ = 0 [$\mu$eV]')
        row[3].set_xlabel(r'$E_Z$ [meV]'); row[3].set_ylabel(r'$\phi/\pi$')
        row[3].set_title('gap at $k_x$ = 0 (cyan dashed: Q = -1 boundary)', fontsize=9)
    plt.tight_layout(); st.save_fig(fig, 'k0_spectra'); plt.show()


# %% [W] --- 4. (heavy) ldos map on a wide junction, with the Q = -1 boundary ---
# Takes hours at W = 800 nm and ~0.3 GB per worker.  Uses ldos's own map and plots.
if run_part('W'):
    RUN_MAP = False
    if RUN_MAP:
        import ldos as S
        W_MAP = 800.0
        PW = st.case(W_MAP)
        EZ_MAP = np.linspace(0.0, max(0.3, 2.0 * PW.E_T), 41)
        PHIS_MAP = np.linspace(0.0, 2 * np.pi, 41)
        R = S.run_map_parallel(pp=PW, phis=PHIS_MAP, b_vals=PW.B_of_E_Z(EZ_MAP), n_workers=10,
                               save=str(st.DATA / f'scharf_absW_map_rashba_W{W_MAP:.0f}.npz'))
        TOPO = (R['E_Z'], PHIS_MAP, st.q_map(PW, R['E_Z'], PHIS_MAP, cache=f'scharf_abs_qmap_rashba_W{W_MAP:.0f}'))
        S.plot_map(R, PW, topo=TOPO)
        S.plot_curvature(R, PW, topo=TOPO)
        S.plot_transport(R, PW, topo=TOPO)


# %% [wbands] --- WHY WIDTH CLOSES THE GAP AT FINITE k: bands E(k) at phi = pi, MIXED SOC (2026-11-11) ---
# Figure for the width section.  The phase lines of [W] cell 2 (pure Rashba, scharf_absR_line_W*) and of the archived
# scharf_abs.py (mixed SOC, scharf_absW_line_W*), a 20, phi pi, give Q and the gap at k = 0 per field.  Their "gap over
# all k" came from the 88-point K_GRID and MISSES the narrow dips at W >= 300 (8.4 -> 0.48 ueV, 1.7 -> 0 at the old
# best fields), so Delta_top is refined here at EVERY field (313-point grid + 1D refinement of the 3 lowest minima) and
# the best topological field is chosen from the refined values.  At that field: the bands of the infinite junction,
# k along the interfaces.  k = 0 is normal incidence: the lowest Andreev level sits ~ half a level spacing
# pi hbar v_F / (W + xi_S) from zero.  At finite k (trajectories ~ parallel to the interfaces, near the outer Fermi
# point) the lowest band dips to Delta_top.  Wider W -> more ABS, smaller spacing, deeper dip.  plots/W_bands_mixed.png.
W_BANDS = (100.0, 300.0, 800.0)
W_SUM = (100.0, 300.0, 800.0, 1000.0)
W_RASHBA = (100.0, 300.0, 800.0)
HBVF_A20, XI_S = 51.4, 51.4 / 0.25                   # hbar v_F (meV nm, lattice a 20, [fermi]) and hbar v_F / Delta (nm)
K_REF = np.r_[np.linspace(0.0, 1.5, 301), np.linspace(1.6, np.pi, 12)]    # k a at a = 20: 4x the K_GRID density
CLOSED = 1e-5                                         # meV: a refined Delta_top below 0.01 ueV counts as closed


def refined_gap_point(q, E_Z, phi=np.pi):
    """[smallest gap over all k (meV), its k (1/nm)] at a = 20: dense grid + 1D refinement of the 3 lowest minima."""
    p = q.params(E_Z=float(E_Z), phi=float(phi))
    g, k = st.gap_refined(p, st.n_S_of(q), K_REF)
    return np.array([g * q.t, k / q.a])


def band_point(q, E_Z, k, n_levels=160):
    """The n_levels BdG levels closest to zero (meV) at phi = pi, Bloch momentum k (units of 1/a)."""
    return st.levels(q, E_Z, np.pi, k, n_levels)


def refined_line(prefix, W, q):
    """Cached phase line (Q, gap at k = 0) plus Delta_top refined at every field: dict E_Z, Q, gap0, gap_grid, gap, k."""
    d = np.load(st.DATA / f'scharf_{prefix}_line_W{W:.0f}.npz')
    v, ez = d['values'][:, 0], d['axis_E_Z']
    R = st.run_scan(partial(refined_gap_point, q), {'E_Z': ez}, f'scharf_{prefix}ref_line_W{W:.0f}', chunksize=1)
    return dict(E_Z=ez, Q=v[:, 0], gap0=v[:, 1], gap_grid=v[:, 2], gap=R[:, 0], k=R[:, 1])


if run_part('wbands'):
    am, bm = st.mixed_soc()
    LN = {}
    print("\n  a 20, phi = pi.  Best topological field per W from the REFINED Delta_top (grid value at the same field):")
    print("  SOC     W [nm]  ABS  Q=-1 for E_Z     best E_Z  Delta_top refined / grid [ueV]  k [1/nm]  gap k=0 [ueV]"
          "  closed on (of Q=-1 fields)")
    for soc, prefix, Ws_ in (('mixed', 'absW', W_SUM), ('Rashba', 'absR', W_RASHBA)):
        for W in Ws_:
            q = st.case(W, alpha=am, beta=bm) if soc == 'mixed' else st.case(W)
            assert q.a == 20.0
            ln = refined_line(prefix, W, q)
            topo = ln['Q'] < 0
            i = int(np.argmax(np.where(topo, ln['gap'], -1)))
            LN[(soc, W)] = dict(ln, q=q, i=i, n=st.count_abs(q))
            print(f"  {soc:7s} {W:6.0f}  {LN[(soc, W)]['n']:3d}  {ln['E_Z'][topo].min():.3f}-{ln['E_Z'][topo].max():.3f}"
                  f"    {ln['E_Z'][i]:7.4f}   {ln['gap'][i] * 1e3:8.3f} / {ln['gap_grid'][i] * 1e3:7.2f}"
                  f"             {ln['k'][i]:.4f}    {ln['gap0'][i] * 1e3:8.1f}       "
                  f"{np.sum(topo & (ln['gap'] < CLOSED))} / {topo.sum()}", flush=True)
    BW = {}
    for W in W_SUM:
        L_ = LN[('mixed', W)]
        BW[W] = dict(q=L_['q'], ez=L_['E_Z'][L_['i']], g0=L_['gap0'][L_['i']], g=L_['gap'][L_['i']],
                     k=L_['k'][L_['i']], n=L_['n'])

    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.15), height_ratios=(1.25, 1), wspace=0.32, hspace=0.32)
    INKB, INK2B, GRIDB = '#0b0b0b', '#52514e', '#d8d7d2'
    C_TOP, C_K0, C_RB = '#eb6834', '#2a78d6', '#1baf7a'
    for c, W in enumerate(W_BANDS):
        b = BW[W]
        q, Delta_ueV = b['q'], b['q'].Delta * 1e3
        ks = np.unique(np.round(np.r_[np.linspace(0.0, 1.2, 601), b['k'] * q.a + np.linspace(-0.03, 0.03, 301)], 8))
        ks = ks[(ks >= 0) & (ks <= 1.2)]
        E = st.run_scan(partial(band_point, q, float(b['ez'])), {'k': ks},
                        f'scharf_absWbands_W{W:.0f}_EZ{b["ez"]:g}', chunksize=8) * 1e3          # ueV
        kk = ks / q.a
        for r, ylim in ((0, Delta_ueV), (1, 60.0)):
            ax = fig.add_subplot(gs[r, c])
            Em = np.where(np.abs(E) <= 1.02 * ylim, E, np.nan)
            ax.plot(kk, Em, color=INK2B, lw=0.7)
            ax.axhline(0, color=INK2B, lw=0.6, ls=':')
            ax.plot([0, 0], [-b['g0'] * 1e3, b['g0'] * 1e3], ls='none', marker='o', ms=6, color=C_K0, mec='w', zorder=5)
            ax.plot([b['k']] * 2, [-b['g'] * 1e3, b['g'] * 1e3], ls='none', marker='o', ms=6, color=C_TOP, mec='w',
                    zorder=5)
            ax.set_ylim(-ylim, ylim); ax.set_xlim(0, kk[-1])
            ax.grid(color=GRIDB, lw=0.5, alpha=0.6)
            ax.tick_params(colors=INK2B, labelcolor=INKB)
            ax.set_xlabel(r'$k$ along the interfaces (1/nm)', color=INKB)
            if r == 0:
                ax.set_title(f'W = {W:.0f} nm: {b["n"]} ABS, $E_Z$ = {b["ez"]:.3f} meV', color=INKB, fontsize=10)
                ax.set_ylabel(r'$E$ ($\mu$eV), $|E| \leq \Delta$' if c == 0 else '', color=INKB)
            else:
                gtxt = 'closed' if b['g'] < CLOSED else rf'{b["g"] * 1e3:.2g} $\mu$eV'
                ax.set_title(rf'zoom: gap at $k=0$ {b["g0"] * 1e3:.0f} $\mu$eV, over all $k$ {gtxt}',
                             color=INKB, fontsize=9)
                ax.set_ylabel(r'$E$ ($\mu$eV)' if c == 0 else '', color=INKB)
    Ws = np.array(W_SUM)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(Ws, [BW[W]['g'] * 1e3 for W in W_SUM], '-o', color=C_TOP, lw=2, ms=6, mec='w',
            label=r'gap over all $k$ ($\Delta_{top}$)')
    ax.plot(Ws, [BW[W]['g0'] * 1e3 for W in W_SUM], '-o', color=C_K0, lw=2, ms=6, mec='w', label=r'gap at $k = 0$')
    rb = [LN[('Rashba', W)] for W in W_RASHBA]
    ax.plot(W_RASHBA, [r['gap'][r['i']] * 1e3 for r in rb], '--o', color=C_TOP, lw=1.2, ms=5, mfc='w',
            label=r'$\Delta_{top}$, pure Rashba 16 (comparison)')
    for W in W_SUM:
        if BW[W]['g'] < CLOSED:
            ax.annotate('closed', (W, 0), textcoords='offset points', xytext=(-14, -12), fontsize=8, color=C_TOP)
    ax.axhline(0, color=INK2B, lw=0.6)
    ax.set_ylim(-16, None); ax.set_xlabel('W (nm)', color=INKB); ax.set_ylabel(r'gap ($\mu$eV)', color=INKB)
    ax.set_title(r'gaps at the best topological field ($\phi = \pi$)', color=INKB, fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(color=GRIDB, lw=0.5, alpha=0.6)
    ax.tick_params(colors=INK2B, labelcolor=INKB)
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(Ws, np.pi * HBVF_A20 / (Ws + XI_S) * 1e3, '-o', color=INK2B, lw=2, ms=6, mec='w')
    for W in W_SUM:
        ax.annotate(f'{BW[W]["n"]} ABS', (W, np.pi * HBVF_A20 / (W + XI_S) * 1e3), textcoords='offset points',
                    xytext=(4, 6), fontsize=8, color=INKB)
    ax.set_xlabel('W (nm)', color=INKB); ax.set_ylabel(r'level spacing $\pi\hbar v_F/(W+\xi_S)$ ($\mu$eV)', color=INKB)
    ax.set_title(r'Andreev level spacing at $k = 0$ ($E_Z$ = 0 count)', color=INKB, fontsize=10)
    ax.set_ylim(0, 620); ax.grid(color=GRIDB, lw=0.5, alpha=0.6); ax.tick_params(colors=INK2B, labelcolor=INKB)
    fig.suptitle(r'Infinite junction, mixed SOC, $\mu_C$ = 0.7 meV, $\Delta$ = 0.25 meV, a = 20 nm, $\phi = \pi$:  '
                 r'blue = gap at $k = 0$, orange = smallest gap (finite $k$)', color=INKB, fontsize=11)
    st.save_fig(fig, 'W_bands_mixed', dpi=150); plt.close(fig)
    print("figure: plots/W_bands_mixed.png", flush=True)


# ====================================================================================================================
# PART 'delta'  (was study_delta.py)   renamed: CASES -> CASES_D, tag -> tag_D
# ====================================================================================================================
# STUDY: more ABS WITHOUT widening -- a larger Delta (the ABS count goes with Delta / E_T), and
# stronger SOC.  Pure Rashba, a = 20 nm; results in CLAUDE.md ("LARGER DELTA DOES WHAT WIDENING CANNOT").
#
#   1. Per (W, Delta, alpha): ABS count, and along phi = pi Q, the gap at k = 0, the gap over all k.
#   2. kappa, G, local Andreev vs E_Z at phi = pi, L = 2 um.
#   3. 4 ABS (W = 300 nm, Delta 0.75 / 1.0 meV) at L = 4 and 8 um: does length help, as for 2 ABS?
#
# Caveat: Delta >= 0.75 meV approaches mu_S = 1 meV (strong-coupling superconductor); read as a trend.
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [delta] --- PARAMETERS ---

CASES_D = [(W, D, 16.0) for W in (100.0, 300.0) for D in (0.25, 0.5, 0.75, 1.0)] \
      + [(300.0, 0.25, 32.0), (300.0, 0.5, 32.0)]                   # (W nm, Delta meV, alpha meV nm)
CASES_LONG = [(300.0, 0.75, 4000.0), (300.0, 0.75, 8000.0), (300.0, 1.0, 4000.0)]   # (W, Delta, L)


def tag_D(W, D, alpha):
    return f'W{W:.0f}_D{D:g}_a{alpha:g}'


# %% [delta] --- 1. ABS count and gaps along phi = pi ---
if run_part('delta'):
    RES = {}
    print(" W[nm] Delta alpha  ABS   E_T   topological E_Z (phi=pi)  max gap over all k [ueV] (at E_Z)   gap/Delta")
    for W, D, al in CASES_D:
        q = st.case(W, Delta=D, alpha=al)
        line = st.phase_line(q, st.field_grid(q, 41), cache=f'scharf_absD_line_{tag_D(W, D, al)}')
        topo = line['Q'] < 0
        i = st.best_topological(line)
        rng = f"{line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f}" if topo.any() else "none"
        print(f"{W:6.0f} {D:5.2f} {al:5.1f}  {st.count_abs(q):3d}  {q.E_T:5.3f}   {rng:>16s}"
              f"          {line['gap'][i] * 1e3:8.2f} ({line['E_Z'][i]:.3f})        {line['gap'][i] / D:.3f}")
        RES[(W, D, al)] = dict(q=q, line=line)


# %% [delta] --- 2. transport at L = 2 um ---
if run_part('delta'):
    for key, r in RES.items():
        q = r['q']
        r['tr'] = st.transport_line(q, st.field_grid(q, 121), cache='scharf_absK_D3_' + tag_D(*key))
        st.print_peaks(f"W {key[0]:.0f}, Delta {key[1]:g}, alpha {key[2]:g} ({st.count_abs(q)} ABS)", r['tr'])

    fig, axs = plt.subplots(len(RES), 2, figsize=(14, 2.3 * len(RES)))
    for row, (key, r) in zip(axs, RES.items()):
        line, t = r['line'], r['tr']
        topo = line['Q'] < 0
        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        row[0].fill_between(line['E_Z'], 0, top, where=topo, color='0.88', step='mid')
        row[0].plot(line['E_Z'], line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        row[0].plot(line['E_Z'], line['gap'] * 1e3, 'r-', label='gap over all k')
        row[0].set_ylim(0, top)
        row[0].set_ylabel(f'W {key[0]:.0f}, $\\Delta$ {key[1]:g}\n$\\alpha$ {key[2]:g}: {st.count_abs(r["q"])} ABS', fontsize=8)
        tm = np.interp(t['E_Z'], line['E_Z'], topo.astype(float)) > 0.5
        row[1].fill_between(t['E_Z'], 0, 1.05, where=tm, color='0.9', step='mid')
        row[1].plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
        row[1].plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
        row[1].plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
        row[1].axhline(0.5, color='0.5', ls='--', lw=0.6)
        row[1].set_ylim(0, 1.05)
    axs[0, 0].legend(fontsize=7); axs[0, 1].legend(fontsize=7, ncol=3)
    axs[-1, 0].set_xlabel(r'$E_Z$ [meV]  ($\phi=\pi$; grey: Q = -1)  gap [$\mu$eV]')
    axs[-1, 1].set_xlabel(r'$E_Z$ [meV]  ($\phi=\pi$, L = 2 $\mu$m, E = 0)')
    plt.tight_layout(); st.save_fig(fig, 'D_delta'); plt.show()


# %% [delta] --- 3. 4 ABS in longer junctions ---
if run_part('delta'):
    fig, axs = plt.subplots(len(CASES_LONG), 1, figsize=(10, 2.6 * len(CASES_LONG)))
    for ax, (W, D, L) in zip(axs, CASES_LONG):
        q = st.case(W, L=L, Delta=D)
        t = st.transport_line(q, st.field_grid(q, 121), cache=f'scharf_absK_DL_W{W:.0f}_D{D:g}_L{L:.0f}')
        s = st.peak_stats(t)
        st.print_peaks(f"W {W:.0f}, Delta {D:g}, L {L / 1000:.0f} um (4 ABS)", t)
        print(f"      peaks > 0.3: {s['n']},  neutral: {s['neutral']},  Majorana-like: {s['good']},  above 0.55: {s['over']}")
        ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
        ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
        ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
        ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f'W {W:.0f}, $\\Delta$ {D:g}\nL {L / 1000:.0f} $\\mu$m', fontsize=8)
    axs[0].legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel(r'$E_Z$ [meV]   ($\phi = \pi$, E = 0, tc_barr = 0.2; 4 ABS)')
    plt.tight_layout(); st.save_fig(fig, 'DL_4abs'); plt.show()


# ====================================================================================================================
# PART 'tc'  (was study_tc.py)   renamed: CASES -> CASES_TC
# ====================================================================================================================
# STUDY: can a different probe coupling tc_barr_transport replace a longer junction?
# (it cannot -- results in CLAUDE.md, "THE PROBE COUPLING DOES NOT REPLACE LENGTH")
#
# kappa, G, local Andreev and det r vs E_Z at phi = pi, E = 0, for five junctions and five couplings.
# Majorana-like kappa peak: 0.45 <= kappa <= 0.55 and |G|/kappa < 0.05.
# Two-Majorana picture: kappa = 2 delta^2 Gamma^2 / (delta^2 + Gamma^2)^2, and tc only moves Gamma.
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [tc] --- PARAMETERS ---

TCS = (0.05, 0.1, 0.2, 0.5, 1.0)
CASES_TC = {'W100 L2 (1 ABS, ref)':   dict(tag='W100_L2', W=100.0, L=2000.0),
         'W300 L2 (2 ABS)':        dict(tag='W300_L2', W=300.0, L=2000.0),
         'W300 L4 (2 ABS, works)': dict(tag='W300_L4', W=300.0, L=4000.0),
         'W800 L4 (3 ABS)':        dict(tag='W800_L4', W=800.0, L=4000.0),
         'W300 D0.75 L4 (4 ABS)':  dict(tag='W300_D0.75_L4', W=300.0, L=4000.0, Delta=0.75)}


# %% [tc] --- RUN ---
if run_part('tc'):
    RES = {}
    print("case                      tc    peaks>0.3  Majorana-like  kappa>0.55  max kappa  median |G|/kappa"
          "  frac(det r < 0)")
    for name, c in CASES_TC.items():
        for tc in TCS:
            q = st.case(c['W'], L=c['L'], Delta=c.get('Delta', 0.25), tc=tc)
            t = st.transport_line(q, st.field_grid(q, 121), cache=f"scharf_absTC_{c['tag']}_tc{tc:g}")
            RES[(name, tc)] = t
            s = st.peak_stats(t)
            print(f"{name:24s} {tc:5.2f}   {s['n']:6d}     {s['good']:6d}        {s['over']:6d}     {s['kmax']:6.3f}"
                  f"      {s['rmed']:7.3f}          {s['det_neg']:.2f}")


# %% [tc] --- FIGURE: cases as columns, couplings as rows ---
if run_part('tc'):
    fig, axs = plt.subplots(len(TCS), len(CASES_TC), figsize=(4.2 * len(CASES_TC), 2.2 * len(TCS)), sharex='col')
    for j, name in enumerate(CASES_TC):
        for i, tc in enumerate(TCS):
            ax, t = axs[i, j], RES[(name, tc)]
            ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
            ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
            ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
            ax.fill_between(t['E_Z'], 0, 0.06, where=t['det'] < 0, color='g', alpha=0.4, step='mid', lw=0)
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'tc_barr = {tc:g}', fontsize=9)
        axs[-1, j].set_xlabel(r'$E_Z$ [meV]')
    axs[0, 0].legend(fontsize=6, loc='upper right')
    fig.suptitle(r'$\phi = \pi$, E = 0, pure Rashba, a = 20 nm; green bar: det r < 0', fontsize=9)
    plt.tight_layout(); st.save_fig(fig, 'TC_scan'); plt.show()


# ====================================================================================================================
# PART 'checks'  (was study_checks.py)
# ====================================================================================================================
# STUDY: checks on the Majorana resonances (pure Rashba; results in CLAUDE.md).
#
#   1. Lattice constant a = 20 / 10 / 5 nm at fixed physical W: ABS count, Q and the gaps (cross-section).
#   2. W = 300 nm, L = 4 um: fine scan through a double peak + Majorana splitting from the end LDOS.
#      The dip between the two kappa ~ 1/2 peaks is where the splitting delta crosses zero.
#   3. W = 300 nm, L = 4 um: a = 20 vs 10 nm in transport.
#   4. Longer junctions: W = 300 nm at L = 8 um, W = 800 nm (3 ABS) at L = 4 and 8 um.
#
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [checks] --- PARAMETERS ---


def plot_lines(rows, labels, name):
    """kappa, |G| and local Andreev vs E_Z, one panel per transport line."""
    fig, axs = plt.subplots(len(rows), 1, figsize=(10, 2.3 * len(rows)))
    for ax, t, lab in zip(np.atleast_1d(axs), rows, labels):
        ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
        ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
        ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
        ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
        ax.set_ylim(0, 1.05); ax.set_ylabel(lab, fontsize=8)
    np.atleast_1d(axs)[0].legend(fontsize=7, ncol=3)
    np.atleast_1d(axs)[-1].set_xlabel(r'$E_Z$ [meV]   ($\phi = \pi$, E = 0, tc_barr = 0.2)')
    plt.tight_layout(); st.save_fig(fig, name); plt.show()


# %% [checks] --- 1. lattice constant, cross-section only ---
if run_part('checks'):
    print("  W[nm]  a[nm]  ABS  E_Z[meV]   Q   gap(k=0)[ueV]  gap(all k)[ueV]  at k[1/nm]")
    for W, EZs in ((100.0, (0.3, 0.593)), (300.0, (0.15, 0.323))):
        for a in (20.0, 10.0, 5.0):
            q = st.case(W, a=a)
            for EZ in EZs:
                Q, g0, g, km = st.phase_point(q, EZ, np.pi, ks=st.k_grid(a))
                print(f"  {W:5.0f}  {a:5.0f}  {st.count_abs(q):3d}  {EZ:7.3f}  {Q:+3.0f}  {g0 * 1e3:12.2f}"
                      f"  {g * 1e3:14.2f}   {km / a:9.4f}", flush=True)


# %% [checks] --- 2. the double peak at W = 300 nm, L = 4 um ---
if run_part('checks'):
    q4 = st.case(300.0, L=4000.0)
    EZ1 = np.round(np.linspace(0.12, 0.16, 41), 5)
    K1 = st.transport_line(q4, EZ1, cache='scharf_absK_C1_W300_L4000_a20')
    i_d = int(np.argmin(K1['kappa'][5:-5])) + 5                                   # the dip
    i_l = int(np.argmax(np.where(EZ1 < EZ1[i_d], K1['kappa'], -1)))              # left peak
    i_r = int(np.argmax(np.where(EZ1 > EZ1[i_d], K1['kappa'], -1)))              # right peak
    print("    E_Z [meV]   kappa    |G|      Andreev   delta/2 from end LDOS [ueV]")
    SPL = {}
    for tag, i in (('left peak', i_l), ('dip', i_d), ('right peak', i_r)):
        d2, E, end = st.splitting(q4, EZ1[i])
        SPL[tag] = (EZ1[i], E, end)
        print(f"    {EZ1[i]:8.4f}  {K1['kappa'][i]:7.4f}  {abs(K1['G'][i]):7.4f}  {K1['andreev'][i]:7.4f}"
              f"   {d2:8.3f}    ({tag})", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(EZ1, K1['kappa'], 'k.-', label=r'$\kappa/\kappa_0$')
    axs[0].plot(EZ1, np.abs(K1['G']), 'r.-', label=r'$|G|/G_0$')
    axs[0].plot(EZ1, K1['andreev'] / 2, 'b.-', label=r'Andreev $/\,2e^2/h$')
    axs[0].axhline(0.5, color='0.5', ls='--', lw=0.6)
    axs[0].set_xlabel(r'$E_Z$ [meV]'); axs[0].legend(fontsize=8)
    axs[0].set_title('W = 300 nm, L = 4 $\\mu$m, $\\phi = \\pi$, E = 0', fontsize=9)
    for tag, (ez, E, end) in SPL.items():
        axs[1].plot(E * 1e3, end, label=f'{tag}, $E_Z$ = {ez:.4f}')
    axs[1].set_xlabel(r'$E$ [$\mu$eV]'); axs[1].set_ylabel(r'end LDOS, $\eta$ = 0.2 $\mu$eV')
    axs[1].legend(fontsize=8)
    plt.tight_layout(); st.save_fig(fig, 'K_C1'); plt.show()


# %% [checks] --- 3. and 4. a = 10 vs 20 nm, and longer junctions ---
if run_part('checks'):
    rows, labels = [], []
    for a in (20.0, 10.0):
        q = st.case(300.0, L=4000.0, a=a)
        t = st.transport_line(q, st.field_grid(q, 121), cache=f'scharf_absK_C2_W300_L4000_a{a:.0f}')
        st.print_peaks(f"W = 300 nm, L = 4 um, a = {a:.0f} nm", t)
        rows.append(t); labels.append(f'W 300, L 4 um, a {a:.0f}')
    for W, L in ((300.0, 8000.0), (800.0, 4000.0), (800.0, 8000.0)):
        q = st.case(W, L=L)
        t = st.transport_line(q, st.field_grid(q, 121), cache=f'scharf_absK_C34_W{W:.0f}_L{L:.0f}_a20')
        st.print_peaks(f"W = {W:.0f} nm, L = {L / 1000:.0f} um", t)
        rows.append(t); labels.append(f'W {W:.0f}, L {L / 1000:.0f} um')
    plot_lines(rows, labels, 'K_C234')


# ====================================================================================================================
# PART 'T'  (was study_T.py)
# ====================================================================================================================
# STUDY: the Majorana kappa resonance at finite temperature (pure Rashba, a = 20 nm, phi = pi).
# Results in CLAUDE.md, Measurability section ("PURE-RASHBA CASES ..." and "THE COUPLING HAS A
# TEMPERATURE OPTIMUM").
#
#   1. The best T = 0 neutral peaks of three clean junctions, heated to 0-50 mK (tc_barr = 0.2).
#   2. Full fine E_Z scans at several couplings tc_barr and T = 0 / 10 / 20 / 50 mK.
#
# Method: the six channels are computed on a fixed energy grid E >= 0 (the expensive part, cached),
# then st.thermal() does the thermal average for any T.  The signal survives while kT <~ Gamma ~ delta.
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [T] --- PARAMETERS ---

T_MK = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)       # temperatures of cell 1
E_PEAK = np.unique(np.concatenate([np.linspace(0.0, 3e-3, 61), np.linspace(3e-3, 20e-3, 35),
                                   np.geomspace(20e-3, 0.1, 10)]))             # meV, cell 1
E_SCAN = np.unique(np.concatenate([np.linspace(0.0, 3e-3, 31), np.linspace(3e-3, 35e-3, 33)]))   # meV, cell 2

# cell 1: junction -> (parameters, cached T = 0 transport line to pick the peaks from)
PEAK_CASES = {'W100 L2 (1 ABS)':      (st.case(100.0), 'scharf_absTC_W100_L2_tc0.2'),
              'W100 L2 D0.5 (2 ABS)': (st.case(100.0, Delta=0.5), 'scharf_absK_D3_W100_D0.5_a16'),
              'W300 L4 (2 ABS)':      (st.case(300.0, L=4000.0), 'scharf_absTC_W300_L4_tc0.2')}

# cell 2: junction -> (parameters, field range, couplings)
SCAN_CASES = {'W100_L2':      (st.case(100.0), np.round(np.arange(0.55, 1.25 + 1e-9, 0.0025), 5),
                               (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0)),
              'W100_L2_D0.5': (st.case(100.0, Delta=0.5), np.round(np.arange(0.3, 1.0 + 1e-9, 0.0025), 5),
                               (0.2,)),           # 0.3-0.5 not run yet (stopped 2026-10-21)
              # mixed Rashba + Dresselhaus, same window: is pure Rashba's temperature robustness the SOC?
              'W100_L2_mixed': (st.case(100.0, alpha=st.mixed_soc()[0], beta=st.mixed_soc()[1]),
                                np.round(np.arange(0.55, 1.25 + 1e-9, 0.0025), 5), (0.2, 0.3, 0.4, 0.7)),
              # (0.7 added 2026-11-11, computed by study_mixed [deck] cell 2: the 'too strong' slide-10 panel)
              # probe-lead chemical potential: the 5-site lead strip has subbands from 0.67 / 2.49 / 4.98 meV,
              # so mu_N = 1-2 is a single-channel probe, 3.0 (default) two channels, 5.5 three
              **{f'W100_L2_muN{m:g}': (st.replace(st.case(100.0), mu_N=m),
                                      np.round(np.arange(0.55, 1.25 + 1e-9, 0.005), 5), (0.3, 0.4))
                 for m in (1.0, 2.0, 5.5)},
              # the longer 2-ABS junction: narrow resonances, but kappa(phi = 0) ~ 0 (a clean reference)
              'W300_L4':       (st.case(300.0, L=4000.0), np.round(np.arange(0.1, 0.42 + 1e-9, 0.0025), 5),
                                (0.2, 0.3, 0.4))}
T_SCAN = (0.0, 10.0, 20.0, 50.0)


def best_neutral_peaks(t, n=2):
    """Indices of the n highest Majorana-like kappa peaks of a transport line."""
    pk = st.kappa_peaks(t)
    good = pk[st.is_majorana_like(t, pk)]
    return good[np.argsort(t['kappa'][good])[::-1][:n]]


# %% [T] --- 1. the best T = 0 peaks, heated ---
if run_part('T'):
    for name, (q, cache) in PEAK_CASES.items():
        d = np.load(st.DATA / f'{cache}.npz')
        t0 = dict(E_Z=d['axis_E_Z'], kappa=d['values'][:, 0], G=d['values'][:, 1])
        step = t0['E_Z'][1] - t0['E_Z'][0]
        for i in best_neutral_peaks(t0):
            tag = f"{name.split(' (')[0].replace(' ', '_')}_EZ{t0['E_Z'][i]:.4f}"
            # re-locate the peak on a fine E = 0 scan, then a window of 7 fields around it
            EZ_fine = np.round(np.linspace(t0['E_Z'][i] - 1.5 * step, t0['E_Z'][i] + 1.5 * step, 41), 6)
            fine = st.transport_line(q, EZ_fine, cache=f'scharf_absK_T_fine_{tag}')
            j = int(np.argmax(np.where(np.abs(fine['G']) < 0.05 * fine['kappa'], fine['kappa'], -1)))
            hw = max(3 * (EZ_fine[1] - EZ_fine[0]), 0.25 * step)
            win = np.round(np.linspace(EZ_fine[j] - hw, EZ_fine[j] + hw, 7), 6)
            V = st.run_scan(partial(st.channel_point, q, phis=(np.pi, 0.0)), {'E_Z': win, 'E': E_PEAK},
                            cache=f'scharf_absTT_{tag}')                  # (7, n_E, 12): phi = pi, then 0
            print(f"\n{name}, peak at E_Z = {EZ_fine[j]:.4f} meV")
            print("   T [mK]   best kappa(pi)  at E_Z     G(pi)    kappa(0)   Andreev(pi)")
            for T in T_MK:
                r_pi = st.thermal_line(V, st.kT_of_mK(T), E_PEAK, phase=0)
                r_0 = st.thermal_line(V, st.kT_of_mK(T), E_PEAK, phase=1)
                a = int(np.argmax(r_pi[:, 0]))
                print(f"   {T:6.1f}   {r_pi[a, 0]:9.4f}     {win[a]:.4f}   {r_pi[a, 1]:8.4f}  {r_0[a, 0]:8.4f}"
                      f"   {r_pi[a, 2]:8.4f}")


# %% [T] --- 2. couplings at finite temperature, full fine field scans ---
if run_part('T'):
    for case_name, (q0, EZ, tcs) in SCAN_CASES.items():
        print(f"\n{case_name}\n  tc    T[mK]  best kappa (|G|/kappa)   best NEUTRAL kappa at E_Z"
              "   E_Z range neutral kappa > 0.25   best kappa with |G|/kappa < 0.1")
        RES = {}
        for tc in tcs:
            q = st.replace(q0, tc_barr_transport=tc)
            V = st.run_scan(partial(st.channel_point, q), {'E_Z': EZ, 'E': E_SCAN},
                            cache=f'scharf_absTtc_{case_name}_tc{tc:g}')      # (n_EZ, n_E, 6)
            RES[tc] = {T: st.thermal_line(V, st.kT_of_mK(T), E_SCAN) for T in T_SCAN}
            for T in T_SCAN:
                k, G = RES[tc][T][:, 0], RES[tc][T][:, 1]
                ratio = np.abs(G) / np.maximum(k, 1e-12)
                i = int(np.argmax(k))
                j = int(np.argmax(np.where(ratio < 0.05, k, -1)))
                l = int(np.argmax(np.where(ratio < 0.1, k, -1)))
                span = ((ratio < 0.05) & (k > 0.25)).sum() * (EZ[1] - EZ[0])
                print(f"  {tc:4.2f}  {T:5.1f}   {k[i]:6.3f} ({ratio[i]:5.3f})         {k[j]:6.3f} at {EZ[j]:.4f}"
                      f"          {span:.4f} meV              {k[l]:6.3f} at {EZ[l]:.4f}")

        fig, axs = plt.subplots(len(tcs), 1, figsize=(11, 2.2 * len(tcs)), sharex=True, squeeze=False)
        for ax, tc in zip(axs[:, 0], tcs):
            for T, col in zip(T_SCAN, ('k', 'tab:orange', 'tab:red', 'tab:purple')):
                ax.plot(EZ, RES[tc][T][:, 0], '-', color=col, lw=0.9, label=rf'$\kappa/\kappa_0$, T = {T:g} mK')
                ax.plot(EZ, np.abs(RES[tc][T][:, 1]), ':', color=col, lw=0.8)
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05); ax.set_ylabel(f'tc_barr = {tc:g}', fontsize=9)
        axs[0, 0].legend(fontsize=7, ncol=4)
        axs[-1, 0].set_xlabel(rf'$E_Z$ [meV]   ({case_name}, $\phi=\pi$; dotted: $|G|/G_0$)')
        plt.tight_layout(); st.save_fig(fig, f'T_tc_{case_name}'); plt.show()


# ====================================================================================================================
# PART 'transition'  (was study_transition.py)   renamed: WIDTHS -> WIDTHS_TR
# ====================================================================================================================
# STUDY (UNFINISHED): the topological transition with 2-3 ABS (W = 300 / 800 nm), seen in the LDOS.
# Pure Rashba, a = 20 nm, L = 2 um, phi = pi, probes decoupled (tc_barr = 0) and eta = 0.05 Delta (Scharf).
#
#   1. LDOS(E, E_Z) at the junction END and in the MIDDLE: the gap closes at the transition;
#      a zero-energy peak only at the end = end Majoranas, at both = bulk low-energy states.
#   2. Real-space LDOS(E = 0) at four fields: trivial / gap closing / best-gapped topological / high field.
#
# Status: W = 300 nm finished; W = 800 nm partly cached (resumes where it stopped).
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [transition] --- PARAMETERS ---

WIDTHS_TR = (300.0, 800.0)
E_LDOS = np.linspace(-1.0, 1.0, 81)               # energies in units of Delta
ETA = 0.05                                        # LDOS broadening in units of Delta (Scharf)
EVERY = {300.0: 1, 800.0: 4}                      # use every n-th column for the real-space maps (memory)
N_WORKERS = 10                                    # LocalGreen at W = 800 needs ~0.3 GB per worker


# %% [transition] --- 1. LDOS(E, E_Z) at the end and in the middle ---
if run_part('transition'):
    RES = {}
    for W in WIDTHS_TR:
        q = st.case(W)
        line = st.phase_line(q, st.field_grid(q, 41), cache=f'scharf_absR_line_W{W:.0f}')
        EZ, topo = line['E_Z'], line['Q'] < 0
        spec = st.run_scan(partial(st.spectrum_point, q, E_meV=E_LDOS * q.Delta, eta=ETA * q.Delta),
                           {'E_Z': EZ}, cache=f'scharf_absT_spec_W{W:.0f}', n_workers=N_WORKERS)   # (n_EZ, 2, n_E)
        RES[W] = dict(q=q, line=line, spec=spec)
        print(f"W = {W:.0f} nm ({st.count_abs(q)} ABS): Q = -1 for E_Z in {EZ[topo].min():.3f}..{EZ[topo].max():.3f} meV")

    fig, axs = plt.subplots(len(RES), 3, figsize=(16, 3.8 * len(RES)), squeeze=False)
    for row, (W, r) in zip(axs, RES.items()):
        line, spec = r['line'], r['spec']
        EZ, topo = line['E_Z'], line['Q'] < 0
        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        row[0].fill_between(EZ, 0, top, where=topo, color='0.88', step='mid', label='Q = -1')
        row[0].plot(EZ, line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        row[0].plot(EZ, line['gap'] * 1e3, 'r-', label='gap over all k')
        row[0].set_ylim(0, top); row[0].set_xlabel(r'$E_Z$ [meV]'); row[0].set_ylabel(r'gap [$\mu$eV]')
        row[0].legend(fontsize=7); row[0].set_title(f'W = {W:.0f} nm, $\\phi = \\pi$', fontsize=9)
        for ax, k, where in ((row[1], 0, 'end'), (row[2], 1, 'middle')):
            im = ax.pcolormesh(EZ, E_LDOS, spec[:, k, :].T, shading='nearest', cmap='cividis')
            fig.colorbar(im, ax=ax, label='LDOS')
            for e in (EZ[topo].min(), EZ[topo].max()):
                ax.axvline(e, color='w', ls='--', lw=0.8)
            ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$E/\Delta$')
            ax.set_title(f'LDOS at the {where}; dashed: Q = -1 edges', fontsize=9)
    plt.tight_layout(); st.save_fig(fig, 'T_transition'); plt.show()


# %% [transition] --- 2. real-space LDOS(E = 0) at four fields ---
if run_part('transition'):
    fig, axs = plt.subplots(len(RES), 5, figsize=(20, 4.2 * len(RES)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.6]))
    for row, (W, r) in zip(axs, RES.items()):
        q, line = r['q'], r['line']
        EZ, topo = line['E_Z'], line['Q'] < 0
        i_t = int(np.argmax(topo))                                    # first topological field
        picks = {'trivial': max(i_t - 3, 0), 'gap closing': max(i_t - 1, 0),
                 'best topological': st.best_topological(line), 'high field': len(EZ) - 4}
        for ax, (lab, i) in zip(row[:4], picks.items()):
            y, x, A = st.ldos_map_E0(q, EZ[i], ETA * q.Delta, every=EVERY[W])
            ax.pcolormesh(x, y, A, shading='nearest', cmap='cividis')
            ax.set_title(f'W {W:.0f}: {lab}\n$E_Z$ = {EZ[i]:.3f} meV', fontsize=9)
            ax.set_xlabel('x [nm]')
            row[4].plot(y, A.sum(1) / A.sum(1).max(), label=f'{lab} ({EZ[i]:.3f})')
        row[0].set_ylabel('y [nm] (along)')
        row[4].set_xlabel('y [nm]'); row[4].set_ylabel(r'$\sum_x$ LDOS(E=0) (norm.)')
        row[4].legend(fontsize=7)
    plt.tight_layout(); st.save_fig(fig, 'T_ldos_E0'); plt.show()


# ====================================================================================================================
# PART 'soc_shift'  (was study_soc_shift.py)   renamed: CASES -> CASES_SOC
# ====================================================================================================================
# STUDY: what changes if the constant SOC shift E_so = m(alpha^2 + beta^2)/2 hbar^2 is left out of the
# onsite energy (my_functions.hamiltonians.onsite_block adds it, so mu is measured from the bottom of the
# lower SOC band).  Every region has the same alpha, so dropping the shift == raising mu_S, mu_N and mu_C
# by E_so.  Both conventions on the SAME field grid (a = 20 nm, L = 2 um, tc_barr = 0.2, phi = pi):
# the gaps along phi = pi and kappa / G at E = 0.

# %% [soc_shift] --- PARAMETERS ---

HB2M = 76.1996 / 0.038                                   # hbar^2 / m, meV nm^2
CASES_SOC = {'W100 a16': dict(W=100.0, alpha=16.0), 'W100 a32': dict(W=100.0, alpha=32.0),
         'W300 a16': dict(W=300.0, alpha=16.0)}


def no_shift(q):
    """The same junction without the constant SOC shift: every mu raised by E_so."""
    E_so = (q.alpha ** 2 + q.beta ** 2) / (2 * HB2M)
    return st.replace(q, mu_S=q.mu_S + E_so, mu_N=q.mu_N + E_so, mu_C=q.mu_centre + E_so)


# %% [soc_shift] --- RUN ---
if run_part('soc_shift'):
    RES = {}
    for name, c in CASES_SOC.items():
        q0 = st.case(c['W'], alpha=c['alpha'])
        EZ_line, EZ_tr = st.field_grid(q0, 41), st.field_grid(q0, 121)     # same grids for both conventions
        tag = name.replace(' ', '_')
        E_so = c['alpha'] ** 2 / (2 * HB2M)
        print(f"\n{name}: E_so = {E_so * 1e3:.0f} ueV;  v_F(junction) with shift {np.sqrt(2 * q0.mu_centre * HB2M):.1f},"
              f" without {np.sqrt(2 * (q0.mu_centre + E_so) * HB2M):.1f} meV nm")
        for conv, q in (('with shift', q0), ('no shift', no_shift(q0))):
            line = st.phase_line(q, EZ_line, cache=f'scharf_absS_line_{tag}_{conv.replace(" ", "")}')
            t = st.transport_line(q, EZ_tr, cache=f'scharf_absS_tr_{tag}_{conv.replace(" ", "")}')
            RES[(name, conv)] = (line, t)
            topo = line['Q'] < 0
            i = st.best_topological(line)
            pk = st.kappa_peaks(t)
            s = st.peak_stats(t)
            print(f"  {conv:10s}: ABS {st.count_abs(q)}, Q = -1 for {line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f} meV,"
                  f" Delta_top {line['gap'][i] * 1e3:.1f} ueV at {line['E_Z'][i]:.3f};  Majorana-like {s['good']}/{s['n']}")
            print(f"              kappa peaks at {np.round(t['E_Z'][pk], 3)}")
            print(f"              heights {np.round(t['kappa'][pk], 3)},  |G|/kappa {np.round(np.abs(t['G'][pk]) / t['kappa'][pk], 3)}")


# %% [soc_shift] --- FIGURE ---
if run_part('soc_shift'):
    fig, axs = plt.subplots(len(CASES_SOC), 2, figsize=(14, 3 * len(CASES_SOC)))
    for row, name in zip(axs, CASES_SOC):
        for conv, ls in (('with shift', '-'), ('no shift', '--')):
            line, t = RES[(name, conv)]
            row[0].plot(line['E_Z'], line['gap'] * 1e3, ls, label=f'Delta_top, {conv}')
            row[1].plot(t['E_Z'], t['kappa'], ls, lw=1, label=rf'$\kappa$, {conv}')
        row[0].set_ylabel(f'{name}\ngap [$\\mu$eV]'); row[1].set_ylabel(r'$\kappa/\kappa_0$')
        row[0].legend(fontsize=7); row[1].legend(fontsize=7); row[1].set_ylim(0, 1.05)
    axs[-1, 0].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[-1, 1].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$, E = 0)')
    plt.tight_layout(); st.save_fig(fig, 'soc_shift'); plt.show()


# ====================================================================================================================
# PART 'fermi'  (was check_fermi.py)
# ====================================================================================================================
# CHECK: Fermi momenta and velocities of the ACTUAL lattice model (my_functions blocks, normal state,
# E_Z = 0, Delta = 0) against the two continuum formulas:
#   with shift (code, Scharf):  k_F = k_mu -+ k_so,                 hbar v_F = (hbar^2/m) k_mu
#   without shift:              k_F = sqrt(k_mu^2 + k_so^2) -+ k_so, hbar v_F = (hbar^2/m) sqrt(k_mu^2 + k_so^2)
# with k_mu = sqrt(2 m mu)/hbar, k_so = m alpha / hbar^2.  Bulk of the junction material (mu_C), k along
# the junction (package x) and across it (package y); a -> 0 should approach the continuum.

HB2M = 76.1996 / 0.038                         # hbar^2 / m, meV nm^2


def bands(q, ka, direction):
    """Electron bands (meV) of the bulk junction material at momenta ka along 'x' or 'y'."""
    p = q.params(E_Z=0.0, phi=0.0)
    ons = model_onsite(p, 'c')[:2, :2]                                  # electron block (c_up, c_dn)
    V = model_hop(p, direction, p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))[:2, :2]
    Vo = model_hop(p, 'y' if direction == 'x' else 'x', p.t_c, p.m0_c, alpha=region_alpha(p, 'c'))[:2, :2]
    ph = np.exp(1j * np.asarray(ka))[:, None, None]
    H = ons + V * ph + V.conj().T / ph + Vo + Vo.conj().T              # (n_k, 2, 2); other momentum = 0
    return np.linalg.eigvalsh(H) * q.t


def fermi_points(q, direction):
    """[(k_F in 1/nm, hbar v_F in meV nm)] of both bands, from the zero crossings (linear interpolation
    between grid points; slope = dE/d(ka) * a)."""
    ka = np.linspace(1e-6, min(np.pi, 0.1 * q.a), 200001)              # k up to 0.1 1/nm (k_F ~ 0.035)
    E = bands(q, ka, direction)
    res = []
    for b in (0, 1):
        i = np.flatnonzero(np.diff(np.sign(E[:, b])))[0]
        kF = ka[i] - E[i, b] * (ka[i + 1] - ka[i]) / (E[i + 1, b] - E[i, b])
        res.append((kF / q.a, (E[i + 1, b] - E[i, b]) / (ka[i + 1] - ka[i]) * q.a))
    return sorted(res)


if run_part('fermi'):
    for alpha in (16.0, 32.0):
        q0 = st.case(100.0, alpha=alpha)
        mu = q0.mu_centre
        k_mu, k_so = np.sqrt(2 * mu / HB2M), alpha / HB2M
        A = (k_mu - k_so, k_mu + k_so, HB2M * k_mu)
        B = (np.hypot(k_mu, k_so) - k_so, np.hypot(k_mu, k_so) + k_so, HB2M * np.hypot(k_mu, k_so))
        print(f"\nalpha = {alpha:g} meV nm, mu_C = {mu} meV")
        print(f"  formula WITH shift   : k_F = {A[0]:.5f} / {A[1]:.5f} 1/nm,  hbar v_F = {A[2]:.2f} meV nm")
        print(f"  formula WITHOUT shift: k_F = {B[0]:.5f} / {B[1]:.5f} 1/nm,  hbar v_F = {B[2]:.2f} meV nm")
        print("   a [nm]  dir   k_F inner / outer [1/nm]    hbar v_F inner / outer [meV nm]")
        for a in (20.0, 10.0, 5.0, 2.5, 1.0):
            q = st.case(100.0, alpha=alpha, a=a)
            for d in ('x', 'y'):
                (k1, v1), (k2, v2) = fermi_points(q, d)
                print(f"   {a:5.1f}   {d}    {k1:.5f} / {k2:.5f}          {v1:6.2f} / {v2:6.2f}")


# ====================================================================================================================
# PART 'a'  (was study_a.py)
# ====================================================================================================================
# STUDY: does the result converge as the lattice constant a shrinks?  (W = 100 nm, L = 2 um, pure Rashba)
#
#   1. Scaling check: every lattice parameter maps back to the same physical value for each a.
#   2. Why tc_barr cannot be used: a weakened bond's transmission falls like a^2 (1D model);
#      a physical barrier (height V_barr over d_barr nm) converges.
#   3. Cross-section: ABS count, gap at k = 0 and Delta_top vs a.
#   4. Transport (phi = pi, E = 0) with a physical barrier, tc_barr_transport = 1: kappa, G, Andreev
#      vs E_Z for each a -- peak heights, neutrality and positions.
#
# Expectation: heights and neutrality converge quickly; peak POSITIONS converge like a^2, because the
# splitting oscillates as cos(k_F L) and the lattice k_F differs from the continuum one by ~(k_F a)^2/24.
# Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
# The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
# (they re-import this file); in the interactive window the cells run normally.

# %% [a] --- PARAMETERS ---

A_LIST = (20.0, 10.0, 5.0, 4.0, 2.5)                       # nm
BARRIERS = {'medium': (6.0, 20.0), 'weak': (6.0, 40.0)}      # (V_barr meV, d_barr nm); 1D T ~ 0.30 / 0.04
A_TRANSPORT = {'medium': (20.0, 10.0, 5.0, 4.0), 'weak': (20.0, 10.0, 5.0, 4.0)}
# a = 2.5 nm: ~500 s per point and several GB per worker (14 workers ran out of memory) -> cell 5,
# a narrow window around the neutral peaks with 4 workers
EZ_FINE = np.round(np.arange(0.77, 0.93 + 1e-9, 0.002), 5)
N_WORKERS_SMALL_A = 4
EZ_TR = np.round(np.arange(0.55, 1.25 + 1e-9, 0.002), 5)    # meV, the resonance region of W = 100 nm
EZ_XS = (0.3, 0.593)                                         # meV, cross-section points


SITE_BARRIERS = {'site120': 120.0, 'site240': 240.0}   # lam_barr meV nm: ONE site of lam/a; 1D T ~ 0.28 / 0.09
A_SITE = (20.0, 10.0, 5.0, 4.0)


def junction(a, barrier=None):
    """W = 100 nm, L = 2 um at lattice constant a, optionally with a physical probe barrier
    (square: 'medium'/'weak', or one-site: 'site120'/'site240')."""
    q = st.case(100.0, a=a, tc=1.0 if barrier else 0.2)
    if barrier in SITE_BARRIERS:
        q = st.replace(q, lam_barr=SITE_BARRIERS[barrier])
    elif barrier:
        q = st.replace(q, V_barr=BARRIERS[barrier][0], d_barr=BARRIERS[barrier][1])
    return q


def contact_1d(a, tau=1.0, V_b=0.0, d_b=0.0, mu_N=3.0, mu_C=0.7, n_pad=4):
    """Transmission at E = 0 of a 1D chain: lead (mu_N) | weak bond tau or barrier V_b over d_b | junction (mu_C)."""
    t = 76.1996 / 0.038 / (2 * a * a)                                    # hbar^2 / (2 m a^2), meV
    nb = int(round(d_b / a))
    mus = [mu_N] * n_pad + [mu_N - V_b] * nb + [mu_C] * n_pad
    H = np.diag([2 * t - m for m in mus]).astype(complex)
    for j in range(len(mus) - 1):
        H[j, j + 1] = H[j + 1, j] = -t
    if tau != 1.0:
        j = n_pad + nb - 1
        H[j, j + 1] = H[j + 1, j] = -tau * t
    sig = lambda mu: -t * np.exp(1j * np.arccos(1 - mu / (2 * t)))      # semi-infinite chain, E = 0
    S = np.zeros_like(H)
    S[0, 0], S[-1, -1] = sig(mu_N), sig(mu_C)
    G = np.linalg.inv(-H - S)
    return 4 * S[0, 0].imag * S[-1, -1].imag * abs(G[0, -1]) ** 2


def contact_scattering(q, E_Z, phi=np.pi, normal=False):
    """
    The LEFT probe's reflection matrix at E = 0 from the full Green's function, split into electron and
    hole blocks: r_ab = -delta_ab + i Gamma_a^1/2 G_LL[a, b] Gamma_b^1/2 on the open channels of each block.
    normal=True sets Delta = 0 everywhere (the ribbons become normal leads): the contact's normal-state
    scattering, everything else (geometry, SOC, field, barrier, all four terminals) unchanged.
    Returns dict: n_e open electron channels, R_ee / R_he = Tr r^dag r for an incoming electron (normal /
    Andreev reflection), T_out = n_e - R_ee - R_he (electron flux leaving through the OTHER terminals),
    tau = eigenvalues of 1 - r_ee^dag r_ee (per channel, before Andreev), n_bad failed lead solves.
    """
    n0 = st.myf.leads.N_UNRELIABLE
    p = q.params_transport(E_Z=float(E_Z))
    if normal:
        p = st.replace(p, delta=0.0)
    rgf = st.myf.RGFFourTerminal(st.myf.FourTerminalJunction(p), np.array([0.0]))
    G = rgf.green_blocks(float(phi))[0][0]
    Gam = rgf.Gamma_L[0]
    S, U = {}, {}
    for a, idx in (('e', rgf.e_idx), ('h', rgf.h_idx)):
        w, V = np.linalg.eigh(Gam[np.ix_(idx, idx)])
        keep = w > 1e-4 * w.max()                # open channels; evanescent ones have Gamma ~ eta
        S[a], U[a] = np.sqrt(w[keep]), V[:, keep]
    idx = {'e': rgf.e_idx, 'h': rgf.h_idx}
    r = lambda a, b: (-np.eye(len(S[a])) * (a == b)
                      + 1j * S[a][:, None] * (U[a].conj().T @ G[np.ix_(idx[a], idx[b])] @ U[b]) * S[b][None, :])
    r_ee, r_he = r('e', 'e'), r('h', 'e')
    R_ee, R_he = np.sum(np.abs(r_ee)**2), np.sum(np.abs(r_he)**2)
    return dict(n_e=len(S['e']), R_ee=R_ee, R_he=R_he, T_out=len(S['e']) - R_ee - R_he, gamma_e=S['e']**2,
                tau=np.sort(1 - np.linalg.eigvalsh(r_ee.conj().T @ r_ee))[::-1],
                n_bad=st.myf.leads.N_UNRELIABLE - n0)


# %% [a] --- 1. scaling check ---
if run_part('a'):
    print(" a [nm]  t [meV]   mu_S  mu_N  mu_C  Delta  alpha  E_Z   eta_tr [meV]   W     L    barrier (meV, nm)  E_T")
    for a in A_LIST:
        q = junction(a, 'medium')
        p, t = q.params_transport(E_Z=0.3), q.t
        print(f"{a:6.1f}  {t:7.2f}   {p.mu_s * t:.2f}  {p.mu_n * t:.2f}  {p.mu_c * t:.2f}  {p.delta * t:.3f}  {p.alpha * a * t:5.1f}"
              f"  {p.Bxy_c * t:.2f}   {p.eta * t:.1e}   {p.ny * a:4.0f}  {p.nx * a:5.0f}   ({p.barrier_V * t:.1f}, {p.barrier_n * a:.0f})"
              f"         {q.E_T:.3f}")


# %% [a] --- 2. the probe contact in 1D: weak bond vs physical barrier ---
if run_part('a'):
    print("\n a [nm]   T(weak bond 0.2)   T(weak bond 0.4)   T(barrier 6 meV, 20 nm)   T(barrier 6 meV, 40 nm)")
    for a in A_LIST + (1.0,):
        print(f"{a:6.1f}       {contact_1d(a, tau=0.2):.4f}             {contact_1d(a, tau=0.4):.4f}"
              f"               {contact_1d(a, V_b=6, d_b=20):.4f}                    {contact_1d(a, V_b=6, d_b=40):.4f}")


# %% [a] --- 3. cross-section vs a ---
if run_part('a'):
    print("\n a [nm]  ABS   E_Z [meV]   Q   gap(k=0) [ueV]   Delta_top [ueV]")
    for a in A_LIST:
        q = junction(a)
        V = st.run_scan(partial(st.phase_point, q, ks=st.k_grid(a)),
                        {'E_Z': np.array(EZ_XS), 'phi': np.array([np.pi])}, cache=f'scharf_absA_xs_a{a:g}')[:, 0]
        for ez, (Q, g0, g, km) in zip(EZ_XS, V):
            print(f"{a:6.1f}  {st.count_abs(q):3d}   {ez:7.3f}   {Q:+.0f}   {g0 * 1e3:12.2f}   {g * 1e3:13.2f}")


# %% [a] --- 4. transport vs a, physical barrier ---
if run_part('a'):
    RES = {}
    for barrier, a_list in A_TRANSPORT.items():
        print(f"\nbarrier '{barrier}' (V = {BARRIERS[barrier][0]} meV, d = {BARRIERS[barrier][1]} nm), phi = pi, E = 0")
        print(" a [nm]  peaks>0.3  Majorana-like  >0.55   Majorana-like heights            at E_Z [meV]")
        for a in a_list:
            t = st.transport_line(junction(a, barrier), EZ_TR, cache=f'scharf_absA_tr_{barrier}_a{a:g}')
            RES[(barrier, a)] = t
            s = st.peak_stats(t)
            pk = st.kappa_peaks(t)
            good = pk[st.is_majorana_like(t, pk)]
            print(f"{a:6.1f}  {s['n']:6d}     {s['good']:6d}       {s['over']:4d}    {np.round(t['kappa'][good], 3)}   {np.round(t['E_Z'][good], 3)}")

    for barrier, a_list in A_TRANSPORT.items():
        fig, axs = plt.subplots(len(a_list), 1, figsize=(11, 2.1 * len(a_list)), sharex=True)
        for ax, a in zip(axs, a_list):
            t = RES[(barrier, a)]
            ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
            ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
            ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05); ax.set_ylabel(f'a = {a:g} nm', fontsize=9)
        axs[0].legend(fontsize=7, ncol=3)
        axs[-1].set_xlabel(rf'$E_Z$ [meV]   (W = 100 nm, L = 2 $\mu$m, $\phi=\pi$, barrier {barrier}: '
                           rf'{BARRIERS[barrier][0]:g} meV, {BARRIERS[barrier][1]:g} nm)')
        plt.tight_layout(); st.save_fig(fig, f'a_transport_{barrier}'); plt.show()


# %% [a] --- 5. a = 2.5 nm, narrow window around the neutral peaks (slow: hours, 4 workers) ---
if run_part('a'):
    print("\nmedium barrier, E_Z 0.77-0.93 meV: neutral peak positions vs a")
    FINE = {}
    for a in (5.0, 4.0, 2.5):
        q = junction(a, 'medium')
        if a == 2.5:
            FINE[a] = st.transport_line(q, EZ_FINE, cache='scharf_absA_trfine_medium_a2.5', n_workers=N_WORKERS_SMALL_A)
        else:                                                       # reuse the full lines
            t = RES[('medium', a)]
            m = (t['E_Z'] >= EZ_FINE[0] - 1e-9) & (t['E_Z'] <= EZ_FINE[-1] + 1e-9)
            FINE[a] = {k: v[m] for k, v in t.items()}
        pk = st.kappa_peaks(FINE[a])
        print(f"  a = {a:4.1f} nm: peaks at E_Z = {np.round(FINE[a]['E_Z'][pk], 3)},  kappa {np.round(FINE[a]['kappa'][pk], 3)},"
              f"  |G|/kappa {np.round(np.abs(FINE[a]['G'][pk]) / FINE[a]['kappa'][pk], 3)}")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for a, t in FINE.items():
        ax.plot(t['E_Z'], t['kappa'], lw=1, label=f'a = {a:g} nm')
    ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
    ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$\kappa/\kappa_0$'); ax.legend(fontsize=8)
    ax.set_title('medium barrier, W = 100 nm, L = 2 um, phi = pi: convergence of the peak positions', fontsize=9)
    plt.tight_layout(); st.save_fig(fig, 'a_fine_medium'); plt.show()


# %% [a] --- 6. the ONE-SITE barrier (lam_barr = V * a fixed): convergence with a ---
if run_part('a'):
    SITE = {}
    for barrier in SITE_BARRIERS:
        print(f"\none-site barrier '{barrier}' (lam = {SITE_BARRIERS[barrier]:g} meV nm), phi = pi, E = 0")
        print(" a [nm]  site [meV]  peaks>0.3  Majorana-like  >0.55   peak positions E_Z [meV] / heights / |G|/kappa")
        for a in A_SITE:
            t = st.transport_line(junction(a, barrier), EZ_TR, cache=f'scharf_absA_tr_{barrier}_a{a:g}')
            SITE[(barrier, a)] = t
            s, pk = st.peak_stats(t), st.kappa_peaks(t)
            print(f"{a:6.1f}  {SITE_BARRIERS[barrier] / a:8.1f}   {s['n']:6d}     {s['good']:6d}       {s['over']:4d}    "
                  f"{np.round(t['E_Z'][pk], 3)}")
            print(f"{'':44s}{np.round(t['kappa'][pk], 3)}  {np.round(np.abs(t['G'][pk]) / t['kappa'][pk], 3)}")
    for barrier in SITE_BARRIERS:
        fig, axs = plt.subplots(len(A_SITE), 1, figsize=(11, 2.1 * len(A_SITE)), sharex=True)
        for ax, a in zip(axs, A_SITE):
            t = SITE[(barrier, a)]
            ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
            ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05); ax.set_ylabel(f'a = {a:g} nm', fontsize=9)
        axs[0].legend(fontsize=7, ncol=2)
        axs[-1].set_xlabel(rf'$E_Z$ [meV]   (W = 100 nm, L = 2 $\mu$m, $\phi=\pi$, one-site barrier '
                           rf'{SITE_BARRIERS[barrier]:g} meV nm)')
        plt.tight_layout(); st.save_fig(fig, f'a_transport_{barrier}'); plt.show()


# %% [a] --- 7. the probe contact from the FULL model (supersedes the 1D chain of cell 2) ---
# contact_scattering (above): left-probe reflection matrix from the real G_LL and Gamma_L, normal state
# (Delta = 0, everything else unchanged).  T_out = n_e - Tr r_ee^dag r_ee = electron flux leaving the
# probe into the rest of the device; tau = per-channel transmissions.
CONTACTS_FULL = [('tc_barr 0.2', 20.0, dict(tc_barr_transport=0.2)),
                 ('tc_barr 0.2', 10.0, dict(tc_barr_transport=0.2)),
                 ('tc_barr 0.3387', 10.0, dict(tc_barr_transport=0.3387)),
                 ('lam 240', 20.0, dict(tc_barr_transport=1.0, lam_barr=240.0)),
                 ('lam 240', 10.0, dict(tc_barr_transport=1.0, lam_barr=240.0))]
if run_part('a'):
    print("\nfull model, left probe, E = 0, normal state (Delta = 0, E_Z = 0)")
    print("  contact          a    open e-channels   T_out    per-channel transmission")
    for name, a, kw in CONTACTS_FULL:
        n = contact_scattering(st.replace(st.case(100.0, a=a), **kw), 0.0, normal=True)
        print(f"  {name:15s} {a:4.0f}        {n['n_e']}          {n['T_out']:.4f}   {np.round(n['tau'], 4)}"
              f"   (failed lead solves {n['n_bad']})")
