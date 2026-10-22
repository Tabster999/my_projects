"""
STUDY: what the junction width W does to the Andreev spectrum and the topological phase.
(pure Rashba, a = 20 nm; results in CLAUDE.md, "SCHARF: more ABS by widening W")

  1. ABS at E_Z = 0: how many, for which W.
  2. Along phi = pi: Q, the gap at k = 0 and the gap over all k (= Scharf's Delta_top).
  3. k_x = 0 (normal incidence): E(phi), E(E_Z) and the (E_Z, phi) phase diagram.
  4. (heavy, optional) the ldos-style map on a wide junction, with the Q = -1 boundary.

Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st
from functools import partial

W_COUNT = (100.0, 300.0, 500.0, 800.0, 1000.0)     # widths for the ABS count (cell 1)
WIDTHS = (100.0, 300.0, 800.0)                      # widths studied in detail (cells 2-3)
PHI = np.linspace(0, 2 * np.pi, 61)                 # phase grid of the k_x = 0 maps
N_EZ = 61                                           # field points of the k_x = 0 maps


# %% --- 1. ABS at E_Z = 0 ---
if __name__ == "__main__":
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


# %% --- 2. gaps along phi = pi ---
if __name__ == "__main__":
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


# %% --- 3. k_x = 0: spectra and phase diagram ---
if __name__ == "__main__":
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


# %% --- 4. (heavy) ldos map on a wide junction, with the Q = -1 boundary ---
# Takes hours at W = 800 nm and ~0.3 GB per worker.  Uses ldos's own map and plots.
if __name__ == "__main__":
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
