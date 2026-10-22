"""
STUDY (UNFINISHED): the topological transition with 2-3 ABS (W = 300 / 800 nm), seen in the LDOS.
Pure Rashba, a = 20 nm, L = 2 um, phi = pi, probes decoupled (tc_barr = 0) and eta = 0.05 Delta (Scharf).

  1. LDOS(E, E_Z) at the junction END and in the MIDDLE: the gap closes at the transition;
     a zero-energy peak only at the end = end Majoranas, at both = bulk low-energy states.
  2. Real-space LDOS(E = 0) at four fields: trivial / gap closing / best-gapped topological / high field.

Status: W = 300 nm finished; W = 800 nm partly cached (resumes where it stopped).
Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st
from functools import partial

WIDTHS = (300.0, 800.0)
E_LDOS = np.linspace(-1.0, 1.0, 81)               # energies in units of Delta
ETA = 0.05                                        # LDOS broadening in units of Delta (Scharf)
EVERY = {300.0: 1, 800.0: 4}                      # use every n-th column for the real-space maps (memory)
N_WORKERS = 10                                    # LocalGreen at W = 800 needs ~0.3 GB per worker


# %% --- 1. LDOS(E, E_Z) at the end and in the middle ---
if __name__ == "__main__":
    RES = {}
    for W in WIDTHS:
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


# %% --- 2. real-space LDOS(E = 0) at four fields ---
if __name__ == "__main__":
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
