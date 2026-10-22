"""
STUDY: more ABS WITHOUT widening -- a larger Delta (the ABS count goes with Delta / E_T), and
stronger SOC.  Pure Rashba, a = 20 nm; results in CLAUDE.md ("LARGER DELTA DOES WHAT WIDENING CANNOT").

  1. Per (W, Delta, alpha): ABS count, and along phi = pi Q, the gap at k = 0, the gap over all k.
  2. kappa, G, local Andreev vs E_Z at phi = pi, L = 2 um.
  3. 4 ABS (W = 300 nm, Delta 0.75 / 1.0 meV) at L = 4 and 8 um: does length help, as for 2 ABS?

Caveat: Delta >= 0.75 meV approaches mu_S = 1 meV (strong-coupling superconductor); read as a trend.
Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st

CASES = [(W, D, 16.0) for W in (100.0, 300.0) for D in (0.25, 0.5, 0.75, 1.0)] \
      + [(300.0, 0.25, 32.0), (300.0, 0.5, 32.0)]                   # (W nm, Delta meV, alpha meV nm)
CASES_LONG = [(300.0, 0.75, 4000.0), (300.0, 0.75, 8000.0), (300.0, 1.0, 4000.0)]   # (W, Delta, L)


def tag(W, D, alpha):
    return f'W{W:.0f}_D{D:g}_a{alpha:g}'


# %% --- 1. ABS count and gaps along phi = pi ---
if __name__ == "__main__":
    RES = {}
    print(" W[nm] Delta alpha  ABS   E_T   topological E_Z (phi=pi)  max gap over all k [ueV] (at E_Z)   gap/Delta")
    for W, D, al in CASES:
        q = st.case(W, Delta=D, alpha=al)
        line = st.phase_line(q, st.field_grid(q, 41), cache=f'scharf_absD_line_{tag(W, D, al)}')
        topo = line['Q'] < 0
        i = st.best_topological(line)
        rng = f"{line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f}" if topo.any() else "none"
        print(f"{W:6.0f} {D:5.2f} {al:5.1f}  {st.count_abs(q):3d}  {q.E_T:5.3f}   {rng:>16s}"
              f"          {line['gap'][i] * 1e3:8.2f} ({line['E_Z'][i]:.3f})        {line['gap'][i] / D:.3f}")
        RES[(W, D, al)] = dict(q=q, line=line)


# %% --- 2. transport at L = 2 um ---
if __name__ == "__main__":
    for key, r in RES.items():
        q = r['q']
        r['tr'] = st.transport_line(q, st.field_grid(q, 121), cache='scharf_absK_D3_' + tag(*key))
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


# %% --- 3. 4 ABS in longer junctions ---
if __name__ == "__main__":
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
