"""
STUDY: can a different probe coupling tc_barr_transport replace a longer junction?
(it cannot -- results in CLAUDE.md, "THE PROBE COUPLING DOES NOT REPLACE LENGTH")

kappa, G, local Andreev and det r vs E_Z at phi = pi, E = 0, for five junctions and five couplings.
Majorana-like kappa peak: 0.45 <= kappa <= 0.55 and |G|/kappa < 0.05.
Two-Majorana picture: kappa = 2 delta^2 Gamma^2 / (delta^2 + Gamma^2)^2, and tc only moves Gamma.
Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st

TCS = (0.05, 0.1, 0.2, 0.5, 1.0)
CASES = {'W100 L2 (1 ABS, ref)':   dict(tag='W100_L2', W=100.0, L=2000.0),
         'W300 L2 (2 ABS)':        dict(tag='W300_L2', W=300.0, L=2000.0),
         'W300 L4 (2 ABS, works)': dict(tag='W300_L4', W=300.0, L=4000.0),
         'W800 L4 (3 ABS)':        dict(tag='W800_L4', W=800.0, L=4000.0),
         'W300 D0.75 L4 (4 ABS)':  dict(tag='W300_D0.75_L4', W=300.0, L=4000.0, Delta=0.75)}


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    print("case                      tc    peaks>0.3  Majorana-like  kappa>0.55  max kappa  median |G|/kappa"
          "  frac(det r < 0)")
    for name, c in CASES.items():
        for tc in TCS:
            q = st.case(c['W'], L=c['L'], Delta=c.get('Delta', 0.25), tc=tc)
            t = st.transport_line(q, st.field_grid(q, 121), cache=f"scharf_absTC_{c['tag']}_tc{tc:g}")
            RES[(name, tc)] = t
            s = st.peak_stats(t)
            print(f"{name:24s} {tc:5.2f}   {s['n']:6d}     {s['good']:6d}        {s['over']:6d}     {s['kmax']:6.3f}"
                  f"      {s['rmed']:7.3f}          {s['det_neg']:.2f}")


# %% --- FIGURE: cases as columns, couplings as rows ---
if __name__ == "__main__":
    fig, axs = plt.subplots(len(TCS), len(CASES), figsize=(4.2 * len(CASES), 2.2 * len(TCS)), sharex='col')
    for j, name in enumerate(CASES):
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
