"""
STUDY: checks on the Majorana resonances (pure Rashba; results in CLAUDE.md).

  1. Lattice constant a = 20 / 10 / 5 nm at fixed physical W: ABS count, Q and the gaps (cross-section).
  2. W = 300 nm, L = 4 um: fine scan through a double peak + Majorana splitting from the end LDOS.
     The dip between the two kappa ~ 1/2 peaks is where the splitting delta crosses zero.
  3. W = 300 nm, L = 4 um: a = 20 vs 10 nm in transport.
  4. Longer junctions: W = 300 nm at L = 8 um, W = 800 nm (3 ABS) at L = 4 and 8 um.

Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st


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


# %% --- 1. lattice constant, cross-section only ---
if __name__ == "__main__":
    print("  W[nm]  a[nm]  ABS  E_Z[meV]   Q   gap(k=0)[ueV]  gap(all k)[ueV]  at k[1/nm]")
    for W, EZs in ((100.0, (0.3, 0.593)), (300.0, (0.15, 0.323))):
        for a in (20.0, 10.0, 5.0):
            q = st.case(W, a=a)
            for EZ in EZs:
                Q, g0, g, km = st.phase_point(q, EZ, np.pi, ks=st.k_grid(a))
                print(f"  {W:5.0f}  {a:5.0f}  {st.count_abs(q):3d}  {EZ:7.3f}  {Q:+3.0f}  {g0 * 1e3:12.2f}"
                      f"  {g * 1e3:14.2f}   {km / a:9.4f}", flush=True)


# %% --- 2. the double peak at W = 300 nm, L = 4 um ---
if __name__ == "__main__":
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


# %% --- 3. and 4. a = 10 vs 20 nm, and longer junctions ---
if __name__ == "__main__":
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
