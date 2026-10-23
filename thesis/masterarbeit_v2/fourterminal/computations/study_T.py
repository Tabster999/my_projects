"""
STUDY: the Majorana kappa resonance at finite temperature (pure Rashba, a = 20 nm, phi = pi).
Results in CLAUDE.md, Measurability section ("PURE-RASHBA CASES ..." and "THE COUPLING HAS A
TEMPERATURE OPTIMUM").

  1. The best T = 0 neutral peaks of three clean junctions, heated to 0-50 mK (tc_barr = 0.2).
  2. Full fine E_Z scans at several couplings tc_barr and T = 0 / 10 / 20 / 50 mK.

Method: the six channels are computed on a fixed energy grid E >= 0 (the expensive part, cached),
then st.thermal() does the thermal average for any T.  The signal survives while kT <~ Gamma ~ delta.
Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st
from functools import partial

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
                                np.round(np.arange(0.55, 1.25 + 1e-9, 0.0025), 5), (0.2, 0.3, 0.4)),
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


# %% --- 1. the best T = 0 peaks, heated ---
if __name__ == "__main__":
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


# %% --- 2. couplings at finite temperature, full fine field scans ---
if __name__ == "__main__":
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
