"""
Does a stronger probe coupling make the Majorana kappa resonance survive higher temperatures?
W = 100 nm, L = 2 um, pure Rashba, a = 20 nm, phi = pi; Delta = 0.25 (1 ABS) and 0.5 meV (2 ABS).  A larger Gamma widens the resonance in
energy (good for kT) but needs a larger delta = Gamma (other E_Z) and couples the higher levels
(G != 0) -- scharf_abs_tc.py found G turning on at tc >= 0.5 at T = 0.
Full fine E_Z scans (the resonance moves with tc), channels on a fixed energy grid, thermal
integrals with scharf_abs_T.thermal.  Per (tc, T): best kappa_T, best NEUTRAL kappa_T
(|G_T|/kappa_T < 0.05), and the E_Z range with neutral kappa_T > 0.25.
"""
# %% --- IMPORTS ---
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
import scharf_abs as A
import scharf_abs_rashba as R
import scharf_abs_T as T
import scharf_abs_delta as D
from scharf_abs import myf, DATA

CASES = {'W100_L2': (replace(R.PR, W=100.0, L=2000.0), np.round(np.arange(0.55, 1.25 + 1e-9, 0.0025), 5),
                     (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0)),
         'W100_L2_D0.5': (D.case_pp(100.0, 0.5, 16.0), np.round(np.arange(0.3, 1.0 + 1e-9, 0.0025), 5),
                          (0.2, 0.3, 0.4, 0.5))}
T_MK = (0.0, 10.0, 20.0, 50.0)
E_GRID = np.unique(np.concatenate([np.linspace(0.0, 3e-3, 31), np.linspace(3e-3, 35e-3, 33)]))   # meV
N_WORKERS = 14


class EPointPi:
    """Picklable for myf.scan: (E_Z, E) in meV -> the six channels at phi = pi."""
    def __init__(self, q):
        self.q = q

    def __call__(self, E_Z, E):
        q = self.q
        rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=float(E_Z))),
                                  np.array([float(E) / q.t]))
        c = rgf.channels_at_phi(np.pi, 'left')
        return np.array([float(np.asarray(c[k])[0]) for k in T.KEYS])


def tc_scan(case, tc):
    import scharf_abs_T_tc
    q0, EZ, _ = CASES[case]
    q = replace(q0, tc_barr_transport=tc)
    res = myf.scan(scharf_abs_T_tc.EPointPi(q), {'E_Z': EZ, 'E': E_GRID}, n_workers=N_WORKERS,
                   save=str(DATA / f'scharf_absTtc_{case}_tc{tc:g}.npz'))
    V = res['values']                                           # (n_EZ, n_E, 6)
    out = {}
    for Tm in T_MK:
        kT = T.K_B * Tm * 1e-3
        out[Tm] = np.array([T.thermal({k: V[i, :, n] for n, k in enumerate(T.KEYS)}, kT, E_GRID)
                            for i in range(len(EZ))])           # (n_EZ, 3): kappa, G, Andreev
    return out


# %% --- RUN ---
if __name__ == "__main__":
    for case, (q0, EZ, TCS) in CASES.items():
        RES = {}
        print(f"\n{case}\n  tc    T[mK]  best kappa (|G|/kappa)   best NEUTRAL kappa at E_Z"
              "   E_Z range neutral kappa > 0.25   best kappa with |G|/kappa < 0.1", flush=True)
        for tc in TCS:
            RES[tc] = tc_scan(case, tc)
            for Tm in T_MK:
                r = RES[tc][Tm]
                k, G = r[:, 0], r[:, 1]
                i = int(np.argmax(k))
                neutral = np.abs(G) < 0.05 * np.maximum(k, 1e-12)
                j = int(np.argmax(np.where(neutral, k, -1)))
                span = (neutral & (k > 0.25)).sum() * (EZ[1] - EZ[0])
                l = int(np.argmax(np.where(np.abs(G) < 0.1 * np.maximum(k, 1e-12), k, -1)))
                print(f"  {tc:4.2f}  {Tm:5.1f}   {k[i]:6.3f} ({abs(G[i]) / k[i]:5.3f})         {k[j]:6.3f} at {EZ[j]:.4f}"
                      f"          {span:.4f} meV              {k[l]:6.3f} at {EZ[l]:.4f}", flush=True)

        fig, axs = plt.subplots(len(TCS), 1, figsize=(11, 2.2 * len(TCS)), sharex=True)
        for ax, tc in zip(axs, TCS):
            for Tm, col in zip(T_MK, ('k', 'tab:orange', 'tab:red', 'tab:purple')):
                r = RES[tc][Tm]
                ax.plot(EZ, r[:, 0], '-', color=col, lw=0.9, label=rf'$\kappa/\kappa_0$, T = {Tm:g} mK')
                ax.plot(EZ, np.abs(r[:, 1]), ':', color=col, lw=0.8)
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05); ax.set_ylabel(f'tc_barr = {tc:g}', fontsize=9)
        axs[0].legend(fontsize=7, ncol=4)
        axs[-1].set_xlabel(rf'$E_Z$ [meV]   ({case}, $\phi=\pi$; dotted: $|G|/G_0$)')
        plt.tight_layout(); A._save(fig, f'T_tc_{case}'); plt.close(fig)
    print("done", flush=True)
