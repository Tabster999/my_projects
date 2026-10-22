"""
More ABS WITHOUT widening: larger Delta (the ABS count goes with Delta / E_T) and stronger SOC.
Pure Rashba (scharf_abs_rashba.PR), a = 20 nm, transport geometry for kappa (scharf_abs_checks).

For each (W, Delta, alpha):
  D1  ABS count at E_Z = 0 (levels < 0.9 Delta at k = 0, phi = 0).
  D2  phi = pi line: Q, gap at k = 0, gap over all k (cross-section, 900 nm SC strips).
  D3  kappa, G, local Andreev vs E_Z at phi = pi, L = 2 um, weak probe.
Question: does a larger Delta give 2-3 ABS at W = 100-300 nm while keeping the gap over all k
open, so that kappa stays a neutral 1/2 at L = 2 um (unlike widening W)?
Caveat: Delta = 1 meV equals mu_S = 1 meV (strong-coupling SC); read it as a trend.
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
import scharf_abs_checks as K
from scharf_abs import myf, DATA

PR = R.PR
CASES = [(W, D, 16.0) for W in (100.0, 300.0) for D in (0.25, 0.5, 0.75, 1.0)] \
      + [(300.0, 0.25, 32.0), (300.0, 0.5, 32.0)]


def case_pp(W, D, alpha):
    q = replace(PR, W=W, Delta=D, alpha=alpha)
    q.theta_z = q.theta_z_optimal() + np.pi
    return q


def tag(W, D, alpha):
    return f'W{W:.0f}_D{D:g}_a{alpha:g}'


def phase_line(q, name, n=41):
    import scharf_abs
    EZ = R.ez_grid(q, n)
    res = myf.scan(scharf_abs.PhasePoint(q, ks=A.K_GRID), {'E_Z': EZ, 'phi': np.array([np.pi])},
                   n_workers=14, save=str(DATA / f'scharf_absD_line_{name}.npz'))
    V = res['values'][:, 0]
    return dict(E_Z=EZ, Q=V[:, 0], gap0=V[:, 1], gap=V[:, 2])


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    print(" W[nm] Delta alpha  ABS   E_T   topological E_Z (phi=pi)  max gap over all k [ueV] (at E_Z)"
          "   gap/Delta", flush=True)
    for W, D, al in CASES:
        q = case_pp(W, D, al)
        n_abs = A.count_abs(q, below=0.9)[0]
        line = phase_line(q, tag(W, D, al))
        topo = line['Q'] < 0
        i = int(np.argmax(np.where(topo, line['gap'], -1)))
        rng = f"{line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f}" if topo.any() else "none"
        print(f"{W:6.0f} {D:5.2f} {al:5.1f}  {n_abs:3d}  {q.E_T:5.3f}   {rng:>16s}"
              f"          {line['gap'][i] * 1e3:8.2f} ({line['E_Z'][i]:.3f})        {line['gap'][i] / D:.3f}",
              flush=True)
        RES[(W, D, al)] = dict(q=q, n_abs=n_abs, line=line)

    print("\n kappa at phi = pi, L = 2 um, weak probe tc_barr = 0.2", flush=True)
    for key, r in RES.items():
        q = r['q']
        t = K.channel_line(q, R.ez_grid(q, 121), 'D3_' + tag(*key))
        r['tr'] = t
        K.summary_line(f"W {key[0]:.0f}, Delta {key[1]:g}, alpha {key[2]:g} ({r['n_abs']} ABS)", t)

    fig, axs = plt.subplots(len(RES), 2, figsize=(14, 2.3 * len(RES)))
    for row, (key, r) in zip(axs, RES.items()):
        line, t = r['line'], r['tr']
        topo = line['Q'] < 0
        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        row[0].fill_between(line['E_Z'], 0, top, where=topo, color='0.88', step='mid')
        row[0].plot(line['E_Z'], line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        row[0].plot(line['E_Z'], line['gap'] * 1e3, 'r-', label='gap over all k')
        row[0].set_ylim(0, top)
        row[0].set_ylabel(f'W {key[0]:.0f}, $\\Delta$ {key[1]:g}\n$\\alpha$ {key[2]:g}: {r["n_abs"]} ABS', fontsize=8)
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
    plt.tight_layout(); A._save(fig, 'D_delta'); plt.close(fig)
    print("done", flush=True)
