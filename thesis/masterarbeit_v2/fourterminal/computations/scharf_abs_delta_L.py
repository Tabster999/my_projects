"""
4 ABS from a larger Delta (W = 300 nm, scharf_abs_delta.py): does a longer junction give genuine
Majoranas, as it did for W = 300 nm at Delta = 0.25 (2 ABS, L >= 4 um)?
kappa, G, local Andreev vs E_Z at phi = pi, weak probe, pure Rashba, a = 20 nm.
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
import scharf_abs_delta as D

CASES = [(300.0, 0.75, 4000.0), (300.0, 0.75, 8000.0), (300.0, 1.0, 4000.0)]


def neutral_stats(t):
    k, G = t['kappa'], t['G']
    pk = np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > 0.3)) + 1
    r = np.abs(G[pk]) / k[pk]
    return len(pk), int((r < 0.05).sum()), int(((k[pk] >= 0.48) & (r < 0.05)).sum()), \
        int(((k[pk] > 0.52) & (r < 0.05)).sum())


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    for W, Dl, L in CASES:
        q = replace(D.case_pp(W, Dl, 16.0), L=L)
        t = K.channel_line(q, R.ez_grid(q, 121), f'DL_W{W:.0f}_D{Dl:g}_L{L:.0f}')
        RES[(W, Dl, L)] = t
        n, nn, ng, nover = neutral_stats(t)
        K.summary_line(f"W {W:.0f}, Delta {Dl:g}, L {L / 1000:.0f} um (4 ABS)", t)
        print(f"      peaks > 0.3: {n},  neutral: {nn},  neutral and >= 0.48: {ng},  neutral but > 0.52: {nover}",
              flush=True)

    fig, axs = plt.subplots(len(RES), 1, figsize=(10, 2.6 * len(RES)))
    for ax, ((W, Dl, L), t) in zip(axs, RES.items()):
        ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
        ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
        ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
        ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f'W {W:.0f}, $\\Delta$ {Dl:g}\nL {L / 1000:.0f} $\\mu$m', fontsize=8)
    axs[0].legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel(r'$E_Z$ [meV]   ($\phi = \pi$, E = 0, weak probe; 4 ABS)')
    plt.tight_layout(); A._save(fig, 'DL_4abs'); plt.close(fig)
    print("done", flush=True)
