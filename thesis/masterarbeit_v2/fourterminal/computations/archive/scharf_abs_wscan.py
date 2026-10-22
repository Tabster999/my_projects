"""
More Andreev bands by widening the junction, studied the scharf_ldos way.

  1. E(phi) at E_Z = 0, k = 0 for several W: count the ABS, choose W with 3-4 of them.
  2. The topological transition: the scharf_ldos map over (E_Z, phi) on that junction --
     edge/bulk LDOS(E=0), its energy curvature, kappa and G -- with the boundary of the class-D
     invariant Q = -1 (scharf_abs cross-section) drawn on top as a dashed line.
  3. Points with kappa closest to 1/2 at phi = pi: channels, kappa/G against T, real-space LDOS.

Genuine Majorana pair = inside Q = -1, LDOS(E=0) peaked at the ends (curv_end < 0, end >> mid),
kappa -> 1/2 with G ~ 0.  The ABS count = distinct levels below 0.9 Delta at phi = 0 (k = 0).
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
import scharf_ldos as S
from scharf_abs import DATA

# %% --- SETUP ---
PB = replace(A.PP, tc_barr_transport=0.2)         # a = 20 nm, L = 2000 nm, weak probe
W_CANDIDATES = (100.0, 300.0, 500.0, 800.0, 1000.0)
W_CHOSEN = 800.0                                  # 3 ABS below 0.9 Delta (4 below 0.98 Delta)
N_WORKERS = 10                                    # LocalGreen at W = 800 needs ~0.3 GB per worker
N_EZ, N_PHI = 41, 41


# %% --- 1. ABS at E_Z = 0: how many for which W ---
if __name__ == "__main__":
    PHI = np.linspace(0, 2 * np.pi, 101)
    fig, axs = plt.subplots(1, len(W_CANDIDATES), figsize=(3.4 * len(W_CANDIDATES), 3.6), sharey=True)
    print("W [nm]   E_T [meV]   ABS (< 0.9 Delta)")
    for ax, W in zip(axs, W_CANDIDATES):
        q = replace(PB, W=W)
        n = A.count_abs(q, below=0.9)[0]
        print(f"{W:6.0f}   {q.E_T:9.3f}   {n:5d}")
        ax.plot(PHI / np.pi, A.abs_spectrum(q, PHI, E_Z=0.0, n_levels=20) / q.Delta, 'k-', lw=0.8)
        ax.axhline(0.9, color='r', ls=':', lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$\phi/\pi$')
        ax.set_title(f'W = {W:.0f} nm: {n} ABS', fontsize=10)
    axs[0].set_ylabel(r'$E/\Delta$  ($E_Z$ = 0, $k$ = 0)')
    plt.tight_layout(); A._save(fig, 'W_abs'); plt.show()


# %% --- 2. the topological transition on the wide junction (scharf_ldos map) ---
if __name__ == "__main__":
    PW = replace(PB, W=W_CHOSEN)
    S.show_summary(PW)
    EZ = np.linspace(0.0, max(0.3, 2.0 * PW.E_T), N_EZ)              # E_T sets the field scale
    B_VALS = PW.B_of_E_Z(EZ)
    PHIS = np.linspace(0.0, 2 * np.pi, N_PHI)
    R = S.run_map_parallel(pp=PW, phis=PHIS, b_vals=B_VALS, n_workers=N_WORKERS,
                           save=str(DATA / f'scharf_absW_map_W{W_CHOSEN:.0f}.npz'))
    TOPO = (R['E_Z'], PHIS, A.q_map(PW, R['E_Z'], PHIS, name=f'qmap_W{W_CHOSEN:.0f}'))
    S.plot_map(R, PW, topo=TOPO)
    S.plot_curvature(R, PW, topo=TOPO)
    S.plot_transport(R, PW, topo=TOPO)


# %% --- 3. points with kappa closest to 1/2 at phi = pi ---
if __name__ == "__main__":
    pts = S.pick_points(R, n=2, require_phi=np.pi)
    S.show_points(pts)
    S.point_report(PW, pts, every=4)                  # every 4th column: the full map is ~4 GB at W = 800
