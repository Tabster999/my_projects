"""
The infinite junction at k_x = 0 (momentum ALONG the superconductors, package x = paper y).
Pure Rashba (scharf_abs_rashba.PR), a = 20 nm; W = 100 / 300 / 800 nm (1 / 2 / 3 ABS).

Per W:
  (a) E(phi) at E_Z = 0                       -- the ABS
  (b) E(phi) at a field inside Q = -1          -- how they split and cross zero
  (c) E(E_Z) at phi = pi, Q = -1 shaded        -- the levels through the transition
  (d) gap at k_x = 0 over (E_Z, phi)           -- the diamond, Q = -1 boundary dashed
Check: semi-infinite ribbons (myf.junction_spectrum_kx, the transport model's ribbons) at W = 300 nm,
spectral density A(E, E_Z) at k_x = 0, phi = pi, with the finite-strip levels on top.
k_x = 0 locates the transition (Q changes only where this gap closes) but NOT the protecting gap,
which closes at finite k_x for wide junctions (compare 'gap over all k' in scharf_abs_rashba).
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
from scharf_abs import myf

# %% --- SETUP ---
PR = R.PR
WIDTHS = (100.0, 300.0, 800.0)                   # 1000 nm dropped: too slow
N_EZ, N_PHI = 61, 61
PHI = np.linspace(0, 2 * np.pi, N_PHI)


def ez_axis(q, n=N_EZ):
    return np.round(np.linspace(0.0, max(0.3, 1.5 * q.E_T), n), 5)


def gap0_map(q, EZ, phis):
    """min |E| at k_x = 0 over (E_Z, phi), meV."""
    n_S = A.n_S_of(q)
    return np.array([[np.abs(np.linalg.eigvalsh(A.H_cross(q.params(E_Z=ez, phi=ph), 0.0, n_S))).min() * q.t
                      for ph in phis] for ez in EZ])


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    for W in WIDTHS:
        q = replace(PR, W=W)
        EZ = ez_axis(q)
        Q = A.q_map(q, EZ, PHI, name=f'k0_qmap_W{W:.0f}')
        G0 = gap0_map(q, EZ, PHI)
        j = int(np.argmin(np.abs(PHI - np.pi)))
        topo = Q[:, j] < 0
        i_b = int(np.argmax(np.where(topo, G0[:, j], -1)))          # largest k_x = 0 gap inside Q = -1
        E_ez = np.array([A.abs_spectrum(q, [np.pi], E_Z=ez, n_levels=16)[0] for ez in EZ])
        RES[W] = dict(q=q, EZ=EZ, Q=Q, G0=G0, E_ez=E_ez, EZ_b=float(EZ[i_b]), n_abs=A.count_abs(q, below=0.9)[0],
                      E_phi0=A.abs_spectrum(q, PHI, E_Z=0.0, n_levels=20),
                      E_phiB=A.abs_spectrum(q, PHI, E_Z=float(EZ[i_b]), n_levels=20))
        tz = EZ[topo]
        print(f"W = {W:.0f} nm ({RES[W]['n_abs']} ABS): phi = pi topological for E_Z in {tz.min():.3f}..{tz.max():.3f} meV;"
              f"  k_x=0 gap: {G0[i_b, j] * 1e3:.1f} ueV max at E_Z = {EZ[i_b]:.3f};"
              f"  topological fraction of the (E_Z, phi) map {np.mean(Q < 0):.2f}", flush=True)

    fig, axs = plt.subplots(len(RES), 4, figsize=(19, 3.7 * len(RES)))
    for row, (W, r) in zip(axs, RES.items()):
        q, EZ, Q, G0 = r['q'], r['EZ'], r['Q'], r['G0']
        j = int(np.argmin(np.abs(PHI - np.pi)))
        topo = Q[:, j] < 0
        for ax, E, ttl in ((row[0], r['E_phi0'], f'$E_Z$ = 0: {r["n_abs"]} ABS'),
                           (row[1], r['E_phiB'], f'$E_Z$ = {r["EZ_b"]:.3f} meV (Q = -1 at $\\phi=\\pi$)')):
            ax.plot(PHI / np.pi, E / q.Delta, 'k-', lw=0.8)
            ax.set_ylim(-1.05, 1.05); ax.set_xlabel(r'$\phi/\pi$'); ax.set_ylabel(r'$E/\Delta$')
            ax.set_title(f'W = {W:.0f} nm, $k_x$ = 0, {ttl}', fontsize=9)
        row[2].fill_between(EZ, -1.05, 1.05, where=topo, color='0.9', step='mid')
        row[2].plot(EZ, r['E_ez'] / q.Delta, 'k-', lw=0.8)
        row[2].set_ylim(-1.05, 1.05); row[2].set_xlabel(r'$E_Z$ [meV]'); row[2].set_ylabel(r'$E/\Delta$')
        row[2].set_title(r'$k_x$ = 0, $\phi = \pi$ (grey: Q = -1)', fontsize=9)
        im = row[3].pcolormesh(EZ, PHI / np.pi, G0.T * 1e3, shading='nearest', cmap='magma')
        row[3].contour(EZ, PHI / np.pi, Q.T, levels=[0], colors='c', linestyles='--', linewidths=1)
        fig.colorbar(im, ax=row[3], label=r'gap at $k_x$ = 0 [$\mu$eV]')
        row[3].set_xlabel(r'$E_Z$ [meV]'); row[3].set_ylabel(r'$\phi/\pi$')
        row[3].set_title('gap at $k_x$ = 0 (cyan dashed: Q = -1 boundary)', fontsize=9)
    plt.tight_layout(); A._save(fig, 'k0_spectra'); plt.close(fig)

    # --- check: semi-infinite ribbons (transport model) at W = 300 nm ---
    q, r = replace(PR, W=300.0), RES[300.0]
    E = np.linspace(-1.05, 1.05, 421) * q.Delta
    Amap = np.array([myf.junction_spectrum_kx(replace(q.params(E_Z=ez, phi=np.pi), eta=5e-3 * q.Delta / q.t),
                                              [0.0], E / q.t)[:, 0] for ez in r['EZ']])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(r['EZ'], E / q.Delta, Amap.T, shading='nearest', cmap='cividis',
                       vmax=np.percentile(Amap, 99))
    ax.plot(r['EZ'], r['E_ez'] / q.Delta, 'r-', lw=0.5)
    fig.colorbar(im, ax=ax, label=r'$A(E)$, semi-infinite ribbons')
    ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$E/\Delta$'); ax.set_ylim(-1.05, 1.05)
    ax.set_title(r'W = 300 nm, $k_x$ = 0, $\phi=\pi$: semi-infinite ribbons (colour) vs 900 nm strips (red)',
                 fontsize=9)
    plt.tight_layout(); A._save(fig, 'k0_semiinfinite_W300'); plt.close(fig)
    print("done", flush=True)
