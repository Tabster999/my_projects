"""
The W study of scharf_abs_wscan.py with PURE RASHBA SOC (beta = 0), a subset of the tests:
  1. ABS count at E_Z = 0 vs W.
  2. For a few W, along phi = pi: Q, gap at k = 0, gap over all k (cross-section).
  3. Finite device at the best-protected topological point: delta, E_1, LDOS(E = 0) profile.
  4. kappa, G vs E_Z at phi = pi and 0 (weak probe).
alpha = 16 meV nm keeps |SOC| = sqrt(alpha^2 + beta^2) of the beta != 0 runs; the field is turned
to the pure-Rashba optimum (theta_z_optimal = pi/2, plus pi as in scharf_ldos.PP).
Caches: plot_npz/scharf_absR_*.npz (separate from the beta != 0 files).
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
from scharf_abs import myf, DATA

# %% --- SETUP ---
SOC_AMP = 16.0                                    # meV nm
PR = replace(A.PP, alpha=SOC_AMP, beta=0.0, tc_barr_transport=0.2)
PR.theta_z = PR.theta_z_optimal() + np.pi         # = 3 pi / 2 for pure Rashba
TAG = 'R'
W_COUNT = (100.0, 300.0, 500.0, 800.0, 1000.0)
W_STUDY = (100.0, 300.0, 800.0)
ETA_SCHARF, ETA_FINE = 0.05 * PR.Delta, 0.5e-3    # meV
N_STATES = 60


def ez_grid(q, n):
    return np.round(np.linspace(0.0, max(0.3, 1.5 * q.E_T), n), 5)


def phase_line(q, n=41, n_workers=14):
    """Q, gap(k=0), gap(all k), k_min along phi = pi (cached)."""
    import scharf_abs
    EZ = ez_grid(q, n)
    res = myf.scan(scharf_abs.PhasePoint(q, ks=A.K_GRID), {'E_Z': EZ, 'phi': np.array([np.pi])},
                   n_workers=n_workers, save=str(DATA / f'scharf_abs{TAG}_line_W{q.W:.0f}.npz'))
    V = res['values'][:, 0]
    return dict(E_Z=EZ, Q=V[:, 0], gap0=V[:, 1], gap=V[:, 2], kmin=V[:, 3])


def transport_line(q, n=121, n_workers=14):
    """kappa, G at phi = pi and 0 vs E_Z, T -> 0, weak probe (cached)."""
    EZ = ez_grid(q, n)
    f = myf.ThermalPoint(q.params_transport(E_Z=0.0), kT=0.0, phis=(np.pi, 0.0))
    res = myf.scan(f, {'Bxy_c': EZ / q.t}, n_workers=n_workers,
                   save=str(DATA / f'scharf_abs{TAG}_transport_W{q.W:.0f}_tc{q.tc_barr_transport:g}.npz'))
    V = res['values']
    return dict(E_Z=EZ, kappa=V[:, 0], G=V[:, 1], kappa0=V[:, 2], G0=V[:, 3], n_bad=V[:, 4])


def ldos_E0(E, rho, eta):
    return A.ldos_from_states(E, rho, [0.0], eta)[0]


# %% --- 1. ABS at E_Z = 0 ---
if __name__ == "__main__":
    print(f"pure Rashba: alpha = {PR.alpha} meV nm, beta = {PR.beta}, theta_z = {PR.theta_z / np.pi:.3f} pi")
    PHI = np.linspace(0, 2 * np.pi, 101)
    fig, axs = plt.subplots(1, len(W_COUNT), figsize=(3.4 * len(W_COUNT), 3.6), sharey=True)
    print("W [nm]   E_T [meV]   ABS (< 0.9 Delta)")
    for ax, W in zip(axs, W_COUNT):
        q = replace(PR, W=W)
        n = A.count_abs(q, below=0.9)[0]
        print(f"{W:6.0f}   {q.E_T:9.3f}   {n:5d}")
        ax.plot(PHI / np.pi, A.abs_spectrum(q, PHI, E_Z=0.0, n_levels=20) / q.Delta, 'k-', lw=0.8)
        ax.axhline(0.9, color='r', ls=':', lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$\phi/\pi$')
        ax.set_title(f'W = {W:.0f} nm: {n} ABS', fontsize=10)
    axs[0].set_ylabel(r'$E/\Delta$  ($E_Z$ = 0, $k$ = 0)')
    fig.suptitle(r'pure Rashba, $\beta$ = 0', fontsize=9)
    plt.tight_layout(); A._save(fig, 'R_abs'); plt.show()


# %% --- 2-4. phase line, finite device, transport for a few W ---
if __name__ == "__main__":
    RES = {}
    for W in W_STUDY:
        q = replace(PR, W=W)
        line = phase_line(q)
        topo = line['Q'] < 0
        i = int(np.argmax(np.where(topo, line['gap'], -1)))
        EZ_best = float(line['E_Z'][i])
        E, rho, n_S = A.device_states(q, EZ_best, np.pi, n_states=N_STATES)
        RES[W] = dict(q=q, n_abs=A.count_abs(q, below=0.9)[0], line=line, EZ=EZ_best,
                      E=E, rho=rho, n_S=n_S, tr=transport_line(q))
        print(f"W = {W:.0f} nm done", flush=True)

    print("\n W[nm] ABS  topological E_Z   best E_Z  gap(k=0)  gap(all k)  delta    E_1   end/mid  end/mid"
          "  max kappa(pi) in Q=-1   |G|/kappa  kappa(0)")
    print("            (phi=pi) [meV]   [meV]     [ueV]     [ueV]     [ueV]   [ueV]  (fine)  (Scharf)"
          "                           there      there")
    for W, r in RES.items():
        q, line, E, rho, n_S, tr = r['q'], r['line'], r['E'], r['rho'], r['n_S'], r['tr']
        topo = line['Q'] < 0
        i = int(np.argmin(np.abs(line['E_Z'] - r['EZ'])))
        n0 = len(E) // 2
        centre = slice(n_S, n_S + q.n_across)
        prof = {eta: ldos_E0(E, rho, eta)[:, centre].sum(1) for eta in (ETA_FINE, ETA_SCHARF)}
        con = {eta: p[0] / p[len(p) // 2] for eta, p in prof.items()}
        tmask = np.interp(tr['E_Z'], line['E_Z'], topo.astype(float)) > 0.5
        j = int(np.nanargmax(np.where(tmask, tr['kappa'], -1)))
        rng = f"{line['E_Z'][topo].min():5.3f}..{line['E_Z'][topo].max():5.3f}" if topo.any() else "    none    "
        print(f"{W:6.0f} {r['n_abs']:3d}   {rng}   {r['EZ']:6.3f}  {line['gap0'][i] * 1e3:8.2f}  {line['gap'][i] * 1e3:9.3f}"
              f"  {2 * E[n0] * 1e3:7.3f} {E[n0 + 1] * 1e3:6.2f}  {con[ETA_FINE]:6.1f}  {con[ETA_SCHARF]:7.1f}"
              f"   {tr['kappa'][j]:.4f} @ {tr['E_Z'][j]:.3f}        {abs(tr['G'][j]) / tr['kappa'][j]:8.2e}  {tr['kappa0'][j]:8.2e}")
        if np.nansum(tr['n_bad']):
            print(f"       WARNING: {int(np.nansum(tr['n_bad']))} failed lead solves")

    # figure: gaps along phi = pi | LDOS(E=0) profile | transport, one row per W
    fig, axs = plt.subplots(len(RES), 3, figsize=(16, 3.4 * len(RES)))
    for row, (W, r) in zip(axs, RES.items()):
        q, line, E, rho, n_S, tr = r['q'], r['line'], r['E'], r['rho'], r['n_S'], r['tr']
        topo = line['Q'] < 0
        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        row[0].fill_between(line['E_Z'], 0, top, where=topo, color='0.88', step='mid', label='Q = -1')
        row[0].plot(line['E_Z'], line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        row[0].plot(line['E_Z'], line['gap'] * 1e3, 'r-', label='gap over all k')
        row[0].axvline(r['EZ'], color='k', ls='--', lw=0.8)
        row[0].set_ylim(0, top)
        row[0].set_xlabel(r'$E_Z$ [meV]'); row[0].set_ylabel(f'W = {W:.0f} nm ({r["n_abs"]} ABS)\ngap [$\\mu$eV]')
        row[0].legend(fontsize=7); row[0].set_title(r'$\phi = \pi$', fontsize=9)

        y = (np.arange(q.n_along) + 0.5) * q.a - q.L / 2
        centre = slice(n_S, n_S + q.n_across)
        for eta, ls in ((ETA_FINE, '-'), (ETA_SCHARF, '--')):
            p = ldos_E0(E, rho, eta)[:, centre].sum(1)
            row[1].plot(y, p / p.max(), ls, label=f'$\\eta$ = {eta * 1e3:g} $\\mu$eV')
        n0 = len(E) // 2
        row[1].set_xlabel('y [nm] (along)'); row[1].set_ylabel('LDOS(E=0), normal strip (norm.)')
        row[1].legend(fontsize=7)
        row[1].set_title(f'$E_Z$ = {r["EZ"]:.3f} meV: $\\delta$ = {2 * E[n0] * 1e3:.2f} $\\mu$eV, '
                         f'$E_1$ = {E[n0 + 1] * 1e3:.2f} $\\mu$eV', fontsize=9)

        tm = np.interp(tr['E_Z'], line['E_Z'], topo.astype(float)) > 0.5
        row[2].fill_between(tr['E_Z'], 0, 1.05, where=tm, color='0.9', step='mid')
        row[2].plot(tr['E_Z'], tr['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$, $\phi=\pi$')
        row[2].plot(tr['E_Z'], np.abs(tr['G']), 'r-', lw=0.8, label=r'$|G|/G_0$, $\phi=\pi$')
        row[2].plot(tr['E_Z'], tr['kappa0'], 'b:', lw=0.8, label=r'$\kappa/\kappa_0$, $\phi=0$')
        row[2].axhline(0.5, color='0.5', ls='--', lw=0.6)
        row[2].set_ylim(0, 1.05)
        row[2].set_xlabel(r'$E_Z$ [meV]  (grey: Q = -1)')
        row[2].legend(fontsize=7, ncol=3)
    fig.suptitle(fr'pure Rashba ($\alpha$ = {PR.alpha} meV nm, $\beta$ = 0), weak probe tc_barr = {PR.tc_barr_transport:g}',
                 fontsize=9)
    plt.tight_layout(); A._save(fig, 'R_study'); plt.show()
