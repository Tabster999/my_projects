"""
The topological transition with 2-4 ABS (tuned by W), seen in the LDOS of the transport geometry
(finite centre, semi-infinite SC ribbons and leads; probes decoupled, tc_barr = 0, as in Scharf).
Pure Rashba (scharf_abs_rashba.PR), a = 20 nm, L = 2 um, phi = pi.

  1. E(phi) at E_Z = 0: the ABS count per W.
  2. phi = pi line from the cross-section: Q, gap at k = 0, gap over all k.
  3. LDOS(E, E_Z) at the junction END and in the MIDDLE (Scharf Fig. 8 vs field): the gap closes at
     the transition; a zero-energy peak at the end only = end Majoranas, at both = bulk states.
  4. Real-space LDOS(E = 0) at four fields: trivial / gap closing / best-gapped topological / high field.
Caches plot_npz/scharf_absT_*.npz.
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
from scharf_abs import myf, DATA

# %% --- SETUP ---
PR = R.PR                                         # pure Rashba, a = 20 nm, L = 2000 nm
WIDTHS = (300.0, 800.0, 1000.0)                   # 2, 3, 4 ABS
N_EZ = 41
E_LDOS = np.linspace(-1.0, 1.0, 81)               # in units of Delta
ETA = 0.05                                        # LDOS broadening in units of Delta (Scharf)
N_END = 3                                         # columns summed at the end / in the middle
EVERY = {300.0: 1, 800.0: 4, 1000.0: 4}           # column stride of the real-space maps (memory)
N_WORKERS = 10


class SpectrumPoint:
    """Picklable for myf.scan: E_Z (meV) -> (2, n_E): LDOS(E) at the end and in the middle, phi = pi."""
    def __init__(self, q, E_meV, eta_meV, n_end=N_END, chunk=6):
        self.q, self.E, self.eta, self.n_end, self.chunk = q, np.asarray(E_meV), eta_meV, n_end, chunk

    def __call__(self, E_Z):
        q, n = self.q, self.n_end
        p = replace(q.params(E_Z=float(E_Z), phi=np.pi), eta=self.eta / q.t, tc_barr=0.0)
        J = myf.FourTerminalJunction(p)
        mid = q.n_along // 2
        cols = list(range(n)) + list(range(mid - n // 2, mid - n // 2 + n))
        out = np.empty((2, len(self.E)))
        for k in range(0, len(self.E), self.chunk):
            e, h = myf.LocalGreen(J, self.E[k:k + self.chunk] / q.t, cols=cols).ldos(np.pi)
            D = (e + h).sum(2)                                  # (n_E, n_cols)
            out[0, k:k + self.chunk] = D[:, :n].sum(1)
            out[1, k:k + self.chunk] = D[:, n:].sum(1)
        return out


def spectrum_map(q, EZ, name):
    import scharf_abs_transition
    f = scharf_abs_transition.SpectrumPoint(q, E_LDOS * q.Delta, ETA * q.Delta)
    res = myf.scan(f, {'E_Z': np.asarray(EZ)}, n_workers=N_WORKERS,
                   save=str(DATA / f'scharf_absT_spec_{name}.npz'))
    return res['values']                                        # (n_EZ, 2, n_E)


def ldos_map_E0(q, EZ, eta_meV, every):
    """Real-space LDOS(E = 0) at phi = pi: (y_nm, x_nm, A[y, x]) over every `every`-th column."""
    p = replace(q.params(E_Z=float(EZ), phi=np.pi), eta=eta_meV / q.t, tc_barr=0.0)
    cols = list(range(0, q.n_along, every))
    e, h = myf.LocalGreen(myf.FourTerminalJunction(p), np.array([0.0]), cols=cols).ldos(np.pi)
    x_nm, y_nm = q.coords(cols)
    return y_nm, x_nm, (e + h)[0]


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    for W in WIDTHS:
        q = replace(PR, W=W)
        n_abs = A.count_abs(q, below=0.9)[0]
        line = R.phase_line(q, n=N_EZ)                          # cached scharf_absR_line_W*.npz
        EZ = line['E_Z']
        spec = spectrum_map(q, EZ, f'W{W:.0f}')
        topo = line['Q'] < 0
        i_t = int(np.argmax(topo))                              # first topological field
        i_b = int(np.argmax(np.where(topo, line['gap'], -1)))   # best-gapped topological field
        picks = {'trivial': max(i_t - 3, 0), 'gap closing': max(i_t - 1, 0),
                 'best topological': i_b, 'high field': len(EZ) - 4}
        maps = {lab: (EZ[i],) + ldos_map_E0(q, EZ[i], ETA * q.Delta, EVERY[W]) for lab, i in picks.items()}
        RES[W] = dict(q=q, n_abs=n_abs, line=line, spec=spec, maps=maps)
        end0 = spec[:, 0, len(E_LDOS) // 2]
        mid0 = spec[:, 1, len(E_LDOS) // 2]
        print(f"W = {W:.0f} nm ({n_abs} ABS): Q = -1 for E_Z in {EZ[topo].min():.3f}..{EZ[topo].max():.3f} meV;"
              f" best gap over all k {line['gap'][i_b] * 1e3:.1f} ueV at E_Z = {EZ[i_b]:.3f}", flush=True)
        for lab, (ez, y, x, Am) in maps.items():
            prof = Am.sum(1)
            j = int(np.argmin(np.abs(EZ - ez)))
            print(f"    {lab:17s} E_Z = {ez:.3f}: LDOS(E=0) end/mid (spectrum cols) = {end0[j] / mid0[j]:5.2f},"
                  f"  max/mean along y = {prof.max() / prof.mean():4.2f}", flush=True)

    # --- figure 1: per W -- ABS at E_Z = 0, gaps along phi = pi, LDOS(E, E_Z) end and middle ---
    PHI = np.linspace(0, 2 * np.pi, 101)
    fig, axs = plt.subplots(len(RES), 4, figsize=(19, 3.8 * len(RES)))
    for row, (W, r) in zip(axs, RES.items()):
        q, line, spec = r['q'], r['line'], r['spec']
        EZ, topo = line['E_Z'], line['Q'] < 0
        row[0].plot(PHI / np.pi, A.abs_spectrum(q, PHI, E_Z=0.0, n_levels=20) / q.Delta, 'k-', lw=0.8)
        row[0].axhline(0.9, color='r', ls=':', lw=0.8)
        row[0].set_ylim(-1.05, 1.05); row[0].set_xlabel(r'$\phi/\pi$'); row[0].set_ylabel(r'$E/\Delta$')
        row[0].set_title(f'W = {W:.0f} nm: {r["n_abs"]} ABS ($E_Z$ = 0, $k$ = 0)', fontsize=9)

        top = 1.05 * np.nanmax(line['gap0']) * 1e3
        row[1].fill_between(EZ, 0, top, where=topo, color='0.88', step='mid', label='Q = -1')
        row[1].plot(EZ, line['gap0'] * 1e3, 'b-', label='gap at k = 0')
        row[1].plot(EZ, line['gap'] * 1e3, 'r-', label='gap over all k')
        row[1].set_ylim(0, top); row[1].set_xlabel(r'$E_Z$ [meV]'); row[1].set_ylabel(r'gap [$\mu$eV]')
        row[1].legend(fontsize=7); row[1].set_title(r'$\phi = \pi$ (infinite junction)', fontsize=9)

        for ax, k, ttl in ((row[2], 0, 'end'), (row[3], 1, 'middle')):
            im = ax.pcolormesh(EZ, E_LDOS, spec[:, k, :].T, shading='nearest', cmap='cividis')
            fig.colorbar(im, ax=ax, label='LDOS')
            for e in (EZ[topo].min(), EZ[topo].max()):
                ax.axvline(e, color='w', ls='--', lw=0.8)
            ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$E/\Delta$')
            ax.set_title(f'LDOS at the {ttl} ($\\phi=\\pi$, $\\eta$ = {ETA:g} $\\Delta$); dashed: Q = -1 edges',
                         fontsize=9)
    plt.tight_layout(); A._save(fig, 'T_transition'); plt.close(fig)

    # --- figure 2: real-space LDOS(E = 0) at four fields per W ---
    fig, axs = plt.subplots(len(RES), 5, figsize=(20, 4.2 * len(RES)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.6]))
    for row, (W, r) in zip(axs, RES.items()):
        for ax, (lab, (ez, y, x, Am)) in zip(row[:4], r['maps'].items()):
            ax.pcolormesh(x, y, Am, shading='nearest', cmap='cividis')
            ax.set_title(f'W {W:.0f}: {lab}\n$E_Z$ = {ez:.3f} meV', fontsize=9)
            ax.set_xlabel('x [nm]')
            row[4].plot(y, Am.sum(1) / Am.sum(1).max(), label=f'{lab} ({ez:.3f})')
        row[0].set_ylabel('y [nm] (along)')
        row[4].set_xlabel('y [nm]'); row[4].set_ylabel(r'$\sum_x$ LDOS(E=0) (norm.)')
        row[4].legend(fontsize=7)
    fig.suptitle(f'LDOS(E = 0), phi = pi, eta = {ETA:g} Delta, pure Rashba, L = {PR.L:.0f} nm', fontsize=9)
    plt.tight_layout(); A._save(fig, 'T_ldos_E0'); plt.close(fig)
    print("done", flush=True)
