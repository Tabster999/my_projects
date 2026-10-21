"""
Does the probe coupling tc_barr_transport recover a Majorana signature for wide junctions, the way a
longer L did (W = 300 nm: no at L = 2 um, yes at L >= 4 um)?
kappa, G, local Andreev and det r of the left lead (Fulga et al., PRB 83, 155429) vs E_Z at phi = pi,
E = 0, pure Rashba, a = 20 nm, for tc_barr_transport = 0.05 ... 1.

Expectation (two-Majorana resonance, kappa = 2 d^2 G^2 / (d^2 + G^2)^2): a larger Gamma reaches the
splitting delta of a long xi_M, so the peak height can return to 1/2; but once Gamma ~ E_1 (next
level, ~ E_T) the probe couples to the extra ABS too, giving kappa > 1/2 and G != 0.
Majorana signature per peak: 0.45 <= kappa <= 0.55 and |G|/kappa < 0.05.
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
import scharf_abs_delta as D
from scharf_abs import myf, DATA
from scharf_disorder import _det_r

PR = R.PR
TCS = (0.05, 0.1, 0.2, 0.5, 1.0)
CASES = {'W100 L2 (1 ABS, ref)': replace(PR, W=100.0, L=2000.0),
         'W300 L2 (2 ABS)': replace(PR, W=300.0, L=2000.0),
         'W300 L4 (2 ABS, works)': replace(PR, W=300.0, L=4000.0),
         'W800 L4 (3 ABS)': replace(PR, W=800.0, L=4000.0),
         'W300 D0.75 L4 (4 ABS)': replace(D.case_pp(300.0, 0.75, 16.0), L=4000.0)}
N_EZ = 121
N_WORKERS = 14


class TcPoint:
    """Picklable for myf.scan: E_Z (meV) -> [kappa, G, local Andreev, det r, failed lead solves] at E = 0."""
    def __init__(self, q, phi=np.pi):
        self.q, self.phi = q, float(phi)

    def __call__(self, E_Z):
        n0 = myf.leads.N_UNRELIABLE
        rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(self.q.params_transport(E_Z=float(E_Z))), np.array([0.0]))
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(self.phi, 'left').items()}
        return np.array([c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'],
                         _det_r(rgf, self.phi), myf.leads.N_UNRELIABLE - n0])


def tc_line(q, tc, name):
    import scharf_abs_tc
    qt = replace(q, tc_barr_transport=tc)
    EZ = R.ez_grid(q, N_EZ)
    res = myf.scan(scharf_abs_tc.TcPoint(qt), {'E_Z': EZ}, n_workers=N_WORKERS,
                   save=str(DATA / f'scharf_absTC_{name}_tc{tc:g}.npz'))
    V = res['values']
    if np.nansum(V[:, 4]):
        print(f"    WARNING: {int(np.nansum(V[:, 4]))} failed lead solves in {name}, tc {tc:g}", flush=True)
    return dict(E_Z=EZ, kappa=V[:, 0], G=V[:, 1], andreev=V[:, 2], det=V[:, 3])


def stats(t):
    k, G = t['kappa'], t['G']
    pk = np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > 0.3)) + 1
    r = np.abs(G[pk]) / k[pk]
    good = (k[pk] >= 0.45) & (k[pk] <= 0.55) & (r < 0.05)
    return dict(n=len(pk), good=int(good.sum()), over=int((k[pk] > 0.55).sum()),
                kmax=float(k.max()), rmed=float(np.median(r)) if len(pk) else np.nan,
                det_neg=float(np.mean(t['det'] < 0)))


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    print("case                      tc    peaks>0.3  Majorana-like  kappa>0.55  max kappa  median |G|/kappa"
          "  frac(det r < 0)", flush=True)
    for name, q in CASES.items():
        tag = name.split(' (')[0].replace(' ', '_')
        for tc in TCS:
            t = tc_line(q, tc, tag)
            RES[(name, tc)] = t
            s = stats(t)
            print(f"{name:24s} {tc:5.2f}   {s['n']:6d}     {s['good']:6d}        {s['over']:6d}     {s['kmax']:6.3f}"
                  f"      {s['rmed']:7.3f}          {s['det_neg']:.2f}", flush=True)

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
    axs[0, 0].legend(fontsize=6, ncol=1, loc='upper right')
    fig.suptitle(r'$\phi = \pi$, E = 0, pure Rashba, a = 20 nm; green bar: det r < 0', fontsize=9)
    plt.tight_layout(); A._save(fig, 'TC_scan'); plt.close(fig)
    print("done", flush=True)
