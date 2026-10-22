"""
kappa and G at finite temperature on the genuine Majorana resonances found at T = 0
(pure Rashba, a = 20 nm, tc_barr_transport = 0.2, phi = pi; phi = 0 for the contrast).

Cases (the clean ones from scharf_abs_tc.py / scharf_abs_delta.py):
    W 100, L 2 um, Delta 0.25 (1 ABS)    W 100, L 2 um, Delta 0.5 (2 ABS)    W 300, L 4 um (2 ABS)
Per case: the two highest Majorana-like T = 0 peaks (0.45 <= kappa <= 0.55, |G|/kappa < 0.05),
re-located on a fine E = 0 scan, then the channels on a fixed energy grid (0.05 ueV steps below
3 ueV, coarser to 0.1 meV) at N_WIN fields around each peak.  The thermal integrals for every T
reuse those channels (same kernels as myf.linear_response_nodes_phs, trapezoid in x = E/kT).
The quantity an experiment sees is the best kappa_T over the field window at each T.
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
from scharf_abs import myf, DATA
from my_functions.transport import _kernels
_trapz = getattr(np, 'trapezoid', None) or np.trapz

PR = R.PR
T_MK = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
K_B = 0.0861733                                   # meV / K
E_GRID = np.unique(np.concatenate([np.linspace(0.0, 3e-3, 61), np.linspace(3e-3, 20e-3, 35),
                                   np.geomspace(20e-3, 0.1, 10)]))   # meV
CASES = {'W100 L2 (1 ABS)': (replace(PR, W=100.0, L=2000.0), 'scharf_absTC_W100_L2_tc0.2.npz'),
         'W100 L2 D0.5 (2 ABS)': (D.case_pp(100.0, 0.5, 16.0), 'scharf_absK_D3_W100_D0.5_a16.npz'),
         'W300 L4 (2 ABS)': (replace(PR, W=300.0, L=4000.0), 'scharf_absTC_W300_L4_tc0.2.npz')}
N_PEAKS, N_FINE, N_WIN = 2, 41, 7
N_WORKERS = 14
KEYS = ('ee', 'hh', 'eh_cross', 'he_cross', 'eh_local', 'he_local')


class EPoint:
    """Picklable for myf.scan: (E_Z, E) in meV -> the six channels at phi = pi, then at phi = 0."""
    def __init__(self, q):
        self.q = q

    def __call__(self, E_Z, E):
        q = self.q
        rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=float(E_Z))),
                                  np.array([float(E) / q.t]))
        out = []
        for ph in (np.pi, 0.0):
            c = rgf.channels_at_phi(ph, 'left')
            out += [float(np.asarray(c[k])[0]) for k in KEYS]
        return np.array(out)


def majorana_peaks(t, n=N_PEAKS):
    k, G = t['kappa'], t['G']
    pk = np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > 0.3)) + 1
    good = pk[(k[pk] >= 0.45) & (k[pk] <= 0.55) & (np.abs(G[pk]) / k[pk] < 0.05)]
    return good[np.argsort(k[good])[::-1][:n]]


def thermal(ch, kT, E=E_GRID):
    """(kappa, G, Andreev) from channels on the grid E (E >= 0, particle-hole folded) at kT in meV."""
    S_th = ch['ee'] + ch['he_cross'] + ch['hh'] + ch['eh_cross']
    S_el = ch['ee'] - ch['he_cross'] + ch['hh'] - ch['eh_cross']
    S_a = ch['eh_local'] + ch['he_local']
    if kT == 0:
        return S_th[0] / 2, S_el[0] / 2, S_a[0]
    x = np.asarray(E) / kT
    m = x <= 16.0
    W_th, W_el = _kernels(x[m])
    return _trapz(W_th * S_th[m], x[m]), _trapz(W_el * S_el[m], x[m]), 2 * _trapz(W_el * S_a[m], x[m])


def run_case(name, q, cache):
    import scharf_abs_T
    d = np.load(DATA / cache)
    EZ0 = d['axis_E_Z']
    t0 = dict(E_Z=EZ0, kappa=d['values'][:, 0], G=d['values'][:, 1])
    step = EZ0[1] - EZ0[0]
    out = []
    for i in majorana_peaks(t0):
        tag = f"{name.split(' (')[0].replace(' ', '_')}_EZ{EZ0[i]:.4f}"
        fine = K.channel_line(q, np.round(np.linspace(EZ0[i] - 1.5 * step, EZ0[i] + 1.5 * step, N_FINE), 6),
                              f'T_fine_{tag}')
        j = int(np.argmax(np.where(np.abs(fine['G']) < 0.05 * fine['kappa'], fine['kappa'], -1)))
        ez_pk = fine['E_Z'][j]
        hw = max(3 * (fine['E_Z'][1] - fine['E_Z'][0]), 0.25 * step)
        win = np.round(np.linspace(ez_pk - hw, ez_pk + hw, N_WIN), 6)
        res = myf.scan(scharf_abs_T.EPoint(q), {'E_Z': win, 'E': E_GRID}, n_workers=N_WORKERS,
                       save=str(DATA / f'scharf_absTT_{tag}.npz'))
        V = res['values']                                       # (N_WIN, n_E, 12)
        chs = [[{k: V[a, :, b * 6 + n] for n, k in enumerate(KEYS)} for b in range(2)] for a in range(N_WIN)]
        out.append(dict(EZ=ez_pk, win=win, fine=fine, chs=chs))
    return out


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    for name, (q, cache) in CASES.items():
        RES[name] = run_case(name, q, cache)
        for pk in RES[name]:
            print(f"\n{name}, peak at E_Z = {pk['EZ']:.4f} meV (window +-{pk['win'][-1] - pk['EZ']:.4f})", flush=True)
            print("   T [mK]   best kappa(pi)  at E_Z     G(pi)    kappa(0)  contrast   Andreev(pi)", flush=True)
            pk['table'] = []
            for T in T_MK:
                kT = K_B * T * 1e-3
                r = np.array([[thermal(c[b], kT) for b in range(2)] for c in pk['chs']])   # (N_WIN, 2, 3)
                a = int(np.argmax(r[:, 0, 0]))
                pk['table'].append((T, r[a, 0, 0], pk['win'][a], r[a, 0, 1], r[a, 1, 0], r[a, 0, 2]))
                print(f"   {T:6.1f}   {r[a, 0, 0]:9.4f}     {pk['win'][a]:.4f}   {r[a, 0, 1]:8.4f}  {r[a, 1, 0]:8.4f}"
                      f"  {r[a, 0, 0] - r[a, 1, 0]:8.4f}   {r[a, 0, 2]:8.4f}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, peaks in RES.items():
        for pk in peaks:
            tb = np.array(pk['table'])
            lab = f"{name}, $E_Z$ = {pk['EZ']:.3f}"
            l, = axs[0].plot(tb[:, 0], tb[:, 1], 'o-', label=lab)
            axs[0].plot(tb[:, 0], np.abs(tb[:, 3]), 's:', color=l.get_color(), ms=3)
            axs[1].plot(tb[:, 0], tb[:, 1] - tb[:, 4], 'o-', color=l.get_color(), label=lab)
    axs[0].axhline(0.5, color='0.5', ls='--', lw=0.6)
    axs[0].set_xlabel('T [mK]'); axs[0].set_ylabel(r'best $\kappa/\kappa_0$ ($\phi=\pi$); dotted: $|G|/G_0$')
    axs[1].set_xlabel('T [mK]'); axs[1].set_ylabel(r'$\kappa(\pi) - \kappa(0)$')
    axs[0].legend(fontsize=7)
    fig.suptitle('pure Rashba, a = 20 nm, tc_barr = 0.2: Majorana resonances at finite temperature', fontsize=9)
    plt.tight_layout(); A._save(fig, 'T_finiteT'); plt.close(fig)
    print("done", flush=True)
