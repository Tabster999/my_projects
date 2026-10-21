"""
Checks on the pure-Rashba W = 300 nm result (scharf_abs_conv.py T3), transport geometry throughout
(finite centre, semi-infinite ribbons and leads; LDOS with decoupled probes, kappa/G with the weak probe).

  C1  The double peaks at L = 4 um: fine E_Z scan through one of them (kappa, G, local Andreev at
      E = 0) and the Majorana splitting from the end LDOS at the left peak, the dip and the right peak.
      Expected if the dip is the delta = 0 node: kappa -> 0 and Andreev -> 2 at the dip, and the
      splitting ~ Gamma / ~ 0 / ~ Gamma.
  C2  a = 10 vs 20 nm at L = 4 um: kappa, G, Andreev vs E_Z.
  C3  W = 300 nm at L = 8 um (a = 20 nm).
  C4  W = 800 nm (3 ABS) at L = 4 and 8 um (a = 20 nm).
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
import scharf_abs_conv as C
from scharf_abs import myf, DATA

PR = R.PR
N_WORKERS = 14


class ChannelPoint:
    """Picklable for myf.scan: E_Z (meV) -> [kappa, G, local Andreev, failed lead solves] at E = 0."""
    def __init__(self, q, phi=np.pi):
        self.q, self.phi = q, float(phi)

    def __call__(self, E_Z):
        n0 = myf.leads.N_UNRELIABLE
        J = myf.FourTerminalJunction(self.q.params_transport(E_Z=float(E_Z)))
        c = {k: float(np.asarray(v)[0])
             for k, v in myf.RGFFourTerminal(J, np.array([0.0])).channels_at_phi(self.phi, 'left').items()}
        return np.array([c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'],
                         myf.leads.N_UNRELIABLE - n0])


def channel_line(q, EZ, name, n_workers=N_WORKERS):
    """kappa, G, Andreev at phi = pi along E_Z (cached in plot_npz/scharf_absK_<name>.npz)."""
    import scharf_abs_checks
    res = myf.scan(scharf_abs_checks.ChannelPoint(q), {'E_Z': np.asarray(EZ)}, n_workers=n_workers,
                   save=str(DATA / f'scharf_absK_{name}.npz'))
    V = res['values']
    if np.nansum(V[:, 3]):
        print(f"    WARNING: {int(np.nansum(V[:, 3]))} failed lead solves in {name}", flush=True)
    return dict(E_Z=np.asarray(EZ), kappa=V[:, 0], G=V[:, 1], andreev=V[:, 2])


def splitting(q, EZ, half_window=0.02, n=161, eta=0.2e-3):
    """Lowest non-negative peak of the end LDOS(E) = delta / 2 (0 if unresolved below eta), in ueV."""
    E = np.linspace(-half_window, half_window, n)
    end, _ = C.ldos_spectrum(q, EZ, E, eta=eta)
    pk = C.peaks_pos(E, end)
    return (pk[0] * 1e3 if len(pk) else np.nan), E, end


def summary_line(label, t):
    pk = np.flatnonzero((t['kappa'][1:-1] > t['kappa'][:-2]) & (t['kappa'][1:-1] >= t['kappa'][2:])
                        & (t['kappa'][1:-1] > 0.3)) + 1
    print(f"  {label}: kappa peaks > 0.3 at E_Z = {np.round(t['E_Z'][pk], 3)}", flush=True)
    print(f"      heights {np.round(t['kappa'][pk], 3)},  |G|/kappa {np.round(np.abs(t['G'][pk]) / t['kappa'][pk], 3)}",
          flush=True)


# %% --- C1: the double peak at L = 4 um, W = 300 nm ---
if __name__ == "__main__":
    q4 = replace(PR, W=300.0, L=4000.0)
    EZ1 = np.round(np.linspace(0.12, 0.16, 41), 5)
    K1 = channel_line(q4, EZ1, 'C1_W300_L4000_a20')
    i_l = int(np.argmax(np.where(EZ1 < EZ1[np.argmin(K1['kappa'][5:-5]) + 5], K1['kappa'], -1)))
    i_d = int(np.argmin(K1['kappa'][5:-5])) + 5
    i_r = int(np.argmax(np.where(EZ1 > EZ1[i_d], K1['kappa'], -1)))
    print("C1  W = 300 nm, L = 4 um, a = 20 nm, phi = pi: fine E_Z scan through the double peak")
    print("    E_Z [meV]   kappa    |G|      Andreev   delta/2 from end LDOS [ueV]")
    SPL = {}
    for tag, i in (('left peak', i_l), ('dip', i_d), ('right peak', i_r)):
        d2, E, end = splitting(q4, EZ1[i])
        SPL[tag] = (EZ1[i], E, end)
        print(f"    {EZ1[i]:8.4f}  {K1['kappa'][i]:7.4f}  {abs(K1['G'][i]):7.4f}  {K1['andreev'][i]:7.4f}"
              f"   {d2:8.3f}    ({tag})", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(EZ1, K1['kappa'], 'k.-', label=r'$\kappa/\kappa_0$')
    axs[0].plot(EZ1, np.abs(K1['G']), 'r.-', label=r'$|G|/G_0$')
    axs[0].plot(EZ1, K1['andreev'] / 2, 'b.-', label=r'Andreev $/\,2e^2/h$')
    axs[0].axhline(0.5, color='0.5', ls='--', lw=0.6)
    axs[0].set_xlabel(r'$E_Z$ [meV]'); axs[0].legend(fontsize=8)
    axs[0].set_title('W = 300 nm, L = 4 $\\mu$m, $\\phi = \\pi$, E = 0', fontsize=9)
    for tag, (ez, E, end) in SPL.items():
        axs[1].plot(E * 1e3, end, label=f'{tag}, $E_Z$ = {ez:.4f}')
    axs[1].set_xlabel(r'$E$ [$\mu$eV]'); axs[1].set_ylabel(r'end LDOS, $\eta$ = 0.2 $\mu$eV')
    axs[1].legend(fontsize=8)
    plt.tight_layout(); A._save(fig, 'K_C1'); plt.close(fig)


# %% --- C2: a = 10 vs 20 nm at L = 4 um ---
if __name__ == "__main__":
    print("\nC2  W = 300 nm, L = 4 um: a = 20 vs 10 nm")
    EZ2 = R.ez_grid(q4, 121)
    K2 = {a: channel_line(replace(q4, a=a), EZ2, f'C2_W300_L4000_a{a:.0f}') for a in (20.0, 10.0)}
    for a, t in K2.items():
        summary_line(f"a = {a:.0f} nm", t)


# %% --- C3 / C4: longer junctions, and W = 800 nm ---
if __name__ == "__main__":
    print("\nC3/C4  longer junctions (a = 20 nm)")
    K3 = {}
    for W, L in ((300.0, 8000.0), (800.0, 4000.0), (800.0, 8000.0)):
        q = replace(PR, W=W, L=L)
        EZ = R.ez_grid(q, 121)
        t = channel_line(q, EZ, f'C34_W{W:.0f}_L{L:.0f}_a20')
        K3[(W, L)] = t
        summary_line(f"W = {W:.0f} nm, L = {L / 1000:.0f} um", t)

    rows = [(300.0, 4000.0, K2[20.0]), (300.0, 4000.0, K2[10.0])] + [(W, L, t) for (W, L), t in K3.items()]
    labels = ['W 300, L 4 um, a 20', 'W 300, L 4 um, a 10', 'W 300, L 8 um', 'W 800, L 4 um', 'W 800, L 8 um']
    fig, axs = plt.subplots(len(rows), 1, figsize=(10, 2.3 * len(rows)))
    for ax, (W, L, t), lab in zip(axs, rows, labels):
        ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
        ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
        ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
        ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
        ax.set_ylim(0, 1.05); ax.set_ylabel(lab, fontsize=8)
    axs[0].legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel(r'$E_Z$ [meV]   ($\phi = \pi$, E = 0, weak probe)')
    plt.tight_layout(); A._save(fig, 'K_C234'); plt.close(fig)
    print("done", flush=True)
