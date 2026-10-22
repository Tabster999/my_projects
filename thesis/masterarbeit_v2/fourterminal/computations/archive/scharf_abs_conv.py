"""
Pure Rashba (scharf_abs_rashba.PR): is the 1 ABS -> 2 ABS result converged, and is L the problem?

  T1  lattice constant a = 20 / 10 / 5 nm at FIXED physical W, L: ABS count, Q, gap(k=0),
      gap over all k at the best points of scharf_abs_rashba (cross-section only, cheap).
  T2  W = 300 nm at a = 10 nm: LDOS (E = 0 profile, E-resolved at end / middle) and kappa/G vs E_Z.
  T3  W = 300 nm with L = 4000 nm (a = 20 nm): does the splitting fall and the pair localise?

T2/T3 use the transport geometry of scharf_ldos: finite centre, SEMI-INFINITE SC ribbons and normal
leads (myf.LocalGreen / RGF).  LDOS with the probes decoupled (tc_barr = 0, Scharf); kappa/G with the
weak probe.  Lengths stay physical (W sets E_T and the ABS count, L the MBS overlap); the a = 5 nm
check is done on the cross-section only.
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

PR = R.PR
ETA_FINE = R.ETA_FINE                             # meV, LDOS(E = 0) that resolves the MBS splitting
ETA_SPEC = 1.0e-3                                 # meV, broadening of the E-resolved LDOS


def k_grid(a):
    """Same physical k range as A.K_GRID (dense up to 1.5 / 20 nm^-1), in units of 1/a."""
    s = a / 20.0
    return np.r_[np.linspace(0, 1.5 * s, 76), np.linspace(1.5 * s + 0.1, np.pi, 12)]


def ldos_profile(q, EZ, eta, every=1, E=0.0):
    """LDOS at energy E (meV) summed across the normal strip, vs y, phi = pi (decoupled probes)."""
    p = replace(q.params(E_Z=EZ, phi=np.pi), eta=eta / q.t, tc_barr=0.0)
    cols = list(range(0, q.n_along, every))
    e, h = myf.LocalGreen(myf.FourTerminalJunction(p), np.array([E / q.t]), cols=cols).ldos(np.pi)
    return (np.array(cols) + 0.5) * q.a - q.L / 2, (e + h)[0].sum(1)


def ldos_spectrum(q, EZ, E_meV, eta=ETA_SPEC, n_end=3, chunk=12):
    """LDOS(E) summed over the first n_end columns (end) and the n_end middle columns, phi = pi."""
    p = replace(q.params(E_Z=EZ, phi=np.pi), eta=eta / q.t, tc_barr=0.0)
    J = myf.FourTerminalJunction(p)
    mid = q.n_along // 2
    cols = list(range(n_end)) + list(range(mid - n_end // 2, mid - n_end // 2 + n_end))
    end, cen = [], []
    for k in range(0, len(E_meV), chunk):
        e, h = myf.LocalGreen(J, np.asarray(E_meV[k:k + chunk]) / q.t, cols=cols).ldos(np.pi)
        D = (e + h).sum(2)
        end.append(D[:, :n_end].sum(1)); cen.append(D[:, n_end:].sum(1))
    return np.concatenate(end), np.concatenate(cen)


def peaks_pos(E, D):
    """Non-negative-energy local maxima of D(E), lowest first."""
    i = np.flatnonzero((D[1:-1] > D[:-2]) & (D[1:-1] >= D[2:])) + 1
    return E[i][E[i] >= 0]


def device_report(q, EZ, label, every=1):
    """End/middle contrast of LDOS(E = 0) (Scharf eta and fine eta) and the low-energy peaks."""
    out = {}
    for eta in (ETA_FINE, R.ETA_SCHARF):
        y, prof = ldos_profile(q, EZ, eta, every=every)
        out[eta] = (y, prof)
    E = np.linspace(-0.08, 0.08, 321)
    end, cen = ldos_spectrum(q, EZ, E)
    pk = peaks_pos(E, end)
    yF, pF = out[ETA_FINE]
    print(f"  {label}: E_Z = {EZ:.3f} meV   end/mid LDOS(E=0) = {pF[0] / pF[len(pF) // 2]:5.1f} (eta 0.5 ueV),"
          f" {out[R.ETA_SCHARF][1][0] / out[R.ETA_SCHARF][1][len(pF) // 2]:5.1f} (Scharf eta)"
          f"   end-LDOS peaks at E = {np.round(pk[:4] * 1e3, 2)} ueV", flush=True)
    return dict(y=yF, p=pF, E=E, end=end, cen=cen)


def transport_line(q, name, n=121, n_workers=14):
    EZ = R.ez_grid(q, n)
    f = myf.ThermalPoint(q.params_transport(E_Z=0.0), kT=0.0, phis=(np.pi, 0.0))
    res = myf.scan(f, {'Bxy_c': EZ / q.t}, n_workers=n_workers, save=str(DATA / f'scharf_absC_{name}.npz'))
    V = res['values']
    return dict(E_Z=EZ, kappa=V[:, 0], G=V[:, 1], kappa0=V[:, 2], n_bad=V[:, 4])


# %% --- T1: lattice-constant convergence of the cross-section ---
if __name__ == "__main__":
    print("T1  pure Rashba, phi = pi, cross-section (S strips 900 nm)")
    print("  W[nm]  a[nm]  ABS  E_Z[meV]   Q   gap(k=0)[ueV]  gap(all k)[ueV]  at k[1/nm]")
    for W, EZs in ((100.0, (0.3, 0.593)), (300.0, (0.15, 0.323))):
        for a in (20.0, 10.0, 5.0):
            q = replace(PR, W=W, a=a)
            n_abs = A.count_abs(q, below=0.9, phis=np.array([0.0]))[0]
            f = A.PhasePoint(q, ks=k_grid(a))
            for EZ in EZs:
                Q, g0, g, km = f(EZ, np.pi)
                print(f"  {W:5.0f}  {a:5.0f}  {n_abs:3d}  {EZ:7.3f}  {Q:+3.0f}  {g0 * 1e3:12.2f}  {g * 1e3:14.2f}"
                      f"   {km / a:9.4f}", flush=True)


# %% --- T2: W = 300 nm at a = 10 nm, finite device and transport ---
if __name__ == "__main__":
    print("\nT2  W = 300 nm, L = 2000 nm: a = 20 vs 10 nm")
    T2 = {}
    for a in (20.0, 10.0):
        q = replace(PR, W=300.0, a=a)
        d = device_report(q, 0.323, f"a = {a:.0f} nm", every=1 if a == 20 else 4)
        T2[a] = dict(q=q, d=d, tr=transport_line(q, f'W300_L2000_a{a:.0f}'))


# %% --- T3: W = 300 nm, longer junction ---
if __name__ == "__main__":
    print("\nT3  W = 300 nm, a = 20 nm: L = 2000 vs 4000 nm")
    T3 = {}
    for L in (2000.0, 4000.0):
        q = replace(PR, W=300.0, L=L)
        d = device_report(q, 0.323, f"L = {L:.0f} nm", every=1 if L == 2000 else 2)
        T3[L] = dict(q=q, d=d, tr=transport_line(q, f'W300_L{L:.0f}_a20'))

    fig, axs = plt.subplots(2, 3, figsize=(18, 7))
    for row, (D, lab) in zip(axs, ((T2, 'a = {:.0f} nm'), (T3, 'L = {:.0f} nm'))):
        for key, d in D.items():
            q, dv = d['q'], d['d']
            row[0].plot(dv['y'] / q.L, dv['p'] / dv['p'].max(), label=lab.format(key))
            row[1].plot(dv['E'] * 1e3, dv['end'], label='end, ' + lab.format(key))
            row[1].plot(dv['E'] * 1e3, dv['cen'], ':', label='middle, ' + lab.format(key))
            row[2].plot(d['tr']['E_Z'], d['tr']['kappa'], lw=1, label=r'$\kappa$, ' + lab.format(key))
            row[2].plot(d['tr']['E_Z'], np.abs(d['tr']['G']), ':', lw=1, label=r'$|G|$, ' + lab.format(key))
        row[0].set_xlabel('y / L'); row[0].set_ylabel('LDOS(E=0), normal strip (norm.)')
        row[0].set_title(r'W = 300 nm, $E_Z$ = 0.323 meV, $\phi = \pi$', fontsize=9)
        row[1].set_xlabel(r'$E$ [$\mu$eV]'); row[1].set_ylabel(f'LDOS, $\\eta$ = {ETA_SPEC * 1e3:g} $\\mu$eV')
        row[1].set_title('E-resolved LDOS at the end and in the middle', fontsize=9)
        row[2].axhline(0.5, color='0.5', ls='--', lw=0.6)
        row[2].set_ylim(0, 1.05); row[2].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)')
        for ax in row:
            ax.legend(fontsize=7)
    fig.suptitle('pure Rashba, W = 300 nm (2 ABS)', fontsize=9)
    plt.tight_layout(); A._save(fig, 'C_W300'); plt.show()
