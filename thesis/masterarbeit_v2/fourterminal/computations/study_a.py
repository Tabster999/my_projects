"""
STUDY: does the result converge as the lattice constant a shrinks?  (W = 100 nm, L = 2 um, pure Rashba)

  1. Scaling check: every lattice parameter maps back to the same physical value for each a.
  2. Why tc_barr cannot be used: a weakened bond's transmission falls like a^2 (1D model);
     a physical barrier (height V_barr over d_barr nm) converges.
  3. Cross-section: ABS count, gap at k = 0 and Delta_top vs a.
  4. Transport (phi = pi, E = 0) with a physical barrier, tc_barr_transport = 1: kappa, G, Andreev
     vs E_Z for each a -- peak heights, neutrality and positions.

Expectation: heights and neutrality converge quickly; peak POSITIONS converge like a^2, because the
splitting oscillates as cos(k_F L) and the lattice k_F differs from the continuum one by ~(k_F a)^2/24.
Run the cells top to bottom; everything is cached in plot_npz/, so reruns are fast.
The `if __name__ == "__main__":` lines stop the parallel workers from re-running the cells
(they re-import this file); in the interactive window the cells run normally.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st
from functools import partial

A_LIST = (20.0, 10.0, 5.0, 4.0, 2.5)                       # nm
BARRIERS = {'medium': (6.0, 20.0), 'weak': (6.0, 40.0)}      # (V_barr meV, d_barr nm); 1D T ~ 0.30 / 0.04
A_TRANSPORT = {'medium': (20.0, 10.0, 5.0, 4.0), 'weak': (20.0, 10.0, 5.0, 4.0)}
# a = 2.5 nm: ~500 s per point and several GB per worker (14 workers ran out of memory) -> cell 5,
# a narrow window around the neutral peaks with 4 workers
EZ_FINE = np.round(np.arange(0.77, 0.93 + 1e-9, 0.002), 5)
N_WORKERS_SMALL_A = 4
EZ_TR = np.round(np.arange(0.55, 1.25 + 1e-9, 0.002), 5)    # meV, the resonance region of W = 100 nm
EZ_XS = (0.3, 0.593)                                         # meV, cross-section points


def junction(a, barrier=None):
    """W = 100 nm, L = 2 um at lattice constant a, optionally with a physical probe barrier."""
    q = st.case(100.0, a=a, tc=1.0 if barrier else 0.2)
    if barrier:
        q = st.replace(q, V_barr=BARRIERS[barrier][0], d_barr=BARRIERS[barrier][1])
    return q


def contact_1d(a, tau=1.0, V_b=0.0, d_b=0.0, mu_N=3.0, mu_C=0.7, n_pad=4):
    """Transmission at E = 0 of a 1D chain: lead (mu_N) | weak bond tau or barrier V_b over d_b | junction (mu_C)."""
    t = 76.1996 / 0.038 / (2 * a * a)                                    # hbar^2 / (2 m a^2), meV
    nb = int(round(d_b / a))
    mus = [mu_N] * n_pad + [mu_N - V_b] * nb + [mu_C] * n_pad
    H = np.diag([2 * t - m for m in mus]).astype(complex)
    for j in range(len(mus) - 1):
        H[j, j + 1] = H[j + 1, j] = -t
    if tau != 1.0:
        j = n_pad + nb - 1
        H[j, j + 1] = H[j + 1, j] = -tau * t
    sig = lambda mu: -t * np.exp(1j * np.arccos(1 - mu / (2 * t)))      # semi-infinite chain, E = 0
    S = np.zeros_like(H)
    S[0, 0], S[-1, -1] = sig(mu_N), sig(mu_C)
    G = np.linalg.inv(-H - S)
    return 4 * S[0, 0].imag * S[-1, -1].imag * abs(G[0, -1]) ** 2


# %% --- 1. scaling check ---
if __name__ == "__main__":
    print(" a [nm]  t [meV]   mu_S  mu_N  mu_C  Delta  alpha  E_Z   eta_tr [meV]   W     L    barrier (meV, nm)  E_T")
    for a in A_LIST:
        q = junction(a, 'medium')
        p, t = q.params_transport(E_Z=0.3), q.t
        print(f"{a:6.1f}  {t:7.2f}   {p.mu_s * t:.2f}  {p.mu_n * t:.2f}  {p.mu_c * t:.2f}  {p.delta * t:.3f}  {p.alpha * a * t:5.1f}"
              f"  {p.Bxy_c * t:.2f}   {p.eta * t:.1e}   {p.ny * a:4.0f}  {p.nx * a:5.0f}   ({p.barrier_V * t:.1f}, {p.barrier_n * a:.0f})"
              f"         {q.E_T:.3f}")


# %% --- 2. the probe contact in 1D: weak bond vs physical barrier ---
if __name__ == "__main__":
    print("\n a [nm]   T(weak bond 0.2)   T(weak bond 0.4)   T(barrier 6 meV, 20 nm)   T(barrier 6 meV, 40 nm)")
    for a in A_LIST + (1.0,):
        print(f"{a:6.1f}       {contact_1d(a, tau=0.2):.4f}             {contact_1d(a, tau=0.4):.4f}"
              f"               {contact_1d(a, V_b=6, d_b=20):.4f}                    {contact_1d(a, V_b=6, d_b=40):.4f}")


# %% --- 3. cross-section vs a ---
if __name__ == "__main__":
    print("\n a [nm]  ABS   E_Z [meV]   Q   gap(k=0) [ueV]   Delta_top [ueV]")
    for a in A_LIST:
        q = junction(a)
        V = st.run_scan(partial(st.phase_point, q, ks=st.k_grid(a)),
                        {'E_Z': np.array(EZ_XS), 'phi': np.array([np.pi])}, cache=f'scharf_absA_xs_a{a:g}')[:, 0]
        for ez, (Q, g0, g, km) in zip(EZ_XS, V):
            print(f"{a:6.1f}  {st.count_abs(q):3d}   {ez:7.3f}   {Q:+.0f}   {g0 * 1e3:12.2f}   {g * 1e3:13.2f}")


# %% --- 4. transport vs a, physical barrier ---
if __name__ == "__main__":
    RES = {}
    for barrier, a_list in A_TRANSPORT.items():
        print(f"\nbarrier '{barrier}' (V = {BARRIERS[barrier][0]} meV, d = {BARRIERS[barrier][1]} nm), phi = pi, E = 0")
        print(" a [nm]  peaks>0.3  Majorana-like  >0.55   Majorana-like heights            at E_Z [meV]")
        for a in a_list:
            t = st.transport_line(junction(a, barrier), EZ_TR, cache=f'scharf_absA_tr_{barrier}_a{a:g}')
            RES[(barrier, a)] = t
            s = st.peak_stats(t)
            pk = st.kappa_peaks(t)
            good = pk[st.is_majorana_like(t, pk)]
            print(f"{a:6.1f}  {s['n']:6d}     {s['good']:6d}       {s['over']:4d}    {np.round(t['kappa'][good], 3)}   {np.round(t['E_Z'][good], 3)}")

    for barrier, a_list in A_TRANSPORT.items():
        fig, axs = plt.subplots(len(a_list), 1, figsize=(11, 2.1 * len(a_list)), sharex=True)
        for ax, a in zip(axs, a_list):
            t = RES[(barrier, a)]
            ax.plot(t['E_Z'], t['kappa'], 'k-', lw=1, label=r'$\kappa/\kappa_0$')
            ax.plot(t['E_Z'], np.abs(t['G']), 'r-', lw=0.8, label=r'$|G|/G_0$')
            ax.plot(t['E_Z'], t['andreev'] / 2, 'b:', lw=0.8, label=r'Andreev $/\,2e^2/h$')
            ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
            ax.set_ylim(0, 1.05); ax.set_ylabel(f'a = {a:g} nm', fontsize=9)
        axs[0].legend(fontsize=7, ncol=3)
        axs[-1].set_xlabel(rf'$E_Z$ [meV]   (W = 100 nm, L = 2 $\mu$m, $\phi=\pi$, barrier {barrier}: '
                           rf'{BARRIERS[barrier][0]:g} meV, {BARRIERS[barrier][1]:g} nm)')
        plt.tight_layout(); st.save_fig(fig, f'a_transport_{barrier}'); plt.show()


# %% --- 5. a = 2.5 nm, narrow window around the neutral peaks (slow: hours, 4 workers) ---
if __name__ == "__main__":
    print("\nmedium barrier, E_Z 0.77-0.93 meV: neutral peak positions vs a")
    FINE = {}
    for a in (5.0, 4.0, 2.5):
        q = junction(a, 'medium')
        if a == 2.5:
            FINE[a] = st.transport_line(q, EZ_FINE, cache='scharf_absA_trfine_medium_a2.5', n_workers=N_WORKERS_SMALL_A)
        else:                                                       # reuse the full lines
            t = RES[('medium', a)]
            m = (t['E_Z'] >= EZ_FINE[0] - 1e-9) & (t['E_Z'] <= EZ_FINE[-1] + 1e-9)
            FINE[a] = {k: v[m] for k, v in t.items()}
        pk = st.kappa_peaks(FINE[a])
        print(f"  a = {a:4.1f} nm: peaks at E_Z = {np.round(FINE[a]['E_Z'][pk], 3)},  kappa {np.round(FINE[a]['kappa'][pk], 3)},"
              f"  |G|/kappa {np.round(np.abs(FINE[a]['G'][pk]) / FINE[a]['kappa'][pk], 3)}")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for a, t in FINE.items():
        ax.plot(t['E_Z'], t['kappa'], lw=1, label=f'a = {a:g} nm')
    ax.axhline(0.5, color='0.5', ls='--', lw=0.6)
    ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$\kappa/\kappa_0$'); ax.legend(fontsize=8)
    ax.set_title('medium barrier, W = 100 nm, L = 2 um, phi = pi: convergence of the peak positions', fontsize=9)
    plt.tight_layout(); st.save_fig(fig, 'a_fine_medium'); plt.show()
