"""
STUDY: does the Majorana splitting follow  eps(E_Z) ~ A Delta_top e^{-L/xi_M} cos(k_osc L + phi0) ?
PREDICTION from the INFINITE cross-section, no fit: at E = 0 solve H(k) psi = 0 for complex
k = k_osc + i/xi_M and take the slowest-decaying solution (the Majorana tail).
MEASUREMENT on the FINITE junction (probes decoupled): eps = lowest peak of the end LDOS (tools.splitting).
W 100 nm, L 2 um, a = 20 nm, phi = pi.  Tests:
  1. between consecutive zeros of eps, k_osc L must advance by exactly pi (no free parameter);
  2. the envelope eps_max / (Delta_top e^{-L/xi_M}) must be constant;
  3. xi_M against the estimate hbar v_F / Delta_top.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from functools import partial
import tools as st

HBAR_VF = 51.43                                  # meV nm, lattice value at a = 20 (check_fermi.py)
EZ = np.round(np.arange(0.50, 1.3001, 0.005), 3)
CASE = st.case(100.0)


def bloch_blocks(p, n_S):
    """H0, H1 with H(k) = H0 + H1 e^{ik} + H1^dag e^{-ik} (nearest-neighbour hopping along the junction)."""
    Hp, Hm, Hh = (st.H_cross(p, k, n_S) for k in (0.0, np.pi, np.pi / 2))
    H0 = (Hp + Hm) / 2
    H1 = ((Hp - Hm) / 2 - 1j * (Hh - H0)) / 2
    return H0, H1


def evanescent_point(q, E_Z, phi=np.pi):
    """[xi_M (nm), k_osc (1/nm), Delta_top (meV), Q, k of the gap minimum (1/nm)] at one field."""
    p = q.params(E_Z=float(E_Z), phi=phi)
    n_S = st.n_S_of(q)
    H0, H1 = bloch_blocks(p, n_S)
    n = len(H0)
    I, Z = np.eye(n), np.zeros((n, n))
    # (H1 lam^2 + H0 lam + H1^dag) psi = 0 with lam = e^{ika}, linearised with chi = lam psi
    lam = sla.eig(np.block([[Z, I], [-H1.conj().T, -H0]]), np.block([[I, Z], [Z, H1]]), right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1 - 1e-9) & (np.abs(lam) > 1e-12)]
    i = np.argmax(np.abs(lam))                   # |lam| closest to 1 = slowest decay
    g0, gmin, kmin = st.gaps(p, n_S)
    return np.array([-q.a / np.log(np.abs(lam[i])), abs(np.angle(lam[i])) / q.a, gmin * q.t,
                     st.invariant(p, n_S), kmin / q.a])


def split_point(q, E_Z):
    """[eps (ueV)]: lowest end-LDOS peak of the closed junction."""
    return np.array([st.splitting(q, float(E_Z), half_window=0.025, n=201)[0]])


def zeros_of(EZ, eps, rel=0.3, half=0.04):
    """E_Z of the zeros of eps: local minima below rel x the largest eps within +-half meV."""
    out = []
    for i in range(1, len(EZ) - 1):
        if not (eps[i] <= eps[i - 1] and eps[i] <= eps[i + 1]):     # also skips NaN neighbours
            continue
        if eps[i] < rel * np.nanmax(eps[np.abs(EZ - EZ[i]) <= half]):
            out.append(EZ[i])
    return np.array(out)


# %% --- 1. PREDICTION: complex k of the infinite cross-section ---
if __name__ == "__main__":
    V = st.run_scan(partial(evanescent_point, CASE), {'E_Z': EZ}, 'scharf_absO_evanescent_W100')
    xi_M, k_osc, D_top, Q, k_gap = V.T
    topo = Q < 0
    print(f"Q = -1 for E_Z {EZ[topo].min():.3f} .. {EZ[topo].max():.3f} meV")
    print(" E_Z    Delta_top   xi_M   hbar v/Delta_top   k_osc   k(gap min)")
    for j in range(0, len(EZ), 10):
        print(f" {EZ[j]:.3f}  {D_top[j] * 1e3:6.1f} ueV  {xi_M[j]:6.0f}   {HBAR_VF / D_top[j]:6.0f} nm"
              f"        {k_osc[j]:.4f}   {k_gap[j]:.4f} 1/nm   Q {Q[j]:+.0f}")


# %% --- 2. MEASUREMENT: eps of the finite junction ---
if __name__ == "__main__":
    eps = st.run_scan(partial(split_point, CASE), {'E_Z': EZ}, 'scharf_absO_split_W100_L2')[:, 0]
    print(f"\nmeasured eps: {np.sum(np.isfinite(eps))}/{len(eps)} points, max {np.nanmax(eps[topo]):.2f} ueV")


# %% --- 3. COMPARISON ---
if __name__ == "__main__":
    L = CASE.L
    env = D_top * np.exp(-L / xi_M) * 1e3                       # ueV, prediction without prefactor
    z = zeros_of(EZ, np.where(topo, eps, np.nan))
    Phi = np.interp(z, EZ, k_osc) * L
    print(f"\nTEST 1: zeros of eps at {z}")
    print(f"        k_osc L advances by {np.round(np.diff(Phi) / np.pi, 2)} pi between them (model: 1)")
    print(f"        predicted spacing pi/(L dk/dE_Z): "
          f"{np.round(np.pi / (L * np.interp(z[:-1] + np.diff(z) / 2, EZ, np.gradient(k_osc, EZ))), 3)} meV,"
          f" measured {np.round(np.diff(z), 3)} meV")
    print("TEST 2: lobe   E_Z    eps_max [ueV]   Delta_top e^{-L/xi_M} [ueV]   ratio A")
    for lo, hi in zip(z[:-1], z[1:]):
        w = (EZ > lo) & (EZ < hi) & np.isfinite(eps)
        j = np.flatnonzero(w)[np.argmax(eps[w])]
        print(f"        {lo:.3f}-{hi:.3f}  {EZ[j]:.3f}   {eps[j]:6.2f}          {env[j]:6.2f}"
              f"                   {eps[j] / env[j]:.2f}")
    # best overall fit A |cos(k_osc L + phi0)| env: two parameters
    m = topo & np.isfinite(eps)
    best = None
    for phi0 in np.linspace(0, np.pi, 361):
        mod = env[m] * np.abs(np.cos(k_osc[m] * L + phi0))
        A = (eps[m] @ mod) / (mod @ mod)
        r = np.sqrt(np.mean((eps[m] - A * mod) ** 2))
        if best is None or r < best[0]:
            best = (r, A, phi0)
    r, A, phi0 = best
    print(f"FIT:    A = {A:.2f}, phi0 = {phi0 / np.pi:.2f} pi, rms residual {r:.2f} ueV"
          f" = {r / np.nanmax(eps[m]):.1%} of max eps")
    print(f"TEST 3: xi_M / (hbar v_F / Delta_top) = {np.round((xi_M / (HBAR_VF / D_top))[topo][::10], 2)}")


# %% --- 4. ZEROS SEEN BY TRANSPORT: det r -> -1 with the weak probe (study_a cache, a = 20) ---
if __name__ == "__main__":
    try:
        d = np.load(st.DATA / 'scharf_absA_tr_weak_a20.npz', allow_pickle=True)
        Et, det = d['axis_E_Z'], d['values'][:, 3]
        i = np.flatnonzero((det[1:-1] < det[:-2]) & (det[1:-1] <= det[2:]) & (det[1:-1] < -0.9)) + 1
        print(f"\ndet r minima (< -0.9), weak probe: {np.round(Et[i], 3)}")
    except FileNotFoundError:
        print("\nno weak-probe transport cache at a = 20")


# %% --- FIGURE ---
if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(EZ, eps, 'o', ms=3, label=r'$\varepsilon$ measured (end LDOS)')
    axs[0].plot(EZ, np.where(topo, A * env * np.abs(np.cos(k_osc * L + phi0)), np.nan), '-',
                label=rf'${A:.2f}\,\Delta_{{top}} e^{{-L/\xi_M}} |\cos(k_{{osc}}L + \phi_0)|$')
    axs[0].plot(EZ, np.where(topo, A * env, np.nan), '--', lw=1, label='envelope')
    for x in z:
        axs[0].axvline(x, color='k', lw=0.5, alpha=0.3)
    axs[0].set_ylabel(r'$\varepsilon$ [$\mu$eV]'); axs[0].legend(fontsize=8)
    axs[1].plot(EZ, np.where(topo, xi_M, np.nan), label=r'$\xi_M$ (complex $k$)')
    axs[1].plot(EZ, np.where(topo, HBAR_VF / D_top, np.nan), '--', label=r'$\hbar v_F/\Delta_{top}$')
    axs[1].set_ylabel('length [nm]'); axs[1].legend(fontsize=8)
    axs[2].plot(EZ, np.where(topo, k_osc, np.nan), label=r'$k_{osc}$ (complex $k$)')
    axs[2].plot(EZ, np.where(topo, k_gap, np.nan), '--', label=r'$k$ of the gap minimum')
    axs[2].set_ylabel(r'$k$ [1/nm]'); axs[2].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[2].legend(fontsize=8)
    plt.tight_layout(); st.save_fig(fig, 'oscillation'); plt.show()
