"""
The Majorana splitting eps(E_Z) (pure Rashba, a = 20): the oscillation law, which gap branch sets the tail, and the
oscillations as a thermal fingerprint.

PARTS (one per former script; the cells of each part are tagged [part]):
  oscillation  (was study_oscillation.py)  STUDY: does the Majorana splitting follow  eps(E_Z) ~ A Delta_out e^{-L/xi_M} cos(k_osc L + phi0) ?
  branches     (was study_branches.py)  STUDY: which gap branch sets the Majorana tail -- the outer dip (k0 ~ k_F, oscillating tail) or the inner
  fingerprint  (was study_fingerprint.py)  STUDY: the splitting oscillations as a thermal fingerprint.

Run one part:   python study_oscillation.py oscillation      (several: list them; no argument: every part, in order)
In the interactive window the cells run as usual.  run_part(part) replaces the old `if __name__ == "__main__":`
guard: the parallel workers re-import this file and must not re-run the cells.
Names that collided between parts were renamed (see each PART header); cache names are unchanged.
"""
# %% --- IMPORTS AND THE PART SELECTOR ---
import sys
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from functools import partial
import tools as st

PARTS = ('oscillation', 'branches', 'fingerprint')
_SELECTED = {a for a in sys.argv[1:] if a in PARTS}


def run_part(part):
    """True for the cells of `part` when this file runs as a script: every part, or only those named on
    the command line.  False in the parallel workers (they import this file)."""
    return __name__ == "__main__" and (not _SELECTED or part in _SELECTED)


# ====================================================================================================================
# PART 'oscillation'  (was study_oscillation.py)
# ====================================================================================================================
# STUDY: does the Majorana splitting follow  eps(E_Z) ~ A Delta_out e^{-L/xi_M} cos(k_osc L + phi0) ?
# PREDICTION from the INFINITE cross-section, no fit: at E = 0 solve H(k) psi = 0 for complex
# k = k_osc + i/xi_M and take the slowest-decaying solution (the Majorana tail).
# MEASUREMENT on the FINITE junction: eps = lowest |eigenvalue| of the closed box (L along, the same
# cross-section with W_S = 900 nm SC on both sides, hard walls), sparse shift-invert, ~5 s per field.
# Cross-check against the transport geometry (semi-infinite SC, probes decoupled, end-LDOS peak,
# tools.splitting, ~8 min per field): E_Z = 0.8 gives 4.09 (box) vs 4.0 ueV (LDOS, 0.25 ueV grid).
# W 100 nm, L 2 um, a = 20 nm, phi = pi.  Tests:
#   1. between consecutive zeros of eps, k_osc L must advance by exactly pi (no free parameter);
#   2. the envelope eps_max / (Delta_out e^{-L/xi_M}) must be constant (Delta_out: gap at the outer dip, 1b;
#      Delta_top from tools.gaps misses the narrow dip on K_GRID and overestimates it by up to ~20%);
#   3. xi_M against the estimate hbar v_F / Delta_top  (FAILS: 0.41 -> 1.5), and
#   1b. against the REAL band at the outer gap dip (dip_point): hbar v_A / Delta_out with the Andreev-band
#       velocity hbar v_A = sqrt(Delta_out E''), and the dip continued to complex k by a polynomial.

# %% [oscillation] --- PARAMETERS ---

HBAR_VF = 51.43                                  # meV nm, lattice value at a = 20 (check_fermi.py)
EZ = np.round(np.arange(0.50, 1.3001, 0.005), 3)
CASE = st.case(100.0)


def bloch_blocks(p, n_S):
    """H0, H1 with H(k) = H0 + H1 e^{ik} + H1^dag e^{-ik} (nearest-neighbour hopping along the junction)."""
    Hp, Hm, Hh = (st.H_cross(p, k, n_S) for k in (0.0, np.pi, np.pi / 2))
    H0 = (Hp + Hm) / 2
    H1 = ((Hp - Hm) / 2 - 1j * (Hh - H0)) / 2
    return H0, H1


def slowest_mode(p, n_S):
    """lam = e^{ika} of the slowest-decaying E = 0 solution (|lam| < 1 closest to 1)."""
    H0, H1 = bloch_blocks(p, n_S)
    n = len(H0)
    I, Z = np.eye(n), np.zeros((n, n))
    # (H1 lam^2 + H0 lam + H1^dag) psi = 0 with lam = e^{ika}, linearised with chi = lam psi
    lam = sla.eig(np.block([[Z, I], [-H1.conj().T, -H0]]), np.block([[I, Z], [Z, H1]]), right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1 - 1e-9) & (np.abs(lam) > 1e-12)]
    return lam[np.argmax(np.abs(lam))]


def evanescent_point(q, E_Z, phi=np.pi):
    """[xi_M (nm), k_osc (1/nm), Delta_top (meV), Q, k of the gap minimum (1/nm)] at one field."""
    p = q.params(E_Z=float(E_Z), phi=phi)
    n_S = st.n_S_of(q)
    lam = slowest_mode(p, n_S)
    g0, gmin, kmin = st.gaps(p, n_S)
    return np.array([-q.a / np.log(np.abs(lam)), abs(np.angle(lam)) / q.a, gmin * q.t,
                     st.invariant(p, n_S), kmin / q.a])


def dip_point(q, E_Z, phi=np.pi, h=2e-3, kc=None):
    """
    xi_M from the REAL band: the gap dip at the outer Fermi point k0 (the local minimum next to k_osc).
    Hyperbola E ~ sqrt(Delta^2 + (hbar v_A dk)^2):  hbar v_A = sqrt(Delta E''), xi_hyp = hbar v_A / Delta.
    xi_poly: E(k)^2 fitted by a degree-6 polynomial over k0 +- 2/xi_hyp, continued to its complex zero.
    kc: k_osc a if already known (skips the complex-k solve).
    -> [k0 (1/nm), Delta_out (meV), hbar v_A (meV nm), xi_hyp (nm), xi_poly (nm)]
    """
    p = q.params(E_Z=float(E_Z), phi=phi)
    n_S = st.n_S_of(q)
    E1 = lambda k: np.min(np.abs(np.linalg.eigvalsh(st.H_cross(p, k, n_S))))
    kc = abs(np.angle(slowest_mode(p, n_S))) if kc is None else kc
    # near the bulk transition the dip is ~a/xi_M wide and a wide Brent search can miss it: also search kc +- 0.02
    k0 = min((minimize_scalar(E1, bounds=(kc - w, kc + w), method='bounded', options={'xatol': 1e-7}).x
              for w in (0.15, 0.02)), key=E1)
    D = E1(k0)
    Epp = (E1(k0 + h) - 2 * D + E1(k0 - h)) / h**2
    if Epp > 0 and h > 0.1 * np.sqrt(D / Epp):              # step not << the dip width 1/xi_hyp: redo finer
        h = 0.1 * np.sqrt(D / Epp)                          # (h = 2e-3 at mixed a 10, E_Z 1.15 is 0.8/xi_hyp: v_A 2.5% low)
        Epp = (E1(k0 + h) - 2 * D + E1(k0 - h)) / h**2
    if not Epp > 0:                                         # no minimum at k0: no dip to fit
        return np.array([k0 / q.a, D * q.t, np.nan, np.nan, np.nan])
    kap = np.sqrt(D / Epp)                                  # 1 / xi_hyp, units 1/a
    x = np.linspace(-2, 2, 41)
    E2 = np.array([E1(k0 + kap * xx) for xx in x]) ** 2 / D**2
    z = np.roots(np.polyfit(x, E2, 6)) * kap
    z = z[z.imag > 0]
    xi_poly = q.a / z[np.argmin(np.abs(z))].imag if len(z) else np.nan      # no complex zero: the fit is not a dip
    return np.array([k0 / q.a, D * q.t, np.sqrt(Epp * D) * q.t * q.a, q.a / kap, xi_poly])


def box_point(q, E_Z, phi=np.pi, k=6):
    """The k/2 lowest positive levels (ueV) of the closed box; the first is eps (each comes as +-E)."""
    p = q.params(E_Z=float(E_Z), phi=phi)
    H0, H1 = bloch_blocks(p, st.n_S_of(q))
    off = sp.diags(np.ones(q.n_along - 1), 1)
    H1 = sp.csr_matrix(H1)
    H = (sp.kron(sp.eye(q.n_along), sp.csr_matrix(H0)) + sp.kron(off, H1) + sp.kron(off.T, H1.conj().T)).tocsc()
    E = spla.eigsh(H, k=k, sigma=0.0, which='LM', return_eigenvectors=False)
    return np.sort(np.abs(E))[::2] * q.t * 1e3


def split_point(q, E_Z):
    """[eps (ueV)]: lowest end-LDOS peak in the transport geometry (slow; the cross-check above)."""
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


# %% [oscillation] --- 1. PREDICTION: complex k of the infinite cross-section ---
if run_part('oscillation'):
    V = st.run_scan(partial(evanescent_point, CASE), {'E_Z': EZ}, 'scharf_absO_evanescent_W100')
    xi_M, k_osc, D_top, Q, k_gap = V.T
    topo = Q < 0
    print(f"Q = -1 for E_Z {EZ[topo].min():.3f} .. {EZ[topo].max():.3f} meV")
    print(" E_Z    Delta_top   xi_M   hbar v/Delta_top   k_osc   k(gap min)")
    for j in range(0, len(EZ), 10):
        print(f" {EZ[j]:.3f}  {D_top[j] * 1e3:6.1f} ueV  {xi_M[j]:6.0f}   {HBAR_VF / D_top[j]:6.0f} nm"
              f"        {k_osc[j]:.4f}   {k_gap[j]:.4f} 1/nm   Q {Q[j]:+.0f}")


# %% [oscillation] --- 1b. xi_M FROM THE REAL BAND: the gap dip at the outer Fermi point ---
if run_part('oscillation'):
    k0, D_out, hv_A, xi_hyp, xi_poly = st.run_scan(partial(dip_point, CASE), {'E_Z': EZ},
                                                   'scharf_absO_dip_W100').T
    print("\n E_Z   Delta_out  hbar v_A  v_A/v_F   xi_hyp/xi_M   xi_poly/xi_M")
    for j in range(0, len(EZ), 10):
        print(f" {EZ[j]:.2f}  {D_out[j] * 1e3:5.1f} ueV  {hv_A[j]:5.1f}     {hv_A[j] / HBAR_VF:.2f}"
              f"       {xi_hyp[j] / xi_M[j]:.3f}         {xi_poly[j] / xi_M[j]:.3f}")
    print(f"growth 0.5 -> 1.3: xi_M x{xi_M[-1] / xi_M[0]:.1f} = v_A x{hv_A[-1] / hv_A[0]:.1f}"
          f" / Delta_out x{D_out[-1] / D_out[0]:.2f}  (hyperbola, times xi_hyp/xi_M)")


# %% [oscillation] --- 2. MEASUREMENT: eps of the finite junction ---
if run_part('oscillation'):
    lev = st.run_scan(partial(box_point, CASE), {'E_Z': EZ}, 'scharf_absO_box_W100_L2')
    eps = lev[:, 0]
    print(f"\nmeasured eps: {np.sum(np.isfinite(eps))}/{len(eps)} points, max {np.nanmax(eps[topo]):.2f} ueV")


# %% [oscillation] --- 3. COMPARISON ---
if run_part('oscillation'):
    L = CASE.L
    env = D_out * np.exp(-L / xi_M) * 1e3                       # ueV, prediction without prefactor
    z = zeros_of(EZ, np.where(topo, eps, np.nan))
    Phi = np.interp(z, EZ, k_osc) * L
    print(f"\nTEST 1: zeros of eps at {z}")
    print(f"        k_osc L advances by {np.round(np.diff(Phi) / np.pi, 2)} pi between them (model: 1)")
    print(f"        predicted spacing pi/(L dk/dE_Z): "
          f"{np.round(np.pi / (L * np.interp(z[:-1] + np.diff(z) / 2, EZ, np.gradient(k_osc, EZ))), 3)} meV,"
          f" measured {np.round(np.diff(z), 3)} meV")
    print("TEST 2: lobe   E_Z    eps_max [ueV]   Delta_out e^{-L/xi_M} [ueV]   ratio A   eps_max/Delta_out")
    for lo, hi in zip(z[:-1], z[1:]):
        w = (EZ > lo) & (EZ < hi) & np.isfinite(eps)
        j = np.flatnonzero(w)[np.argmax(eps[w])]
        print(f"        {lo:.3f}-{hi:.3f}  {EZ[j]:.3f}   {eps[j]:6.2f}          {env[j]:6.2f}"
              f"                   {eps[j] / env[j]:.2f}      {eps[j] / (D_out[j] * 1e3):.2f}")
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


# %% [oscillation] --- 4. ZEROS SEEN BY TRANSPORT: det r -> -1 with the weak probe (study_a cache, a = 20) ---
if run_part('oscillation'):
    try:
        d = np.load(st.DATA / 'scharf_absA_tr_weak_a20.npz', allow_pickle=True)
        Et, det = d['axis_E_Z'], d['values'][:, 3]
        i = np.flatnonzero((det[1:-1] < det[:-2]) & (det[1:-1] <= det[2:]) & (det[1:-1] < -0.9)) + 1
        print(f"\ndet r minima (< -0.9), weak probe: {np.round(Et[i], 3)}")
    except FileNotFoundError:
        print("\nno weak-probe transport cache at a = 20")


# %% [oscillation] --- FIGURE ---
if run_part('oscillation'):
    fig, axs = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    axs[0].plot(EZ, eps, 'o', ms=3, label=r'$\varepsilon$ measured (closed junction)')
    axs[0].plot(EZ, np.where(topo, A * env * np.abs(np.cos(k_osc * L + phi0)), np.nan), '-',
                label=rf'${A:.2f}\,\Delta_{{out}} e^{{-L/\xi_M}} |\cos(k_{{osc}}L + \phi_0)|$')
    axs[0].plot(EZ, np.where(topo, A * env, np.nan), '--', lw=1, label='envelope')
    for x in z:
        axs[0].axvline(x, color='k', lw=0.5, alpha=0.3)
    axs[0].set_ylabel(r'$\varepsilon$ [$\mu$eV]'); axs[0].legend(fontsize=8)
    axs[1].plot(EZ, np.where(topo, xi_M, np.nan), label=r'$\xi_M$ (complex $k$)')
    axs[1].plot(EZ, np.where(topo, HBAR_VF / D_top, np.nan), ':', label=r'$\hbar v_F/\Delta_{top}$ (2DEG $v_F$, global gap)')
    axs[1].plot(EZ, np.where(topo, xi_hyp, np.nan), '--', label=r'$\hbar v_A/\Delta_{out}$ (hyperbola at the dip)')
    axs[1].plot(EZ, np.where(topo, xi_poly, np.nan), 'k.', ms=2, label='dip continued to complex $k$')
    axs[1].set_ylabel('length [nm]'); axs[1].legend(fontsize=8)
    axs[2].plot(EZ, np.where(topo, hv_A, np.nan), label=r'$\hbar v_A = \sqrt{\Delta_{out} E^{\prime\prime}}$ [meV nm]')
    axs[2].axhline(HBAR_VF, color='k', lw=0.5, ls=':', label=r'$\hbar v_F$ (2DEG)')
    axs[2].plot(EZ, np.where(topo, D_out * 1e3, np.nan), '--', label=r'$\Delta_{out}$ [$\mu$eV]')
    axs[2].plot(EZ, np.where(topo, D_top * 1e3, np.nan), ':', label=r'$\Delta_{top}$ [$\mu$eV]')
    axs[2].legend(fontsize=8)
    axs[3].plot(EZ, np.where(topo, k_osc, np.nan), label=r'$k_{osc}$ (complex $k$)')
    axs[3].plot(EZ, np.where(topo, k_gap, np.nan), '--', label=r'$k$ of the global gap minimum')
    axs[3].plot(EZ, np.where(topo, k0, np.nan), 'k:', label=r'$k_0$ of the outer dip')
    axs[3].set_ylabel(r'$k$ [1/nm]'); axs[3].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[3].legend(fontsize=8)
    plt.tight_layout(); st.save_fig(fig, 'oscillation'); plt.show()


# %% [oscillation] --- 5. THE PREFACTOR A FOR SEVERAL L (box only): does it settle once L >> xi_M? (2026-11-03) ---
# The fit of cell 3 per length.  L = 1.5 / 2 / 2.5 um are the study_fingerprint box caches (same E_Z grid),
# 3 um is new.  If the drift of the per-lobe A (1.66 -> 2.01 at 2 um) comes from L/xi_M -> 1, a longer box
# must give flatter, L-independent values at the low-field lobes.
LS_A = (1500.0, 2000.0, 2500.0, 3000.0)
EZ_F = np.round(np.arange(0.50, 1.3001, 0.002), 3)


def fit_A(eps, env, k_osc, L):
    """eps = A env |cos(k_osc L + phi0)| by least squares in A on a phi0 grid: (A, phi0, rms ueV)."""
    best = None
    for phi0 in np.linspace(0, np.pi, 361):
        mod = env * np.abs(np.cos(k_osc * L + phi0))
        A = (eps @ mod) / (mod @ mod)
        r = np.sqrt(np.mean((eps - A * mod) ** 2))
        if best is None or r < best[2]:
            best = (A, phi0, r)
    return best


def a_vs_L(xi_M, k_osc, D_out, n_workers=6):
    """Fit A for every L in LS_A (box caches scharf_absF_box_L*), print, and save plots/oscillation_A_vs_L."""
    xi_f, k_f, D_f = (np.interp(EZ_F, EZ, v) for v in (xi_M, k_osc, D_out))
    print("\n  L [um]   fit A   phi0/pi   rms [ueV]      per lobe: E_Z of max, A = eps_max/env, L/xi_M there")
    fig, axs = plt.subplots(len(LS_A) + 1, 1, figsize=(10, 2.6 * (len(LS_A) + 1)), layout='constrained')
    out = {}
    for n, L_ in enumerate(LS_A):
        eps_ = st.run_scan(partial(box_point, st.case(100.0, L=L_)), {'E_Z': EZ_F},
                           f'scharf_absF_box_L{L_ / 1000:g}', n_workers=n_workers)[:, 0]
        env_ = D_f * np.exp(-L_ / xi_f) * 1e3
        A_, ph_, r_ = fit_A(eps_, env_, k_f, L_)
        z_ = zeros_of(EZ_F, eps_)
        lobes = []
        for lo, hi in zip(z_[:-1], z_[1:]):
            w = (EZ_F > lo) & (EZ_F < hi)
            j = np.flatnonzero(w)[np.argmax(eps_[w])]
            lobes.append((EZ_F[j], eps_[j] / env_[j], L_ / xi_f[j]))
        out[L_] = (A_, ph_, r_, lobes)
        print(f"  {L_ / 1000:4.1f}     {A_:.2f}    {ph_ / np.pi:.2f}     {r_:.2f} ({r_ / np.nanmax(eps_):.1%})    "
              + "  ".join(f"{e:.2f}:{a:.2f}({x:.1f})" for e, a, x in lobes))
        ax = axs[n]
        ax.plot(EZ_F, eps_, '.', ms=2, label=r'$\varepsilon$ (closed box)')
        ax.plot(EZ_F, A_ * env_ * np.abs(np.cos(k_f * L_ + ph_)), '-', lw=0.9,
                label=rf'${A_:.2f}\,\Delta_{{out}}e^{{-L/\xi_M}}|\cos(k_{{osc}}L + {ph_ / np.pi:.2f}\pi)|$')
        ax.plot(EZ_F, A_ * env_, '--', lw=0.7, color='gray', label='envelope')
        ax.set_ylabel(r'$\varepsilon$ [$\mu$eV]')
        ax.set_title(f'L = {L_ / 1000:g} um', fontsize=9, loc='left')
        ax.legend(fontsize=7, loc='upper left')
        lb = np.array(lobes)
        axs[-1].plot(lb[:, 2], lb[:, 1], 'o-', ms=4, label=f'L = {L_ / 1000:g} um')
    axs[-2].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$, W 100, a 20)')
    axs[-1].set_xscale('log')
    axs[-1].set_xlabel(r'$L/\xi_M$ at the lobe maximum')
    axs[-1].set_ylabel(r'per-lobe $A = \varepsilon_{max}/(\Delta_{out}e^{-L/\xi_M})$')
    axs[-1].legend(fontsize=8)
    fig.suptitle(r'prefactor $A$ of $\varepsilon = A\,\Delta_{out}e^{-L/\xi_M}|\cos(k_{osc}L+\phi_0)|$ for several L',
                 fontsize=10)
    st.save_fig(fig, 'oscillation_A_vs_L'); plt.close(fig)
    return out


if run_part('oscillation'):
    A_L = a_vs_L(xi_M, k_osc, D_out)


# ====================================================================================================================
# PART 'branches'  (was study_branches.py)   renamed: EZ -> EZ_B
# ====================================================================================================================
# STUDY: which gap branch sets the Majorana tail -- the outer dip (k0 ~ k_F, oscillating tail) or the inner
# branch near k = 0 (non-oscillating tail, composite-MF picture of Klinovaja & Loss PRB 86, 085408)?
# Expectation: the k ~ 0 branch has an effective velocity ~ alpha Delta / E_Z, so it wins at strong SOC.
# Complex k of the infinite cross-section (study_oscillation.slowest_mode, all modes kept): W 100, a 20,
# phi pi, alpha = 16 / 32 / 48 / 64 meV nm at fixed mu measured from the band bottom (the +E_so convention
# of hamiltonians.py, so k_F of the outer band is k_mu + k_so and the inner one k_mu - k_so).
# Per field: the three slowest distinct decaying modes (xi, Re k), Q, and the gap at k = 0.

# %% [branches] --- PARAMETERS ---

ALPHAS = (16.0, 32.0, 48.0, 64.0)
EZ_B = np.round(np.arange(0.30, 1.8001, 0.01), 3)
N_MODES = 3


def modes_point(q, E_Z, phi=np.pi):
    """[xi_1, k_1, xi_2, k_2, xi_3, k_3, Q, E1(k=0) meV]: the slowest distinct E = 0 modes (nm, 1/nm)."""
    p = q.params(E_Z=float(E_Z), phi=phi); n_S = st.n_S_of(q)
    H0, H1 = bloch_blocks(p, n_S); n = len(H0); I, Z = np.eye(n), np.zeros((n, n))
    lam = sla.eig(np.block([[Z, I], [-H1.conj().T, -H0]]), np.block([[I, Z], [Z, H1]]), right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1 - 1e-9) & (np.abs(lam) > 1e-12)]
    xi, k = -q.a / np.log(np.abs(lam)), np.abs(np.angle(lam)) / q.a
    out = []
    for j in np.argsort(-xi):                                   # +-k partners: same xi and |Re k|
        if any(abs(xi[j] - a) < 1e-4 * a and abs(k[j] - b) < 1e-7 for a, b in out):
            continue
        out.append((xi[j], k[j]))
        if len(out) == N_MODES:
            break
    E0 = np.min(np.abs(np.linalg.eigvalsh(st.H_cross(p, 0.0, n_S)))) * q.t
    return np.r_[np.ravel(out), st.invariant(p, n_S), E0]


# %% [branches] --- 1. SCANS ---
if run_part('branches'):
    R = {}
    for al in ALPHAS:
        q = st.case(100.0, alpha=al)
        R[al] = st.run_scan(partial(modes_point, q), {'E_Z': EZ_B}, f'scharf_absB_modes_a{al:g}')


# %% [branches] --- 2. WHICH BRANCH WINS ---
if run_part('branches'):
    for al in ALPHAS:
        V = R[al]; topo = V[:, 6] < 0
        k_so, k_mu = al / 2006.0, 51.43 / 2006.0
        print(f"\nalpha {al:g} meV nm: E_so = {al**2 / 4012:.3f} meV, Fermi points k_mu -+ k_so = "
              f"{abs(k_mu - k_so):.4f} / {k_mu + k_so:.4f} 1/nm;  Q = -1 for E_Z "
              f"{EZ_B[topo].min() if topo.any() else np.nan:.2f} .. {EZ_B[topo].max() if topo.any() else np.nan:.2f}")
        print("   E_Z    slowest: xi [nm]  Re k     2nd: xi   Re k     3rd: xi   Re k     gap(k=0) [ueV]  Q")
        for j in range(0, len(EZ_B), 10):
            v = V[j]
            print(f"   {EZ_B[j]:.2f}   {v[0]:8.0f}  {v[1]:.4f}   {v[2]:8.0f}  {v[3]:.4f}   {v[4]:8.0f}  {v[5]:.4f}"
                  f"     {v[7] * 1e3:7.1f}        {v[6]:+.0f}")


# %% [branches] --- 3. alpha 48: does kappa lose its zeros where the k ~ 0 branch takes over (1.41-1.46)? ---
# Box eps + NEGF at E = 0 (as study_fingerprint), L 2 um, tc_barr 0.2, E_Z 0.90-1.50 step 0.002.
EZ48 = np.round(np.arange(0.90, 1.5001, 0.002), 3)
if run_part('branches'):
    q48 = st.case(100.0, alpha=48.0)
    eps48 = st.run_scan(partial(box_point, q48), {'E_Z': EZ48}, 'scharf_absB_box_a48_L2')[:, 0]
    tr48 = st.transport_line(q48, EZ48, cache='scharf_absB_tr_a48_L2')
    V = R[48.0]
    slow_k = np.interp(EZ48, EZ_B, V[:, 1]); slow_xi = np.interp(EZ48, EZ_B, V[:, 0])
    topo48 = np.interp(EZ48, EZ_B, V[:, 6]) < 0
    z = zeros_of(EZ48, np.where(topo48, eps48, np.nan), half=0.03)
    print(f"\nalpha 48, L 2 um: zeros of the box eps at {z}")
    print("   E_Z    eps [ueV]  kappa    G       det r    slowest: xi [nm]  Re k")
    for j in range(0, len(EZ48), 10):
        print(f"   {EZ48[j]:.2f}   {eps48[j]:7.2f}   {tr48['kappa'][j]:.3f}  {tr48['G'][j]:+.3f}  {tr48['det'][j]:+.3f}"
              f"      {slow_xi[j]:7.0f}     {slow_k[j]:.4f}   {'Q -1' if topo48[j] else 'Q +1'}")
    d = tr48['det']
    i = np.flatnonzero((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < 0)) + 1
    print(f"det r local minima (< 0): {np.round(EZ48[i], 3)}  values {np.round(d[i], 3)}")
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].semilogy(EZ48, eps48, 'k', lw=1.2); axs[0].set_ylabel(r'$\varepsilon$ box [$\mu$eV]')
    axs[1].plot(EZ48, tr48['kappa'], label=r'$\kappa/\kappa_0$'); axs[1].plot(EZ48, tr48['G'], label=r'$G/G_0$')
    axs[1].plot(EZ48, tr48['det'], ':', label='det r'); axs[1].axhline(0.5, color='k', lw=0.6)
    axs[1].legend(fontsize=8); axs[1].set_ylabel('NEGF, T = 0')
    axs[2].semilogy(EZ48, slow_xi, 'k', lw=1.2); axs[2].set_ylabel(r'$\xi$ slowest mode [nm]')
    ax2 = axs[2].twinx(); ax2.plot(EZ48, slow_k, 'C1.', ms=2); ax2.set_ylabel('Re k slowest [1/nm]', color='C1')
    for ax in axs:
        ax.axvspan(1.405, 1.465, color='0.9', zorder=0)
        for x in z:
            ax.axvline(x, color='k', lw=0.5, ls=':')
    axs[2].set_xlabel(r'$E_Z$ [meV]   (grey: k ~ 0 branch slowest; dotted: box zeros)')
    axs[0].set_title(r'$\alpha$ = 48 meV nm, W 100, L 2 $\mu$m, $\phi = \pi$, tc_barr 0.2', fontsize=10, loc='left')
    plt.tight_layout(); st.save_fig(fig, 'branches_a48'); plt.close(fig)


# %% [branches] --- FIGURE ---
if run_part('branches'):
    fig, axs = plt.subplots(2, len(ALPHAS), figsize=(16, 7), sharex=True)
    for c, al in enumerate(ALPHAS):
        V = R[al]; topo = V[:, 6] < 0
        for m, (mk, s) in enumerate((('o', 14), ('s', 8), ('^', 6))):
            xi, k = np.where(topo, V[:, 2 * m], np.nan), V[:, 2 * m + 1]
            sc = axs[0, c].scatter(EZ_B, xi, c=k, s=s, marker=mk, cmap='viridis', vmin=0, vmax=0.06)
            axs[1, c].plot(EZ_B, np.where(topo, k, np.nan), mk, ms=3 if m else 4, label=f'mode {m + 1}')
        axs[0, c].set_yscale('log'); axs[0, c].axhline(2000, color='k', lw=0.6, ls='--')
        axs[0, c].set_title(rf'$\alpha$ = {al:g} meV nm  ($E_{{so}}$ = {al**2 / 4012:.2f} meV)', fontsize=10)
        axs[1, c].set_xlabel(r'$E_Z$ [meV]')
    axs[0, 0].set_ylabel(r'$\xi$ of the 3 slowest modes [nm] (colour: Re k)')
    axs[1, 0].set_ylabel(r'Re $k$ [1/nm]'); axs[1, 0].legend(fontsize=8)
    fig.colorbar(sc, ax=axs[0, :], label='Re k [1/nm]')
    st.save_fig(fig, 'branches'); plt.show()


# ====================================================================================================================
# PART 'fingerprint'  (was study_fingerprint.py)   renamed: EZ -> EZ_F, EZ_O -> EZ
# ====================================================================================================================
# STUDY: the splitting oscillations as a thermal fingerprint.
# Two-Majorana model: kappa/kappa0 = 2x^2/(1+x^2)^2, det r = (x^2-1)/(x^2+1), x = eps/Gamma.  So kappa = 0 at
# every zero of eps (det r = -1) and kappa = 1/2 wherever eps = Gamma, whatever Gamma is.
# Three lengths L = 1.5, 2, 2.5 um (W 100 nm, a = 20, phi = pi, tc_barr = 0.2), on one E_Z grid:
#   eps   : closed box (study_oscillation.box_point), ~5 s per field
#   kappa : full NEGF at E = 0, T -> 0 (tools.transport_point), ~0.6 s per field
# Tests:
#   1. collapse: the zeros sit at k_osc(E_Z) L / pi = n + 1/2 - phi0/pi for EVERY L (k_osc from the
#      infinite cross-section, scharf_absO_evanescent_W100; no free parameter except phi0);
#   2. the kappa zeros (det r minima) of the NEGF coincide with the zeros of the box eps;
#   3. the two-Majorana model: Gamma = eps / x with x from det r; kappa_model(eps_box, Gamma) vs kappa_NEGF;
#   4. the kappa maxima between zeros are 1/2 once eps_max > Gamma;
#   5. depth of the kappa zeros (fine NEGF scan around each box zero): once xi_M ~ L each Majorana's tail
#      reaches the opposite probe and the zeros fill in by electron cotunnelling (G ~ kappa); against L/xi_M.

# %% [fingerprint] --- PARAMETERS ---

LS = (1500.0, 2000.0, 2500.0)
EZ_F = np.round(np.arange(0.50, 1.3001, 0.002), 3)
CASES = {L: st.case(100.0, L=L) for L in LS}

ev = np.load(st.DATA / 'scharf_absO_evanescent_W100.npz')['values']   # xi_M, k_osc, Delta_top, Q, k_gap
K_OSC = np.interp(EZ_F, EZ, ev[:, 1])                                  # 1/nm, independent of L
XI_M = np.interp(EZ_F, EZ, ev[:, 0])                                   # nm
TOPO = np.interp(EZ_F, EZ, ev[:, 3]) < 0


def kappa_model(x):
    return 2 * x**2 / (1 + x**2)**2


def x_of_det(d):
    """x = eps/Gamma from det r = (x^2-1)/(x^2+1)."""
    return np.sqrt((1 + d) / (1 - d))


def smooth_gamma(eps, det):
    """Gamma = eps/x (x from det r) away from the zeros, smoothed by a quadratic in E_Z (as in cell 3)."""
    G = eps / x_of_det(np.clip(det, -0.999999, 0.999999))
    good = TOPO & (np.abs(det) < 0.95) & np.isfinite(G)
    return np.polyval(np.polyfit(EZ_F[good], G[good], 2), EZ_F)


def kappa_E(q, E_Z, E):
    """(kappa/kappa0, G/G0) at energy E (meV), phi = pi.  E != 0: all four channels (hh(E) = ee(-E))."""
    c = dict(zip(st.CHANNELS, st.channel_point(q, E_Z, E)))
    return ((c['ee'] + c['he_cross'] + c['hh'] + c['eh_cross']) / 2,
            (c['ee'] - c['he_cross'] + c['hh'] - c['eh_cross']) / 2)


def resonance_point(q, i, EZ_sel, E_star, Gamma):
    """
    Largest kappa(E) near the two-Majorana resonance E* = sqrt(eps^2 - Gamma^2) (meV) at EZ_sel[i]:
    13 energies over E* +- 3 Gamma, then a bounded 1D maximisation.  -> [E_peak (ueV), kappa, G there, kappa(E=0)]
    """
    from scipy.optimize import minimize_scalar
    EZ_i, Es, Gm = float(EZ_sel[i]), float(E_star[i]), float(Gamma[i])
    grid = np.clip(Es + Gm * np.linspace(-3, 3, 13), 0.0, None)
    kg = np.array([kappa_E(q, EZ_i, e)[0] for e in grid])
    j = int(np.argmax(kg))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    r = minimize_scalar(lambda e: -kappa_E(q, EZ_i, e)[0], bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-3 * Gm})
    E_pk = r.x if -r.fun > kg[j] else grid[j]
    k_pk, G_pk = kappa_E(q, EZ_i, E_pk)
    return np.array([E_pk * 1e3, k_pk, G_pk, kappa_E(q, EZ_i, 0.0)[0]])


def finite_energy_line(q, eps, det, every=5, n_workers=6, cache='scharf_absF_res_L2'):
    """resonance_point on every `every`-th field with eps > Gamma; returns (E_Z, eps, Gamma, values)."""
    Gam = smooth_gamma(eps, det)
    sel = np.flatnonzero(TOPO & np.isfinite(eps) & (eps > Gam) & (np.arange(len(EZ_F)) % every == 0))
    E_star = np.sqrt(eps[sel] ** 2 - Gam[sel] ** 2) * 1e-3                      # meV
    V = st.run_scan(partial(resonance_point, q, EZ_sel=EZ_F[sel], E_star=E_star, Gamma=Gam[sel] * 1e-3),
                    {'i': np.arange(len(sel))}, cache, n_workers=n_workers)
    return EZ_F[sel], eps[sel], Gam[sel], V


def plot_finite_energy(E_r, eps_r, Gam_r, V_r, name='fingerprint_finite_energy'):
    """kappa at E = 0 and at the resonance E* against E_Z, plus eps, Gamma, E*; saved to plots/<name>."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, layout='constrained')
    a1.plot(E_r, V_r[:, 3], 'o', ms=3, label=r'$\kappa(E = 0)$')
    a1.plot(E_r, V_r[:, 1], 's', ms=3, label=r'$\kappa(E_{peak})$, largest near $E^* = \sqrt{\varepsilon^2-\Gamma^2}$')
    a1.plot(E_r, np.abs(V_r[:, 2]), 'x', ms=3, label=r'$|G(E_{peak})|$')
    a1.axhline(0.5, color='k', lw=0.6, ls='--')
    a1.set_ylabel(r'$\kappa/\kappa_0$, $|G|/G_0$'); a1.legend(fontsize=8)
    a1.set_title(r'two-Majorana model: $\max_E \kappa(E) = 1/2$ at $E^*$ for every $\varepsilon \geq \Gamma$', fontsize=9)
    a2.semilogy(E_r, eps_r, '.', label=r'$\varepsilon$ (closed box)')
    a2.semilogy(E_r, Gam_r, '-', label=r'$\Gamma$ (from det r, smoothed)')
    a2.semilogy(E_r, V_r[:, 0], 'x', ms=3, label=r'$E_{peak}$ (NEGF)')
    a2.set_ylabel(r'energy [$\mu$eV]'); a2.set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$, W 100, L 2 $\mu$m, a 20, tc_barr 0.2)')
    a2.legend(fontsize=8)
    st.save_fig(fig, name); plt.close(fig)


# %% [fingerprint] --- 1. eps (box) and kappa, G, det r (NEGF) for each L ---
if run_part('fingerprint'):
    R = {}
    for L, q in CASES.items():
        eps = st.run_scan(partial(box_point, q), {'E_Z': EZ_F}, f'scharf_absF_box_L{L / 1000:g}')[:, 0]
        tr = st.transport_line(q, EZ_F, cache=f'scharf_absF_tr_L{L / 1000:g}')
        R[L] = dict(eps=np.where(TOPO, eps, np.nan), **tr)


# %% [fingerprint] --- 2. TESTS 1 + 2: zeros of eps and of kappa, and the collapse on k_osc L / pi ---
if run_part('fingerprint'):
    print("\n  L      zeros of eps (box)                     det r minima (NEGF)                   "
          "k_osc L/pi at the eps zeros (mod 1)")
    for L in LS:
        r = R[L]
        z = zeros_of(EZ_F, r['eps'], half=0.03)
        d = r['det']
        i = np.flatnonzero((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < -0.9)) + 1
        r['z'], r['z_tr'] = z, EZ_F[i]
        ph = np.interp(z, EZ_F, K_OSC) * L / np.pi
        print(f"  {L / 1000:g}  {np.round(z, 3)}  {np.round(EZ_F[i], 3)}  {np.round(ph % 1, 2)}")
        print(f"        steps of k_osc L/pi between zeros: {np.round(np.diff(ph), 2)}")


# %% [fingerprint] --- 3. TEST 3: Gamma from det r, and the two-Majorana model against the NEGF ---
if run_part('fingerprint'):
    print("\n  L     Gamma = eps/x [ueV] at E_Z 0.6 / 0.8 / 1.0 / 1.2      rms(kappa_model - kappa_NEGF)")
    for L in LS:
        r = R[L]
        x = x_of_det(np.clip(r['det'], -0.999999, 0.999999))
        G = r['eps'] / x
        good = TOPO & (np.abs(r['det']) < 0.95)                # x well defined, away from the zeros
        r['Gamma'] = np.where(good, G, np.nan)
        Gs = [np.nanmedian(r['Gamma'][np.abs(EZ_F - e) < 0.05]) for e in (0.6, 0.8, 1.0, 1.2)]
        r['Gamma_s'] = np.polyval(np.polyfit(EZ_F[good], G[good], 2), EZ_F)   # smooth: cannot follow the wiggles
        r['k_mod'] = kappa_model(r['eps'] / r['Gamma_s'])
        rms = np.sqrt(np.nanmean((r['k_mod'] - r['kappa'])[TOPO] ** 2))
        print(f"  {L / 1000:g}   {'  '.join(f'{g:6.2f}' for g in Gs)}                        {rms:.3f}")


# %% [fingerprint] --- 4. TEST 4: heights of the kappa maxima, lobe by lobe ---
if run_part('fingerprint'):
    print("\n  L     lobe          kappa maxima        eps_max/Gamma    max |G|/kappa at them")
    for L in LS:
        r = R[L]
        for lo, hi in zip(r['z'][:-1], r['z'][1:]):
            w = np.flatnonzero((EZ_F > lo) & (EZ_F < hi))
            k = r['kappa'][w]
            pk = w[1:-1][(k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:])]
            j = w[np.nanargmax(r['eps'][w])]
            print(f"  {L / 1000:g}   {lo:.3f}-{hi:.3f}   {np.round(r['kappa'][pk], 3)}"
                  f"   {r['eps'][j] / r['Gamma_s'][j]:5.2f}          "
                  f"{np.nanmax(np.abs(r['G'][pk]) / r['kappa'][pk]) if len(pk) else np.nan:.3f}")


# %% [fingerprint] --- 5. TEST 5: how deep the kappa zeros are (0.1 ueV field steps around each box zero) ---
if run_part('fingerprint'):
    print("\n  L     zero    L/xi_M   kappa_min   G there   det r there")
    for L, q in CASES.items():
        r = R[L]; rows = []
        for z in r['z']:
            Ef = np.round(np.arange(z - 0.006, z + 0.006 + 1e-9, 0.0001), 5)
            V = st.run_scan(partial(st.transport_point, q), {'E_Z': Ef}, f'scharf_absF_zero_L{L / 1000:g}_{z:.3f}')
            i = np.argmin(V[:, 0])
            rows.append([Ef[i], L / np.interp(Ef[i], EZ_F, XI_M), V[i, 0], V[i, 1], V[i, 3]])
            print(f"  {L / 1000:g}   {Ef[i]:.4f}   {rows[-1][1]:5.2f}    {V[i, 0]:.4f}    {V[i, 1]:+.4f}   {V[i, 3]:+.4f}")
        rows = np.array(rows)
        r['zf'] = rows[rows[:, 4] < 0]                         # det r > 0: the box zero has no kappa zero


# %% [fingerprint] --- FIGURE ---
if run_part('fingerprint'):
    COL = {1500.0: '#2a78d6', 2000.0: '#eb6834', 2500.0: '#1baf7a'}
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(4, 2)
    a_k, a_m = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    a_c, a_e, a_z, a_g = (fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                          fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]))
    for L in LS:
        r, c, lab = R[L], COL[L], f'L = {L / 1000:g} µm'
        a_k.plot(EZ_F, r['kappa'], color=c, lw=1.4, label=lab)
        for x in r['z']:
            a_k.axvline(x, color=c, lw=0.6, ls=':')
        a_m.plot(EZ_F, r['kappa'], color=c, lw=1.4, label=f'NEGF, {lab}')
        a_m.plot(EZ_F, r['k_mod'], color='k', lw=0.6, ls='--')
        zf = r['zf']
        a_c.plot(zf[:, 0], (np.interp(zf[:, 0], EZ_F, K_OSC) * L / np.pi) % 1, 'o', color=c, label=lab)
        a_e.semilogy(EZ_F, r['eps'], color=c, lw=1.4, label=lab)
        a_e.semilogy(EZ_F, r['Gamma'], color=c, lw=0.8, ls='--')
        a_z.semilogy(zf[:, 1], zf[:, 2], 'o', color=c, label=lab)
        a_g.plot(zf[:, 1], zf[:, 3] / zf[:, 2], 'o', color=c, label=lab)
    a_k.axhline(0.5, color='k', lw=0.6); a_k.set_xlabel(r'$E_Z$ [meV]')
    a_k.set_ylabel(r'$\kappa/\kappa_0$ (NEGF, T = 0)'); a_k.legend(fontsize=8)
    a_k.set_title('dotted: zeros of the box splitting', fontsize=9, loc='left')
    a_m.set_xlabel(r'$E_Z$ [meV]'); a_m.set_ylabel(r'$\kappa/\kappa_0$'); a_m.legend(fontsize=8)
    a_m.set_title(r'black dashed: two-Majorana model $2x^2/(1+x^2)^2$, $x = \varepsilon_{box}/\Gamma$, smooth $\Gamma(E_Z)$',
                  fontsize=9, loc='left')
    a_c.set_xlabel(r'$E_Z$ of the $\kappa$ zero [meV]'); a_c.set_ylabel(r'$k_{osc} L/\pi$ mod 1')
    a_c.set_title('collapse: one phase curve for every L', fontsize=9, loc='left'); a_c.legend(fontsize=8)
    a_e.set_xlabel(r'$E_Z$ [meV]'); a_e.set_ylabel(r'$\varepsilon$ (solid), $\Gamma$ (dashed) [$\mu$eV]')
    a_e.legend(fontsize=8)
    a_z.set_xlabel(r'$L/\xi_M$ at the zero'); a_z.set_ylabel(r'$\kappa_{min}/\kappa_0$')
    a_z.set_title('depth of the zeros', fontsize=9, loc='left'); a_z.legend(fontsize=8)
    a_g.set_xlabel(r'$L/\xi_M$ at the zero'); a_g.set_ylabel(r'$G/\kappa$ at the minimum (1: pure cotunnelling)')
    a_g.legend(fontsize=8)
    plt.tight_layout(); st.save_fig(fig, 'fingerprint'); plt.show()


# %% [fingerprint] --- 6. FINITE ENERGY: does kappa(E) reach 1/2 at E* = sqrt(eps^2 - Gamma^2) wherever eps > Gamma? ---
# Two-Majorana model (2026-11-03): kappa(E)/kappa0 = 2 G^2 e^2 / ((E^2 - e^2 - G^2)^2 + 4 E^2 G^2), maximal = 1/2
# exactly at E* for every eps >= Gamma.  A linear-response kappa(T) cannot pick out E* (window ~ kT >> Gamma).
if run_part('fingerprint'):
    r = R[2000.0]
    E_r, eps_r, Gam_r, V_r = finite_energy_line(CASES[2000.0], r['eps'], r['det'])
    plot_finite_energy(E_r, eps_r, Gam_r, V_r)
    print("\n  E_Z    eps [ueV]  Gamma   E* [ueV]  E_peak   kappa(E_peak)   G(E_peak)   kappa(E=0)")
    for j in range(len(E_r)):
        print(f"  {E_r[j]:.3f}  {eps_r[j]:7.2f}   {Gam_r[j]:5.2f}  {np.sqrt(eps_r[j]**2 - Gam_r[j]**2):7.2f}"
              f"  {V_r[j, 0]:7.2f}     {V_r[j, 1]:.3f}         {V_r[j, 2]:+.3f}      {V_r[j, 3]:.3f}")
