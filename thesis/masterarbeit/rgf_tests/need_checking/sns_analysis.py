"""
SNS Junction — Comprehensive Analysis Script
=============================================
Verified and consolidated from all previous debugging sessions.

Physics:
  BdG Nambu basis: (c↑, c↓, c↓†, c↑†)
  SNS junction: SC lead | Normal region | SC lead
  Gauge convention (confirmed numerically):
    phase_matrix(θ) = diag(1, 1, e^{iθ}, e^{iθ})
    U(θ) @ H(Δ) @ U†(θ) = H(Δ·e^{-iθ})
    → To get Δ·e^{+iφ} on right lead: UR = phase_matrix(-φ)
    → To get Δ·e^{-iφ/2} on left lead (symmetric): UL = phase_matrix(+φ/2)
    → To get Δ·e^{+iφ/2} on right lead (symmetric): UR = phase_matrix(-φ/2)

Verified fixes included:
  - Correct gauge rotation signs (confirmed by numerical self-test)
  - get_rgf_sns/finite always return 5-tuple
  - Lower triangle G_blocks recursion: G[i,j] = G[i,j+1] @ V† @ GL[j]
  - Josephson current: I = -T * Re(Tr[V @ G_0N - V† @ G_N0]) per Matsubara freq
  - Matsubara surface GFs use eta=0 (iwn already off real axis)

Contents:
  [1]  Parameters
  [2]  Static objects and helpers
  [3]  Sanity checks (PH symmetry, gauge self-test, topological phase)
  [4]  1D energy sweep: finite RGF vs Sancho-López
  [5]  2D LDOS map (phase × energy), parallelised over energies
  [6]  Josephson current (Matsubara), parallelised over frequencies
  [7]  All plots
"""

#%% ── Imports ────────────────────────────────────────────────────────────────────
import sys, os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from joblib import Parallel, delayed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
import my_functions as myf

# ══════════════════════════════════════════════════════════════════════════════
#%% [1] PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# gauge
SYMMETRIC = True   # True  → ±φ/2 on left/right leads
                   # False → full φ on right lead only

# system geometry
SL, SM, SR = 100, 50, 100   # sites: left SC | normal | right SC

# physical
t      = 1.0
alpha  = 0.2
mu_sc  = 0.0025
mu_n   = 0.0001
Delta  = 0.1
B      = 0.3
eta    = 1e-4    # broadening for retarded GF (LDOS)
target_E = 0.0
# Matsubara current
T_mat   = 1e-4   # temperature
N_MATS  = 100    # number of Matsubara frequencies

# grids
N_E   = 301
N_PHI = 301
energies = np.linspace(-1.5, 1.5, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

# probe sites inside N region
probe_sites  = [1, SM // 2, SM - 2]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# parallelism — leave one core free
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ══════════════════════════════════════════════════════════════════════════════
#%% [2] STATIC OBJECTS AND HELPERS
# define helper functions and matrices ══════════════════════════════════════════════════════════════════════════════

V         = myf.t_matrix(t, alpha)
Vdag      = V.conj().T
onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)          # real Delta
onsite_n  = myf.onsite_matrix(t, mu_n,  B, 0.0)

H_mid_slices, _ = myf.build_middle_region(t, mu_n, alpha, B, SM)


def phase_matrix(theta: float) -> np.ndarray:
    """
    Nambu gauge rotation: U(θ) @ H(Δ) @ U†(θ) = H(Δ·e^{-iθ})
    Confirmed numerically — signal is in Re of trace.
    """
    return np.diag([1.0, 1.0,
                    np.exp(1j * theta),
                    np.exp(1j * theta)]).astype(np.complex128)


def get_surface_gfs(E, phi: float, symmetric: bool = True):
    """
    Surface GFs with SC phase applied as gauge rotation.

    Convention (matches build_sns_junction_sliced):
      symmetric=True:
        Δ_L = Δ·e^{-iφ/2}  →  UL = phase_matrix(+φ/2)
        Δ_R = Δ·e^{+iφ/2}  →  UR = phase_matrix(-φ/2)
      symmetric=False:
        Δ_L = Δ (real)      →  UL = identity
        Δ_R = Δ·e^{+iφ}    →  UR = phase_matrix(-φ)
    """
    if symmetric:
        UL = phase_matrix(+phi / 2.0)
        UR = phase_matrix(-phi / 2.0)
    else:
        UL = np.eye(4, dtype=np.complex128)
        UR = phase_matrix(-phi)

    g_L, _ = myf.get_surface_gf(E, onsite_sc, Vdag, eta=eta)
    g_R, _ = myf.get_surface_gf(E, onsite_sc, V,    eta=eta)

    return UL @ g_L @ UL.conj().T, UR @ g_R @ UR.conj().T


def get_surface_gfs_matsubara(iwn):
    """Surface GFs for Matsubara frequencies — eta=0, phase applied separately."""
    g_L, _ = myf.get_surface_gf(iwn, onsite_sc, Vdag, eta=0)
    g_R, _ = myf.get_surface_gf(iwn, onsite_sc, V,    eta=0)
    return g_L, g_R


def get_ldos(G_block: np.ndarray) -> float:
    return -np.imag(np.trace(G_block)) / np.pi


def matsubara_freqs(T: float, n_max: int) -> np.ndarray:
    return 1j * (2 * np.arange(n_max) + 1) * np.pi * T


# ══════════════════════════════════════════════════════════════════════════════
#%% [3] SANITY CHECKS
# sanity checks ══════════════════════════════════════════════════════════════════════════════

C_ph = np.fliplr(np.eye(4))   # PH conjugation matrix for this basis

def ph_check(name, M):
    ok = np.allclose(M, -C_ph @ M.conj() @ C_ph, atol=1e-12)
    print(f"  PH symmetry [{name}]: {'✓ PASS' if ok else '✗ FAIL'}")

print("\n══ Hamiltonian PH checks ══")
ph_check("onsite_sc", onsite_sc)
ph_check("onsite_N",  onsite_n)
ph_check("V hopping", V)

print("\n══ Gauge rotation self-test ══")
print("  U(-φ) @ H(Δ) @ U†(-φ) == H(Δ·e^{+iφ}) ?")
for phi_test in [0.0, np.pi / 2, np.pi]:
    U   = phase_matrix(-phi_test)
    err = np.max(np.abs(
        U @ onsite_sc @ U.conj().T -
        myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(1j * phi_test))
    ))
    print(f"  φ={phi_test/np.pi:.2f}π  err={err:.2e}  {'✓' if err < 1e-12 else '✗'}")

topo = np.sqrt(Delta**2 + mu_sc**2)
in_topo = B > topo
print(f"\n══ Topological phase ══")
print(f"  B={B:.3f}  √(Δ²+μ²)={topo:.4f}")
print(f"  In topological phase: {'YES ✓' if in_topo else 'NO  ✗'}")
print(f"\n══ Gauge: {'symmetric ±φ/2' if SYMMETRIC else 'asymmetric full φ on right'} ══")

# ══════════════════════════════════════════════════════════════════════════════
#%% [4] 1D ENERGY SWEEP AND PHASE SWEEP (φ = π fixed, E = 0 fixed, finite RGF vs Sancho-López) 
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ Running 1D energy sweep (φ=π) ══")

H_full_slices, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, np.pi, SL, SR, SM, symmetric=SYMMETRIC)

# shape: (n_probes, N_E, 2)  — method 0=finite, 1=Sancho-López
ldos_e = np.zeros((len(probe_sites), N_E, 2))

for e_idx, E in enumerate(energies):
    G_fin, _, _, _, _ = myf.get_rgf_finite_system(
        H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, np.pi, SYMMETRIC)
    G_inf, _, _, _, _ = myf.get_rgf_sns(
        H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        ldos_e[s_idx, e_idx, 0] = get_ldos(G_fin[SL + s])
        ldos_e[s_idx, e_idx, 1] = get_ldos(G_inf[s])

# PH symmetry check on results
e_sym = np.allclose(energies, -energies[::-1], atol=1e-10)
if e_sym:
    for s_idx, lbl in enumerate(probe_labels):
        for m, mname in enumerate(["Finite RGF", "Sancho-López"]):
            diff = np.max(np.abs(ldos_e[s_idx, :, m] - ldos_e[s_idx, ::-1, m]))
            print(f"  PH sym [{lbl} | {mname}]: max|Δ|={diff:.2e}  "
                  f"{'✓' if diff < 5e-3 else '✗'}")
            
# PHASE SWEEP  (E = 0)
print(f"\n══ Running 1D phase sweep (E={target_E:1f}) ══")



# shape: (n_probes, N_E, 2)  — method 0=finite, 1=Sancho-López
ldos_p = np.zeros((len(probe_sites), N_PHI, 2))

for p_idx, phi in enumerate(phases):
    H_full_slices, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi, SL, SR, SM, symmetric=SYMMETRIC)
    G_fin, _, _, _, _ = myf.get_rgf_finite_system(
        H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, np.pi, SYMMETRIC)
    G_inf, _, _, _, _ = myf.get_rgf_sns(
        H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        ldos_p[s_idx, p_idx, 0] = get_ldos(G_fin[SL + s])
        ldos_p[s_idx, p_idx, 1] = get_ldos(G_inf[s])

# PH symmetry check on results
p_sym = np.allclose(phases, -phases[::-1], atol=1e-10)
if p_sym:
    for s_idx, lbl in enumerate(probe_labels):
        for m, mname in enumerate(["Finite RGF", "Sancho-López"]):
            diff = np.max(np.abs(ldos_p[s_idx, :, m] - ldos_p[s_idx, ::-1, m]))
            print(f"  PH sym [{lbl} | {mname}]: max|Δ|={diff:.2e}  "
                  f"{'✓' if diff < 5e-3 else '✗'}")
            
#%% [5] 2D LDOS MAP  (phase × energy, parallelised over energies)
# ══════════════════════════════════════════════════════════════════════════════

def _ldos_row(e_idx, E, phases, probe_site):
    """One row of the 2D map — all phases at fixed energy."""
    row = np.zeros(len(phases), dtype=float)
    for p_idx, phi in enumerate(phases):
        g_L, g_R = get_surface_gfs(E, phi, SYMMETRIC)
        G_inf, _, _, _, _ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        row[p_idx] = get_ldos(G_inf[probe_site])
    return e_idx, row


def compute_ldos_2d(phases, energies, probe_site, n_jobs=N_JOBS):
    """2D LDOS array shape (N_PHI, N_E), parallelised over energies."""
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_ldos_row)(e_idx, E, phases, probe_site)
        for e_idx, E in enumerate(energies)
    )
    ldos_2d = np.zeros((len(phases), len(energies)), dtype=float)
    for e_idx, row in results:
        ldos_2d[:, e_idx] = row
    return ldos_2d


print(f"\n══ Running 2D LDOS map (cores: {N_JOBS}) ══")
ldos_2d = compute_ldos_2d(phases, energies, probe_site=SM // 2)

# Phase symmetry check: ldos(φ) == ldos(2π - φ)
sym_err = np.max(np.abs(ldos_2d - ldos_2d[::-1, :]))
print(f"  Phase symmetry error: {sym_err:.3e}  {'✓' if sym_err < 5e-3 else '✗'}")

# ══════════════════════════════════════════════════════════════════════════════
#%% [6] JOSEPHSON CURRENT  (Matsubara, parallelised over frequencies)
# ══════════════════════════════════════════════════════════════════════════════
# Formula (derived and verified):
#   I(φ) = Σ_n  -T * Re( Tr[ V @ G_0N(iωn) - V† @ G_N0(iωn) ] )
# where G_0N = corner block [site 0, site N-1] of the full GF
# Signal is in Re(trace), not Im — confirmed numerically.

def _current_row(iwn, phases):
    """Contribution of one Matsubara frequency to I(φ)."""
    g_L, g_R = get_surface_gfs_matsubara(iwn)
    N   = len(H_mid_slices)
    dof = 4
    row = np.zeros(len(phases), dtype=float)

    for p_idx, phi in enumerate(phases):
        # asymmetric gauge: left at φ=0, right gets full φ
        UR      = phase_matrix(-phi)
        g_R_phi = UR @ g_R @ UR.conj().T

        _, _, _, G_blocks, _ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R_phi, iwn, eta=0,
            return_full=True, return_dense=False)

        G_0N = G_blocks[0,   N-1]   # top-right corner block
        G_N0 = G_blocks[N-1, 0  ]   # bottom-left corner block

        tr = np.trace(V @ G_0N - Vdag @ G_N0)
        row[p_idx] = -T_mat * np.real(tr)

    return row


def compute_current(phases, T, n_max, n_jobs=N_JOBS):
    """Josephson current I(φ), parallelised over Matsubara frequencies."""
    iwn_list = matsubara_freqs(T, n_max)
    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_current_row)(iwn, phases)
        for iwn in iwn_list
    )
    return np.sum(rows, axis=0)


print(f"\n══ Running Josephson current (Matsubara, cores: {N_JOBS}) ══")
current = compute_current(phases, T_mat, N_MATS)

# Basic checks
print(f"  I(φ=0)  = {current[0]:.4e}   (should be ~0)")
print(f"  I(φ=2π) = {current[-1]:.4e}  (should be ~0, periodicity)")
print(f"  max|I|  = {np.max(np.abs(current)):.4e}")

# ══════════════════════════════════════════════════════════════════════════════
#%% [7] PLOTS
# ══════════════════════════════════════════════════════════════════════════════

C_FIN = "#d6604d"
C_INF = "#4dac26"

# ── Figure 1: 1D energy sweep ─────────────────────────────────────────────────
fig1, axes1 = plt.subplots(len(probe_sites), 1, figsize=(8, 7), sharex=True)
fig1.suptitle(f"LDOS vs Energy  (φ=π, B={B}, Δ={Delta}, α={alpha})",
              fontsize=13, fontweight="bold")

styles = [
    dict(color=C_FIN, lw=1.5, ls="dashed", label=f"Finite RGF ({SL}+{SM}+{SR})", alpha=0.85),
    dict(color=C_INF, lw=2,   ls="solid",  label="Sancho-López (∞ leads)",        alpha=0.85),
]

for s_idx, (s, lbl) in enumerate(zip(probe_sites, probe_labels)):
    ax = axes1[s_idx]
    for m in range(2):
        ax.plot(energies, ldos_e[s_idx, :, m], **styles[m])
    ax.axvline(-Delta, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(+Delta, color="gray", lw=0.8, ls="--", alpha=0.5, label="±Δ")
    ax.axvline(0,      color="k",    lw=0.5, ls=":",  alpha=0.3)
    ax.set_ylabel("LDOS", fontsize=10)
    ax.set_title(lbl, fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.set_ylim(bottom=0)

axes1[-1].set_xlabel("Energy (E/t)", fontsize=11)
fig1.tight_layout()

# ── Figure 2: 2D LDOS map ─────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 7))
pcm = ax2.pcolormesh(
    phases / np.pi, energies, ldos_2d.T,
    shading="auto", cmap="magma"
)
ax2.axhline(-Delta, color="white", lw=1,   ls="--", alpha=0.6, label="±Δ")
ax2.axhline(+Delta, color="white", lw=1,   ls="--", alpha=0.6)
ax2.axhline(0,      color="white", lw=0.5, ls=":",  alpha=0.4)
ax2.axvline(1,      color="gray",  lw=0.8, ls="--", alpha=0.5, label="φ=π")
ax2.set_xlabel(r"$\phi / \pi$", fontsize=12)
ax2.set_ylabel("Energy (E/t)", fontsize=12)
ax2.set_title(f"LDOS  (B={B}, Δ={Delta}, α={alpha})", fontsize=13)
ax2.legend(fontsize=9)
fig2.colorbar(pcm, ax=ax2, label="LDOS")
fig2.tight_layout()

# ── Figure 3: Josephson current ───────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(phases / np.pi, current, lw=2, color="#2166ac")
ax3.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
ax3.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.5, label="φ=π")
ax3.set_xlabel(r"$\phi / \pi$", fontsize=12)
ax3.set_ylabel("Current (arb.)", fontsize=12)
ax3.set_title(f"Josephson current (Matsubara, T={T_mat})", fontsize=12)
ax3.grid(True, alpha=0.2)
ax3.legend(fontsize=9)
fig3.tight_layout()

plt.show()

# %%
