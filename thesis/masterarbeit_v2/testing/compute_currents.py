#%%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent
sys.path.insert(0, str(module_root))
import modules as myf

#%% set up parameters
# PARAMETERS
# ═════════════════════════════════════════════════════════════
t      = 1.0
alpha  = 0.5
mu_sc  = 0.00025
mu_n   = 0.0
Delta  = 0.1
B      = 0.3

SL, SM, SR = 150, 80, 150

T      = 1e-4
N_MATS = 100

N_PHI  = 101
N_E    = 101                              # energy points for 2D sweep
phases   = np.linspace(0, 2*np.pi, N_PHI)
energies = np.linspace(-Delta*3, Delta*3, N_E)   # focus on sub-gap region

probe = 0                                 # site index into N region
#%% define stuff
# HELPERS
# ═════════════════════════════════════════════════════════════
V         = myf.t_matrix(t, alpha)
Vdag      = V.conj().T
onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)
H_mid_slices, _ = myf.build_middle_region(t, mu_n, alpha, B, SM)


def phase_matrix(phi: float) -> np.ndarray:
    """
    U(theta) @ H(Delta) @ U(theta)† = H(Delta * exp(-i*theta))
    Asymmetric gauge: left=0, right=phi → UR = phase_matrix(-phi)
    """
    return np.diag([1.0, 1.0,
                    np.exp(1j * phi),
                    np.exp(1j * phi)]).astype(np.complex128)

def matsubara_freqs(T: float, n_max: int) -> np.ndarray:
    n = np.arange(n_max)
    return 1j * (2*n + 1) * np.pi * T

def get_surface_gfs(iwn):
    """Bare surface GFs with real Delta — phase applied separately."""
    g_L, _ = myf.get_surface_gf(iwn, onsite_sc, Vdag, eta=0)
    g_R, _ = myf.get_surface_gf(iwn, onsite_sc, V,    eta=0)
    return g_L, g_R

def current_integrand(GR: np.ndarray, gl: np.ndarray) -> complex:
    """Tr[ V·GR·V†·gl  −  V†·gl·V·GR ]"""
    return np.trace(V @ GR @ Vdag @ gl - Vdag @ gl @ V @ GR)

def get_ldos(G_block: np.ndarray) -> float:
    return -np.imag(np.trace(G_block)) / np.pi

# ═════════════════════════════════════════════════════════════
# 1D JOSEPHSON CURRENT  (parallelised over Matsubara freqs)
# ═════════════════════════════════════════════════════════════
def _current_one_matsubara(iwn, phases):
    g_L, g_R = get_surface_gfs(iwn)

    N   = len(H_mid_slices)
    dof = 4

    row = np.zeros(len(phases), dtype=float)
    for p_idx, phi in enumerate(phases):
        UR      = phase_matrix(-phi)
        g_R_phi = UR @ g_R @ UR.conj().T

        G_full, _, _ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R_phi, iwn, eta=0, return_full=True
        )

        G_0N = G_full[:dof, (N-1)*dof:N*dof]
        G_N0 = G_full[(N-1)*dof:N*dof, :dof]

        tr = np.trace(V @ G_0N - V.conj().T @ G_N0)
        row[p_idx] = -T * np.real(tr)

    return row

def compute_current(phases: np.ndarray, T: float, n_max: int,
                    n_jobs: int = -1) -> np.ndarray:
    """
    Josephson current vs phase.
    Parallelised over Matsubara frequencies — each frequency is independent.
    """
    iwn_list = matsubara_freqs(T, n_max)

    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_current_one_matsubara)(iwn, phases)
        for iwn in iwn_list
    )
    return np.sum(rows, axis=0)
#%% 2D LDOS MAP
# LDOS energy vs phase
# ═════════════════════════════════════════════════════════════
def _ldos_one_energy(e_idx, E, phases, probe, eta=0):
    """LDOS(phi) at a fixed energy — one row of the 2D map."""
    row = np.zeros(len(phases), dtype=float)
    for p_idx, phi in enumerate(phases):
        g_L, _ = myf.get_surface_gf(E, onsite_sc, Vdag, eta=eta)
        g_R, _ = myf.get_surface_gf(E, onsite_sc, V,    eta=eta)

        UR      = phase_matrix(-phi)
        g_R_phi = UR @ g_R @ UR.conj().T

        G_inf, _, _ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R_phi, E, eta=0, return_full=False
        )
        row[p_idx] = get_ldos(G_inf[probe])

    return e_idx, row

def compute_ldos_2d(phases: np.ndarray, energies: np.ndarray,
                    probe: int, eta: float = 0,
                    n_jobs: int = -1) -> np.ndarray:
    """
    2D LDOS map: shape (N_PHI, N_E).
    Parallelised over energies — each energy slice is independent.
    """
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_ldos_one_energy)(e_idx, E, phases, probe, eta)
        for e_idx, E in enumerate(energies)
    )

    ldos_2d = np.zeros((len(phases), len(energies)), dtype=float)
    for e_idx, row in results:
        ldos_2d[:, e_idx] = row

    return ldos_2d

#%% RUN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1D current ──────────────────────────────────────────
    print("Computing Josephson current (parallel over Matsubara freqs)...")
    current = compute_current(phases, T, N_MATS, n_jobs=-1)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(phases / np.pi, current, lw=2)
    ax1.set_xlabel(r"$\phi / \pi$")
    ax1.set_ylabel("Current (arb.)")
    ax1.set_title("Josephson current (Matsubara RGF)")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # ── 2D LDOS map ─────────────────────────────────────────
    print("\nComputing 2D LDOS map (parallel over energies)...")
    ldos_2d = compute_ldos_2d(phases, energies, probe=probe, n_jobs=-1, eta=1e-3)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pcm = ax2.pcolormesh(
        phases / np.pi, energies, ldos_2d.T,
        shading="auto", cmap="magma"
    )
    ax2.axhline(0,      color="white", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(-Delta, color="cyan",  lw=0.8, ls="--", alpha=0.5)
    ax2.axhline(+Delta, color="cyan",  lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel(r"$\phi / \pi$")
    ax2.set_ylabel("Energy")
    ax2.set_title(f"LDOS vs phase and energy  (site {probe})")
    fig2.colorbar(pcm, ax=ax2, label="LDOS")
    fig2.tight_layout()

    plt.show()
# %% phi diagnostic 
iwn_list = matsubara_freqs(T, N_MATS)  
g_L, g_R = get_surface_gfs(iwn_list[0])
N   = len(H_mid_slices)
dof = 4

print("=== Current trace with fixed G[N-1,0] ===")
for phi_test in [0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
    UR      = phase_matrix(-phi_test)
    g_R_phi = UR @ g_R @ UR.conj().T

    G_full, _, _ = myf.get_rgf_sns(
        H_mid_slices, V, g_L, g_R_phi, iwn_list[0], eta=0, return_full=True
    )

    G_0N = G_full[:dof, (N-1)*dof:N*dof]
    G_N0 = G_full[(N-1)*dof:N*dof, :dof]

    tr = np.trace(V @ G_0N - V.conj().T @ G_N0)
    print(f"phi={phi_test/np.pi:.2f}pi  Re={np.real(tr):+.6e}  Im={np.imag(tr):+.6e}  -T*Re={-T*np.real(tr):+.6e}")

# %%
fig2, ax2 = plt.subplots(figsize=(8, 6))
pcm = ax2.pcolormesh(
    phases / np.pi, energies, ldos_2d.T,
    shading="auto", cmap="magma"
)
ax2.axhline(0,      color="white", lw=0.8, ls="--", alpha=0.6)
ax2.axhline(-Delta, color="cyan",  lw=0.8, ls="--", alpha=0.5)
ax2.axhline(+Delta, color="cyan",  lw=0.8, ls="--", alpha=0.5)
ax2.set_xlabel(r"$\phi / \pi$")
ax2.set_ylabel("Energy")
ax2.set_title(f"LDOS vs phase and energy  (site {probe})")
fig2.colorbar(pcm, ax=ax2, label="LDOS")
fig2.tight_layout()
# %%
