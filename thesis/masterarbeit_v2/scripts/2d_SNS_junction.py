#%%
"""After verifying the RGf method works, by comparing it to the exact diagonalization, it is now used to calculate the GF for a 2d SNS junction and plot the LDOS as a function of energy and phase difference, as always."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.linalg import inv, block_diag
from math import pi, cos, sin, sqrt

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent
sys.path.insert(0, str(module_root))

import modules as myf
#%% Create parameter class and set plotting style 

class SNSJunction:
    def __init__(self):
        self.t = 1.0
        self.mu_sc = 0.025 * self.t
        self.mu_m = 0.1 * self.t
        self.alpha = 0.4 * self.t
        self.delta = 0.1 * self.t
        self.eta = 1e-4 * self.delta
        self.h = 0.2 * self.t
        self.phi_steps = 61
        self.phi_array = np.linspace(0, 4*pi, self.phi_steps)
        self.dof = 4
        self.sites_l = 50
        self.sites_r = 50
        self.sites_m = 25
        self.sites_tot = (self.sites_l + self.sites_r + self.sites_m)
        self.N_tot = self.dof * self.sites_tot

    def get_textstr(self):
        return '\n'.join((
            r'$\mu_m/t = {:.3f}$'.format(self.mu_m),
            r'$\mu_s/t $= {:.3f}'.format(self.mu_sc),
            r'$t = {:.1f}$'.format(self.t),
            r'$\alpha/t = {:.1f}$'.format(self.alpha),
            r'$\Delta/t = {:.1f}$'.format(self.delta)
        ))



# Plotting styles
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)

#%% Define functions 
def get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def intraslice_hopping(hopping_matrix, N_slices):
    """Construct the intraslice hopping matrix for the 2D junction."""
    return block_diag(*[hopping_matrix for _ in range(N_slices)])

def phase_matrix(phi):
    """U(1) gauge rotation in Nambu basis (up,dn,dn†,up†)."""
    h = phi / 2
    return np.diag([np.exp(1j*h), np.exp(1j*h), np.exp(-1j*h), np.exp(-1j*h)]).astype(np.complex128)

def get_surface_gfs(E, t, mu_sc, alpha, B, Delta, phi, eta, symmetric, matsubara=None):
    """
    Returns (g_L, g_R) surface GFs in the correct gauge.
    symmetric=True  → ±φ/2 on left/right leads
    symmetric=False → full φ on right lead only
    """
    phase_L = -phi / 2 if symmetric else 0.0
    phase_R = +phi / 2 if symmetric else phi
    V = myf.t_matrix(t, alpha)
    onsite_L = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(1j * phase_L))
    onsite_R = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(1j * phase_R))

    g_L, _ = myf.get_surface_gf(E, onsite_L, V.conj().T, eta=eta, matsubara=matsubara)
    g_R, _ = myf.get_surface_gf(E, onsite_R, V, eta=eta, matsubara=matsubara)
    return g_L, g_R

def get_pairing(G_block):
    """Anomalous amplitude |G[0,3]| = |⟨c↑ c↑†⟩| proxy."""
    return np.abs(G_block[0, 3])

#%% 
# ── Parameters ────────────────────────────────────────────────────────────────
SYMMETRIC = True    # True  → ±φ/2 on left/right
                    # False → full φ on right only

SL, SM, SR = 200, 50, 200        # sites: left SC | normal | right SC
DOF        = 4
Delta      = 0.10
mu_sc      = 0.0025
mu_n       = 0.01
t          = 1.0
alpha      = 0.60
B          = 0.40              # Zeeman; topological when B > sqrt(Delta²+mu_sc²)
eta        = 1e-4
phi_fixed  = np.pi

N_E   = 121                    # energy points
N_PHI = 121                    # phase points
energies = np.linspace(-.8, .8, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

probe_sites = [1, SM // 2, SM - 2]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# ── Fixed objects ───────────────────────────────────────────────────────
V      = myf.t_matrix(t, alpha)
V_dag  = V.conj().T
onsite_sc   = myf.onsite_matrix(t, mu_sc, B, Delta)
H_mid_slices, _ = myf.build_middle_region(t, mu_n, alpha, B, SM)

H_full_slices, _ = myf.build_sns_junction_sliced(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, SL, SR, SM, symmetric=SYMMETRIC)
H_full_matrix = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, SL, SR, SM, 4, symmetric=SYMMETRIC)
dim_full = H_full_matrix.shape[0]
I_full = np.eye(dim_full, dtype=np.complex128)

# ── Particle-hole symmetry check on Hamiltonians ─────────────────────────────
C = np.fliplr(np.eye(DOF))   # antidiag identity — PH matrix

def ph_check(name, M):
    ok = np.allclose(M, -C @ M.conj() @ C, atol=1e-12)
    print(f"  PH symmetry [{name}]: {'✓ PASS' if ok else '✗ FAIL'}")

print("\n══ Hamiltonian PH checks ══")
ph_check("onsite_sc", onsite_sc)
ph_check("V hopping", V)
ph_check("onsite_N",  myf.onsite_matrix(t, mu_n, B, 0.0))

# ── Topological phase check ───────────────────────────────────────────────────
topo_threshold = np.sqrt(Delta**2 + mu_sc**2)
in_topo = B > topo_threshold
print(f"\n══ Topological phase ══")
print(f"  B = {B:.3f},  √(Δ²+μ²) = {topo_threshold:.4f}")
print(f"  In topological phase: {'YES ✓' if in_topo else 'NO  ✗'}")
print(f"\n══ Gauge convention: {'symmetric ±φ/2' if SYMMETRIC else 'asymmetric, full φ on right'} ══")
#%%
# ═════════════════════════════════════════════════════════════════════════════
# ENERGY SWEEP  (φ fixed)
# ═════════════════════════════════════════════════════════════════════════════
print("\n══ Running energy sweep ══")

# shape: (probe, energy, method)  methods: 0=inv 1=fin 2=inf
ldos_e  = np.zeros((len(probe_sites), N_E, 3))
pair_e  = np.zeros((len(probe_sites), N_E, 3))
Gblk_e  = np.zeros((len(probe_sites), N_E, 3, DOF, DOF), dtype=np.complex128)
ldos_map     = np.zeros((SM, N_E))
ldos_all_fin = np.zeros((SM, N_E))
ldos_all_inf = np.zeros((SM, N_E))

for e_idx, E in enumerate(energies):
    z = E + 1j * eta

    G_inv = inv(z * I_full - H_full_matrix)
    G_fin, _, _ = myf.get_rgf_finite_system(H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, t, mu_sc, alpha, B, Delta,phi_fixed, eta, SYMMETRIC)
    G_inf = myf.get_rgf_sns(H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_inv = G_inv[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF]
        blk_fin = G_fin[SL + s]
        blk_inf = G_inf[s]
        for m, blk in enumerate([blk_inv, blk_fin, blk_inf]):
            ldos_e[s_idx, e_idx, m] = get_ldos(blk)
            pair_e[s_idx, e_idx, m] = get_pairing(blk)
            Gblk_e[s_idx, e_idx, m] = blk

    for s in range(SM):
        ldos_map[s, e_idx]     = get_ldos(G_inv[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF])
        ldos_all_fin[s, e_idx] = get_ldos(G_fin[SL + s])
        ldos_all_inf[s, e_idx] = get_ldos(G_inf[s])
#%%
# ═════════════════════════════════════════════════════════════════════════════
# PHASE SWEEP  (E = 0)
# ═════════════════════════════════════════════════════════════════════════════
print("══ Running phase sweep  ══")

energy_fixed = 0 
ldos_p = np.zeros((len(probe_sites), N_PHI, 3))
pair_p = np.zeros((len(probe_sites), N_PHI, 3))


for p_idx, phi in enumerate(phases):
    H_full_p, _ = myf.build_sns_junction_sliced(t, mu_sc, mu_n, alpha, B, Delta, phi, SL, SR, SM, symmetric=SYMMETRIC)
    H_mat_p = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi, SL, SR, SM, DOF, symmetric=SYMMETRIC)
    
    I_p = np.eye(H_mat_p.shape[0], dtype=np.complex128)
    z   = energy_fixed + 1j * eta

    G_inv_p = inv(z * I_p - H_mat_p)
    G_fin_p, _, _ = myf.get_rgf_finite_system(
        H_full_p, V, energy_fixed, eta=eta, return_full=False)

    g_L_p, g_R_p = get_surface_gfs(E, t, mu_sc, alpha, B, Delta,phi_fixed, eta, SYMMETRIC)
    G_inf_p = myf.get_rgf_sns(H_mid_slices, V, g_L_p, g_R_p, energy_fixed, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_inv = G_inv_p[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF]
        blk_fin = G_fin_p[SL + s]
        blk_inf = G_inf_p[s]
        for m, blk in enumerate([blk_inv, blk_fin, blk_inf]):
            ldos_p[s_idx, p_idx, m] = get_ldos(blk)
            pair_p[s_idx, p_idx, m] = get_pairing(blk)
#%% 