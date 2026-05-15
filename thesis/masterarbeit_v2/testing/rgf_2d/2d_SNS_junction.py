"""
After verifying the RGf method works, by comparing it to the exact diagonalization, it is now used to calculate the GF for a 2d SNS junction and plot the LDOS as a function of energy and phase difference, as always.
"""
#%% ── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Add parent directory to path for relative imports
current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent.parent
sys.path.insert(0, str(module_root))

import modules 

# Plotting styles
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)


#%% helper functions
'''
Convention: Slice in y-direction (h_slcies has the V_y hopping matrix) and the columns are in x-direction.
'''
def build_sns_slice_2d(N_y, t, mu, h, alpha, delta):
    '''
    Args:   N_y: int, number of sites in y-direction (width of the junction)
            t: float, hopping amplitude
            mu: float, chemical potential
            h: float, Zeeman energy
            alpha: float, spin-orbit coupling strength
            delta: complex, SC pairing potential (can include phase)
    
    Within-slice Hamiltonian: y-direction chain.
    y-hops use t_matrix_y (imaginary Rashba: σ_y)
    '''
    dof = 4
    # Onsite Hamiltonian for each site (same for all sites in the slice)
    h_0 = modules.onsite_matrix(t, mu, h, delta)
    V_y = modules.t_matrix_y(t, alpha)  # y-direction hops: imaginary Rashba
    V_y_dag = V_y.conj().T
    I_y = np.eye(N_y, dtype=np.complex128)
    off_y = np.eye(N_y, k=1, dtype=np.complex128)
    H = np.kron(I_y, h_0) + np.kron(off_y, V_y) + np.kron(off_y.T, V_y_dag)
    return H

def build_sns_junction_sliced_2d(N_y, t, mu_sc, mu_m, h, alpha, delta, phi, SL, SR, SM, symmetric=False):
    '''
    Args:   N_y: int, number of sites in y-direction (width of the junction)
            t: float, hopping amplitude
            mu_sc: float, chemical potential in the SC leads
            mu_m: float, chemical potential in the middle normal region
            h: float, Zeeman energy
            alpha: float, spin-orbit coupling strength
            delta: float, SC pairing potential (magnitude)
            phi: float, phase difference between the SC leads
            SL: int, number of sites in left SC lead
            SR: int, number of sites in right SC lead
            SM: int, number of sites in middle normal region
            symmetric: bool, whether to use symmetric gauge (±φ/2) or asymmetric (full φ on right)

    Returns: A tuple (H_slices, V_2d) where H_slices is a list of Hamiltonian slices for each region and V_2d is the inter-slice hopping matrix.
    '''
    phi_L = -phi/2 if symmetric else 0.0
    phi_R =  phi/2 if symmetric else phi

    H_L = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j*phi_L))
    H_R = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j*phi_R))
    H_M = build_sns_slice_2d(N_y, t, mu_m,  h, alpha, 0.0)

    V_2d = np.kron(np.eye(N_y), modules.t_matrix(t, alpha))  # x-direction inter-slice hopping: real Rashba

    H_slices = ([H_L.copy() for _ in range(SL)] +
                [H_M.copy() for _ in range(SM)] +
                [H_R.copy() for _ in range(SR)])

    return H_slices, V_2d 


def get_Vx_2d(N_y, t, alpha):
    """
    Constructs the 4Ny x 4Ny hopping matrix between two 2D slices.
    """
    # 1. Get the base 4x4 hopping matrix in x-direction
    # This matrix contains the kinetic hopping and Rashba SOC terms
    Vx_4x4 = modules.t_matrix(t, alpha)
    
    # 2. Use Kronecker product to repeat it for each transverse site
    # I_y is the identity matrix of size Ny
    I_y = np.eye(N_y, dtype=np.complex128)
    
    # This creates a block-diagonal matrix where each diagonal block is Vx_4x4
    Vx_2d = np.kron(I_y, Vx_4x4)
    
    return Vx_2d

def get_local_block(G_slice, y, dof=4):
    s = y * dof
    e = (y + 1) * dof
    return G_slice[s:e, s:e]

def build_sns_junction_2d():
    H_slices, V = build_sns_junction_sliced_2d(
        N_y, t, mu_sc, mu_n, B, alpha, Delta, 0.0,
        SL, SR, SM, symmetric=SYMMETRIC
    )
    return H_slices, V

#%% ── Parameters ─────────────────────────────────────────────────────────────
SYMMETRIC = False

SL, SM, SR = 30, 10, 30
N_y = 10

DOF = 4
Delta = 0.1
mu_sc = 0.0025
mu_n = 0.01
t = 1.0
alpha = 0.4
B = 0.3
eta = 1e-3
phi_fixed = np.pi
N_E = 61
N_PHI = 61

energies = np.linspace(-0.15, 0.15, N_E)
phases = np.linspace(0, 2*np.pi, N_PHI)

probe_x = [1, SM//2, SM-1]
probe_y = [N_y // 2]

probe_labels = [f"x={x}, y={probe_y[0]}" for x in probe_x]


#%% ── building blocks ──────────────────────────────────────────────────────

H_lead_slice = build_sns_slice_2d(N_y, t, mu_sc, B, alpha, Delta * np.exp(1j*phi_fixed))

V_x_2d = get_Vx_2d(N_y, t, alpha)

H_slices_list, V_list = build_sns_junction_2d()

onsite_sc = modules.onsite_matrix(t, mu_sc, B, Delta)

V_x = modules.t_matrix(t, alpha)
V_y = modules.t_matrix_y(t, alpha)


#%% ── Energy sweep ───────────────────────────────────────────────────────────
print("\n══ Energy sweep ══")

ldos_e = np.zeros((len(probe_x), N_E, 2))
pair_e = np.zeros_like(ldos_e)

ldos_map_fin = np.zeros((SM, N_E))
ldos_map_inf = np.zeros((SM, N_E))

# First iteration with debug info
E = energies[0]

# finite system
G_fin, *_ = modules.get_rgf_finite_system(
    H_slices_list, V_list, E, eta=eta, return_full=False
)

# infinite leads
gL, gR = modules.get_surface_gfs_2d_phased(E, H_lead_slice, V_x_2d, N_y, phi_fixed, symmetric=SYMMETRIC, eta=eta)
G_inf, *_ = modules.get_rgf_sns(
    H_slices_list[SL:SL+SM], V_list, gL, gR, E, eta=eta, return_full=False
)

print(f"\n  E={E:.4f}:")
print(f"    G_fin shape: {G_fin.shape}")
print(f"    G_inf shape: {G_inf.shape}")
print(f"    Testing probe indexing:")
for i, x in enumerate(probe_x):
    idx_fin = SL + x
    print(f"      probe_x[{i}]={x}: G_fin[{idx_fin}].shape={G_fin[idx_fin].shape}, G_inf[{x}].shape={G_inf[x].shape}")
    blk_fin = get_local_block(G_fin[idx_fin], probe_y[0])
    blk_inf = get_local_block(G_inf[x], probe_y[0])
    print(f"        Local blocks: fin={blk_fin.shape}, inf={blk_inf.shape}")
    print(f"        LDOS: fin={-np.imag(np.trace(blk_fin))/np.pi:.6f}, inf={-np.imag(np.trace(blk_inf))/np.pi:.6f}")

# Now loop through all energies
for e_idx, E in enumerate(energies):

    # finite system
    G_fin, *_ = modules.get_rgf_finite_system(
        H_slices_list, V_list, E, eta=eta, return_full=False
    )

    # infinite leads
    gL, gR = modules.get_surface_gfs_2d_phased(E, H_lead_slice, V_x_2d, N_y, phi_fixed, symmetric=SYMMETRIC, eta=eta)
    G_inf, *_ = modules.get_rgf_sns(
        H_slices_list[SL:SL+SM], V_list, gL, gR, E, eta=eta, return_full=False
    )

    for i, x in enumerate(probe_x):

        blk_fin = get_local_block(G_fin[SL + x], probe_y[0])
        blk_inf = get_local_block(G_inf[x], probe_y[0])

        for m, blk in enumerate([blk_fin, blk_inf]):
            ldos_e[i, e_idx, m] = -np.imag(np.trace(blk)) / np.pi
            pair_e[i, e_idx, m] = modules.get_pairing_amplitude(blk)

    # full LDOS map (center slice only in y)
    for x in range(SM):
        blk_f = get_local_block(G_fin[SL + x], probe_y[0])
        blk_i = get_local_block(G_inf[x], probe_y[0])

        ldos_map_fin[x, e_idx] = -np.imag(np.trace(blk_f)) / np.pi
        ldos_map_inf[x, e_idx] = -np.imag(np.trace(blk_i)) / np.pi


#%% ── Phase sweep ────────────────────────────────────────────────────────────
print("\n══ Phase sweep ══")

ldos_p = np.zeros((len(probe_x), N_PHI, 2))
pair_p = np.zeros_like(ldos_p)

E0 = 0.0

for p_idx, phi in enumerate(phases):

    H_full, V_2d_phi = build_sns_junction_sliced_2d(
        N_y, t, mu_sc, mu_n, B, alpha, Delta, phi,
        SL, SR, SM, symmetric=SYMMETRIC
    )

    G_fin, *_ = modules.get_rgf_finite_system(H_full, V_2d_phi, E0, eta=eta)

    gL, gR = modules.get_surface_gfs_2d_phased(E0, H_lead_slice, V_x_2d, N_y, phi, symmetric=SYMMETRIC, eta=eta)

    G_inf, *_ = modules.get_rgf_sns(
        H_full[SL:SL+SM], V_2d_phi, gL, gR, E0, eta=eta, return_full=False
    )

    for i, x in enumerate(probe_x):

        blk_f = get_local_block(G_fin[SL + x], probe_y[0])
        blk_i = get_local_block(G_inf[x], probe_y[0])

        for m, blk in enumerate([blk_f, blk_i]):
            ldos_p[i, p_idx, m] = -np.imag(np.trace(blk)) / np.pi
            pair_p[i, p_idx, m] = modules.get_pairing_amplitude(blk)

#%% ── FIGURE 1: LDOS vs Energy ───────────────────────────────────────────────
fig, axes = plt.subplots(len(probe_x), 1, figsize=(8, 6), sharex=True)

for i, lbl in enumerate(probe_labels):
    ax = axes[i]

    ax.plot(energies, ldos_e[i, :, 0], '--', label='finite')
    ax.plot(energies, ldos_e[i, :, 1], '-', label='infinite')

    ax.set_title(lbl)
    ax.set_ylabel("LDOS")
    ax.legend()
    ax.grid()

axes[-1].set_xlabel("Energy")
plt.tight_layout()


#%% ── FIGURE 2: Phase sweep ────────────────────────────────────────────────
fig, axes = plt.subplots(len(probe_x), 2, figsize=(10, 6), sharex=True)

for i, lbl in enumerate(probe_labels):

    ax1, ax2 = axes[i]

    ax1.plot(phases/np.pi, ldos_p[i, :, 0], '--')
    ax1.plot(phases/np.pi, ldos_p[i, :, 1], '-')
    ax1.set_ylabel("LDOS")

    ax2.plot(phases/np.pi, pair_p[i, :, 0], '--')
    ax2.plot(phases/np.pi, pair_p[i, :, 1], '-')
    ax2.set_ylabel("Pairing")

    ax1.set_title(lbl)
    ax1.grid()
    ax2.grid()

axes[-1, 0].set_xlabel("φ/π")
axes[-1, 1].set_xlabel("φ/π")
plt.tight_layout()


#%% ── FIGURE 3: LDOS map (x vs energy) ──────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

im0 = ax[0].imshow(ldos_map_fin.T, aspect='auto', origin='lower',
                   extent=[0, SM, energies[0], energies[-1]])
ax[0].set_title("Finite system")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Energy")

im1 = ax[1].imshow(ldos_map_inf.T, aspect='auto', origin='lower',
                   extent=[0, SM, energies[0], energies[-1]])
ax[1].set_title("Infinite leads")
ax[1].set_xlabel("x")

plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")
plt.show()

print("Finished 2D sweep.")
#%%