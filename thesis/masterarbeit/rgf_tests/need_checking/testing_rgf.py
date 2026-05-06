#%% all in one main execution
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --- Environment Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
import my_functions as myf

# --- Physics Helpers ---
def phase_matrix(theta):
    """U(1) gauge rotation in Nambu basis."""
    return np.diag([
        np.exp(-1j * theta/2), np.exp(-1j * theta/2),
        np.exp(1j * theta/2), np.exp(1j * theta/2)
    ]).astype(np.complex128)

def get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def get_surface_gfs(E, phi, params, symmetric=True):
    onsite_sc = myf.onsite_matrix(params['t'], params['mu_sc'], params['B'], params['Delta'])
    V = params['V']
    
    if symmetric:
        UL, UR = phase_matrix(+phi / 2.0), phase_matrix(-phi / 2.0)
    else:
        UL, UR = np.eye(4, dtype=np.complex128), phase_matrix(-phi)
    
    g_L, _ = myf.get_surface_gf(E, onsite_sc, V.conj().T, eta=params['eta'])
    g_R, _ = myf.get_surface_gf(E, onsite_sc, V, eta=params['eta'])
    
    return UL @ g_L @ UL.conj().T, UR @ g_R @ UR.conj().T

# --- Parallel Workers ---
def worker_infinite(e_idx, E, phases, H_mid_slices, params, probes, symmetric):
    """Calculates LDOS for multiple probes in the Infinite Lead setup."""
    # probes = [idx_interface, idx_middle]
    res_int = np.zeros(len(phases))
    res_mid = np.zeros(len(phases))
    
    for p_idx, phi in enumerate(phases):
        g_L, g_R = get_surface_gfs(E, phi, params, symmetric)
        G_inf, *_ = myf.get_rgf_sns(H_mid_slices, params['V'], g_L, g_R, E, eta=params['eta'], return_full=False)
        res_int[p_idx] = get_ldos(G_inf[probes[0]])
        res_mid[p_idx] = get_ldos(G_inf[probes[1]])
        
    return e_idx, res_int, res_mid

def worker_finite(e_idx, E, phases, params, probes_fin, symmetric):
    """Calculates LDOS for multiple probes in the Finite setup."""
    res_int = np.zeros(len(phases))
    res_mid = np.zeros(len(phases))
    
    for p_idx, phi in enumerate(phases):
        H_full_p, _ = myf.build_sns_junction_sliced(
            params['t'], params['mu_sc'], params['mu_n'], params['alpha'], 
            params['B'], params['Delta'], phi,
            params['SL'], params['SR'], params['SM'], symmetric=symmetric
        )
        G_fin, *_ = myf.get_rgf_finite_system(H_full_p, params['V'], E, eta=params['eta'], return_full=False)
        res_int[p_idx] = get_ldos(G_fin[probes_fin[0]])
        res_mid[p_idx] = get_ldos(G_fin[probes_fin[1]])
        
    return e_idx, res_int, res_mid

# --- Main Execution ---
def main():
    p = {
        't': 1.0, 'alpha': 0.15, 'Delta': 0.01, 'B': 0.025, 
        'mu_sc': 0.0, 'mu_n': 0.0, 'eta': 0.0001,
        'SL': 150, 'SM': 40, 'SR': 150  
    }
    p['V'] = myf.t_matrix(p['t'], p['alpha'])
    
    SYMMETRIC = True
    N_E, N_PHI = 100, 80 
    energies = np.linspace(-p['Delta']*2.5, p['Delta']*2.5, N_E)
    phases = np.linspace(0, 2*np.pi, N_PHI)
    
    # Define Probes
    # In Sancho-Lopez: indexing 0 to SM-1
    probe_inf = [0, p['SM'] // 2] 
    # In Finite: indexing 0 to (SL+SM+SR)-1. Interface is at index SL.
    probe_fin = [p['SL'], p['SL'] + (p['SM'] // 2)]
    
    H_mid_slices, _ = myf.build_middle_region(p['t'], p['mu_n'], p['alpha'], p['B'], p['SM'])
    
    print("--> Running Infinite Lead Simulations...")
    results_inf = Parallel(n_jobs=-2)(
        delayed(worker_infinite)(i, E, phases, H_mid_slices, p, probe_inf, SYMMETRIC)
        for i, E in enumerate(energies)
    )
    
    print("--> Running Finite System Simulations...")
    results_fin = Parallel(n_jobs=-2)(
        delayed(worker_finite)(i, E, phases, p, probe_fin, SYMMETRIC)
        for i, E in enumerate(energies)
    )
    
    # Data Assembly
    ldos_inf_int = np.zeros((N_PHI, N_E))
    ldos_inf_mid = np.zeros((N_PHI, N_E))
    ldos_fin_int = np.zeros((N_PHI, N_E))
    ldos_fin_mid = np.zeros((N_PHI, N_E))
    
    for i, r_int, r_mid in results_inf:
        ldos_inf_int[:, i], ldos_inf_mid[:, i] = r_int, r_mid
        
    for i, r_int, r_mid in results_fin:
        ldos_fin_int[:, i], ldos_fin_mid[:, i] = r_int, r_mid

    plot_comparison_grid(phases, energies, ldos_inf_int, ldos_inf_mid, ldos_fin_int, ldos_fin_mid, p)

def plot_comparison_grid(phases, energies, inf_int, inf_mid, fin_int, fin_mid, params):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    
    data = [
        (inf_int, "Infinite: Interface (Site 0)"),
        (inf_mid, f"Infinite: Middle (Site {params['SM']//2})"),
        (fin_int, "Finite: Interface"),
        (fin_mid, "Finite: Middle")
    ]
    
    for ax, (d, title) in zip(axes.flatten(), data):
        im = ax.pcolormesh(phases/np.pi, energies, d.T, cmap='magma', shading='auto', rasterized=True)
        ax.set_title(title, fontweight='bold')
        ax.axhline(params['Delta'], color='cyan', ls=':', alpha=0.6)
        ax.axhline(-params['Delta'], color='cyan', ls=':', alpha=0.6)
        plt.colorbar(im, ax=ax, label="LDOS")

    # Labels
    axes[1,0].set_xlabel(r"$\phi / \pi$")
    axes[1,1].set_xlabel(r"$\phi / \pi$")
    axes[0,0].set_ylabel("Energy (E)")
    axes[1,0].set_ylabel("Energy (E)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
import my_functions as myf
# --- Setup Parameters ---
p = {
    't': 1.0, 'alpha': 0.15, 'Delta': 0.01, 'B': 0.03, 
    'mu_sc': 0.0, 'mu_n': 0.0, 'eta': 0.0002,
    'SL': 120, 'SM': 60, 'SR': 120  
}
p['V'] = myf.t_matrix(p['t'], p['alpha'])

# Energy and Phase Grids
N_E, N_PHI = 80, 80
energies = np.linspace(-p['Delta']*2.5, p['Delta']*2.5, N_E)
phases = np.linspace(0, 2*np.pi, N_PHI)

def get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def phase_matrix(phi):
    h = phi / 2
    return np.diag([np.exp(1j*h), np.exp(1j*h), np.exp(-1j*h), np.exp(-1j*h)]).astype(np.complex128)

def get_surface_gfs(E, phi, params):
    onsite_sc = myf.onsite_matrix(params['t'], params['mu_sc'], params['B'], params['Delta'])
    # Symmetric gauge: -phi/2 on Left, +phi/2 on Right
    UL, UR = phase_matrix(-phi/2), phase_matrix(phi/2)
    g_L_bare, _ = myf.get_surface_gf(E, onsite_sc, params['V'].conj().T, eta=params['eta'])
    g_R_bare, _ = myf.get_surface_gf(E, onsite_sc, params['V'], eta=params['eta'])
    return UL @ g_L_bare @ UL.conj().T, UR @ g_R_bare @ UR.conj().T

# --- Core Computation Worker ---
def compute_point(E, phi):
    # 1. Infinite (Sancho-Lopez)
    H_mid, _ = myf.build_middle_region(p['t'], p['mu_n'], p['alpha'], p['B'], p['SM'])
    gL, gR = get_surface_gfs(E, phi, p)
    G_inf, *_ = myf.get_rgf_sns(H_mid, p['V'], gL, gR, E, eta=p['eta'], return_full=False)
    
    inf_interface = get_ldos(G_inf[0])             # Index 0
    inf_middle    = get_ldos(G_inf[p['SM'] // 2])  # Index SM/2

    # 2. Finite (Full RGF)
    H_full, _ = myf.build_sns_junction_sliced(p['t'], p['mu_sc'], p['mu_n'], p['alpha'], 
                                              p['B'], p['Delta'], phi, p['SL'], p['SR'], p['SM'], True)
    G_fin, *_ = myf.get_rgf_finite_system(H_full, p['V'], E, eta=p['eta'], return_full=False)
    
    fin_interface = get_ldos(G_fin[p['SL']])                # Index SL
    fin_middle    = get_ldos(G_fin[p['SL'] + p['SM']//2])   # Index SL + SM/2
    
    return inf_interface, inf_middle, fin_interface, fin_middle

# --- Execution ---
print("Starting 2D Sweep...")
# Parallelizing over phases for speed
results = Parallel(n_jobs=-1)(
    delayed(lambda phi: [compute_point(E, phi) for E in energies])(ph) 
    for ph in phases
)

# Reshape results: (Phase, Energy, 4_probes)
data = np.array(results) 

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
titles = [
    "Infinite: Interface (Site 0)", "Infinite: Middle (Site SM/2)",
    "Finite: Interface (Site SL)",  "Finite: Middle (Site SL + SM/2)"
]

# Mapping data indices to the 2x2 plot
plot_map = [data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]]

for i, ax in enumerate(axes.flatten()):
    im = ax.pcolormesh(phases/np.pi, energies, plot_map[i].T, 
                       cmap='magma', shading='auto', rasterized=True)
    ax.set_title(titles[i], fontweight='bold')
    ax.axhline(p['Delta'], color='cyan', ls=':', alpha=0.5)
    ax.axhline(-p['Delta'], color='cyan', ls=':', alpha=0.5)
    plt.colorbar(im, ax=ax, label="LDOS")

axes[1,0].set_xlabel(r"$\phi / \pi$")
axes[1,1].set_xlabel(r"$\phi / \pi$")
axes[0,0].set_ylabel("Energy (E)")
axes[1,0].set_ylabel("Energy (E)")

plt.tight_layout()
plt.show()
# %%
