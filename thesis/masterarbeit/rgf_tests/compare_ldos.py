#%%
from scipy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(parent_dir)

import my_functions as myf
def slices_to_matrix(H_slices, V):
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    dim = N * dof

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(N):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = H_slices[i]

    for i in range(N-1):
        H[i*dof:(i+1)*dof, (i+1)*dof:(i+2)*dof] = V
        H[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof] = V.conj().T

    return H

def get_phase_matrix(phi):
    half = phi / 2
    return np.diag([np.exp(1j*half), np.exp(1j*half), np.exp(-1j*half), np.exp(-1j*half)])


def finite_lead_surface_gf(E, n_lead, onsite, V, dof, eta):
    """
    Compute surface Green's function from a finite lead by inversion.
    """
    # build uniform lead Hamiltonian
    H_slices = [onsite for _ in range(n_lead)]
    H_lead = slices_to_matrix(H_slices, V)


    dim = H_lead.shape[0]
    I = np.eye(dim, dtype=np.complex128)

    G = inv((E + 1j*eta)*I - H_lead)

    # surface block = first unit cell
    return G[:dof, :dof]

#%% Parameters
sites_left, sites_mid, sites_right = 100, 30, 100
dof = 4

Delta = 0.1
mu_sc = 0.025
mu_n = 0.1
t = 1.0
alpha = 0.5
B = 0.4
eta = 1e-4
phi_fixed = np.pi

V = myf.t_matrix(t, alpha)

# ---------- Build systems ----------
H_slices_full, V_slices_full = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi_fixed,
    sites_left, sites_right, sites_mid
)

H_slices_mid, _ = myf.build_middle_region(
    t, mu_n, alpha, B, sites_mid
)

# For inversion reference
H_full_matrix = slices_to_matrix(H_slices_full, V)
dim_full = H_full_matrix.shape[0]
I_full = np.eye(dim_full, dtype=np.complex128)

sites_to_plot = [ 1, sites_mid//2, sites_mid-1]
energies = np.linspace(-t, t, 71)


textstr = '\n'.join((
r'$\mu_n/t = {:.3f}$'.format(mu_n),
r'$\mu_sc/t = {:.3f}$'.format(mu_sc),
r'$t = {:.1f}$'.format(t),
r'$\alpha/t = {:.1f}$'.format(alpha),
r'$\Delta/t = {:.1f}$'.format(Delta),
r'$h/t = {:.3f}$'.format(B)
))
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20


#%% 1. Energy Sweep
results_e = {s: {'inf': [], 'fin': [], 'inv': []} for s in sites_to_plot}

for E in energies:

    # --- Infinite leads ---
    onsite_phase_l = myf.onsite_matrix(t, mu_sc, B, Delta)
    onsite_phase_r = myf.onsite_matrix(t, mu_sc, B, Delta)
    onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)
    g_L_raw, _ = myf.get_surface_gf(E, onsite_phase_l, V.conj().T, eta=eta)
    g_R_raw, _ = myf.get_surface_gf(E, onsite_phase_r, V, eta=eta)

    # 2. Apply Phase via Gauge Rotation
    # Rotate Left by -phi/2 and Right by +phi/2
    U_R = get_phase_matrix(phi_fixed)
    U_L = get_phase_matrix(-phi_fixed)

    g_R = U_R @ g_R_raw @ U_R.conj().T
    g_L = U_L @ g_L_raw @ U_L.conj().T

    G_inf = myf.get_rgf_sns(
        H_slices_mid, V, g_L, g_R, E, eta=eta, return_full=True
    )

    # --- Finite RGF ---
    G_fin = myf.get_rgf_finite_system(
        H_slices_full, V, E, eta=eta, return_full=True
    )

    # --- Direct inversion ---
    G_inv = inv((E + 1j*eta)*I_full - H_full_matrix)

    for s in sites_to_plot:
        idx_m = s * dof
        idx_f = (sites_left + s) * dof

        b_inf = G_inf[idx_m:idx_m+dof, idx_m:idx_m+dof]
        b_fin = G_fin[idx_f:idx_f+dof, idx_f:idx_f+dof] #type: ignore
        b_inv = G_inv[idx_f:idx_f+dof, idx_f:idx_f+dof]

        results_e[s]['inf'].append(
            -np.imag(np.trace(b_inf)) / np.pi #type: ignore
        )
        results_e[s]['fin'].append(
            -np.imag(np.trace(b_fin)) / np.pi #type: ignore
        )
        results_e[s]['inv'].append(
            -np.imag(np.trace(b_inv)) / np.pi
        )


#%% 2. Phase Sweep
phases = np.linspace(0, 2*np.pi, 71)
target_E = 0

results_p = {s: {'inv': [], 'fin': [], 'inf': [], 'p_inv': [], 'p_fin': [], 'p_inf': []} for s in sites_to_plot}

for p_val in phases:

    H_slices_full, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta, p_val,
        sites_left, sites_right, sites_mid
    )
    H_slices_mid = H_slices_full[sites_left:sites_left+sites_mid]
    H_full_matrix = slices_to_matrix(H_slices_full, V)

    n_lead = 300
    onsite_phase_l = myf.onsite_matrix(t, mu_sc, B, Delta)
    onsite_phase_r = myf.onsite_matrix(t, mu_sc, B, Delta)
    onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)
    g_L_raw, _ = myf.get_surface_gf(target_E, onsite_phase_l, V.conj().T, eta=eta)
    g_R_raw, _ = myf.get_surface_gf(target_E, onsite_phase_r, V, eta=eta)
    
    U_R = get_phase_matrix(p_val)
    U_L = get_phase_matrix(-p_val)

    g_R = U_R @ g_R_raw @ U_R.conj().T
    g_L = U_L @ g_L_raw @ U_L.conj().T
    
    G_inf = myf.get_rgf_sns(H_slices_mid, V, g_L, g_R, target_E, eta=eta, return_full=True)
    G_fin = myf.get_rgf_finite_system(H_slices_full, V, target_E, eta=eta, return_full=True)
    G_inv = inv((target_E + 1j*eta)*I_full - H_full_matrix)

    for s in sites_to_plot:
        idx_f = (sites_left + s) * dof
        idx_m = s * dof

        b_inv = G_inv[idx_f:idx_f+dof, idx_f:idx_f+dof]
        b_fin = G_fin[idx_f:idx_f+dof, idx_f:idx_f+dof] #type: ignore
        b_inf = G_inf[idx_m:idx_m+dof, idx_m:idx_m+dof] #type: ignore

        results_p[s]['inv'].append(-np.imag(np.trace(b_inv)) / np.pi)
        results_p[s]['fin'].append(-np.imag(np.trace(b_fin)) / np.pi)
        results_p[s]['inf'].append(-np.imag(np.trace(b_inf)) / np.pi)

        results_p[s]['p_inv'].append(np.abs(b_inv[0,3]))
        results_p[s]['p_fin'].append(np.abs(b_fin[0,3]))
        results_p[s]['p_inf'].append(np.abs(b_inf[0,3]))


#%% Plotting 
titles = ["Direct Inversion (Finite)", "Recursive GF (Finite)", "Sancho-Lopez (Infinite)"]
keys = ['inv', 'fin', 'inf']
p_keys = ['p_inv', 'p_fin', 'p_inf']
row_colors = ['tab:green', 'tab:red', 'tab:blue']

for s_idx, s in enumerate(sites_to_plot):
    # Create a new large figure for this specific site
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Site {s}", fontsize=20, fontweight='bold', y=0.98)
    
    # 3x3 Grid: (Rows: Energy, Phase-LDOS, Phase-Pairing) x (Cols: Inv, Fin, Inf)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    for m_idx in range(3):  # 0: Inversion, 1: Finite RGF, 2: Infinite Leads
        
        # --- ROW 1: LDOS vs Energy (at fixed phi) ---
        ax_e = fig.add_subplot(gs[0, m_idx])
        ax_e.plot(energies, results_e[s][keys[m_idx]], color=row_colors[m_idx], lw=2)
        ax_e.set_title(titles[m_idx], fontsize=15, pad=10)
        ax_e.set_ylabel("LDOS")
        ax_e.set_xlabel("Energy ($E/t$)")
        ax_e.grid(True, alpha=0.3)

        # --- ROW 2: LDOS vs Phase (at E=0) ---
        ax_p = fig.add_subplot(gs[1, m_idx])
        ax_p.plot(phases/np.pi, results_p[s][keys[m_idx]], color=row_colors[m_idx], lw=2)
        ax_p.set_ylabel("LDOS ($E=0$)")
        ax_p.set_xlabel(r"Phase $\phi/\pi$")
        ax_p.grid(True, alpha=0.3)

        # --- ROW 3: Pairing Amplitude vs Phase (at E=0) ---
        ax_pair = fig.add_subplot(gs[2, m_idx])
        ax_pair.plot(phases/np.pi, results_p[s][p_keys[m_idx]], color=row_colors[m_idx], linestyle='--', lw=2)
        ax_pair.set_ylabel("Pairing Amp $|G_{03}|$")
        ax_pair.set_xlabel(r"Phase $\phi/\pi$")
        ax_pair.grid(True, alpha=0.3)

    # Add the parameter text box to the bottom of each figure
    fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props, verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) #type: ignore
    plt.show() # This will pop up a separate window/plot for each site


# %%
import numpy as np
from my_functions import onsite_matrix, get_surface_gf, t_matrix
def check_phase_in_g(g, phi_expected):
    g_eh = g[0:2, 2:4]
    g_he = g[2:4, 0:2]
    
    # We check both blocks to see which one carries the expected phase
    phase_eh = np.angle(g_eh[0, 0])
    phase_he = np.angle(g_he[0, 0])
    
    print(f"--- Phase Analysis ---")
    print(f"Expected:  {phi_expected:.4f}")
    print(f"In g_eh:   {phase_eh:.4f} (Diff: {phase_eh - phi_expected:.4f})")
    print(f"In g_he:   {phase_he:.4f} (Diff: {phase_he + phi_expected:.4f})")
    
    # Correct symmetry for (c_up, c_dn, c_dn_dag, c_up_dag) basis
    is_bdg = np.allclose(g_eh, g_he.conj().T)
    print(f"BdG Consistency (g_eh == g_he.H): {is_bdg}")

    if not is_bdg:
        # If that fails, check the standard antisymmetric basis requirement
        is_alt_bdg = np.allclose(g_eh, -g_he.conj().T)
        print(f"Alt BdG Check (g_eh == -g_he.H): {is_alt_bdg}")

phi = np.pi / 2
delta_mag = 0.1
t = 1.0
mu = 0.025

# Approach 1: phase in eps
delta_phased = delta_mag * np.exp(1j * phi)
eps_phased = onsite_matrix(t, mu, h=0, delta=delta_phased)
g_approach1, _ = get_surface_gf(0.0, eps_phased, t_matrix(t, alpha=0))
print("=== Approach 1: phase in eps ===")
check_phase_in_g(g_approach1, phi)

# Approach 2: gauge rotation after, with CORRECT U
g_0, _ = get_surface_gf(0.0, onsite_matrix(t, mu, h=0, delta=delta_mag), t_matrix(t, alpha=0))
half = phi / 2 
# Electrons (rows/cols 0,1) get e^{+i*phi/2}, holes (rows/cols 2,3) get e^{-i*phi/2}
half = phi / 2 
# To get +phi phase, use +half for electrons, -half for holes
U = np.diag([np.exp(1j*half), np.exp(1j*half), np.exp(-1j*half), np.exp(-1j*half)])
g_approach2 = U @ g_0 @ U.conj().T
print("\n=== Approach 2: gauge rotation ===")
check_phase_in_g(g_approach2, phi)
# %%
