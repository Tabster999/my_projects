#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import my_functions as myf
#%% Parameters & Setup
sites_left, sites_mid, sites_right = 150, 50, 150
sites_tot = sites_left + sites_mid + sites_right
dof = 4
Delta, mu_sc, mu_n = 0.1, 0.1, 0.001
t, alpha, B, eta = 1.0, 0.5, 0.3, 1e-3
phi_fixed = np.pi

V = myf.t_matrix(t, alpha)
H_full = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, sites_left, sites_right, sites_mid, dof)
H_mid = myf.build_middle_region(t, mu_n, alpha, B, sites_mid, dof)
I_full = np.eye(H_full.shape[0], dtype=np.complex128)

sites_to_plot = [0]
energies = np.concatenate([
    np.linspace(-1.0, -0.1, 100),
    np.linspace(-0.1, 0.1, 150), 
    np.linspace(0.1, 1.0, 100)
])

#%% 1. Energy Sweep (LDOS vs Energy)
results_e = {s: {'inf': [], 'fin': [], 'inv': []} for s in sites_to_plot}

for E in energies:
    # Method 1: Infinite leads (Self-Energies)
    g_L, _ = myf.get_surface_gf(E, myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(-1j*phi_fixed/2)), V, eta=eta)
    g_R, _ = myf.get_surface_gf(E, myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(1j*phi_fixed/2)), V, eta=eta)
    G_inf = myf.get_rgf_sns(H_mid, V, g_L, g_R, E, eta=eta, return_full=True)
    
    # Method 2 & 3: Finite system
    G_fin = myf.get_rgf_finite_system(H_full, V, E, eta=eta, return_full=True)[0]
    G_inv = np.linalg.inv((E + 1j*eta)*I_full - H_full)
    
    for s in sites_to_plot:
        idx_m, idx_f = s * dof, (sites_left + s) * dof
        results_e[s]['inf'].append(-np.imag(np.trace(G_inf[idx_m:idx_m+dof, idx_m:idx_m+dof])) / np.pi)
        results_e[s]['fin'].append(-np.imag(np.trace(G_fin[idx_f:idx_f+dof, idx_f:idx_f+dof])) / np.pi)
        results_e[s]['inv'].append(-np.imag(np.trace(G_inv[idx_f:idx_f+dof, idx_f:idx_f+dof])) / np.pi)

#%% 2. Phase Sweep at fixed E
phases = np.linspace(0, 2 * np.pi, 100)
target_E = 0
results_p = {s: {'inv': [], 'fin': [], 'inf': [], 'p_inv': [], 'p_fin': [], 'p_inf': []} for s in sites_to_plot}

for p_val in phases:
    H_phi = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, p_val, sites_left, sites_right, sites_mid, dof)
    g_L0, _ = myf.get_surface_gf(target_E, myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(-1j*p_val/2)), V, eta=eta)
    g_R0, _ = myf.get_surface_gf(target_E, myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(1j*p_val/2)), V, eta=eta)
    
    G_inf = myf.get_rgf_sns(H_mid, V, g_L0, g_R0, target_E, eta=eta, return_full=True)
    G_fin = myf.get_rgf_finite_system(H_phi, V, target_E, eta=eta, return_full=True)[0]
    G_inv = np.linalg.inv((target_E + 1j*eta) * I_full - H_phi)

    for s in sites_to_plot:
        idx_f, idx_m = (sites_left + s) * dof, s * dof
        b_inv, b_fin, b_inf = G_inv[idx_f:idx_f+dof, idx_f:idx_f+dof], G_fin[idx_f:idx_f+dof, idx_f:idx_f+dof], G_inf[idx_m:idx_m+dof, idx_m:idx_m+dof]
        
        results_p[s]['inv'].append(-np.imag(np.trace(b_inv)) / np.pi)
        results_p[s]['fin'].append(-np.imag(np.trace(b_fin)) / np.pi)
        results_p[s]['inf'].append(-np.imag(np.trace(b_inf)) / np.pi)
        results_p[s]['p_inv'].append(np.abs(b_inv[0, 3]))
        results_p[s]['p_fin'].append(np.abs(b_fin[0, 3]))
        results_p[s]['p_inf'].append(np.abs(b_inf[0, 3]))

#%% Plotting
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig)
titles = ["Inversion", "Finite (RGF)", "Infinite (Self-E)"]
keys = ['inv', 'fin', 'inf']
colors = ['g', 'r', 'b']

# ROW 1: LDOS vs Energy 
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    for s in sites_to_plot:
        ax.plot(energies, results_e[s][keys[i]], color=colors[i], label=f'Site {s}')
    ax.set_title(f'{titles[i]}: LDOS vs Energy')
    ax.set_xlabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

# ROW 2: LDOS vs Phase 
for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    for s in sites_to_plot:
        ax.plot(phases/np.pi, results_p[s][keys[i]], color=colors[i], label=f'Site {s}')
    ax.set_title(f'{titles[i]}: LDOS vs Phase')
    ax.set_xlabel(r'$\phi/\pi$')
    ax.legend()
    ax.grid(True, alpha=0.3)

# ROW 3: Anomalous Correlation |F_{↑↑}| vs Phase
ax_p = fig.add_subplot(gs[2, :])
for s in sites_to_plot:
    ax_p.plot(phases/np.pi, results_p[s]['p_inv'], 'g:', label=f'Inv Site {s}')
    ax_p.plot(phases/np.pi, results_p[s]['p_fin'], 'r--', label=f'RGF Site {s}')
    ax_p.plot(phases/np.pi, results_p[s]['p_inf'], 'b-', label=f'Inf Site {s}')
ax_p.set_title(r'Anomalous Correlation $|F_{\uparrow\uparrow}|$ at $E=0$')
ax_p.set_xlabel(r'$\phi/\pi$')
ax_p.set_ylabel('Magnitude')
ax_p.legend(ncol=3)
ax_p.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()