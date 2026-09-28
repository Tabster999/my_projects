#%% --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

#%% --- HAMILTONIAN BUILDERS (4x4 Nambu-Spin Basis) ---
# Basis: [c_up, c_down, -c_down^dag, c_up^dag]

def H_onsite(eps, Delta):
    """Creates the 4x4 onsite block for a given energy and Superconducting gap."""
    H = np.zeros((4, 4), dtype=complex)

    H[0, 0] = eps;  H[1, 1] = eps
    H[2, 2] = -eps; H[3, 3] = -eps

    H[0, 2] = Delta; H[1, 3] = Delta
    H[2, 0] = np.conj(Delta); H[3, 1] = np.conj(Delta)
    return H

def V_hop(t):
    """Creates the 4x4 hopping block: -t * tau_z"""
    return np.diag([-t, -t, t, t]).astype(complex)

def build_SC_ribbon(Nx, eps, tx, ty, Delta):
    """Builds H_intra and H_inter for a 2D SC lead of width Nx."""
    H_intra = np.zeros((4*Nx, 4*Nx), dtype=complex)
    H_inter = np.zeros((4*Nx, 4*Nx), dtype=complex)
    
    H_on = H_onsite(eps, Delta)
    Vx = V_hop(tx)
    Vy = V_hop(ty)
    
    for i in range(Nx):
        H_intra[i*4:(i+1)*4, i*4:(i+1)*4] = H_on
        H_inter[i*4:(i+1)*4, i*4:(i+1)*4] = Vy
        
    for i in range(Nx - 1):
        H_intra[i*4:(i+1)*4, (i+1)*4:(i+2)*4] = Vx
        H_intra[(i+1)*4:(i+2)*4, i*4:(i+1)*4] = np.conj(Vx).T
        
    return H_intra, H_inter

def build_Central(Nx, eps, tx):
    """Builds H_C for the central 1D Normal region."""
    H_C = np.zeros((4*Nx, 4*Nx), dtype=complex)
    H_on = H_onsite(eps, 0.0)
    Vx = V_hop(tx)
    for i in range(Nx):
        H_C[i*4:(i+1)*4, i*4:(i+1)*4] = H_on
    for i in range(Nx - 1):
        H_C[i*4:(i+1)*4, (i+1)*4:(i+2)*4] = Vx
        H_C[(i+1)*4:(i+2)*4, i*4:(i+1)*4] = np.conj(Vx).T
    return H_C

#%% --- BATCHED SANCHO-RUBIO (The Performance Engine) ---
def batch_sancho_rubio(z_batch, H_00, H_01, max_iter=200, tol=1e-14):
    """
    Computes the surface Green's function for all energies simultaneously.
    z_batch shape: (N_E, M, M)
    """
    N_E = z_batch.shape[0]
    M = H_00.shape[0]
    
    eps_s = np.broadcast_to(H_00, (N_E, M, M)).copy()
    eps_b = np.broadcast_to(H_00, (N_E, M, M)).copy()
    alpha = np.broadcast_to(H_01, (N_E, M, M)).copy()
    beta = np.broadcast_to(H_01.conj().T, (N_E, M, M)).copy()
    
    for _ in range(max_iter):
        g = np.linalg.inv(z_batch - eps_b)
        
        alpha_g = alpha @ g  # Batched 3D matrix multiplication
        beta_g = beta @ g
        
        eps_s += alpha_g @ beta
        eps_b += alpha_g @ beta + beta_g @ alpha
        
        alpha_new = alpha_g @ alpha
        beta_new = beta_g @ beta
        
        err = max(np.max(np.abs(alpha_new)), np.max(np.abs(beta_new)))
        alpha, beta = alpha_new, beta_new
        
        if err < tol:
            break
            
    return np.linalg.inv(z_batch - eps_s)

#%% --- TRANSPORT KERNEL ---
def batched_trace(Gamma1, GR12, Gamma2, GA21):
    """Computes Trace(Gamma1 @ GR12 @ Gamma2 @ GA21) for the whole energy sweep."""
    # (N_E, M, M) @ (N_E, M, M) natively performs batched matmul
    T_matrix = Gamma1 @ GR12 @ Gamma2 @ GA21
    return np.trace(T_matrix, axis1=1, axis2=2).real

def compute_all_channels_batched(E_sweep, sys_params):
    Nx = sys_params['Nx']
    N_E = len(E_sweep)
    
    # --- 1. Prepare Batch z = E + i*eta ---
    # z_4 is for the 1D L/R leads (4x4), z_4N is for Central & Top/Bottom (4Nx x 4Nx)
    z_4 = (E_sweep + 1j * sys_params['eta'])[:, None, None] * np.eye(4)[None, :, :]
    z_4N = (E_sweep + 1j * sys_params['eta'])[:, None, None] * np.eye(4*Nx)[None, :, :]
    
    # --- 2. Compute Surface Green's Functions (Batched) ---
    g_L = batch_sancho_rubio(z_4, sys_params['H_L'], sys_params['V_L'])
    g_R = batch_sancho_rubio(z_4, sys_params['H_R'], sys_params['V_R'])
    g_T = batch_sancho_rubio(z_4N, sys_params['H_T_intra'], sys_params['H_T_inter'])
    g_B = batch_sancho_rubio(z_4N, sys_params['H_B_intra'], sys_params['H_B_inter'])
    
    # --- 3. Compute Self Energies ---
    # Left and Right leads only couple to the first and last site of the central region
    V_Lc = sys_params['V_Lc'][None, :, :]
    Sigma_L_4x4 = V_Lc @ g_L @ np.conj(V_Lc).transpose(0, 2, 1)
    
    V_Rc = sys_params['V_Rc'][None, :, :]
    Sigma_R_4x4 = V_Rc @ g_R @ np.conj(V_Rc).transpose(0, 2, 1)
    
    # Assemble Total Self Energy in the full 4Nx x 4Nx space
    Sigma_tot = np.zeros((N_E, 4*Nx, 4*Nx), dtype=complex)
    Sigma_tot[:, :4, :4] += Sigma_L_4x4
    Sigma_tot[:, -4:, -4:] += Sigma_R_4x4
    
    # Top and Bottom couple completely site-to-site
    V_Tc = sys_params['V_Tc'][None, :, :]
    Sigma_T = V_Tc @ g_T @ np.conj(V_Tc).transpose(0, 2, 1)
    
    V_Bc = sys_params['V_Bc'][None, :, :]
    Sigma_B = V_Bc @ g_B @ np.conj(V_Bc).transpose(0, 2, 1)
    
    Sigma_tot += Sigma_T + Sigma_B
    
    # --- 4. Central Green's Function ---
    H_C_batch = sys_params['H_C'][None, :, :]
    GR = np.linalg.inv(z_4N - H_C_batch - Sigma_tot)
    GA = np.conj(GR).transpose(0, 2, 1)
    
    # Broadening Matrices for Left and Right Leads (Extracted as 4x4 blocks)
    Gamma_L = 1j * (Sigma_L_4x4 - np.conj(Sigma_L_4x4).transpose(0, 2, 1))
    Gamma_R = 1j * (Sigma_R_4x4 - np.conj(Sigma_R_4x4).transpose(0, 2, 1))
    
    # Extract structural blocks
    G_LL = GR[:, :4, :4]
    G_RR = GR[:, -4:, -4:]
    G_LR = GR[:, :4, -4:]
    G_RL = GR[:, -4:, :4]
    
    G_LL_A = GA[:, :4, :4]
    G_RR_A = GA[:, -4:, -4:]
    G_LR_A = GA[:, :4, -4:]
    G_RL_A = GA[:, -4:, :4]
    
    # --- 5. Extract Channels ---
    e = slice(0, 2)
    h = slice(2, 4)
    
    def extract(out_lead):
        res = {}
        if out_lead == 'left':
            res['ee'] = batched_trace(Gamma_L[:, e, e], G_LR[:, e, e], Gamma_R[:, e, e], G_RL_A[:, e, e])
            res['hh'] = batched_trace(Gamma_L[:, h, h], G_LR[:, h, h], Gamma_R[:, h, h], G_RL_A[:, h, h])
            res['eh_cross'] = batched_trace(Gamma_L[:, e, e], G_LR[:, e, h], Gamma_R[:, h, h], G_RL_A[:, h, e])
            res['he_cross'] = batched_trace(Gamma_L[:, h, h], G_LR[:, h, e], Gamma_R[:, e, e], G_RL_A[:, e, h])
            res['eh_local'] = batched_trace(Gamma_L[:, e, e], G_LL[:, e, h], Gamma_L[:, h, h], G_LL_A[:, h, e])
            res['he_local'] = batched_trace(Gamma_L[:, h, h], G_LL[:, h, e], Gamma_L[:, e, e], G_LL_A[:, e, h])
        else: # right
            res['ee'] = batched_trace(Gamma_R[:, e, e], G_RL[:, e, e], Gamma_L[:, e, e], G_LR_A[:, e, e])
            res['hh'] = batched_trace(Gamma_R[:, h, h], G_RL[:, h, h], Gamma_L[:, h, h], G_LR_A[:, h, h])
            res['eh_cross'] = batched_trace(Gamma_R[:, e, e], G_RL[:, e, h], Gamma_L[:, h, h], G_LR_A[:, h, e])
            res['he_cross'] = batched_trace(Gamma_R[:, h, h], G_RL[:, h, e], Gamma_L[:, e, e], G_LR_A[:, e, h])
            res['eh_local'] = batched_trace(Gamma_R[:, e, e], G_RR[:, e, h], Gamma_R[:, h, h], G_RR_A[:, h, e])
            res['he_local'] = batched_trace(Gamma_R[:, h, h], G_RR[:, h, e], Gamma_R[:, e, e], G_RR_A[:, e, h])
        return res

    return {'left': extract('left'), 'right': extract('right')}

#%% --- YOUR POST-PROCESSING (Unchanged) ---
def f_electron(E, mu, kT):
    if kT == 0: return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    x = np.clip((E - mu) / kT, -700, 700)
    return 1.0 / (1.0 + np.exp(x))

def f_hole(E, mu, kT):
    if kT == 0: return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    x = np.clip((E + mu) / kT, -700, 700)
    return 1.0 / (1.0 + np.exp(x))

def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    mu_out, mu_in = bias[out_name], bias[in_name]
    
    f_out_e = f_electron(E_sweep, mu_out, kT)
    f_out_h = f_hole(E_sweep, mu_out, kT)
    f_in_e  = f_electron(E_sweep, mu_in, kT)
    f_in_h  = f_hole(E_sweep, mu_in, kT)

    integrand_EC  = ch['ee']*(f_out_e - f_in_e) - ch['hh']*(f_out_h - f_in_h)
    integrand_CAR = ch['eh_cross']*(f_out_e - f_in_h) - ch['he_cross']*(f_out_h - f_in_e)
    integrand_LAR = (ch['eh_local'] + ch['he_local'])*(f_out_e - f_out_h)

    I_EC  = np.trapezoid(integrand_EC, E_sweep)
    I_CAR = np.trapezoid(integrand_CAR, E_sweep)
    I_LAR = np.trapezoid(integrand_LAR, E_sweep)
    return {'EC': I_EC, 'CAR': I_CAR, 'LAR': I_LAR, 'total': I_EC + I_CAR + I_LAR}

def other_name(name): return 'right' if name=='left' else 'left'

def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    G = {}
    for i in leads:
        ch_i = channel_sweeps[i]
        in_name = other_name(i)
        for j in leads:
            bias_plus  = bias0.copy(); bias_plus[j]  += dV
            bias_minus = bias0.copy(); bias_minus[j] -= dV
            r_plus  = dc_current_channels(ch_i, E_sweep, bias_plus,  kT, i, in_name)
            r_minus = dc_current_channels(ch_i, E_sweep, bias_minus, kT, i, in_name)
            G[(i,j)] = (r_plus['total'] - r_minus['total']) / (2*dV)
    return G

#%% --- SYSTEM SETUP ---
Nx = 5            # Number of sites in central region
delta = 0.1       # SC Gap amplitude
phase_diff = np.pi # Phase difference between top and bottom SC leads

t_n, t_s = 1.0, 1.0
tc = 0.6          # Lead coupling
mu_n, mu_s = 0.3, 0.3
eta = 1e-5
kT = 1e-3 * delta

# On-site energies relative to the band bottom
# 1D leads: 2t - mu
# 2D leads: 4t - mu (hopping in x and y)
eps_1D = 2 * t_n - mu_n
eps_2D = 4 * t_s - mu_s

sys_params = {
    'Nx': Nx, 'eta': eta,
    
    # Central Region (1D Normal)
    'H_C': build_Central(Nx, eps_1D, t_n),
    
    # Normal 1D Leads (L and R)
    'H_L': H_onsite(eps_1D, 0.0), 'V_L': V_hop(t_n),
    'H_R': H_onsite(eps_1D, 0.0), 'V_R': V_hop(t_n),
    'V_Lc': V_hop(tc), 'V_Rc': V_hop(tc),
    
    # 2D Superconducting Top Lead
    'H_T_intra': build_SC_ribbon(Nx, eps_2D, t_s, t_s, delta * np.exp(1j * phase_diff/2))[0],
    'H_T_inter': build_SC_ribbon(Nx, eps_2D, t_s, t_s, delta * np.exp(1j * phase_diff/2))[1],
    'V_Tc': np.kron(np.eye(Nx), V_hop(tc)), # Couples completely across Central
    
    # 2D Superconducting Bottom Lead
    'H_B_intra': build_SC_ribbon(Nx, eps_2D, t_s, t_s, delta * np.exp(-1j * phase_diff/2))[0],
    'H_B_inter': build_SC_ribbon(Nx, eps_2D, t_s, t_s, delta * np.exp(-1j * phase_diff/2))[1],
    'V_Bc': np.kron(np.eye(Nx), V_hop(tc)),
}

E_sweep = np.linspace(-4.05*delta, 4.05*delta, 5001)
V_range = np.linspace(-2.05*delta, 2.05*delta, 302)

#%% --- RUN COMPUTATIONS ---
print(f"Computing {len(E_sweep)} energy points for {Nx} sites (Batch processing)...")
channel_sweeps = compute_all_channels_batched(E_sweep, sys_params)
print("Finished compute.")

ch = channel_sweeps['left']
results = {'sym': {'EC': [], 'CAR': [], 'LAR': [], 'total': []},
           'anti': {'EC': [], 'CAR': [], 'LAR': [], 'total': []}}

for Vb in V_range:
    sym = {"left": +Vb , "right": +Vb }
    anti = {"left": -Vb , "right": +Vb}
    r_sym = dc_current_channels(ch, E_sweep, sym, kT, "right", "left")
    r_anti = dc_current_channels(ch, E_sweep, anti, kT, "right", "left")
    for key in ['EC', 'CAR', 'LAR', 'total']:
        results['sym'][key].append(r_sym[key])
        results['anti'][key].append(r_anti[key])

#%% --- CONDUCTANCE COMPUTATIONS (Symmetric & Antisymmetric) + PLOTTING---
V_bias_cond = V_range 
leads_names = ['left', 'right']
dV = 1e-5

G_channels = {
    'sym':  {'EC': [], 'CAR': [], 'LAR': [], 'total': []},
    'anti': {'EC': [], 'CAR': [], 'LAR': [], 'total': []}
}
G_matrix_elems = {
    'sym':  {'G_LL': [], 'G_LR': []},
    'anti': {'G_LL': [], 'G_LR': []}
}

for Vb in V_bias_cond:
    for scheme in ['sym', 'anti']:
        if scheme == 'sym':
            bias0   = {"left": +Vb , "right": +Vb }
            b_plus  = {"left": +(Vb + dV) , "right": +(Vb + dV) }
            b_minus = {"left": +(Vb - dV) , "right": +(Vb - dV) }
        else: 
            bias0   = {"left": +Vb , "right": -Vb }
            b_plus  = {"left": +(Vb + dV) , "right": -(Vb + dV) }
            b_minus = {"left": +(Vb - dV) , "right": -(Vb - dV) }
        
        G_mat = conductance_matrix(bias0, leads_names, channel_sweeps, E_sweep, kT, dV=1e-5)
        G_matrix_elems[scheme]['G_LL'].append(G_mat[('left', 'left')])
        G_matrix_elems[scheme]['G_LR'].append(G_mat[('left', 'right')])
        
        r_plus  = dc_current_channels(channel_sweeps['left'], E_sweep, b_plus,  kT, "left", "right")
        r_minus = dc_current_channels(channel_sweeps['left'], E_sweep, b_minus, kT, "left", "right")
        
        for key in ['EC', 'CAR', 'LAR', 'total']:
            dI_key = (r_plus[key] - r_minus[key]) / (2 * dV)
            G_channels[scheme][key].append(dI_key)

#%% --- 2x2 CONDUCTANCE PLOTS ---
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
schemes_info = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
                ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')]

for col, (scheme, title) in enumerate(schemes_info):
    ax_top = axes[0, col]
    ax_top.plot(V_bias_cond, G_channels[scheme]['EC'], linewidth=2, label=r'$G_{\rm EC}$')
    ax_top.plot(V_bias_cond, G_channels[scheme]['CAR'], linewidth=2, label=r'$G_{\rm CAR}$')
    ax_top.plot(V_bias_cond, G_channels[scheme]['LAR'], linewidth=2, label=r'$G_{\rm LAR}$')
    ax_top.plot(V_bias_cond, G_channels[scheme]['total'], linewidth=2, color='k', linestyle='--', label=r'$G_{\rm total}$', alpha=.5)
    ax_top.axvline(x=-delta, color='gray', linestyle=':')
    ax_top.axvline(x=delta, color='gray', linestyle=':')
    ax_top.set_title(title)
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc='upper right')
    if col == 0: ax_top.set_ylabel(r'$dI_{\rm left} / dV_{\rm bias}$ (a.u.)')

    ax_bot = axes[1, col]
    G_total_direct = np.asarray(G_channels[scheme]['total'])
    G_LL = np.asarray(G_matrix_elems[scheme]['G_LL'])
    G_LR = np.asarray(G_matrix_elems[scheme]['G_LR'])

    ax_bot.plot(V_bias_cond, G_total_direct, linewidth=2, color='k', linestyle='--', label=r'Direct $dI_L/dV_{\rm bias}$', alpha = .5)
    ax_bot.plot(V_bias_cond, G_LL, linewidth=2, color='tab:green', label=r'$G_{LL}$', alpha = .7)
    ax_bot.plot(V_bias_cond, G_LR, linewidth=2, color='tab:gray', label=r'$G_{LR}$', alpha = .7)
    
    if scheme == 'sym':
        ax_bot.plot(V_bias_cond, (G_LL + G_LR), linewidth=2, color='tab:orange', label=r'$(G_{LL} + G_{LR})$', alpha=.5)
    else:
        ax_bot.plot(V_bias_cond, (G_LL - G_LR), linewidth=2, color='tab:orange', label=r'$(G_{LL} - G_{LR})$', alpha=.5)
        
    ax_bot.axvline(x=-delta, color='gray', linestyle=':')
    ax_bot.axvline(x=delta, color='gray', linestyle=':')
    ax_bot.set_title(r'Comparison with total differential G')
    ax_bot.set_xlabel(r'$V_{\rm bias}$')
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc='upper right')
    if col == 0: ax_bot.set_ylabel(r'Differential conductance (e^2 / h)')

plt.tight_layout()
plt.show()
# %%
