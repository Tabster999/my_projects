#%% --- IMPORTS ---
import numpy as np 
import matplotlib.pyplot as plt 
import os, sys
from scipy.linalg import norm
from numpy.linalg import inv
from scipy.integrate import simpson
#%% --- HELPER FUNCTIONS ---

def self_energy(g_surf, V_coupling):
    return V_coupling @ g_surf @ V_coupling.conj().T

def gamma_from_self_energy(Sigma):
    return -2.0 * np.imag(Sigma)

def make_block_diagonal(block_4x4, ny):
    """ny copies of a 4×4 block on the diagonal."""
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)

def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    x = np.clip((E - mu) / kT, -700, 700)
    return 1.0 / (1.0 + np.exp(x))

def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    x = np.clip((E + mu) / kT, -700, 700)
    return 1.0 / (1.0 + np.exp(x))

def kron(a, b):
    return np.kron(a, b)

#%% --- SURFACE GREEN'S FUNCTION AND TRANSMISSION COEFFICIENTS ---
def surface_gf(z_batch, H_onsite, V_hop, max_iter=400, tol=1e-14):

    N_E = z_batch.shape[0]
    M = H_onsite.shape[0]

    eps_s = np.broadcast_to(H_onsite, (N_E, M, M)).copy()
    eps_b = np.broadcast_to(H_onsite, (N_E, M, M)).copy()
    alpha = np.broadcast_to(V_hop, (N_E, M, M)).copy()
    beta = np.broadcast_to(V_hop.conj().T, (N_E, M, M)).copy()
    
    for _ in range(max_iter):
        g = inv(z_batch - eps_b)
        alpha_g = alpha @ g
        beta_g = beta @ g
        
        eps_s += alpha_g @ beta
        eps_b += alpha_g @ beta + beta_g @ alpha
        
        alpha_new = alpha_g @ alpha
        beta_new = beta_g @ beta
        err = max(np.max(np.abs(alpha_new)), np.max(np.abs(beta_new)))            
        if err < tol:
            break
        alpha, beta = alpha_new, beta_new

    return inv(z_batch - eps_s)

#%% --- BUILDING BLOCK MATRICES --- 

def onsite_block(t, mu, delta, phi):
    '''
    Onsite 4x4 Hamiltonian of the superconducting region
    '''
    ons = np.zeros((4, 4), dtype=np.complex128)
    pairing = delta * np.exp(1j * phi)

    ons[0,0] = 2*t - mu;    ons[1,1] = 2*t - mu
    ons[2,2] = -2*t + mu;   ons[3,3] = -2*t + mu

    ons[0,2] = pairing;    ons[1,3] = pairing
    ons[2,0] = pairing.conj();   ons[3,1] = pairing.conj()
    
    return ons

def Vx(t):
    '''
    Hopping 4x4 Hamiltonian in x-direction
    '''
    hop = np.zeros((4, 4), dtype=np.complex128)
    hop[0,0] = -t; hop[1,1] = -t
    hop[2,2] = t;  hop[3,3] = t
    return hop

def Vy(t):
    '''
    Hopping 4x4 Hamiltonian in y-direction
    '''
    hop = np.zeros((4, 4), dtype=np.complex128)
    hop[0,0] = -t; hop[1,1] = -t
    hop[2,2] = t;  hop[3,3] = t
    return hop

def build_sc_lead(nx, t, mu_s, delta, phi):
    """
    Build a superconducting lead with given parameters.
    Returns a Lead object.
    """
    dim_slice = 4 * nx 
    H_intra = np.zeros((dim_slice, dim_slice), dtype=np.complex128)
    H_inter = np.zeros((dim_slice, dim_slice), dtype=np.complex128)

    V_x = Vx(t)
    V_y = Vy(t)
    
    for i in range(nx):
        H_intra[i*4:(i+1)*4, i*4:(i+1)*4] = onsite_block(t, mu_s, delta, phi)
        H_inter[i*4:(i+1)*4, i*4:(i+1)*4] = V_y

    for i in range(nx - 1):
        H_intra[i*4:(i+1)*4, (i+1)*4:(i+2)*4] = V_x
        H_intra[(i+1)*4:(i+2)*4, i*4:(i+1)*4] = V_x.conj().T

    return H_inter, H_intra

def build_central(t, mu, nx):

    H_c = np.zeros((4*nx, 4*nx), dtype=np.complex128)
    H_on = onsite_block(t, mu, 0.0, 0.0)

    V_x = Vx(t)

    for i in range(nx):
        H_c[i*4:(i+1)*4, i*4:(i+1)*4] = H_on

    for i in range(nx - 1):
        H_c[i*4:(i+1)*4, (i+1)*4:(i+2)*4] = V_x
        H_c[(i+1)*4:(i+2)*4, i*4:(i+1)*4] = V_x.conj().T

    return H_c
#%% --- TRANSMISSION COEFFICIENTS AND COMPUTATIONS FUNCTIONS ---

def get_transmission_coefficients(Gamma1, GR, Gamma2, GA):

    T_matrix = Gamma1 @ GR @ Gamma2 @ GA
    return np.trace(T_matrix, axis1=1, axis2=2).real

def compute_all_channels(E_sweep, sys):

    z1 = (E_sweep + 1j * sys['eta'])[:, None, None] * np.eye(4, dtype=np.complex128)[None, :, :]
    zN = (E_sweep + 1j * sys['eta'])[:, None, None] * np.eye(4 * sys['nx'], dtype=np.complex128)[None, :, :]

    N_E = len(E_sweep)
    nx  = sys['nx']

    # 1. Surface green's functions for the leads
    g_L = surface_gf(z1, sys['H_L'], sys['V_L'], E_sweep)
    g_R = surface_gf(z1, sys['H_R'], sys['V_R'], E_sweep)
    g_T = surface_gf(zN, sys['H_T_intra'], sys['H_T_inter'], E_sweep)
    g_B = surface_gf(zN, sys['H_B_intra'], sys['H_B_inter'], E_sweep)

    # 2. Self-energies for the leads
    V_Lc = sys['V_Lc'][None, :, :]
    V_Rc = sys['V_Rc'][None, :, :]
    V_Tc = sys['V_Tc'][None, :, :]
    V_Bc = sys['V_Bc'][None, :, :]

    Sigma_l = V_Lc @ g_L @ V_Lc.conj().transpose(0, 2, 1)
    Sigma_r = V_Rc @ g_R @ V_Rc.conj().transpose(0, 2, 1)
    Sigma_t = V_Tc @ g_T @ V_Tc.conj().transpose(0, 2, 1)
    Sigma_b = V_Bc @ g_B @ V_Bc.conj().transpose(0, 2, 1)

    Sigma_tot = np.zeros((N_E, 4*nx, 4*nx), dtype=np.complex128)
    Sigma_tot[:, :4, :4]    = Sigma_l
    Sigma_tot[:, -4:, -4:]  = Sigma_r
    Sigma_tot              += Sigma_t + Sigma_b

    # 3. Compute central regions Green's function
    H_C_batch   =   sys['H_C'][None, :, :]
    GR          =   inv(zN - H_C_batch - Sigma_tot)
    GA          =   GR.conj().transpose(0, 2, 1)

    # 4. Compute Gamma matrices
    Gamma_L = -2.0 * np.imag(Sigma_l)
    Gamma_R = -2.0 * np.imag(Sigma_r)

    # 5. Extract Green's function for the left and right interfaces
    G_LR = GR[:, :4, -4:]   ;   G_LR_A = GA[:, -4:, :4]
    G_RL = GR[:, -4:, :4]   ;   G_RL_A = GA[:, :4, -4:]
    G_LL = GR[:, :4, :4]    ;   G_LL_A = GA[:, :4, :4]
    G_RR = GR[:, -4:, -4:]  ;   G_RR_A = GA[:, -4:, -4:]
    G_RL = GR[:, -4:, :4]   ;   G_RL_A = GA[:, :4, -4:]
    
    e = slice(0,2)
    h = slice(2,4)

    def get_transmission(out_lead_name):
        res = {}
        if out_lead_name == 'left':
            res['ee'] = get_transmission_coefficients(Gamma_L[:, e, e], G_LR[:, e, e], Gamma_R[:, e, e], G_LR_A[:, e, e])
            res['hh'] = get_transmission_coefficients(Gamma_L[:, h, h], G_LR[:, h, h], Gamma_R[:, h, h], G_LR_A[:, h, h])
            res['eh_cross'] = get_transmission_coefficients(Gamma_L[:, e, e], G_LR[:, e, h], Gamma_R[:, h, h], G_LR_A[:, h, e])
            res['he_cross'] = get_transmission_coefficients(Gamma_L[:, h, h], G_LR[:, h, e], Gamma_R[:, e, e], G_LR_A[:, e, h])
            res['eh_local'] = get_transmission_coefficients(Gamma_L[:, e, e], G_LL[:, e, h], Gamma_L[:, h, h], G_LL_A[:, h, e])
            res['he_local'] = get_transmission_coefficients(Gamma_L[:, h, h], G_LL[:, h, e], Gamma_L[:, e, e], G_LL_A[:, e, h])
        else:
            res['ee'] = get_transmission_coefficients(Gamma_R[:, e, e], G_RL[:, e, e], Gamma_L[:, e, e], G_RL_A[:, e, e])
            res['hh'] = get_transmission_coefficients(Gamma_R[:, h, h], G_RL[:, h, h], Gamma_L[:, h, h], G_RL_A[:, h, h])
            res['eh_cross'] = get_transmission_coefficients(Gamma_R[:, e, e], G_RL[:, e, h], Gamma_L[:, h, h], G_RL_A[:, h, e])
            res['he_cross'] = get_transmission_coefficients(Gamma_R[:, h, h], G_RL[:, h, e], Gamma_L[:, e, e], G_RL_A[:, e, h])
            res['eh_local'] = get_transmission_coefficients(Gamma_R[:, e, e], G_RR[:, e, h], Gamma_R[:, h, h], G_RR_A[:, h, e])
            res['he_local'] = get_transmission_coefficients(Gamma_R[:, h, h], G_RR[:, h, e], Gamma_R[:, e, e], G_RR_A[:, e, h])
        return res

    return {'left': get_transmission('left'), 'right': get_transmission('right')}


def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    mu_out, mu_in = bias[out_name], bias[in_name]
    
    f_out_e = f_electron(E_sweep, mu_out, kT)
    f_out_h = f_hole(E_sweep, mu_out, kT)
    f_in_e  = f_electron(E_sweep, mu_in, kT)
    f_in_h  = f_hole(E_sweep, mu_in, kT)

    integrand_EC  = ch['ee']*(f_out_e - f_in_e) - ch['hh']*(f_out_h - f_in_h)
    integrand_CAR = ch['eh_cross']*(f_out_e - f_in_h) - ch['he_cross']*(f_out_h - f_in_e)
    integrand_LAR = (ch['eh_local'] + ch['he_local'])*(f_out_e - f_out_h)

    I_EC  = simpson(integrand_EC, E_sweep)
    I_CAR = simpson(integrand_CAR, E_sweep)
    I_LAR = simpson(integrand_LAR, E_sweep)
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

# %% --- PARAMETERS ---

t_n = 1; t_s = 1; tc = 1.0
mu_n = 1.5; mu_s = 1.0
phi = np.pi / 2
delta = 0.1
nx = 10
eta = 1e-4
kT = 5e-2*delta

sys_params = {
    'nx': nx, 
    'eta': eta,
    'H_C': build_central(t_n, mu_n, nx),
    'H_L': onsite_block(t_n, mu_n, 0.0, 0.0), 'V_L': Vx(t_n),
    'H_R': onsite_block(t_n, mu_n, 0.0, 0.0), 'V_R': Vx(t_n),
    'V_Lc': Vx(tc), 'V_Rc': Vx(tc),

    'H_T_intra': build_sc_lead(nx, t_s, mu_s, delta, +phi/2)[1],
    'H_T_inter': build_sc_lead(nx, t_s, mu_s, delta, +phi/2)[0],
    'V_Tc': np.kron(np.eye(nx), Vy(tc)),
    
    'H_B_intra': build_sc_lead(nx, t_s, mu_s, delta, -phi/2)[1],
    'H_B_inter': build_sc_lead(nx, t_s, mu_s, delta, -phi/2)[0],
    'V_Bc': np.kron(np.eye(nx), Vy(tc)),
}

V_range = np.linspace(-2.05*delta, 2.05*delta, 102)
E_sweep = np.linspace(-4.05*delta, 4.05*delta, 1001)

#%% --- RUN COMPUTATIONS ---
print(f"Computing {len(E_sweep)} energy points for {nx} sites...")
channel_sweeps = compute_all_channels(E_sweep, sys_params)
print("Finished compute.")

ch = channel_sweeps['left']
results = {'sym': {'EC': [], 'CAR': [], 'LAR': [], 'total': []},
           'anti': {'EC': [], 'CAR': [], 'LAR': [], 'total': []}}

for Vb in V_range:
    sym = {"left": +Vb , "right": +Vb }
    anti = {"left": +Vb , "right": -Vb}
    r_sym = dc_current_channels(ch, E_sweep, sym, kT, "left", "right")
    r_anti = dc_current_channels(ch, E_sweep, anti, kT, "left", "right")
    for key in ['EC', 'CAR', 'LAR', 'total']:
        results['sym'][key].append(r_sym[key])
        results['anti'][key].append(r_anti[key])

#%% --- CONDUCTANCE COMPUTATIONS (Symmetric & Antisymmetric) ---
V_bias_cond = V_range 
leads_names = ['left', 'right']
dV = 1e-6

G_channels = {
    'sym':  {'EC': [], 'CAR': [], 'LAR': [], 'total': []},
    'anti': {'EC': [], 'CAR': [], 'LAR': [], 'total': []}
}
G_matrix_elems = {
    'sym':  {'G_LL': [], 'G_LR': [], 'G_RR': [], 'G_RL': []},
    'anti': {'G_LL': [], 'G_LR': [], 'G_RR': [], 'G_RL': []}
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
        G_matrix_elems[scheme]['G_RR'].append(G_mat[('right', 'right')])
        G_matrix_elems[scheme]['G_RL'].append(G_mat[('right', 'left')])
        
        r_plus  = dc_current_channels(channel_sweeps['left'], E_sweep, b_plus,  kT, "left", "right")
        r_minus = dc_current_channels(channel_sweeps['left'], E_sweep, b_minus, kT, "left", "right")
        
        for key in ['EC', 'CAR', 'LAR', 'total']:
            dI_key = (r_plus[key] - r_minus[key]) / (2 * dV)
            G_channels[scheme][key].append(dI_key)

#%% --- CURRENT PLOTS ---
fig_curr, axes_curr = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
schemes_info = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
                ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')]

for col, (scheme, title) in enumerate(schemes_info):
    ax = axes_curr[col]
    ax.plot(V_range, results[scheme]['EC'], linewidth=2, label=r'$I_{\rm EC}$')
    ax.plot(V_range, results[scheme]['CAR'], linewidth=2, label=r'$I_{\rm CAR}$')
    ax.plot(V_range, results[scheme]['LAR'], linewidth=2, label=r'$I_{\rm LAR}$')
    ax.plot(V_range, results[scheme]['total'], linewidth=2, color='k', linestyle='--', label=r'$I_{\rm total}$', alpha=.5)
    ax.axvline(x=-delta, color='gray', linestyle=':')
    ax.axvline(x=delta, color='gray', linestyle=':')
    ax.set_title(f"Currents - {title}")
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    if col == 0: ax.set_ylabel(r'Current ($e / h$ units)')

plt.tight_layout()
plt.show()

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
    if col == 0: ax_bot.set_ylabel(r'Differential conductance ($e^2 / h$)')

plt.tight_layout()
plt.show()

#%% --- PHASE-DEPENDENT CONDUCTANCE AT FIXED ENERGY AND BIAS ---
phi_values = np.linspace(-np.pi, np.pi, 81)
E_fixed = np.array([0.05]) 

T_ee = []
T_car = [] 
T_lar = [] 

for phi_val in phi_values:
  sys_phi = sys_params.copy()
  H_T_intra, H_T_inter = build_sc_lead(nx, t_s, mu_s, delta, +phi_val)
  H_B_intra, H_B_inter = build_sc_lead(nx, t_s, mu_s, delta, 0)
  sys_phi['H_T_intra'] = H_T_intra
  sys_phi['H_T_inter'] = H_T_inter
  sys_phi['H_B_intra'] = H_B_intra
  sys_phi['H_B_inter'] = H_B_inter

  channel_sweeps_phi = compute_all_channels(E_fixed, sys_phi)

  ch_left = channel_sweeps_phi['left']
  T_ee.append(ch_left['ee'][0])
  T_car.append(ch_left['eh_cross'][0])
  T_lar.append(ch_left['eh_local'][0])

fig_phi, ax_phi = plt.subplots(figsize=(9, 5))
ax_phi.plot(phi_values, T_ee, linewidth=2, label=r'EC ($e \rightarrow e$)')
ax_phi.plot(phi_values, T_car, linewidth=2, label=r'CAR ($e \rightarrow h$ cross)')
ax_phi.plot(phi_values, T_lar, linewidth=2, label=r'LAR ($e \rightarrow h$ local)')

ax_phi.set_title(
    r'Transmission Channels for fixed Energy $E = %.2f$' % E_fixed[0]
)
ax_phi.set_xlabel(r'Phase difference $\phi$')
ax_phi.set_ylabel(r'Transmission Probability ($e^2/h$ units)')
ax_phi.grid(alpha=0.5)
ax_phi.legend(loc='upper right')
#ax_phi.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
# ax_phi.set_xticklabels([r'$-\pi / 2$', r'$0$', r'$\pi / 2$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
ax_phi.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax_phi.set_xticklabels([r'$-\pi$', r'$-\pi / 2$', r'$0$', r'$\pi / 2$', r'$\pi$'])
plt.tight_layout()
plt.show()
#%% --- SYMMETRIC-BIAS CONDUCTANCE: SYMMETRIZATION TEST ---
'''
The symmetrization is performed as:
    G^S(V) = [G(+V) + G(-V)] / 2

    G^A(V) = [G(+V) - G(-V)] / 2

The paper's relation to test is:
    G_LL^A(V) = -G_LR^A(V)
'''

# Get the symmetric-bias conductance matrix elements
G_LL = np.asarray(G_matrix_elems['sym']['G_LL'])
G_LR = np.asarray(G_matrix_elems['sym']['G_LR'])

# G^S(V) = [G(+V) + G(-V)] / 2
G_LL_S = 0.5 * (G_LL + G_LL[::-1])
G_LR_S = 0.5 * (G_LR + G_LR[::-1])

# G^A(V) = [G(+V) - G(-V)] / 2
G_LL_A = 0.5 * (G_LL - G_LL[::-1])
G_LR_A = 0.5 * (G_LR - G_LR[::-1])


# LEFT: EVEN / SYMMETRIC PARTS
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(V_bias_cond, G_LL_S, linewidth=2, label=r'$G_{LL}^{S}$')
ax.plot(V_bias_cond, G_LR_S, linewidth=2, label=r'$G_{LR}^{S}$')
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.axvline(-delta, color='gray', linestyle=':')
ax.axvline(delta, color='gray', linestyle=':')
ax.set_title('Symmetric-bias: even parts')
ax.set_xlabel(r'$V_{\rm bias}$')
ax.set_ylabel(r'Conductance ($e^2/h$)')
ax.grid(alpha=0.3)
ax.legend(loc='upper right')

# RIGHT: ODD / ANTISYMMETRIC PARTS
ax = axes[1]
ax.plot(V_bias_cond, G_LL_A, linewidth=2, label=r'$G_{LL}^{A}$')
ax.plot(V_bias_cond, G_LR_A, linewidth=2, label=r'$G_{LR}^{A}$')
ax.plot(V_bias_cond, G_LL_A + G_LR_A, linewidth=2, linestyle='-', label=r'$G_{LL}^{A} + G_{LR}^{A}$')
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.axvline(-delta, color='gray', linestyle=':')
ax.axvline(delta, color='gray', linestyle=':')
ax.set_title(r'Symmetric-bias: $G_{LL}^{A}-G_{LR}^{A}$')
ax.set_xlabel(r'$V_{\rm bias}$')
ax.set_ylabel(r'Conductance ($e^2/h$)')
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
# %%
