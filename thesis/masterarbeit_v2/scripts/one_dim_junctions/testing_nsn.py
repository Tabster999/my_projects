#%% --- IMPORTS ---
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import norm

#%% --- HELPER FUNCTIONS ---
def tau_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def tau_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def site_slice(site_index, dim_per_site):
    start = site_index * dim_per_site
    return slice(start, start + dim_per_site)

def build_lattice_hamiltonian(onsite_blocks, bonds):
    dim = onsite_blocks[0].shape[0]
    n_sites = len(onsite_blocks)
    H = np.zeros((n_sites * dim, n_sites * dim), dtype=complex)
    for i, block in enumerate(onsite_blocks):
        H[site_slice(i, dim), site_slice(i, dim)] = block
    for site_a, site_b, hop in bonds:
        H[site_slice(site_a, dim), site_slice(site_b, dim)] = hop
        H[site_slice(site_b, dim), site_slice(site_a, dim)] = hop.conj().T
    return H

def embed_lead_coupling(local_couplings, n_sites, dim_per_site):
    dim_lead = next(iter(local_couplings.values())).shape[1]
    V = np.zeros((n_sites * dim_per_site, dim_lead), dtype=complex)
    for site_index, block in local_couplings.items():
        V[site_slice(site_index, dim_per_site), :] = block
    return V

def gamma_from_sigma(sigma_r):
    return -2.0 * np.imag(sigma_r)


#%% --- CLASSES ---

class Lead:
    def __init__(self, name, H_onsite, V_hop, V_coupling, eta=1e-6):
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_hop, dtype=complex)
        self.V_coupling = np.asarray(V_coupling, dtype=complex)
        self.eta = eta

    def surface_gf(self, E, max_iter=400, tol=1e-14):
        dim = self.H_onsite.shape[0]
        z = (E + 1j * self.eta) * np.eye(dim, dtype=complex)
        alpha = self.V_hop.copy()
        beta = self.V_hop.conj().T.copy()
        eps_s = self.H_onsite.copy()
        eps_b = self.H_onsite.copy()
        for _ in range(max_iter):
            g = inv(z - eps_b)
            alpha_g = alpha @ g
            beta_g = beta @ g
            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha
            alpha = alpha_g @ alpha
            beta = beta_g @ beta
            if norm(alpha, np.inf) < tol and norm(beta, np.inf) < tol:
                break
        return inv(z - eps_s)

    def self_energy(self, E):
        g_surf = self.surface_gf(E)
        return self.V_coupling @ g_surf @ self.V_coupling.conj().T


class CentralRegion:
    def __init__(self, H_C, leads):
        self.H_C = np.asarray(H_C, dtype=complex)
        self.leads = list(leads)
        self._by_name = {lead.name: lead for lead in self.leads}

    def solve_system(self, E, eta=1e-6):
        dim = self.H_C.shape[0]
        sigmas = {lead.name: lead.self_energy(E) for lead in self.leads}
        sigma_tot = sum(sigmas.values()) #type: ignore
        GR = inv((E + 1j * eta) * np.eye(dim, dtype=complex) - self.H_C - sigma_tot)
        GA = GR.conj().T
        gammas = {name: gamma_from_sigma(sigma) for name, sigma in sigmas.items()}
        return {"GR": GR, "GA": GA, "sigmas": sigmas, "gammas": gammas}


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

def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    """
    Integrates the channel-resolved spectral current over energy, split into:
       EC  -- normal electron/hole cotunneling between out_name and in_name
       CAR -- crossed Andreev reflection between out_name and in_name
       LAR -- local Andreev reflection at out_name itself
    """
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


#%% --- TRANSPORT FUNCTIONS ---
tau_idx = {"e": 0, "h": 1}

def transmission_channel(state, l1, tau1, l2, tau2, site_of):

    GR, GA, gammas = state['GR'], state['GA'], state['gammas']
    idx1 = 2 * site_of[l1] + tau_idx[tau1]
    idx2 = 2 * site_of[l2] + tau_idx[tau2]
    Gamma1 = gammas[l1][idx1, idx1].real
    Gamma2 = gammas[l2][idx2, idx2].real
    return Gamma1 * Gamma2 * (GR[idx1, idx2] * GA[idx2, idx1]).real

def compute_all_channels(E_sweep, central, eta, site_of):
    """Computes channel sweeps for BOTH leads simultaneously to save 50% runtime."""
    leads = ["left", "right"]
    results = {
        out_lead: {k: np.zeros(len(E_sweep)) for k in ["ee", "eh_cross", "he_cross", "hh", "eh_local", "he_local"]}
        for out_lead in leads
    }

    # Define channel mapping: (out_lead, tau_out, in_lead, tau_in)
    definitions = {
        "left": {
            "ee":       ("left", "e", "right", "e"),
            "eh_cross": ("left", "e", "right", "h"),
            "he_cross": ("left", "h", "right", "e"),
            "hh":       ("left", "h", "right", "h"),
            "eh_local": ("left", "e", "left", "h"),
            "he_local": ("left", "h", "left", "e"),
        },
        "right": {
            "ee":       ("right", "e", "left", "e"),
            "eh_cross": ("right", "e", "left", "h"),
            "he_cross": ("right", "h", "left", "e"),
            "hh":       ("right", "h", "left", "h"),
            "eh_local": ("right", "e", "right", "h"),
            "he_local": ("right", "h", "right", "e"),
        }
    }

    for idx, E in enumerate(E_sweep):
        state = central.solve_system(E, eta=eta)
        for out_name in leads:
            for ch_name, (l1, t1, l2, t2) in definitions[out_name].items():
                results[out_name][ch_name][idx] = transmission_channel(state, l1, t1, l2, t2, site_of)
                
    return results

def other_name(name):
    if name not in ['left', 'right']:
        raise ValueError("Invalid lead name. Must be 'left' or 'right'.")
    else:
        return 'right' if name=='left' else 'left'

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

#%% --- SYSTEM PARAMETERS ---
t_n, t_s = 1.0, 1.0
mu_n, mu_s = 0.3, 0.3
delta = 0.1
eta = 1e-5
tc = 0.6
kT = 1e-3 * delta  
n_sites = 5
dim_per_site = 2


H_N = (2 * t_n - mu_n) * tau_z()
V_N = -t_n * tau_z()
H_S = (2 * t_s - mu_s) * tau_z() + delta * tau_x()

onsite_blocks = [H_S] * n_sites
hop = -t_s * tau_z()
bonds = [(i, i + 1, hop) for i in range(n_sites - 1)]
H_C = build_lattice_hamiltonian(onsite_blocks, bonds)

V_left_full = embed_lead_coupling({0: -tc * tau_z()}, n_sites, dim_per_site)
V_right_full = embed_lead_coupling({n_sites - 1: -tc * tau_z()}, n_sites, dim_per_site)

lead_left = Lead("left", H_N, V_N, V_left_full, eta=eta)
lead_right = Lead("right", H_N, V_N, V_right_full, eta=eta)

central = CentralRegion(H_C, [lead_left, lead_right])
site_of = {"left": 0, "right": n_sites - 1}

E_sweep = np.linspace(-4.05*delta , 4.05*delta, 5001)
V_range = np.linspace(-2.05 * delta, 2.05 * delta, 302)

#%% --- COMPUTATIONS ---

channel_sweeps = compute_all_channels(E_sweep, central, eta=eta, site_of=site_of)
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


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, scheme, title in [(axes[0], 'sym', r'Symmetric bias  ($\mu_L=\mu_R=+V$)'),
                          (axes[1], 'anti', r'Antisymmetric bias  ($\mu_L=+V,\ \mu_R=-V$)')]:
    ax.plot(V_range, results[scheme]['EC'], linewidth=2, label='EC (normal)')
    ax.plot(V_range, results[scheme]['CAR'], linewidth=2, label='CAR (crossed Andreev)')
    ax.plot(V_range, results[scheme]['LAR'], linewidth=2, label='LAR (local Andreev)')
    ax.plot(V_range, results[scheme]['total'], linewidth=2, color='k', linestyle='--', label='Total')
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel(r'DC current $I_{\rm left}$ (a.u.)')
plt.show()

for Vb in [0.3, 1.0, 2.0]:
    sym = {"left": +Vb , "right": +Vb }
    anti = {"left": +Vb , "right": -Vb }
    r_sym = dc_current_channels(ch, E_sweep, sym, kT, "left", "right")
    r_anti = dc_current_channels(ch, E_sweep, anti, kT, "left", "right")
    print(f"V_bias={Vb}")
    print(f"  symmetric:     EC={r_sym['EC']:+.6f}  CAR={r_sym['CAR']:+.6f}  LAR={r_sym['LAR']:+.6f}  total={r_sym['total']:+.6f}")
    print(f"  antisymmetric: EC={r_anti['EC']:+.6f}  CAR={r_anti['CAR']:+.6f}  LAR={r_anti['LAR']:+.6f}  total={r_anti['total']:+.6f}")

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
        
        # 1. Partial derivative matrix elements (G_LL and G_LR) at bias0
        G_mat = conductance_matrix(bias0, leads_names, channel_sweeps, E_sweep, kT, dV=1e-5)
        G_matrix_elems[scheme]['G_LL'].append(G_mat[('left', 'left')])
        G_matrix_elems[scheme]['G_LR'].append(G_mat[('left', 'right')])
        
        # 2. Path differential conductances dI_channel / dV_bias
        r_plus  = dc_current_channels(channel_sweeps['left'], E_sweep, b_plus,  kT, "left", "right")
        r_minus = dc_current_channels(channel_sweeps['left'], E_sweep, b_minus, kT, "left", "right")
        
        for key in ['EC', 'CAR', 'LAR', 'total']:
            dI_key = (r_plus[key] - r_minus[key]) / (2 * dV)
            G_channels[scheme][key].append(dI_key)

#%% --- 2x2 CONDUCTANCE PLOTS ---
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

schemes_info = [
    ('sym',  r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
    ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')
]

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

    if col == 0:
        ax_top.set_ylabel(r'$dI_{\rm left} / dV_{\rm bias}$ (a.u.)')


    # Symmetric bias:
    #   dI_L/dV_bias = (G_LL + G_LR)/2

    # Antisymmetric bias:
    #   dI_L/dV_bias = (G_LL - G_LR)/2
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

    if col == 0:
        ax_bot.set_ylabel(r'Differential conductance (e^2 / h)')
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
