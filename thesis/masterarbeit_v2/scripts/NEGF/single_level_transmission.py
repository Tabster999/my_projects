#%% imports 
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from IPython.display import display, Math 
#%% --- PHYSICAL CONSTANTS ---
e = const.e
h = const.h
G0 = 2 * e**2 / h 

#%% --- HELPER FUNCTIONS ---

def get_surface_green_function_1d(E, eps, t, eta=1e-4, max_iter=400, tol=1e-14):
    """Sancho-Rubio decimation for a 1D semi-infinite lead surface Green's function."""
    z = E + 1j * eta
    a = b = t
    es = eb = eps
    
    for _ in range(max_iter):
        g_eff = 1 / (z - eb)
        es += a * g_eff * b
        eb += a * g_eff * b + b * g_eff * a
        a *= g_eff * a
        b *= g_eff * b
        if abs(a) < tol and abs(b) < tol:
            break   
            
    return 1 / (z - es)

def get_self_energy(g_surf, t_coupling):
    return (t_coupling**2) * g_surf

def get_gamma(sigma):
    return -np.imag(sigma)

def get_retarded_green_function(E, epsilon_atom, sigma_l, sigma_r, eta=1e-6):
    return 1.0 / (E + 1j * eta - epsilon_atom - sigma_l - sigma_r)

def get_transmission(GR, gamma_l, gamma_r):
    return 4 * gamma_l * gamma_r * (np.abs(GR)**2)

#%% --- SYSTEM PARAMETERS ---
epsilon_atom = 0.0     # Fixed atom energy
t_lead       = 1.0     # Lead hopping 
eps_lead     = 0.0     # Lead band center
eta          = 1e-4    
mu_bias = 0.0
symmetric = True 

if symmetric:
    mu_L, mu_R = mu_bias / 2 , mu_bias / 2  
    display(Math(r"\mu_{L} = \mu_R " + f"= {mu_L}"))
else:
    mu_L, mu_R = mu_bias / 2 , -mu_bias / 2 
    display(Math(r"\mu_{L} =-\mu_R " + f"= {mu_L}"))
#%% --- CALCULATION: SWEEP ELECTRON ENERGY E ---

E_sweep = np.linspace(-3*t_lead, 3*t_lead, 301)
t_couplings = [0.4, 1.0]

plt.figure(figsize=(10, 6))

for tc in t_couplings:
    transmission_exact = []
    g0 = get_surface_green_function_1d(0.0, eps_lead, -t_lead, eta=eta)
    sigma0 = get_self_energy(g0, -tc)

    for E in E_sweep:

        g_surf_L = get_surface_green_function_1d(E - mu_L, eps_lead, -t_lead, eta=eta)
        g_surf_R = get_surface_green_function_1d(E - mu_R, eps_lead, -t_lead, eta=eta) 
        
        sigma_L = get_self_energy(g_surf_L, -tc)
        sigma_R = get_self_energy(g_surf_R, -tc)
        
        gamma_L = get_gamma(sigma_L)
        gamma_R = get_gamma(sigma_R)
        
        GR = get_retarded_green_function(E, epsilon_atom, sigma_L, sigma_R, eta=eta)
        
        T_E = get_transmission(GR, gamma_L, gamma_R)
        transmission_exact.append(T_E)

    plt.plot(E_sweep, transmission_exact, linewidth=2, label=f'$t_c = {tc}$')

plt.axvline(x=-2*t_lead, color='gray', linestyle='--', alpha=0.7, label=r'Lead Band Edges ($E = \pm 2t$)')
plt.axvline(x=2*t_lead, color='gray', linestyle='--', alpha=0.7)
plt.xlabel(r'Energy $E$')
plt.ylabel(r'Transmission $T(E)$')
plt.title(r'Transmission for a Single-Level System')
plt.ylim(-0.05, 1.15)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%