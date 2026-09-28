#%% --- IMPORTS ---
from dataclasses import dataclass
from re import A
from traceback import print_exception
import numpy as np
from numpy.linalg import inv
from scipy.integrate import simpson
from scipy.linalg import block_diag
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from IPython.display import display, Math, Pretty
import pandas as pd 
#%% --- PARAMS ---

@dataclass
class Params:
    nx: int = 10
    ny: int = 1
    t_n: float = 1.0;  mu_n: float = 0.0
    t_c: float = 1.0;  mu_c: float = 0.0
    t_s: float = 1.0;  mu_s: float = 0.0
    delta: float = 0.1
    phi: float = np.pi 
    tc_top: float = 0.6
    tc_bot: float = 0.6
    tc_barr: float = 0.5
    alpha: float = 0.0               # Rashba 
    beta: float = 0.0                # Dresselhaus
    Bz: float = 0.0                  # Zeeman out-of-plane
    Bxy: float = 0.0                 # Zeeman in-plane magnitude
    theta_z: float = 0.0             # Zeeman in-plane angle
    eta: float = 1e-5
    max_iter: int = 450
    tol: float = 1e-14
    kT: float = 1e-3 

#%% --- BUILDING BLOCKS (4x4) + HELPER FUNCTIONS ---

def onsite_block(t, mu, delta=0.0, phi=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0, alpha=0.0, beta=0.0, twod=False):
    ons = np.zeros((4, 4), dtype=np.complex128)
    onsite_val = ((4*t - mu) if twod else (2*t-mu)) + (alpha**2 + beta**2) / 4 
    pairing = delta * np.exp(1j * phi)
    Bxy_c = Bxy * np.exp(1j * theta_z)

    ons[0,0] =  onsite_val + Bz;  ons[1,1] =  onsite_val - Bz
    ons[2,2] = -onsite_val + Bz;  ons[3,3] = -onsite_val - Bz

    ons[0,1] = Bxy_c;  ons[1,0] = np.conj(Bxy_c)
    ons[2,3] = Bxy_c;  ons[3,2] = np.conj(Bxy_c)

    ons[0,2] = pairing;         ons[1,3] = pairing
    ons[2,0] = pairing.conj();  ons[3,1] = pairing.conj()
    return ons

def Vx(t, alpha=0.0, beta=0.0):
    s = 0.5 * (alpha - beta)

    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    hop[0,1] = -s; hop[1,0] = s; hop[2,3] = s; hop[3,2] = -s
    return hop

def Vy(t, alpha=0.0, beta=0.0):
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([[0,1j,0,0], [1j,0,0,0], [0,0,0,-1j], [0,0,-1j,0]], dtype=np.complex128) * (alpha + beta) / 2
    return hop + soc

def make_row_hamiltonian(n, onsite, hop_x):
    dim = 4 * n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        H[4*i:4*i+4, 4*i:4*i+4] = onsite
    for i in range(n - 1):
        H[4*i:4*i+4, 4*i+4:4*i+8] = hop_x
        H[4*i+4:4*i+8, 4*i:4*i+4] = hop_x.conj().T
    return H

def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E - mu)/kT, -1000, 1000)))

def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E + mu)/kT, -1000, 1000)))

def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    mu_out, mu_in = bias[out_name], bias[in_name]
    f_out_e, f_out_h = f_electron(E_sweep, mu_out, kT), f_hole(E_sweep, mu_out, kT)
    f_in_e,  f_in_h  = f_electron(E_sweep, mu_in, kT),  f_hole(E_sweep, mu_in, kT)

    I_EC  = np.trapezoid(ch['ee']*(f_out_e-f_in_e) - ch['hh']*(f_out_h-f_in_h), E_sweep)
    I_CAR = np.trapezoid(ch['eh_cross']*(f_out_e-f_in_h) - ch['he_cross']*(f_out_h-f_in_e), E_sweep) 
    I_LAR = np.trapezoid(ch['eh_local']*(f_out_e-f_out_h), E_sweep)
    return {'EC': I_EC, 'CAR': I_CAR, 'LAR': I_LAR, 'total': I_EC+I_CAR+I_LAR}

def other_name(name): return 'right' if name == 'left' else 'left'

def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    G = {}
    for i in leads:
        ch_i, in_name = channel_sweeps[i], other_name(i)
        for j in leads:
            bp, bm = bias0.copy(), bias0.copy()
            bp[j] += dV; bm[j] -= dV
            rp = dc_current_channels(ch_i, E_sweep, bp, kT, i, in_name)
            rm = dc_current_channels(ch_i, E_sweep, bm, kT, i, in_name)
            G[(i,j)] = (rp['total'] - rm['total']) / (2*dV)
    return G

def get_2d_hamiltonian(nx, ny, onsite, Vx, Vy, dof=4):

    dim = 4* nx * ny 
    H = np.zeros((dim, dim), dtype=np.complex128)
    def idx(ix, iy): return ix + nx * iy
    for iy in range(ny):
        for ix in range(nx):
            j = idx(ix, iy)
            H[dof*j:dof*(j+1), dof*j:dof*(j+1)] = onsite

    for iy in range(ny):
        for ix in range(nx -1):
            i, j = idx(ix, iy), idx(ix+1, iy)
            H[dof*i:dof*(i+1), dof*j:dof*(j+1)] = Vx
            H[dof*j:dof*(j+1), dof*i:dof*(i+1)] = Vx.conj().T

    for iy in range(ny - 1):
            for ix in range(nx):
                i, j = idx(ix, iy), idx(ix, iy+1)
                H[dof*i:dof*(i+1), dof*j:dof*(j+1)] = Vy
                H[dof*j:dof*(j+1), dof*i:dof*(i+1)] = Vy.conj().T
    return H 


def _style(ax, title):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.set_title(title)

def print_params(p):
    latex_str = rf"\begin{{array}}{{ll | ll | ll}}\hline\text{{Chemical Potentials}} & \text{{Value}} & \text{{Hoppings}} & \text{{Value}} & \text{{Other}} & \text{{Value}} \\\hline\mu_n & {p.mu_n} & t_n & {p.t_n} & n_x & {p.nx} \\\mu_c & {p.mu_c} & t_c & {p.t_c} & \Delta & {p.delta} \\\mu_s & {p.mu_s} & t_s & {p.t_s} & \phi & {p.phi:.3g} \\& & t_{{\text{{top}}}} & {p.tc_top} & \eta & {p.eta} \\& & t_{{\text{{bot}}}} & {p.tc_bot} & kT & {p.kT} \\& & t_{{\text{{barr}}}} & {p.tc_barr} & & \\\hline\end{{array}}"
    display(Math(latex_str))

def flat_site_idx(sites):
    '''
    Using the convention physical_site__idx = ix + nx * iy, this recovers the matrix elements corresponding to the given site
    '''
    return np.concatenate([np.arange(4*s, 4*(s+1)) for s in sites])
#%% --- CLASSES: LEAD, JUNCTION ---
class Lead:
    def __init__(self, name, H_onsite, V_hop, V_coupling, p: Params, br=False):
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_hop, dtype=complex)
        Vc = np.asarray(V_coupling, dtype=complex)
        self.V_coupling = Vc.conj().T if br else Vc        
        self.p = p

    def surface_gf(self, z_batch):
        M = self.H_onsite.shape[0]
        N_E = z_batch.shape[0]
        eps_s = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        eps_b = np.broadcast_to(self.H_onsite, (N_E, M, M)).copy()
        alpha = np.broadcast_to(self.V_hop, (N_E, M, M)).copy()
        beta  = np.broadcast_to(self.V_hop.conj().T, (N_E, M, M)).copy()
        for _ in range(self.p.max_iter):
            g = inv(z_batch - eps_b)
            alpha_g, beta_g = alpha @ g, beta @ g
            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha
            alpha, beta = alpha_g @ alpha, beta_g @ beta
            if (np.max(np.sum(np.abs(alpha), axis=2)) < self.p.tol and np.max(np.sum(np.abs(beta), axis=2)) < self.p.tol):
                break
        return inv(z_batch - eps_s)

    def self_energy(self, z_batch):
        g = self.surface_gf(z_batch)
        Vc = self.V_coupling[None, :, :]
        return Vc @ g @ Vc.conj().transpose(0, 2, 1)

class FourTerminalJunction:
    def __init__(self, p: Params):
        self.p = p
        self._build()

    def _build(self):
        p = self.p
        nx, ny = p.nx, p.ny 

        onsite_C = onsite_block(p.t_c, p.mu_c, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z, alpha=p.alpha, beta=p.beta, twod=True)
        self.H_C = get_2d_hamiltonian(nx, ny, onsite_C, Vx(p.t_c, p.alpha, p.beta), Vy(p.t_c, p.alpha, p.beta))

        onsite_N = onsite_block(p.t_n, p.mu_n, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z, alpha=p.alpha, beta=p.beta, twod=True)
        H_layer_N = make_row_hamiltonian(ny, onsite_N, Vy(p.t_n, p.alpha, p.beta))
        V_n = block_diag(*[Vx(p.t_n, p.alpha, p.beta)]* ny)
        V_coupling_LR = block_diag(*[Vx(p.tc_barr)] * ny)

        self.lead_L = Lead('L', H_layer_N, V_n, V_coupling_LR, p, br=False)
        self.lead_R = Lead('R', H_layer_N, V_n, V_coupling_LR, p, br=True)
        self.ribbon_top = self._make_ribbon(p.phi, p.tc_top)
        self.ribbon_bot = self._make_ribbon(0, p.tc_bot)
        left_sites  = [ix + nx*iy for iy in range(ny) for ix in [0]]
        right_sites = [ix + nx*iy for iy in range(ny) for ix in [nx-1]]
        self.idx_L = flat_site_idx(left_sites)
        self.idx_R = flat_site_idx(right_sites)

    def _make_ribbon(self, phi_lead, tc, is_bot=False):
        p = self.p
        onsite_SC = onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=phi_lead, twod=True)
        H_intra = make_row_hamiltonian(p.nx, onsite_SC, Vx(p.t_s))
        H_inter = block_diag(*([Vy(p.t_s)] * p.nx))
        V_coupling = block_diag(*([Vy(tc)] * p.nx))
        return Lead('ribbon', H_intra, H_inter, V_coupling, p, br=is_bot)

    def _z_batches(self, E_sweep):
        p = self.p
        dim = 4 * p.nx * p.ny
        z1 = (E_sweep + 1j*p.eta)[:,None,None] * np.eye(4*p.ny, dtype=complex)[None,:,:]
        zN = (E_sweep + 1j*p.eta)[:,None,None] * np.eye(dim, dtype=complex)[None,:,:]
        return z1, zN

    def channels(self, E_sweep, side_name= None, precompute_sigmas_n = None):
        p = self.p
        nx, ny = p.nx, p.ny
        dim = 4* nx * ny
        N_E = len(E_sweep)

        z_1D = (E_sweep + 1j*p.eta)[:, None, None]
        z_lead_N = z_1D * np.eye(4*ny, dtype=complex)[None, :, :]
        z_lead_SC = z_1D * np.eye(4*nx, dtype=complex)[None, :, :]
        z_center = z_1D * np.eye(dim, dtype=complex)[None, :, :]

        if precompute_sigmas_n is None:
            Sigma_l = self.lead_L.self_energy(z_lead_N)
            Sigma_r = self.lead_R.self_energy(z_lead_N)   
        else:
            Sigma_l, Sigma_r = precompute_sigmas_n

        Sigma_t = self.ribbon_top.self_energy(z_lead_SC)
        Sigma_b = self.ribbon_bot.self_energy(z_lead_SC)

        Sigma_tot = np.zeros((N_E, dim, dim), dtype=complex)
        Sigma_tot[:, :4*nx, :4*nx]   += Sigma_t                        
        Sigma_tot[:, -4*nx:, -4*nx:] += Sigma_b                       
        iL, iR = self.idx_L, self.idx_R
        Sigma_tot[:, iL[:,None], iL[None,:]] += Sigma_l               
        Sigma_tot[:, iR[:,None], iR[None,:]] += Sigma_r     

        GR = inv(z_center - self.H_C[None,:,:] - Sigma_tot)
        GA = GR.conj().transpose(0, 2, 1)
        Gamma_L = -1.0 * np.imag(Sigma_l)
        Gamma_R = -1.0 * np.imag(Sigma_r)

        def block(M, i, j): return M[:, i[:,None], j[None,:]]
        G_LR, G_LR_A = block(GR, iL, iR), block(GA, iL, iR)
        G_RL, G_RL_A = block(GR, iR, iL), block(GA, iR, iL)
        G_LL, G_LL_A = block(GR, iL, iL), block(GA, iL, iL)
        G_RR, G_RR_A = block(GR, iR, iR), block(GA, iR, iR)

        e_idx = np.where(np.tile([True,True,False,False], ny))[0]
        h_idx = np.where(np.tile([False,False,True,True], ny))[0]

        e = lambda M: M[:, e_idx[:, None], e_idx[None, :]]
        h = lambda M: M[:, h_idx[:, None], h_idx[None, :]]
        eh = lambda M: M[:, e_idx[:, None], h_idx[None, :]]
        he = lambda M: M[:, h_idx[:, None], e_idx[None, :]]

        def T(G1, B1, G2, B2):
            return np.einsum('nij,njk,nkl,nli->n', G1, B1, G2, B2).real

        def side(name):
            if name == 'left':
                GL, GLA, GaL, GaO = G_RL, G_RL_A, Gamma_L, Gamma_R
                GLL, GLLA, GaLL   = G_LL, G_LL_A, Gamma_L
            else:
                GL, GLA, GaL, GaO = G_LR, G_LR_A, Gamma_R, Gamma_L
                GLL, GLLA, GaLL   = G_RR, G_RR_A, Gamma_R
            return {
                "ee": T(e(GaL), e(GL), e(GaO), e(GLA)),
                "hh": T(h(GaL), h(GL), h(GaO), h(GLA)),
                "eh_cross": T(e(GaL), eh(GL), h(GaO), he(GLA)),
                "he_cross": T(h(GaL), he(GL), e(GaO), eh(GLA)),
                "eh_local": T(e(GaLL), eh(GLL), h(GaLL), he(GLLA)),
                "he_local": T(h(GaLL), he(GLL), e(GaLL), eh(GLLA)),
            }
        if side_name is None: 
            return {'left': side('left'),
                    'right': side('right')}
        if side_name == 'left':
            return side('left')
        if side_name == 'right':
            return side('right')
        raise ValueError(f"Invalid side_name: {side_name}. Must be 'left', 'right', or None.")

#%% --- BUILD JUNCTION & RUN CORE COMPUTATION ---
'''To resolve the energy integrals, the value of eta should be larger than the energy step size, as to smooth out numerical integration'''
p = Params(
        nx=11, ny=2, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0, 
        delta=0.35, phi = np.pi / 4, tc_top=1.00, tc_bot=1.00, tc_barr=1.00, 
        eta=2e-5, kT=1e-2)

V_range = np.linspace(-1.5*p.delta, 1.5*p.delta, 121)
E_sweep = np.linspace(-1.5*p.delta, 1.5*p.delta, 1001)

# V_range = np.linspace(-0.5, 0.5, 301)
# E_sweep = np.linspace(-0.5, 0.5, 301)

print(f"Computing {len(E_sweep)} energy points for {p.nx} sites...")
junction = FourTerminalJunction(p)
channel_sweeps = junction.channels(E_sweep)
print("Finished compute.")
print_params(p)
#%% --- CURRENTS (sym / anti bias) ---
'''ch_sweeps['left'] and ch_sweeps['right'] contain the transmission channels for the left and right leads, respectively ch_sweeps['left']['ee'] is for example the electron-to-electron transmission from left to right, evaluated at all energies in E_sweep'''
ch = channel_sweeps['left']
results = {'sym': {k: [] for k in ['EC','CAR','LAR','total']},
           'anti': {k: [] for k in ['EC','CAR','LAR','total']}}

for Vb in V_range:
    r_sym  = dc_current_channels(ch, E_sweep, {"left": Vb, "right": Vb}, p.kT, "left", "right")
    r_anti = dc_current_channels(ch, E_sweep, {"left": Vb, "right": -Vb}, p.kT, "left", "right")
    for key in ['EC','CAR','LAR','total']:
        results['sym'][key].append(r_sym[key])
        results['anti'][key].append(r_anti[key])    
        # Results are now stored in results['sym'] and results['anti'], each containing lists of currents for EC, CAR, LAR, and total. Accessible via results['sym']['EC'], results['anti']['CAR'], etc.

fig_curr, axes_curr = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
schemes_info = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
                ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')]
for col, (scheme, title) in enumerate(schemes_info):

    ax = axes_curr[col]
    for key, lbl in [('EC', r'$I_{\rm EC}$'), ('CAR', r'$I_{\rm CAR}$'), ('LAR', r'$I_{\rm LAR}$')]:
        ax.plot(V_range, results[scheme][key], linewidth=2, label=lbl)

    ax.plot(V_range, results[scheme]['total'], linewidth=2, color='k', linestyle='--', label=r'$I_{\rm total}$', alpha=.5)
    ax.axvline(-p.delta, color='gray', linestyle=':'); ax.axvline(p.delta, color='gray', linestyle=':')
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.5); ax.legend(loc='upper left'); _style(ax, f"Currents - {title}")
    if col == 0: ax.set_ylabel(r'Current ($e/h$ units)')

plt.tight_layout(); plt.show()
#%% --- 2x2 CONDUCTANCE PLOTS ---
leads_names = ['left', 'right']
dV = 1e-5

#Initialize dictionaries to store conductances
G_channels = {'sym': {k: [] for k in ['EC','CAR','LAR','total']},
              'anti': {k: [] for k in ['EC','CAR','LAR','total']}}

G_matrix_elems = {'sym': {k: [] for k in ['G_LL','G_LR','G_RR','G_RL']},
                  'anti': {k: [] for k in ['G_LL','G_LR','G_RR','G_RL']}}

for Vb in V_range:
    for scheme in ['sym', 'anti']:
        #Loop over bias voltages and compute conductances for both symmetric and antisymmetric bias schemes
        sign = 1 if scheme == 'sym' else -1
        bias0 = {"left": Vb, "right": sign*Vb}
        b_plus  = {"left": Vb+dV, "right": sign*(Vb+dV)}
        b_minus = {"left": Vb-dV, "right": sign*(Vb-dV)}

        G_mat = conductance_matrix(bias0, leads_names, channel_sweeps, E_sweep, p.kT, dV=1e-5)
        for key, tup in [('G_LL',('left','left')), ('G_LR',('left','right')),
                          ('G_RR',('right','right')), ('G_RL',('right','left'))]:
            G_matrix_elems[scheme][key].append(G_mat[tup])

        r_plus  = dc_current_channels(channel_sweeps['left'], E_sweep, b_plus,  p.kT, out_name="left", in_name="right")
        r_minus = dc_current_channels(channel_sweeps['left'], E_sweep, b_minus, p.kT, out_name="left", in_name="right")
        for key in ['EC','CAR','LAR','total']:
            G_channels[scheme][key].append((r_plus[key]-r_minus[key])/(2*dV))
# Conductances are now stored in G_channels and G_matrix_elems for both symmetric and antisymmetric bias schemes. Accessible via G_channels['sym']['EC'], G_matrix_elems['anti']['G_LL'], etc.
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for col, (scheme, title) in enumerate(schemes_info):
    ax_top = axes[0, col]

    for key, lbl in [('EC', r'$G_{\rm EC}$'), ('CAR', r'$G_{\rm CAR}$'), ('LAR', r'$G_{\rm LAR}$')]:
        ax_top.plot(V_range, G_channels[scheme][key], linewidth=2, label=lbl)
    ax_top.plot(V_range, G_channels[scheme]['total'], linewidth=2, color='k', linestyle='--', label=r'$G_{\rm total}$', alpha=.5)
    ax_top.axvline(-p.delta, color='gray', linestyle=':'); ax_top.axvline(p.delta, color='gray', linestyle=':')
    ax_top.grid(alpha=0.3); ax_top.legend(loc='upper right'); _style(ax_top, title)
    if col == 0: ax_top.set_ylabel(r'$dI_{\rm left}/dV_{\rm bias}$ (a.u.)')

    ax_bot = axes[1, col]
    G_LL = np.asarray(G_matrix_elems[scheme]['G_LL'])
    G_LR = np.asarray(G_matrix_elems[scheme]['G_LR'])
    ax_bot.plot(V_range, np.asarray(G_channels[scheme]['total']), linewidth=2, color='k', linestyle='--', label=r'Direct $dI_L/dV$', alpha=.5)
    ax_bot.plot(V_range, G_LL, linewidth=2, color='tab:green', label=r'$G_{LL}$', alpha=.7)
    ax_bot.plot(V_range, G_LR, linewidth=2, color='tab:gray', label=r'$G_{LR}$', alpha=.7)
    combo = (G_LL+G_LR) if scheme == 'sym' else (G_LL-G_LR)
    lbl = r'$(G_{LL}+G_{LR})$' if scheme == 'sym' else r'$(G_{LL}-G_{LR})$'
    ax_bot.plot(V_range, combo, linewidth=2, color='tab:orange', label=lbl, alpha=.5)
    ax_bot.axvline(-p.delta, color='gray', linestyle=':'); ax_bot.axvline(p.delta, color='gray', linestyle=':')
    ax_bot.set_xlabel(r'$V_{\rm bias}$')
    ax_bot.grid(alpha=0.5); ax_bot.legend(loc='upper right'); _style(ax_bot, 'Comparison with total differential G')
    if col == 0: ax_bot.set_ylabel(r'Differential conductance ($e^2/h$)')
plt.tight_layout(); plt.show()
print_params(p)
#%% --- PHASE-DEPENDENT TRANSMISSION AT FIXED ENERGY (COMPUTATION + PLOTS) ---
phi_values = np.linspace(0, 2*np.pi, 81)
E_fixed = np.array([0.1])
T_ee, T_car, T_lar = [], [], []

for phi_val in phi_values:  
    p_phi = Params(**{**p.__dict__, 'phi': phi_val})
    junction_phi = FourTerminalJunction(p_phi)
    ch_left = junction_phi.channels(E_fixed)['left']
    T_ee.append(ch_left['ee'][0] + ch_left['hh'][0])
    T_car.append(ch_left['eh_cross'][0] + ch_left['he_cross'][0])
    T_lar.append(ch_left['eh_local'][0] + ch_left['he_local'][0])

fig_phi, ax_phi = plt.subplots(figsize=(9, 5))
ax_phi.plot(phi_values , T_ee, linewidth=2, label=r'EC ($e\to e$)')
ax_phi.plot(phi_values, T_car, linewidth=2, label=r'CAR ($e\to h$ cross)')
ax_phi.plot(phi_values, T_lar, linewidth=2, label=r'LAR ($e\to h$ local)')
ax_phi.set_xlabel(r'Phase difference $\phi$'); ax_phi.set_ylabel(r'Transmission ($e^2/h$)')
ax_phi.set_xticks([0,np.pi/2,np.pi, np.pi, 3 * np.pi / 2, 2* np.pi])
ax_phi.grid(alpha=0.5); ax_phi.legend(loc='upper right'); _style(ax_phi, r'Transmission channels at fixed $E=%.2f$' % E_fixed[0])
plt.tight_layout(); plt.show()
#%% --- SYMMETRIZATION TEST ---
G_LL = np.asarray(G_matrix_elems['sym']['G_LL'])
G_LR = np.asarray(G_matrix_elems['sym']['G_LR'])

G_LL_S = 0.5*(G_LL + G_LL[::-1]); G_LR_S = 0.5*(G_LR + G_LR[::-1]) # Even part: [G(V) + G(-V)]/2 , symmetric around V=0
G_LL_A = 0.5*(G_LL - G_LL[::-1]); G_LR_A = 0.5*(G_LR - G_LR[::-1]) # Odd part: [G(V) - G(-V)]/2 , antisymmetric around V=0
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(V_range, G_LL_S, linewidth=2, label=r'$G_{LL}^S$')
axes[0].plot(V_range, G_LR_S, linewidth=2, label=r'$G_{LR}^S$')
axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
_style(axes[0], 'Symmetric-bias: even parts'); axes[0].legend(); axes[0].set_xlabel(r'$V_{\rm bias}$'); axes[0].grid(alpha=0.5)

axes[1].plot(V_range, G_LL_A, linewidth=2, label=r'$G_{LL}^A$')
axes[1].plot(V_range, G_LR_A, linewidth=2, label=r'$G_{LR}^A$')
axes[1].plot(V_range, G_LL_A+G_LR_A, linewidth=2, label=r'$G_{LL}^A+G_{LR}^A != 0$')
axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
_style(axes[1], r'Symmetric-bias: odd parts'); axes[1].legend(); axes[1].set_xlabel(r'$V_{\rm bias}$'); axes[1].grid(alpha=0.5)

plt.tight_layout(); plt.show()
print_params(p)
#%% --- PHASE / BIAS CONDUCTANCE MAP ---
pc = Params(nx=7, ny = 2, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.00, t_s=1.0, mu_s=1.0, 
            delta=0.*35, phi=0.0, tc_top=1.0, tc_bot=1.0, tc_barr=1.0,
            alpha=0.0, beta=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0, eta=1e-4, kT=5e-2
)

phi_vals = np.linspace(-np.pi, np.pi, 101)
V_bias_map = np.linspace(-2.5*pc.delta, 2.5*pc.delta, 101)
#V_bias_map = np.linspace(-0.5, 0.5,41)
dV_map = 1e-5

G_LL_map = np.empty((len(phi_vals),len(V_bias_map)))
G_LR_map = np.empty_like(G_LL_map)
G_sym_map = np.empty_like(G_LL_map)
G_anti_map = np.empty_like(G_LL_map)

junction0 = FourTerminalJunction(pc)
z1,_ = junction0._z_batches(E_sweep)
precomputed_normal = (
    junction0.lead_L.self_energy(z1),
    junction0.lead_R.self_energy(z1)
    )

def partial_G_vectorized(ch, E, V, dV, kT, scheme="sym"):
    """
    Computes partial conductances G_LL and G_LR evaluated around the baseline
    specified by 'scheme' ('sym': V_R = +V, 'anti': V_R = -V).
    """
    V = V[:, None]

    if scheme == "sym":
        VR_base = V
    elif scheme == "anti":
        VR_base = -V
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")

    def I_eval(VL, VR):
        fLe, fLh = f_electron(E[None, :], VL, kT), f_hole(E[None, :], VL, kT)
        fRe, fRh = f_electron(E[None, :], VR, kT), f_hole(E[None, :], VR, kT)
        I_curr = (
            ch["ee"][None, :] * (fLe - fRe)
            - ch["hh"][None, :] * (fLh - fRh)
            + ch["eh_cross"][None, :] * (fLe - fRh)
            - ch["he_cross"][None, :] * (fLh - fRe)
            + (ch["eh_local"] + ch["he_local"])[None, :] * (fLe - fLh)
        )
        return np.trapezoid(I_curr, E, axis=1)
    G_LL = (I_eval(V + dV, VR_base) - I_eval(V - dV, VR_base)) / (2 * dV)    # 1. G_LL=dI_L/dV_L (Vary V_L around V, V_R fixed at VR_base)
    G_LR = (I_eval(V, VR_base + dV) - I_eval(V, VR_base - dV)) / (2 * dV)    # 2. G_LR=dI_L/dV_R (V_L fixed at V, vary V_R around VR_base)

    return G_LL, G_LR

def eval_I_total(ch, E, VL, VR, kT):
    fLe, fLh = f_electron(E[None, :], VL, kT), f_hole(E[None, :], VL, kT)
    fRe, fRh = f_electron(E[None, :], VR, kT), f_hole(E[None, :], VR, kT)
    I_curr = (
        ch["ee"][None, :] * (fLe - fRe)
        - ch["hh"][None, :] * (fLh - fRh)
        + ch["eh_cross"][None, :] * (fLe - fRh)
        - ch["he_cross"][None, :] * (fLh - fRe)
        + (ch["eh_local"] + ch["he_local"])[None, :] * (fLe - fLh)
    )
    return np.trapezoid(I_curr, E, axis=1)

def total_dIdV_map(ch, E, V, dV, kT, scheme):
    """Direct derivative of I_L along the actual bias line — 'sym': V_R=+V,
    'anti': V_R=-V — evaluated at the correct point for each scheme.
    """
    V = V[:, None]
    if scheme == "sym":
        sign = 1
    elif scheme == "anti":
        sign = -1
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")
    Ip = eval_I_total(ch, E, V + dV, sign * (V + dV), kT)
    Im = eval_I_total(ch, E, V - dV, sign * (V - dV), kT)
    return (Ip - Im) / (2 * dV)

G_LL_sym_map = np.empty((len(phi_vals), len(V_bias_map)))
G_LR_sym_map = np.empty_like(G_LL_sym_map)
G_LL_anti_map = np.empty_like(G_LL_sym_map)
G_LR_anti_map = np.empty_like(G_LL_sym_map)
G_sym_map = np.empty_like(G_LL_sym_map)
G_anti_map = np.empty_like(G_LL_sym_map)


# --- SIMULATION LOOP ---
for i, phi in enumerate(tqdm(phi_vals, desc="Phase sweep")):
    p_i = Params(**{**pc.__dict__, "phi": phi})
    ch_i = FourTerminalJunction(p_i).channels(E_sweep, side_name="left", precompute_sigmas_n=precomputed_normal)

    # 1. Partial conductances evaluated along symmetric baseline (V_R = +V)
    G_LL_sym_map[i], G_LR_sym_map[i] = partial_G_vectorized(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="sym")
    # 2. Partial conductances evaluated along antisymmetric baseline (V_R = -V)
    G_LL_anti_map[i], G_LR_anti_map[i] = partial_G_vectorized(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="anti")
    # 3. Total directional derivatives along the bias trajectories
    G_sym_map[i] = total_dIdV_map(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="sym")
    G_anti_map[i] = total_dIdV_map(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="anti")


# %% --- PLOT 2D MAP---
G_tot_sym   = G_LL_sym_map + G_LR_sym_map
G_tot_anti  = G_LL_anti_map - G_LR_anti_map

maps = [
    (G_LL_sym_map, r"$G_{LL}=\partial I_L/\partial V_L$ (Symmetric baseline)",),
    (G_LR_sym_map, r"$G_{LR}=\partial I_L/\partial V_R$ (Symmetric baseline)",),
    (G_LL_anti_map, r"$G_{LL}=$\partial I_L/\partial V_L$ (Antisymmetric baseline)"),
    (G_LR_anti_map, r"$G_{LR}=$\partial I_L/\partial V_R$ (Antisymmetric baseline)"),
    (G_sym_map, r"$dI_L/dV$ Total Symmetric Directional"),
    (G_anti_map, r"$dI_L/dV$ Total Antisymmetric Directional"),
]

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
for ax, (G, title) in zip(axs.flat, maps):
    cf = ax.contourf(phi_vals / np.pi, V_bias_map / pc.delta, G.T, levels=100, cmap="inferno")
    fig.colorbar(cf, ax=ax, label=r"$e^2/h$")
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$V_{\rm bias}/\Delta$")
    _style(ax, title)
plt.tight_layout()
plt.show()
print_params(pc)
#%% --- PARAMETER ANALYSIS FUNCTIONS---

def compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map=1e-5):
    '''One parameter instance -> returns G_LL and G_LR maps.'''
    junction0 = FourTerminalJunction(pc)
    z1, _ = junction0._z_batches(E_sweep)
    precomputed_normal = (junction0.lead_L.self_energy(z1), junction0.lead_R.self_energy(z1))

    G_LL_map = np.empty((len(phi_vals), len(V_bias_map)))
    G_LR_map = np.empty_like(G_LL_map)

    for i, phi in enumerate(phi_vals):
        p_i = Params(**{**pc.__dict__, 'phi': phi})
        ch_i = FourTerminalJunction(p_i).channels(E_sweep, side_name='left', precompute_sigmas_n=precomputed_normal)

        G_LL_map[i], G_LR_map[i] = partial_G_vectorized(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="sym")

    return G_LL_map, G_LR_map

def summarize_map(G_LL_map, G_LR_map):
    '''
    Summarizes the non-local conductance. 
    Positive peak indicates CAR dominates, negative indicates EC dominates.
    '''
    flat_GLR = G_LR_map.flatten()
    max_idx = np.argmax(np.abs(flat_GLR))
    peak_GLR = flat_GLR[max_idx]
    
    return {
        'mean_GLR': np.mean(G_LR_map),
        'peak_GLR': peak_GLR,
        'peak_GLL': np.max(np.abs(G_LL_map))
    }

def scan_grid(base_params, name1, vals1, name2, vals2, E_sweep, phi_vals, V_bias_map, dV_map=1e-5):
    results = np.empty((len(vals1), len(vals2)), dtype=object)
    for i, v1 in enumerate(tqdm(vals1, desc=f'{name1} sweep')):
        for j, v2 in enumerate(vals2):
            pc = Params(**{**base_params.__dict__, name1: v1, name2: v2})
            
            G_LL_map, G_LR_map = compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map)
            
            results[i, j] = {
                'G_LL_map': G_LL_map, 
                'G_LR_map': G_LR_map,
                'summary': summarize_map(G_LL_map, G_LR_map),
            }
    return results



def plot_grid_thumbnails(results, name1, vals1, name2, vals2, phi_vals, V_bias_map, delta):
    '''Grid of small contour plots of Non-Local Conductance G_LR, one per (v1,v2) cell.'''
    n1, n2 = len(vals1), len(vals2)
    fig, axs = plt.subplots(n1, n2, figsize=(2.6*n2, 2.4*n1), squeeze=False)
    
    for i in range(n1):
        for j in range(n2):
            ax = axs[i, j]
            G_LR = results[i, j]['G_LR_map']
            
            # Find symmetrical bounds for the diverging colormap
            vmax = np.max(np.abs(G_LR))
            if vmax == 0: vmax = 1e-10 # avoid singular norm
            
            # Red/Blue colormap centered exactly at 0. Red = CAR (+), Blue = EC (-)
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            cf = ax.contourf(phi_vals/np.pi, V_bias_map/delta, G_LR.T, levels=40,
                             cmap='RdBu_r', norm=norm)
            
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f'{name2}={vals2[j]:.2g}', fontsize=9)
            if j == 0: ax.set_ylabel(f'{name1}={vals1[i]:.2g}', fontsize=9)
            
    fig.suptitle(r'Non-local Conductance $G_{LR}$ across ' + f'{name1} x {name2} \n(Red = CAR, Blue = EC)', y=1.05)
    plt.tight_layout()
    return fig, axs

def plot_summary_trends(results, name1, vals1, name2, vals2):
    '''Line plot of Peak G_LR vs name1, tracking the transition from EC to CAR.'''
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for j, v2 in enumerate(vals2):
        peak_glr = [results[i, j]['summary']['peak_GLR'] for i in range(len(vals1))]
        ax.plot(vals1, peak_glr, marker='o', label=f'{name2}={v2:.2g}')
    ax.set_xlabel(name1)
    ax.set_ylabel(r'Peak Non-local Conductance $G_{LR}$ ($e^2/h$)')
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.7)
    ylim = ax.get_ylim()
    ax.text(vals1[0], 0.15 * abs(ylim[1] - ylim[0]), 'CAR Dominates ($G_{LR} > 0$)', 
            color='tab:red', fontsize=10, va='bottom')
    ax.text(vals1[0], -0.15 * abs(ylim[1] - ylim[0]), 'EC Dominates ($G_{LR} < 0$)', 
            color='tab:blue', fontsize=10, va='top')
    ax.legend()
    _style(ax, r'Mechanism Tuning via $G_{LR}$')
    plt.tight_layout()
    return fig, ax


#%% --- RUN ANALYSIS---

base = Params(nx=7, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.0, t_s=1.0, mu_s=1.0, delta=0.35, tc_top=1.0, tc_bot=1.0, tc_barr=1.0, eta=1e-4, kT=5e-2)

phi_vals_scan = np.linspace(-np.pi, np.pi, 21)
V_bias_scan   = np.linspace(-1.1*base.delta, 1.1*base.delta, 31)

var1_name, var1_vals = 'mu_s', np.array([0.0, 1.0])
var2_name, var2_vals = 't_c',  np.array([0.4, 0.9])

results = scan_grid(base, var1_name, var1_vals, var2_name, var2_vals, E_sweep, phi_vals_scan, V_bias_scan)

# Plot the 2D heatmaps of GLR
plot_grid_thumbnails(results, var1_name, var1_vals, var2_name, var2_vals, phi_vals_scan, V_bias_scan, base.delta)

# Plot the tuning summary (Zero crossing = EC to CAR phase transition)
plot_summary_trends(results, var1_name, var1_vals, var2_name, var2_vals)

plt.show()
# %%
pT = Params(
        nx=5, ny=10, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0, 
        delta=0.35, phi = 0.0, tc_top=1.00, tc_bot=1.00, tc_barr=1.00, 
        eta=1e-5, kT=2e-3)

T_ee = []
T_eh = []
T_he = []
T_lar_eh = []
T_lar_he = []

E_fixed = np.array([0.0])
phi_vals = np.linspace(0.0, 2*np.pi, 201)

for phi_val in phi_vals:
    p_phi = Params(**{**pT.__dict__, 'phi': phi_val})
    junction_phi = FourTerminalJunction(p_phi)
    ch_left = junction_phi.channels(E_fixed)['left']

    T_ee.append(ch_left['ee'])
    T_eh.append(ch_left['eh_cross'])
    T_he.append(ch_left['he_cross'])
    T_lar_eh.append(ch_left['eh_local'])
    T_lar_he.append(ch_left['he_local'])

#%%
fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

#ax[0].plot(phi_vals, T_ee, label=r"$T_{ee}$ (ECT)")
ax[0].plot(phi_vals, T_eh, label=r"$T_{eh}$ (CAR)")
ax[0].set_title(fr'Transmissions at fixed E={E_fixed[0]:.2f} vs. $\phi$')
ax[0].plot(phi_vals, T_he, label=r"$T_{he}$ (CAR)")
ax[0].set_ylabel("Transmission")
ax[0].legend()
ax[0].grid(True)
lar_comb = np.array(T_lar_eh) + np.array(T_lar_he)
ax[1].plot(phi_vals,T_lar_eh, label=r"$T_{eh}$ (LAR)")
#ax[1].plot(phi_vals, T_lar_he, label=r"$T_{he}$ (LAR)")
ax[1].set_xlabel(r"$\phi / \pi$")
ax[1].set_ylabel("Transmission")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
# %%
