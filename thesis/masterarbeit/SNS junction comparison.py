#%% 
"""Comparison of brute-force diagonalization and RGF method for calculating the energy spectrum and LDOS of a 1d SNS junction. This was mainly done to verify the RGF calculations, by comparing to the brute-force inversion results and Sancho Lopez results obtained in my Bachelor thesis."""
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import inv
from scipy.linalg import block_diag
from math import pi, sin, cos, exp
import my_functions as myf

#%% Parameters and configuration

class SNSParams:
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
        self.t_matrix = myf.t_matrix(self.t, self.alpha)
    def get_textstr(self):
        return '\n'.join((
            r'$\mu_m/t = {:.3f}$'.format(self.mu_m),
            r'$\mu_s/t $= {:.3f}'.format(self.mu_sc),
            r'$t = {:.1f}$'.format(self.t),
            r'$\alpha/t = {:.1f}$'.format(self.alpha),
            r'$\Delta/t = {:.1f}$'.format(self.delta)
        ))

params = SNSParams()

# Plotting styles
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)

# %% Define SNS junction Hamiltonian function

'''SNS junction Hamiltonian as block diagonals with hopping matrices coupling the leads'''

def build_sns_junction(t, mu_sc, mu_m, alpha, h, delta, phi, sites_left, sites_right, sites_mid, dof):
    r"""
    Builds a 1d SNS-junction Hamiltonian optimized for RGF calculations.
    Constructs the system as a concatenated chain: Left lead - Middle - Right lead
    
    Args:
        t:float, hopping
        mu:float, chemical potential
        alpha:float, spin-orbit strength
        h:float, zeeman energy
        delta:float, SC pairing 
        phi:float, phase difference between the SC leads
        sites_left:int, number of sites left lead
        sites_right:int, number of sites right lead
        sites_mid:int, number of sites middle lead
        dof:int, degrees of freedom per site
    
    Returns: np.ndarray of dimensions [dof * (sites_left + sites_right + sites_mid), ...]
    """

    # Get left, middle and right Hamiltonians
    H_L = myf.get_tb_hamiltonian(myf.onsite_matrix(t, mu_sc, h, delta), myf.t_matrix(t, alpha), sites_left)
    H_M = myf.get_tb_hamiltonian(myf.onsite_matrix(t, mu_m, h, 0), myf.t_matrix(t, alpha), sites_mid)
    H_R = myf.get_tb_hamiltonian(myf.onsite_matrix(t, mu_sc, h, delta*np.exp(1j*phi)), myf.t_matrix(t, alpha), sites_right)
    
    H = block_diag(H_L, H_M, H_R)
    
    # Add coupling between leads
    V = myf.t_matrix(t, alpha)
    idx_L_end = (sites_left - 1) * dof
    idx_M_start = sites_left * dof 
    idx_M_end = (sites_left + sites_mid - 1) * dof
    idx_R_start = (sites_left + sites_mid) * dof
    
    # L-M coupling
    H[idx_L_end:idx_L_end + dof, idx_M_start:idx_M_start + dof] = V 
    H[idx_M_start:idx_M_start + dof, idx_L_end:idx_L_end + dof] = V.conj().T
    
    # M-R coupling
    H[idx_M_end:idx_M_end + dof, idx_R_start:idx_R_start + dof] = V 
    H[idx_R_start:idx_R_start + dof, idx_M_end:idx_M_end + dof] = V.conj().T
    
    return H

#%% BRUTE FORCE: Calculation of energy spectrum for different phase differences 

energies = []
for phi in params.phi_array:
    H = myf.build_sns_junction(params.t, params.mu_m, params.mu_sc, params.alpha, params.h, params.delta, phi, params.sites_l, params.sites_r, params.sites_m, params.dof)
    eval = np.linalg.eigvalsh(H) 
    energies.append(np.sort(eval))

energies = np.array(energies)

#%% BRUTE FORCE: Plotting the brute force energy spectrum

plt.figure(figsize=(8, 6))
plt.plot(params.phi_array/(np.pi), energies, color='orange', lw=1)
plt.axvline(1, color='gray', linestyle=':', label=r'$\phi=\pi$')
plt.ylim(-params.delta, params.delta)
plt.xlim(0, 4)
plt.xlabel(r'$\Delta \phi / \pi$')
plt.text(0.02, 0.95, params.get_textstr(), transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', bbox=props)
plt.ylabel(r'Energy $E$')
plt.title(r'Energy spectrum vs $\Delta \phi$ (Brute Force)')
plt.grid(alpha=0.2)
plt.legend(loc="upper right")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(params.phi_array/(np.pi), energies, color='orange', lw=1)
plt.axvline(1, color='gray', linestyle=':', label=r'$\phi=\pi$')
plt.ylim(-.055, .055)
plt.xlim(0, 4)
plt.xlabel(r'$\Delta \phi / \pi$')
plt.text(0.02, 0.95, params.get_textstr(), transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', bbox=props)
plt.ylabel(r'Energy $E$')
plt.title(r'Energy spectrum vs $\Delta \phi$ (Brute Force) zoomed')
plt.grid(alpha=0.2)
plt.legend(loc="upper right")
plt.show()

#%% BRUTE FORCE: Plotting the spatial density of the zero-energy state at phi = pi

H_pi = build_sns_junction(params.t, params.mu_m, params.mu_sc, params.alpha, params.h, params.delta, np.pi, params.sites_l, params.sites_r, params.sites_m, params.dof)
evals_pi, evecs_pi = np.linalg.eigh(H_pi)

idx_zero = np.argmin(np.abs(evals_pi))
zero_state = evecs_pi[:, idx_zero]
psi_sd = zero_state.reshape(-1, 4)
spatial_density = np.einsum('sd, sd -> s', psi_sd, psi_sd.conj()).real

plt.figure(figsize=(8,6))
plt.title(r'$|\Psi|^2$ vs. sites for $\Phi=\pi$ (Brute Force)')
plt.xlabel('Site index')
plt.axvline(x=params.sites_l+1, label='Surface left', ls=':', c='blue', alpha=.5, lw=1)
plt.axvline(x=params.sites_l+params.sites_m+1, label='Surface right', ls=':', c='blue', alpha=.5, lw=1)
plt.ylabel(r'$|\Psi|^2$')
plt.text(0.02, 0.95, params.get_textstr(), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.plot(range(len(spatial_density)), spatial_density, color='green', alpha=0.5)
plt.legend(loc=9)
plt.show()

#%% Brute Force: Calculation of the LDOS as a function of energy for phi = 2pi and plotting it

energy_array = np.linspace(-params.delta, params.delta, 61)
G_brute = myf.calc_G_lehmann(energy_array, evals_pi, evecs_pi, params.eta, ra='r')

idx_start = params.sites_l * params.dof
idx_end = (params.sites_l + params.sites_m) * params.dof

ldos_brute_middle = np.zeros_like(energy_array)
for i, energy in enumerate(energy_array):
    G_energy = G_brute[i]

    G_middle = G_energy[idx_start:idx_end, idx_start:idx_end]
    ldos_brute_middle[i] = -1/np.pi * np.imag(np.trace(G_middle))

plt.figure()
plt.plot(energy_array, ldos_brute_middle)
plt.xlabel('Energy')
plt.ylabel('DOS middle lead')
plt.title(r'DOS middle lead vs Energy ($\phi=\pi$, Brute Force)')
plt.show()
#%%

# Choose a specific energy index from your energy_array
# For example, the index closest to E = 0
target_energy = 0
energy_idx = np.argmin(np.abs(energy_array - target_energy))

G_at_E0 = G_brute[energy_idx]

ldos_spatial = np.zeros(params.sites_tot)

for s in range(params.sites_tot):
    idx = s * params.dof
    G_site = G_at_E0[idx : idx + params.dof, idx : idx + params.dof]
    
    ldos_spatial[s] = -1/np.pi * np.imag(np.trace(G_site))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(np.arange(params.sites_tot), ldos_spatial, label=f'E = {energy_array[energy_idx]:.4f}')
plt.axvspan(params.sites_l, params.sites_l + params.sites_m, color='gray', alpha=0.2, label='Normal Region')
plt.xlabel('Site Index')
plt.ylabel('LDOS')
plt.title('Spatial Profile of LDOS')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%% RGF: calculation of energy spectrum for different phase differences

energy_array = np.linspace(-params.delta, params.delta, 61)

ldos_matrix = np.zeros((len(params.phi_array), len(energy_array)))

for i, phi in enumerate(params.phi_array):
    H_sns = build_sns_junction(params.t, params.mu_m, params.mu_sc, params.alpha, params.h, params.delta, phi, params.sites_l, params.sites_r, params.sites_m, params.dof)
    for j, energy in enumerate(energy_array):
        G, _, _ = myf.get_rgf_finite_system(H_sns, params.t_matrix, energy, eta=params.eta, ra='r', return_full=True)
        idx_start = params.sites_l * params.dof
        idx_end = (params.sites_l + params.sites_m) * params.dof
        G_middle = G[idx_start:idx_end, idx_start:idx_end]
        ldos = -1/np.pi * np.imag(np.trace(G_middle))
        ldos_matrix[i, j] = ldos

#%% RGF: Plotting the LDOS as a function of phase difference and energy (colormap)

ldos_clipped = np.clip(ldos_matrix, 0, 10)
plt.figure(figsize=(8, 6))
plt.contourf(params.phi_array/(np.pi), energy_array, ldos_clipped.T, levels=100, cmap='cool')
plt.colorbar(label='LDOS')
plt.axhline(0, color='lightgray', linestyle='--', linewidth=.5)
plt.xlabel(r'$\Delta \phi / \pi$')
plt.ylabel(r'Energy $E$')
plt.title(r'LDOS vs $\Delta \phi$ (RGF Method)')
plt.text(0.02, 0.95, params.get_textstr(), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.show()

# %% RGF: Plotting the spatial density of the zero-energy state at phi = pi

G_pi, _, _ = myf.get_rgf_finite_system(myf.build_sns_junction(params.t, params.mu_m, params.mu_sc, params.alpha, params.h, params.delta, np.pi, params.sites_l, params.sites_r, params.sites_m, params.dof), params.t_matrix, 0, eta=params.eta, ra='r', return_full=True)
psi_sd_rgf = (-1/np.pi) * np.diag(G_pi.imag).reshape(-1, params.dof)
spatial_density_rgf = np.einsum('sd, sd -> s', psi_sd_rgf, psi_sd_rgf.conj()).real
plt.figure(figsize=(8,6))
plt.title(r'$|\Psi|^2$ vs. sites for $\Phi=\pi$ (RGF Method)')
plt.xlabel('Site index')
plt.axvline(x=params.sites_l+1, label='Surface left', ls=':', c='blue', alpha=.5, lw=1)
plt.axvline(x=params.sites_l+params.sites_m+1, label='Surface right', ls=':', c='blue', alpha=.5, lw=1)
plt.ylabel(r'$|\Psi|^2$')
plt.plot(range(len(spatial_density_rgf)), spatial_density_rgf, color='green', alpha=0.5)
plt.ylim(0, 0.05)
plt.legend(loc=9)
plt.text(0.02, 0.95, params.get_textstr(), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.show()
# %% RGF: LDOS of middle lead at phi = 2pi
idx_start = params.sites_l * params.dof
idx_end = (params.sites_l + params.sites_m) * params.dof
# Build Hamiltonian at specific phi
phi_target = np.pi
H_sns_target = myf.build_sns_junction(params.t, params.mu_m, params.mu_sc, params.alpha, params.h, params.delta, phi_target, params.sites_l, params.sites_r, params.sites_m, params.dof)

ldos_middle = []

for j, energy in enumerate(energy_array):
    G, _, _ = myf.get_rgf_finite_system(H_sns_target, params.t_matrix, energy, eta=params.eta, ra='r', return_full=True)
    G_middle = G[idx_start:idx_end, idx_start:idx_end]
    ldos = -1/np.pi * np.imag(np.trace(G_middle))
    ldos_middle.append(ldos)

plt.figure()
plt.plot(energy_array, ldos_middle)
plt.xlabel('Energy')
plt.ylabel('LDOS middle lead')
plt.title(f'LDOS middle lead vs Energy (phi={phi_target/np.pi:.1f}π)')
plt.show()

# %%
