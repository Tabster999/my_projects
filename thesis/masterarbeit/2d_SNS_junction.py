#%%
"""After verifying the RGf method works, by comparing it to the exact diagonalization, it is now used to calculate the GF for a 2d SNS junction and plot the LDOS as a function of energy and phase difference, as always."""
import numpy as np
import matplotlib.pyplot as plt
import my_functions as myf
import scipy.constants as const
from scipy.linalg import inv, block_diag
from math import pi, cos, sin, sqrt
#%% Create parameter class and set plotting style 

class SNSJunction:
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

    def get_textstr(self):
        return '\n'.join((
            r'$\mu_m/t = {:.3f}$'.format(self.mu_m),
            r'$\mu_s/t $= {:.3f}'.format(self.mu_sc),
            r'$t = {:.1f}$'.format(self.t),
            r'$\alpha/t = {:.1f}$'.format(self.alpha),
            r'$\Delta/t = {:.1f}$'.format(self.delta)
        ))
    def build_sns_junction(self, phi):
        """
        Builds a 1d SNS-junction Hamiltonian for RGF calculations.
        Constructs the system as a segmented chain: Left lead - Middle - Right lead
        
        Args:
            phi: float, phase difference between the SC leads

        Returns:
            np.ndarray of dimensions [dof * (sites_left + sites_right + sites_mid), ...]
        """

        t = self.t
        mu_sc = self.mu_sc
        mu_m = self.mu_m
        alpha = self.alpha
        h = self.h
        delta = self.delta
        sites_left = self.sites_l
        sites_right = self.sites_r
        sites_mid = self.sites_m
        dof = self.dof

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


one_dim_junction = SNSJunction()

# Plotting styles
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)

#%% Define functions 

