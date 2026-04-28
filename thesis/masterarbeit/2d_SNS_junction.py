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

def intraslice_hopping(hopping_matrix, N_slices):
    """Construct the intraslice hopping matrix for the 2D junction."""
    return block_diag(*[hopping_matrix for _ in range(N_slices)])





