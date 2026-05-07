#%% import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import my_functions as myf
import time
#%% initialize parameters 
r''' hamiltonian structure
in the basis \Psi^\dagger = (\Psi_\uparrow^\dagger, \Psi_\downarrow^\dagger, \Psi_\downarrow, \Psi_\uparrow) the Hamiltonian operator is given by:
    H = H_{kin} + H_\alpha + H_z + H_\Delta
with:
    H_kin = \sum_{i,j} c_i^\dagger (1/2m) *(p_x^2 + p_y^2) \sigma_0 \tau_z c_j
    H_\alpha = \alpha \sum_i c_i^\dagger (p_x \sigma_x + p_y \sigma_y) \tau_z c_i
    H_z = h \sum_i c_i^\dagger \sigma_z c_i
    H_\Delta = \sum_i [\Delta * (c_{i,\uparrow} c_{i,\downarrow}) + \Delta^\star * (c_{i,\uparrow}^\dagger c_{i,\downarrow}^\dagger)]

the index i is a superindex for (x,y)
'''

'Variables'
sites = 81
t = 1.0
mu = 0.025 * t
alpha = 0.4 * t
delta = 0.1 * t
eta = 1e-4 * delta
h = 0.2 * t
phase_transition = np.sqrt(delta**2+mu**2)


'Arrays'
e_steps = 101
e_min = - delta
e_max = -e_min
energy_array = np.linspace(e_min, e_max, e_steps)

h_steps = 91
h_min = .0
h_max = .25
h_array = np.linspace(h_min, h_max, h_steps)

ky_steps = 71
ky_min = -np.pi
ky_max = np.pi
ky_array = np.linspace(ky_min, ky_max, ky_steps)

textstr = '\n'.join((
r'$\mu/t = {:.3f}$'.format(mu),
r'$t = {:.1f}$'.format(t),
r'$\alpha/t = {:.1f}$'.format(alpha),
r'$\Delta/t = {:.1f}$'.format(delta)
))
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
#%% Define Hamiltonian 
r''' hamiltonian after fourier transform
After partial FT y -> k_y:
    H_kin -> -t * \sum_{x,k_y} c_{x,k_y}^\dagger *(c_{x-1,k_y} + c_{x+1,k_y} - 4*c_{x,k_y} + 2*cos(k_y^\tilde*a)*c_{x,k_y})     
    H_\alpha -> -i*\alpha^\tilde * \sum_{x,k_y} c_{x,k_y}^\dagger *[\sigma_x*(c_{x+1,k_y} - c_{x-1,k_y}) - \sigma_y * (2i*sin(k_y^\tilde*a))] 
with:
    t = \hbar^2/(2ma)
    \alpha^\tilde = \hbar*\alpha / 2a
    k_y^\tilde = (2m*k_y) / \hbar
'''

def get_hopping(t, alpha):
    t_matrix = np.array([[-t, alpha, 0, 0], 
                       [-alpha, -t, 0, 0],
                       [0, 0, t, alpha], 
                       [0, 0, -alpha, t]], 
                        dtype=np.complex128)
    
    return t_matrix

def get_h0(t, mu, h, alpha, delta, k_y):
    onsite_matrix = np.array(
        [[-2*t*(np.cos(k_y) - 2) - mu + h, -2*np.sin(k_y)*alpha, delta, 0], 
        [-2*np.sin(k_y)*alpha, -2*t*(np.cos(k_y) - 2) - mu - h, 0, -delta], 
        [delta, 0, 2*t*(np.cos(k_y) - 2) + mu + h, -2*np.sin(k_y)*alpha], 
        [0, -delta, -2*np.sin(k_y)*alpha, 2*t*(np.cos(k_y) - 2) + mu - h]], 
        dtype=np.complex128)

    return onsite_matrix