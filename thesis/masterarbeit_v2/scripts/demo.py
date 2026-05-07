"""
new_minimal_workflow.py
========================

A small example using the new minimal files:
  - hamiltonians.py
  - solvers.py
  - helpers.py

This is the light version of your workflow.
"""
#%%
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hamiltonians import onsite_matrix, t_matrix, build_sns_junction
from solvers import calc_G, get_G_energy, get_surface_gfs_phased
#%%
# Parameters
params = {
    't': 1.0,
    'mu_sc': 0.0025,
    'mu_m': 0.01,
    'alpha': 0.60,
    'B': 0.40,
    'Delta': 0.10,
    'SL': 50,
    'SM': 20,
    'SR': 50,
    'dof': 4,
    'phi': np.pi,
    'eta': 1e-4,
}

# Build Hamiltonian
H = build_sns_junction(
    t=params['t'],
    mu_sc=params['mu_sc'],
    mu_m=params['mu_m'],
    alpha=params['alpha'],
    h=params['B'],
    delta=params['Delta'],
    phi=params['phi'],
    sites_left=params['SL'],
    sites_right=params['SR'],
    sites_mid=params['SM'],
    dof=params['dof'],
    symmetric=True,
)

print('Hamiltonian built:', H.shape)
print('Hermitian check:', np.allclose(H, H.conj().T))

# Solve at a single energy
energy = 0.0
G = calc_G(energy, H, eta=params['eta'])
print('Green function shape:', G.shape)
print('LDOS at site 0:', -np.imag(np.trace(G[:4, :4])) / np.pi)

# Solve over energy range
energies = np.linspace(-0.2, 0.2, 21)
G_array = get_G_energy(energies, H, eta=params['eta'])
print('G array shape:', G_array.shape)

# Demonstrate surface GF phase correction
onsite_sc = onsite_matrix(params['t'], params['mu_sc'], params['B'], params['Delta'])
V = t_matrix(params['t'], params['alpha'])
g_L, g_R = get_surface_gfs_phased(energy, onsite_sc, V, params['phi'], eta=params['eta'], symmetric=True)
print('Surface GF shapes:', g_L.shape, g_R.shape)
print('Surface pair phase diff:', np.angle(g_R[0,3]) - np.angle(g_L[0,3]))
#%%