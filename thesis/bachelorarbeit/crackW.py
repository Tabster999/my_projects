#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Funktion zur Berechnung des Hamiltonians
def my_hamiltonian(k, h, m, alpha, Delta, mu):
    H = np.array([[k**2/(2*m) - mu + h, 1j*alpha*k, 0, Delta],
                  [-1j*alpha*k, k**2/(2*m) - mu - h, -Delta, 0],
                  [0, -Delta, -k**2/(2*m) + mu - h, -1j*alpha*k],
                  [Delta, 0, 1j*alpha*k, -k**2/(2*m) + mu + h]])
    return H

# %%
m = 1
Delta = 1
mu = 2

k_array = np.linspace(-6.5, 6.5, 600)

# Define Pauli matrices
sigma_z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
sigma_3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])  # particle-hole

# Parameters for different scenarios
scenarios = [
    {"h": 0.0, r"$\alpha$": 0.0, "title": fr"Energy spectrum ($\Delta={Delta}$)"},
    {"h": 0.5, r"$\alpha$": 0.0, "title": r"Energy spectrum ($\Delta=1 , h=0.5$)"},
    {"h": 1.5, r"$\alpha$": 0.0, "title": r"Energy spectrum ($\Delta=1 , h=1.5$)"},
    {"h": 0.0, r"$\alpha$": 0.5, "title": r"Energy spectrum ($\Delta=1 , h=0.0 , \alpha=0.5$)"},
    {"h": 0.0, r"$\alpha$": 1.0, "title": r"Energy spectrum ($\Delta=1, h=0.0 , \alpha=1.0$)"},
    {"h": 0.5, r"$\alpha$": 0.5, "title": r"Energy spectrum ($\Delta=1 , h=0.5 , \alpha=0.5$)"},
    {"h": 1.5, r"$\alpha$": 1.0, "title": r"Energy spectrum ($\Delta=1 , h=1.5 , \alpha=1.0$)"}
]

for scenario in scenarios:
    h = scenario["h"]
    alpha = scenario[r"$\alpha$"]
    
    # Use physical units directly, no normalization
    energy_bands = np.zeros((4, len(k_array)))
    spin_expectation = np.zeros((4, len(k_array)))
    
    # Calculate energy bands and spin expectation values
    for i, k in enumerate(k_array):
        H = my_hamiltonian(k, h, m, alpha, Delta, mu)
        energies, eigvecs = np.linalg.eigh(H)
        energy_bands[:, i] = energies
        
        for j in range(4):
            psi = eigvecs[:, j]
            spin_expectation[j, i] = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
    
    # Band labels - easily customizable
    band_labels = ['Hole up', 'Hole down', 'Electron down', 'Electron up']
    colors = ['red', 'orange', 'blue', 'cyan']
    
    # Plot each scenario separately
    fig, ax = plt.subplots(dpi=100)
    
    for j in range(4):
        ax.plot(k_array, energy_bands[j, :], color=colors[j], lw=2, label=band_labels[j])
    
    ax.set_ylim(-4, 4)
    ax.set_xlim(-6, 6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, lw=0.5)
    ax.set_title(scenario["title"], fontsize=12)
    ax.set_xlabel(r'$k$', fontsize=11)
    ax.set_ylabel(r'$E$', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Physical Parameters (Adjust as needed)
v = 1.0      # Velocity
mu = 2     # Chemical potential
delta = 1  # Superconducting gap
eta = 0.01   # Small broadening for numerical stability
# Define the grid for Momentum (p) and Energy (E)
p = np.linspace(-4, 4, 500)
E = np.linspace(-4, 4, 500)
P, E_grid = np.meshgrid(p, E)

# We use a complex energy to avoid singularities at the poles
E_c = E_grid + 1j * eta

# Common Denominator Term: (E^2 - (vp - mu)^2 - Delta^2) * (E^2 - (vp + mu)^2 - Delta^2)
term1 = E_c**2 - (v*P - mu)**2 - delta**2
term2 = E_c**2 - (v*P + mu)**2 - delta**2
denominator = term1 * term2

# f0 calculation (Singlet)
numerator_f0 = delta * (E_c**2 - mu**2 - v**2 * P**2 - delta**2)
f0 = numerator_f0 / denominator

# f vector calculation (Triplet Magnitude)
# In your model, f is effectively the scalar magnitude of the triplet part
numerator_f = -2 * delta * mu * v * P
f_triplet = numerator_f / denominator

# Plotting the results (Real parts)
# Plot f0 (Singlet) - Real part
fig, ax = plt.subplots()
im0 = ax.pcolormesh(p, E, np.real(f0), cmap='RdBu_r', shading='auto', vmin=-5, vmax=5)
ax.set_title(r'Real part $d_0$')
ax.set_xlabel('$k$')
ax.set_ylabel('$E$')
fig.colorbar(im0, ax=ax)
ax.axhline(y=0, color='k', linestyle='--', lw=1)
plt.tight_layout()
plt.show()

# Plot f (Triplet) - Real part
fig, ax = plt.subplots()
im1 = ax.pcolormesh(p, E, np.real(f_triplet), cmap='RdBu_r', shading='auto', vmin=-5, vmax=5)
ax.set_title(r'Real part $d_x$')
ax.set_xlabel('$k$')
ax.set_ylabel('$E$')
fig.colorbar(im1, ax=ax)
ax.axhline(y=0, color='k', linestyle='--', lw=1)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Physical Parameters (Adjust as needed)
m = 1.0        # Effective mass for epsilon_k
alpha = 0.8    # Spin-orbit coupling
h = .5        # Zeeman field
mu = 2       # Chemical potential
delta = 1.0    # Superconducting gap
eta = 1e-6     # Small broadening for numerical stability (Increased slightly for better visuals)

# Define the grid for Momentum (k) and Energy (E)
k = np.linspace(-6, 6, 500)
E = np.linspace(-6, 6, 500)
K, E_grid = np.meshgrid(k, E)

# We use a complex energy to avoid singularities at the poles
E_c = E_grid + 1j * eta

# Kinetic energy
epsilon_k = K**2 / (2 * m)

# ==========================================
# LET ME SET 'w' MYSELF
# ==========================================
# Define your denominator 'w' here. 
# As a placeholder, this is the standard denominator for the Rashba wire 
# w = (E^2 - E_+^2)(E^2 - E_-^2) based on the energy spectrum from your previous slides.
inner_sqrt = np.sqrt((epsilon_k - mu)**2 * (h**2 + alpha**2 * K**2) + h**2 * delta**2)
E_plus_sq = (epsilon_k - mu)**2 + alpha**2 * K**2 + h**2 + delta**2 + 2 * inner_sqrt
E_minus_sq = (epsilon_k - mu)**2 + alpha**2 * K**2 + h**2 + delta**2 - 2 * inner_sqrt

w = ((delta**2 - E_c**2) + (K*alpha + mu*epsilon_k)**2) *(delta**2 - epsilon_k**2 + (K*alpha - mu + epsilon_k)**2)+ h**4 + 2*h**2*((K*alpha)**2 - delta**2 - E_c**2 - (epsilon_k - mu)**2)  
# ==========================================

# Calculate the 4 components from the image
F0 = (((K * alpha)**2 + delta**2 + (epsilon_k - mu)**2 - h**2 - E_c**2) * delta) / w
Fx = (-2j * K * alpha * h * delta) / w
Fy = (-2 * K * alpha * delta * (epsilon_k - mu)) / w
Fz = (2 * h * E_c * delta) / w

# Store in a list to plot iteratively
components = [
    (r'F_0', F0, 'Real'),
    (r'F_x', Fx, 'Imaginary'), # Fx has an explicit 'i' in the numerator!
    (r'F_y', Fy, 'Real'),
    (r'F_z', Fz, 'Real')
]

# Plotting the results
for name, data, part in components:
    fig, ax = plt.subplots()
    
    # Extract the requested part (Real or Imaginary)
    if part == 'Real':
        plot_data = np.real(data)
    else:
        plot_data = np.imag(data)
        
    im = ax.pcolormesh(k, E, plot_data, cmap='RdBu_r', vmin=-5, vmax=5)
    
    ax.set_title(f'{part} part ${name}$')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$E$')
    
    fig.colorbar(im, ax=ax)
    ax.axhline(y=0, color='k', linestyle='--', lw=1)
    
    plt.tight_layout()
    plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Physical Parameters (Adjust as needed)
m = 1.0        # Effective mass for epsilon_k
alpha = 1.2    # Spin-orbit coupling
h = 1.2        # Zeeman field
mu = 0.25     # Chemical potential
delta = 1.0    # Superconducting gap
eta = 1e-4     # Small broadening for numerical stability (Set to 0.05 for beautiful smooth bands)

# Define the grid for Momentum (k) and Energy (E)
k = np.linspace(-4, 4, 601)
E = np.linspace(-4, 4, 601)
K, E_grid = np.meshgrid(k, E)

# We use a complex energy to avoid singularities at the poles
E_c = E_grid + 1j * eta

# Kinetic energy
epsilon_k = K**2 / (2 * m)
xi_k = epsilon_k - mu

# ==========================================
# THE CORRECTED 'w' (Characteristic Polynomial)
# ==========================================
# w = (E^2 - E_+^2)(E^2 - E_-^2) expanded analytically
A = xi_k**2 + (K * alpha)**2 + h**2 + delta**2
B = xi_k**2 * (h**2 + (K * alpha)**2) + h**2 * delta**2

w = (E_c**2 - A)**2 - 4 * B
# ==========================================

# Calculate the 4 components from the image
F0 = (((K * alpha)**2 + delta**2 + xi_k**2 - h**2 - E_c**2) * delta) / w
Fx = (-2j * K * alpha * h * delta) / w
Fy = (-2 * K * alpha * delta * xi_k) / w
Fz = (2 * h * E_c * delta) / w

# Store in a list to plot iteratively
components = [
    (r'd_0', F0, 'Imaginary'),
    (r'd_x', Fx, 'Real'), # Fx has an explicit 'i' in the numerator!
    (r'd_y', Fy, 'Imaginary'),
    (r'd_z', Fz, 'Imaginary')
]

# Plotting the results
for name, data, part in components:
    fig, ax = plt.subplots()
    
    # Extract the requested part (Real or Imaginary)
    if part == 'Imaginary':
        plot_data = np.real(data)
    else:
        plot_data = np.imag(data)
        
    im = ax.pcolormesh(k, E, plot_data, cmap='RdBu_r', shading='auto', vmin=-5, vmax=5)
    
    ax.set_title(f'${name}$')
    ax.set_xlabel('$k$')
    ax.set_ylabel(' $E$')
    
    fig.colorbar(im, ax=ax)
    ax.axhline(y=0, color='k', linestyle='--', lw=1)
    
    plt.tight_layout()
    plt.show()
# %%
