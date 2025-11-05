#%%
import numpy as np 
import matplotlib.pyplot as plt
#%%
"""Calculation of Green-function in Nambu-Basis up,down,up_dagger,down_dagger

-k: momentum
-h: Zeeman-energy
-m: mass of quasiparticle
-alpha: Rashaba-coefficient
-Delta: superconducting pairing potential
-µ: chemical potential
-e: Total energy
eta np.complex128 broadening
"""

'Set-up Hamiltonian (H) and calculate Green-function (G). h_bar := 1'
def my_hamiltonian(k, h, m, alpha, Delta, mu):
    H = np.array([[k**2/(2*m) - mu + h, 1j*alpha*k, 0, Delta],[-1j*alpha*k, k**2/(2*m) -  mu - h, -Delta, 0],[0, -Delta, -k**2/(2*m) + mu - h, -1j*alpha*k],[Delta, 0, 1j*alpha*k, -k**2/(2*m) + mu + h]])
    return H

def tkachov_hamiltonian(v, k, mu, delta):
    ha = np.array([[-mu,v*k,0,delta],[v*k, -mu, -delta,0],[0,-delta,mu,v*k],[delta,0,v*k,mu]])
    return ha

def tkachov_green_function_r(v, k, mu, delta,e):
    eta = .0001j
    matrix = (np.eye(4)*(e+eta) - tkachov_hamiltonian(v, k, mu, delta))
    gr = np.linalg.inv(matrix)
    return gr

def tkachov_green_function_a(v, k, mu, delta,e):
    eta = .0001j
    matrix = (np.eye(4)*(e-eta) - tkachov_hamiltonian(v, k, mu, delta))
    ga = np.linalg.inv(matrix)
    return ga

def my_green_function_a(k, h, m, alpha, Delta, mu,e):
    eta = .01j
    matrix = (np.eye(4)*(e-eta) - my_hamiltonian(k,h,m,alpha,Delta,mu))
    G_a = np.linalg.inv(matrix)
    return G_a

def my_green_function_r(k, h, m, alpha, Delta, mu,e):
    eta = .01j
    matrix = (np.eye(4)*(e+eta) - my_hamiltonian(k,h,m,alpha,Delta,mu))
    G_r = np.linalg.inv(matrix)
    return G_r

#%%
'''Set-up the parameters'''
m = 1
alpha = .8
Delta = .1
mu = 0.00025
k = .001

Delta_tilde = 1
mu_tilde = mu / Delta
alpha_tilde = alpha * np.sqrt(2*m/Delta)
k_tilde = k/np.sqrt(2*m*Delta)

e_min = -1
e_max = 1
phase_transition = np.sqrt(Delta_tilde**2 + mu_tilde**2)
h_array = np.linspace(0,.5,200)
energy_array = np.linspace(e_min,e_max,200)


#Create the arrays of anomalos gf elements (g12 if G=[[g11,g12],[g21,g22]]). it is a 2x2 matrix
r_uu = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
r_ud = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
r_du = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
r_dd = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)

a_uu = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
a_ud = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
a_du = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
a_dd = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)

for i, h in enumerate(h_array):
    for j, e in enumerate(energy_array):
        
        
        G_r = my_green_function_r(k_tilde, h/Delta, m, alpha_tilde, Delta_tilde, mu_tilde, e/Delta)
        G_a = my_green_function_a(k_tilde, h/Delta, m, alpha_tilde, Delta_tilde, mu_tilde, e/Delta)
        
        g_r = G_r[0:2,2:4]
        g_a = G_a[0:2,2:4]
        
        r_uu[i, j] = g_r[0,0]
        r_ud[i, j] = g_r[1,0]
        r_du[i, j] = g_r[0,1]
        r_dd[i, j] = g_r[1,1]
        
        a_uu[i, j] = g_a[0,0]
        a_ud[i, j] = g_a[1,0]
        a_du[i, j] = g_a[0,1]
        a_dd[i, j] = g_a[1,1]

d0 = .5j*((a_ud - r_ud) - (a_du - r_du))
dx = -.5j*((a_uu - r_uu) - (a_dd - r_dd))
dy = .5*((a_uu - r_uu) - (a_dd - r_dd))
dz = .5j*((a_ud - r_ud) + (a_du - r_du))
#%% Plots over h 
titlestr = ['d0','dx','dy','dz']
energy_index = 100
d_list = [d0[:,energy_index].T,dx[:,energy_index].T, dy[:,energy_index].T, dz[:,energy_index].T]

fig, axs = plt.subplots(2,2, figsize=(12,8),dpi=600)
axs = axs.flatten()

for i in range(4):
    ax = axs[i]
    plot = ax.plot(h_array/Delta, np.imag(d_list[i]), color='red', label='imag')
    plot = ax.plot(h_array/Delta, np.real(d_list[i]), color='black', label='real')
    ax.set_title(titlestr[i]+' mit minus bei y')
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.legend()
    ax.set_xlim(.5,1.5)
    textstr = '\n'.join((
    r'$\mu = {:.3f}$'.format(mu),
    r'$k = {:.3f}$'.format(k),
    r'$\alpha = {:.1f}$'.format(alpha),
    r'$\Delta = {:.1f}$'.format(Delta)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.5, 0.5, textstr, fontsize=8, verticalalignment='center', horizontalalignment='center', bbox=props)
    ax.axvline(x=phase_transition, color='black', linestyle='--', label='phase transition')
plt.show()
#%% Contour plots
phase_transition = np.sqrt((mu_tilde)**2+Delta_tilde**2)

titlestr_d = ['d0 imag ','dx real','dy imag ','dz imag']

max_clip = 2
d0_clip_i = np.clip(d0.imag,-max_clip,max_clip)
dx_clip_i = np.clip(dx.imag,-max_clip,max_clip)
dy_clip_i = np.clip(dy.imag,-max_clip,max_clip)
dz_clip_i = np.clip(dz.imag,-max_clip,max_clip)
d_list_i = [d0_clip_i,dx_clip_i,dy_clip_i,dz_clip_i]

d0_clip_r = np.clip(d0.real,-max_clip,max_clip)
dx_clip_r = np.clip(dx.real,-max_clip,max_clip)
dy_clip_r = np.clip(dy.real,-max_clip,max_clip)
dz_clip_r = np.clip(dz.real,-max_clip,max_clip)
d_list_r = [d0_clip_r,dx_clip_r,dy_clip_r,dz_clip_r]
clips = [d0_clip_i.T,dx_clip_r.T,dy_clip_i.T,dz_clip_i.T]
#plot imag part of d_i in a contourplot for h, e
fig, axs = plt.subplots(2,2, figsize=(12,8), dpi=200)
axs = axs.flatten()
for i in range(4):
    ax = axs[i]
    plot = ax.contourf(h_array/Delta,energy_array/Delta, clips[i], levels=100, cmap='plasma')
    textstr = '\n'.join((
    r'$\mu = {:.3f}$'.format(mu),
    r'$k = {:.3f}$'.format(k_tilde),
    r'$\alpha = {:.1f}$'.format(alpha),
    r'$\Delta = {:.1f}$'.format(Delta),
    r'clip value = {}'.format(max_clip)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.5, 0.5, textstr, fontsize=8, verticalalignment='center', horizontalalignment='center', bbox=props)
    ax.set_title(titlestr_d[i]+' numerical')
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, color='black', linestyle='--')
    fig.colorbar(plot, ax=ax)
plt.tight_layout()
plt.show()

#plot real part of d_i in a contourplot for h, e
# fig, axs = plt.subplots(2,2, figsize=(12,8))
# axs = axs.flatten()
# for i in range(4):
#     ax = axs[i]
#     plot = ax.contourf(h_array,energy_array, d_list_r[i])
#     ax.set_title(titlestr_d[i]+' real numerical')
#     ax.set_xlabel('h')
#     ax.set_facecolor('lightyellow')
#     ax.axvline(x=phase_transition, color='black', linestyle='--')
#     fig.colorbar(plot, ax=ax)

# plt.tight_layout()
# plt.show()
#%% Calculate the integrals over e for d_i
d_x = (e_max - e_min)/201
d0_integrated = np.trapz(d0,energy_array/Delta,axis=0)
dx_integrated = np.trapz(dx,energy_array/Delta,axis=0)
dy_integrated = np.trapz(dy,energy_array/Delta,axis=0)
dz_integrated = np.trapz(dz,energy_array/Delta,axis=0)

integral_array = [d0_integrated, dx_integrated, dy_integrated, dz_integrated]
titlestr = ['d0', 'dx', 'dy', 'dz']
fig, axs = plt.subplots(2,2,figsize=(12,10),dpi=600)
axs = axs.flatten()

for i in range(4):
    ax = axs[i]
    plot = ax.plot(h_array/Delta, integral_array[i].real, c='black', label='real')
    plot = ax.plot(h_array/Delta, integral_array[i].imag, c='red', label='imag')
    ax.set_title('integral over e ' +titlestr[i])
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, c='black', linestyle='--')
    ax.legend()
plt.show()
# %%plots for the LDOS of spin components
m = 1
alpha = .5
Delta = .1
mu = -0.12
h = .2


Delta_tilde = 0
mu_tilde = mu / Delta
alpha_tilde = alpha * np.sqrt(2*m/Delta)
h_tilde = h/Delta

phase_transition = np.sqrt((mu_tilde)**2+Delta_tilde**2)
k_array = np.linspace(-3,3,400)
energy_array = np.linspace(-.75,.75,400)

electron_up = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
electron_down = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_up = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_down = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)

electron_up_soc = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
electron_down_soc = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_up_soc = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_down_soc = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)

electron_up_h = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
electron_down_h = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_up_h = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_down_h = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)


for i, k in enumerate(k_array):
    for j, e in enumerate(energy_array):
        
        Gr = my_green_function_r(k/np.sqrt(2*m*Delta),h, m, alpha_tilde, Delta_tilde, mu_tilde, e/Delta)
        
        electron_up[i, j] = -Gr[0,0]
        electron_down[i, j] = -Gr[1,1]
        hole_up[i, j] = -Gr[2,2]
        hole_down[i, j] = -Gr[3,3]
        

# %% plots for ldos of spin components
fig = plt.figure(figsize=(6, 8), dpi=600)
clip_value = 2
spin_components = [np.clip(electron_up.imag,-clip_value, clip_value), np.clip(electron_down.imag,-clip_value, clip_value), np.clip(hole_up.imag,-clip_value, clip_value), np.clip(hole_down.imag,-clip_value, clip_value)]
title = ['electron up','electron down', 'hole up', 'hole down']


contour_colors = ['darkblue', 'green', 'darkred', 'goldenrod']
contour_labels = ['Spin Component 0', 'Spin Component 1', 'Spin Component 2', 'Spin Component 3']

# Konturlinien
for i in range(4):
    c = plt.contour(k_array, energy_array, spin_components[i].T, colors=contour_colors[i], levels=100, linewidths=0.08)

plt.xticks([])
#plt.yticks([])
plt.tight_layout()
plt.show()
# %%

# Given parameters
m = 4
alpha = 0.4
Delta = 0.1
mu = 0.12
h = 0.0

# Rescaled dimensionless parameters
Delta_tilde = 1
mu_tilde = mu / Delta
alpha_tilde = alpha * np.sqrt(2*m/Delta)
h_tilde = h / Delta

# Define k and energy arrays
k_array = np.linspace(-3, 3, 400)
energy_array = np.linspace(-0.75, 0.75, 400)

# Initialize arrays to hold the Green's function components
electron_up = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
electron_down = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_up = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)
hole_down = np.zeros((len(k_array), len(energy_array)), dtype=np.complex128)

# Populate these arrays using your Green's function calculations
for i, k in enumerate(k_array):
    for j, e in enumerate(energy_array):
        # Assuming Gr is your Green's function calculation
        Gr = my_green_function_r(k/np.sqrt(2*m*Delta), h, m, alpha_tilde, Delta_tilde, mu_tilde, e/Delta)

        # Populate the arrays with the imaginary part of Green's function components
        electron_up[i, j] = -Gr[0, 0].imag
        electron_down[i, j] = -Gr[1, 1].imag
        hole_up[i, j] = -Gr[2, 2].imag
        hole_down[i, j] = -Gr[3, 3].imag

# Clip the values for better visualization
clip_value = 200000
spin_components = [
    np.clip(electron_up, -clip_value, clip_value),
    np.clip(electron_down, -clip_value, clip_value),
    np.clip(hole_up, -clip_value, clip_value),
    np.clip(hole_down, -clip_value, clip_value)
]

# Initialize a figure with subplots for each component
fig, axes = plt.subplots(4, 1, figsize=(6, 12), dpi=600)

# Titles for each subplot
titles = ['Spin-Up Electrons', 'Spin-Down Electrons', 'Spin-Up Holes', 'Spin-Down Holes']

# Colors corresponding to each component
colors = ['Reds', 'Blues', 'Oranges', 'Greens']

# Plot each component on its own subplot
for idx, (component, title, color) in enumerate(zip(spin_components, titles, colors)):
    ax = axes[idx]
    
    # Create a contour plot for each component
    c = ax.contourf(k_array, energy_array, component.T, levels=100, cmap=color, alpha=0.8)
    

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



# %% maybe scrap
# Parameterdefinition
m = 10
alpha = .01
Delta = .1
mu = -.005
h = .2  # Zeeman-Feld auf Null gesetzt

Delta_tilde = 0
mu_tilde = mu
alpha_tilde = alpha 
h_tilde = h
k_array = np.linspace(-1, 1, 500)
energy_bands = np.zeros((4, len(k_array)))

# Farben für Spin-Richtungen
colors = np.zeros((4, len(k_array), 3))  # RGB-Farbwerte für jeden Punkt in den Bändern

# Pauli-Matrizen für y- und z-Richtungen in der Nambu-Basis
sigma_y = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
sigma_z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Berechnung der Energiebänder und Spin-Erwartungswerte für jeden Wert von k
for i, k in enumerate(k_array):
    H = my_hamiltonian(k, h_tilde, m, alpha_tilde, Delta_tilde, mu_tilde)
    
    # Verwenden Sie scipy.linalg.eigh für bessere numerische Stabilität
    energies, eigvecs = np.linalg.eigh(H)
    energy_bands[:, i] = energies
    # Berechne die Spin-Erwartungswerte für jeden Zustand
    for j in range(4):  # Für jedes Band
        psi = eigvecs[:, j]
        S_y = np.real(np.vdot(psi, np.dot(sigma_y, psi)))
        S_z = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
        
        # Normiere S_y und S_z auf den Bereich [-1, 1]
        S_y_norm = (S_y + 1) / 2
        S_z_norm = (S_z + 1) / 2
        
        # Definiere Farben für Interpolation
        color_blau = np.array([0, 0, 1])  # Blau für S_z > 0
        color_rot = np.array([1, 0.5, 0])  # Orange für S_z < 0 (anstatt Rot)
        color_cyan = np.array([0.5, 0, .5])  # Grün für S_y > 0 (anstatt Cyan)
        color_magenta = np.array([0, 0, 0.8])  # Lila für S_y < 0 (anstatt Magenta)

        # Interpolation der Farben
        color_z_interp = (1 - S_z_norm) * color_rot + S_z_norm * color_blau
        color_y_interp = (1 - S_y_norm) * color_magenta + S_y_norm * color_cyan
        
        # Kombination der beiden Interpolationen
        colors[j, i] = 0.5 * color_z_interp + 0.5 * color_y_interp

# Plot der Energiebänder mit unterschiedlicher Linienbreite für Elektronen und Löcher
fig, ax = plt.subplots(figsize=(12, 8), dpi=600)

for j in range(4):
    for i in range(len(k_array) - 1):
        avg_color = (colors[j, i] + colors[j, i+1]) / 2
        
        # Unterscheidung zwischen Elektronen- und Löcherbändern
        if j < 2:  # Elektronenbänder (niedrigste beiden Energien)
            ax.plot(k_array[i:i+2], energy_bands[j, i:i+2], color=avg_color, lw=3)  # Dickere Linien
        else:  # Löcherbänder (höchste beiden Energien)
            ax.plot(k_array[i:i+2], energy_bands[j, i:i+2], color=avg_color, lw=3)  # Dünnere, gestrichelte Linien
plt.gca().set_facecolor('lightyellow')
plt.xticks([0])
plt.yticks([])
plt.tight_layout()
plt.show()

# %%
def my_hamiltonian(k, h, m, alpha, Delta, mu):
    H = np.array([
        [k**2/(2*m) - mu + h, 1j*alpha*k, 0, Delta],
        [-1j*alpha*k, k**2/(2*m) - mu - h, -Delta, 0],
        [0, -Delta, -k**2/(2*m) + mu - h, -1j*alpha*k],
        [Delta, 0, 1j*alpha*k, -k**2/(2*m) + mu + h]
    ])
    return H

# Parameters
m = 1
alpha = 0.2
Delta = 0.1
mu =  -.5
h = 0.5  # Zeeman field

Delta_tilde = 0
mu_tilde = mu / Delta
alpha_tilde = alpha * np.sqrt(2*m/Delta)
h_tilde = h / Delta
k_array = np.linspace(-3, 3, 400)
energy_bands = np.zeros((2, len(k_array)))  # Only 2 bands for holes

# Colors for Spin-Directions
colors = np.zeros((2, len(k_array), 3))  # RGB color values for each point in the bands

# Pauli matrices for y- and z-directions in the Nambu basis
sigma_y = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
sigma_z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Calculation of energy bands and spin expectation values for each value of k
for i, k in enumerate(k_array):
    H = my_hamiltonian(k/np.sqrt(2*m*Delta), h_tilde, m, alpha_tilde, Delta_tilde, mu_tilde)
    
    # Use scipy.linalg.eigh for better numerical stability
    energies, eigvecs = np.linalg.eigh(H)
    energy_bands[:, i] = energies[2:]  # Only take the last two (highest) energy states (hole states)
    
    # Calculate the spin expectation values for each state
    for j in range(2):  # For each band
        psi = eigvecs[:, j + 2]  # Use the eigenvectors for the hole states
        S_y = np.real(np.vdot(psi, np.dot(sigma_y, psi)))
        S_z = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
        
        # Normalize S_y and S_z to the range [-1, 1]
        S_y_norm = (S_y + 1) / 2
        S_z_norm = (S_z + 1) / 2
        
        # Define colors for interpolation
        color_blue = np.array([0, 0, 1])  # Blue for S_z > 0
        color_red = np.array([1, 0.5, 0])  # Orange for S_z < 0 (instead of Red)
        color_cyan = np.array([0, 1, 0])  # Green for S_y > 0 (instead of Cyan)
        color_magenta = np.array([0.5, 0, 0.5])  # Purple for S_y < 0 (instead of Magenta)

        # Interpolate the colors
        color_z_interp = (1 - S_z_norm) * color_red + S_z_norm * color_blue
        color_y_interp = (1 - S_y_norm) * color_magenta + S_y_norm * color_cyan
        
        # Combine the two interpolations
        colors[j, i] = 0.5 * color_z_interp + 0.5 * color_y_interp

# Plot the energy bands with varying line widths for holes
fig, ax = plt.subplots(figsize=(10, 8))

for j in range(2):  # Only plotting the two hole bands
    for i in range(len(k_array) - 1):
        avg_color = (colors[j, i] + colors[j, i+1]) / 2
        
        # Hole bands (highest two energies)
        ax.plot(k_array[i:i+2]/np.sqrt(2*m*Delta), energy_bands[j, i:i+2], color=avg_color, lw=2)  # Thicker lines
plt.xticks([])
plt.yticks([])
plt.show()
# %%


def my_uncoupled_hamiltonian(k, h, m, alpha, mu):
    H = np.array([
        [k**2/(2*m) - mu + h, 1j*alpha*k],
        [-1j*alpha*k, k**2/(2*m) - mu - h]
    ])
    return H

# Parameters
m = 1
alpha = 3
mu = .0
h = 1
alpha_tilde = alpha/np.sqrt(2*m)
mu_tilde = mu/(2*m)
h_tilde = h/(2*m)
k_array = np.linspace(-8, 8, 2000)
energy_bands = np.zeros((2, len(k_array)))  # Two energy bands (for two spin states)

# Colors for Spin-Directions
colors = np.zeros((2, len(k_array), 3))  # RGB color values for each point in the bands

# Pauli matrices for y- and z-directions
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Calculation of energy bands and spin expectation values for each value of k
for i, k in enumerate(k_array):
    H = my_uncoupled_hamiltonian(k/np.sqrt(2*m), h_tilde, m, alpha_tilde, mu_tilde)
    
    # Use scipy.linalg.eigh for better numerical stability
    energies, eigvecs = np.linalg.eigh(H)
    energy_bands[:, i] = energies  # The two energy states correspond to spin-up and spin-down electrons
    
    # Calculate the spin expectation values for each state
    for j in range(2):  # For each band
        psi = eigvecs[:, j]
        S_y = np.real(np.vdot(psi, np.dot(sigma_y, psi)))
        S_z = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
        
        # Normalize S_y and S_z to the range [-1, 1]
        S_y_norm = (S_y + 1) / 2
        S_z_norm = (S_z + 1) / 2
        
        # Define colors for interpolation
        color_blue = np.array([0, 0, 1])  # Blue for S_z > 0
        color_red = np.array([1, 0, 0])  # Orange for S_z < 0
        color_cyan = np.array([0, 1, .0])  # Green for S_y > 0
        color_magenta = np.array([0.5, 0, 0.5])  # Purple for S_y < 0

        # Interpolate the colors
        color_z_interp = (1 - S_z_norm) * color_red + S_z_norm * color_blue
        color_y_interp = (1 - S_y_norm) * color_magenta + S_y_norm * color_cyan
        
        # Combine the two interpolations
        colors[j, i] = 0.5 * color_z_interp + 0.5 * color_y_interp

# Plot the energy bands with colors representing spin orientation
fig, ax = plt.subplots(figsize=(3,4), dpi=600)
ax.spines['left'].set_position(('data',0))  # Position the y-axis at x = 0
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(-3, 4)
ax.spines['bottom'].set_bounds(k_array.min()/np.sqrt(2*m), k_array.max()/np.sqrt(2*m))
for j in range(2):  # Plotting the two electron bands
    for i in range(len(k_array) - 1):
        avg_color = (colors[j, i] + colors[j, i+1]) / 2
        ax.plot(k_array[i:i+2]/np.sqrt(2*m), energy_bands[j, i:i+2], color=avg_color, lw=1)  # Thicker lines
plt.xticks([])
ax.set_ylim(-3,4)
ax.set_xlim(k_array.min()/np.sqrt(2*m), k_array.max()/np.sqrt(2*m))
plt.yticks([])
plt.axhline(y=mu, c='black', linestyle=(0,{11,28}), alpha=.8, lw=.4)
plt.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.3)
plt.show()

# %%
