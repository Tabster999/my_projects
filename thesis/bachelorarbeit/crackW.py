#%%
import numpy as np
import matplotlib.pyplot as plt

# Funktion zur Berechnung des Hamiltonians
def my_hamiltonian(k, h, m, alpha, Delta, mu):
    H = np.array([[k**2/(2*m) - mu + h, 1j*alpha*k, 0, Delta],[-1j*alpha*k, k**2/(2*m) -  mu - h, -Delta, 0],[0, -Delta, -k**2/(2*m) + mu - h, -1j*alpha*k],[Delta, 0, 1j*alpha*k, -k**2/(2*m) + mu + h]])
    return H

# %%
m = 1
alpha = 0.2
Delta = .1
mu = .3
h = .4

Delta_tilde = 0 # We normalize everything with respect to Delta
mu_tilde = mu / Delta
alpha_tilde = alpha*np.sqrt(2*m/ Delta  ) # Note the correction here
h_tilde = h / Delta

k_array = np.linspace(-10, 10, 600)
energy_bands = np.zeros((4, len(k_array)))

# Pauli-Matrizen für y- und z-Richtungen in der Nambu-Basis
sigma_y = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
sigma_z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Erstellung einer Farbfunktion
def interpolate_colors(S_y_norm, S_z_norm):
    # Farben für S_z
    color_blau = np.array([0, 0, 1])  # Blau für S_z > 0
    color_rot = np.array([1, .5, 0])   # Rot für S_z < 0
    # Farben für S_y
    color_cyan = np.array([0, 1, 1])  # Cyan für S_y > 0
    color_magenta = np.array([.8, 0, .8])  # Magenta für S_y < 0
    
    # Interpolationen zwischen den Farben
    color_z_interp = (1 - S_z_norm) * color_rot + S_z_norm * color_blau
    color_y_interp = (1 - S_y_norm) * color_magenta + S_y_norm * color_cyan
    
    # Rückgabe eines Durchschnitts der beiden Interpolationen
    return 0.5 * color_z_interp + 0.5 * color_y_interp

# Berechnung der Energiebänder und Spin-Erwartungswerte für jeden Wert von k
colors = np.zeros((4, len(k_array), 3))  # RGB-Farbwerte für jeden Punkt in den Bändern
for i, k in enumerate(k_array):
    H = my_hamiltonian(k/np.sqrt(2*m*Delta), h_tilde, m, alpha_tilde, Delta_tilde, mu_tilde)
    
    # Verwenden Sie scipy.linalg.eigh für bessere numerische Stabilität
    energies, eigvecs = np.linalg.eigh(H)
    energy_bands[:, i] = energies
    
    # Berechne die Spin-Erwartungswerte für jeden Zustand
    for j in range(4):  # Für jedes Band
        psi = eigvecs[:, j]
        S_y = np.real(np.vdot(psi, np.dot(sigma_y, psi)))
        S_z = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
        
        # Normiere S_y und S_z auf den Bereich [0, 1] für die Interpolation
        S_y_norm = (S_y + 1) / 2
        S_z_norm = (S_z + 1) / 2
        
        # Verwende die Farbfunktion zur Interpolation
        colors[j, i] = interpolate_colors(S_y_norm, S_z_norm)

# Plot der Energiebänder mit mehr Farbverläufen
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

for j in range(4):
    for i in range(len(k_array) - 1):
        avg_color = (colors[j, i] + colors[j, i+1]) / 2
        
        # Unterscheidung zwischen Elektronen- und Löcherbändern
        if j < 2:  # Elektronenbänder (niedrigste beiden Energien)
            ax.plot(k_array[i:i+2]/np.sqrt(2*m*Delta), energy_bands[j, i:i+2], color=avg_color, lw=1.2)  # Dickere Linien
        else:  # Löcherbänder (höchste beiden Energien)
            ax.plot(k_array[i:i+2]/np.sqrt(2*m*Delta), energy_bands[j, i:i+2], color=avg_color, lw=2.5)  # Dünnere, gestrichelte Linien

ax.set_ylim(-12,12)
ax.set_xlim(-7,7)
ax.set_facecolor('lightyellow')
plt.yticks([])
plt.xticks([])
plt.show()

# %%
