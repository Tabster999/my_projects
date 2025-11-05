#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg  # Use scipy's linear algebra functions
#%%
def my_hamiltonian(k, h, m, alpha, Delta, mu):
    H = np.array([[k**2/(2*m) - mu + h, 1j*alpha*k, 0, Delta],[-1j*alpha*k, k**2/(2*m) -  mu - h, -Delta, 0],[0, -Delta, -k**2/(2*m) + mu - h, -1j*alpha*k],[Delta, 0, 1j*alpha*k, -k**2/(2*m) + mu + h]])
    return H
#%%
m = 1
alpha = .3
Delta = .0
mu = 0.0
h = .0


k_array = np.linspace(-1, 1, 400)
energy_bands = np.zeros((4, len(k_array)))

colors = np.zeros((4, len(k_array), 3))  # RGB-Farbwerte für jeden Punkt in den Bändern

# Pauli-Matrizen für y- und z-Richtungen in der Nambu-Basis
sigma_y = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
sigma_z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
def my_uncoupled_hamiltonian(k, h, m, alpha, mu):
    H = np.array([
        [k**2/(2*m) - mu + h, 1j*alpha*k],
        [-1j*alpha*k, k**2/(2*m) - mu - h]
    ])
    return H
# Berechnung der Energiebänder und Spin-Erwartungswerte für jeden Wert von k
for i, k in enumerate(k_array):
    H = my_uncoupled_hamiltonian(k, h, m, alpha, mu)
    
    # Verwenden Sie scipy.linalg.eigh für bessere numerische Stabilität
    energies, eigvecs = scipy.linalg.eigh(H)
    energy_bands[:, i] = energies
    # Berechne die Spin-Erwartungswerte für jeden Zustand
    for j in range(2):  # Für jedes Band
        psi = eigvecs[:, j]
        S_y = np.real(np.vdot(psi, np.dot(sigma_y, psi)))
        S_z = np.real(np.vdot(psi, np.dot(sigma_z, psi)))
        
        # Normiere S_y und S_z auf den Bereich [-1, 1]
        S_y_norm = (S_y + 1) / 2
        S_z_norm = (S_z + 1) / 2
        
        # Definiere Farben für Interpolation
        color_blau = np.array([0, 0, 1])  # Blau für S_z > 0
        color_rot = np.array([1, 0.3, 0])  # Orange für S_z < 0 (anstatt Rot)
        color_cyan = np.array([0, .8, 0])  # Grün für S_y > 0 (anstatt Cyan)
        color_magenta = np.array([0.5, 0, 0.5])  # Lila für S_y < 0 (anstatt Magenta)

        # Interpolation der Farben
        color_z_interp = (1 - S_z_norm) * color_rot + S_z_norm * color_blau
        color_y_interp = (1 - S_y_norm) * color_magenta + S_y_norm * color_cyan
        
        # Kombination der beiden Interpolationen
        colors[j, i] = 0.4 * color_z_interp + 0.6 * color_y_interp

# Plot der Energiebänder mit unterschiedlicher Linienbreite für Elektronen und Löcher
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)

for j in range(4):
    for i in range(len(k_array) - 1):
        avg_color = (colors[j, i] + colors[j, i+1]) / 2
        
        # Unterscheidung zwischen Elektronen- und Löcherbändern
        if j < 2:  # Elektronenbänder (niedrigste beiden Energien)
            ax.plot(k_array[i:i+2], energy_bands[j, i:i+2], color=avg_color, lw=2)  # Dickere Linien
        else:  # Löcherbänder (höchste beiden Energien)
            ax.plot(k_array[i:i+2], energy_bands[j, i:i+2], color=avg_color, lw=2)  # Dünnere, gestrichelte Linien

ax.set_ylim(-0,.4)
ax.set_facecolor('lightyellow')
plt.xticks([])
plt.yticks([])
plt.show()


# %%
def parabola(k):
    par = k**2
    return par
k_array = np.linspace(-20,20,300)
plus = []
minus = []

for k in k_array:
    minus.append(parabola(k+3))
    plus.append(parabola(k-3))


plt.figure(figsize=(8,8), dpi=600)
plt.tight_layout()
plt.plot(k_array, plus, c='blue')
plt.plot(k_array, minus, c='red')
plt.xlim(-23,23)
# %%
