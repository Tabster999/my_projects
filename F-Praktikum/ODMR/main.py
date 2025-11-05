#%% Imports
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.signal import find_peaks
import os
import glob
import re
from scipy.spatial.transform import Rotation as R
import scipy.constants as cons
from scipy.stats import linregress
#%% initialize data
# --- Physical Constants and Spin Matrices ---
D = 2.87e9  # Zero-field splitting in Hz
g_e = 2.0028
mu_B = 9.274e-24 # J/T
h = 6.626e-34    # J*s
gamma_e = g_e * mu_B / h # Hz/T

# |m_s = +1, 0, -1> basis for S=1 spin
Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
Sz = np.diag([1.0, 0.0, -1.0])
S_vec = np.array([Sx, Sy, Sz])

# Standard NV orientations in the crystal frame
NV_orientations = (1/np.sqrt(3))*np.array([
    [ 1,  1,  1],
    [-1, 1, 1],
    [ 1, -1,  1],
    [ 1,  1, -1]
], dtype=float)

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20

#%% define functions
def get_hamiltonian(B_vec, n_vec):
    """Calculates the spin Hamiltonian for a given B-field and NV orientation."""
    Sx, Sy, Sz = S_vec
    H_zeeman = gamma_e * (B_vec[0]*Sx + B_vec[1]*Sy + B_vec[2]*Sz)
    S_dot_n = n_vec[0]*Sx + n_vec[1]*Sy + n_vec[2]*Sz
    H_zfs = D * (S_dot_n @ S_dot_n)
    return H_zfs + H_zeeman

def get_transition_frequencies(B_vec, n_vec):
    """
    Calculates the two ODMR transition frequencies by finding the eigenvalues.
    """
    ham = get_hamiltonian(B_vec=B_vec, n_vec=n_vec)
    evals = np.sort(np.linalg.eigvalsh(ham))
    # CORRECTED: Transitions are the energy differences from the lowest state (evals[0])
    return evals[1] , evals[2]
#%%
h = 6.62607015e-34      # Planck [J*s]
mu_B = 9.274009994e-24  # Bohr magneton [J/T]
theta = np.deg2rad(40)  # angle in radians

# --- Map B-field values (in Tesla) to your spectra ---
B_map = {
    "B=-3mT": -0.003,
    "B=-2mT": -0.002,
    "B=-1mT": -0.001,
    "B=1mT":   0.001,
    "B=2mT":   0.002,
    "B=3mT":   0.003,
}

# %%
try:
    path = r'C:\Main\my_projects\F-Praktikum\my_odmr_data'
    zeeman_dat1 = np.loadtxt(os.path.join(path, 'Measurement_106.dat'), skiprows=4) # -3mT
    zeeman_dat2 = np.loadtxt(os.path.join(path, 'Measurement_107.dat'), skiprows=4) # -2mT
    zeeman_dat3 = np.loadtxt(os.path.join(path, 'Measurement_108.dat'), skiprows=4) # -1mT
    zeeman_dat4 = np.loadtxt(os.path.join(path, 'Measurement_109.dat'), skiprows=4) # +1mT
    zeeman_dat5 = np.loadtxt(os.path.join(path, 'Measurement_110.dat'), skiprows=4) # +2mT
    zeeman_dat6 = np.loadtxt(os.path.join(path, 'Measurement_111.dat'), skiprows=4) # +3mT
except FileNotFoundError:
    print("Error: One or more data files were not found. Please check the file path.")
    # Create dummy data so the script can still run
    zeeman_dat1 = zeeman_dat2 = zeeman_dat3 = zeeman_dat4 = zeeman_dat5 = zeeman_dat6 = np.random.rand(10, 5)

# --- 2. Organize the data into lists for easy looping ---
fz_list = [
    zeeman_dat1[:,0], zeeman_dat2[:,0], zeeman_dat3[:,0],
    zeeman_dat4[:,0], zeeman_dat5[:,0], zeeman_dat6[:,0]
]
rz_list = [
    zeeman_dat1[:,4], zeeman_dat2[:,4], zeeman_dat3[:,4],
    zeeman_dat4[:,4], zeeman_dat5[:,4], zeeman_dat6[:,4]
]
titles_list = ['B=-3mT', 'B=-2mT', 'B=-1mT', 'B=1mT', 'B=2mT', 'B=3mT']
all_peaks = []
# --- 3. Create the 2x3 subplot grid ---
fig, axs = plt.subplots(2, 3, figsize=(20, 12))

# --- 4. Loop through each spectrum to find peaks, plot, and annotate ---
for ax, fz, rz, title in zip(axs.flat, fz_list, rz_list, titles_list):
    
    # Plot the full ODMR spectrum line
    ax.plot(fz / 1e9, rz, label='ODMR Data')
    
    # Find the peaks (dips) in the intensity data.
    # You may need to adjust 'prominence' for your data's noise level.
    peak_indices, _ = find_peaks(1-rz, prominence=0.008, height=0.003)
    
    # Get the frequency and intensity values at the peak locations
    peak_freqs = fz[peak_indices]
    peak_intensities = rz[peak_indices]
    all_peaks.append(peak_freqs)
    # --- Loop through each found peak to plot a marker and add a label ---
    for freq, intensity in zip(peak_freqs, peak_intensities):
        freq_ghz = freq / 1e9
        
        # Plot a single marker on the peak
        ax.plot(freq_ghz, intensity, 'o', color='red', markersize=8)
        
        # Create the text label
        label = f"{freq_ghz:.3f} GHz"
        
        # Add the text label slightly above the marker
        # You can adjust the y_offset for better placement
        y_offset = 0.000 
        ax.text(freq_ghz, intensity + y_offset, label, ha='center', va='bottom', fontsize=10)
    
    # --- Formatting for each subplot ---
    ax.set_title(title)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Intensity Ratio (a.u.)')
    ax.grid(True)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout()
plt.show()
# %%
# --- Physical Constants ---
h = 6.62607015e-34  # J*s
mu_B = 9.274009994e-24 # J/T


x_pos = np.array([0.0,0.001**2,0.002**2,0.003**2])
y_pos = np.array([(0.00667*1e9)**2,(0.024*1e9)**2,(0.041*1e9)**2,(0.062*1e9)**2])

x_neg = -np.array([0.003,0.002,0.001,0.0])**2
y_neg = -np.array([(0.055e9),(0.041e9),(0.021*1e9),(0.00667*1e9)])**2

#%% Perform Fits and Calculate g-factors
# =============================================================================

# --- Fit 1: Inner Peaks --
theta = np.deg2rad(40)
fit_pos = linregress(x_pos, y_pos)
slope_pos = fit_pos.slope
g_pos = (np.sqrt(slope_pos) * h) / (mu_B*np.cos(theta))

fit_neg = linregress(x_neg, y_neg)
slope_neg = fit_neg.slope
g_neg = (np.sqrt(slope_neg) * h) / (mu_B*np.cos(theta))

print("--- Fit Results ---")
print(f"g-factor calculated from pos: {g_pos:.4f}")
print(f"g-factor calculated from neg: {g_neg:.4f}")


#%% Plotting
# =============================================================================
plt.figure(figsize=(10, 7))

# Plot the squared data points
plt.scatter(x_pos, y_pos, label='pos Data', color='blue', marker='o')
plt.scatter(x_neg, y_neg, label='neg Data', color='blue', marker='o')
# Plot the linear fit lines
plt.plot(x_pos, fit_pos.intercept + slope_pos * x_pos, 
         color='blue', linestyle='--', label=f'pos (g_eff = {g_pos:.2f})')
plt.plot(x_neg, fit_neg.intercept + slope_neg * x_neg, 
         color='blue', linestyle='--', label=f'neg (g_eff = {g_pos:.2f})')

plt.xlabel(r'Magnetic Field Squared ($B^2$) [$T^2$]')
plt.ylabel(r'Frequency Splitting Squared ($\Delta\nu^2$) [$Hz^2$]')
plt.title(r'Linearized ODMR Data')
plt.legend()
plt.grid(True)
plt.show()
# %%
