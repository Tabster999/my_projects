#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, least_squares
from scipy.signal import find_peaks
import os
import sys
import glob
import re
from scipy.spatial.transform import Rotation as R
#%% initialize data
theta_folder = r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Theta'
phi_folder = r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Phi'
B_magnitude = 0.002 #in Tesla 

mu_B = 9.274009994e-24      # J/T
h = 6.62607015e-34          # J s
g_e = 2.0028                # electron g-factor for NV electron
gamma_e = g_e * mu_B/h     # Hz / T
D = 2.87e9                  #Hz
# |m=+1,0,-1> basis
Sx = np.array([[0, np.sqrt(1/2), 0],
               [np.sqrt(1/2), 0, np.sqrt(1/2)],
               [0, np.sqrt(1/2), 0]], dtype=float)
Sy = np.array([[0, -1j*np.sqrt(1/2), 0],
               [1j*np.sqrt(1/2), 0, -1j*np.sqrt(1/2)],
               [0, 1j*np.sqrt(1/2), 0]], dtype=complex)
Sz = np.diag([1.0, 0.0, -1.0])

Sx = Sx.real
Sz = Sz.real
S_vec = np.array([Sx, Sy, Sz])

NV_orientations = (1/np.sqrt(3))*np.array([
    [ 1,  1,  1],
    [-1,  1,  1],
    [ 1, -1,  1],
    [ 1,  1, -1]
], dtype=float)
#%% define functions
r'''
x = r cos(\phi) sin(\theta)
y = r sin(\phi) sin(\theta)
z = r cos(\theta)
for theta measurements phi=90°
for phi measurements theta=90°
'''

def get_hamiltonian(B_vec, n_vec, S_vec=S_vec, D=D, E=0):
    '''
    B_vec: Magnetic field vector in spherical coordinates
    n_vec: NV orientation in spherical coordinates
    '''
    gamma_e = 28031679103.16072 #Hz/T
    nx, ny, nz = n_vec
    Sx, Sy, Sz = S_vec
    S_dot_n = nx*Sx+ny*Sy+nz*Sz
    
    H_zfs = D*(S_dot_n @ S_dot_n) 
    H_z = gamma_e * (B_vec[0]*Sx+B_vec[1]*Sy+B_vec[2]*Sz) #Hz
    
    H = H_zfs + H_z
    
    return np.array(H, dtype=np.complex128)

def get_transition_frequencies(B_vec, n_vec):
    
    ham = get_hamiltonian(B_vec=B_vec, n_vec=n_vec)
    evals = np.sort(np.linalg.eigvalsh(ham))
    return np.array([evals[1] - evals[0], evals[2] - evals[0]])
    
def minimize_angles(params, B_mag, thetas, phis, freq_low, freq_high, D=D, E=0):
    
    theta, phi = params
    n_vec = np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)])
    freq_low_model, freq_high_model = get_transition_frequencies(n_vec, B_mag, thetas, phis, D=D, E=E)


    resid_low = (freq_low_model - freq_low)
    resid_high = (freq_high_model - freq_high)
    
    return np.concatenate([resid_low, resid_high])    
    
#%% plotting theoretical spectra for theta sweep
thetas_model1 = np.linspace(0,np.pi,200)
phis_model1 = np.pi/2*np.ones_like(thetas_model1)

fig, ax = plt.subplots()
for n_vec in NV_orientations:
    freq1 = []
    freq2 = []
    
    for theta, phi in zip(thetas_model1, phis_model1):
        B_vec = B_magnitude*np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        f1, f2 = get_transition_frequencies(B_vec, n_vec)
        freq1.append(f1)
        freq2.append(f2)
    ax.plot(np.rad2deg(thetas_model1), freq1, label=f'NV {n_vec}')
    ax.plot(np.rad2deg(thetas_model1), freq2, linestyle='--')
    
ax.set_xlabel(r'$\theta$ (deg, yz-plane)')
ax.set_ylabel('Frequency (GHz)')
ax.set_title(r'ODMR vs $\theta$ for all NV orientations')
#ax.legend()
plt.show()
#%% plotting theoretical spectra for phi sweep
# --- Sweep phi in xy-plane (theta=90°) ---
phis_model2 = np.linspace(0, np.pi, 180)
thetas_model2 = np.pi/2*np.ones_like(phis_model2)

fig, ax = plt.subplots()
for n_vec in NV_orientations:
    freq1, freq2 = [], []
    for th, ph in zip(thetas_model2, phis_model2):
        B = B_magnitude*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        f1, f2 = get_transition_frequencies(B, n_vec)
        freq1.append(f1/1e9)
        freq2.append(f2/1e9)
    ax.plot(np.rad2deg(phis_model2), freq1, label=f'NV {n_vec}')
    ax.plot(np.rad2deg(phis_model2), freq2, linestyle='--')

ax.set_xlabel(r'$\phi$ (deg, xy-plane)')
ax.set_ylabel('Frequency (GHz)')
ax.set_title(r'ODMR vs $\phi$ for all NV orientations')
#ax.legend(loc=0)
plt.show()
#%% extracting the angle measurments for theta 
folder_theta = r"C:\Main\my_projects\F-Praktikum\my_odmr_data\Theta"


files_theta = sorted(glob.glob(os.path.join(folder_theta, "*.dat")),
               key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))


data_ref_theta = np.loadtxt(files_theta[0], skiprows=3)
freq_common_theta = data_ref_theta[:,0] / 1e9

ratios_thetas = []
thetas = []

for f in files_theta:
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    angle = (number - 85) * 10  # 66 for phi, 85 for theta
    thetas.append(angle)
    
    data = np.loadtxt(f, skiprows=3)
    freq = data[:,0] 
    ratio = data[:,4]
    
    ratios_thetas.append(ratio)


ratios_theta = np.array(ratios_thetas).T  # shape: (n_freqs, n_angles)
thetas = np.array(thetas)

# %% extracting the angle measurments for phi
folder_phi = r"C:\Main\my_projects\F-Praktikum\my_odmr_data\Phi"
files_phi = sorted(glob.glob(os.path.join(folder_phi, "*.dat")),
                   key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

data_ref_phi = np.loadtxt(files_phi[0], skiprows=3)
freq_common_phi = data_ref_phi[:,0] # Keep in Hz for now

ratios_phi = []
phis = []

for f in files_phi:
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    angle = (number - 66) * 10
    phis.append(angle)
    data = np.loadtxt(f, skiprows=3)
    ratios_phi.append(data[:,4])

ratios_phi = np.array(ratios_phi).T
phis = np.array(phis)
# %% extracting the peaks of both spectra using scipy.optimize.find_peaks
all_peak_freqs_phi = []
all_peak_freqs_theta = []

for i in range(len(thetas)):
    spectrum_to_peaks = 1 - ratios_theta.T[:, i]  
    peaks_indices = find_peaks(spectrum_to_peaks, height=.01, distance=1, prominence=.005)
    peaks_freqs = freq_common_theta[peaks_indices[0]]
    all_peak_freqs_theta.append(peaks_freqs)

for i in range(len(phis)):
    spectrum_to_peaks = 1 - ratios_phi[:, i]  
    peaks_indices = find_peaks(spectrum_to_peaks, height=.01, distance=1, prominence=.005)
    peaks_freqs = freq_common_phi[peaks_indices[0]]
    all_peak_freqs_phi.append(peaks_freqs)

# peak1 = [p1[0] for p1 in all_peak_freqs_theta]
# peak2 = [p2[1] for p2 in all_peak_freqs_theta]

# dummy_angles = np.linspace(0,180,19)

# plt.plot(dummy_angles, peak1, label='1')
# plt.plot(dummy_angles, peak2, label='2')
# plt.legend()
#%% use scipy.optimize.least_squares to solve for the orientations
n1_vec, n2_vec, n3_vec, n4_vec = NV_orientations[0], NV_orientations[1], NV_orientations[2], NV_orientations[3]
B_vec = B_magnitude*np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

f_low_model, f_high_model = get_transition_frequencies(B_vec=B_vec, n_vec=n_vec)

# %% testing the simple approach
manual_euler_angles = [133.0, 29.0, 116.0]
colors = ['red', 'cyan', 'lime', 'magenta'] 
rotation = R.from_euler('xyz', manual_euler_angles, degrees=True)
nv_axes_in_lab_frame = rotation.apply(NV_orientations)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
fig.suptitle(f'Manual Fit for Angles: a={manual_euler_angles[0]}°, b={manual_euler_angles[1]}°, g={manual_euler_angles[2]}°', fontsize=16)

ax1.set_title('Theta Sweep (yz-plane)')
if 'thetas' in locals() and thetas is not None:
    ax1.contourf(thetas, freq_common_theta, ratios_theta, levels=50, cmap='viridis')
    
    angles_model = np.linspace(0, 180, 181)
    for i, n_vec in enumerate(nv_axes_in_lab_frame):
        freq1, freq2 = [], []
        for th_deg in angles_model:
            th_rad = np.deg2rad(th_deg)
            # B-field rotates in yz-plane, so phi is fixed at 90°
            B_vec = B_magnitude * np.array([0, np.sin(th_rad), np.cos(th_rad)])
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            freq1.append(f1)
            freq2.append(f2)
        ax1.plot(angles_model, np.array(freq1)/1e9, color=colors[i], linewidth=1.5, linestyle='--')
        ax1.plot(angles_model, np.array(freq2)/1e9, color=colors[i], linewidth=1.5, linestyle='--')

ax2.set_title('Phi Sweep (xy-plane)') # Use ax2
if 'phis' in locals() and phis is not None:
    # Use ax2 for the contour plot
    ax2.contourf(phis, freq_common_phi/1e9, ratios_phi, levels=50, cmap='viridis')
    
    angles_model = np.linspace(0, 180, 181)
    for i, n_vec in enumerate(nv_axes_in_lab_frame):
        freq1, freq2 = [], []
        for ph_deg in angles_model:
            ph_rad = np.deg2rad(ph_deg)
            B_vec = B_magnitude * np.array([np.cos(ph_rad), np.sin(ph_rad), 0])
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            freq1.append(f1)
            freq2.append(f2)
        # Use ax2 for the line plots
        ax2.plot(angles_model, np.array(freq1)/1e9, color=colors[i], linewidth=1.5, linestyle='--')
        ax2.plot(angles_model, np.array(freq2)/1e9, color=colors[i], linewidth=1.5, linestyle='--')
# %%
def fit_cost_function(euler_angles_rad, experimental_data, B_mag):
    """
    Calculates the total squared error between the experimental data and the
    theoretical model for a given set of Euler angles.
    """
    rotation = R.from_euler('xyz', euler_angles_rad, degrees=False)
    nv_axes_lab = rotation.apply(NV_orientations)
    total_squared_error = 0.0

    for measurement in experimental_data:
        angle_rad = np.deg2rad(measurement['angle_deg'])
        
        if measurement['sweep_type'] == 'phi':
            B_vec = B_mag * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        else: # 'theta'
            # CORRECTED: Was incorrectly using a non-existent 'th_rad'.
            # Now correctly uses 'angle_rad' for the theta sweep calculation.
            B_vec = B_mag * np.array([0, np.sin(angle_rad), np.cos(angle_rad)])
        
        # Calculate the 8 possible theoretical frequencies for this angle
        theoretical_freqs = []
        for n_vec in nv_axes_lab:
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            theoretical_freqs.extend([f1, f2])
        
        # Compare every experimental peak to the closest theoretical one
        for peak in measurement['peaks']:
            min_diff = np.min(np.abs(peak - np.array(theoretical_freqs)))
            total_squared_error += min_diff**2
            
    return total_squared_error

# --- 1. Combine all your experimental data into one list ---
# (This requires you to have run the peak extraction cell first)
all_experimental_peaks = []
for angle, peaks in zip(thetas, all_peak_freqs_theta):
    # Convert peaks from GHz back to Hz for the cost function
    all_experimental_peaks.append({'angle_deg': angle, 'sweep_type': 'theta', 'peaks': peaks * 1e9})

for angle, peaks in zip(phis, all_peak_freqs_phi):
    # Convert peaks from GHz back to Hz for the cost function
    all_experimental_peaks.append({'angle_deg': angle, 'sweep_type': 'phi', 'peaks': peaks * 1e9})
    
# --- 2. Set your manual guess as the starting point ---
# ❗ACTION: Put your best manually-found angles here
manual_guess_deg = [120.0, 20.0, 100.0]
initial_guess_rad = np.deg2rad(manual_guess_deg)

print(f"Starting automated fit with initial guess: {manual_guess_deg}...")

# --- 3. Run the optimization ---
tighter_tolerances = {'xatol':1e-8, 'fatol':1e-8} 
result = minimize(
    fit_cost_function,
    initial_guess_rad,
    args=(all_experimental_peaks, B_magnitude), # Extra arguments for the function
    method='Nelder-Mead', options=tighter_tolerances
)

# --- 4. Print the final, optimized angles ---
best_euler_angles_deg = np.rad2deg(result.x)

print("\n--- ✅ Automated Fit Complete ---")
if result.success:
    print("The optimization was successful.")
    print(f"The best-fit Euler Angles are:")
    print(f"  Alpha = {best_euler_angles_deg[0]:.3f}°")
    print(f"  Beta  = {best_euler_angles_deg[1]:.3f}°")
    print(f"  Gamma = {best_euler_angles_deg[2]:.3f}°")
else:
    print("❌ The optimization did not converge successfully.")
    print(result.message)
# %%
