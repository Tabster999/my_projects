#%% Imports
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
import os
import glob
import re
from scipy.spatial.transform import Rotation as R
import scipy.constants as cons
from scipy.stats import linregress
import matplotlib.markers as markers
#%% initialize data
theta_folder = r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Theta'
phi_folder = r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Phi'
B_magnitude = 0.002 #in Tesla 

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

def linear_function(x, m, c):
    return m*x+c
#%% extracting the angle measurments for theta 
folder_theta = r"C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Theta"
files_theta = sorted(glob.glob(os.path.join(folder_theta, "*.dat")),
                     key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

data_ref_theta = np.loadtxt(files_theta[0], skiprows=3)
freq_common_theta = data_ref_theta[:,0] # Keep in Hz for consistency

ratios_list = []
thetas = []

for f in files_theta:
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    angle = (number - 85) * 10
    thetas.append(angle)
    data = np.loadtxt(f, skiprows=3)
    ratios_list.append(data[:,4])

ratios_theta = np.array(ratios_list).T
thetas = np.array(thetas)
print("Theta data loaded.")

# %% extracting the angle measurments for phi
# CORRECTED: This entire cell was fixed to use the correct phi variables
folder_phi = r"C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Phi"
files_phi = sorted(glob.glob(os.path.join(folder_phi, "*.dat")),
                   key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

data_ref_phi = np.loadtxt(files_phi[0], skiprows=3)
freq_common_phi = data_ref_phi[:,0] # Keep in Hz for consistency

ratios_list = []
phis = []

for f in files_phi:
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    angle = (number - 66) * 10
    phis.append(angle)
    data = np.loadtxt(f, skiprows=3)
    ratios_list.append(data[:,4])

ratios_phi = np.array(ratios_list).T
phis = np.array(phis)
print("Phi data loaded.")

# %% extracting the peaks of both spectra using scipy.signal.find_peaks
all_peak_freqs_phi = []
all_peak_freqs_theta = []

print("Finding peaks in experimental data...")
for i in range(len(thetas)):
    # CORRECTED: The indexing was wrong. This now correctly gets the i-th spectrum.
    spectrum_to_peaks = 1 - ratios_theta[:, i]
    peaks_indices, _ = find_peaks(spectrum_to_peaks, height=0.01, prominence=0.005)
    peaks_freqs = freq_common_theta[peaks_indices]
    all_peak_freqs_theta.append(peaks_freqs)

for i in range(len(phis)):
    # CORRECTED: The indexing was wrong. This now correctly gets the i-th spectrum.
    spectrum_to_peaks = 1 - ratios_phi[:, i]
    peaks_indices, _ = find_peaks(spectrum_to_peaks, height=0.01, prominence=0.005)
    peaks_freqs = freq_common_phi[peaks_indices]
    all_peak_freqs_phi.append(peaks_freqs)
print("Peak finding complete.")

#%% Manual Fitting Plot
# ===========================================================================
manual_euler_angles = [132.3, 172.881, 148.118]

colormap = 'viridis'
colors = ['red', 'magenta', 'lime', 'cyan']
crystal_axes_labels = ['[111]','[-111]', '[1-11]', '[11-1]'] 
linewidths = [1.5,1,1.5,1.5]
rotation = R.from_euler('xyz', manual_euler_angles, degrees=True)
nv_axes_in_lab_frame = rotation.apply(NV_orientations)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
#fig.suptitle(f': a={manual_euler_angles[0]}°, b={manual_euler_angles[1]}°, g={manual_euler_angles[2]}°', fontsize=16)

ax1.set_title('Theta Sweep (yz-plane)')
if 'thetas' in locals():
    ax1.contourf(thetas, freq_common_theta / 1e9, ratios_theta, levels=50, cmap=colormap)
    angles_model = np.linspace(0, 180, 181)
    for i, n_vec in enumerate(nv_axes_in_lab_frame):
        freq1, freq2 = [], []
        for th_deg in angles_model:
            th_rad = np.deg2rad(th_deg)
            B_vec = B_magnitude * np.array([0, np.sin(th_rad), np.cos(th_rad)])
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            freq1.append(f1)
            freq2.append(f2)
        ax1.plot(angles_model, np.array(freq1)/1e9, color=colors[i], linewidth=linewidths[i], linestyle='--', label=crystal_axes_labels[i])
        ax1.plot(angles_model, np.array(freq2)/1e9, color=colors[i], linewidth=linewidths[i], linestyle='--')
        ax1.tick_params(which="both", direction="in", top=True, right=True)

        #ax1.axvline(x=147)

    ax1.legend()
ax1.set_ylim(2.8,2.95)

ax2.set_title('Phi Sweep (xy-plane)')
if 'phis' in locals():
    ax2.contourf(phis, freq_common_phi / 1e9, ratios_phi, levels=50, cmap=colormap)
    angles_model = np.linspace(0, 180, 181)
    for i, n_vec in enumerate(nv_axes_in_lab_frame):
        freq1, freq2 = [], []
        for ph_deg in angles_model:
            ph_rad = np.deg2rad(ph_deg)
            B_vec = B_magnitude * np.array([np.cos(ph_rad), np.sin(ph_rad), 0])
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            freq1.append(f1)
            freq2.append(f2)
        ax2.plot(angles_model, np.array(freq1)/1e9, color=colors[i], linewidth=linewidths[i], linestyle='--', label=crystal_axes_labels[i])
        ax2.plot(angles_model, np.array(freq2)/1e9, color=colors[i], linewidth=linewidths[i], linestyle='--')
        #ax2.axvline(x=53)
    ax2.legend()
    ax2.tick_params(which="both", direction="in", top=True, right=True)

for ax in [ax1, ax2]:
    ax.set_ylabel("Frequency [GHz]")
ax2.set_xlabel("Angle [deg]")
ax2.set_ylim(2.8,2.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% Automated Fitter

def fit_cost_function(euler_angles_rad, experimental_data, B_mag):
    rotation = R.from_euler('xyz', euler_angles_rad, degrees=False)
    nv_axes_lab = rotation.apply(NV_orientations)
    total_squared_error = 0.0

    for measurement in experimental_data:
        angle_rad = np.deg2rad(measurement['angle_deg'])
        if measurement['sweep_type'] == 'phi':
            B_vec = B_mag * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        else: # 'theta'
            B_vec = B_mag * np.array([0, np.sin(angle_rad), np.cos(angle_rad)])
        
        theoretical_freqs = []
        for n_vec in nv_axes_lab:
            f1, f2 = get_transition_frequencies(B_vec, n_vec)
            theoretical_freqs.extend([f1, f2])
        
        for peak in measurement['peaks']:
            min_diff = np.min(np.abs(peak - np.array(theoretical_freqs)))
            total_squared_error += min_diff**2
    return total_squared_error

all_experimental_peaks = []
for angle, peaks in zip(thetas, all_peak_freqs_theta):
    all_experimental_peaks.append({'angle_deg': angle, 'sweep_type': 'theta', 'peaks': peaks})

for angle, peaks in zip(phis, all_peak_freqs_phi):
    all_experimental_peaks.append({'angle_deg': angle, 'sweep_type': 'phi', 'peaks': peaks})
    
manual_guess_deg = [132.0, 8.0, 140.0]
initial_guess_rad = np.deg2rad(manual_guess_deg)

print(f"\nStarting automated fit with initial guess: {manual_guess_deg}...")

tighter_tolerances = {'xatol':1e-6, 'fatol':1e-6} 
result = minimize(
    fit_cost_function,
    initial_guess_rad,
    args=(all_experimental_peaks, B_magnitude),
    method='Nelder-Mead', 
    options=tighter_tolerances
)

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
# %% zeeman splitting by plotting the 6 spectra for each B value and extracting the peaks positions 
zeeman_dat1 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_106.dat', skiprows=4)
zeeman_dat2 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_107.dat', skiprows=4)
zeeman_dat3 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_108.dat', skiprows=4)
zeeman_dat4 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_109.dat', skiprows=4)
zeeman_dat5 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Measurement_110.dat', skiprows=4)
zeeman_dat6 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Measurement_111.dat', skiprows=4)


# %%
zf1 = zeeman_dat1[:,0]#
zf2 = zeeman_dat2[:,0]
zf3 = zeeman_dat3[:,0]
zf4 = zeeman_dat4[:,0]
zf5 = zeeman_dat5[:,0]
zf6 = zeeman_dat6[:,0]

rz1 = zeeman_dat1[:,4]
rz2 = zeeman_dat2[:,4]
rz3 = zeeman_dat3[:,4]
rz4 = zeeman_dat4[:,4]
rz5 = zeeman_dat5[:,4]
rz6 = zeeman_dat6[:,4]

fz_list = [zf1, zf2, zf3, zf4, zf5, zf6]
rz_list  = [rz1,rz2,rz3,rz4,rz5,rz6]

x_points_list = [
    np.array([2.806, 2.848, 2.903, 2.936]),
    np.array([2.827, 2.852, 2.893, 2.912]),
    np.array([2.859, 2.88]),
    np.array([2.858, 2.882]),
    np.array([2.829, 2.85, 2.891, 2.909]),
    np.array([2.808, 2.843, 2.905, 2.934])
]

y_points_list = [
    np.array([0.989164, 0.9841268, 0.98334112, 0.9883762]),
    np.array([0.98676656, 0.98151514, 0.97974574, 0.98816407]),
    np.array([0.97605931, 0.97555524]),
    np.array([0.97169201, 0.97158712]),
    np.array([0.98888627, 0.9770377, 0.97679165, 0.98815664]),
    np.array([0.98918671, 0.97962307, 0.98131049, 0.98831948])
]
# %% played around and got this plot for zeeman analysis
fig, axs = plt.subplots(2, 3, figsize=(18, 10)) # Increased figure width for labels

# Loop through each subplot and its corresponding data
for ax, fz, rz in zip(axs.flat, fz_list, rz_list):
    # Plot the original line data
    ax.plot(fz / 1e9, rz, color='gray')
    
    # --- Find Peaks ---
    # You may need to adjust the 'prominence' value for your data
    peak_indices, _ = find_peaks(1-rz, prominence=0.005, height=0.005)
    
    # Get the coordinates of all peaks found in this subplot
    peak_frequencies = fz[peak_indices]
    peak_intensities = rz[peak_indices]
    
    # --- NEW: Loop through each peak to plot and label it individually ---
    
    # Generate a unique color for each peak from the 'viridis' colormap
    num_peaks = len(peak_frequencies)
    colors = plt.cm.viridis(np.linspace(0, 1, num_peaks))
    
    for i in range(num_peaks):
        # Get the coordinates and color for the current peak
        freq_ghz = peak_frequencies[i] / 1e9
        intensity = peak_intensities[i]
        
        # Plot the single peak as a colored circle
        ax.plot(freq_ghz, intensity, 'o', color=colors[i], markersize=8)
        
        # Add a text label with the frequency value above the point
        label = f"{freq_ghz:.3f}" # Format frequency to 3 decimal places
        ax.text(freq_ghz, intensity + 0.001, label, ha='center', va='bottom', fontsize=9)

    # --- Formatting ---
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Intensity [a.u.]')
    ax.grid(True)
    ax.axvline(x=2.843)

plt.tight_layout()
plt.show()
#%%
capsize = 4
markersize =  10
markeredgewidth = 2
h = 6.62607015e-34  # J*s
mu_B = 9.274009994e-24 # J/T
nu_err = 0.005e9
prefactor = np.sqrt(8)
# --- 1. Filtered Experimental Data (B >= 0, Inner Peaks Only) ---
b_fields_fit = np.array([0.0,0.001, 0.002, 0.003])
theta = 61.97
# Corresponding frequency splittings (Δν) in GHz
inner_splittings_hz = np.array([0.00667, 0.024, 0.041, 0.062])

outer_splittings_hz = np.array([0.055, 0.041, 0.021])
# --- 2. Linearize the Data by Squaring ---
x_data = b_fields_fit**2        # This is your X-axis (B^2)
y_data = (inner_splittings_hz*1e9)**2   # This is your Y-axis (Δν^2)
y_err = prefactor*nu_err*np.sqrt(y_data)
x_data2 = -np.array([0.003,0.002,0.001,0])**2
y_data2 = -np.array([0.060e9,0.041e9,0.021e9,0.00667e9])**2
y_err2 = prefactor*nu_err*np.sqrt(-y_data2)
# --- 3. Perform a Linear Fit on the Squared Data ---
fit_result = linregress(x_data, y_data)
slope1 = fit_result.slope
slope1_err = 6.561e+08
intercept1 = fit_result.intercept

fit_result2 = linregress(x_data2, y_data2)
slope2 = fit_result2.slope
intercept2 = fit_result2.intercept
print(f"Fit Results for Inner Peaks (B>=0):")
print(f"Slope for positive measurments (m) = {slope1:.3e}")
print(f"Slope2 for negative measurments (m) = {slope2:.3e}")

g_factor = (np.sqrt(slope1) * h) / (mu_B*np.abs(np.cos(np.deg2rad(theta))))
g_factor2 = (np.sqrt(slope2) * h) / (mu_B*np.abs(np.cos(np.deg2rad(theta))))

print("\nCalculated Physical Parameters from this subset:")
print(f"g-factor for pos = {g_factor:.4f}")
print(f"g-factor for neg = {g_factor2:.4f}")



plt.figure(figsize=(8, 6))
plt.errorbar(x_data[1:] , y_data[1:],yerr=y_err[1:], label='Measurement', color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.errorbar(x_data[0]+1e-7,y_data[0],yerr=y_err[0], color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.plot(x_data, intercept1 + slope1 * x_data, 'r-', 
         label=f'Linear Fit 1(g = {g_factor:.4f})')
plt.plot(x_data2, -(intercept2 + slope2*x_data2), 
         label=f'Linear Fit 2(g = {g_factor2:.4f})')
plt.errorbar(x_data2[:3], -y_data2[:3],yerr=y_err2[:3], color='gray',fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.errorbar(x_data2[3]-1e-7,-y_data2[3],yerr=y_err2[3], color='gray', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.xlabel(r'$B^2 [T^2]$')
plt.ylabel(r'$\Delta\nu^2 [Hz^2]$')
plt.title('')
plt.legend()
plt.grid(True)
plt.show()
# %% weighted fit 
h = 6.62607015e-34      # Planck [J*s]
mu_B = 9.274009994e-24  # Bohr magneton [J/T]
theta = np.deg2rad(131.82)  # angle in radians

# --- Map B-field values (in Tesla) to your spectra ---
B_map = {
    "B=-3mT": -0.003,
    "B=-2mT": -0.002,
    "B=-1mT": -0.001,
    "B=1mT":   0.001,
    "B=2mT":   0.002,
    "B=3mT":   0.003,
}

# --- Your spectra lists ---
fz_list = [zf1, zf2, zf3, zf4, zf5, zf6]
rz_list = [rz1, rz2, rz3, rz4, rz5, rz6]
titles_list = list(B_map.keys())

# --- Automatic peak detection ---
all_peak_freqs = []
all_B_fields = []

for fz, rz, title in zip(fz_list, rz_list, titles_list):
    # Invert for dip detection
    spectrum_to_peaks = 1 - rz
    peaks_indices, _ = find_peaks(spectrum_to_peaks, height=0.01, prominence=0.005)
    peaks_freqs = fz[peaks_indices] / 1e9  # convert to GHz
    
    all_peak_freqs.extend(peaks_freqs)
    all_B_fields.extend([B_map[title]] * len(peaks_freqs))

# --- Convert to arrays ---
all_peak_freqs = np.array(all_peak_freqs)      # in GHz
all_B_fields   = np.array(all_B_fields)        # in Tesla
# given constants (from your snippet)
h = 6.62607015e-34      # J*s
mu_B = 9.274009994e-24  # J/T
nu_err = 0.005e9        # Hz (uncertainty on frequency splitting)
prefactor = np.sqrt(8)
theta = np.deg2rad(52)

# --- your data (as in your message) ---
b_fields_fit = np.array([0,0.001, 0.002, 0.003])*1e9   # T
inner_splittings_hz = np.array([0.00667, 0.024e9, 0.041e9, 0.062e9])  # GHz

y_data = (inner_splittings_hz) ** 2  
x_data = b_fields_fit ** 2               

x_data2 = np.array([0.003e9, 0.002e9, 0.001e9,0.00])**2
y_data2 = np.array([0.060e9, 0.041e9, 0.021e9, 0.00667e9])**2 # Hz^2 (negative)

# --- Compute y uncertainties according to your formula: sigma_y = sqrt(8) * y * y_err ---
sigma_y1 = prefactor * np.sqrt(np.abs(y_data)) * nu_err   # use abs(y) to keep sigma positive
sigma_y2 = prefactor * np.sqrt(np.abs(y_data2)) * nu_err  # same

# --- Weighted linear fit (degree=1). np.polyfit uses w = 1/sigma ---
# For dataset 1:
p1, cov1 = curve_fit(linear_function, x_data, y_data, sigma=sigma_y1, absolute_sigma=True)
slope1, intercept1 = p1[0], p1[1]
slope1_err = np.sqrt(cov1[0, 0])
intercept1_err = np.sqrt(cov1[1, 1])

# For dataset 2:
# If any sigma is zero or extremely small, polyfit will complain. We guard against zeros:
sigma_y2[sigma_y2 == 0] = np.min(sigma_y2[sigma_y2 > 0]) * 1e-6
p2, cov2 = curve_fit(linear_function, x_data2, y_data2,sigma=sigma_y2, absolute_sigma=True)
slope2, intercept2 = p2[0]*1e20, p2[1]
slope2_err = np.sqrt(cov2[0, 0])
intercept2_err = np.sqrt(cov2[1, 1])

# --- Propagate slope uncertainty to g-factor ---
# g = (sqrt(slope) * h) / (mu_B * |cos(theta)|)
# d g / d slope = h / (2 * mu_B * |cos(theta)| * sqrt(slope))
def g_from_slope(slope):
    if slope <= 0:
        return np.nan
    return (np.sqrt(slope) * h) / (mu_B * np.abs(np.cos(theta)))

def g_err_from_slope(slope, slope_err):
    if slope <= 0:
        return np.nan
    dg_ds = h / (2.0 * mu_B * np.abs(np.cos(theta)) * np.sqrt(slope))
    return np.abs(dg_ds) * slope_err

g1 = g_from_slope(slope1*1e6)
g1_err = g_err_from_slope(slope1*1e6, slope1_err*1e6)

g2 = g_from_slope(slope2*1e6)
g2_err = g_err_from_slope(slope2*1e6, slope2_err*1e6)

# --- Print results ---
print("Dataset 1 (B >= 0):")
print(f"  slope = {slope1:.6e} ± {slope1_err:.2e}  [units: Hz^2 / T^2]")
print(f"  intercept = {intercept1:.6e} ± {intercept1_err:.2e}")
print(f"  g = {g1:.5f} ± {g1_err:.5f}")

print("\nDataset 2 (negative B entries):")
print(f"  slope = {slope2:.6e} ± {slope2_err:.2e}  [units: Hz^2 / T^2]")
print(f"  intercept = {intercept2:.6e} ± {intercept2_err:.2e}")
print(f"  g = {g2:.5f} ± {g2_err:.5f}")

# --- Plot with errorbars and fits ---
x_plot = np.linspace(np.min(np.concatenate([x_data,x_data2]))*1.1,
                     np.max(np.concatenate([x_data,x_data2]))*1.1, 200)

plt.figure(figsize=(8,6))
# dataset1 points with y errorbars
lines, caplines, barlines = plt.errorbar(x_data, y_data, yerr=sigma_y1, fmt='o', label='Measurement', capsize=capsize)
plt.plot(-x_plot, intercept1 + slope1 * x_plot, '-', label=f'Fit 2')

# dataset2 points with y errorbars
lines2, caplines2, barlines2 = plt.errorbar(-x_data2, y_data2, yerr=sigma_y2, fmt='s', label='Measurement', capsize=capsize)
plt.plot(x_plot, intercept2 + slope2 * x_plot, '--', label=f'Fit 1')

lines.set_marker('+')
lines.set_markersize(10)
lines.set_markerfacecolor('green')
lines.set_markeredgewidth(2)

lines2.set_marker('+')
lines2.set_markersize(4)
lines2.set_markerfacecolor('green')
lines2.set_markeredgewidth(2)

plt.xlabel(r'$B^2\; [T^2]$')
plt.ylabel(r'$\Delta\nu^2\; [Hz^2]$')
plt.title('')
plt.legend()
plt.grid(True)
plt.show()
# %%
slope1_max = slope1 + slope1_err
slope1_min = slope1 - slope1_err

# keep same intercept (simple approach)
y_line_central1 = intercept1 + slope1 * x_plot
y_line_max1 = intercept1 + slope1_max * x_plot
y_line_min1 = intercept1 + slope1_min * x_plot

# --- g-factors for maximal/minimal slopes ---
g1_max = g_from_slope(slope1_max)
g1_min = g_from_slope(slope1_min)

print("Dataset 1 g-factor range from slope uncertainty:")
print(f"  g_min = {g1_min:.5f}")
print(f"  g_central = {g1:.5f}")
print(f"  g_max = {g1_max:.5f}")

# --- Plot ---
plt.figure(figsize=(8,6))
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', capsize=2, label='Measurement')

# central fit
plt.plot(x_plot, y_line_central1, 'r-', label=f'Central fit (g={g1:.4f})')

# envelope fits
plt.plot(x_plot, y_line_max1, 'r--', label=f'Max slope (g={g1_max:.4f})')
plt.plot(x_plot, y_line_min1, 'r--', label=f'Min slope (g={g1_min:.4f})')

plt.xlabel(r'B² [T²]')
plt.ylabel('Δν² [Hz²]')
plt.title('Linear fit with maximal & minimal slope (dataset 1)')
plt.legend()
plt.grid(True)
plt.show()
# %%
nv_crystals = [
    np.array([1, -1, 1]),
    #np.array([1, 1, 1]),
    np.array([1, 1, -1]),
    np.array([-1,1,1])
]

# Euler angles (degrees)
euler = [132.3, -7.119, 148.118]
rot = R.from_euler('xyz', euler, degrees=True)

# Rotate NV axes into lab frame
nv_lab_units = [rot.apply(v)/np.linalg.norm(rot.apply(v)) for v in nv_crystals]

# Applied B-field in spherical coordinates
theta_deg = 40.0
phi_deg   = 20.0
theta = np.deg2rad(theta_deg)
phi   = np.deg2rad(phi_deg)
b_lab = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
b_lab_unit = b_lab / np.linalg.norm(b_lab)

# Compute angles
for nv_crystal, nv_lab in zip(nv_crystals, nv_lab_units):
    dot = np.clip(np.dot(nv_lab, b_lab_unit), -1.0, 1.0)
    vector_angle = np.degrees(np.arccos(dot))          # 0–180°
    axis_angle   = min(vector_angle, 180 - vector_angle)  # 0–90°
    print(f"NV crystal: {nv_crystal}")
    print(f"  NV (lab unit): {nv_lab.round(6)}")
    print(f"  Vector angle: {vector_angle:.2f}°")
    print(f"  Unsigned axis angle: {axis_angle:.2f}°\n")
# %%
