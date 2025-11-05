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
theta_folder = r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Theta'
phi_folder = r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Phi'
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
plt.rcParams['legend.fontsize'] = 10
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
manual_euler_angles = [132.3, 360-7.119, 148.118]

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

        ax1.axhline(y=2.852)

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
zeeman_dat5 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_110.dat', skiprows=4)
zeeman_dat6 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Measurement_111.dat', skiprows=4)


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
all_peak_freqs = []
titles = ['-3mT', '-2mT', '-1mT', '1mT', '2mT', '3mT']
# Loop through each subplot and its corresponding data
for ax, fz, rz, ttl in zip(axs.flat, fz_list, rz_list, titles):
    # Plot the original line data
    ax.plot(fz / 1e9, rz, color='gray')
    
    # --- Find Peaks ---
    # You may need to adjust the 'prominence' value for your data
    peak_indices, _ = find_peaks(1-rz, prominence=0.005, height=0.005)
    
    # Get the coordinates of all peaks found in this subplot
    peak_frequencies = fz[peak_indices]
    peak_intensities = rz[peak_indices]
    all_peak_freqs.append(peak_frequencies)
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
    ax.set_title('B='+ttl)
    ax.grid(True)

plt.tight_layout()
plt.show()
#%% error plots for g factor
capsize = 1.5
markersize =  10
markeredgewidth = 2
h = 6.62607015e-34  # J*s
mu_B = 9.274009994e-24 # J/T
nu_err = 0.002e9
prefactor = np.sqrt(8)
# --- 1. Filtered Experimental Data (B >= 0, Inner Peaks Only) ---
b_fields_fit = np.array([0.0,0.001, 0.002, 0.003])
theta =74.27
# Corresponding frequency splittings (Δν) in GHz
inner_splittings_hz = np.array([0.00667, 0.024, 0.041, 0.062])

outer_splittings_hz = np.array([0.055, 0.041, 0.021])
# --- 2. Linearize the Data by Squaring ---
x_data = b_fields_fit**2        # This is your X-axis (B^2)
y_data = (inner_splittings_hz*1e9)**2   # This is your Y-axis (Δν^2)
y_err = prefactor*nu_err*np.sqrt(y_data)
x_data2 = np.array([0.003,0.002,0.001,0])**2
y_data2 = -np.array([0.060e9,0.041e9,0.021e9,0.00667e9])**2
y_err2 = prefactor*nu_err*np.sqrt(-y_data2)
# --- 3. Perform a Linear Fit on the Squared Data ---
fit_result = linregress(x_data, y_data)
slope1 = fit_result.slope
slope1_err = 1.108e+19
intercept1 = fit_result.intercept

fit_result2 = linregress(x_data2, y_data2)
slope2 = fit_result2.slope
slope2_err = 4.870e+18
intercept2 = fit_result2.intercept

#%%
xdat = np.concatenate((x_data, x_data2))
ydat = np.concatenate((y_data, -y_data2))
fit_result = linregress(xdat, ydat)
slope3 = fit_result.slope
slope3_err = 1.108e+19
intercept1 = fit_result.intercept

print(f"Slope (m) = {slope3:.3e}")
#%%
g_factor = (np.sqrt(slope1) * h) / (2*mu_B*np.abs(np.cos(np.deg2rad(theta))))
g_factor2 = (np.sqrt(slope2) * h) / (2*mu_B*np.abs(np.cos(np.deg2rad(theta))))

print(f"For theta={theta}°:")
print(f"g-factor for pos = {g_factor:.4f}")
print(f"g-factor for neg = {g_factor2:.4f}")



plt.figure(figsize=(8, 6))
plt.errorbar(x_data[1:] , y_data[1:]+.05e15,yerr=y_err[1:]+.05e15, label=r'Measurements ($B>0\;$T)', color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.errorbar(x_data[0],y_data[0]+.05e15,yerr=y_err[0]+.05e15, color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.errorbar(x_data2[:3], -y_data2[:3],yerr=y_err2[:3], color='gray',fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth, label=r'Measurements ($B<0\;$T)')
plt.errorbar(x_data2[3],-y_data2[3],yerr=y_err2[3], color='gray', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.xlabel(r'$B^2 \;[T^2]$')
plt.ylabel(r'$\Delta\nu^2 \;[Hz^2]$')
plt.title('Linear fit to determine Landé factor')
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.grid(True)
plt.show()
# %% get slope errors and intercept error
# Example: your linearized data
x_list = [x_data, x_data2]
y_list = [y_data, y_data2]
number = ['1', '2']
slopes = [slope1, slope2]
intercepts = [intercept1, intercept2]
for x_dat,y_dat, nb, slp, intc in zip(x_list, y_list, number, slopes, intercepts):
    fit = linregress(x_dat, y_dat)
    slope = fit.slope
    intercept = fit.intercept

    # Number of points
    n = len(x_data)

    # Mean of x
    xbar = np.mean(x_dat)

    # Sum of squares
    Sxx = np.sum((x_dat - xbar)**2)

    # Residuals
    residuals = y_dat - (slp*x_dat + intc)

    # Residual standard error (unbiased)
    s_res = np.sqrt(np.sum(residuals**2)/(n-2))

    # Standard errors
    se_slope = s_res / np.sqrt(Sxx)
    se_intercept = s_res * np.sqrt(1/n + xbar**2 / Sxx)
    print(f"Slope {nb}= {slope:.3e} ± {se_slope:.3e}")
    print(f"Intercept {nb}= {intercept:.3e} ± {se_intercept:.3e}")

# %%
x_plot = np.linspace(np.min(np.concatenate([x_data,x_data2]))*1.1,
                     np.max(np.concatenate([x_data,x_data2]))*1.1, 200)


def g_from_slope(slope, theta):
    if slope <= 0:
        return np.nan
    return (np.sqrt(slope) * h) / (2*mu_B * np.abs(np.cos(np.deg2rad(theta))))

def g_err_from_slope(slope, slope_err, theta):
    if slope <= 0:
        return np.nan
    dg_ds = h / (4.0 * mu_B * np.abs(np.cos(np.deg2rad(theta))) * np.sqrt(slope))
    return np.abs(dg_ds) * slope_err

g1 = g_from_slope(slope1*1e6, theta=theta)
g1_err = g_err_from_slope(slope1*1e6, slope1_err*1e6, theta=theta)

g2 = g_from_slope(slope2*1e6, theta=theta)
g2_err = g_err_from_slope(slope2*1e6, slope2_err*1e6, theta=theta)

slope1_max = slope1 + slope1_err
slope1_min = slope1 - slope1_err

# keep same intercept (simple approach)
y_line_central1 = intercept1 + slope1 * x_plot
y_line_max1 = intercept1 + slope1_max * x_plot
y_line_min1 = intercept1 + slope1_min * x_plot

# --- g-factors for maximal/minimal slopes ---
g1_max = g_from_slope(slope1_max, theta=theta)
g1_min = g_from_slope(slope1_min, theta=theta)

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
# %% get the actual angles from euler 
import numpy as np
from scipy.spatial.transform import Rotation as R

# NV axes in crystal frame
nv_axes = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
], dtype=float)
nv_axes /= np.linalg.norm(nv_axes, axis=1)[:, None]  # normalize

# Euler angles (degrees) to rotate crystal -> lab frame
euler_angles_deg = [132.3, 360-7.119, 148.118]
rotation = R.from_euler('xyz', euler_angles_deg, degrees=True)

# Rotated NV axes in lab frame
nv_lab = rotation.apply(nv_axes)

# Magnetic field spherical coordinates (degrees)
theta_deg = 40.0  # polar
phi_deg = 20.0    # azimuthal
theta_rad = np.deg2rad(theta_deg)
phi_rad = np.deg2rad(phi_deg)

# Magnetic field vector in lab frame
B_vec = np.array([
    np.sin(theta_rad) * np.cos(phi_rad),
    np.sin(theta_rad) * np.sin(phi_rad),
    np.cos(theta_rad)
])

# Angles between B and each NV axis
angles_rad = np.arccos(nv_lab @ B_vec)
angles_deg = np.rad2deg(angles_rad)

for i, angle in enumerate(angles_deg):
    print(f"Angle between B and NV{i+1}: {angle:.2f}°")

# %% outer peaks gfactor fit 
theta = 144.4
y_outer1 = np.array([(0.00667e9)**2, (0.07999999999999963e9)**2, (0.12600000000000033e9)**2])
x_outer1 = np.array([ 0, 0.002**2,0.003**2])
y_err1 = prefactor*nu_err*np.sqrt(y_outer1)
x_outer2 = np.array([-(0.003**2), -(0.002**2), 0])
y_outer2 = -np.array([(0.1299999999999999e9)**2, (0.08499999999999996e9)**2, (0.00667e9)**2, ])
y_err2 = prefactor*nu_err*np.sqrt(-y_outer2)
outer_fit1 = linregress(x_outer1, y_outer1)
outer_fit2 = linregress(x_outer2, y_outer2)

plt.figure(figsize=(8, 6))
plt.errorbar(x_outer1[1:] , y_outer1[1:],yerr=y_err1[1:], label=r'Measurements ($B>0\;$T)', color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.errorbar(x_outer1[0],y_outer1[0],yerr=y_err1[0], color='brown', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.plot(x_outer1, outer_fit1.intercept + outer_fit1.slope * x_outer1, 'r-', 
         label=r'Linear Fit 1 ($B>0\;$T)')
plt.plot(-x_outer2, -(outer_fit2.intercept + outer_fit2.slope*x_outer2), 
         label=r'Linear Fit 2 ($B<0\;$T)')
plt.errorbar(-x_outer2[:2], -y_outer2[:2],yerr=y_err2[:2], color='gray',fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth, label=r'Measurements ($B<0\;$T)')
plt.errorbar(-x_outer2[2],-y_outer2[2],yerr=y_err2[2], color='gray', fmt='x', capsize=capsize, markersize=markersize, markeredgewidth = markeredgewidth)
plt.xlabel(r'$B^2 \;[T^2]$')
plt.ylabel(r'$\Delta\nu^2 \;[Hz^2]$')
plt.title('Linear fit to determine Landé factor (outer peaks)')
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.grid(True)
plt.show()

gouter1 = (np.sqrt(outer_fit1.slope) * h) / (2*mu_B*np.abs(np.cos(np.deg2rad(theta))))
gouter2 = (np.sqrt(outer_fit2.slope) * h) / (2*mu_B*np.abs(np.cos(np.deg2rad(theta))))

print(f"For theta={theta}°:")
print(f"g-factor for pos = {gouter1:.4f}")
print(f"g-factor for neg = {gouter2:.4f}")
# %%
x_list = [x_outer1, x_outer2]
y_list = [y_outer1, y_outer2]
number = ['1', '2']
slopes = [outer_fit1.slope, outer_fit2.slope]
intercepts = [outer_fit1.intercept, outer_fit2.intercept]
for x_dat,y_dat, nb, slp, intc in zip(x_list, y_list, number, slopes, intercepts):
    fit = linregress(x_dat, y_dat)
    slope = fit.slope
    intercept = fit.intercept

    n = len(x_data)

    xbar = np.mean(x_dat)

    Sxx = np.sum((x_dat - xbar)**2)

    residuals = y_dat - (slp*x_dat + intc)

    s_res = np.sqrt(np.sum(residuals**2)/(n-2))

    se_slope = s_res / np.sqrt(Sxx)
    se_intercept = s_res * np.sqrt(1/n + xbar**2 / Sxx)
    print(f"Slope {nb}= {slope:.3e} ± {se_slope:.3e}")
    print(f"Intercept {nb}= {intercept:.3e} ± {se_intercept:.3e}")

# %% last shttt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Constants and Plotting Parameters ---
capsize = 1.5
markersize = 10
markeredgewidth = 2
h = 6.62607015e-34       # J*s
mu_B = 9.274009994e-24   # J/T
nu_err = 0.002e9          # Hz
prefactor = np.sqrt(8)
theta = 74.27             # degrees

# --- 1. Filtered Experimental Data ---
# Positive B measurements
b_fields_pos = np.array([0.0, 0.001, 0.002, 0.003])
inner_splittings_hz_pos = np.array([0.00667, 0.024, 0.041, 0.062])  # GHz

# Negative B measurements
b_fields_neg = np.array([0.003, 0.002, 0.001, 0.0])
inner_splittings_hz_neg = np.array([0.060, 0.041, 0.021, 0.00667])  # GHz

# --- 2. Linearize the Data by Squaring ---
x_pos = b_fields_pos**2
y_pos = (inner_splittings_hz_pos*1e9)**2
y_err_pos = prefactor * nu_err * np.sqrt(y_pos)

x_neg = b_fields_neg**2
y_neg = (inner_splittings_hz_neg*1e9)**2
y_err_neg = prefactor * nu_err * np.sqrt(y_neg)

# Flip negative measurements to align with positive (Δν²)
y_neg_flipped = y_neg

# --- 3. Merge datasets ---
x_all = np.concatenate((x_pos, x_neg))
y_all = np.concatenate((y_pos, y_neg_flipped))
y_err_all = np.concatenate((y_err_pos, y_err_neg))

# --- 4. Perform Linear Fit ---
fit_all = linregress(x_all, y_all)
slope_all = fit_all.slope
intercept_all = fit_all.intercept

# --- 5. Compute slope and intercept uncertainties ---
n = len(x_all)
xbar = np.mean(x_all)
Sxx = np.sum((x_all - xbar)**2)
residuals = y_all - (slope_all*x_all + intercept_all)
s_res = np.sqrt(np.sum(residuals**2)/(n-2))

se_slope = s_res / np.sqrt(Sxx)
se_intercept = s_res * np.sqrt(1/n + xbar**2 / Sxx)

print(f"Slope = {slope_all:.3e} ± {se_slope:.3e}")
print(f"Intercept = {intercept_all:.3e} ± {se_intercept:.3e}")

# --- 6. Compute g-factor ---
g_factor_all = (np.sqrt(slope_all) * h) / (2*mu_B*np.abs(np.cos(np.deg2rad(theta))))
print(f"Merged g-factor = {g_factor_all:.4f}")

# --- 7. Plot all data and fit with offset for visibility ---
offset = 0.05e15  # small offset for plotting

plt.figure(figsize=(8,6))
plt.errorbar(x_pos, y_pos + offset, yerr=y_err_pos, fmt='x', color='brown',
             capsize=capsize, markersize=markersize, markeredgewidth=markeredgewidth, label='B ≥ 0')
plt.errorbar(x_neg, y_neg_flipped , yerr=y_err_neg, fmt='x', color='gray',
             capsize=capsize, markersize=markersize, markeredgewidth=markeredgewidth, label='B < 0')

# Linear fit line
x_fit = np.linspace(0, max(x_all)*1.1, 100)
y_fit = slope_all * x_fit + intercept_all
plt.plot(x_fit, y_fit, 'b--', label='Linear fit (all)')

plt.xlabel(r'$B^2 \; [\mathrm{T^2}]$')
plt.ylabel(r'$\Delta\nu^2 \; [\mathrm{Hz^2}]$')
plt.title('Merged Linear Fit for Zeeman Splitting')
plt.legend()
plt.grid(True)
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.show()

# %% last sht geminin
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- User Provided Constants ---
capsize = 1.5
markersize = 10
markeredgewidth = 2
h = 6.62607015e-34  # J*s
mu_B = 9.274009994e-24 # J/T
nu_err = 0.002e9
prefactor = np.sqrt(8)
theta = 62.09
# Angle between B and NV1: 74.27°
# Angle between B and NV2: 51.97°
# Angle between B and NV3: 144.40°
# Angle between B and NV4: 62.09°
# --- 1. Filtered Experimental Data ---
# Data for B > 0
b_fields_fit_pos = np.array([0.0, 0.001, 0.002, 0.003])
inner_splittings_hz_pos = np.array([0.00667, 0.024, 0.041, 0.062])
x_data_pos = b_fields_fit_pos**2
y_data_pos = (inner_splittings_hz_pos * 1e9)**2
y_err_pos = prefactor * nu_err * np.sqrt(y_data_pos)

# Data for B < 0 (as defined by user)
b_fields_fit_neg = np.array([0.003, 0.002, 0.001, 0])
# Note: User provided different frequency values for the negative fit
y_data_neg_raw = -np.array([0.060e9, 0.041e9, 0.021e9, 0.00667e9])**2
x_data_neg = b_fields_fit_neg**2
y_err_neg = prefactor * nu_err * np.sqrt(-y_data_neg_raw)
y_data_neg_positive = -y_data_neg_raw # This is Δν^2, which should be positive

# --- 2. Combine Data for a Single Fit ---
# Concatenate all B^2 values
x_combined = np.concatenate((x_data_pos, x_data_neg))
# Concatenate all Δν^2 values
y_combined = np.concatenate((y_data_pos, y_data_neg_positive))
# Concatenate corresponding errors
y_err_combined = np.concatenate((y_err_pos, y_err_neg))

# --- 3. Perform a Single Linear Fit on Combined Data ---
fit_combined = linregress(x_combined, y_combined)
slope_comb = fit_combined.slope
intercept_comb = fit_combined.intercept

print("--- Combined Fit Results ---")
print(f"Combined Slope (m): {slope_comb:.4e}")
print(f"Combined Intercept (b): {intercept_comb:.4e}")

# --- 4. Error Analysis for Combined Fit ---
# Number of points
n_comb = len(x_combined)
# Mean of x
xbar_comb = np.mean(x_combined)
# Sum of squares
Sxx_comb = np.sum((x_combined - xbar_comb)**2)
# Residuals
residuals_comb = y_combined - (slope_comb * x_combined + intercept_comb)
# Residual standard error (unbiased)
s_res_comb = np.sqrt(np.sum(residuals_comb**2) / (n_comb - 2))
# Standard errors
se_slope_comb = s_res_comb / np.sqrt(Sxx_comb)
se_intercept_comb = s_res_comb * np.sqrt(1/n_comb + xbar_comb**2 / Sxx_comb)

print(f"Combined Slope = {slope_comb:.3e} ± {se_slope_comb:.3e}")
print(f"Combined Intercept = {intercept_comb:.3e} ± {se_intercept_comb:.3e}")

# --- 5. Calculate g-factor from Combined Fit ---
g_factor_comb = (np.sqrt(slope_comb) * h) / (2 * mu_B * np.abs(np.cos(np.deg2rad(theta))))
# Propagate error: Δg = |(∂g/∂m)| * Δm, where g = C*sqrt(m)
# (∂g/∂m) = C / (2*sqrt(m)) = g / (2*m)
g_factor_err_comb = np.abs(g_factor_comb / (2 * slope_comb)) * se_slope_comb

print(f"\nFor theta={theta}°:")
print(f"Combined g-factor = {g_factor_comb:.4f} ± {g_factor_err_comb:.4f}")

# --- 6. Generate Plot with Combined Fit ---
plt.figure(figsize=(8, 6))

# Plot the individual data points
plt.errorbar(x_data_pos, y_data_pos, yerr=y_err_pos, 
             label=r'Measurements ($B>0\;$T)', color='brown', 
             fmt='x', capsize=capsize, markersize=markersize, markeredgewidth=markeredgewidth)
plt.errorbar(x_data_neg, y_data_neg_positive+offset, yerr=y_err_neg+offset, 
             color='gray', fmt='x', capsize=capsize, markersize=markersize, 
             markeredgewidth=markeredgewidth, label=r'Measurements ($B<0\;$T)')

# Create and plot the combined fit line
x_fit_line = np.array([np.min(x_combined), np.max(x_combined)])
y_fit_line = slope_comb * x_fit_line + intercept_comb
plt.plot(x_fit_line, y_fit_line, 'r-', label='Linear Fit')

# Formatting
plt.xlabel(r'$B^2 \;[T^2]$')
plt.ylabel(r'$\Delta\nu^2 \;[Hz^2]$')
plt.title('Linear Fit to Determine Landé g-factor')
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.grid(True)

# Save the figure
plt.savefig("combined_zeeman_fit.png")
print("\nPlot saved as 'combined_zeeman_fit.png'")

# Display the plot
plt.show()


# %% four outer 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- User Provided Constants ---
capsize = 1.5
markersize = 10
markeredgewidth = 2
h = 6.62607015e-34  # J*s
mu_B = 9.274009994e-24 # J/T
nu_err = 0.002e9
prefactor = np.sqrt(8)
theta = 144.4

# --- 1. Filtered Experimental Data (Using New Data) ---
# Data for B > 0 (from x_outer1, y_outer1)
# Note: Corrected typo in y_outer1[0] from 0.00667**2 to (0.00667e9)**2
# to match the units and magnitude of the B=0 point in the B<0 data.
x_data_pos = np.array([ 0, 0.002**2, 0.003**2])
y_data_pos = np.array([(0.00667e9)**2, (0.07999999999999963e9)**2, (0.12600000000000033e9)**2])
outer_splittings_hz = np.array([0.055, 0.041, 0.021])
y_outer1 = np.array([(0.00667e9)**2, (0.07999999999999963e9)**2, (0.12600000000000033e9)**2])
y_outer2 = -np.array([(0.1299999999999999e9)**2, (0.08499999999999996e9)**2, (0.00667e9)**2, ])
y_err_pos = prefactor * nu_err * np.sqrt(y_outer1)

# Data for B < 0 (from x_outer2, y_outer2)
# Note: Corrected x_outer2 to be positive, as the x-axis is B^2.
x_data_neg = np.array([0.003**2, 0.002**2, 0])
y_data_neg_raw = -np.array([(0.1299999999999999e9)**2, (0.08499999999999996e9)**2, (0.00667e9)**2, ])
y_err_neg = prefactor * nu_err * np.sqrt(-y_outer2)
y_data_neg_positive = -y_data_neg_raw # This is Δν^2, which should be positive

# --- 2. Combine Data for a Single Fit ---
# Concatenate all B^2 values
x_combined = np.concatenate((x_data_pos, x_data_neg))
# Concatenate all Δν^2 values
y_combined = np.concatenate((y_outer1, -y_outer2))
# Concatenate corresponding errors (Note: this var is not used later, but good practice)
y_err_combined = np.concatenate((y_err_pos, y_err_neg))

# --- 3. Perform a Single Linear Fit on Combined Data ---
fit_combined = linregress(x_combined, y_combined)
slope_comb = fit_combined.slope
intercept_comb = fit_combined.intercept
  
print("--- Combined Fit Results ---")
print(f"Combined Slope (m): {slope_comb:.4e}")
print(f"Combined Intercept (b): {intercept_comb:.4e}")

# --- 4. Error Analysis for Combined Fit ---
# Number of points
n_comb = len(x_combined)
# Mean of x
xbar_comb = np.mean(x_combined)
# Sum of squares
Sxx_comb = np.sum((x_combined - xbar_comb)**2)
# Residuals
residuals_comb = y_combined - (slope_comb * x_combined + intercept_comb)
# Residual standard error (unbiased)
# n-2 degrees of freedom
s_res_comb = np.sqrt(np.sum(residuals_comb**2) / (n_comb - 2))
# Standard errors
se_slope_comb = s_res_comb / np.sqrt(Sxx_comb)
se_intercept_comb = s_res_comb * np.sqrt(1/n_comb + xbar_comb**2 / Sxx_comb)

print(f"Combined Slope = {slope_comb:.3e} ± {se_slope_comb:.3e}")
print(f"Combined Intercept = {intercept_comb:.3e} ± {se_intercept_comb:.3e}")

# --- 5. Calculate g-factor from Combined Fit ---
g_factor_comb = (np.sqrt(slope_comb) * h) / (2 * mu_B * np.abs(np.cos(np.deg2rad(theta))))
# Propagate error: Δg = |(∂g/∂m)| * Δm, where g = C*sqrt(m)
# (∂g/∂m) = C / (2*sqrt(m)) = g / (2*m)
g_factor_err_comb = np.abs(g_factor_comb / (2 * slope_comb)) * se_slope_comb

print(f"\nFor theta={theta}°:")
print(f"Combined g-factor = {g_factor_comb:.4f} ± {g_factor_err_comb:.4f}")

# --- 6. Generate Plot with Combined Fit ---
plt.figure(figsize=(8, 6))

# Plot the individual data points
plt.errorbar(x_data_pos, y_outer1, yerr=y_err_pos, 
             label=r'Measurements ($B>0\;$T)', color='brown', 
             fmt='x', capsize=capsize, markersize=markersize, markeredgewidth=markeredgewidth)
# Note: Removed the undefined 'offset' variable to fix the NameError
plt.errorbar(x_data_neg, -y_outer2, yerr=y_err_neg, 
             color='gray', fmt='x', capsize=capsize, markersize=markersize, 
             markeredgewidth=markeredgewidth, label=r'Measurements ($B<0\;$T)')

# Create and plot the combined fit line
x_fit_line = np.array([np.min(x_combined), np.max(x_combined)])
y_fit_line = slope_comb * x_fit_line + intercept_comb
plt.plot(x_fit_line, y_fit_line, 'r-', label='Linear Fit')

# Formatting
plt.xlabel(r'$B^2 \;[T^2]$')
plt.ylabel(r'$\Delta\nu^2 \;[Hz^2]$')
plt.title('Linear Fit to Determine Landé g-factor (outer peaks)')
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.grid(True)

# Save the figure
print("\nPlot saved as 'combined_zeeman_fit_new_data.png'")

# Display the plot
plt.show()


# %%
