#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
import os
import glob
import sys
from pathlib import Path
import pandas as pd 
import re
from lmfit import Model
#%%
num_peaks = 2
prominence=0.05
guess_width= 10

B_MAGNITUDE_MT = 1.5
DATA_FOLDER = r'C:\Main\my_projects\F-Praktikum\my_odmr_data\Phi'

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
#%% define functions 
def lorentzian_form(x, xc, w, A):
    """
    y = (2A)/pi * w / (4 (x-xc)^2 + w^2)
    xc: center
    w: FWHM
    A: integrated area (i.e. integral of y over x)
    """
    x = np.asarray(x)  # ensure array
    return (2.0 * A / np.pi) * (w / (4.0*(x - xc)**2 + w**2))

def multi_lorentzian(x, *params):
    x = np.atleast_1d(x).astype(float)  # ensure array
    if len(params) == 0:
        raise ValueError("No parameters supplied to multi_lorentzian")
    if len(params) == 1:
        p = np.asarray(params[0], dtype=float).ravel()   # handles popt (array)
    else:
        p = np.asarray(params, dtype=float).ravel()     # handles separate args

    if p.size < 3:
        raise ValueError("Need at least 3 parameters (xc, w, A) for one Lorentzian")

    # optional trailing baseline: length 3*N + 1
    baseline = 0.0
    if p.size % 3 == 1:
        baseline = float(p[-1])
        p = p[:-1]

    if p.size % 3 != 0:
        raise ValueError("Parameter length must be 3*N, or 3*N+1 with trailing baseline")

    n = p.size // 3
    y = np.full_like(x, baseline, dtype=float)

    for i in range(n):
        xc, w, A = p[3*i:3*i+3]
        y += lorentzian_form(x, xc, w, A)

    return y


def load_pl_spectrum(text_dat):
    data = np.loadtxt(text_dat, skiprows=4)
    freq_hz = data[:,0]
    intensity = data[:,-1]
    
    return freq_hz, intensity

def load_data_for_colormap(folder_path):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    full_folder_path = os.path.join(script_dir, folder_path)
    
    data_files = glob.glob(os.path.join(full_folder_path, '*.dat'))
    data_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f) or '0'))))

    all_ratios = []
    theta_angles = []
    freq_hz = None
    

    for filepath in data_files:
        try:
            current_theta = None
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip().startswith("#B-Field Theta"):
                        parts = line.split()
                        if len(parts) > 2:
                            current_theta = float(parts[2])
                        break
            
            if current_theta is None:
                print(f"Warning: Could not find '#B-Field Theta' in header of {os.path.basename(filepath)}. Skipping file.")
                continue

            data = np.loadtxt(filepath, skiprows=3)
            
            if freq_hz is None:
                freq_hz = data[:, 0]
            
            if len(data[:, 4]) == len(freq_hz):
                all_ratios.append(data[:, 4])
                theta_angles.append(current_theta)
            else:
                print(f"Warning: Data in {os.path.basename(filepath)} has a different length. Skipping file.")

        except Exception as e:
            print(f"Warning: Could not process file {os.path.basename(filepath)}: {e}")
            
    if not theta_angles:
        print("Error: No valid data could be loaded. Cannot create plot.")
        return None, None, None

    return np.array(theta_angles), freq_hz / 1e9, np.array(all_ratios).T
#%% zero field plot and spectra
x, y = load_pl_spectrum(
    r'C:\Main\my_projects\F-Praktikum\ProbstHofmann\Measurement_64.dat'
)
y_err = np.ones_like(y)*0.002
# initial guesses: [xc1, w1, A1, xc2, w2, A2]
initial_guess = [2.864e9, .5*1e6, 1e9,
        2.874e9, .5*1e6, 1e9]
popt, pcov = curve_fit(multi_lorentzian, x, 1-y, p0=initial_guess, sigma=y_err, absolute_sigma = True)

perr = np.sqrt(np.diag(pcov))

print("Fit parameters:")
for i, (val, err) in enumerate(zip(popt, perr), start=1):
    print(f"p{i} = {val:.6g} ± {err:.2g}")

# --- goodness of fit ---
y_fit = multi_lorentzian(x, *popt)          # model values
residuals = (1 - y) - y_fit                 # difference data - fit
ss_res = np.sum(residuals**2)               # residual sum of squares
ss_tot = np.sum(((1 - y) - np.mean(1 - y))**2)   # total sum of squares

r_squared = 1 - ss_res / ss_tot             # R²
n = len(x)                                  # number of data points
p = len(popt)                               # number of fit params
r2_adj = 1 - (1-r_squared)*(n-1)/(n-p-1)    # adjusted R²

print("Fit results:")
for i, (val, err) in enumerate(zip(popt, perr), start=1):
    print(f"p{i} = {val:.6g} ± {err:.2g}")

print(f"R² = {r_squared:.5f}")
print(f"Adjusted R² = {r2_adj:.5f}")

x_fine = np.linspace(x.min(), x.max(), 1000)
y_fit_fine = multi_lorentzian(x_fine, *popt)

# plotting
plt.figure(figsize=(9,6))
plt.errorbar(x=x/1e9, y=y, yerr=y_err,fmt="o", capsize=3, label="Measurement", ecolor='black', color='black')
plt.plot(x_fine/1e9, 1 - y_fit_fine, "-", label="fit", lw=2)
plt.legend(loc=3)
plt.xlabel("Frequency [GHz]")
plt.ylabel("Intensity [a.u.]")
plt.title('Zero-field ODMR spectrum')
plt.tight_layout()
plt.xlim(2.84, 2.9)
plt.grid()
plt.tick_params(which='both', direction='in', top=True, right=True)
plt.show()

#%% colorplot of angle sweeps
folder = r"C:\Main\my_projects\F-Praktikum\my_odmr_data\Theta"
files = sorted(glob.glob(os.path.join(folder, "*.dat")),
               key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

ratios_list = []
thetas = []

# --- Choose the first file's frequency grid as the common reference ---
data_ref = np.loadtxt(files[0], skiprows=3)
freq_common = data_ref[:,0] / 1e9  # GHz

for f in files:
    # --- Extract angle from filename ---
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    theta = (number - 85) * 10  # adjust so 66 -> 0°
    thetas.append(theta)
    
    # --- Load measurement ---
    data = np.loadtxt(f, skiprows=3)
    freq = data[:,0] 
    ratio = data[:,4]
    
    ratios_list.append(ratio)

# --- Convert to arrays ---
ratios = np.array(ratios_list).T  # shape: (n_freqs, n_angles)
thetas = np.array(thetas)

# --- Meshgrid for contour ---
tick_positions = [0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
T, F = np.meshgrid(thetas, freq_common)

plt.figure(figsize=(10,6))
plt.gca().tick_params(which="both", direction="in", top=True, right=True)
plt.contourf(T, F, ratios, levels=50, cmap='viridis')
plt.colorbar(label='Photon Counter Ratio [a.u.]', ticks=tick_positions)
plt.xlabel(r'Angle $\theta$[°]')
plt.ylabel('Frequency [GHz]')
plt.title(r'ODMR Spectrum vs Magnetic Field Angle sweep $\theta$')
plt.ylim(2.8,2.95)
plt.show()

# %%
folder = r"C:\Main\my_projects\F-Praktikum\ODMR\my_odmr_data\Phi"
files = sorted(glob.glob(os.path.join(folder, "*.dat")),
               key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

ratios_list = []
thetas = []

# --- Choose the first file's frequency grid as the common reference ---
data_ref = np.loadtxt(files[0], skiprows=3)
freq_common = data_ref[:,0] / 1e9  # GHz

for f in files:
    # --- Extract angle from filename ---
    number = int(re.search(r'\d+', os.path.basename(f)).group())
    theta = (number - 66) * 10  # adjust so 66 -> 0°
    thetas.append(theta)
    
    # --- Load measurement ---
    data = np.loadtxt(f, skiprows=3)
    freq = data[:,0] 
    ratio = data[:,4]
    
    ratios_list.append(ratio)

# --- Convert to arrays ---
ratios = np.array(ratios_list).T  # shape: (n_freqs, n_angles)
thetas = np.array(thetas)

# --- Meshgrid for contour ---
tick_positions = [0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
T, F = np.meshgrid(thetas, freq_common)

plt.figure(figsize=(10,6))
plt.gca().tick_params(which="both", direction="in", top=True, right=True)
plt.contourf(T, F, ratios, levels=50, cmap='viridis')
plt.colorbar(label='Photon Counter Ratio [a.u.]', ticks=tick_positions)
plt.xlabel(r'Angle $\phi$[°]')
plt.ylabel('Frequency [GHz]')
plt.title(r'ODMR Spectrum vs Magnetic Field Angle sweep $\phi$')
plt.ylim(2.8,2.95)
plt.show()
# %%
