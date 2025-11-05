#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.signal import find_peaks

# --- Define constants and your functions ---
h = const.h
e = const.elementary_charge
c = const.speed_of_light
deadtime = 90e-6
systematic_angle_err_deg = 0.05

def get_wavelength(theta, d, order=1):
    wl = 2 * d * np.sin(np.deg2rad(theta)) / order
    return wl

def get_energy_from_wl(wl):
    '''wl: wavelength in meters'''
    energy_ev = h * c / (e * wl)
    return energy_ev

def multi_peak_model_with_bkg(x, *params):
    """A model for multiple Gaussian peaks plus an integrated polynomial background."""
    a, b, c = params[-3:]
    background = a * x**2 + b * x + c
    peak_params = params[:-3]
    num_peaks = len(peak_params) // 3
    peaks_sum = 0
    for i in range(num_peaks):
        peaks_sum += params[i*3] * np.exp(-(x - params[i*3+1])**2 / (2 * params[i*3+2]**2))
    return peaks_sum + background

def linear_func(x, m, c):
    """A simple linear function: y = m*x + c"""
    return m * x + c

#%% --- Load your data (ensure file paths are correct) ---

try:
    fe_data = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Fe2s35kV1mA', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
    mo_data_full = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Mo2s_4-65', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
    cu_data_full = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\2mmganzesspektrum', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
except FileNotFoundError:
    print("ERROR: Data files not found. Please update the file paths.")
    # Create dummy data to allow the script to run for demonstration
    fe_data = np.array([[20,100]])
    mo_data_full = np.array([[10, 200]])
    cu_data_full = np.array([[15, 150]])


element_data = {
    'Copper': {'angle': cu_data_full[:,0] + 0.15, 'counts': cu_data_full[:,1]/(1-deadtime*cu_data_full[:,1]), 'd': 2.01e-10},
    'Iron': {'angle': fe_data[:,0], 'counts': fe_data[:,1]/(1-deadtime*fe_data[:,1]), 'd': 2.01e-10},
    'Molybdenum': {'angle': mo_data_full[:,0], 'counts': mo_data_full[:,1]/(1-deadtime*mo_data_full[:,1]), 'd': 2.01e-10}
}

sensitive_peak_params = {
    'Copper': {'height': 70, 'prominence': 150},
    'Iron': {'height': 70, 'prominence': 300},
    'Molybdenum': {'height': 50, 'prominence': 30}
}

#%%  --- Main analysis loop to find peak energies ---
results = {}
plt.rcParams['font.size'] = 14

for element, data in element_data.items():
    angle = data['angle']
    counts = data['counts']
    d = data['d']
    if len(counts) < 10: continue

    if element == 'Iron':
        search_slice = slice(74, None)
        all_indices, properties = find_peaks(counts[search_slice], **sensitive_peak_params[element])
        all_indices += search_slice.start
    else:
        all_indices, properties = find_peaks(counts, **sensitive_peak_params[element])

    if element == 'Molybdenum':
        all_peak_angles = angle[all_indices]
        angle_mask = (all_peak_angles <= 10.5) | (all_peak_angles >= 16.3)
        all_indices = all_indices[angle_mask]
        for key in properties: properties[key] = properties[key][angle_mask]

    if len(all_indices) > 4:
        top_4_prominence_indices = np.argsort(properties['prominences'])[-4:]
        peak_indices = np.sort(all_indices[top_4_prominence_indices])
    else:
        peak_indices = np.sort(all_indices)

    peak_angles, peak_heights = angle[peak_indices], counts[peak_indices]
    initial_guesses = [val for p_angle, p_height in zip(peak_angles, peak_heights) for val in (p_height, p_angle, 0.2)]
    initial_guesses.extend([0, 0, np.min(counts)])
    
    try:
        # (Fit performing code is omitted for brevity but is unchanged)
        popt, pcov = curve_fit(multi_peak_model_with_bkg, angle, counts, p0=initial_guesses, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError) as e:
        print(f"Could not fit {element} spectrum: {e}")
        continue
    # --- NEW: PLOTTING OF THE GAUSSIAN FITS (re-integrated from previous script) ---
    num_peaks_fitted = len(peak_indices)
    plt.figure(figsize=(10, 7))
    plt.plot(angle, counts, 'o', color='gray', markersize=3, label='Raw Data')
    plt.plot(angle, multi_peak_model_with_bkg(angle, *popt), 'k-', label='Overall Fit')
    
    a_fit, b_fit, c_fit = popt[-3:]
    fit_bkg = a_fit * angle**2 + b_fit * angle + c_fit
    plt.plot(angle, fit_bkg, 'b--', label='Fitted Background')
    
    for i in range(num_peaks_fitted):
        fit_peak_params = popt[i*3:(i+1)*3]
        fit_peak = fit_peak_params[0] * np.exp(-(angle - fit_peak_params[1])**2 / (2 * fit_peak_params[2]**2))
        plt.plot(angle, fit_peak + fit_bkg, 'r-', lw=2)
        
    plt.title(f'{element} X-ray Spectrum with Gaussian Fits')
    plt.tick_params(which="both", direction="in", top=True, right=True)
    plt.xlabel(r'$\theta$ [°]')
    plt.ylabel(r'Counts [s⁻¹]')
    plt.grid(True)
    plt.legend()
    plt.show()
    # --- END OF NEW PLOTTING BLOCK ---
    element_results = []
    for i in range(len(peak_indices)):
        center_idx = i*3 + 1
        peak_center, statistical_angle_err = popt[center_idx], perr[center_idx]
        
        en_n1_keV = get_energy_from_wl(get_wavelength(peak_center, d, order=1)) / 1000
        
        statistical_en_err_n1_keV = en_n1_keV * (np.deg2rad(statistical_angle_err) / np.tan(np.deg2rad(peak_center)))
        systematic_en_err_n1_keV = en_n1_keV * (np.deg2rad(systematic_angle_err_deg) / np.tan(np.deg2rad(peak_center)))
        
        final_en_err_n1 = max(statistical_en_err_n1_keV, systematic_en_err_n1_keV)
        
        element_results.append({
            'angle': peak_center, 'angle_err': statistical_angle_err,
            'energy_n1_keV': en_n1_keV, 'energy_n1_err_keV': final_en_err_n1
        })

    results[element] = sorted(element_results, key=lambda x: x['energy_n1_keV'], reverse=True)


#%% --- NEW: Final Summary of All Measured Peak Data ---

z_map = {'Iron': 26, 'Copper': 29, 'Molybdenum': 42}
print("\n" + "="*80)
print("--- Final Summary of Fitted Peak Energies (First Order) ---")
print("="*80)
print(f"{'Element':<12} | {'Line':<8} | {'Angle [°]':<24} | {'Energy [keV]':<28}")
print("-" * 80)

for element, res_list in sorted(results.items(), key=lambda item: z_map[item[0]]):
    if len(res_list) >= 2:
        kb_peak = res_list[0]
        ka_peak = res_list[1]
        
        for line_name, peak_data in [('K-beta', kb_peak), ('K-alpha', ka_peak)]:
            peak_center = peak_data['angle']
            stat_angle_err = peak_data['angle_err']
            energy = peak_data['energy_n1_keV']
            final_energy_err = peak_data['energy_n1_err_keV']
            
            # Check if systematic error was dominant
            stat_energy_err = energy * (np.deg2rad(stat_angle_err) / np.tan(np.deg2rad(peak_center)))
            is_systematic_dominant = (final_energy_err > stat_energy_err)

            if is_systematic_dominant:
                angle_str = f"{peak_center:8.4f} ± {systematic_angle_err_deg:.4f}*"
            else:
                angle_str = f"{peak_center:8.4f} ± {stat_angle_err:.4f}"
            
            energy_str = f"{energy:8.4f} ± {final_energy_err:.4f}"

            print(f"{element:<12} | {line_name:<8} | {angle_str:<24} | {energy_str:<28}")

print("-" * 80)
print("*Indicates that the systematic angle error (0.05°) was the dominant source of uncertainty.")
print("="*80 + "\n")


#%% --- Data Preparation for Moseley's Law ---


moseley_data = {
    'atomic_numbers': [], 'ka_energies_eV': [], 'ka_errors_eV': [],
    'kb_energies_eV': [], 'kb_errors_eV': []
}

for element, res_list in sorted(results.items(), key=lambda item: z_map[item[0]]):
    if len(res_list) >= 2:
        kb_peak = res_list[0]
        ka_peak = res_list[1]
        
        moseley_data['atomic_numbers'].append(z_map[element])
        moseley_data['ka_energies_eV'].append(ka_peak['energy_n1_keV'] * 1000)
        moseley_data['ka_errors_eV'].append(ka_peak['energy_n1_err_keV'] * 1000)
        moseley_data['kb_energies_eV'].append(kb_peak['energy_n1_keV'] * 1000)
        moseley_data['kb_errors_eV'].append(kb_peak['energy_n1_err_keV'] * 1000)

atomic_numbers = np.array(moseley_data['atomic_numbers'])
ka_energies = np.array(moseley_data['ka_energies_eV'])
ka_errors = np.array(moseley_data['ka_errors_eV'])
kb_energies = np.array(moseley_data['kb_energies_eV'])
kb_errors = np.array(moseley_data['kb_errors_eV'])

#%% --- Reusable function for Moseley's Law Analysis ---


def perform_moseley_analysis(line_name, atomic_numbers, energies_eV, energy_errors_eV):
    """
    Performs a complete Moseley's Law analysis for a given spectral line.
    """
    if 'alpha' in line_name.lower():
        n_factor = (1/1**2 - 1/2**2)
        title = r"Moseley's Law ($K_\alpha$ line)"
    else:
        n_factor = (1/1**2 - 1/3**2)
        title = r"Moseley's Law ($K_\beta$ line)"

    sqrt_energies = np.sqrt(energies_eV)
    sqrt_errors = energy_errors_eV / (2 * sqrt_energies)

    popt, pcov = curve_fit(linear_func, atomic_numbers, sqrt_energies, sigma=sqrt_errors, absolute_sigma=True)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    rydberg_energy = slope**2 / n_factor
    rydberg_err = 2 * rydberg_energy * (slope_err / slope)
    
    sigma = -intercept / slope
    sigma_err = abs(sigma) * np.sqrt( (intercept_err / intercept)**2 + (slope_err / slope)**2 )

    print(f"\n--- Analysis Results for {line_name} ---")
    print(f"Rydberg Energy = ({rydberg_energy:.2f} +/- {rydberg_err:.2f}) eV")
    print(f"Screening Factor (σ) = {sigma:.2f} +/- {sigma_err:.2f}")

    # Plotting
    x_fit = np.linspace(atomic_numbers.min(), atomic_numbers.max(), 200)
    y_fit = linear_func(x_fit, slope, intercept)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(atomic_numbers, sqrt_energies, yerr=sqrt_errors,
                 fmt='o', capsize=5, ecolor='black', color='darkorange', label='Data points')
    plt.plot(x_fit, y_fit, c='orange', label='Linear Fit')
    plt.tick_params(which="both", direction="in", top=True, right=True)
    plt.title(title)
    plt.grid(True)
    plt.ylabel(r'$\sqrt{\nu}$  [$\sqrt{eV}$]')
    plt.xlabel('Atomic Number Z')
    plt.legend()
    plt.show()

#%% --- Run the final analysis for both K-alpha and K-beta lines ---

if len(atomic_numbers) > 0:
    perform_moseley_analysis("K-alpha", atomic_numbers, ka_energies, ka_errors)
    perform_moseley_analysis("K-beta", atomic_numbers, kb_energies, kb_errors)
else:
    print("\nSkipping final analysis because no valid peak data was found.")
    print("Please check your data files and peak finding parameters.")
# %%
