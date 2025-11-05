#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.constants as const
from scipy.optimize import curve_fit
#%%

def analyze_spectrum(angles_deg, counts, background_level, window_size=25, 
                     search_start_angle=3.0, search_end_angle=8.0):
    """
    Finds lambda_min by fitting the raw data in a specified range and finding
    its intersection with the provided background noise level.
    """
    d_spacing = 2.010e-10

    search_indices = np.where((angles_deg >= search_start_angle) & (angles_deg <= search_end_angle))
    angles_search, counts_search = angles_deg[search_indices], counts[search_indices]

    if len(angles_search) < window_size:
        print(f"Warning: Search range [{search_start_angle}, {search_end_angle}] is too small.")
        return None, None, None

    best_r_squared, best_slope, best_intercept, best_window_slice = -1, 0, 0, None
    for i in range(len(angles_search) - window_size):
        window_slice = slice(i, i + window_size)
        slope, intercept, r_value, _, _ = linregress(angles_search[window_slice], counts_search[window_slice])
        if r_value**2 > best_r_squared:
            best_r_squared, best_slope, best_intercept, best_window_slice = r_value**2, slope, intercept, window_slice
            
    if best_window_slice is None: return None, None, None

    theta_min_deg = (background_level - best_intercept) / best_slope
    lambda_min = 2 * d_spacing * np.sin(np.deg2rad(theta_min_deg))
    
    original_start_index = search_indices[0][0]
    best_fit_indices = slice(original_start_index + best_window_slice.start,
                             original_start_index + best_window_slice.stop)
    
    return lambda_min, theta_min_deg, best_fit_indices


filepath = r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Messreihe2_2Spannung'

analysis_masks = {
    11000: (15.8, 18.5), 13000: (13.7, 17), 15000: (11, 13.8), 17000: (10, 13),
    19000: (9.1, 12.5), 21000: (7, 11), 23000: (7.7, 10.1), 25000: (6.7, 9.7),
    27000: (6.3, 910), 29000: (5.8, 8.7), 31000: (5.5, 8), 33000: (5.2, 8),
    35000: (4.2, 8)
}
background_levels = {
    11000: 0, 13000: 0, 15000: 0, 17000: 0, 19000: 0,
    21000: 0, 23000: 0, 25000: .8, 27000: 1.2, 29000: 3.1,
    31000: 3.5, 33000: 8, 35000: 10
}

voltage_data = np.loadtxt(filepath, skiprows=2, converters=lambda s: s.replace(b',', b'.'))
voltage_angles = voltage_data[:, 0]

# --- 2. DATA PROCESSING LOOP ---
voltages, found_lambda_mins = [], []

for i in range(13):
    current_voltage = (11 + 2 * i) * 1000
    current_counts = voltage_data[:, (i + 1)]/(1- 90e-6*voltage_data[:, (i + 1)])
    
    start_angle, end_angle = analysis_masks[current_voltage]
    bg_level = background_levels[current_voltage]
    
    lambda_min, theta_min, fit_indices = analyze_spectrum(
        voltage_angles, current_counts, background_level=bg_level,
        search_start_angle=start_angle, search_end_angle=end_angle
    )
    
    if lambda_min is not None:
        voltages.append(current_voltage)
        found_lambda_mins.append(lambda_min)
        print(f"U = {current_voltage/1000} kV: Using BG={bg_level} -> Found onset θ_min = {theta_min:.2f}°")

        plt.figure(figsize=(10, 6))
        plt.plot(voltage_angles, current_counts, label='Measurement', color='gray', alpha=0.7)
        plt.scatter(voltage_angles[fit_indices], current_counts[fit_indices], color='red', zorder=5, label='Data used for Fit')
        slope, intercept, _, _, _ = linregress(voltage_angles[fit_indices], current_counts[fit_indices])
        x_fit = np.linspace(theta_min - 1, voltage_angles[fit_indices.stop], 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'b--', label='Linear Fit')
        plt.axhline(bg_level, color='orange', linestyle=':', label=f'Background ({bg_level})')
        plt.ylabel(r'Counts $[s^{-1}]$'), plt.xlabel(r'$\theta$ [°]'), plt.title(f'Spectrum for U = {current_voltage/1000} kV')
        plt.grid(True), plt.legend(), plt.xlim(voltage_angles[0], 20)
        plt.ylim(0,50)
        plt.tick_params(which="both", direction="in", top=True, right=True)
        plt.show()

# =============================================================================
# 3. FINAL CALCULATION & PLOTTING (UPDATED FOR λ vs 1/V)
# =============================================================================
voltages, found_lambda_mins = np.array(voltages), np.array(found_lambda_mins)
e_charge, c_light, h_accepted = const.e, const.c, const.h

# --- Step 1: Calculate the y-error (s_λ) for each data point ---
# Re-calculate the list of theta_mins that correspond to your found_lambda_mins
theta_mins_rad = np.arcsin(found_lambda_mins / (2 * 2.01e-10))

# Goniometer uncertainty in radians
s_theta_rad = np.deg2rad(0.05)

# Calculate the error in lambda for each point using the propagation formula
# s_λ = |2d*cos(θ)| * s_θ
lambda_errors =  abs(2 * 2.01e-10 * np.cos(theta_mins_rad)) * s_theta_rad


# --- Step 2: Perform the Weighted Linear Fit with curve_fit ---
# Define the function we want to fit (a straight line)
def linear_func(x, m, b):
    return m * x + b

# Prepare the x and y data for the λ vs 1/V plot
x_data = 1 / voltages
y_data = found_lambda_mins

# Call curve_fit, passing the lambda_errors to the 'sigma' parameter
# This tells the function to perform a weighted least squares fit.
popt, pcov = curve_fit(linear_func, x_data, y_data, sigma=lambda_errors)

# Extract the results from the fit
slope = popt[0]
intercept = popt[1]

# The standard errors are the square root of the diagonal of the covariance matrix (pcov)
slope_error = np.sqrt(pcov[0, 0])


# --- Step 3: Calculate h and Propagate the New Error ---
# Use the slope from the weighted fit to calculate h
h_experimental = (slope * e_charge) / c_light

# Propagate the new, more accurate slope error to find the error in h
# IMPORTANT: This single error value now accounts for both the data scatter
# and the goniometer uncertainty, so we do not need to combine errors at the end.
h_total_error = (e_charge / c_light) * slope_error


# --- Print Final Results ---
print("\n" + "="*40 + "\n     FINAL RESULTS (Weighted Fit Method)\n" + "="*40)
print(f"Fit Quality (R-squared is not directly given by curve_fit, but check plot)")
print("-" * 40)
print("Uncertainty Analysis:")
print(f"  - The total propagated error now includes the goniometer uncertainty.")
print("-" * 40)
print(f"Experimental Planck's Constant (h): ({h_experimental/1e-34:.3f} ± {h_total_error/1e-34:.3f}) x 10⁻³⁴ J·s")
print(f"Accepted Planck's Constant (h):     {h_accepted:.4e} J·s")
print(f"Percent Error: {abs((h_experimental - h_accepted) / h_accepted) * 100:.2f}%")


# --- Final Plot: λ_min vs. 1/V with Error Bars ---
plt.figure(figsize=(10, 6))

# Use plt.errorbar to show the individual uncertainty for each point
plt.errorbar(x_data, y_data*1e9, yerr=0, fmt='o', color='red', 
             ecolor='lightgray', elinewidth=3, capsize=1, label='Experimental Data')

# Plot the best-fit line from curve_fit
plt.plot(x_data, linear_func(x_data, slope, intercept)*1e9, color='blue', label='Linear Fit')

plt.title(r'$\lambda_{min}$ vs. $\frac{1}{U_A}$')
plt.xlabel(r'$\frac{1}{U_A} [V^{-1}]$')
plt.ylabel(r'$λ_{min}$ [nm]')
plt.grid(True)
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.show()
# %%
