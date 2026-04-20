#%%
import os
import re
import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
import scipy as sp 
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pandas as pd
import mplcursors 
from io import StringIO
import scipy.constants as const
with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_1.dat') as f:
    measurement_1 = np.loadtxt(
        StringIO(f.read().replace(',', '.')),
        skiprows=15   # IMPORTANT: skip metadata + header line
    )
with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_0.dat') as f:
    cal_data = np.loadtxt(
        StringIO(f.read().replace(',', '.')),
        skiprows=15  
    )
#%%
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20

def temp_from_pressure(p, regime):
    #p in Pascal, T in Kelvin
    if regime == 'low':
            a_list = [1.392408, 0.527153, 0.166756, 0.050988, 0.026514, 0.001975, -0.017976, 0.005409, 0.013259, 0]
            b = 5.6
            c = 2.9
    elif regime == 'high':
            a_list = [3.146631, 1.357655, 0.413923, 0.091159, 0.016349, 0.001826, -0.004325, -0.004973, 0, 0]
            b = 10.3
            c = 1.9
    else:
        raise ValueError("Regime must be 'low' or 'high'")
    x = (np.log(p) - b) / c
    temp_k = a_list[0] + sum(a * x**i for i, a in enumerate(a_list[1:], start=1))
    return temp_k

def fit_func1(T, a, b, c1):
    return a + b* np.exp(-T/c1) 

def compute_r2(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def lin_fit(x, m, b):
    return m * x + b

def temp_from_ab(U_AB, regime):
    if regime == 'low':
        a, b, c1 = popt_low
    elif regime == 'high':
        a, b, c1 = popt_high
    else:
        raise ValueError("Regime must be 'low' or 'high'")

    arg = (U_AB - a) / b
    return -c1 * np.log(arg)

def get_h_and_err(V, dV):
    """Calculates H (in Tesla) and its propagated error."""
    # Coil constants
    r_i, dr_i = 10.5e-3, 0.5e-3
    r_o, dr_o = 20.4e-3, 0.5e-3
    L, dL = 193e-3, 3e-3
    R, N = 384.5e-6, 10353
    
    I, dI = V / R, dV / R
    a = L / 2
    sqrt_o, sqrt_i = np.sqrt(r_o**2 + a**2), np.sqrt(r_i**2 + a**2)
    diff_r = r_o - r_i
    log_val = np.log((r_o + sqrt_o) / (r_i + sqrt_i))
    
    H = (0.5 * N * I / diff_r) * log_val
    
    # Sensitivity (Derivatives)
    dH_dI = (N / (2 * diff_r)) * log_val
    dH_dro = (N * I / (2 * diff_r)) * (1/sqrt_o - log_val/diff_r)
    dH_dri = (N * I / (2 * diff_r)) * (log_val/diff_r - 1/sqrt_i)
    term_o, term_i = L/(sqrt_o*(r_o+sqrt_o)), L/(sqrt_i*(r_i+sqrt_i))
    dH_dL = (N * I / (8 * diff_r)) * (term_o - term_i)
    
    err_H = np.sqrt((dH_dI*dI)**2 + (dH_dro*dr_o)**2 + (dH_dri*dr_i)**2 + (dH_dL*dL)**2)
    return H * const.mu_0, err_H * const.mu_0

def calculate_h_error(V, dV):
    """
    Calculates H and propagated error specifically for:
    dV (Voltage error), dr_o, dr_i, and dL.
    Assumes R and N are constants with zero error.
    """
    r_i = 10.5e-3
    dr_i = 0.5e-3
    r_o = 20.4e-3
    dr_o = 0.5e-3
    L = 193e-3
    dL = 3e-3
    R = 384.5e-6
    N = 10353

    # 1. Calculate Current and its Error (from Voltage only)
    I = V / R
    dI = dV / R  # Simple linear scaling because R is constant
    
    # 2. Geometric Constants
    a = L / 2
    sqrt_o = np.sqrt(r_o**2 + a**2)
    sqrt_i = np.sqrt(r_i**2 + a**2)
    diff_r = r_o - r_i
    
    log_val = np.log((r_o + sqrt_o) / (r_i + sqrt_i))
    
    # 3. Calculate H
    H = (0.5 * N * I / diff_r) * log_val
    
    # 4. Partial Derivatives (Sensitivity Coefficients)
    # Effect of Current (Voltage)
    dH_dI = (N / (2 * diff_r)) * log_val
    
    # Effect of Outer Radius
    dH_dro = (N * I / (2 * diff_r)) * (1/sqrt_o - log_val/diff_r)
    
    # Effect of Inner Radius
    dH_dri = (N * I / (2 * diff_r)) * (log_val/diff_r - 1/sqrt_i)
    
    # Effect of Length
    term_o = L / (sqrt_o * (r_o + sqrt_o))
    term_i = L / (sqrt_i * (r_i + sqrt_i))
    dH_dL = (N * I / (8 * diff_r)) * (term_o - term_i)
    
    # 5. Combine Errors (Sum of Squares)
    # Each term represents (Sensitivity * Uncertainty)^2
    err_sq_V = (dH_dI * dI)**2
    err_sq_ro = (dH_dro * dr_o)**2
    err_sq_ri = (dH_dri * dr_i)**2
    err_sq_L = (dH_dL * dL)**2
    
    delta_H = np.sqrt(err_sq_V + err_sq_ro + err_sq_ri + err_sq_L)
    
    return H*const.mu_0, delta_H*const.mu_0

def get_temp_maximal_error(U_AB, popt_low, perr_low, popt_high, perr_high):
    """Calculates T and asymmetric errors via maximal error analysis."""
    # Logic: Which regime are we in? 
    # Usually U_AB > ~0.015V is the 'high' regime for this sensor
    if U_AB < 0.034099: 
        p, p_err = popt_high, perr_high
    else:
        p, p_err = popt_low, perr_low

    def T_calc(u, a, b, c):
        val = (u - a) / b
        return -c * np.log(val) if val > 0 else np.nan

    dU = 0.00005 * U_AB + 0.0000035
    Tc = T_calc(U_AB, *p)
    
    # Maximal Error: Shift parameters and voltage to find worst-case T
    T_max = T_calc(U_AB - dU, p[0]-p_err[0], p[1]-p_err[1], p[2]+p_err[2])
    T_min = T_calc(U_AB + dU, p[0]+p_err[0], p[1]+p_err[1], p[2]-p_err[2])
    
    return Tc, np.abs(Tc - T_min), np.abs(T_max - Tc)

# %% Fit for vapour pressure vs. voltage of manometer 

start_idx = 100

ab_data = np.array([0.009184, 0.009409, 0.009521, 0.0096367, 0.009879, 0.010739, 0.011039, 0.011314, 0.01145, 0.01183, 0.012097, 0.01238, 0.012683, 0.012895, 0.01358, 0.014367, 0.015075, 0.017633, 0.019340, 0.024503, 0.031261])

mano_data = np.array([0.953671, 0.901634, 0.875302, 0.849757, 0.799083, 0.654431, 0.612501, 0.577452, 0.562078, 0.519941, 0.493119, 0.467356, 0.441701, 0.425199, 0.377189, .331608, 0.297266, 0.209065, 0.170809, 0.102019, 0.0602943])

p_data = np.array([1001, 947, 919, 892, 838, 687, 643, 606, 590, 546, 517, 491, 464, 446, 396, 348, 311, 219, 179, 107, 63])*100  # Convert from mbar to Pa

ab_err = ab_data * 0.00005 + 0.0000035
p_minus = p_data / 1.03 - 100
p_plus  = p_data / 0.97 + 100
mano_err = 0.0025 * mano_data + 0.006
# central fit
popt_lin_c, pcov_lin_c = curve_fit(lin_fit, mano_data, p_data)

# lower and upper extreme fits
popt_lin_minus, _ = curve_fit(lin_fit, mano_data-mano_err, p_minus)
popt_lin_plus, _  = curve_fit(lin_fit, mano_data+mano_err, p_plus)

# parameter shifts
err_lin_minus = np.abs(popt_lin_c - popt_lin_minus)
err_lin_plus  = np.abs(popt_lin_plus - popt_lin_c)

print("Linear calibration:")
for name, val, em, ep in zip(['slope', 'intercept'], popt_lin_c, err_lin_minus, err_lin_plus):
    print(f"{name} = {val:.6f} (-{em:.6f}, +{ep:.6f})")

x_plot = np.linspace(mano_data.min(), mano_data.max()+.08, 300)

plt.figure(figsize=(8, 6))
plt.scatter(mano_data, p_data, label='Data', color='black', s=15)

plt.plot(x_plot, lin_fit(x_plot, *popt_lin_c), label='Central fit', color='tab:blue')
plt.plot(x_plot, lin_fit(x_plot, *popt_lin_minus), '--', label='Lower bound fit', color='red', alpha=.7)
plt.plot(x_plot, lin_fit(x_plot, *popt_lin_plus), '--', label='Upper bound fit', color='orange', alpha=.7)

# optional asymmetric error bars on the calibration points
yerr_minus = p_data - p_minus
yerr_plus = p_plus - p_data
plt.errorbar(
    mano_data, p_data,
    yerr=[yerr_minus, yerr_plus],
    xerr=mano_err,
    fmt='none',
    ecolor='gray',
    alpha=0.7,
    capsize=3
)

plt.xlabel(r'$U_{\mathrm{Mano}}$ (V)')
plt.ylabel(r'$P$ (Pa)')
plt.title(r'Linear fit $p_{He}$ vs. $U_M$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0,1)
plt.ylim(lin_fit(x_plot, *popt_lin_minus).min()-1000,lin_fit(x_plot, *popt_lin_plus).max()+2000)
plt.show()


#%% Calibration p(U_m) via linear fit and plot, then compute temperature from pressure using its-90 and fit U_AB vs. T for low and high regime separately

mask_pressure = 4984  # Pa

ab_cal = np.array(cal_data[start_idx:, 1])
mano_cal = cal_data[start_idx:, 2]
ab_err = 0.00005 * ab_cal + 0.0000035
p_cal_c = popt_lin_c[0] * mano_cal + popt_lin_c[1]

# pressure bounds: 3% calibration + 1 mbar reading error = 100 Pa
p_cal_minus = p_cal_c / 1.03 + 100
p_cal_plus  = p_cal_c / 0.97 + 100

p_floor = 1
p_cal_c = np.clip(p_cal_c, p_floor, None)
p_cal_minus = np.clip(p_cal_minus, p_floor, None)
p_cal_plus  = np.clip(p_cal_plus, p_floor, None)

temp0 = np.empty_like(p_cal_c)
temp_low_bound = np.empty_like(p_cal_c)
temp_high_bound = np.empty_like(p_cal_c)

for i in range(len(p_cal_c)):
    regime = 'low' if p_cal_c[i] < mask_pressure else 'high'
    temp0[i] = temp_from_pressure(p_cal_c[i], regime)
    temp_low_bound[i] = temp_from_pressure(p_cal_minus[i], regime)
    temp_high_bound[i] = temp_from_pressure(p_cal_plus[i], regime)

dT_minus = np.abs(temp0 - temp_low_bound)
dT_plus  = np.abs(temp_high_bound - temp0)

print("Typical dT_minus:", np.min(dT_minus), np.median(dT_minus), np.max(dT_minus))
print("Typical dT_plus :", np.min(dT_plus), np.median(dT_plus), np.max(dT_plus))

low_mask = p_cal_c < mask_pressure
high_mask = p_cal_c >= mask_pressure

temp_low = temp0[low_mask]
temp_high = temp0[high_mask]

y_low = ab_cal[low_mask]
y_high = ab_cal[high_mask]


# 1. In your calibration section, create the 'extreme' fits
popt_low_max_err, _ = curve_fit(fit_func1, temp_low - dT_minus[low_mask], y_low + ab_err[low_mask])
popt_low_min_err, _ = curve_fit(fit_func1, temp_low + dT_plus[low_mask], y_low - ab_err[low_mask])

popt_high_max_err, _ = curve_fit(fit_func1, temp_high - dT_minus[high_mask], y_high + ab_err[high_mask])
popt_high_min_err, _ = curve_fit(fit_func1, temp_high + dT_plus[high_mask], y_high - ab_err[high_mask])

# fits
popt_low, pcov_low = curve_fit(
    fit_func1, temp_low, y_low,
    p0=[0.01, 0.01, 1.0],
    maxfev=10000
)

popt_high, pcov_high = curve_fit(
    fit_func1, temp_high, y_high,
    p0=[0.01, 0.01, 1.0],
    maxfev=10000
)

y_fit_low = fit_func1(temp_low, *popt_low)
y_fit_high = fit_func1(temp_high, *popt_high)

perr_low = np.sqrt(np.diag(pcov_low))
perr_high = np.sqrt(np.diag(pcov_high))

R2_low = compute_r2(y_low, y_fit_low)
R2_high = compute_r2(y_high, y_fit_high)

# 2. Calculate the 'Physical Shift' for each parameter (instead of perr)
# This represents the total ordeal of the 3% pressure error + manometer reading error
p_shift_low = np.abs(popt_low - popt_low_max_err) 
p_shift_high = np.abs(popt_high - popt_high_max_err)
# 3. Use THIS shift in your error function
        
def get_temp_with_err(U_AB, regime):
    """
    Calculates T and combines Gaussian fit uncertainty with 
    inherited systematic calibration uncertainty.
    """
    # 1. Select Fit Parameters
    if regime == 'low':
        a, b, c = popt_low
        da, db, dc = perr_low # Errors from pcov
    else:
        a, b, c = popt_high
        da, db, dc = perr_high

    # 2. Measurement uncertainty of the Multimeter for U_AB
    dU = 0.00005 * U_AB + 0.0000035

    # 3. Calculate Temperature
    arg = (U_AB - a) / b
    if arg <= 1e-8:
        return np.nan, np.nan, np.nan
    T = -c * np.log(arg)

    # 4. GAUSSIAN PROPAGATION (Statistical Fit Error)
    dT_dU = -c / (U_AB - a)
    dT_da =  c / (U_AB - a)
    dT_db =  c / b
    dT_dc = -np.log(arg)
    
    dT_fit = np.sqrt(
        (dT_dU * dU)**2 +
        (dT_da * da)**2 +
        (dT_db * db)**2 +
        (dT_dc * dc)**2
    )

    # 5. LOOK-UP (Systematic Calibration Error)
    # This finds the error inherited from the Manometer accuracy (3%)
    # Ensure 'ab_cal', 'dT_cal_minus', and 'dT_cal_plus' are defined globally
    idx = np.argmin(np.abs(ab_cal - U_AB))
    sys_m = dT_minus[idx]
    sys_p = dT_plus[idx]

    # 6. COMBINE IN QUADRATURE
    # Total error = sqrt( Fit_Error^2 + Calibration_Error^2 )
    total_m = np.sqrt(dT_fit**2 + sys_m**2)
    total_p = np.sqrt(dT_fit**2 + sys_p**2)

    return T, total_m, total_p

print("High Regime Fit:")
print(f"a  = {popt_high[0]:.5f} ± {perr_high[0]:.5f}")
print(f"b  = {popt_high[1]:.5f} ± {perr_high[1]:.5f}")
print(f"c1 = {popt_high[2]:.5f} ± {perr_high[2]:.5f}")
print(f"R² = {R2_high:.5f}")
print()

print("Low Regime Fit:")
print(f"a  = {popt_low[0]:.5f} ± {perr_low[0]:.5f}")
print(f"b  = {popt_low[1]:.5f} ± {perr_low[1]:.5f}")
print(f"c1 = {popt_low[2]:.5f} ± {perr_low[2]:.5f}")
print(f"R² = {R2_low:.5f}")
print()

T_low_plot = np.linspace(temp_low.min(), temp_low.max(), 400)
T_high_plot = np.linspace(temp_high.min(), temp_high.max(), 400)

cs = 1
ms = 2
ewidth = 3
alph = 1
plt.figure(figsize=(8, 6), dpi=150)

plt.errorbar(
    temp_low,
    y_low,
    xerr=[dT_minus[low_mask], dT_plus[low_mask]],
    yerr = ab_err[low_mask],
    fmt='b.',
    ecolor='lightblue',
    elinewidth=ewidth,
    capsize=cs,
    alpha=alph,
    ms=ms,
    zorder=1
)

plt.errorbar(
    temp_high,
    y_high,
    xerr=[dT_minus[high_mask], dT_plus[high_mask]],
    yerr = ab_err[high_mask],
    fmt='r.',
    ecolor='lightblue',
    elinewidth=3,
    capsize=1,
    alpha=alph,
    ms=2,
    zorder=1
)

#plt.scatter(temp_low, y_low, s=1, color='tab:blue', label='Low regime data', zorder=3, alpha=.2)
#plt.scatter(temp_high, y_high, s=1, color='yellow', label='High regime data', zorder=3, alpha=.2)

plt.plot(
    T_low_plot,
    fit_func1(T_low_plot, *popt_low),
    color='black',
    lw=1.5,
    label='Fit low regime',
    zorder=2
)
plt.plot(
    T_high_plot,
    fit_func1(T_high_plot, *popt_high),
    color='orange',
    lw=1.5,
    label='Fit high regime',
    zorder=2
)

plt.axvline(x=2.1768, color='gray', linestyle='--', label='Lambda point', zorder=0)

plt.xlabel('Temperature (K)')
plt.ylabel(r'$U_{AB}$ (V)')
plt.title(r'Temperature calibration curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Extract critical field data from measurements 2-14, plot U_Sn vs. H for each measurement, then extract the transition voltage U_Spule and calculate the corresponding critical field H_c for each measurement.

ciritcal_field_path = r'C:\coding\my_projects\F-Praktikum\Supercondcutivity\sc_data\ciritcal_field_data'

def extract_number(filename):
    return int(re.search(r'(\d+)\.dat$', filename).group(1)) # type: ignore

files = sorted(
    [f for f in os.listdir(ciritcal_field_path) if f.endswith('.dat')],
    key=extract_number
)
print("Sorted files:", files)
critical_field_data = []

for filename in files:
    filepath = os.path.join(ciritcal_field_path, filename)

    with open(filepath) as f:
        data = np.loadtxt(
            StringIO(f.read().replace(',', '.')),
            skiprows=15
        )

    critical_field_data.append(data)

fig, axs = plt.subplots(3,4,figsize=(12, 8))
axs = axs.flatten()

for i in range(12):
    critical_field_measurement = critical_field_data[i]
    sn_data = critical_field_measurement[:, 3]
    coil_data = critical_field_measurement[:, 4]
    ax = axs[i]
    ax.scatter(coil_data, sn_data, label=f'Measurement {i+1}', s=10)
    ax.set_title(f'Measurement {i+1}')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.locator_params(axis='x', nbins=6)
    ax.set_xlim(left=0)  
    #ax.set_xlabel('Magnetic Field (H)')
    #ax.set_ylabel(r'$U_{SN}$ (V)')

fig.supxlabel('Magnetic Field (H)', fontsize=16)
fig.supylabel(r'$U_{SN}$ (V)', fontsize=16)
fig.suptitle('Critical Field Measurements')
plt.tight_layout()
plt.show()
    
# %% plot single example measurement for H_c 

R_series = 384.5e-6
sn_ex = critical_field_data[5][:,3]
coil_ex = critical_field_data[5][:,4]

coil_err = coil_ex *  0.00005 + 0.0000035 
xerr_ex = []
H_values = []
yerr_ex = sn_ex * 0.00005 + 0.0000035

for V, dV in zip(coil_ex, coil_err):
    H, dH = calculate_h_error(V, dV)
    H_values.append(H)
    xerr_ex.append(dH)

#%%

plt.figure(figsize=(8,6))
plt.errorbar(coil_ex[:100], sn_ex[:100]*1e3, xerr= coil_err[:100], yerr=yerr_ex[:100] , ms=11, label="up sweep",
    elinewidth=1,
    capsize=1             
)
plt.errorbar(coil_ex[100:], sn_ex[100:]*1e3, xerr= coil_err[100:], yerr=yerr_ex[100:], ms=1, label="down sweep",
    elinewidth=1,
    capsize=1             
)
plt.title(r"Critical field measurement at $T\simeq3.11\,K$")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.locator_params(axis="x", nbins=6)
plt.xlabel(r"Magnetic field $H \,(T)$")
plt.ylabel(r"$U_{Sn}\; (mV)$")
plt.grid(True, alpha=.3)
plt.legend()
plt.show()

# %% Extract the transition voltages U_Spule by using (U_Sn.min + U_Sn.max) / 2 as threshold for each measurement. then calculate H_c from U_Spule

transition_voltages = np.zeros(12)
ab_mean = np.zeros(12)

for i in range(12):
    critical_field_measurement = critical_field_data[i]
    sn_data = critical_field_measurement[:, 3]
    coil_data = critical_field_measurement[:, 4]
    ab_data = critical_field_measurement[:, 1]
    
    difference = np.abs(np.diff(sn_data))
    idx = np.argmax(difference) + 10
    ab_mean[i] = np.mean(ab_data)
    transition_voltages[i] = coil_data[idx]


ab_rep = np.array(ab_mean)

T_low = np.array([temp_from_ab(u, 'low') for u in ab_rep])
T_high = np.array([temp_from_ab(u, 'high') for u in ab_rep])

T_vals = []
dT_lows = []
dT_highs = []

for u in ab_rep:
    Tc, dTm, dTp = get_temp_with_err(u, 'high' if u > 0.0341 else 'low') 
    T_vals.append(Tc)
    dT_lows.append(dTm)
    dT_highs.append(dTp)


print("T_low :", T_low)
print("T_high:", T_high)

def theoretical_field(T):
    T_c = 3.722  # K (tin)
    B_0 = 0.0305  # Tesla
    return B_0 * (1 - (T / T_c)**2)

transition_voltages_err = transition_voltages * 0.00005 + 0.0000035

field_values, field_err = calculate_h_error(transition_voltages, transition_voltages_err)
temperature_values = np.array(temp_from_ab(ab_mean, 'high'))
temp_aux = np.linspace(1, 3.8, 100)


plt.figure(figsize=(8, 5))
plt.plot(temp_aux, theoretical_field(temp_aux), label='Theoretical Critical Field', color='black', linestyle='--')
plt.errorbar(temperature_values, field_values, fmt='b.', label='Data', xerr=[dT_lows, dT_highs], yerr=field_err)
plt.xlabel('Temperature (K)')
plt.ylabel('Critical Magnetic Field (T)')
plt.title('Critical Magnetic Field vs. Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% Extraction of H_c(T) from transition voltages and AB voltages, then fit to theoretical curve to extract T_c and H_0

transition_voltages = np.zeros(12)
ab_rep = np.zeros(12)

for i in range(12):
    m = critical_field_data[i]

    sn_data = m[:, 3]
    coil_data = m[:, 4]
    ab_data = m[:, 1]

    order = np.argsort(coil_data)
    sn_data = sn_data[order]
    coil_data = coil_data[order]
    ab_data = ab_data[order]

    diff_sn = np.abs(np.diff(sn_data))
    idx = np.argmax(diff_sn)

    transition_voltages[i] = 0.5 * (coil_data[idx] + coil_data[idx+1])

    # average of ab around transition
    lo = max(idx - 20, 0)
    hi = min(idx + 20, len(ab_data))
    ab_rep[i] = np.mean(ab_data[lo:hi])

transition_voltages_err = transition_voltages * 0.00005 + 0.0000035


field_values, field_err = calculate_h_error(transition_voltages, transition_voltages_err)
T_lambda = 2.1768  # K

temperature_values = []

for u in ab_rep:
    candidates = []

    for regime in ['low', 'high']:
        try:
            T = temp_from_ab(u, regime)
            if np.isfinite(T):
                candidates.append((regime, T))
        except:
            pass

    chosen_T = None

    for regime, T in candidates:
        if regime == 'low' and T < T_lambda:
            chosen_T = T
        elif regime == 'high' and T >= T_lambda:
            chosen_T = T

    # fallback: take first valid
    if chosen_T is None and candidates:
        chosen_T = candidates[0][1]

    temperature_values.append(chosen_T)

temperature_values = np.array(temperature_values)

# %% Fitting of H_c(T) to extract T_c and H_0 and plotting of theoretical curve, fit, and data points

def fit_func2(T, H_0, T_c):
    return H_0 * (1 - (T / T_c)**2)

t_array = np.linspace(1, 3.8, 100)
popt, pcov = curve_fit(fit_func2, temperature_values, field_values, p0=[0.03, 3.7], maxfev=10000)


H_0_fit, T_c_fit = popt
H_0_err, T_c_err = np.sqrt(np.diag(pcov))
print(f"H_0 = {H_0_fit:.4f} ± {H_0_err:.4f} T")
print(f"T_c = {T_c_fit:.4f} ± {T_c_err:.4f} K")

plt.figure(figsize=(8, 5))
plt.plot(t_array, theoretical_field(t_array), label='Theoretical $B_c(T)$', color='black', linestyle='--', zorder=1)
plt.plot(t_array, fit_func2(t_array, *popt), label='Fitted $B_c(T)$', color='blue', linestyle='-', zorder=1)
#plt.errorbar(temperature_values, field_values, yerr=0.01, xerr=0.05, fmt='o', color='red', capsize=5, elinewidth=1, markeredgewidth=1, label='Measured $B_c(T)$')
plt.errorbar(T_vals, field_values, fmt='r.', label='Measured $B_c(T)$', yerr=field_err, zorder=2, alpha=.6, xerr=[dT_lows, dT_highs])
plt.xlabel('Temperature (K)')
plt.ylabel('Critical Magnetic Field (T)')
plt.title('Critical Magnetic Field vs. Temperature with Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# %% Direct measurement of T_c from multiple sweeps around the transition temperature, then plot U_AB vs. T
#Load in data and plot it

sweep1_start = 240
sweep1_end = 500

sweep2_start = 500
sweep2_end = 750

sweep3_start = 1000
sweep3_end = 1300

sweep4_start = 1300
sweep4_end = 1703

with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_15.dat') as f:
    cal_data = np.loadtxt(
        StringIO(f.read().replace(',', '.')),
        skiprows=15  
    )

sn_multiple = cal_data[:, 3]
ab_multiple = cal_data[:, 1]
t_multiple = np.array([temp_from_ab(u, 'high') for u in ab_multiple])

xerr_minus = []
xerr_plus = []

for U_AB in ab_multiple:
    _, xerr_l, xerr_h = get_temp_with_err(U_AB, 'high')  # or 'low' depending on regime
    xerr_minus.append(xerr_l)
    xerr_plus.append(xerr_h)

xerr = [np.array(xerr_minus), np.array(xerr_plus)]  
yerr = ab_multiple * 0.00005 + 0.0000035

plt.figure(figsize=(8, 6))
scatter1 = plt.scatter(t_multiple[sweep1_start:sweep1_end], sn_multiple[sweep1_start:sweep1_end] * 1e6, label='1st (down)', color='blue', s=3)
mplcursors.cursor(scatter1, hover=True)
plt.show()
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(t_multiple[sweep2_start:sweep2_end], sn_multiple[sweep2_start:sweep2_end] * 1e6, label='2nd (up)', color='red', s=3)
mplcursors.cursor(scatter2, hover=True)
plt.show()
plt.figure(figsize=(8, 6))
scatter3 = plt.scatter(t_multiple[sweep3_start:sweep3_end], sn_multiple[sweep3_start:sweep3_end] * 1e6, label='3rd (down)', color='orange', s=3)
mplcursors.cursor(scatter3, hover=True)
plt.show()
plt.figure(figsize=(8, 6))
scatter4 = plt.scatter(t_multiple[sweep4_start:sweep4_end], sn_multiple[sweep4_start:sweep4_end] * 1e6, label='4th (up)', color='magenta', s=3)
mplcursors.cursor(scatter4, hover=True)
plt.show()
plt.figure(figsize=(8,6))
plt.errorbar(
    t_multiple,
    sn_multiple*1e6,
    xerr=xerr,
    yerr=yerr,
    fmt='o',
    ms=3,
    capsize=2,
    label='Data',
    ecolor='lightblue',
    elinewidth=.5,
    alpha=.8
)
plt.xlabel('Temperature (K)')
plt.ylabel(r'$U_{Sn}$ ($\mu$V)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=3.7, c='black', label='Approximate lower value', ls = '--', lw=.8, alpha=.6)
plt.axvline(x=3.7363, c='black', label='Approximate upper value', ls = '--', lw=.8, alpha=.6)
plt.xlim(3.65, 3.76)
plt.legend()
plt.show()


# %% Extract T_c for both hysteresis loops and take mean value 
low_vals = np.array([3.7001, 3.7004, 3.7005, 3.6968])
high_vals = np.array([3.7390, 3.7351, 3.7384, 3.7327])
added_vals = (low_vals + high_vals)/2
avg_val = np.mean(added_vals)

diff_T = np.mean(np.abs(high_vals - low_vals)/2)
print(r'Average T_c =' + f' {avg_val:.4f}' + r" pm" + f" {diff_T:.4f}")

T_err = diff_T + np.abs(dT_plus[736])
print(f'T_err = {T_err:.4f}')
# %%
