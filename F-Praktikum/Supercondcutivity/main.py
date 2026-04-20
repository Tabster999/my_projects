#%%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from io import StringIO

#%% Load data
with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_1.dat') as f:
    measurement_1 = np.loadtxt(
        StringIO(f.read().replace(',', '.')),
        skiprows=15
    )
with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_0.dat') as f:
    cal_data = np.loadtxt(
        StringIO(f.read().replace(',', '.')),
        skiprows=15
    )

#%% Matplotlib style
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20

#%% Helper functions
def temp_from_pressure(p, regime):
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
    return a + b * np.exp(-T/c1)

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

def temp_from_ab_safe(U_AB):
    # compute both
    try:
        T_low = temp_from_ab(U_AB, 'low')
    except:
        T_low = np.nan
    try:
        T_high = temp_from_ab(U_AB, 'high')
    except:
        T_high = np.nan
    # pick physically reasonable value
    if np.isfinite(T_low) and T_low < 2.1768:
        return T_low, 'low'
    elif np.isfinite(T_high) and T_high > 2.1768:
        return T_high, 'high'
    else:
        # fallback
        return np.nan, None
    
def calculate_h_error(V, dV):
    # Coil parameters
    r_i, r_o = 10.5e-3, 20.4e-3
    dr_i, dr_o = 0.5e-3, 0.5e-3
    L, dL = 193e-3, 3e-3
    R = 384.5e-6
    N = 10353
    mu0 = const.mu_0

    # Current and its absolute error
    I = V / R
    dI = dV / R

    # Geometry terms
    a = L / 2
    da = dL / 2
    diff_r = r_o - r_i
    # We treat the error in (r_o - r_i) as dr_diff = sqrt(dr_o^2 + dr_i^2)
    ddiff_r = np.sqrt(dr_o**2 + dr_i**2)
    
    sqrt_o = np.sqrt(r_o**2 + a**2)
    sqrt_i = np.sqrt(r_i**2 + a**2)
    log_val = np.log((r_o + sqrt_o) / (r_i + sqrt_i))
    
    # Central Value B = mu0 * H
    B = mu0 * 0.5 * N * I / diff_r * log_val

    # Partial derivatives (simplified for practical propagation)
    # 1. dB/dI = B / I
    err_I = (B / I) * dI
    
    # 2. dB/d(diff_r) approx -B / diff_r (dominant term for radial error)
    err_r = (B / diff_r) * ddiff_r
    
    # 3. dB/dL (The log term depends on L via a=L/2)
    # This is often smaller than I and r errors, but for completeness:
    err_L = (B / L) * dL 

    # Total Absolute Error (Quadrature)
    dB = np.sqrt(err_I**2 + err_r**2 + err_L**2)

    return B, dB

#%% Linear calibration of manometer
start_idx = 100
ab_data = np.array([0.009184, 0.009409, 0.009521, 0.0096367, 0.009879, 0.010739, 0.011039, 0.011314, 0.01145, 0.01183, 0.012097, 0.01238, 0.012683, 0.012895, 0.01358, 0.014367, 0.015075, 0.017633, 0.019340, 0.024503, 0.031261])
mano_data = np.array([0.953671, 0.901634, 0.875302, 0.849757, 0.799083, 0.654431, 0.612501, 0.577452, 0.562078, 0.519941, 0.493119, 0.467356, 0.441701, 0.425199, 0.377189, 0.331608, 0.297266, 0.209065, 0.170809, 0.102019, 0.0602943])
p_data = np.array([1001, 947, 919, 892, 838, 687, 643, 606, 590, 546, 517, 491, 464, 446, 396, 348, 311, 219, 179, 107, 63])*100

ab_err = ab_data * 0.00005 + 0.0000035
p_minus = p_data / 1.03 - 100
p_plus  = p_data / 0.97 + 100
mano_err = 0.00004 * mano_data + 0.0007

popt_lin_c, pcov_lin_c = curve_fit(lin_fit, mano_data, p_data)
popt_lin_minus, _ = curve_fit(lin_fit, mano_data-mano_err, p_plus)
popt_lin_plus, _  = curve_fit(lin_fit, mano_data+mano_err, p_minus)

x_plot = np.linspace(mano_data.min(), mano_data.max()+.08, 300)
plt.figure(figsize=(8, 6), dpi=100)
plt.scatter(mano_data, p_data, label='Data', color='black', s=15)
plt.plot(x_plot, lin_fit(x_plot, *popt_lin_c), label='Central fit', color='tab:blue')
plt.plot(x_plot, lin_fit(x_plot, *popt_lin_minus), '--', label='Lower bound fit', color='red', alpha=.7)
plt.plot(x_plot, lin_fit(x_plot, *popt_lin_plus), '--', label='Upper bound fit', color='orange', alpha=.7)
yerr_minus = p_data - p_minus
yerr_plus = p_plus - p_data
plt.errorbar(mano_data, p_data, yerr=[yerr_minus, yerr_plus], xerr=mano_err, fmt='none', ecolor='gray', alpha=0.7, capsize=3)
plt.xlabel(r'$U_{\mathrm{Mano}}$ (V)')
plt.ylabel(r'$P$ (Pa)')
plt.title(r'Linear fit $p_{He}$ vs. $U_M$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0,1)
plt.show()

# --- Calculation of Fit Parameters and Errors ---
m_c, b_c = popt_lin_c
m_min, b_min = popt_lin_minus
m_max, b_max = popt_lin_plus

# Maximal absolute deviations
dm_plus = abs(m_max - m_c)
dm_minus = abs(m_c - m_min)
db_plus = abs(b_max - b_c)
db_minus = abs(b_c - b_min)

# Print results
print("-" * 30)
print(f"LINEAR FIT RESULTS (p = m * U_mano + b)")
print("-" * 30)
print(f"Slope (m):     {m_c:10.2f}  [+{dm_plus:8.2f} / -{dm_minus:8.2f}] Pa/V")
print(f"Intercept (b): {b_c:10.2f}  [+{db_plus:8.2f} / -{db_minus:8.2f}] Pa")
print("-" * 30)

#%% Calibration of U_AB vs T
mask_pressure = 5000  # Pa
ab_cal = np.array(cal_data[start_idx:, 1])
mano_cal = cal_data[start_idx:, 2]
ab_err = 0.00005 * ab_cal + 0.0000035
p_cal_c = lin_fit(mano_cal, *popt_lin_c)

# Pressure bounds
p_cal_minus = p_cal_c / 1.03 + 100
p_cal_plus  = p_cal_c / 0.97 + 100
p_floor = 5
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

low_mask = p_cal_c < mask_pressure
high_mask = p_cal_c >= mask_pressure
temp_low = temp0[low_mask]
temp_high = temp0[high_mask]
y_low = ab_cal[low_mask]
y_high = ab_cal[high_mask]

popt_low, pcov_low = curve_fit(fit_func1, temp_low, y_low, p0=[0.01, 0.01, 1.0], maxfev=10000)
popt_high, pcov_high = curve_fit(fit_func1, temp_high, y_high, p0=[0.01, 0.01, 1.0], maxfev=10000)

perr_low = np.sqrt(np.diag(pcov_low))
perr_high = np.sqrt(np.diag(pcov_high))

#%% Plot calibration
T_low_plot = np.linspace(temp_low.min(), temp_low.max(), 400)
T_high_plot = np.linspace(temp_high.min(), temp_high.max(), 400)

plt.figure(figsize=(8, 6), dpi=150)
plt.errorbar(temp_low, y_low, xerr=[dT_minus[low_mask], dT_plus[low_mask]], yerr=ab_err[low_mask], fmt='b.', ecolor='lightblue', elinewidth=3, capsize=1, ms=1, zorder=1, alpha=.6)
plt.errorbar(temp_high, y_high, xerr=[dT_minus[high_mask], dT_plus[high_mask]], yerr=ab_err[high_mask], fmt='r.', ecolor='lightblue', elinewidth=3, capsize=1, ms=1, zorder=1, alpha=.6)
plt.plot(T_low_plot, fit_func1(T_low_plot, *popt_low), color='black', lw=1, label='Fit low regime', zorder=2)
plt.plot(T_high_plot, fit_func1(T_high_plot, *popt_high), color='orange', lw=1, label='Fit high regime')
plt.axvline(x=2.1768, color='gray', linestyle='--', label='Lambda point', zorder=2)
plt.xlabel('Temperature (K)')
plt.ylabel(r'$U_{AB}$ (V)')
plt.title(r'Temperature calibration curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("a, b and c in low = " + f"{popt_low[0]:.4f}, {popt_low[1]:.4f}, {popt_low[2]:.4f}")
print("a, b and c in high = " + f"{popt_high[0]:.4f}, {popt_high[1]:.4f}, {popt_high[2]:.4f}")
#%% Critical field data extraction
ciritcal_field_path = r'C:\coding\my_projects\F-Praktikum\Supercondcutivity\sc_data\ciritcal_field_data'
def extract_number(filename):
    return int(re.search(r'(\d+)\.dat$', filename).group(1)) # type: ignore

files = sorted([f for f in os.listdir(ciritcal_field_path) if f.endswith('.dat')], key=extract_number)
critical_field_data = []
for filename in files:
    with open(os.path.join(ciritcal_field_path, filename)) as f:
        data = np.loadtxt(StringIO(f.read().replace(',', '.')), skiprows=15)
        critical_field_data.append(data)

#%% Plot single example measurement
sn_ex = critical_field_data[5][:,3]
coil_ex = critical_field_data[5][:,4]
coil_err = coil_ex*0.00005 + 0.0000035

xerr_ex = []
H_values = []
yerr_ex = sn_ex*0.00005 + 0.0000035


for V, dV in zip(coil_ex, coil_err):
    H, dH = calculate_h_error(V, dV)
    H_values.append(H)
    xerr_ex.append(dH)

H_array = np.array(H_values)
H_err_array = np.array(xerr_ex)

plt.figure(figsize=(8,6))
plt.errorbar(H_values[:100], sn_ex[:100], xerr=H_err_array[:100], yerr=yerr_ex[:100], ms=11, label="up sweep", elinewidth=1, capsize=1)
plt.errorbar(H_values[100:], sn_ex[100:], xerr=H_err_array[100:], yerr=yerr_ex[100:], ms=1, label="down sweep", elinewidth=1, capsize=1)
plt.title(r"Critical field measurement at $T\simeq3.11\,K$")
plt.xlabel(r"Magnetic field $B \,(T)$")
plt.ylabel(r"$U_{Sn}\; (V)$")
plt.grid(True, alpha=.3)
plt.legend()
plt.show()

#%% Extract transition voltages and fit H_c(T)
# --- extract transition voltages and average AB voltages ---
transition_voltages = np.zeros(len(critical_field_data))
ab_mean = np.zeros(len(critical_field_data))

for i, m in enumerate(critical_field_data):
    sn_data = m[:, 3]
    coil_data = m[:, 4]
    ab_data = m[:, 1]

    order = np.argsort(coil_data)
    sn_data = sn_data[order]
    coil_data = coil_data[order]
    ab_data = ab_data[order]

    diff_sn = np.abs(np.diff(sn_data))
    idx = np.argmax(diff_sn)

    # midpoint for transition voltage
    transition_voltages[i] = 0.5 * (coil_data[idx] + coil_data[idx+1])

    # average AB voltage around transition
    lo = max(idx-20, 0)
    hi = min(idx+20, len(ab_data))
    ab_mean[i] = np.mean(ab_data[lo:hi])

def get_temp_with_err(U_AB, regime):
    """
    Calculates T and its asymmetric maximal errors by shifting fit parameters.
    """
    if regime == 'low':
        p, p_err = popt_low, perr_low
    elif regime == 'high':
        p, p_err = popt_high, perr_high
    else:
        raise ValueError("Regime must be 'low' or 'high'")

    # Current U_AB measurement error
    dU = 0.00005 * U_AB + 0.0000035
    
    # 1. Central Temperature
    T_central = temp_from_ab(U_AB, regime)
    
    # 2. Maximal Error Analysis: 
    # We shift the parameters (a, b, c1) by their standard deviations 
    # in directions that create the min/max temperature.
    
    # Helper to calculate T from specific parameters
    def T_calc(u, a, b, c):
        arg = (u - a) / b
        # Ensure we don't log a negative number during shifts
        if arg <= 0: return np.nan
        return -c * np.log(arg)

    # Calculate variations (shfting parameters to extremes)
    # Note: For -c * ln((U-a)/b), increasing 'a' and 'b' and 'c' usually increases T
    T_plus_params = T_calc(U_AB + dU, p[0]+p_err[0], p[1]+p_err[1], p[2]+p_err[2])
    T_minus_params = T_calc(U_AB - dU, p[0]-p_err[0], p[1]-p_err[1], p[2]-p_err[2])
    
    # Asymmetric errors
    dT_plus = np.abs(T_plus_params - T_central)
    dT_minus = np.abs(T_central - T_minus_params)
    
    return T_central, dT_minus, dT_plus

# --- calculate magnetic field and its error properly ---
transition_voltages_err = transition_voltages * 0.00004 + 0.000007

field_values = []
field_err = []

for V, dV in zip(transition_voltages, transition_voltages_err):
    H, dH = calculate_h_error(V, dV)
    field_values.append(H)
    field_err.append(dH)

field_values = np.array(field_values)
field_err = np.array(field_err)

# --- calculate temperature and asymmetric errors using safe regime ---
T_vals = []
dT_lows = []
dT_highs = []
dTc = []
for U_AB in ab_mean:
    # use your existing temp_from_ab_safe
    Tc, regime = temp_from_ab_safe(U_AB)
    
    # compute asymmetric errors for this regime
    dTc, dTc_low, dTc_high = get_temp_with_err(U_AB, regime)  # make sure get_temp_err returns (dT_minus, dT_plus)
    
    T_vals.append(Tc)
    dT_lows.append(dTc_low)
    dT_highs.append(dTc_high)

T_vals = np.array(T_vals)
dT_lows = np.array(dT_lows)
dT_highs = np.array(dT_highs)

# --- calculate magnetic field and its error properly ---
field_values = []
field_err = []

for V, dV in zip(transition_voltages, transition_voltages_err):
    H, dH = calculate_h_error(V, dV)
    field_values.append(H)
    field_err.append(dH)

field_values = np.array(field_values)
field_err = np.array(field_err)



t_array = np.linspace(1, 3.8, 100)

# --- theoretical curve ---
def theoretical_field(T):
    T_c = 3.722  # K (tin)
    B_0 = 0.0305  # Tesla
    return B_0 * (1 - (T / T_c)**2)

t_array = np.linspace(1, 3.9, 100)

# --- fit H_c(T) to extract H_0 and T_c ---
def fit_func2(T, H_0, T_c):
    return H_0 * (1 - (T / T_c)**2)
# %%
# central fit
popt_central, _ = curve_fit(fit_func2, T_vals, field_values, p0=[0.03, 3.7], maxfev=10000)

# upper bound fit (add errors)
T_upper = T_vals - dT_highs
H_upper = field_values - field_err
popt_upper, _ = curve_fit(fit_func2, T_upper, H_upper, p0=[0.03, 3.7], maxfev=10000)

# lower bound fit (subtract errors)
T_lower = T_vals + dT_lows
H_lower = field_values + field_err
popt_lower, _ = curve_fit(fit_func2, T_lower, H_lower, p0=[0.03, 3.7], maxfev=10000)

# generate smooth curve for plotting
t_array = np.linspace(0, 4.2, 200)
H_central = fit_func2(t_array, *popt_central)
H_max     = fit_func2(t_array, *popt_upper)
H_min     = fit_func2(t_array, *popt_lower)

# plot
plt.figure(figsize=(8,5))
plt.plot(t_array, H_max, label='maximum error fit 1')
plt.plot(t_array, H_min, label='maximum error fit 2')
plt.plot(t_array, H_central, '-', color='blue', label='Central fit')
plt.errorbar(T_vals, field_values, xerr=[dT_lows, dT_highs], yerr=field_err, fmt='o', color='black', capsize=3, label='Extracted $B_c(T)$', markersize=3)
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$B_c$ (T)')
plt.title('Critical Field vs. Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0,4.2)
plt.ylim(0,.036)
plt.show()

# %%
# Central fit
popt_central, _ = curve_fit(fit_func2, T_vals, field_values, p0=[0.03, 3.7], maxfev=10000)
H0_central, Tc_central = popt_central

# Upper bound fit (T + dT_highs, H + field_err)
popt_upper, _ = curve_fit(
    fit_func2,
    T_vals + dT_highs,
    field_values + field_err,
    p0=[0.03, 3.7],
    maxfev=10000
)
H0_upper, Tc_upper = popt_upper

# Lower bound fit (T - dT_lows, H - field_err)
popt_lower, _ = curve_fit(
    fit_func2,
    T_vals - dT_lows,
    field_values - field_err,
    p0=[0.03, 3.7],
    maxfev=10000
)
H0_lower, Tc_lower = popt_lower

# Conservative asymmetric error estimate
H0_err = max(abs(H0_upper - H0_central), abs(H0_central - H0_lower))
Tc_err = max(abs(Tc_upper - Tc_central), abs(Tc_central - Tc_lower))

print(f"Maximal-error analysis results:")
print(f"H_0 = {H0_central:.5f} ± {H0_err:.5f} T")
print(f"T_c = {Tc_central:.5f} ± {Tc_err:.5f} K")
# %%
