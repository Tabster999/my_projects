import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import StringIO
import scipy.constants as const

# =============================================================================
# 1. SETTINGS & CONSTANTS
# =============================================================================
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})
T_LAMBDA = 2.1768
MASK_PRESSURE = 4984  # Pa threshold between low/high helium regimes
R_SERIES = 384.5e-6
N_COIL = 10353

# =============================================================================
# 2. CALIBRATION FUNCTIONS (ITS-90 & FITS)
# =============================================================================

def temp_from_pressure(p, regime):
    """ITS-90 standard for Helium vapour pressure."""
    if regime == 'low':
        a = [1.392408, 0.527153, 0.166756, 0.050988, 0.026514, 0.001975, -0.017976, 0.005409, 0.013259, 0]
        b, c = 5.6, 2.9
    else:
        a = [3.146631, 1.357655, 0.413923, 0.091159, 0.016349, 0.001826, -0.004325, -0.004973, 0, 0]
        b, c = 10.3, 1.9
    x = (np.log(np.clip(p, 1, None)) - b) / c
    return a[0] + sum(val * x**i for i, val in enumerate(a[1:], start=1))

def fit_func_ab(T, a, b, c):
    """Exponential model for the AB-sensor voltage vs Temperature."""
    return a + b * np.exp(-T / c)

def lin_fit(x, m, b):
    return m * x + b

def calculate_h_error(V, dV):
    """Propagates coil geometry and voltage errors to Magnetic Field H (Tesla)."""
    r_i, dr_i = 10.5e-3, 0.5e-3
    r_o, dr_o = 20.4e-3, 0.5e-3
    L, dL = 193e-3, 3e-3
    I, dI = V / R_SERIES, dV / R_SERIES
    a = L / 2
    sqrt_o, sqrt_i = np.sqrt(r_o**2 + a**2), np.sqrt(r_i**2 + a**2)
    diff_r = r_o - r_i
    log_val = np.log((r_o + sqrt_o) / (r_i + sqrt_i))
    
    H = (0.5 * N_COIL * I / diff_r) * log_val
    dH_dI = (N_COIL / (2 * diff_r)) * log_val
    dH_dro = (N_COIL * I / (2 * diff_r)) * (1/sqrt_o - log_val/diff_r)
    dH_dri = (N_COIL * I / (2 * diff_r)) * (log_val/diff_r - 1/sqrt_i)
    term_o, term_i = L/(sqrt_o*(r_o+sqrt_o)), L/(sqrt_i*(r_i+sqrt_i))
    dH_dL = (N_COIL * I / (8 * diff_r)) * (term_o - term_i)
    
    delta_H = np.sqrt((dH_dI*dI)**2 + (dH_dro*dr_o)**2 + (dH_dri*dr_i)**2 + (dH_dL*dL)**2)
    return H * const.mu_0, delta_H * const.mu_0

# =============================================================================
# 3. LOADING DATA & INITIAL CALIBRATION
# =============================================================================

# --- Load Calibration Data ---
with open(r'C:/coding/my_projects/F-Praktikum/Supercondcutivity/sc_data/SanC_TalH_Supraleitung_0.dat') as f:
    cal_raw = np.loadtxt(StringIO(f.read().replace(',', '.')), skiprows=15)

# --- Step A: Manometer to Pressure ---
# (Using your manually entered points for the manometer fit)
mano_data = np.array([0.953671, 0.901634, 0.849757, 0.654431, 0.441701, 0.102019, 0.0602943])
p_data = np.array([1001, 947, 892, 687, 464, 107, 63]) * 100
popt_lin_c, _ = curve_fit(lin_fit, mano_data, p_data)

# --- Step B: Pressure to Temperature (The "Reference" T) ---
u_ab_cal = cal_raw[100:, 1]
u_mano_cal = cal_raw[100:, 2]

p_cal_c = lin_fit(u_mano_cal, *popt_lin_c)
p_cal_minus, p_cal_plus = p_cal_c / 1.03 - 100, p_cal_c / 0.97 + 100

T_ref = np.array([temp_from_pressure(p, 'low' if p < MASK_PRESSURE else 'high') for p in p_cal_c])
T_ref_low = np.array([temp_from_pressure(p, 'low' if p < MASK_PRESSURE else 'high') for p in p_cal_minus])
T_ref_high = np.array([temp_from_pressure(p, 'low' if p < MASK_PRESSURE else 'high') for p in p_cal_plus])

# THE ERROR BARS FOR THE LOOK-UP TABLE
dT_cal_minus = np.abs(T_ref - T_ref_low)
dT_cal_plus = np.abs(T_ref_high - T_ref)

# --- Step C: Final AB-Sensor Fit ---
mask_low = p_cal_c < MASK_PRESSURE
popt_low, _ = curve_fit(fit_func_ab, T_ref[mask_low], u_ab_cal[mask_low], p0=[0.01, 0.01, 1.0])
popt_high, _ = curve_fit(fit_func_ab, T_ref[~mask_low], u_ab_cal[~mask_low], p0=[0.01, 0.01, 1.0])

# =============================================================================
# 4. THE MAPPING ENGINE (Inherited Error Approach)
# =============================================================================

def get_T_with_inherited_err(U_AB):
    """Maps measured voltage to T and inherits calibration errors via look-up."""
    # 1. Direct Conversion via Fit
    regime = 'low' if U_AB > 0.0341 else 'high'
    popt = popt_low if regime == 'low' else popt_high
    
    # Inverse of a + b*exp(-T/c) -> T = -c * ln((U-a)/b)
    arg = (U_AB - popt[0]) / popt[1]
    T_val = -popt[2] * np.log(np.clip(arg, 1e-9, None))
    
    # 2. Inherit Errors from Calibration Look-up
    idx = np.argmin(np.abs(u_ab_cal - U_AB))
    return T_val, dT_cal_minus[idx], dT_cal_plus[idx]

# =============================================================================
# 5. ANALYSIS: H_c vs T
# =============================================================================

critical_field_path = r'C:\coding\my_projects\F-Praktikum\Supercondcutivity\sc_data\ciritcal_field_data'
files = sorted([f for f in os.listdir(critical_field_path) if f.endswith('.dat')], 
               key=lambda x: int(re.search(r'(\d+)\.dat$', x).group(1))) # type: ignore

h_c_vals, h_c_errs = [], []
t_trans_vals, t_err_m, t_err_p = [], [], []

for filename in files:
    data = np.loadtxt(os.path.join(critical_field_path, filename), skiprows=15).replace(',', '.') # type: ignore
    
    # Identify transition via steepness in Sn voltage
    sn, coil, ab = data[:, 3], data[:, 4], data[:, 1]
    idx = np.argmax(np.abs(np.diff(sn)))
    
    # H_c calculation
    u_trans = 0.5 * (coil[idx] + coil[idx+1])
    du_trans = u_trans * 0.00005 + 0.0000035
    h, dh = calculate_h_error(u_trans, du_trans)
    h_c_vals.append(h)
    h_c_errs.append(dh)
    
    # T at transition (Inherited)
    u_ab_mean = np.mean(ab[max(0, idx-10):idx+10])
    t, em, ep = get_T_with_inherited_err(u_ab_mean)
    t_trans_vals.append(t)
    t_err_m.append(em)
    t_err_p.append(ep)

# --- Fit H_c vs T ---
def hc_fit_model(T, H0, Tc):
    return H0 * (1 - (T/Tc)**2)

popt_hc, _ = curve_fit(hc_fit_model, t_trans_vals, h_c_vals, p0=[0.03, 3.7])

# =============================================================================
# 6. FINAL T_c EXTRACTION (Hysteresis Analysis)
# =============================================================================
# Using your transition values from the direct Tc sweeps
low_U = np.array([0.009409, 0.009521, 0.009636, 0.009879]) 
high_U = np.array([0.010739, 0.011039, 0.011314, 0.01145])

res_low = [get_T_with_inherited_err(u) for u in low_U]
res_high = [get_T_with_inherited_err(u) for u in high_U]

t_l = np.array([r[0] for r in res_low]); em_l = np.array([r[1] for r in res_low])
t_h = np.array([r[0] for r in res_high]); em_h = np.array([r[1] for r in res_high])

tc_mid = 0.5 * (t_l + t_h)
tc_sys_hyst = 0.5 * np.abs(t_h - t_l)
tc_stat_cal = 0.5 * np.sqrt(em_l**2 + em_h**2)

final_err = np.sqrt(tc_sys_hyst**2 + tc_stat_cal**2)

print(f"Final T_c from Sweeps: {np.mean(tc_mid):.4f} ± {np.mean(final_err):.4f} K")
print(f"Fit Parameters: H0={popt_hc[0]:.4f} T, Tc={popt_hc[1]:.4f} K")

# =============================================================================
# 7. PLOTTING
# =============================================================================
plt.figure(figsize=(10, 6))
plt.errorbar(t_trans_vals, h_c_vals, xerr=[t_err_m, t_err_p], yerr=h_c_errs, fmt='ro', label='Data')
t_fine = np.linspace(1, 4, 100)
plt.plot(t_fine, hc_fit_model(t_fine, *popt_hc), 'b-', label='Fit $H_c(T)$')
plt.xlabel('Temperature (K)')
plt.ylabel('Critical Field $H_c$ (T)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()