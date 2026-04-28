# %%
import os
import re
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from io import StringIO

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20

def read_and_integrate(filepath, starttime, endtime):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
        cols = df.columns.tolist()
        if len(cols) < 2: return 0
        df = df.iloc[:, :2]
        df.columns = ['Time', 'Counts']
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Counts'] = pd.to_numeric(df['Counts'], errors='coerce')
        df = df.dropna()
        sig_mask = (df['Time'] >= starttime) & (df['Time'] <= endtime)
        signal = df.loc[sig_mask, 'Counts'].sum() #type: ignore
        return max(0, signal)
    except Exception: return 0

def get_angle_from_filename(filename):
    match_both = re.search(r'HWP1_(\d+)p?(\d*)_HWP2_(\d+)p?(\d*)', filename)
    if match_both:
        h1 = float(match_both.group(1) + '.' + (match_both.group(2) if match_both.group(2) else '0'))
        h2 = float(match_both.group(3) + '.' + (match_both.group(4) if match_both.group(4) else '0'))
        return h1, h2
    match_h1 = re.search(r'HWP1_(\d+)p?(\d*)', filename)
    if match_h1:
        h1 = float(match_h1.group(1) + '.' + (match_h1.group(2) if match_h1.group(2) else '0'))
        return h1, 0.0
    return None, None
    
def process_folder(folderpath, starttime, endtime):
    data = []
    if not os.path.exists(folderpath): return pd.DataFrame()
    for file in os.listdir(folderpath):
        h1, h2 = get_angle_from_filename(file)
        if h1 is not None and h2 is not None:
            c = read_and_integrate(os.path.join(folderpath, file), starttime, endtime)
            data.append({'HWP1': h1, 'HWP2': h2, 'Counts': c})
    df = pd.DataFrame(data)
    return df.sort_values(by='HWP2').reset_index(drop=True) if not df.empty else df

def process_parent_folder(parent_folder, starttime, endtime):
    all_data = []
    if not os.path.exists(parent_folder): return pd.DataFrame()
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            df = process_folder(subfolder_path, starttime, endtime)
            if not df.empty:
                df['SourceFolder'] = subfolder
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def model(theta_2, theta_1, phi, amp, offset):
    return amp * np.sin(phi + 2 * (np.deg2rad(theta_1) - np.deg2rad(theta_2)))**2 + offset

def fit_sin(x_data, y_data, fixed_theta1):
    fit_func = lambda x, phi, amp, offset: model(x, fixed_theta1, phi, amp, offset)
    weights = np.sqrt(np.maximum(y_data, 1))
    p0 = [.2, np.max(y_data) - np.min(y_data), np.min(y_data)] 
    popt, pcov = curve_fit(fit_func, x_data, y_data, p0=p0, sigma=weights, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov)) 
    return popt, fit_func, pcov

def calc_vis_with_err(amp, offset, pcov):
    denom = abs(amp) + 2 * offset
    if denom <= 0:
        return 0, 0

    v = abs(amp) / denom

    dV_da = (2 * offset) / denom**2
    dV_do = (-2 * abs(amp)) / denom**2

    var = (
        dV_da**2 * pcov[1,1] +
        dV_do**2 * pcov[2,2] +
        2 * dV_da * dV_do * pcov[1,2]
    )

    return v, np.sqrt(var)

def get_E_interpolated(a, b, params_dict):
    try:
        c_ab    = model(b, a, *params_dict[a])
        c_apbp  = model(b + 45, a + 45, *params_dict[a + 45])
        c_apb   = model(b, a + 45, *params_dict[a + 45])
        c_abp   = model(b + 45, a, *params_dict[a])
        n_total = c_ab + c_apbp + c_apb + c_abp
        if n_total <= 0: return 0, 1e-6 
        num, den = (c_ab + c_apbp - c_apb - c_abp), n_total
        e_val = num / den
        e_err = np.sqrt(((den - num) / den**2)**2 * (c_ab + c_apbp) + ((den + num) / den**2)**2 * (c_apb + c_abp))
        return e_val, e_err
    except KeyError as e:
        print(f"Missing parameter for HWP1={e} in S calculation.")
        return 0, 0

# --- Data Loading ---
folderpath0 = r'C:\coding\my_projects\F-Praktikum\Bells\pumping_from_both_sides_DHWP_0'
folderpath45 = r'C:\coding\my_projects\F-Praktikum\Bells\pumping_from_both_sides_DHWP_45'
df0 = process_folder(folderpath0, -180, 40)
df45 = process_parent_folder(folderpath45, -100, 60)

# --- Analysis for DF0 ---
if not df0.empty:
    plt.figure(figsize=(12, 6)); ax0 = plt.gca()
    fits_params0, leg_h0, vis_h0 = {}, [], []
    print("--- DHWP0 Results (Poisson Weighted) ---")
    
    # FIXED: Skip HWP1 = 0.0 using a conditional check and fixed x_f definition
    for hwp1, group in df0.groupby('HWP1'):
#        if hwp1 == 0.0: continue 
        
        # FIXED: Ensure x_f and y_f use the local group data
        x_f = group['HWP2']
        y_f = group['Counts']
        
        try:
            popt, m_func, pcov = fit_sin(x_f, y_f, hwp1)
            fits_params0[hwp1] = popt
            v, v_e = calc_vis_with_err(popt[1], popt[2], pcov)
            print(f"HWP1={hwp1}°: V = {v*100:.2f} ± {v_e*100:.2f}%")
            xr = np.linspace(group['HWP2'].min(), group['HWP2'].max(), 200)
            line, = ax0.plot(xr, m_func(xr, *popt), label=rf'{hwp1}°')
            ax0.errorbar(group['HWP2'], group['Counts'], yerr=np.sqrt(group['Counts']), fmt='o', color=line.get_color(), mfc='none')
            leg_h0.append(line)
            vis_h0.append(plt.Line2D([0], [0], linestyle='none', marker='o', #type: ignore
                color=line.get_color(), mfc='none', label=f'{v*100:.1f}±{v_e*100:.1f}%'))
        except Exception as e: print(f"Fit error HWP1={hwp1}: {e}")
    
    l1 = ax0.legend(handles=leg_h0, title=r"$\theta_1$", bbox_to_anchor=(1.01, 1), loc='upper left'); ax0.add_artist(l1)
    ax0.legend(handles=vis_h0, title="Visibilities", bbox_to_anchor=(1.01, 0.25), loc='upper left')
    plt.xlabel(r"$\theta_2$ (degrees)"); plt.ylabel("Coincidences")
    plt.tight_layout(rect=[0,0,.85,1]); plt.show() #type: ignore
    
    # NOTE: If you exclude 0.0, the S calculation below may fail if it expects 0.0
    e1, er1 = get_E_interpolated(0.0, 11.25, fits_params0)
    e2, er2 = get_E_interpolated(0.0, 33.75, fits_params0)
    e3, er3 = get_E_interpolated(22.5, 11.25, fits_params0)
    e4, er4 = get_E_interpolated(22.5, 33.75, fits_params0)
    s0 = e1 - e2 + e3 + e4
    s0_e = np.sqrt(er1**2 + er2**2 + er3**2 + er4**2)
    print(f"DHWP0 S = {s0:.4f} ± {s0_e:.4f} ({(abs(s0)-2)/s0_e:.1f} sigma)\n")
#%%
# --- Analysis for DF45 ---
if not df45.empty:
    plt.figure(figsize=(12, 6)); ax45 = plt.gca()
    fits_params45, leg_h45, vis_h45 = {}, [], []
    print("--- DHWP45 Results (Poisson Weighted) ---")
    for hwp1, group in df45.groupby('HWP1'):
        x_f_45 = group['HWP2'].iloc[1:] if hwp1 == 0.0 else group['HWP2']
        y_f_45 = group['Counts'].iloc[1:] if hwp1 == 0.0 else group['Counts']
        try:
            popt, m_func, pcov = fit_sin(x_f_45, y_f_45, hwp1)
            fits_params45[hwp1] = popt
            v, v_e = calc_vis_with_err(popt[1], popt[2], pcov)
            print(f"HWP1={hwp1}°: V = {v*100:.2f} ± {v_e*100:.3f}%")
            xr = np.linspace(group['HWP2'].min(), group['HWP2'].max(), 200)
            line, = ax45.plot(xr, m_func(xr, *popt), label=rf'{hwp1}°')
            ax45.errorbar(group['HWP2'], group['Counts'], yerr=np.sqrt(group['Counts']), fmt='o', color=line.get_color(), mfc='none')
            leg_h45.append(line)
            vis_h45.append(plt.Line2D([0], [0], linestyle='none', marker='o', #type: ignore
                color=line.get_color(), mfc='none', label=f'{v*100:.1f}±{v_e*100:.1f}%'))
        except Exception as e: print(f"Fit error HWP1={hwp1}: {e}")
    l2 = ax45.legend(handles=leg_h45, title=r"$\theta_1$", bbox_to_anchor=(1.01, 1), loc='upper left'); ax45.add_artist(l2)
    ax45.legend(handles=vis_h45, title="Visibilities", bbox_to_anchor=(1.01, 0.25), loc='upper left')
    plt.xlabel(r"$\theta_2$ (degrees)"); plt.ylabel("Coincidences")
    plt.tight_layout(rect=[0,0,.85,1]); plt.show() #type: ignore

    e1, er1 = get_E_interpolated(0.0, 11.25, fits_params45)
    e2, er2 = get_E_interpolated(0.0, 33.75, fits_params45)
    e3, er3 = get_E_interpolated(22.5, 11.25, fits_params45)
    e4, er4 = get_E_interpolated(22.5, 33.75, fits_params45)
    s45 = e1 - e2 + e3 + e4
    s45_e = np.sqrt(er1**2 + er2**2 + er3**2 + er4**2)
    print(f"DHWP45 S = {s45:.4f} ± {s45_e:.4f} ({(abs(s45)-2)/s45_e:.1f} sigma)")
# %%
import numpy as np
import matplotlib.pyplot as plt

# --- HWP rotation matrix ---
def hwp_matrix(theta_deg):
    rad = np.deg2rad(2*theta_deg)  # HWP doubles angle
    return np.array([[np.cos(rad), np.sin(rad)],
                     [np.sin(rad), -np.cos(rad)]])

# --- H V product state projection ---
theta1_vals = [0, 22.5, 45, 67.5] 
theta2_vals = np.linspace(-1e-2, 92, 361)                 

# Define |H> and |V> as column vectors
H = np.array([[1],[0]])
V = np.array([[0],[1]])


plt.figure(figsize=(8,5))

for t1 in theta1_vals:
    U1 = hwp_matrix(t1)
    probs = []
    for t2 in theta2_vals:
        amp = np.sin(np.deg2rad((2*t1))) * np.cos(np.deg2rad((2*t2)))
        #amp = np.cos(2*np.deg2rad(t2 + t1))**2
        probs.append(np.abs(amp)**2)
    plt.plot(theta2_vals, probs, label=f'{t1}°')

plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel('Expected counts (normalized)')
plt.legend(ncol = 2, title=r'$\theta_1$')
plt.xticks(np.arange(0, 91, 15))
plt.xlim(0, 90)
plt.ylim(-1.5e-3, 1.01)
plt.show()

# %%
