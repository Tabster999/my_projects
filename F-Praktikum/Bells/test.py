#%%
import os
import re
from turtle import title
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from io import StringIO
#%% Define functions and settings

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20

def read_and_integrate(filepath, starttime, endtime):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#')

        cols = df.columns.tolist()

        if len(cols) < 2:
            print(f"Bad format in {filepath}")
            return 0

        df = df.iloc[:, :2]
        df.columns = ['Time', 'Counts']

        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Counts'] = pd.to_numeric(df['Counts'], errors='coerce')

        df = df.dropna()

        sig_mask = (df['Time'] >= starttime) & (df['Time'] <= endtime)

        if sig_mask.sum() == 0:
            print(f"No data for {filepath}")

        signal = df.loc[sig_mask, 'Counts'].sum() #type: ignore

        return max(0, signal)

    except Exception as e:
        print(f"Error in {filepath}: {e}")
        return 0

def nearest_angle(df, h1, h2_target):
    h2 = df[df['HWP1']==h1]['HWP2'].values
    h2_nearest = h2[np.argmin(np.abs(h2 - h2_target))]
    return h2_nearest


def get_angle_from_filename(filename):
    match_both = re.search(r'HWP1_(\d+)p?(\d*)_HWP2_(\d+)p?(\d*)', filename)
    if match_both:
        h1 = float(match_both.group(1) + '.' + (match_both.group(2) if match_both.group(2) else '0'))
        h2 = float(match_both.group(3) + '.' + (match_both.group(4) if match_both.group(4) else '0'))
        return h1, h2
    
    match_h1 = re.search(r'HWP1_(\d+)p?(\d*)', filename)
    if match_h1:
        h1 = float(match_h1.group(1) + '.' + (match_h1.group(2) if match_h1.group(2) else '0'))
        h2 = 0.0  
        return h1, h2
    
    print(f"Angle not found in filename: {filename}")
    return None, None
    
def process_folder(folderpath, starttime, endtime):
    data = []
    if not os.path.exists(folderpath):
        print(f"Path does not exist: {folderpath}")
        return pd.DataFrame()
    
    for file in os.listdir(folderpath):
            h1, h2 = get_angle_from_filename(file)
            if h1 is not None and h2 is not None:
                path = os.path.join(folderpath, file)
                c = read_and_integrate(path, starttime=starttime, endtime=endtime)
                data.append({'HWP1': h1, 'HWP2': h2, 'Counts': c})
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='HWP2').reset_index(drop=True)
    return df

def process_parent_folder(parent_folder, starttime, endtime):
    all_data = []

    if not os.path.exists(parent_folder):
        print(f"Parent path does not exist: {parent_folder}")
        return pd.DataFrame()

    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")

            df = process_folder(subfolder_path, starttime, endtime)

            if not df.empty:
                df['SourceFolder'] = subfolder  
                all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def get_count(df, h1, h2):
    val = df[(df['HWP1'] == h1) & (df['HWP2'] == h2)]['Counts']
    if len(val) == 0:
        print(f"Missing data for ({h1}, {h2})")
        return 0
    return val.values[0]


def E(df, a, b):
    C_ab = get_count(df, a, b)
    C_apb = get_count(df, a + 45, b)
    C_abp = get_count(df, a, b + 45)
    C_apbp = get_count(df, a + 45, b + 45)

    numerator = C_ab + C_apbp - C_apb - C_abp
    denominator = C_ab + C_apbp + C_apb + C_abp

    if denominator == 0:
        return 0

    return numerator / denominator
def model(x_deg, a, phi, offset):
    return a * np.sin(2*(phi - np.radians(x_deg)))**2 + offset

def fit_sin(x, y):
    p0 = [np.max(y), 0, np.mean(y)] 
    popt, pcov = curve_fit(model, x, y, p0=p0)
    return popt, model

def E_with_error(df, a, b):
    counts = [
        get_count(df, a, b),          # C_ab
        get_count(df, a + 45, b + 45), # C_apbp
        get_count(df, a + 45, b),    # C_apb
        get_count(df, a, b + 45)     # C_abp
    ]

    C_ab = get_count(df, a, b)      # C_ab
    C_apbp = get_count(df, a+45, b+45) # C_apbp
    C_apb = get_count(df, a+45, b)    # C_apb
    C_abp = get_count(df, a, b+45)     # C_abp

    numerator = C_ab + C_apbp - C_apb - C_abp
    denominator = C_ab + C_apbp + C_apb + C_abp

    N_tot = sum(counts)
    if N_tot == 0: return 0, 0
    
    e_val = (counts[0] + counts[1] - counts[2] - counts[3]) / N_tot
    
    e_err = np.sqrt(((denominator - numerator) / denominator**2)**2 * (C_ab + C_apbp) + ((denominator + numerator) / denominator**2)**2 * (C_apb + C_abp))
    
    return e_val, e_err


#%% Read data and calculate counts
folderpath0 = r'C:\coding\my_projects\F-Praktikum\Bells\pumping_from_both_sides_DHWP_0'

folderpath45 = r'C:\coding\my_projects\F-Praktikum\Bells\pumping_from_both_sides_DHWP_45'

starttime0 = -180
endtime0 = 40
starttime45 = -100
endtime45 = 60
df0 = process_folder(folderpath0, starttime=starttime0, endtime=endtime0)
df45 = process_parent_folder(folderpath45, starttime=starttime45, endtime=endtime45)
#%% DHWP0 measurements: plotting, visibility calculation 

if not df0.empty:
    plt.figure(figsize=(8, 5))
    groups = list(df0.groupby('HWP1'))

    for hwp1, group in groups[1:]:
        plt.errorbar(group['HWP2'], group['Counts'], yerr=np.sqrt(group['Counts']), fmt='o-', label=f'HWP1 = {hwp1}°', capsize=4, markersize=5, alpha=0.8, elinewidth=2)
    
    plt.xlabel('HWP2 Angle (degrees)')
    plt.ylabel('Coincidence Counts')
    plt.legend()
    plt.title('Integrated Counts vs HWP2 Angle')
    plt.grid(True, alpha=0.3)
    plt.show()

    visibilities0 = []
    for hwp1, group in groups[1:]:
        max_counts = group['Counts'].max()
        min_counts = group['Counts'].min()
        if max_counts + min_counts > 0:
            V = (max_counts - min_counts) / (max_counts + min_counts)
            visibilities0.append((hwp1, V))
            print(f"DHWP0, HWP1 = {hwp1}°: Visibility = {V:.3f}")
        else:
            print(f"DHWP0, HWP1 = {hwp1}°: Cannot calculate visibility (max + min = 0)")
else:
    print("DataFrame empty")



a0   = 0.0
a_p0 = 22.5

b0  = 7.5
b_p0 = 30

E_ab0   = E(df0, a0, b0)
E_abp0  = E(df0, a0, b_p0)
E_apb0  = E(df0, a_p0, b0)
E_apbp0 = E(df0, a_p0, b_p0)

print("E0(a,b)   =", E_ab0)
print("E0(a,b')  =", E_abp0)
print("E0(a',b)  =", E_apb0)
print("E0(a',b') =", E_apbp0)

S0 = E_ab0 - E_abp0 + E_apb0 + E_apbp0
print("\nCHSH S =", S0)

E_results0 = [
    E_with_error(df0, 0.0, 7.5),   # E(a, b)
    E_with_error(df0, 0.0, 30.0),  # E(a, b')
    E_with_error(df0, 22.5, 7.5),  # E(a', b)
    E_with_error(df0, 22.5, 30.0)  # E(a', b')
]

vals0 = [res[0] for res in E_results0]
errs0 = [res[1] for res in E_results0]

# S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
S0_val = vals0[0] - vals0[1] + vals0[2] + vals0[3]
S0_err = np.sqrt(sum(np.array(errs0)**2))

print(f"S0 = {S0_val:.4f} ± {S0_err:.4f}")
print(f"Violation: {(np.abs(S0_val)-2)/S0_err:.1f} sigma")

if not df0.empty:
    fits_params0 = {}

    for hwp1, group in df0.groupby('HWP1'):
        x_data_0, y_data_0 = group['HWP2'], group['Counts']
        
        if hwp1 == 0.0:
            x_fit_in0, y_fit_in0 = x_data_0.iloc[1:], y_data_0.iloc[1:]
        else:
            x_fit_in0, y_fit_in0 = x_data_0, y_data_0
            
        try:
            popt0, _ = fit_sin(x_fit_in0, y_fit_in0)
            fits_params0[hwp1] = popt0
        except Exception as e:
            print(f"Fit failed for DHWP0, HWP1={hwp1}: {e}")

    def get_E_interpolated0(a, b):
        c_ab    = model(b, *fits_params0[a])
        c_apbp  = model(b + 45, *fits_params0[a + 45])
        c_apb   = model(b, *fits_params0[a + 45])
        c_abp   = model(b + 45, *fits_params0[a])

        n_total = c_ab + c_apbp + c_apb + c_abp

        if n_total <= 0: return 0, 0
        e_val = (c_ab + c_apbp - c_apb - c_abp) / n_total
        e_err = np.sqrt((1 - e_val**2) / n_total)

        return e_val, e_err

    a_0_idx, a_22_idx = 0.0, 22.5
    b_11_ideal, b_33_ideal = 11.25, 33.75

    try:
        e1_0, err1_0 = get_E_interpolated0(a_0_idx, b_11_ideal)
        e2_0, err2_0 = get_E_interpolated0(a_0_idx, b_33_ideal)
        e3_0, err3_0 = get_E_interpolated0(a_22_idx, b_11_ideal)
        e4_0, err4_0 = get_E_interpolated0(a_22_idx, b_33_ideal)

        S_val0 = e1_0 - e2_0 + e3_0 + e4_0
        S_err0 = np.sqrt(err1_0**2 + err2_0**2 + err3_0**2 + err4_0**2)

        print("\n--- DHWP0 Results (Interpolated from Fit) ---")
        print(f"E0(a, b)   = {e1_0:.4f} ± {err1_0:.4f}")
        print(f"E0(a, b')  = {e2_0:.4f} ± {err2_0:.4f}")
        print(f"E0(a', b)  = {e3_0:.4f} ± {err3_0:.4f}")
        print(f"E0(a', b') = {e4_0:.4f} ± {err4_0:.4f}")
        print(f"S0 (Fit)   = {S_val0:.4f} ± {S_err0:.4f}")

        n_sigma0 = (abs(S_val0) - 2) / S_err0
        print(f"Violation: {n_sigma0:.2f} sigma")
        
    except KeyError as e:
        print(f"Missing HWP1 key in fits_params0: {e}")

    plt.figure(figsize=(10, 6))
    for hwp1, group in df0.groupby('HWP1'):
        x_data_plot = group['HWP2']
        y_data_plot = group['Counts']
        
        line, = plt.plot(x_data_plot, y_data_plot, 'o', markersize=4, label=f'{hwp1}°')
        plt.errorbar(x_data_plot, y_data_plot, yerr=np.sqrt(y_data_plot), fmt='none', 
                     color=line.get_color(), capsize=4, alpha=0.5, elinewidth=4)
        
        if hwp1 in fits_params0:
            x_fit_range = np.linspace(x_data_plot.min(), x_data_plot.max(), 200)
            y_fit_curve = model(x_fit_range, *fits_params0[hwp1])
            plt.plot(x_fit_range, y_fit_curve, '--', color=line.get_color(), alpha=0.8)

    plt.xlabel(r"$\theta_2$ (degrees)")
    plt.ylabel("Coincidences")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=r"$\theta_1$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
#%% DHWP45 measurement: plotting, visibility calculation, expectation values calculation and CHSH parameter calculation

if not df45.empty:
    plt.figure(figsize=(8, 5))
    for hwp1, group in df45.groupby('HWP1'):
        plt.errorbar(group['HWP2'], group['Counts'], yerr=np.sqrt(group['Counts']), fmt='o-', label=f'HWP1 = {hwp1}°')
        
    plt.xlabel('HWP2 Angle (degrees)')
    plt.ylabel('Coincidence Counts')
    plt.legend()
    plt.title('Integrated Counts vs HWP2 Angle')
    plt.grid(True, alpha=0.3)
    plt.show()
    visibilities45 = []
    for hwp1, group in groups:
        max_counts = group['Counts'].max()
        min_counts = group['Counts'].min()
        if max_counts + min_counts > 0:
            V = (max_counts - min_counts) / (max_counts + min_counts)
            visibilities45.append((hwp1, V))
            print(f"DHWP45, HWP1 = {hwp1}°: Visibility = {V:.3f}")
        else:
            print(f"DHWP45, HWP1 = {hwp1}°: Cannot calculate visibility (max + min = 0)")
else:
    print("DataFrame empty")

a   = 0.0
a_p = 22.5
b   = 7.5
b_p = 30

E_ab   = E(df45, a, b)
E_abp  = E(df45, a, b_p)
E_apb  = E(df45, a_p, b)
E_apbp = E(df45, a_p, b_p)

# print("E(a,b)   =", E_ab)
# print("E(a,b')  =", E_abp)
# print("E(a',b)  =", E_apb)
# print("E(a',b') =", E_apbp)

S = E_ab - E_abp + E_apb + E_apbp
print("\nCHSH S =", S)


C_ab45 = get_count(df45, a, b)
C_apb45 = get_count(df45, a + 45, b)
C_abp45 = get_count(df45, a, b + 45)
C_apbp45 = get_count(df45, a + 45, b + 45)
numerator45 = C_ab45 + C_apbp45 - C_apb45 - C_abp45
denominator45 = C_ab45 + C_apbp45 + C_apb45 + C_abp45
sigma_E45 = np.sqrt(((denominator45 - numerator45) / denominator45**2)**2 * (C_ab45 + C_apbp45) + ((denominator45 + numerator45) / denominator45**2)**2 * (C_apb45 + C_abp45))


E_results = [
    E_with_error(df45, 0.0, 7.5),   # E(a, b)
    E_with_error(df45, 0.0, 30.0),  # E(a, b')
    E_with_error(df45, 22.5, 7.5),  # E(a', b)
    E_with_error(df45, 22.5, 30.0)  # E(a', b')
]

vals = [res[0] for res in E_results]
errs = [res[1] for res in E_results]

S_val = vals[0] - vals[1] + vals[2] + vals[3]
S_err = np.sqrt(sum(np.array(errs)**2))

print(f"S = {S_val:.4f} ± {S_err:.4f}")
print(f"Violation: {(np.abs(S_val)-2)/S_err:.1f} sigma")

# %% Extract the DHWP45 data and fit it (not ignoring first point)

plt.figure(figsize=(10, 6))

for hwp1, group in df45.groupby('HWP1'):

    x_data = group['HWP2']
    y_data = group['Counts']
    
    popt, model_func = fit_sin(x_data, y_data)
    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
    
    line, caps, bars = plt.errorbar(x_data, y_data, yerr=np.sqrt(y_data), 
                                    fmt='o', markersize=4, capsize=2,
                                    alpha=0.8)
    
    plt.plot(x_fit, model_func(x_fit, *popt), '--', 
                color=line.get_color(), linewidth=1.5,
                label=f'Fit HWP1={hwp1}°')


plt.xlabel("HWP2 Angle (°)")
plt.ylabel("Integrated Counts")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# %% plot and fit of data with first point at 0° ignored
plt.figure(figsize=(12, 6))
ax = plt.gca() # Get current axes to add the artist later

fits_params = {}
visibilities = {}
leg_handles = [] # To store the fit lines for the first legend
vis_handles = [] # To store the markers for the second legend

for hwp1, group in df45.groupby('HWP1'):
    x_data = group['HWP2']
    y_data = group['Counts']
    
    x_fit_input = x_data.iloc[1:] if hwp1 == 0.0 else x_data
    y_fit_input = y_data.iloc[1:] if hwp1 == 0.0 else y_data

    try:
        popt, _ = fit_sin(x_fit_input, y_fit_input)
        fits_params[hwp1] = popt

        # Visibility Logic
        amp, offset = popt[0], popt[2]
        if amp >= 0:
            i_max, i_min = offset + amp, offset
        else:
            i_max, i_min = offset, max(0, amp + offset)
        
        V = (i_max - i_min) / (i_max + i_min)
        visibilities[hwp1] = V
        print(f"HWP1={hwp1: >5}°: Visibility = {V*100:.3f}%")

        # Plotting
        x_fit_range = np.linspace(x_data.min(), x_data.max(), 200)
        line, = ax.plot(x_fit_range, model_func(x_fit_range, *popt), '--', 
                         label=rf'{hwp1}°')
        
        ax.errorbar(x_data, y_data, yerr=np.sqrt(y_data), fmt='o', 
                    color=line.get_color(), markersize=4, capsize=4,
                    alpha=0.7, elinewidth=4, mfc='none') # mfc='none' for open markers
        
        # Store handles for legends
        leg_handles.append(line)
        
        # Create a proxy for visibility legend (just the marker, no line)
        proxy = plt.Line2D([0], [0], linestyle='none', marker='o', 
                           color=line.get_color(), mfc='none',
                           label=f'{V*100:.1f}%')
        vis_handles.append(proxy)
                     
    except Exception as e:
        print(f"Fit failed for HWP1={hwp1}: {e}")

# --- Legend 1 (Top Right) ---
# Use ncol=2 to make it compact if you have 4+ fits
leg1 = ax.legend(handles=leg_handles, title=r"$\theta_1$", 
                 bbox_to_anchor=(1.01, 1), loc='upper left')

# Add it as a separate artist so it isn't overwritten
ax.add_artist(leg1)

# --- Legend 2 (Bottom Right) ---
ax.legend(handles=vis_handles, title="Visibilities",
          bbox_to_anchor=(1.01, 0.25), loc='upper left')

plt.xlabel(r"$\theta_2$ (degrees)")
plt.ylabel("Coincidences")
plt.tight_layout(rect=[0,0,.85,1])
plt.show()

def get_E_interpolated(a, b):

    C_ab    = model(b, *fits_params[a])
    C_apbp  = model(b + 45, *fits_params[a + 45])
    C_apb   = model(b, *fits_params[a + 45])
    C_abp   = model(b + 45, *fits_params[a])

    n_total = C_ab + C_apbp + C_apb + C_abp
    denominator = C_ab + C_apbp + C_apb + C_abp
    numerator = C_ab + C_apbp - C_apb - C_abp
    if n_total <= 0: return 0, 0
    e_val = (C_ab + C_apbp - C_apb - C_abp) / n_total
    e_err = np.sqrt(((denominator - numerator) / denominator**2)**2 * (C_ab + C_apbp) + ((denominator + numerator) / denominator**2)**2 * (C_apb + C_abp))

    return e_val, e_err  
    
a_0, a_22 = 0.0, 22.5
b_11, b_33 = 11.25, 33.75

e1, err1 = get_E_interpolated(a_0, b_11)   # E(a, b)
e2, err2 = get_E_interpolated(a_0, b_33)   # E(a, b')
e3, err3 = get_E_interpolated(a_22, b_11)  # E(a', b)
e4, err4 = get_E_interpolated(a_22, b_33)  # E(a', b')

S_val = e1 - e2 + e3 + e4
S_err = np.sqrt(err1**2 + err2**2 + err3**2 + err4**2)

print("Bell parameter S =")
print(f"S = {S_val:.4f} ± {S_err:.4f}")

n_sigma = (abs(S_val) - 2) / S_err
print(f"Violation: {n_sigma:.1f} sigma")


# %%
