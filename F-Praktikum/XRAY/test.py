#%%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid , quad
#%% Define constants and functions 
h = const.h
hbar = const.hbar
e = const.elementary_charge
c = const.speed_of_light
lattice_constant = 2014e-12

def get_wavelength(theta, d, order=1):
    wl = 2*d*np.sin(np.deg2rad(theta))/order
    return wl

def get_energy_from_wl(wl):
    '''
    Wl: wavelenght in meters
    '''
    energy_ev = h*c / (e*wl) 
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
def linear_func(m, x, c):
    return m*x+c

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
#%% Initialize 
fe_data = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Fe2s35kV1mA', skiprows=2, converters=lambda s: s.replace(b',', b'.'))

mo_data_full = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Mo2s_4-65', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
mo_data_31 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Mo60s31-33', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
mo_data_44 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Mo60s44-46', skiprows=2, converters=lambda s: s.replace(b',', b'.'))

cu_data_full = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\2mmganzesspektrum', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
cu_data_1n1 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\1ordnunggenau', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
cu_data_2n1 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\1ordnunggenau2', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
cu_data_1n2 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\2ordnunggenau', skiprows=2, converters=lambda s: s.replace(b',', b'.'))
cu_data_2n2 = np.loadtxt(r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\2ordnunggenau2', skiprows=2, converters=lambda s: s.replace(b',', b'.'))

fe_angle = fe_data[:,0]
fe_counts = fe_data[:,1]/(1-90e-6*fe_data[:,1])

mo_full_angle = mo_data_full[:,0]
mo_full_counts = mo_data_full[:,1]/(1-90e-6*mo_data_full[:,1])

mo_31_angle = mo_data_31[:,0]
mo_31_counts = mo_data_31[:,1]/(1-90e-6*mo_data_31[:,1])

mo_44_angle = mo_data_44[:,0]
mo_44_counts = mo_data_44[:,1]/(1-90e-6*mo_data_44[:,1])

cu_full_angle = cu_data_full[:,0]+0.15 #set x-axis shift for calibration
cu_full_counts = cu_data_full[:,1]/(1-90e-6*cu_data_full[:,1])

cu_1n1_angle = cu_data_1n1[:,0]
cu_1n1_counts = cu_data_1n1[:,1]

cu_2n1_angle = cu_data_2n1[:,0]
cu_2n1_counts = cu_data_2n1[:,1]

cu_1n2_angle = cu_data_1n2[:,0]
cu_1n2_counts = cu_data_1n2[:,1]

cu_2n2_angle = cu_data_2n2[:,0]
cu_2n2_counts = cu_data_2n2[:,1]
#%% Plotting the different spectra 
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20

angles = [np.array(mo_31_angle), np.array(mo_44_angle), np.array(fe_angle)]
counts = [np.array(mo_31_counts), np.array(mo_44_counts), np.array(fe_counts)/(1-90e-6*np.array(fe_counts))]
titles = ['Mo 3rd order', 'Mo 4th order', 'Fe 2nd order']

for angle, count, title in zip(angles, counts, titles):
    
    plt.figure(figsize=(8,7))
    plt.xlabel(r'$\theta \;[°]$')
    plt.ylabel(r'Counts  $[s^{-1}]$')
    #plt.scatter(angle, count, c='darkorange')
    plt.plot(angle, count, c='orange')
    plt.title(title + r' $K_\alpha$ lines')
    plt.tick_params(which="both", direction="in", top=True, right=True)
    plt.ylim(np.min(count)-1, 2000)
    plt.xlim(73.5, 75)
    plt.grid()
    plt.show()

#%% Find the angles of the characteristic XRAY peaks 
#peaks_cu = 
peak_idx_mo, _ = find_peaks(mo_full_counts, height=400, prominence=100)
peak_idx_fe, _ = find_peaks(fe_counts[74:], prominence=350, height=70)
peak_idx_cu, _ = find_peaks(cu_full_counts, prominence=200, height=70)
peak_idx_mo3, _ = find_peaks(mo_31_counts, height = 50)
peak_idx_mo4, _ = find_peaks(mo_44_counts, height = 25)
peak_angle_mo, peak_angle_mo3, peak_angle_mo4 = [], [], []
peak_angle_fe = []
peak_angle_cu = []
for i in range(len(peak_idx_mo)):
    peak_angle_mo.append(mo_full_angle[peak_idx_mo[i]])
    
for i in range(len(peak_idx_fe)):
    peak_angle_fe.append(fe_angle[peak_idx_fe[i]+74])
    
for i in range(len(peak_idx_cu)):
    peak_angle_cu.append(cu_full_angle[peak_idx_cu[i]])

for i in range(len(peak_idx_mo3)):
    peak_angle_mo3.append(mo_31_angle[peak_idx_mo3[i]])

for i in range(len(peak_idx_mo4)):
    peak_angle_mo4.append(mo_44_angle[peak_idx_mo4[i]])
#%% Plot each spectrum with the peak points added with scatter plot 
capsize = 80
mo_labels = [r'$K_{\beta}$ (n=1)', r'$K_{\alpha}$ (n=1)', r'$K_{\beta}$ (n=2)', r'$K_{\alpha}$ (n=2)'] 
fe_labels = [r'$K_{\beta}$ (n=1)', r'$K_{\alpha}$ (n=1)', r'$K_{\beta}$ (n=2)', r'$K_{\alpha}$ (n=2)']   
cu_labels = [r'$K_{\beta}$ (n=1)', r'$K_{\alpha}$ (n=1)', r'$K_{\beta}$ (n=2)', r'$K_{\alpha} (n=2)$']

# --- Molybdenum spectrum ---
plt.figure(figsize=(8,6))
plt.plot(mo_full_angle, mo_full_counts)

for x, y, label in zip(mo_full_angle[peak_idx_mo],
                       mo_full_counts[peak_idx_mo],
                       mo_labels):
    plt.scatter(x, y, facecolors='none', edgecolors='black', s=capsize, marker='s')
    plt.text(x - 2, y + 100, label, ha='center', va='bottom', fontsize=10)  # label per peak

plt.tick_params(which="both", direction="in", top=True, right=True)
plt.ylabel(r'Counts $s^{-1}$')
plt.title('Molybdenum X-ray spectrum')
plt.ylim(0, np.max(mo_full_counts) + 500)
plt.grid()
plt.xlabel(r'$\theta \;[°]$')
plt.ylabel(r'Counts  $[s^{-1}]$')
plt.show()

print(peak_angle_mo)


# --- Iron spectrum ---
plt.figure(figsize=(8,6))
plt.plot(fe_angle, fe_counts)

for x, y, label in zip(fe_angle[peak_idx_fe+74] + 0,   # +0 for clarity
                       fe_counts[peak_idx_fe + 74],
                       fe_labels):
    plt.scatter(x + 0, y, facecolors='none', edgecolors='black', s=capsize, marker='s')
    plt.text(x - 1, y + 100, label, ha='center', va='bottom', fontsize=10)

plt.tick_params(which="both", direction="in", top=True, right=True)
plt.xlabel(r'$Angle \; \theta \;[°]$')
plt.ylabel(r'Counts $[s^{-1}]$')
plt.title('Iron X-ray spectrum')
plt.ylim(0, np.max(fe_counts) + 500)
plt.grid()
plt.xlabel(r'$\theta \;[°]$')
plt.ylabel(r'Counts  $[s^{-1}]$')
plt.show()

print(peak_angle_fe)


plt.figure(figsize=(8,6))
plt.plot(cu_full_angle, cu_full_counts)
for x, y, label in zip(cu_full_angle[peak_idx_cu],
                       cu_full_counts[peak_idx_cu],
                       cu_labels):
    plt.scatter(x, y, facecolors='none', edgecolors='black', s=capsize, marker='s')
    plt.text(x - 1, y + 7, label, ha='center', va='bottom', fontsize=10) 


plt.tick_params(which="both", direction="in", top=True, right=True)
plt.ylabel(r'Counts $[s^{-1}]$')
plt.title('Copper X-ray spectrum')
plt.ylim(0, np.max(cu_full_counts) + 700)
plt.grid()
plt.xlabel(r'$\theta \;[°]$')
plt.ylabel(r'Counts $[s^{-1}]$')
plt.show()


print(peak_angle_cu)

#%% Calculate the wavelengths and energies from the angles of the characteristic peaks
d = 2.01e-10
cu_peak_wl1, fe_peak_wl1, mo_peak_wl1 = [], [], []  
cu_peak_e1, fe_peak_e1, mo_peak_e1 = [], [], []


for cu, fe, mo in zip(peak_angle_cu, peak_angle_fe, peak_angle_mo):
    l11 = get_wavelength(theta=cu, d=d, order=1)
    l21 = get_wavelength(theta=fe, d=d, order=1)
    l31 = get_wavelength(theta=mo, d=d, order=1)
    
    cu_peak_wl1.append(l11)
    fe_peak_wl1.append(l21)
    mo_peak_wl1.append(l31)
    
    cu_peak_e1.append(get_energy_from_wl(l11))
    fe_peak_e1.append(get_energy_from_wl(l21))
    mo_peak_e1.append(get_energy_from_wl(l31))

# print(r'Wl n=1')
# print('Cu:',cu_peak_wl1)
# print('Fe:',fe_peak_wl1)
# print('Mo:',mo_peak_wl1, end='\n\n')

print(r'Energy in keV (n=1):')
print(f'Cu: {np.array(cu_peak_e1)[:2]/1000}')
print(f'Fe: {np.array(fe_peak_e1)[:2]/1000}')
print(f'Mo: {np.array(mo_peak_e1)[:2]/1000}\n')

cu_peak_wl2, fe_peak_wl2, mo_peak_wl2 = [], [], []
cu_peak_e2, fe_peak_e2, mo_peak_e2 = [], [], []

for cu, fe, mo in zip(peak_angle_cu, peak_angle_fe, peak_angle_mo):
    l12 = get_wavelength(theta=cu, d=d, order=2)
    l22 = get_wavelength(theta=fe, d=d, order=2)
    l32 = get_wavelength(theta=mo, d=d, order=2)
    
    cu_peak_wl2.append(l12)
    fe_peak_wl2.append(l22)
    mo_peak_wl2.append(l32)
    
    cu_peak_e2.append(get_energy_from_wl(l12))
    fe_peak_e2.append(get_energy_from_wl(l22))
    mo_peak_e2.append(get_energy_from_wl(l32))

# print(r'Wl n=2')
# print('Cu:',cu_peak_wl2)
# print('Fe:',fe_peak_wl2)
# print('Mo:',mo_peak_wl2, end='\n\n')

print(r'Energy in keV (n=2):')
print(f'Cu: {np.array(cu_peak_e2)[2:]/1000}')
print(f'Fe: {np.array(fe_peak_e2)[2:]/1000}')
print(f'Mo: {np.array(mo_peak_e2)[2:]/1000}\n')

#%% Calibration of the copper spectrum with 1st and 2nd order USELESS
first_order_mean_angle = np.array(np.array(cu_1n1_angle)+np.array(cu_2n1_angle))/2
first_order_mean_counts = np.array(np.array(cu_1n1_counts)+np.array(cu_2n1_counts))/2

second_order_mean_angle = np.array(np.array(cu_1n2_angle)+np.array(cu_2n2_angle))/2
second_order_mean_counts = np.array(np.array(cu_1n2_counts)+np.array(cu_2n2_counts))/2

fig, axs = plt.subplots(2,1,figsize=(6,9))
axs[0].plot(first_order_mean_angle, first_order_mean_counts, c='blue')
axs[1].plot(second_order_mean_angle, second_order_mean_counts, c='orange')
axs[1].set_xlabel(r'$\theta \;[°]$')
axs[0].set_ylabel(r'Counts $s^{-1}$')

axs[1].set_ylabel(r'Counts $s^{-1}$')

plt.tight_layout()

idx_first, _ = find_peaks(first_order_mean_counts, height=500)
idx_second, _ = find_peaks(second_order_mean_counts, height=100)
peak_angle_first = []
peak_angle_second = []

for i in range(len(idx_first)):
    peak_angle_first.append(first_order_mean_angle[idx_first[i]])
    
for i in range(len(idx_second)):
    peak_angle_second.append(second_order_mean_angle[idx_second[i]])

print('First order peak angles: ')
print( peak_angle_first)
print('Second order peak angles: ')
print(peak_angle_second)
# %% voltage and current sweep intensity dependence (cu)
voltage_data = np.loadtxt(
    r'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\Messreihe2_2Spannung', 
    skiprows=2, converters=lambda s: s.replace(b',', b'.')
)
voltage_angles = voltage_data[:,0]
num_measurements = 10
filenames = [
    fr'C:\Main\my_projects\F-Praktikum\XRAY\xray_data\current_sweep\Messreihe2_2richtig{i+1}'
    for i in range(num_measurements)
]

angles_list = []
counts_list = []

for file in filenames:
    data = np.loadtxt(file, skiprows=2, converters=lambda s: s.replace(b',', b'.'))
    angles_list.append(data[:, 0])
    counts_list.append(data[:, 1])

angles_array_I = np.array(angles_list) 
counts_array_I = np.array(counts_list)

int1 = []
int2 = []
int1_I = []
int2_I = []

for i in range(13):
    int1.append(np.max(voltage_data[160:180, (i+1)]/(1 - 90e-6*voltage_data[160:180, (i+1)])))
    int2.append(np.max(voltage_data[190:200, (i+1)]/(1-90e-6*voltage_data[190:200, (i+1)])))

for i in range(10):
    int1_I.append(np.max(counts_array_I[i,0:15]/(1-90e-6*counts_array_I[i,0:15])))
    int2_I.append(np.max(counts_array_I[i,20:40]/(1-90e-6*counts_array_I[i,20:40])))

# Prepare data, note: only intensity^(2/3) for voltage-dependent measurements
int_list = [np.array(int1)**(2/3), np.array(int2)**(2/3), np.array(int1_I), np.array(int2_I)]
tit_list = [r'Intensity of $K_{\alpha}$ line vs. $U_A$', 
            r'Intensity of $K_{\beta}$ line vs. $U_A$', 
            r'Intensity of $K_{\alpha}$ line vs. $I_A$', 
            r'Intensity of $K_{\beta}$ line vs. $I_A$']
x_list = [np.arange(11,37,2), np.arange(11,37,2), np.arange(.1,1.1,.1), np.arange(.1,1.1,.1)]
xlabel_list = [r'U [kV]', r'U [kV]', r'I [mA]', r'I [mA]']
ylabel_list = [r'$Counts^{2/3}$  $[s^{-{2/3}}]$', r'$Counts^{2/3}$  $[s^{-{2/3}}]$', r'Counts  $[s^{-1}]$', r'Counts  $[s^{-1}]$']

linreg_list = []
which_plot = ['U_A_a1','U_A_b1','I_A_a1','I_A_b1']

for x, y, tit, xlb, ylb, wp in zip(x_list, int_list, tit_list, xlabel_list, ylabel_list, which_plot):
    
    # Linear regression
    res = linregress(x, y)
    linreg_list.append(res)
    
    # Create smooth x-values for plotting the fit line
    x_fit = np.linspace(np.min(x), np.max(x), 200)
    y_fit = res.intercept + res.slope * x_fit
    
    # Plot only the linear fit line
    plt.figure(figsize=(8,7))
    plt.plot(x_fit, y_fit, c='orange', label=f'Linear fit with R²={res.rvalue**2:.3f}')
    plt.scatter(x, y, c='darkorange', marker='+', label='Data')
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.title('Linear Fit ' + tit)
    plt.tick_params(which="both", direction="in", top=True, right=True)
    plt.grid(True)
    plt.legend()
    plt.savefig(r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\F-Praktikum\XRAY\plots\\' + wp + '.png')
    plt.show()

# Print fit parameters
for i, res in enumerate(linreg_list):
    print(f"{tit_list[i]}:")
    print(f"  slope     = {res.slope:.3f}")
    print(f"  intercept = {res.intercept:.3f}")
    print(f"  R²        = {res.rvalue**2:.3f}\n")

# %% Moseley's law 
m_e = const.electron_mass
eps0 = const.epsilon_0
z_mo = 42
z_fe = 26
z_cu = 29
atomic_numbers = np.array([z_fe, z_cu, z_mo])
#characteristic energies in eV 
ka_mo = mo_peak_e1[1]
ka_fe = fe_peak_e1[1]
ka_cu = cu_peak_e1[1]
na = .75

kb_mo = 19.546290e3
kb_fe = 7.047388e3
kb_cu = 8.891904e3
nb = (1 - (1/9))

ya_data = np.sqrt(np.array([ka_fe, ka_cu, ka_mo]))
yb_data = np.sqrt(np.array([kb_fe, kb_cu, kb_mo]))

nu_err = np.sqrt([10.144, 16.854, 84.059])



#slope, intercept, r_value, _, slope_err = linregress(atomic_numbers, ya_data)
popt, pcov = curve_fit(linear_func, atomic_numbers, ya_data, sigma=nu_err, absolute_sigma=True)

slp = popt[0]
icpt = popt[1]
pcov_errs = np.sqrt(np.diag(pcov))
slp_err = pcov_errs[0]
icpt_err = pcov_errs[1]
ryd = slp**2 / (na)
ryd_err = (2*ryd * slp_err) / (slp)
sigma = -icpt/slp
sigma_err = sigma * np.sqrt(-icpt_err/icpt)**2 + (slp_err/slp)**2
print(f'Rydberg constant = ({ryd} +/- {ryd_err} ) eV')
print(f'Screening factor = {sigma} +/- {sigma_err} ')

x_fit = np.linspace(atomic_numbers.min(), atomic_numbers.max(), 200)
y_fit = linear_func(x_fit, *popt) 

# --- 2. Create the Plot ---
plt.figure(figsize=(8, 6))

plt.errorbar(atomic_numbers, ya_data, yerr=nu_err,
             fmt='o',          
             capsize=5,       
             ecolor='black',
             color='darkorange',
             label='Data points')

plt.plot(x_fit, y_fit, c='orange', label='Linear Fit') 

plt.tick_params(which="both", direction="in", top=True, right=True)
plt.title("Moseley's Law with Linear Fit")
plt.grid(True)
plt.ylabel(r'$\sqrt{\nu}$  [$Hz^{1/2}$]')
plt.xlabel('Atomic Number (Z)')
plt.legend()
plt.show()
# %% intensity ratio of the 3rd and 4th order Mo K_alpha lines 
#do gaussian fit for the 4th order

popt, pcov = curve_fit(multi_peak_model_with_bkg, mo_44_angle, mo_44_counts, p0=[37,44.7,.3, 31, 45.1, .25, 0,0,0], maxfev=15000)
pop, pco = curve_fit(gaussian, mo_31_angle, mo_31_counts-44, p0=[128,31.85, .3], maxfev=15000)
y_fit = multi_peak_model_with_bkg(mo_44_angle, *popt)
y_fit_31 = gaussian(mo_31_angle, *pop)

plt.figure(figsize=(8,7))
plt.plot(mo_44_angle, y_fit, c='orange', linestyle='-')
plt.scatter(mo_44_angle, mo_44_counts, c='darkorange', marker='+', label='Data', s=30)
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.title(r"Mo 4th order $K_\alpha$ lines")
plt.grid(True)
plt.ylabel(r'Counts $[s^{-1}]$')
plt.xlabel(r'Angle $\theta$ [°]')
plt.show()

plt.figure(figsize=(8,7))
plt.plot(mo_31_angle, mo_31_counts, c='orange', linestyle='-')
plt.scatter(mo_31_angle, mo_31_counts, c='darkorange', marker='+', label='Data', s=30)
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.title(r"Mo 3rd order $K_\alpha$ lines")
plt.grid(True)
plt.ylabel(r'Counts $[s^{-1}]$')
plt.xlabel(r'Angle $\theta$ [°]')
plt.show()
# %% integrals from the single gaussian peaks from the fit 
A1, mu1, sigma1, A2, mu2, sigma2, b0, b1, b2 = popt
perr = np.sqrt(np.diag(pcov))
g1_fit = A1 * np.exp(-0.5*((mo_44_angle - mu1)/sigma1)**2)
g2_fit = A2 * np.exp(-0.5*((mo_44_angle - mu2)/sigma2)**2)

plt.figure(figsize=(8,7))
plt.plot(mo_44_angle, y_fit, c='orange', label='Total fit')
plt.plot(mo_44_angle, g1_fit, '--', c='blue', label='Gaussian 1')
plt.plot(mo_44_angle, g2_fit, '--', c='green', label='Gaussian 2')
plt.scatter(mo_44_angle, mo_44_counts, c='darkorange', marker='+', s=30, label='Data')
plt.legend()
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.title(r"Gaussian fits of Mo 4th order $K_\alpha$ lines")
plt.xlabel(r'Angle $\theta$ [°]')
plt.ylabel(r'Counts $[s^{-1}]$')
plt.grid(True)
plt.show()

area_1_num = np.trapz(g1_fit, mo_44_angle)
area_2_num = np.trapz(g2_fit, mo_44_angle)
int_ratio_num = area_1_num / area_2_num
print(f'Numerical intensity ratio = {int_ratio_num}')

# %% monte carlo mesig 
# --- Fitted parameters and covariance ---
popt, pcov = curve_fit(multi_peak_model_with_bkg, mo_44_angle, mo_44_counts, 
                       p0=[37,44.7,.3, 31, 45.1,.25, 0,0,0], maxfev=15000)
perr = np.sqrt(np.diag(pcov))

# --- Monte Carlo setup ---
N_MC = 10000          # number of simulations
angle_sigma = 0.05     # angle uncertainty in degrees

ratios = np.zeros(N_MC)

for i in range(N_MC):
    # 1. Randomize fit parameters according to their covariance
    params_MC = np.random.multivariate_normal(popt, pcov)
    
    # Extract parameters for first and second peaks (assuming 3 per peak: h, x0, sigma)
    h1, x01, s1 = params_MC[0:3]
    h2, x02, s2 = params_MC[3:6]
    
    # 2. Optionally, add angle noise
    angles_MC = mo_44_angle + np.random.normal(0, angle_sigma, size=mo_44_angle.size)
    
    A1 = cumulative_trapezoid(g1_fit, mo_44_angle)
    A2 = cumulative_trapezoid(g2_fit, mo_44_angle)
    
    # 4. Compute ratio
    ratios[i] = A1 / A2

# --- Get statistics ---
R_mean = np.mean(ratios)
R_std = np.std(ratios)

print(f"Intensity ratio (MC) = {R_mean:.3f} ± {R_std:.3f}")
# %%
