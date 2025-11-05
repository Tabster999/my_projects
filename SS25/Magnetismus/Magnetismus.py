
#%% exercise 1
mu_b = const.physical_constants['Bohr magneton'][0]
eta_array = np.linspace(0,6,200)
def m_z_normalized(eta):
    g = -const.physical_constants['electron g factor'][0]
    mu_b = const.physical_constants['Bohr magneton'][0]
    m_z = (g/2)*np.tanh(eta/2)
    
    return m_z

m_z_array = np.zeros((len(eta_array)))
for i, eta in enumerate(eta_array):
    m_z_array[i] = m_z_normalized(eta)

 
eta_min_fit = 0
eta_max_fit = .8
fit_mask = (eta_array >= eta_min_fit) & (eta_array <= eta_max_fit)
eta_for_fit = eta_array[fit_mask]
m_z_for_fit = m_z_array[fit_mask]

slope, intercept, r_value, p_value, std_err = linregress(eta_for_fit, m_z_for_fit)
fit_line_y = slope * eta_for_fit + intercept
plt.figure(figsize=(12,7), dpi=150)
plt.plot(eta_array, m_z_array)
#plt.plot(eta_for_fit, fit_line_y)
plt.xlabel(r'$\eta$', fontsize=16)
plt.ylabel(r'$\frac{\langle \mu_z \rangle}{\mu_b}$', fontsize=16)
plt.gca().set_facecolor('lightyellow')
plt.xlim(0,6)
plt.show()

def brillouin_function(J, eta):
    bf = (1/J)*(((2*J+1)/2)*np.coth((2*J+1)*eta/2)-(1/2)*np.coth(eta/2))
    return bf 

eta_array = np.linspace(0,5,200)
j_vals = [.5,1,5]
j_array = np.array(j_vals)
brillouin = np.zeros(len(eta_array))

for i, eta in enumerate(eta_array):
    for k, j in enumerate(j_array): 
        brillouin[i][k] = brillouin_function(j, eta)
        

#%% exercise 5 data initiation
pl_dat = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\Übungen\Pl.dat")
mn_dat = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\Übungen\Mn.dat")

int_pl = pl_dat[:, 0]
energy_pl = pl_dat[:,1]

int_mn = mn_dat[:, 0]
energy_mn = mn_dat[:,1]
#%% Step 2: scaling factor
scaling_factor_pre = np.mean(int_pl[3:105])/np.mean(int_mn[3:105])
scaling_factor_mid = np.mean(int_pl[252:420])/np.mean(int_mn[252:420])
scaling_factor_post = np.mean(int_pl[520:700])/np.mean(int_mn[520:700])
scaling_factor_mean = (scaling_factor_pre+scaling_factor_post+scaling_factor_mid)/3
 
fig, ax_grid = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
ax = ax_grid.flatten()

ax[0].plot(energy_mn, int_mn , label='Mn')
ax[0].plot(energy_pl, int_pl* 1/(scaling_factor_pre), label='Scaled Pl')
ax[0].set_title('Comparison using Pre-edge Scaling')
ax[0].set_xlabel('Energy')
ax[0].axvline(x=energy_pl[0])
ax[0].axvline(x=energy_pl[125])
ax[0].set_ylabel('Intensity')

ax[1].plot(energy_mn, int_mn , label='Mn')
ax[1].plot(energy_pl, int_pl* 1/(scaling_factor_mid), label='Scaled Pl')
ax[1].set_title('Comparison using Mid-edge Scaling')
ax[1].axvline(x=energy_pl[250])
ax[1].axvline(x=energy_pl[420])
ax[1].set_xlabel('Energy')

ax[2].plot(energy_mn, int_mn , label='Mn')
ax[2].plot(energy_pl, int_pl* 1/(scaling_factor_post), label='Scaled Pl')
ax[2].set_title('Comparison using Post-edge Scaling')
ax[2].set_xlabel('Energy')
ax[2].axvline(x=energy_pl[520])
ax[2].axvline(x=energy_pl[700])
ax[2].set_ylabel('Intensity')

ax[3].plot(energy_mn, int_mn , label='Mn')
ax[3].plot(energy_pl, int_pl* (1/scaling_factor_mean), label='Scaled Pl')
ax[3].set_title('Comparison using Mean Scaling')
ax[3].set_xlabel('Energy')

for a in ax:
    a.legend()
    a.grid(True, linestyle='--', alpha=0.6)

fig.tight_layout()
plt.show()
#%% Step 3+4: linear fit 
energy_pl_pre = energy_pl[3:105]
int_pl_pre = int_pl[3:105]
energy_mn_pre = energy_mn[3:105]
int_mn_pre = int_mn[3:105]

slope_pl, intercept_pl, r_value_pl, p_value_pl, std_err_pl = linregress(energy_pl_pre, int_pl_pre)
slope_mn, intercept_mn, r_value_mn, p_value_mn, std_err_mn = linregress(energy_mn_pre, int_mn_pre)

print(f'Slope Pl ={slope_pl}')
print(f'Intercept Pl ={slope_pl}')
print(f'Slope Mn ={slope_mn}')
print(f'Intercept Mn ={slope_mn}')

# fig, ax = plt.subplots(1,2,figsize=(16,10), dpi=150)
# ax[0].plot(energy_pl_pre, slope_pl*energy_pl_pre + intercept_pl, c='blue', label='Pl lin-fit')
# ax[1].plot(energy_mn_pre, slope_mn*energy_mn_pre + intercept_mn, c='orange', label='Mn lin-fit')
# ax[0].set_xlabel('Energy')
# ax[1].set_xlabel('Energy')
# ax[0].set_ylabel('Intensity')
# ax[0].set_title('Pl')
# ax[1].set_title('Mn')
# plt.show()

plt.figure(figsize=(10,7), dpi=100)
plt.plot(energy_pl, int_pl, c='blue', label='+')
plt.plot(energy_mn, int_mn, c='red', label='-')
plt.plot(energy_pl, energy_pl*slope_pl + intercept_pl, c='lightblue', label='fit +', linestyle='--')
plt.plot(energy_pl, energy_mn*slope_mn + intercept_mn, c='salmon', label='fit -', linestyle='--')
plt.title('Intensity comparison')
plt.xlabel('Energy')
plt.legend()
plt.ylabel('Norm. Intensity')
plt.show()
# %% background subtraction
lin_fit_pl = slope_pl * energy_pl + intercept_pl
lin_fit_mn = slope_mn * energy_mn + intercept_mn
background_pl = 1/(scaling_factor_pre)*int_pl - lin_fit_pl
background_mn =  int_mn - lin_fit_pl

norma = 0.0235254969254286
# fig, ax = plt.subplots(1,2,figsize=(16,8), dpi=150)
# ax[0].plot(energy_pl, background_pl, c='blue')
# ax[1].plot(energy_mn, background_mn, c='orange')
# ax[0].set_xlabel('Energy')
# ax[1].set_xlabel('Energy')
# ax[0].set_ylabel('Intensity - Background')
# ax[0].set_title('Pl minus background')
# ax[1].set_title('Mn minus background')
# plt.show()
plt.figure(figsize=(12,7), dpi=100)
plt.plot(energy_pl, background_pl/norma, c='blue', label='+' ,linewidth=.5)
plt.plot(energy_mn, background_mn/norma, c='red', label='-', linewidth=.5)
plt.title('Intensity comparison')
plt.xlabel('Energy')
plt.plot(energy_pl, (background_mn - background_pl)/(2*norma), c='gray', linewidth=.7, label='north - south')
plt.legend()
plt.ylabel('Norm. Intensity')
#%% Step 6: substracting the resonance
avg_background = (background_pl + background_mn)/(2*norma)

#the ratio between A4 and A5 should be similar to the ratio of degeneracies between the different j values degenercy = 2*j+1)
def calc_B(energy, gamma):
    A5 = .01275
    A4 = .705
    EM5 = 1128.4
    EM4 = 1156.2
    pi = np.pi
    B_val = A5 * (((1/pi) * np.arctan((energy - EM5)/gamma) + .5) + A4 * ( (1/pi) * np.arctan((energy - EM4)/gamma) + .5))
    
    return B_val

energies = np.linspace(1110,1180,200)
plt.figure(figsize=(10,7), dpi=100)
plt.plot(energy_pl, calc_B(energy_pl, 1)/norma,linewidth=.75, label='B')
plt.plot(energy_pl, avg_background,linewidth=.75, label='averaged background signal')
plt.legend()
plt.show()


plt.figure(figsize=(10,7), dpi=100)
plt.plot(energy_pl, avg_background - calc_B(energy_pl, 1)/norma, linewidth=.7)
plt.title('Step 6')
plt.xlabel('Energy')
plt.ylabel('Norm. Intensity')
plt.show()
#%% Step 7: Integration
middle_between_peaks = 1142.3

B_array = np.zeros(len(energy_pl))
for i, e_val in enumerate(energy_pl):
    B_array[i] = calc_B(e_val, 1)/norma

integrand = avg_background - B_array
north_south = (background_mn - background_pl)/(2*norma)

integral_no_resonant = cumulative_trapezoid(integrand, energy_pl, initial=0)
integral_north_south = cumulative_trapezoid(north_south, energy_pl, initial=0)
plt.figure(figsize=(10,7), dpi=150)
#plt.plot(energy_pl, integral_no_resonant,linewidth=.75, label='no resonant', c='red')
plt.plot(energy_pl, integral_north_south,linewidth=.75, label='north south', c='blue')
plt.title('Integral north south')
plt.xlabel('Energy')
plt.ylabel('Integrated intensity Intesity')
plt.show()

val_red = integral_no_resonant[700] #height of the xas = q
val_green = integral_north_south[323] #middle of the xmcd
val_blue = integral_north_south[700] - val_green #height of the xmcd
print(f'red ={val_red}')
print(f'blue ={val_blue}')
print(f'green ={val_green}')
#%% Step 8: Calculation of L_z and S_z 
n3d = 7
#height of xmcd is q,
L_z = 21*(val_green + val_blue)/(2*val_red)
S_z = 7*((val_green - 6*val_blue)/val_red)
print(f'L_z = {L_z}')
print(f'S_z =  {S_z}')
# %% final exercise (exam)

