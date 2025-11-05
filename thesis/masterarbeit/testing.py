#%%
import os 
os.chdir(r'C:\Main\my_projects\masterarbeit')
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import my_functions as myf
import time
#%% initialize parameters 
r''' hamiltonian structure
in the basis \Psi^\dagger = (\Psi_\uparrow^\dagger, \Psi_\downarrow^\dagger, \Psi_\downarrow, \Psi_\uparrow) the Hamiltonian operator is given by:
    H = H_{kin} + H_\alpha + H_z + H_\Delta
with:
    H_kin = \sum_{i,j} c_i^\dagger (1/2m) *(p_x^2 + p_y^2) \sigma_0 \tau_z c_j
    H_\alpha = \alpha \sum_i c_i^\dagger (p_x \sigma_x + p_y \sigma_y) \tau_z c_i
    H_z = h \sum_i c_i^\dagger \sigma_z c_i
    H_\Delta = \sum_i [\Delta * (c_{i,\uparrow} c_{i,\downarrow}) + \Delta^\star * (c_{i,\uparrow}^\dagger c_{i,\downarrow}^\dagger)]

the index i is a superindex for (x,y)
'''

sites = 101
t = 1.0
mu = 0.025 * t
alpha = .6 * t
delta = 0.1 * t
eta = 1e-6 * delta
h = 0.2 * t
phase_transition = np.sqrt(delta**2+mu**2)


e_steps = 51
e_min = - delta
e_max = -e_min
energy_array = np.linspace(e_min, e_max, e_steps)


h_steps = 81
h_min = .05
h_max = .3
h_array = np.linspace(h_min, h_max, h_steps)


ky_steps = 81
ky_min = -np.pi
ky_max = np.pi
ky_array = np.linspace(ky_min, ky_max, ky_steps)

textstr = '\n'.join((
r'$\mu/t = {:.3f}$'.format(mu),
r'$t = {:.1f}$'.format(t),
r'$\alpha/t = {:.1f}$'.format(alpha),
r'$\Delta/t = {:.1f}$'.format(delta)
))
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
#%% k_y good quantum number k_x not


r''' hamiltonian after fourier transform
After partial FT y -> k_y:
    H_kin -> -t * \sum_{x,k_y} c_{x,k_y}^\dagger *(c_{x-1,k_y} + c_{x+1,k_y} - 4*c_{x,k_y} + 2*cos(k_y^\tilde*a)*c_{x,k_y})     
    H_\alpha -> -i*\alpha^\tilde * \sum_{x,k_y} c_{x,k_y}^\dagger *[\sigma_x*(c_{x+1,k_y} - c_{x-1,k_y}) - \sigma_y * (2i*sin(k_y^\tilde*a))] 
with:
    t = \hbar^2/(2ma)
    \alpha^\tilde = \hbar*\alpha / 2a
    k_y^\tilde = (2m*k_y) / \hbar
'''

def get_hopping(t, alpha):
    t_matrix = np.array([[-t, alpha, 0, 0], 
                       [-alpha, -t, 0, 0],
                       [0, 0, t, alpha], 
                       [0, 0, -alpha, t]], 
                        dtype=np.complex128)
    
    return t_matrix

def get_h0(t, mu, h, alpha, delta, k_y):
    onsite_matrix = np.array(
        [[-2*t*(np.cos(k_y) - 2) - mu + h, -2*np.sin(k_y)*alpha, delta, 0], 
        [-2*np.sin(k_y)*alpha, -2*t*(np.cos(k_y) - 2) - mu - h, 0, -delta], 
        [delta, 0, 2*t*(np.cos(k_y) - 2) + mu + h, -2*np.sin(k_y)*alpha], 
        [0, -delta, -2*np.sin(k_y)*alpha, 2*t*(np.cos(k_y) - 2) + mu - h]], 
        dtype=np.complex128)

    return onsite_matrix

def get_h0_vectorized(t, mu, h, alpha, delta, ky_array):
    cos_ky = np.cos(ky_array)
    sin_ky = np.sin(ky_array)
    
    onsite_matrix = np.zeros(
        (len(ky_array), 4, 4), dtype=np.complex128)
    
    onsite_matrix[:, 0, 0] = -2 * t * (cos_ky - 2) - mu + h
    onsite_matrix[:, 1, 1] = -2 * t * (cos_ky - 2) - mu - h
    onsite_matrix[:, 2, 2] = 2 * t * (cos_ky - 2) + mu + h
    onsite_matrix[:, 3, 3] = 2 * t * (cos_ky - 2) + mu - h
    onsite_matrix[:, 0, 1] = -2 * sin_ky * alpha
    onsite_matrix[:, 1, 0] = -2 * sin_ky * alpha
    onsite_matrix[:, 0, 2] = delta
    onsite_matrix[:, 2, 0] = delta
    onsite_matrix[:, 1, 3] = -delta
    onsite_matrix[:, 3, 1] = -delta
    onsite_matrix[:, 2, 3] = -2 * sin_ky * alpha
    onsite_matrix[:, 3, 2] = -2 * sin_ky * alpha
    
    return onsite_matrix


def get_tb_hamiltonian_vectorized(h0_matrices, hopping_matrix, sites):
    num_ky = h0_matrices.shape[0]
    H_large_stack = np.zeros(
        (num_ky, 4 * sites, 4 * sites), dtype=np.complex128)
    
    for i in range(sites):
        H_large_stack[:, 4*i:4*i+4, 4*i:4*i+4] = h0_matrices
        if i < sites - 1:
            H_large_stack[:, 4*i:4*i+4, 4*(i+1):4*(i+1)+4] = -hopping_matrix
            H_large_stack[:, 4*(i+1):4*(i+1)+4, 4*i:4*i+4] = -hopping_matrix.conj().T
            
    return H_large_stack
#%% compute eigenvalues E(h,k_y=k_0) for a fixed k_0 and plot


t_matrix = get_hopping(t, alpha)
eigenvalues_over_h = []
k_0 = 0.0
for h_val in h_array:
    h_single = get_h0(t, mu, h_val, alpha, delta, k_0)
    H_large = myf.get_tb_hamiltonian(h_single, t_matrix, sites)
    
    eigenvalues = np.linalg.eigvalsh(H_large)
    eigenvalues_edge = eigenvalues[0]
    eigenvalues_over_h.append(np.sort(eigenvalues))
eigenvalues_over_h_array = np.transpose(np.array(eigenvalues_over_h))

threshold = 0.01
close_to_zero_mask = np.abs(eigenvalues_over_h_array) < threshold
any_close_to_zero = np.any(close_to_zero_mask, axis=0)
indices = np.where(any_close_to_zero)
magnetic_field_values = h_array[indices]
print("Indices of the magnetic field values where eigenvalues are close to zero:", indices)
print("The corresponding magnetic field values are:", magnetic_field_values)


transition_finite = magnetic_field_values[0]
transition_difference = transition_finite - phase_transition

print(f'Difference between finite and infinite transition value for h is dh={transition_difference:.3f}')
#%% plot Eigenvalues(h) of different magnetic field values 
myf.quick_plot(x_values=h_array, y_list=eigenvalues_over_h_array, v_lines=phase_transition,
               v_line_styles={'color' : 'blue', 'linestyle':'--'}, xlabel='h', ylabel='Eigenvalues',
               xlim=(h_min,h_max), title=f'Eigenvalues for $k_y =$ {k_0:.2f}. and varying h', figsize=(12,7),
               dpi=150, ylim=[-.1,.1],colors='b')

#%% calulate and plot the difference of finite and infitnite phase_transition for a varying number of sites 
sites_array = np.arange(40, 251, 20)

transition_differences = []
actual_sites = []

start_total_time = time.time()

for sites in sites_array:
    print(f"\nRunning simulation for {sites} sites...")
    
    start_sites_time = time.time()
    eigenvalues_over_h = []
    
    for h_val in h_array:
        h_single = get_h0(t, mu, h_val, alpha, delta, k_0)
        H_large = myf.get_tb_hamiltonian(h_single, get_hopping(t, alpha), sites)
        eigenvalues = np.linalg.eigvalsh(H_large)
        eigenvalues_over_h.append(np.sort(eigenvalues))
    
    end_eigen_time = time.time()
    print(f"  Eigenvalue calculation took {end_eigen_time - start_sites_time:.2f} seconds.")

    eigenvalues_over_h_array = np.transpose(np.array(eigenvalues_over_h))
    
    close_to_zero_mask = np.abs(eigenvalues_over_h_array) < threshold
    any_close_to_zero = np.any(close_to_zero_mask, axis=0)
    indices = np.where(any_close_to_zero)
    
    if len(indices[0]) > 0:
        transition_finite = h_array[indices[0][0]] 
        transition_difference = transition_finite - phase_transition
        
        transition_differences.append(transition_difference)
        actual_sites.append(sites)
    else:
        print(f"  No transition found for sites = {sites}. Try adjusting h_array range.")
    
    end_sites_time = time.time() 
    print(f"  Total time for {sites} sites: {end_sites_time - start_sites_time:.2f} seconds.")
    
end_total_time = time.time() 
print(f"\nTotal simulation time: {end_total_time - start_total_time:.2f} seconds.")

# --- Plot the results ---
plt.figure(figsize=(10, 6))
plt.scatter(actual_sites, transition_differences, label='Numerical Result')
plt.plot(actual_sites, transition_differences, linestyle='--')
plt.xlabel('Number of Sites (N)')
plt.ylabel('Difference in Transition Field (h_finite - h_infinite)')
plt.title('Finite-Size Scaling of the Topological Phase Transition')
plt.grid(True)
plt.legend()
plt.show()
#%% compute eigenvalues E(h, k_y) for varying k_y and h


eigenvalues_over_ky_h = np.zeros((len(h_array), len(ky_array), 4*sites), dtype=np.complex128)
start_total_time = time.time()
hopping_matrix = get_hopping(t, alpha)

for i, h_val in enumerate(h_array):
    if i//10 == 0:
        
        print(f"Progress: {i+1}/{len(h_array)}")
    
    h0_matrices = get_h0_vectorized(t, mu, h_val, alpha, delta, ky_array)
    
    H_large_stack = get_tb_hamiltonian_vectorized(
        h0_matrices, hopping_matrix, sites)
    

    for j in range(len(ky_array)):
        evals = np.linalg.eigvalsh(H_large_stack[j])
        eigenvalues_over_ky_h[i, j, :] = np.sort(evals)

end_total_time = time.time()
print(f"\nTotal simulation time: {end_total_time - start_total_time:.2f} seconds.")

#%% plots of eigenvalues over ky and h
# Plot 1: E vs. h for a fixed ky
fixed_ky_val = 0.0
fixed_ky_idx = np.abs(ky_array - fixed_ky_val).argmin()
eigenvalues_fixed_ky = np.real(eigenvalues_over_ky_h[:, fixed_ky_idx, :])

fixed_h_val = 0.2
fixed_h_idx = np.abs(h_array - fixed_h_val).argmin()
eigenvalues_fixed_h = np.real(eigenvalues_over_ky_h[fixed_h_idx, :, :])

majorana_idx_neg = 2 * sites - 1
majorana_idx_pos = 2 * sites
majorana_band_neg = np.real(eigenvalues_over_ky_h[:,:, majorana_idx_neg])
majorana_band_pos = np.real(eigenvalues_over_ky_h[:,:, majorana_idx_pos])


plt.figure(figsize=(10, 7))
for band in range(eigenvalues_fixed_ky.shape[1]):
    plt.plot(h_array, eigenvalues_fixed_ky[:, band], color='blue', alpha=0.5)
plt.xlabel('Magnetic Field, h')
plt.ylabel('Eigenenergy, E')
plt.title(f'Eigenenergy vs. Magnetic Field (h) at $k_y$ = {fixed_ky_val:.2f}')
plt.grid(True)
plt.axvline(x=np.sqrt(0.1**2 + 0.025**2), color='black', linestyle='--', label='Theoretical Transition')
plt.legend(loc=1)
plt.ylim((-.5,.5))
plt.show()

# Plot 2: E vs. ky for a fixed h
plt.figure(figsize=(10, 7))
for band in range(eigenvalues_fixed_h.shape[1]):
    plt.plot(ky_array, eigenvalues_fixed_h[:, band], color='blue', alpha=0.5)
plt.xlabel('Momentum, $k_y$')
plt.ylabel('Eigenenergy, E')
plt.title(f'Eigenenergy vs. Momentum ($k_y$) at h = {fixed_h_val:.2f}')
plt.grid(True)
plt.legend()
plt.ylim((-.5,.5))
#plt.savefig('eigenvalues_vs_ky.png')
plt.show()

# Plot 3: Contour plot of a selected band 
H, Ky = np.meshgrid(h_array, ky_array)
vmax_energy = np.max(np.abs(majorana_band_pos))
vmin_energy = -vmax_energy

plt.figure(figsize=(10, 7))
#plt.contour(H, Ky, majorana_band_pos.T, levels=50, cmap='bwr')
plt.pcolormesh(H, Ky, majorana_band_pos.T, cmap='bwr', vmin=-.2,vmax=.2,shading='gouraud')
plt.colorbar(label='Eigenenergy E')
plt.xlabel('Magnetic Field, h')
plt.ylabel('Momentum, $k_y$')
#plt.savefig('eigenenergy_contour.png')
plt.show()

#%% computation of G_r(h, E) using the Lehmann representation
G_r_lehmann = []
sites = 50
ky_val = 0

total_start = time.time()
for i, h_val in enumerate(h_array):
    #print(f"Progress: {i+1}/{len(h_array)}")
    step_start = time.time()
    my_ham = myf.get_tb_hamiltonian(h0_matrix=get_h0(t, mu, h_val, alpha, delta, ky_val), hopping_matrix=get_hopping(t, alpha), sites=55)
    evals, evecs = np.linalg.eigh(my_ham)
    G_r = myf.calc_G_lehmann(evals=evals, evecs=evecs, energy_array=energy_array, eta=eta)
    G_r_lehmann.append(G_r)
    step_end = time.time()
    #print(f'Step {i+1} took {step_end - step_start:.4f} s')
total_end = time.time()
#print(f'Total time {total_end - total_start:.4f} s = {(total_end - total_start)/60} min')

G_r_lehmann_array = np.array(G_r_lehmann)
#%%
em,hm = np.meshgrid(energy_array,h_array)
ldos_aux = -np.einsum('ijkk->ij',G_r_lehmann_array.imag)
#%%
plt.figure()
plt.contourf(hm,em,np.clip(ldos_aux,0,10),levels=300,cmap='hot')
plt.xlabel('Z')
plt.ylabel(r'$\epsilon$')
plt.show()



#%% Plots of G_r(h ,E) calculated by Lehmann representation (either fix h or E)
h_index = 80
fixed_energy = 0.000
energy_index = np.argmin(np.abs(energy_array - fixed_energy))
h_0 = h_array[h_index]
E_0 = energy_array[energy_index]

gr_edge_lehmann_h = G_r_lehmann_array[:, energy_index, 0:4, 0:4]
gr_edge_lehmann_E = G_r_lehmann_array[h_index, :, 0:4, 0:4]

total_ldos_lehmann_h = (-1/np.pi) * np.imag(np.trace(gr_edge_lehmann_h, axis1=1, axis2=2))
total_ldos_lehmann_E = (-1/np.pi) * np.imag(np.trace(gr_edge_lehmann_E, axis1=1, axis2=2))

plt.figure(figsize=(10,7), dpi=100)
plt.plot(h_array, total_ldos_lehmann_h, label='Lehmann', c='orange')
plt.xlabel("h")
plt.ylabel("LDOS")
plt.title(f'LDOS when $k_y$ is a good QN. N={sites}, $E_0=${E_0} and $k_y=${ky_val}')
plt.axvline(x=phase_transition, label='phase transition', linestyle='--')
plt.legend(loc=4)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.show()

plt.figure(figsize=(10,7), dpi=100)
plt.plot(energy_array, total_ldos_lehmann_E, label='Lehmann', c='orange')
plt.xlabel("Energy")
plt.ylabel("LDOS")
plt.title(f'LDOS when $k_y$ is a good QN. N={sites}, $h_0=${h_0} and $k_y=${ky_val}')
plt.legend(loc=4)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.show()
#%% computation of G_r(h, E) by inverting
sites = 61
ky_val = 0.0
my_t = get_hopping(t, alpha)
gr = []

total_start = time.time()
for i, h_val in enumerate(h_array):
    #print(f"Progress: {i+1}/{len(h_array)}")
    step_start = time.time()
    
    my_h0 = get_h0(t, mu, h_val, alpha, delta, ky_val)
    my_H = myf.get_tb_hamiltonian(my_h0, my_t, sites)
    gr_entry = myf.get_G_energy(energy_array, my_H, eta=1e-6, ra='r')
    gr.append(gr_entry)

    step_end = time.time()
    #print(f'Step {i+1} took {step_end - step_start:.4f} s')
total_end = time.time()
#print(f'Total time {total_end - total_start:.4f} s = {(total_end - total_start) // 60 } min {(total_end - total_start) % 60} s')
G_r_array = np.array(gr)
#%% Plots of G_r(h ,E) calculated by Inversion 
fixed_h = 80
h_index = np.argmin(np.abs(h_array - fixed_h))
h_0 = h_array[h_index]

fixed_energy = 0.00
energy_index = np.argmin(np.abs(energy_array - fixed_energy))
E_0 = energy_array[energy_index]

gr_edge_h = G_r_array[:, energy_index, 0:4, 0:4]
gr_edge_E = G_r_array[h_index, :, 0:4, 0:4]

total_ldos_h = (-1/np.pi) * np.imag(np.trace(gr_edge_h, axis1=1, axis2=2))
total_ldos_E = (-1/np.pi) * np.imag(np.trace(gr_edge_E, axis1=1, axis2=2))

'Plot LDOS over h'
plt.figure(figsize=(10,7), dpi=100)
plt.plot(h_array, total_ldos_h, label='Inversion', c='blue')
plt.xlabel("h")
plt.ylabel("LDOS")
plt.title(f'LDOS $k_y$ good qn. N={sites}, $E_0=${E_0}, $k_y=${ky_val}')
plt.axvline(x=phase_transition, label='phase transition', linestyle='--', c='orange')
plt.legend(loc=1)
#plt.ylim(0,6)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.show()

'Plot: LDOS over energy'
plt.figure(figsize=(10,7), dpi=100)
plt.plot(energy_array, total_ldos_E, label='Inversion', c='blue')
plt.xlabel("Energy")
plt.ylabel("LDOS")
plt.title(f'LDOS $k_y$ good qn. N={sites}, $h_0=${h_0}, $k_y=${ky_val}')
plt.legend(loc=1)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.show()
# %%
"""To be added:
t_matrix = get_hopping(t, alpha)
k_0 = 0
h_single = get_h0(t, mu, .2, alpha, delta, k_0)
ham = myf.get_tb_hamiltonian(h_single, t_matrix, sites)

zen = energy_array +1e-4j
zId = zen[:,None,None]*np.eye(ham.shape[0])[None,:,:]

Gret_aux = np.linalg.inv(zId-ham[None,:,:])
Gret_aux = Gret_aux.reshape(len(energy_array),sites,4,sites,4)

ldos = np.einsum('esnsn->e',Gret_aux.imag)/np.pi

plt.figure()
plt.plot(energy_array,ldos)
plt.show()

dos_x = np.einsum('esnsn->es',Gret_aux.imag)/np.pi
plt.figure()
plt.plot(dos_x[len(energy_array)//2,:])
plt.show()
"""
#%% compute DOS over position for Inversion
gr_resh = np.reshape(G_r_array, [len(h_array), len(energy_array), sites, 4, sites, 4]) #h, E, i, s, i, s 
ldos_full = (-1/np.pi)*np.imag(np.einsum('ehisis->ehi', gr_resh))
#%% plot of LDOS over position
ldos_x = ldos_full[h_index, energy_index, :]
plt.figure(figsize=(10,7), dpi=150)
plt.plot(ldos_x, c='blue')
plt.ylim(-.01,1)
plt.xlabel(r'site (index i)')
plt.ylabel(r'DOS')
plt.text(0.5, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.title(f'LDOS over position (ky good qn). N={sites}, h={h_array[h_index]}, E={energy_array[energy_index]}')
plt.show()
#%% Extracting the anomalous Green function (off diagonal elements) and plotting them 
gr_edge = gr_resh[:,:,0,:,0,:]
fr_over_h = gr_edge[:,energy_index, 0:1, 2:3]
ldos_fr = -np.trace(np.imag(fr_over_h), axis1=1, axis2=2)
plt.figure(figsize=(10,7), dpi=150)
plt.plot(h_array, ldos_fr, c='blue')
#plt.ylim(-.01,5)
plt.xlabel(r'h')
plt.ylabel(r'Anomalous Green function')
plt.axvline(x=phase_transition, linestyle='--')
plt.text(0.5, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.title(f'LDOS over position (ky good qn). N={sites}, h={h_array[h_index]}, E={energy_array[energy_index]}')
plt.show()

# %%
gr_over_h = np.imag(gr_edge[:,energy_index, 1, 2])
plt.figure(figsize=(10,7), dpi=150)
plt.plot(h_array, gr_over_h, c='blue')
#plt.ylim(-.01,5)
plt.xlabel(r'h')
plt.ylabel(r'Anomalous Green function')
plt.axvline(x=phase_transition, linestyle='--')
plt.text(0.5, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.title(f'LDOS over position (ky good qn). N={sites}, h={h_array[h_index]}, E={energy_array[energy_index]}')
plt.show()

# %%
