#%% import modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import my_functions as myf
from  bachelorarbeit import my_functions_ba as bach
import pandas as pd 
#%% initialize parameters 
sites = 150
t = 1.0
mu = 0.025 * t
alpha = 0.4 * t
delta = 0.1 * t
eta = 1e-4 * delta
h = 0.2 * t

e_min = - .025 * delta
e_max = -e_min
steps = 301
energy_array = np.linspace(e_min,e_max,steps)

phase_transition = np.sqrt(delta**2+mu**2)

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20

#%% computation of the greens functions
hamiltonian = myf.tb_hamiltonian_1d(sites, t, mu, h, alpha, delta)
gr = myf.get_G_energy(energy_array,hamiltonian, eta=eta, ra='r')
ga = myf.get_G_energy(energy_array, hamiltonian, eta=eta, ra='a')

gr_inf = np.zeros((len(energy_array)), dtype=np.complex128)
gbr_inf = np.zeros((len(energy_array)), dtype=np.complex128)
gr_inf_all = np.zeros((len(energy_array), 4,4), dtype=np.complex128)

d0_inf = np.zeros((len(energy_array)), dtype=np.complex128)
dx_inf = np.zeros((len(energy_array)), dtype=np.complex128)
dy_inf = np.zeros((len(energy_array)), dtype=np.complex128)
dz_inf = np.zeros((len(energy_array)), dtype=np.complex128)

for i,e in enumerate(energy_array):
    Gs_r, Gb_r = bach.Iterator_retarded(e, myf.onsite_matrix(t, mu, h, delta), myf.t_matrix(t, alpha), eta=eta)
    Gs_a, Gb_a =bach.Iterator_advanced(e, myf.onsite_matrix(t, mu, h, delta), myf.t_matrix(t, alpha), eta=eta)
    
    gr_inf[i] = Gs_r[0,0]
    gbr_inf[i] = Gb_r[0,0]
    gr_inf_all[i] = Gs_r
    
    d0_inf[i] = .5*((Gs_a[0,2] - Gs_r[0,2]) - (Gs_a[1,3] - Gs_r[1,3]))
    dx_inf[i] = -.5*((Gs_a[0,3] - Gs_r[0,3]) - (Gs_a[1,2] - Gs_r[1,2]))
    dy_inf[i] = -.5j*((Gs_a[0,3] - Gs_r[0,3]) + (Gs_a[1,2] - Gs_r[1,2]))
    dz_inf[i] = .5*((Gs_a[0,2] - Gs_r[0,2]) + (Gs_a[1,3] - Gs_r[1,3]))
    
    
#%% computation of ldos
A = (-1/np.pi)
nambu_index = 1 #  1-> up-electron, 2-> down-electron, 3-> down-hole, 4-> up-hole
bulk_index = (sites // 2) * 4

ldos_E_surf = A*np.imag(gr[:,0,0])
ldos_inf_surf = A*np.imag(gr_inf)

ldos_E_bulk = A*np.imag(gr[:, bulk_index, bulk_index])
ldos_inf_bulk = A*np.imag(gbr_inf)
#%% plot ldos edge comparison
textstr = '\n'.join((
r'$\mu/t = {:.3f}$'.format(mu),
r'$t = {:.1f}$'.format(t),
r'$\alpha/t = {:.1f}$'.format(alpha),
r'$\Delta/t = {:.1f}$'.format(delta)
))
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)

plt.figure(figsize=(10,7), dpi=130)
plt.plot(energy_array/delta, ldos_inf_surf, c='blue', label='infinite')
plt.plot(energy_array/delta, ldos_E_surf, "r.", label='finite')
plt.legend(loc=0)
plt.ylim(-.01,5)
plt.xlabel(r'$\frac{\epsilon}{\Delta}$')
plt.ylabel(r'LDOS up-$e^-$')
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.title(f'LDOS for N={sites} sites')
#plt.savefig(r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\Masterarbeit\plots\comparison inf and fin 1D (testing)\LDOS egde e_u comparison 2.png')
plt.show()
#%% plot ldos bulk comparison
plt.figure()
plt.plot(energy_array/delta, ldos_inf_bulk, c='blue', label='infinite')
plt.plot(energy_array/delta, ldos_E_bulk, "r.", label='finite')
plt.legend(loc=0)
#plt.ylim(-.01,5)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
plt.xlabel(r'$\frac{\epsilon}{\Delta}$')
plt.ylabel(r'LDOS up-$e^-$')
plt.title(f'bulk (N={sites}, bulk site={bulk_index/4})')
#plt.savefig(r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\Masterarbeit\plots\comparison inf and fin 1D (testing)\LDOS bulk e_u comparison 2.png')
plt.show()
#%% comparison plots with myf.quick_plot (same as the two above)
ldos_list_surf = [ldos_E_surf, ldos_inf_surf]
ldos_titles_surf = [f" $ G_{{r,surf}}(E)$ Finite chain (N={sites})", r'$G_{r,surf}(E)$ Infinite chain ']
colors=['orange', 'blue']
labels = [f'finite (N={sites})', 'infinite']
xlabel = r'$\frac{Energy}{\Delta}$'
ylabel = 'LDOS'
title_surf = r'$G_{r,surf}(E)$'
y_lims = [0,5]

title_bulk = f" $ G_{{r,bulk}}$"
myf.quick_plot(energy_array/delta, [ldos_E_surf, ldos_inf_surf], colors=colors, title=title_surf, ylim=y_lims,
                ylabel=ylabel, xlabel=xlabel, labels=['finite', 'infinite'], legend=True)
myf.quick_plot(energy_array/delta, [ldos_E_bulk, ldos_inf_bulk], colors=colors, title=title_bulk, ylim=y_lims,
                ylabel=ylabel, xlabel=xlabel, legend=1, labels=['finite', 'infinite'])
#%% plot evals over magnetic field 
eval = []
h_array = np.linspace(.05,.3,151)

for h_val in h_array:
    my_ham = myf.tb_hamiltonian_1d(150, t, mu, h_val, alpha, delta)
    eval_entry, evec = np.linalg.eigh(my_ham)
    eval.append(np.sort(eval_entry))


plt.figure(figsize=(12,7), dpi=150)
plt.plot(h_array, eval, c='blue')
plt.axvline(x=phase_transition)
plt.xlabel('Zeeman')
plt.ylabel('Eigenvalues')
plt.title(f'1d finite chain (N={sites})')
plt.ylim(-.15,.15)
plt.show()
#%% dos over energy plots 
diags = np.array(np.trace(gr_inf_all.imag, axis1=1, axis2=2))
dos_E_surf = np.zeros(len(energy_array))
dos_E_surf_inf = A*diags
for i in range(4):
    dos_E_surf += A*np.imag(gr[:,-1-i,-1-i])

plt.figure(figsize=(10,7), dpi=150)
plt.plot(energy_array/delta,dos_E_surf, 'r.', label='finite')
plt.plot(energy_array/delta, dos_E_surf_inf, c='blue', label='infinite')
plt.title(f'Surface DOS for all spin DOFs (N={sites})')
plt.xlabel(r'$\frac{\epsilon}{\Delta}$')
plt.ylabel(r'DOS')
plt.legend(loc=0)
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', bbox=props)
#plt.savefig(r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\Masterarbeit\plots\comparison inf and fin 1D (testing)\DOS egde all dof comparison.png')
plt.show()
#%% dos over position plots 
gfaux = np.reshape(gr,[len(energy_array),sites,2,2,sites,2,2])
dos_x =np.einsum('eisnisn->ei',A*gfaux.imag)

plt.figure(figsize=(10,7), dpi=150)
plt.plot(dos_x[125,:], c='blue')
plt.ylim(-.01,5)
plt.xlabel(r'site (index i)')
plt.ylabel(r'DOS')
plt.title(f'DOS over position  (N={sites})')
plt.text(0.05, 0.95, textstr, fontsize=10, transform=plt.gca().transAxes,
verticalalignment='top', horizontalalignment='left', bbox=props)
#plt.savefig(r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\Masterarbeit\plots\comparison inf and fin 1D (testing)\DOS egde over position.png')
plt.show()
#%% computing and plotting the pairing amplitudes for inf and finite models
gr_rsh = gr.reshape(len(energy_array),sites,2,2,sites,2,2) # indices: energy, position, electron (0)/hole (1), spin
ga_rsh = ga.reshape(len(energy_array),sites,2,2,sites,2,2) # indices: energy, position, electron (0)/hole (1), spin 

fr_edge = gr_rsh[:, 0, 0, :, 0, 1, :] # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)
fa_edge = ga_rsh[:, 0, 0, :, 0, 1, :] # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)

fr_bulk = gr_rsh[:, sites//2+1, 0,:,sites//2+1, 1,:] # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)
fa_bulk = ga_rsh[:, sites//2+1, 0,:,sites//2+1, 1,:] # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)

f_edge = fa_edge - fr_edge # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)
f_bulk = fa_bulk - fr_bulk # indices: energy, electron spin up (0)/spin down (1), hole spin up (1)/spin down (0)

d0 = .5*(f_edge[:,0,0] - f_edge[:,1,1]) # .5*(up,down - down,up)
dx = -.5*(f_edge[:,0,1] - f_edge[:,1,0]) # -.5*(up,up - down,down)
dy = -.5j*(f_edge[:,0,1] +  f_edge[:,1,0]) # -.5j*(up,up + down,down)
dz = .5*(f_edge[:,0,0] + f_edge[:,1,1]) # .5*(up,down + down,up)

d0r_plots = [d0.real,d0_inf.real]
dxr_plots = [dx.real,dx_inf.real]
dyr_plots = [dy.real,dy_inf.real]
dzr_plots = [dz.real,dz_inf.real]

d0i_plots = [d0.imag,d0_inf.imag]
dxi_plots = [dx.imag,dx_inf.imag]
dyi_plots = [dy.imag,dy_inf.imag]
dzi_plots = [dz.imag,dz_inf.imag]

myf.quick_plot(energy_array/delta, d0r_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_0$', title=r'$d_0$ real', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])
myf.quick_plot(energy_array/delta, d0i_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_0$', title=r'$d_0$ imag', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])

myf.quick_plot(energy_array/delta, dxr_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_x$', title=r'$d_x$ real', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])
myf.quick_plot(energy_array/delta, dxi_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_x$', title=r'$d_x$ imag', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])

myf.quick_plot(energy_array/delta, dyr_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_y$', title=r'$d_y$ real', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])
myf.quick_plot(energy_array/delta, dyi_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_y$', title=r'$d_y$ imag', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])

myf.quick_plot(energy_array/delta, dzr_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_z$', title=r'$d_z$ real', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])
myf.quick_plot(energy_array/delta, dzi_plots, xlabel=r'$\frac{\epsilon}{\Delta}$', ylabel=r'$d_z$', title=r'$d_z$ imag', legend=True, colors=['orange', 'blue'], labels=['finite', 'infinite'])
#%% plotting the pairing amplitudes seperately
labels = ['finite', 'infinite']
colors = ['orange', 'blue']

# Data lists: [finite_real, infinite_real, finite_imag, infinite_imag]
d0_list = [d0.real, d0_inf.real, d0.imag, d0_inf.imag]
dx_list = [dx.real, dx_inf.real, dx.imag, dx_inf.imag]
dy_list = [dy.real, dy_inf.real, dy.imag, dy_inf.imag]
dz_list = [dz.real, dz_inf.real, dz.imag, dz_inf.imag]

di_lists = [d0_list, dx_list, dy_list, dz_list]
titles = [r'$d_0$', r'$d_x$', r'$d_y$', r'$d_z$']
ftitle = ['d_0', 'd_x', 'd_y', 'd_z']
save_dir = r'C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\Masterarbeit\plots\comparison inf and fin 1D (testing)'

for d_list, title, fname in zip(di_lists, titles, ftitle):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), dpi=150, sharex=True)
    fig.suptitle(f'{title}', fontsize=14)

    axs[0].plot(energy_array/delta, d_list[0],  'r.', label='finite')
    axs[0].plot(energy_array/delta, d_list[1], label='infinite', c='blue')
    axs[0].set_title('Real Part')
    axs[0].set_ylabel(title)
    axs[0].legend()

    axs[1].plot(energy_array/delta, d_list[2], 'r.', label='finite')
    axs[1].plot(energy_array/delta, d_list[3], label='infinite', c='blue')
    axs[1].set_title('Imaginary Part')
    axs[1].set_xlabel(r'$\epsilon / \Delta$')
    axs[1].set_ylabel(title)
    axs[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    #filename = f'{fname} comparison real imag.png'
    #fig.savefig(f'{save_dir}\\{filename}')
    plt.show()
# %%
