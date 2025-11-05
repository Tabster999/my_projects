#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv
import bachelorarbeit.my_functions_ba as bach
#%% Calculation of the lesser d_i 

t = 1.0
mu = .2*t
delta = .1
alpha = .0*t
e_min = -.2
e_max = .2
steps = 200
h_array = np.linspace(0.00,0.2,steps)
energy_array = np.linspace(e_min,e_max,steps)
phase_transition = np.sqrt(delta**2 + mu**2)
#%% create arrays for d_i and fill them 
d0 = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
dx = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
dy = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
dz = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
ldos_surf = np.zeros(len(energy_array))
ldos_bulk = np.zeros(len(energy_array))


for i,h in enumerate(h_array):
    for j,e in enumerate(energy_array):
        Gs_a, Gb_a = bach.Iterator_advanced(e,bach.onsite_matrix(t, mu, h, delta),bach.t_matrix(t, alpha))
        Gs_r, Gb_r = bach.Iterator_retarded(e, bach.onsite_matrix(t, mu, h, delta), bach.t_matrix(t, alpha))
        
        f_r = Gs_r[0:2,2:4]
        f_a = Gs_a[0:2,2:4]

        # d0[i, j] = 0.5 * ((f_a[0,0] - f_r[0,0]) - (f_a[1,1] - f_r[1,1]))
        # dx[i, j] = -0.5 * ((f_a[0,1] - f_r[0,1]) - (f_a[1,0] - f_r[1,0]))
        # dy[i, j] = -0.5j *((f_a[0,1] - f_r[0,1]) - (f_a[1,0] - f_r[1,0]))
        # dz[i, j] = 0.5 * ((f_a[0,0] - f_r[0,0]) + (f_a[1,1] - f_r[1,1]))
        
titlestr = ['d0', 'dx', 'dy', 'dz']

#%% plots for d_i over h
energy_index =  100
d_list = [d0[:,energy_index], dx[:,energy_index], dy[:,energy_index], dz[:,energy_index]]
fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=600)
axs = axs.flatten()
for i in range(4):
    ax = axs[i]
    plot = ax.plot(h_array, np.imag(d_list[i].T), 'r', label='imag')
    plot = ax.plot(h_array, np.real(d_list[i].T), 'black', label='real')
    ax.set_title(''+titlestr[i])
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, c='black', linestyle='--')
    ax.legend()
    textstr = '\n'.join((
    r'$\mu/t = {:.3f}$'.format(mu),
    r'$t = {:.1f}$'.format(t),
    r'$\alpha/t = {:.1f}$'.format(alpha),
    r'$\Delta/t = {:.1f}$'.format(delta)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.6, 0.5, textstr, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=props)
plt.tight_layout()
plt.show()
#%% clip and plot 
clip_value = .5
d0_clip_i = np.clip(np.imag(d0.T), -clip_value, clip_value)
dx_clip_i = np.clip(np.imag(dx.T), -clip_value, clip_value)
dy_clip_i = np.clip(np.imag(dy.T), -clip_value, clip_value)
dz_clip_i = np.clip(np.imag(dz.T), -clip_value, clip_value)

d0_clip_r = np.clip(np.real(d0.T), -clip_value, clip_value)
dx_clip_r = np.clip(np.real(dx.T), -clip_value, clip_value)
dy_clip_r = np.clip(np.real(dy.T), -clip_value, clip_value)
dz_clip_r = np.clip(np.real(dz.T), -clip_value, clip_value)

clip_array_i = [d0_clip_i, dx_clip_i, dy_clip_i, dz_clip_i]
clip_array_r = [d0_clip_r, dx_clip_r, dy_clip_r, dz_clip_r]

#real part
fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=600)
axs = axs.flatten()
for i in range(4):
    ax = axs[i]
    plot = ax.contourf(h_array,energy_array, clip_array_r[i], levels =100 ,cmap='bwr')
    ax.set_title(titlestr[i]+' real surface')
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, c='black', linestyle='--')
    textstr = '\n'.join((
    r'$\mu/t = {:.3f}$'.format(mu),
    r'$t = {:.1f}$'.format(t),
    r'$\alpha/t = {:.1f}$'.format(alpha),
    r'$\Delta/t = {:.1f}$'.format(delta),
    r'clip value = {:1}'.format(clip_value)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.6, 0.5, textstr, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=props)
    fig.colorbar(plot, ax=ax)

plt.tight_layout()
plt.show()

#imag part
fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=600)
axs = axs.flatten()
for i in range(4):
    ax = axs[i]
    plot = ax.contourf(h_array,energy_array, clip_array_i[i], levels = 100,cmap='bwr')
    ax.set_title(titlestr[i]+' surface imag ')
    ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, c='black', linestyle='--')
    textstr = '\n'.join((
    r'$\mu/t = {:.3f}$'.format(mu),
    r'$t = {:.1f}$'.format(t),
    r'$\alpha/t = {:.1f}$'.format(alpha),
    r'$\Delta/t = {:.1f}$'.format(delta),
    r'clip value = {:1}'.format(clip_value)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.6, 0.5, textstr, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=props)
    fig.colorbar(plot, ax=ax)
plt.tight_layout()
plt.show()
# %%calculate and plot the integrals over e of d_i
interval = (e_max - e_min)/steps
d0_lesser_int = np.trapz(d0,energy_array,axis=1,dx=interval)
dx_lesser_int = np.trapz(dx,energy_array,axis=1,dx=interval)
dy_lesser_int = np.trapz(dy,energy_array,axis=1,dx=interval)
dz_lesser_int = np.trapz(dz,energy_array,axis=1,dx=interval)
integrals = [d0_lesser_int, dx_lesser_int, dy_lesser_int, dz_lesser_int]

fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=600)
axs = axs.flatten()
for i in range(4):
    ax = axs[i]
    plot = ax.plot(h_array, integrals[i].imag, 'r', label='imag')
    plot = ax.plot(h_array, integrals[i].real, 'black', label='real')
    #ax.set_title('Integral over e for '+titlestr[i])
    #ax.set_xlabel('h')
    ax.set_facecolor('lightyellow')
    ax.axvline(x=phase_transition, c='black', linestyle='--')
    ax.legend()
    textstr = '\n'.join((
    r'$\mu/t = {:.3f}$'.format(mu),
    r'$t = {:.1f}$'.format(t),
    r'$\alpha/t = {:.1f}$'.format(alpha),
    r'$\Delta/t = {:.1f}$'.format(delta)
))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)
    fig.text(0.6, 0.5, textstr, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=props)
plt.tight_layout()
plt.show()


#%%
energy_array = np.linspace(-2,2,400)
t = 1.0
mu = .25*t
delta = .1
alpha = .6*t
h = 0.3*t
gr = np.zeros((len(energy_array)), dtype=complex)
ga = np.zeros((len(energy_array)), dtype=complex)
gba = np.zeros((len(energy_array)), dtype=complex)
gbr = np.zeros((len(energy_array)), dtype=complex)
for i,e in enumerate(energy_array):
    Gs_r, Gb_r = bach.Iterator_retarded(e, bach.onsite_matrix(t, mu, h, delta), bach.t_matrix(t, alpha))
    
    gr[i] = -(1/np.pi)*Gs_r[1,1]
    gbr[i] = -(1/np.pi)*Gb_r[0][0]

    
plt.plot(energy_array,gr, label='surface')
#plt.plot(energy_array,gbr, label='bulk')
plt.legend()
plt.show()
# %%
