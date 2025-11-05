#%%
import numpy as np 
import matplotlib.pyplot as plt 
import bachelorarbeit.my_functions_ba as myf
import scipy as sc
#%%
def Iterator(energy:float, eps, t): 
    '''
    Iteration that returns the Gs(advanced surface Green-function) and Gb(advanced bulk Green-function) as a complex 4x4 matrix
    -energy: Onsite energy 
    -eps: onsite Hamiltonian
    -t_matrix: hoppping matrix
    '''
    z = (energy + 1e-5j) * np.eye(2)   
    alpha = t
    beta =  np.transpose(np.conjugate(t))
    Epsilon_surf = eps
    Epsilon_bulk = eps

    for i in range(2000):
        aux = np.linalg.inv((z - Epsilon_bulk))
        Epsilon_surf = Epsilon_surf + alpha@aux@beta
        Epsilon_bulk = Epsilon_bulk + alpha@aux@beta + beta@aux@alpha
        alpha = alpha@aux@alpha
        beta = beta@aux@beta
        if np.linalg.norm(alpha) < 1e-6 :
            break
    Gs = np.linalg.inv(z - Epsilon_surf)
    Gb = np.linalg.inv(z - Epsilon_bulk)
    
    return Gs, Gb
# %%
t = .0
delta = 0
mu = -1
energy_array = np.linspace(-10,10,500)
onsite = np.array([[-mu, 0], [0, mu]])
hopping = np.array([[-t,delta],[-delta,t]])
ldos_surf = np.zeros(len(energy_array), dtype=np.complex128)
ldos_bulk = np.zeros(len(energy_array), dtype=np.complex128)
mu_array = np.linspace(-1,1,300)
for i,e in enumerate(energy_array):
    Gs, Gb = Iterator(e, onsite, hopping)
    ldos_surf[i] = -Gs.imag[0,0]
    ldos_bulk[i] = -Gb.imag[0,0]

    
plt.figure()
plt.plot(energy_array, ldos_surf, label='surface')
plt.plot(energy_array, ldos_bulk, label='bulk')
plt.legend()
plt.gca().set_facecolor('lightgray')
plt.xticks([])
plt.yticks([])
plt.show()
# %%
def my_ham(mu, t, delta):
    ham = np.array([[-mu+delta, t], [t, -mu-delta]]) 
    return ham

elis = [] 
elis2 = []
for mu in mu_array:
    energies = np.linalg.eigvals(my_ham(mu, t, delta))
    elis.append(energies[0])
    elis2.append(energies[1])
plt.plot(mu_array, elis)
plt.plot(mu_array, elis2)
# %%
