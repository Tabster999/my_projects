#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv 
#%%

def Iterator(energy:float, eps, t):
    z = (energy + 1e-4j) * np.eye(2)   #dimension?
    alpha = t 
    beta =  np.conjugate(alpha.T)
    Epsilon_surf = eps  
    Epsilon_bulk = eps 
    
    for i in range(200):
        aux = inv((z - Epsilon_bulk))
        Epsilon_surf = Epsilon_surf + alpha @ aux @ beta 
        Epsilon_bulk = Epsilon_bulk + alpha @ aux @ beta + beta @ aux @ alpha
        alpha = alpha @ aux @ alpha 
        beta = beta @ aux @ beta

    Gs = inv((z - Epsilon_surf)) # replace with np.linalg.inv
    Gb = inv((z - Epsilon_bulk))

    return Gs,Gb

#%%
#Initialize some important parameters
t = 1
delta_s = .0
mu = .25
onsite = 2*t-mu
eps = np.array([[onsite, delta_s], [delta_s, - onsite]]) 
delta_p = .1
t_matrix = np.array([[- t, delta_p], [-delta_p, t]])
energy_array = np.linspace(-2, 2 , 400)

surface_ldos = np.zeros(len(energy_array))
bulk_ldos = np.zeros(len(energy_array))
for i, e in enumerate(energy_array):
    Gs, Gb = Iterator(e, eps, t_matrix)
    bulk_ldos[i] = - 1/(np.pi)*np.trace(Gb.imag)
    surface_ldos[i] = - 1/(np.pi)*np.trace(Gs.imag)

#Setup for the plot
#%%
plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
plt.plot(energy_array, surface_ldos, label='Electrons', c='blue')
plt.xlabel('Energy')
plt.ylabel('LDOS')
plt.title('Surface LDOS')
plt.grid()

plt.subplot(2,2,3)
plt.plot(energy_array,  bulk_ldos, label='Electrons', c='red')
plt.xlabel('Energy')
plt.ylabel('LDOS')
plt.title('Bulk LDOS')
plt.grid()
plt.show()

#%%