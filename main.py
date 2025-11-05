#%%
import numpy as np
import matplotlib.pyplot as plt 
from setup import MySystem as mys
#%%

if __name__ == "__main__":
    z = np.linspace(-5, 5, 100)
    swave = 2
    pwave = 0
    onsite_energy = 0
    t = 100
    energy_value = 0
    instance = mys(t, swave, pwave, onsite_energy, energy_value)
    H, V = instance.get_hamiltonian()
    
    Gs, Gb = mys.get_green_functions(instance)
    
    plt.figure(figsize=(14,8))
    plt.plot(energy_value, t * Gb.imag, label='Electrons', c='blue')
    plt.xlabel('Energy')
    plt.ylabel('DOS')
    plt.ylim([0,2])
    plt.legend()
    plt.title('Bulk DOS')
    plt.grid()
    

# %%
