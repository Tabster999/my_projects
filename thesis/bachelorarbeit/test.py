#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# define some parameters
def Iterator(energy:float, eps, t):
    z = energy-1e-4j
    alpha = t
    beta = np.conjugate(t)
    Epsilon_surf = eps
    Epsilon_bulk = eps
    Epsilon = eps
    for i in range(70):
        aux = 1/(z-Epsilon)
        Epsilon_surf = Epsilon_surf + alpha*aux*beta
        Epsilon = Epsilon + alpha*aux*beta + beta*aux*alpha
        alpha = alpha*aux*alpha
        beta = beta*aux*beta

    Gs = 1/(z-Epsilon_surf)
    Gb = 1/(z-Epsilon)

    return Gs,Gb
#%%
eps =.0
t = 0.9



energy_array = np.linspace(-2, 2, 100)
Gs,Gb = Iterator(energy_array, eps, t)

plt.figure(figsize=(10,9))
plt.subplot(2,1,2)
plt.plot(energy_array,t*Gb.imag,label ='bulk')
plt.grid()
plt.tight_layout()
plt.legend()


plt.subplot(2,2,2)
plt.plot(energy_array,t*Gs.imag,label ='surf')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()





# %%
