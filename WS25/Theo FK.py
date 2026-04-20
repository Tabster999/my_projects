#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def plot_hofstadter_butterfly(max_q=100):
    t = 1.0
    phis = []
    energies = []
    values = []
    for q in range(1, max_q + 1):
        for p in range(0, q + 1):
            if np.gcd(p, q) != 1:
                continue
                
            phi = p / q
            
            H = np.zeros((q, q), dtype=complex)
            
            for n in range(q):
                H[n, n] = 2 * t * np.cos(2 * np.pi * n * phi)
                
                H[n, (n + 1) % q] += t
                H[(n + 1) % q, n] += t
            
            eigvals = np.linalg.eigvalsh(H)
            
            for e in eigvals:
                phis.append(phi)
                energies.append(e)
                values.append((e, phi))
    plt.figure(figsize=(10, 10), facecolor='black')
    #plt.scatter(phis, energies, s=0.1, color='cyan', marker='o', alpha=.5)
    for eval, phi in values:
        plt.plot(eval, phi, c='cyan')
    plt.xlabel(r"Magnetic Flux $\phi$", fontsize=14, color='white')
    plt.ylabel("Energy", fontsize=14, color='white')
    plt.xlim(0, 1)
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    plt.grid(False)
    plt.show()


plot_hofstadter_butterfly(max_q=80)
# %%
