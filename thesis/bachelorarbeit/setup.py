
import numpy as np
from numpy.linalg import inv


class MySystem:
    
    def __init__(self, t, delta_s, delta_p, onsite_energy, energy_value):
        '''Initizalize the parameters: 
                t: Hopping parameter
                delta_s: Self-energy of a swave superconductor
                delta_p: self-energy of a pwave superconductor
                onsite_energy: chemical potential at a site
        
        '''
        self.hopping = t 
        self.swave = delta_s
        self.pwave = delta_p 
        self.epsilon = onsite_energy
        self.energy_value = energy_value

    def get_hamiltonian(self):
        '''Create a function to create and return two 2x2 arrays, with the same parameters as __init__.
        H: BdG-Hamiltonian
        V: Hopping and pwave pairing potential
        '''
        H = np.array([[self.epsilon, self.swave], [self.swave, -self.epsilon]])
        V = np.array([[-self.hopping, self.pwave], [-self.pwave, self.hopping]])
        return H, V
    
    def get_green_functions(self):
        z = (self.energy_value - 1e-4j) * np.eye(2)   
        alpha = self.hopping 
        beta =  np.conjugate(alpha)
        Epsilon_surf = self.epsilon  
        Epsilon_bulk = self.epsilon 
        
        for i in range(100):
            aux = inv((z - Epsilon_bulk))
            Epsilon_surf = Epsilon_surf + alpha @ aux @ beta 
            Epsilon_bulk = Epsilon_bulk + alpha @ aux @ beta + beta @ aux @ alpha
            alpha = alpha @ aux @ alpha 
            beta = beta @ aux @ beta

        Gs = inv((z - Epsilon_surf))
        Gb = inv((z - Epsilon_bulk))

        return Gs, Gb
