#%% --- IMPORTS --- 
"""
General-purpose NEGF transport framework.

Generalizes the single-site N-S BTK script into:
  - Lead:            arbitrary internal dimension, Sancho-Rubio surface GF,
                      self-energy, and broadening matrix Gamma_l = i(Sigma_r - Sigma_a).
  - CentralRegion:    arbitrary H_C (any number of sites / internal degrees of
                      freedom), coupled to an arbitrary number of Lead objects.
  - transmission():   generic Tr{ P_out Gamma_out P_out G^r P_in Gamma_in P_in G^a },
                      with P_out/P_in optional projectors (e.g. onto electron/hole
                      Nambu subspace) to get channel-resolved transmission
                      T_{lead_out, lead_in}^{i,alpha} exactly as derived by hand.

No part of this file assumes 2 leads, a single site, or a 2x2 Nambu structure --
dimensions are inferred from the matrices you pass in.
"""

import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt 

#%%  --- HELPERS --- 

def tau_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def tau_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def nambu_projectors(dim_per_site=1):
    """Electron/hole projectors for a chain of `dim_per_site`-orbital Nambu sites.
    For the simplest case (1 orbital per site, standard 2x2 Nambu block)"""
    n = 2 * dim_per_site
    P_e = np.zeros((n, n), dtype=complex)
    P_h = np.zeros((n, n), dtype=complex)
    for s in range(dim_per_site):
        P_e[2 * s, 2 * s] = 1.0
        P_h[2 * s + 1, 2 * s + 1] = 1.0
    return P_e, P_h


#%% --- DEFINE LEAD AND CENTRAL CLASSES --- 

class Lead:
    """
    A semi-infinite periodic lead, described in its own unit-cell basis by:
        H_onsite : on-site block (n_lead x n_lead)
        V_hop    : hopping between successive unit cells (n_lead x n_lead)
        V_coupling : coupling FROM the central region TO this lead's first
                     unit cell, i.e. V_{C,lead} (n_central x n_lead)

    n_lead need not equal n_central -- e.g. a lead with more internal
    channels than the central site, or a lead written in a different basis.
    """

    def __init__(self, name, H_onsite, V_hop, V_coupling, eta=1e-6):
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_hop, dtype=complex)
        self.V_coupling = np.asarray(V_coupling, dtype=complex)  # V_{C,lead}
        self.eta = eta

    def surface_gf(self, E, max_iter=400, tol=1e-14):
        """Sancho-Rubio decimation for the semi-infinite lead's surface Green's function."""
        dim = self.H_onsite.shape[0]
        z = (E + 1j * self.eta) * np.eye(dim, dtype=complex)

        alpha = self.V_hop.copy()
        beta = self.V_hop.conj().T.copy()
        eps_s = self.H_onsite.copy()
        eps_b = self.H_onsite.copy()

        for _ in range(max_iter):
            g = inv(z - eps_b)
            alpha_g = alpha @ g
            beta_g = beta @ g

            eps_s = eps_s + alpha_g @ beta
            eps_b = eps_b + alpha_g @ beta + beta_g @ alpha

            alpha = alpha_g @ alpha
            beta = beta_g @ beta

            if norm(alpha, np.inf) < tol and norm(beta, np.inf) < tol:
                break

        return inv(z - eps_s)

    def self_energy(self, E):
        """Sigma_l^r = V_{C,l} g_l^r V_{l,C}, mapped into the central-region space."""
        g_surf = self.surface_gf(E)
        V_Cl = self.V_coupling
        V_lC = self.V_coupling.conj().T
        return V_Cl @ g_surf @ V_lC

    def gamma(self, E):
        """Gamma_l(E) = i[Sigma_l^r - Sigma_l^a] = -2 Im(Sigma_l^r), in central-region space."""
        sigma_r = self.self_energy(E)
        return -2.0 * np.imag(sigma_r)


# CentralRegion: arbitrary H_C coupled to an arbitrary list of Leads

class CentralRegion:
    def __init__(self, H_C, leads):
        self.H_C = np.asarray(H_C, dtype=complex)
        self.leads = list(leads)  # list[Lead]
        self._by_name = {lead.name: lead for lead in self.leads}

    def lead(self, name):
        return self._by_name[name]

    def green_functions(self, E, eta=1e-6):
        """Retarded/advanced central-region Green's functions with ALL leads'
        self-energies summed in, plus the individual self-energies (useful for Gamma_l)."""
        dim = self.H_C.shape[0]
        sigmas = {lead.name: lead.self_energy(E) for lead in self.leads}
        sigma_tot = sum(sigmas.values())
        GR = inv((E + 1j * eta) * np.eye(dim, dtype=complex) - self.H_C - sigma_tot)
        GA = GR.conj().T
        return GR, GA, sigmas

    def transmission(self, E, lead_out, lead_in, P_out=None, P_in=None, eta=1e-6):
        """
        Generic transmission:
            T = Tr{ P_out Gamma_out P_out  G^r  P_in Gamma_in P_in  G^a }

        lead_out, lead_in : Lead objects (can be the same lead)
        P_out, P_in        : optional projectors (e.g. nambu_projectors()[0/1])
                              to resolve electron/hole (or any other) channel.
                              Defaults to identity (total transmission).
        """
        GR, GA, _ = self.green_functions(E, eta=eta)
        dim = self.H_C.shape[0]
        if P_out is None:
            P_out = np.eye(dim, dtype=complex)
        if P_in is None:
            P_in = np.eye(dim, dtype=complex)

        Gamma_out = lead_out.gamma(E)
        Gamma_in = lead_in.gamma(E)

        M = P_out @ Gamma_out @ P_out @ GR @ P_in @ Gamma_in @ P_in @ GA
        return float(np.real(np.trace(M)))

#%% --- MAIN EXECUTION --- 

if __name__ == "__main__":
    t_n, t_s = 1.0, 1.0
    mu_n, mu_s = 2.0, 2.0
    delta = 0.1
    eta = 1e-5
    t_couplings = [1.0, 0.4]
    colors = ['blue', 'orange']
    plt.figure(figsize=(10, 7.5))
    for i, tc in enumerate(t_couplings):

        H_N = (2 * t_n - mu_n) * tau_z()
        V_N = -t_n * tau_z()
        H_S = (2 * t_s - mu_s) * tau_z() + delta * tau_x()
        V_S = -t_s * tau_z()
        V_NS = -tc * tau_z()

        lead_N = Lead("N", H_N, V_N, V_N, eta=eta)          # coupling to itself: it's the outer N chain
        lead_S = Lead("S", H_S, V_S, V_NS, eta=eta)         # coupling to central site via V_NS

        central = CentralRegion(H_N, [lead_N, lead_S])

        P_e, P_h = nambu_projectors(dim_per_site=1)

        G = []
        E_array = np.linspace(-3, 3, 301)
        for e in E_array:
            T_he = central.transmission(e, lead_N, lead_N, P_out=P_h, P_in=P_e)
            T_eh = central.transmission(e, lead_N, lead_N, P_out=P_e, P_in=P_h)
            T_eS = central.transmission(e, lead_S, lead_N, P_out=None, P_in=P_e)

            R_ee = 1.0 - T_he - T_eS
            G.append(1.0 - R_ee + T_he)
        plt.plot(E_array, G, label=fr'$t_c ={tc:.1f}$', c=colors[i], alpha = .7)
    plt.xlabel('Energy E')
    plt.ylabel(r'$G / G_0$')
    plt.title('Conductance G vs. particle energy E of N-S interface')
    plt.legend()
    plt.grid(alpha = .3)
    plt.show()
# %%
