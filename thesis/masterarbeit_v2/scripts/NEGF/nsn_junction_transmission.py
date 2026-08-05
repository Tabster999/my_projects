#%% --- IMPORTS --- 
import numpy as np
from numpy.linalg import inv
from scipy.linalg import norm
import matplotlib.pyplot as plt

"""
WORKFLOW:
[ Lead ]  ──>  [ Junction ]  ──>  [ TransmissionSpectrum ]  ──>  [ TransportVisualizer ]
 (Leads &       (Solves Dyson       (Applies Voltage/Bias,        (Plots 1x2 & 2x2
 Self-Energy)    & $T(E)$ curves)    Fermi Distributions & $dI/dV$) Figures)

useful for dicts:   print(dict.keys())
                    print(vars(dict))
"""
#%% --- HELPER FUNCTIONS ---
def tau_z(): return np.array([[1, 0], [0, -1]], dtype=complex)
def tau_x(): return np.array([[0, 1], [1, 0]], dtype=complex)

def site_slice(site_index, dim_per_site):
    start = site_index * dim_per_site
    return slice(start, start + dim_per_site)

def build_lattice_hamiltonian(onsite_blocks, bonds):
    dim = onsite_blocks[0].shape[0]
    n_sites = len(onsite_blocks)
    H = np.zeros((n_sites * dim, n_sites * dim), dtype=complex)
    for i, block in enumerate(onsite_blocks):
        H[site_slice(i, dim), site_slice(i, dim)] = block
    for site_a, site_b, hop in bonds:
        H[site_slice(site_a, dim), site_slice(site_b, dim)] = hop
        H[site_slice(site_b, dim), site_slice(site_a, dim)] = hop.conj().T
    return H

def embed_lead_coupling(local_couplings, n_sites, dim_per_site):
    dim_lead = next(iter(local_couplings.values())).shape[1]
    V = np.zeros((n_sites * dim_per_site, dim_lead), dtype=complex)
    for site_index, block in local_couplings.items():
        V[site_slice(site_index, dim_per_site), :] = block
    return V


#%% --- CLASSES --- 
class Lead:
    """Manages semi-infinite lead surface Green's functions via Lopez-Sancho decimation."""
    def __init__(self, name, H_onsite, V_hop, V_coupling, eta=1e-6):
        self.name = name
        self.H_onsite = np.asarray(H_onsite, dtype=complex)
        self.V_hop = np.asarray(V_hop, dtype=complex)
        self.V_coupling = np.asarray(V_coupling, dtype=complex)
        self.eta = eta

    def surface_gf(self, E, max_iter=400, tol=1e-14):
        dim = self.H_onsite.shape[0]
        z = (E + 1j * self.eta) * np.eye(dim, dtype=complex)
        alpha, beta = self.V_hop.copy(), self.V_hop.conj().T.copy()
        eps_s, eps_b = self.H_onsite.copy(), self.H_onsite.copy()
        for _ in range(max_iter):
            g = inv(z - eps_b)
            alpha_g, beta_g = alpha @ g, beta @ g
            eps_s += alpha_g @ beta
            eps_b += alpha_g @ beta + beta_g @ alpha
            alpha, beta = alpha_g @ alpha, beta_g @ beta
            if norm(alpha, np.inf) < tol and norm(beta, np.inf) < tol:
                break
        return inv(z - eps_s)

    def self_energy(self, E):
        return self.V_coupling @ self.surface_gf(E) @ self.V_coupling.conj().T


class Junction:
    """Represents the Central Scattering Region connected to arbitrary leads."""
    def __init__(self, H_C, leads, site_of):
        self.H_C = np.asarray(H_C, dtype=complex)
        self.leads = list(leads)
        self.site_of = site_of
        self.tau_idx = {"e": 0, "h": 1}

    def solve(self, E, eta=1e-6):
        dim = self.H_C.shape[0]
        sigmas = {lead.name: lead.self_energy(E) for lead in self.leads}
        sigma_tot = sum(sigmas.values())
        GR = inv((E + 1j * eta) * np.eye(dim, dtype=complex) - self.H_C - sigma_tot)
        GA = GR.conj().T
        gammas = {name: -2.0 * np.imag(s) for name, s in sigmas.items()}
        return {"GR": GR, "GA": GA, "gammas": gammas}

    def _channel_trans(self, state, l1, t1, l2, t2):
        idx1 = 2 * self.site_of[l1] + self.tau_idx[t1]
        idx2 = 2 * self.site_of[l2] + self.tau_idx[t2]
        G1 = state['gammas'][l1][idx1, idx1].real
        G2 = state['gammas'][l2][idx2, idx2].real
        return G1 * G2 * (state['GR'][idx1, idx2] * state['GA'][idx2, idx1]).real

    def compute_spectrum(self, E_sweep, eta=1e-5):
        """Precomputes energy-dependent transmissions for all sub-channels."""
        ch_names = ["ee", "eh_cross", "he_cross", "hh", "eh_local", "he_local"]
        data = {l: {k: np.zeros(len(E_sweep)) for k in ch_names} for l in ["left", "right"]}
        defs = {
            "left":  {"ee":("left","e","right","e"), "eh_cross":("left","e","right","h"), 
                      "he_cross":("left","h","right","e"), "hh":("left","h","right","h"), 
                      "eh_local":("left","e","left","h"), "he_local":("left","h","left","e")},
            "right": {"ee":("right","e","left","e"), "eh_cross":("right","e","left","h"), 
                      "he_cross":("right","h","left","e"), "hh":("right","h","left","h"), 
                      "eh_local":("right","e","right","h"), "he_local":("right","h","right","e")}
        }
        for idx, E in enumerate(E_sweep):
            state = self.solve(E, eta=eta)
            for lead in ["left", "right"]:
                for ch, (l1, t1, l2, t2) in defs[lead].items():
                    data[lead][ch][idx] = self._channel_trans(state, l1, t1, l2, t2)
        return TransmissionSpectrum(E_sweep, data)


# TRANSPORT CALCULATOR CLASS
class TransmissionSpectrum:
    """Encapsulates energy integration and voltage sweeps over transmission spectra."""
    def __init__(self, E_sweep, data):
        self.E_sweep = E_sweep
        self.data = data

    @staticmethod
    def _f_e(E, mu, kT):
        if kT == 0: return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
        return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -700, 700)))

    @staticmethod
    def _f_h(E, mu, kT):
        if kT == 0: return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
        return 1.0 / (1.0 + np.exp(np.clip((E + mu) / kT, -700, 700)))

    def calculate_current(self, bias, kT, out_name="left", in_name="right"):
        """Computes channel-resolved DC currents for a specific bias point."""
        ch = self.data[out_name]
        f_out_e, f_out_h = self._f_e(self.E_sweep, bias[out_name], kT), self._f_h(self.E_sweep, bias[out_name], kT)
        f_in_e,  f_in_h  = self._f_e(self.E_sweep, bias[in_name], kT),  self._f_h(self.E_sweep, bias[in_name], kT)

        i_ec  = np.trapezoid(ch['ee']*(f_out_e - f_in_e) - ch['hh']*(f_out_h - f_in_h), self.E_sweep)
        i_car = np.trapezoid(ch['eh_cross']*(f_out_e - f_in_h) - ch['he_cross']*(f_out_h - f_in_e), self.E_sweep)
        i_lar = np.trapezoid((ch['eh_local'] + ch['he_local'])*(f_out_e - f_out_h), self.E_sweep)
        return {'EC': i_ec, 'CAR': i_car, 'LAR': i_lar, 'total': i_ec + i_car + i_lar}

    def calculate_matrix(self, bias0, kT, leads=['left', 'right'], dV=1e-5):
        """Computes partial derivative matrix elements G_ij = dI_i / dV_j."""
        G = {}
        for i in leads:
            in_name = 'right' if i == 'left' else 'left'
            for j in leads:
                b_plus, b_minus = bias0.copy(), bias0.copy()
                b_plus[j] += dV; b_minus[j] -= dV
                r_plus  = self.calculate_current(b_plus, kT, i, in_name)
                r_minus = self.calculate_current(b_minus, kT, i, in_name)
                G[(i, j)] = (r_plus['total'] - r_minus['total']) / (2 * dV)
        return G

    def sweep(self, V_range, kT, dV=1e-4):
        """Runs symmetric & antisymmetric sweeps for both currents and conductances."""
        res = {
            'I':  {'sym': {k: [] for k in ['EC','CAR','LAR','total']}, 'anti': {k: [] for k in ['EC','CAR','LAR','total']}},
            'dI': {'sym': {k: [] for k in ['EC','CAR','LAR','total']}, 'anti': {k: [] for k in ['EC','CAR','LAR','total']}},
            'G':  {'sym': {'G_LL': [], 'G_LR': []}, 'anti': {'G_LL': [], 'G_LR': []}}
        }
        for Vb in V_range:
            for scheme in ['sym', 'anti']:
                sign = +1.0 if scheme == 'sym' else -1.0
                bias0   = {"left": +Vb / 2,         "right": sign * Vb / 2}
                b_plus  = {"left": +(Vb + dV) / 2,  "right": sign * (Vb + dV) / 2}
                b_minus = {"left": +(Vb - dV) / 2,  "right": sign * (Vb - dV) / 2}

                # Currents
                curr = self.calculate_current(bias0, kT)
                for k in curr: res['I'][scheme][k].append(curr[k])

                # Path derivatives dI / dV_bias
                r_p = self.calculate_current(b_plus, kT)
                r_m = self.calculate_current(b_minus, kT)
                for k in r_p: res['dI'][scheme][k].append((r_p[k] - r_m[k]) / (2 * dV))

                # Matrix elements dI_i / dV_j
                G_mat = self.calculate_matrix(bias0, kT, dV=1e-5)
                res['G'][scheme]['G_LL'].append(G_mat[('left', 'left')])
                res['G'][scheme]['G_LR'].append(G_mat[('left', 'right')])
        return res


# PLOTTING VISUALIZER CLASS
class TransportVisualizer:
    """Handles multi-panel figure creation for transport results."""
    @staticmethod
    def plot_currents(V_range, sweep_res):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        titles = {'sym': r'Symmetric bias ($\mu_L=\mu_R=+V/2$)', 'anti': r'Antisymmetric bias ($\mu_L=+V/2, \mu_R=-V/2$)'}
        for ax, scheme in zip(axes, ['sym', 'anti']):
            for key, label in [('EC', 'EC (normal)'), ('CAR', 'CAR (crossed Andreev)'), 
                               ('LAR', 'LAR (local Andreev)'), ('total', 'Total')]:
                style = 'k--' if key == 'total' else '-'
                ax.plot(V_range, sweep_res['I'][scheme][key], style, linewidth=2, label=label)
            ax.set_xlabel(r'$V_{\rm bias}$')
            ax.set_title(titles[scheme])
            ax.grid(alpha=0.3); ax.legend()
        axes[0].set_ylabel(r'DC current $I_{\rm left}$ (a.u.)')
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_conductance_matrix(V_range, sweep_res, delta):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        schemes = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V/2$)'),
                   ('anti', r'Antisymmetric bias ($\mu_L = +V/2, \mu_R = -V/2$)')]
        
        for col, (scheme, title) in enumerate(schemes):
            # Top row: Path derivatives
            ax_top = axes[0, col]
            for key in ['EC', 'CAR', 'LAR', 'total']:
                style = 'k--' if key == 'total' else '-'
                ax_top.plot(V_range, sweep_res['dI'][scheme][key], style, linewidth=2, label=f'$G_{{{key}}}$')
            ax_top.axvline(x=2 * delta, color='gray', linestyle=':', label=r'$V = 2\Delta$')
            ax_top.set_title(title); ax_top.grid(alpha=0.3); ax_top.legend(loc='upper right')

            # Bottom row: Partial derivative matrix elements
            ax_bot = axes[1, col]
            ax_bot.plot(V_range, sweep_res['G'][scheme]['G_LL'], linewidth=2, color='tab:blue', label=r'$G_{LL} = \partial I_L / \partial V_L$')
            ax_bot.plot(V_range, sweep_res['G'][scheme]['G_LR'], linewidth=2, color='tab:red', label=r'$G_{LR} = \partial I_L / \partial V_R$')
            ax_bot.axvline(x=2 * delta, color='gray', linestyle=':', label=r'$V = 2\Delta$')
            ax_bot.set_xlabel(r'$V_{\rm bias}$'); ax_bot.grid(alpha=0.3); ax_bot.legend(loc='upper right')

            if col == 0:
                ax_top.set_ylabel(r'$dI_{\rm left} / dV_{\rm bias}$ (a.u.)')
                ax_bot.set_ylabel(r'Partial Conductance ($e^2/h$)')
        plt.tight_layout(); plt.show()


#%% --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parameters
    t_n, t_s, mu_n, mu_s = 1.0, 1.0, 2.0, 2.0
    delta, eta, tc, n_sites = 0.1, 1e-5, 0.8, 5
    kT = 1e-2 * delta

    # Build System
    H_N, V_N = (2 * t_n - mu_n) * tau_z(), -t_n * tau_z()
    H_S = (2 * t_s - mu_s) * tau_z() + delta * tau_x()
    H_C = build_lattice_hamiltonian([H_S] * n_sites, [(i, i + 1, -t_s * tau_z()) for i in range(n_sites - 1)])

    lead_left = Lead("left", H_N, V_N, embed_lead_coupling({0: -tc * tau_z()}, n_sites, 2), eta=eta)
    lead_right = Lead("right", H_N, V_N, embed_lead_coupling({n_sites - 1: -tc * tau_z()}, n_sites, 2), eta=eta)
    junction = Junction(H_C, [lead_left, lead_right], site_of={"left": 0, "right": n_sites - 1})

    # 1. Compute Quantum Mechanics (Energy Transmission Spectrum)
    E_sweep = np.linspace(-3.0, 3.0, 5001)
    spectrum = junction.compute_spectrum(E_sweep, eta=eta)

    # 2. Compute Voltage Sweeps
    V_range = np.linspace(-5.0, 5.0, 201)
    sweep_results = spectrum.sweep(V_range, kT=kT)

    # 3. Plot Results
    TransportVisualizer.plot_currents(V_range, sweep_results)
    TransportVisualizer.plot_conductance_matrix(V_range, sweep_results, delta=delta)
# %%
