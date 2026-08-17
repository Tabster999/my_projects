"""
four_terminal_junction.py
==========================

4-terminal Josephson junction driver script, built entirely on top of
modules/{matrices,solvers,transport,helpers}.py.

Geometry
--------
    top:     semi-infinite SC lead (phase phi_top), attached along the
             WHOLE junction length via a self-energy embedded at the top
             edge of every x-slice
    bottom:  semi-infinite SC lead (phase phi_bottom = 0, gauge-fixed),
             same idea at the bottom edge
    left/right: semi-infinite NORMAL METAL leads, attached only at the
             two ends of the x-chain (x=0 and x=nx-1), carrying the
             measured current

The SC leads induce the proximity gap / Andreev bound states in the
normal region (as in the original Scharf-Pientka LDOS(E,phi) script);
the metal leads let you additionally compute a genuine transport
current (Landauer transmission) through that region, modulated by the
Josephson phase phi = phi_top - phi_bottom.

Everything physics-specific (Sancho surface GFs, phase gauge, RGF sweep,
self-energy embedding, transmission/current formulas) lives in the
modules; this script only wires them together for this geometry.
"""

#%% IMPORTS
import os, sys
import multiprocessing
from pathlib import Path
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
os.environ['MKL_NUM_THREADS']      = slurm_cpus
os.environ['OMP_NUM_THREADS']      = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
current_dir = Path(__file__).resolve().parent
p_dir = current_dir.parent
parent_dir = p_dir.parent
sys.path.insert(0, str(parent_dir))

import modules as myf
from modules import matrices, solvers, transport, helpers

#%% PHYSICAL CONSTANTS (fixed, never touch)
_hbar = 1.05e-34   # J.s
_e    = 1.602e-19  # C
_m0   = 9.1e-31    # kg
_muB  = 5.78e-2    # meV / T


#%% PARAMETERS
class Params:
    """
    All physical parameters in one place, t-unit conventions matching the
    original LDOS(E,phi) script (Params.tunits sets the meV <-> t-unit scale).
    """
    def __init__(self):
        # --- lattice & effective mass ---
        self.a     = 20.0     # nm
        self.m_eff = 0.038    # m_eff / m0

        # --- superconductor (top & bottom leads) ---
        self.Delta = 0.25     # meV
        self.mu_S  = 1.0      # meV

        # --- normal region ---
        self.mu_N  = 0.7      # meV
        self.alpha = 14.3     # meV.nm, Rashba SOC
        self.beta  = 7.3      # meV.nm, Dresselhaus SOC

        # --- Zeeman (in normal region only; leads are unpolarized) ---
        self.g       = 10
        self.B       = 0.0  # T
        self.theta_z = 0.35 * np.pi
        self.E_z     = 0.0    # meV

        # --- metal leads (left/right, current-carrying) ---
        self.mu_lead = 0.7    # meV; defaults to mu_N (pure continuation)

        # --- junction geometry ---
        self.ny = 12          # transverse sites between top/bottom SC leads
        self.nx = 8            # normal-region x-slices between metal leads

        # --- numerics ---
        self.eta_rel = 0.025
        self.it      = 450
        self.tol     = 1e-15

        # --- scan ranges ---
        self.nphi = 21
        self.nw   = 41
        self.w_range = 1.0     # scan +/- w_range * Delta

        # --- transport ---
        self.kT_lead   = 0.02  # meV, lead temperature for Fermi smearing
        self.mu_bias   = 0.05  # meV, symmetric +/- bias between L/R metal leads for dI/dV-style current

    @property
    def a_m(self): return self.a * 1e-9

    @property
    def tunits(self):
        return (_hbar**2 / (2 * self.m_eff * _m0 * self.a_m**2)) * (1e3 / _e)

    @property
    def t(self): return 1.0

    @property
    def Delta_t(self):  return self.Delta / self.tunits
    @property
    def mu_S_t(self):   return self.mu_S / self.tunits
    @property
    def mu_N_t(self):   return self.mu_N / self.tunits
    @property
    def mu_lead_t(self): return self.mu_lead / self.tunits
    @property
    def alpha_t(self):  return self.alpha / (self.a * self.tunits)
    @property
    def beta_t(self):   return self.beta / (self.a * self.tunits)
    @property
    def E_par_meV(self): return self.B * _muB * self.g
    @property
    def E_par_t(self):  return self.E_par_meV / self.tunits
    @property
    def E_z_t(self):    return self.E_z / self.tunits
    @property
    def eta(self):      return self.eta_rel * self.Delta_t
    @property
    def kT_lead_t(self): return self.kT_lead / self.tunits
    @property
    def mu_bias_t(self): return self.mu_bias / self.tunits

    @property
    def phi_vals(self):
        return np.linspace(0, 2 * np.pi, self.nphi)

    @property
    def energies(self):
        return np.linspace(-self.w_range * self.Delta_t, self.w_range * self.Delta_t, self.nw)

    def summary(self):
        print(f"{'-'*60}")
        print(f"  tunits   = {self.tunits:.3f} meV")
        print(f"  Delta    = {self.Delta:.3f} meV = {self.Delta_t:.4f} t")
        print(f"  mu_S     = {self.mu_S:.3f} meV = {self.mu_S_t:.4f} t")
        print(f"  mu_N     = {self.mu_N:.3f} meV = {self.mu_N_t:.4f} t")
        print(f"  mu_lead  = {self.mu_lead:.3f} meV = {self.mu_lead_t:.4f} t")
        print(f"  E_par    = {self.E_par_meV:.3f} meV = {self.E_par_t:.4f} t")
        print(f"  ny={self.ny}, nx={self.nx}, nphi={self.nphi}, nw={self.nw}")
        print(f"{'-'*60}")


#%% LEAD BUILDING BLOCKS

def get_sc_edge_lead(energy: float, p: Params):
    """
    Bare (phi=0) surface Green's function for the top/bottom SC edge lead,
    plus the coupling matrix used to attach it to a normal slice's edge site.

    The top and bottom leads are made of the same SC material, so (in the
    absence of any y-direction asymmetry in the model) they share this
    same bare surface GF -- only the applied phase differs.

    Returns:
        g_sc   : (4, 4) bare surface GF (phi=0)
        V_sc   : (4, 4) coupling matrix (= hopping_y), same for both edges
    """
    H_sc = matrices.onsite_matrix_sc(p.t, p.mu_S_t, p.Delta_t, p.alpha_t, p.beta_t, twod=True)
    V_sc = matrices.hopping_y(p.alpha_t, p.beta_t, p.t)
    g_sc, _ = solvers.get_surface_gf(energy, H_sc, V_sc, eta=p.eta, max_iter=p.it, tol=p.tol)
    return g_sc, V_sc


def get_metal_lead_gfs(energy: float, p: Params):
    """
    Surface Green's functions for the left/right metal leads (x-direction),
    modeled as an unpolarized continuation of the normal-region lattice at
    chemical potential mu_lead.

    Convention: t_matrix=V0 gives the LEFT lead, t_matrix=Vd gives the
    RIGHT lead -- this was verified numerically against the original
    script's sancho() calls for this same (SP-basis) x-hopping structure;
    it is NOT necessarily the same assignment used elsewhere for the
    plain-Nambu-basis matrices.t_matrix_x_plain (see matrices.py's basis note).

    Returns:
        gL, gR : (4*ny, 4*ny) left/right metal lead surface GFs
        V0     : (4*ny, 4*ny) x-hopping matrix (also used as the
                 lead<->normal-region coupling)
    """
    H_lead = matrices.make_slice_normal(
        p.ny, p.t, p.mu_lead_t, E_par=0.0, theta_z=0.0, E_z=0.0,
        alpha_t=p.alpha_t, beta_t=p.beta_t,
    )
    V0 = matrices.make_x_hopping(p.ny, p.alpha_t, p.beta_t, p.t)
    Vd = V0.conj().T

    gL, _ = solvers.get_surface_gf(energy, H_lead, V0, eta=p.eta, max_iter=p.it, tol=p.tol)  # LEFT
    gR, _ = solvers.get_surface_gf(energy, H_lead, Vd, eta=p.eta, max_iter=p.it, tol=p.tol)  # RIGHT
    return gL, gR, V0


#%% CORE PIPELINE

def build_dressed_normal_slices(energy: float, phi: float, p: Params, symmetric: bool = False):
    """
    Build the nx normal-region slices, each dressed with the top/bottom SC
    lead self-energies at the given energy and Josephson phase phi.

    Gauge (matches build_sns_junction_sliced convention elsewhere in the
    modules): symmetric=False keeps the bottom lead at phi=0 and puts the
    full phi on the top lead; symmetric=True splits it +-phi/2.

    Returns:
        H_slices : list of nx dressed (4*ny, 4*ny) Hamiltonians
    """
    H_bare = matrices.make_slice_normal(
        p.ny, p.t, p.mu_N_t, p.E_par_t, p.theta_z, p.E_z_t, p.alpha_t, p.beta_t
    )
    H_slices = [H_bare.copy() for _ in range(p.nx)]

    g_sc, V_sc = get_sc_edge_lead(energy, p)

    if symmetric:
        g_top = solvers.apply_phase_gauge(g_sc, phi / 2, N_y=1)
        g_bot = solvers.apply_phase_gauge(g_sc, -phi / 2, N_y=1)
    else:
        g_top = solvers.apply_phase_gauge(g_sc, phi, N_y=1)
        g_bot = g_sc

    Sigma_top = transport.self_energy(V_sc, g_top)
    Sigma_bot = transport.self_energy(V_sc, g_bot)

    top_idx = helpers.get_edge_indices(p.ny, n_edge=1, dof=4, side='top')
    bot_idx = helpers.get_edge_indices(p.ny, n_edge=1, dof=4, side='bottom')

    H_slices = transport.build_dressed_slices(H_slices, Sigma_top, top_idx, Sigma_bot, bot_idx)
    return H_slices


def solve_junction(energy: float, phi: float, p: Params, return_full: bool = True):
    """
    Full solve at one (energy, phi): dress the normal slices with the SC
    edge self-energies, attach the metal leads, and run the x-direction RGF.

    Returns:
        G_diag   : (nx, 4*ny, 4*ny) diagonal-block Green's functions
        G_blocks : (nx, nx, 4*ny, 4*ny) full blocks (only if return_full)
        Gamma_L, Gamma_R : (4*ny, 4*ny) metal lead broadening matrices
                           (needed for the transmission formula)
    """
    H_slices = build_dressed_normal_slices(energy, phi, p)
    gL, gR, V0 = get_metal_lead_gfs(energy, p)

    G_diag, GL, GR, G_blocks, _ = solvers.get_rgf_sns(
        H_slices, V0, gL, gR, energy, eta=p.eta, return_full=return_full
    )

    Sigma_L = transport.self_energy(V0.conj().T, gL)
    Sigma_R = transport.self_energy(V0, gR)
    Gamma_L = transport.broadening_matrix(Sigma_L)
    Gamma_R = transport.broadening_matrix(Sigma_R)

    return G_diag, G_blocks, Gamma_L, Gamma_R


#%% DRIVERS

def compute_ldos_E_phi(p: Params, symmetric: bool = False):
    """
    Total LDOS(E, phi) summed over all sites and all x-slices.
    Returns array of shape (nw, nphi).
    """
    result = np.zeros((p.nw, p.nphi))
    idx_all = np.arange(4 * p.ny)
    for iw, w in enumerate(tqdm(p.energies, desc='LDOS(E,phi)')):
        for iphi, phi in enumerate(p.phi_vals):
            H_slices = build_dressed_normal_slices(w, phi, p, symmetric=symmetric)
            gL, gR, V0 = get_metal_lead_gfs(w, p)
            G_diag, *_ = solvers.get_rgf_sns(H_slices, V0, gL, gR, w, eta=p.eta, return_full=False)
            result[iw, iphi] = sum(helpers.ldos_trace(G_diag[i], idx_all) for i in range(p.nx))
    return result


def compute_transmission_E_phi(p: Params, symmetric: bool = False):
    """
    Metal-lead transmission T(E, phi) through the junction.
    Returns array of shape (nw, nphi).
    """
    result = np.zeros((p.nw, p.nphi))
    for iw, w in enumerate(tqdm(p.energies, desc='T(E,phi)')):
        for iphi, phi in enumerate(p.phi_vals):
            _, G_blocks, Gamma_L, Gamma_R = solve_junction(w, phi, p, return_full=True)
            G_LR = G_blocks[0, p.nx - 1]
            result[iw, iphi] = transport.transmission(G_LR, Gamma_L, Gamma_R)
    return result


def compute_current_phase(p: Params, T_of_E_phi: np.ndarray):
    """
    Integrate T(E, phi) against the metal leads' Fermi functions (biased
    symmetrically by +-mu_bias_t/2) to get the current-phase relation
    I(phi) of the metal-probe current.

    Args:
        T_of_E_phi : (nw, nphi) transmission map, e.g. from compute_transmission_E_phi

    Returns:
        (nphi,) array I(phi)
    """
    mu_L =  p.mu_bias_t / 2
    mu_R = -p.mu_bias_t / 2
    I_phi = np.zeros(p.nphi)
    for iphi in range(p.nphi):
        I_phi[iphi] = transport.landauer_current(
            p.energies, T_of_E_phi[:, iphi], mu_L, mu_R,
            kT_p=p.kT_lead_t, kT_q=p.kT_lead_t,
        )
    return I_phi


def compute_thermoelectric_vs_phi(p: Params):
    """
    Electrical conductance G, Seebeck coefficient S, thermal conductance K,
    and the Wiedemann-Franz Lorenz ratio as a function of the Josephson
    phase phi, following Klees et al. (arXiv:2306.17845) Eqs. (1)-(4) and
    the G/Pi/K/S formulas below their Eq. (3) -- generalized here from
    their single-channel quantum-dot Green's functions to our multi-site
    2D-lattice Green's functions via helpers.get_electron_hole_indices +
    transport.resolved_transmission.

    NOTE: like their Eq. (4a/4b), this assumes the bias is applied only to
    the RIGHT metal lead (L stays at the reference mu=0, T) -- see
    transport.total_transmissions_biased_right's docstring.

    Returns:
        dict of (nphi,) arrays: 'G', 'S', 'K', 'lorenz_ratio'
    """
    dim = 4 * p.ny
    electron_idx, hole_idx = helpers.get_electron_hole_indices(dim)

    out = {'G': np.zeros(p.nphi), 'S': np.zeros(p.nphi),
           'K': np.zeros(p.nphi), 'lorenz_ratio': np.zeros(p.nphi)}

    for iphi, phi in enumerate(tqdm(p.phi_vals, desc='G,S,K(phi)')):
        T_ee_RL = np.zeros(p.nw)
        T_eh_RL = np.zeros(p.nw)
        T_eh_RR = np.zeros(p.nw)
        for iw, w in enumerate(p.energies):
            G_diag, G_blocks, Gamma_L, Gamma_R = solve_junction(w, phi, p, return_full=True)
            G_LR = G_blocks[0, p.nx - 1]
            G_RR = G_diag[p.nx - 1]
            T_LR = transport.resolved_transmission(G_LR, Gamma_L, Gamma_R, electron_idx, hole_idx)
            T_RR = transport.resolved_transmission(G_RR, Gamma_R, Gamma_R, electron_idx, hole_idx)
            T_ee_RL[iw] = T_LR['ee']
            T_eh_RL[iw] = T_LR['eh']
            T_eh_RR[iw] = T_RR['eh']

        T11, T12, T21, T22 = transport.total_transmissions_biased_right(T_ee_RL, T_eh_RL, T_eh_RR)
        L = transport.onsager_matrix(p.energies, T11, T12, T21, T22, kT=p.kT_lead_t)
        coeffs = transport.thermoelectric_coefficients(L, p.kT_lead_t)

        out['G'][iphi] = coeffs['G']
        out['S'][iphi] = coeffs['S']
        out['K'][iphi] = coeffs['K']
        out['lorenz_ratio'][iphi] = coeffs['lorenz_ratio']

    return out


#%% PLOTTING

def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos, p: Params):
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.imshow(
        ldos.T, extent=[p.energies[0] / p.Delta_t, p.energies[-1] / p.Delta_t,
                        p.phi_vals[0] / np.pi, p.phi_vals[-1] / np.pi],
        cmap='jet', interpolation='bilinear', aspect='auto', origin='lower',
    )
    fig.colorbar(cf, ax=ax, label='LDOS (a.u.)')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(f'4-terminal LDOS: ny={p.ny}, nx={p.nx}', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_transmission_E_phi(T_map, p: Params):
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.imshow(
        T_map.T, extent=[p.energies[0] / p.Delta_t, p.energies[-1] / p.Delta_t,
                          p.phi_vals[0] / np.pi, p.phi_vals[-1] / np.pi],
        cmap='magma', interpolation='bilinear', aspect='auto', origin='lower',
    )
    fig.colorbar(cf, ax=ax, label='T(E,phi)')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(f'Metal-lead transmission: ny={p.ny}, nx={p.nx}', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_current_phase(I_phi, p: Params):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p.phi_vals / np.pi, I_phi, 'o-', ms=3)
    ax.set_xlabel(r'$\phi\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$I$ (t-units)', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(f'mu_bias={p.mu_bias} meV, kT={p.kT_lead} meV', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_thermoelectric_vs_phi(thermo, p: Params):
    """thermo: dict from compute_thermoelectric_vs_phi. Mirrors Klees et al. Fig. 5 layout."""
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    x = p.phi_vals / np.pi

    axes[0].plot(x, thermo['G'] / (1 / (2 * np.pi)), 'o-', ms=3, color='tab:blue')
    axes[0].set_ylabel(r'$G/G_0$', loc='top', rotation=0, labelpad=20)

    axes[1].plot(x, thermo['S'], 'o-', ms=3, color='tab:orange')
    axes[1].set_ylabel(r'$eS/k_B$', loc='top', rotation=0, labelpad=20)

    axes[2].plot(x, thermo['K'], 'o-', ms=3, color='tab:green')
    axes[2].set_ylabel(r'$K$', loc='top', rotation=0, labelpad=20)

    axes[3].plot(x, thermo['lorenz_ratio'], 'o-', ms=3, color='tab:red')
    axes[3].axhline(1.0, ls=':', color='k', lw=1)
    axes[3].set_ylabel(r'$L/L_0$', loc='top', rotation=0, labelpad=20)
    axes[3].set_xlabel(r'$\phi\,/\,\pi$', loc='right')

    for ax in axes:
        _style(ax)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    plt.tight_layout()
    return fig, axes


#%% MAIN
if __name__ == '__main__':      
    p = Params()
    p.summary()

    ldos = compute_ldos_E_phi(p)
    print(f"LDOS(E,phi) shape {ldos.shape}, max={ldos.max():.3f}")
    fig1, ax1 = plot_ldos_E_phi(ldos, p)

    T_map = compute_transmission_E_phi(p)
    print(f"T(E,phi) shape {T_map.shape}, max={T_map.max():.3f}")
    fig2, ax2 = plot_transmission_E_phi(T_map, p)

    I_phi = compute_current_phase(p, T_map)
    print(f"I(phi) shape {I_phi.shape}, max|I|={np.max(np.abs(I_phi)):.4g}")
    fig3, ax3 = plot_current_phase(I_phi, p)

    thermo = compute_thermoelectric_vs_phi(p)
    print(f"G range: [{thermo['G'].min():.4g}, {thermo['G'].max():.4g}]")
    print(f"S range: [{thermo['S'].min():.4g}, {thermo['S'].max():.4g}]")
    print(f"K range: [{thermo['K'].min():.4g}, {thermo['K'].max():.4g}]")
    fig4, axes4 = plot_thermoelectric_vs_phi(thermo, p)

    plt.show()
    print("Done.")
# %%
