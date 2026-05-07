"""
SNS Junction Solver
===================
A self-contained solver for 1D (and extensible to 2D) SNS junctions.
Initialize with parameters once, then call methods for LDOS, current, etc.

Usage
-----
    from sns_solver import SNSSolver

    solver = SNSSolver(
        t=1.0, mu_sc=0.0025, mu_n=0.01,
        alpha=0.6, B=0.4, Delta=0.1,
        SM=50, SL=200, SR=200,
        symmetric=True, eta=1e-4
    )

    # LDOS vs energy at fixed phi
    ldos = solver.ldos_vs_energy(phi=np.pi, sites=[0, 25, 49])

    # LDOS vs phase at fixed energy
    ldos_phi = solver.ldos_vs_phase(energy=0.0, sites=[0, 25, 49])

    # Josephson current
    Iphi, phases = solver.josephson_current(T=0.001)
"""
#%%
import numpy as np
from scipy.linalg import inv
from dataclasses import dataclass, field
from typing import Optional
import my_functions as myf


# ═════════════════════════════════════════════════════════════════════════════
# Helper: complex frequency
# ═════════════════════════════════════════════════════════════════════════════

def _get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def _get_pairing(G_block):
    return np.abs(G_block[0, 3])



def _get_z(energy, eta, ra='r', matsubara=None):
    if matsubara is not None:
        return matsubara
    return energy + 1j * eta if ra == 'r' else energy - 1j * eta


# ═════════════════════════════════════════════════════════════════════════════
# Parameter container
# ═════════════════════════════════════════════════════════════════════════════
#%%
@dataclass
class SNSParams:
    """
    All physical and numerical parameters for the SNS junction.

    Parameters
    ----------
    t       : hopping amplitude
    mu_sc   : chemical potential in SC leads
    mu_n    : chemical potential in N region
    alpha   : Rashba spin-orbit coupling
    B       : Zeeman field
    Delta   : SC pairing amplitude
    SM      : number of sites in N region
    SL      : number of sites in left SC lead
    SR      : number of sites in right SC lead
    symmetric : True  → phase ±φ/2 on left/right leads
                False → full φ on right lead only
    eta     : imaginary broadening for retarded GF
    dof     : degrees of freedom per site (4 for BdG Nambu basis)
    """
    t        : float = 1.0
    mu_sc    : float = 0.0025
    mu_n     : float = 0.01
    alpha    : float = 0.6
    B        : float = 0.4
    Delta    : float = 0.1
    SM       : int   = 50
    SL       : int   = 200
    SR       : int   = 200
    symmetric: bool  = True
    eta      : float = 1e-4
    dof      : int   = 4
    target_E : float = 0.0
    def __post_init__(self):
        self.topo_threshold = np.sqrt(self.Delta**2 + self.mu_sc**2)
        self.in_topo        = self.B > self.topo_threshold

    def summary(self):
        print(f"\n{'═'*50}")
        print(f"  SNS Junction Parameters")
        print(f"{'─'*50}")
        print(f"  t={self.t}, μ_sc={self.mu_sc}, μ_n={self.mu_n}")
        print(f"  α={self.alpha}, B={self.B}, Δ={self.Delta}")
        print(f"  Sites: SL={self.SL}, SM={self.SM}, SR={self.SR}")
        print(f"  Gauge: {'symmetric ±φ/2' if self.symmetric else 'asymmetric full φ on right'}")
        print(f"  η={self.eta}")
        print(f"  Topological: {'YES ✓' if self.in_topo else 'NO ✗'}"
              f"  (B={self.B:.3f} vs √(Δ²+μ²)={self.topo_threshold:.4f})")
        print(f"{'═'*50}\n")

#%%
# ═════════════════════════════════════════════════════════════════════════════
# Main solver
# ═════════════════════════════════════════════════════════════════════════════

class SNSSolver:
    """
    Solver for SNS junction Green's functions and observables.

    All methods accept optional `method` argument:
        'inf'  → Sancho-López semi-infinite leads  (default, fast)
        'fin'  → finite RGF with SC leads included
        'both' → compute both and return dict

    Parameters
    ----------
    See SNSParams for all parameter definitions.
    Pass either an SNSParams instance or keyword arguments.
    """

    def __init__(self, params: Optional[SNSParams] = None, **kwargs):
        self.p = params if params is not None else SNSParams(**kwargs)
        self._build_fixed_objects()

    def _build_fixed_objects(self):
        """Pre-build objects that don't depend on phi or energy."""
        p = self.p
        self.V     = myf.t_matrix(p.t, p.alpha)
        self.V_dag = self.V.conj().T

        # N-region slices (phi-independent)
        self.H_mid, _ = myf.build_middle_region(
            p.t, p.mu_n, p.alpha, p.B, p.SM)

        # PH symmetry check
        C = np.fliplr(np.eye(p.dof))
        onsite_sc = myf.onsite_matrix(p.t, p.mu_sc, p.B, p.Delta)
        onsite_n  = myf.onsite_matrix(p.t, p.mu_n,  p.B, 0.0)
        self._ph_checks = {
            'onsite_sc': np.allclose(onsite_sc, -C @ onsite_sc.conj() @ C, atol=1e-12),
            'onsite_n' : np.allclose(onsite_n,  -C @ onsite_n.conj()  @ C, atol=1e-12),
            'V_hopping': np.allclose(self.V,     -C @ self.V.conj()    @ C, atol=1e-12),
        }

    # ── Surface GFs ───────────────────────────────────────────────────────────

    def get_surface_gfs(self, E, phi, matsubara=None):
        """
        Return (g_L, g_R) surface GFs for given energy/phi/gauge.
        Pass matsubara=1j*omega_n to use Matsubara frequencies.
        """
        p = self.p
        phase_L = -phi / 2 if p.symmetric else 0.0
        phase_R = +phi / 2 if p.symmetric else phi

        onsite_L = myf.onsite_matrix(p.t, p.mu_sc, p.B,
                                     p.Delta * np.exp(1j * phase_L))
        onsite_R = myf.onsite_matrix(p.t, p.mu_sc, p.B,
                                     p.Delta * np.exp(1j * phase_R))

        g_L, _ = myf.get_surface_gf(E, onsite_L, self.V_dag,
                                     eta=p.eta, matsubara=matsubara)
        g_R, _ = myf.get_surface_gf(E, onsite_R, self.V,
                                     eta=p.eta, matsubara=matsubara)
        return g_L, g_R

    # ── Single-point GF ───────────────────────────────────────────────────────

    def get_gf(self, E, phi, method='inf', matsubara=None, return_full=False):
        """
        Compute Green's function at given energy and phase.

        Parameters
        ----------
        E          : real energy (ignored if matsubara is set)
        phi        : phase difference
        method     : 'inf', 'fin', or 'both'
        matsubara  : complex Matsubara frequency 1j*omega_n (optional)
        return_full: if True returns full matrix; else diagonal blocks only

        Returns
        -------
        G          : array (N_sites, dof, dof) or full matrix
                     dict {'inf': ..., 'fin': ...} if method='both'
        """
        p = self.p

        if method in ('inf', 'both'):
            g_L, g_R = self.get_surface_gfs(E, phi, matsubara=matsubara)
            res_inf = myf.get_rgf_sns(
                self.H_mid, self.V, g_L, g_R, E,
                eta=p.eta, return_full=return_full,
                matsubara=matsubara)

        if method in ('fin', 'both'):
            H_sns, _ = myf.build_sns_junction_sliced(
                p.t, p.mu_sc, p.mu_n, p.alpha, p.B, p.Delta, phi,
                p.SL, p.SR, p.SM, symmetric=p.symmetric)
            res_fin = myf.get_rgf_finite_system(
                H_sns, self.V, E,
                eta=p.eta, return_full=return_full,
                matsubara=matsubara)

        if method == 'inf':
            return res_inf
        if method == 'fin':
            return res_fin
        return {'inf': res_inf, 'fin': (res_fin)}

    # ── LDOS ─────────────────────────────────────────────────────────────────

    def ldos_vs_energy(self, phi, energies, sites=None, method='inf'):
        """
        LDOS as a function of energy at fixed phi.

        Parameters
        ----------
        phi      : phase difference
        energies : array of energies
        sites    : list of N-region site indices (0-indexed within N region)
                   None → all SM sites
        method   : 'inf', 'fin', or 'both'

        Returns
        -------
        ldos : array (n_sites, n_energies) or dict if method='both'
        """
        p    = self.p
        if sites is None:
            sites = list(range(p.SM))
        n_sites = len(sites)
        n_E     = len(energies)

        def _compute(meth):
            out = np.zeros((n_sites, n_E))
            for e_idx, E in enumerate(energies):
                G_diag, GL, GR = self.get_gf(E, phi, method=meth)
                
                for s_idx, s in enumerate(sites):
                        if meth == 'inf':
                            out[s_idx, e_idx] = _get_ldos(G_diag[s])
                        else:
                            out[s_idx, e_idx] = _get_ldos(G_diag[p.SL + s])
            return out

        if method == 'both':
            return {'inf': _compute('inf'), 'fin': _compute('fin')}
        return _compute(method)

    def ldos_vs_phase(self, energy, phases, sites=None, method='inf'):
        """
        LDOS as a function of phase at fixed energy.

        Parameters
        ----------
        energy : fixed energy
        phases : array of phase values
        sites  : list of N-region site indices
        method : 'inf', 'fin', or 'both'

        Returns
        -------
        ldos : array (n_sites, n_phases) or dict if method='both'
        """
        p = self.p
        if sites is None:
            sites = list(range(p.SM))
        n_sites = len(sites)
        n_phi   = len(phases)

        def _compute(meth):
            out = np.zeros((n_sites, n_phi))
            for p_idx, phi in enumerate(phases):
                G_diag, _, _ = self.get_gf(p.target_E, phi, method=meth)

                for s_idx, s in enumerate(sites):
                    if meth == 'inf':
                        out[s_idx, p_idx] = _get_ldos(G_diag[s])
                    else:
                        out[s_idx, p_idx] = _get_ldos(G_diag[p.SL + s])
            return out
        
        if method == 'both':
            return {'inf': _compute('inf'), 'fin': _compute('fin')}
        return _compute(method)

    def spatial_ldos(self, phi, energies, method='inf'):
        """
        Full spatial LDOS map: (SM sites) × (energies).
        Returns array of shape (SM, n_energies).
        """
        return self.ldos_vs_energy(
            phi, energies, sites=list(range(self.p.SM)), method=method)

    # ── Josephson current ─────────────────────────────────────────────────────

    def josephson_current(self, phases, T, n_max=None, method='inf'):
        """
        Josephson current I(φ) via Matsubara frequency sum.

        I(φ) = 2 * 2eT * Σ_{n≥0} Im Tr[ V† G_{01}(iωn) - V G_{10}(iωn) ]
        (factor 2 from summing negative Matsubara frequencies)

        Parameters
        ----------
        phases : array of phase values
        T      : temperature
        n_max  : number of Matsubara frequencies (auto if None)
        method : 'inf', 'fin', or 'both'

        Returns
        -------
        currents : array (n_phases,) or dict if method='both'
        """
        p = self.p

        if n_max is None:
            # enough terms so ω_nmax >> bandwidth (~8t)
            n_max = int(8.0 * p.t / (np.pi * T)) + 100
        omega_ns = (2 * np.arange(0, n_max + 1) + 1) * np.pi * T

        print(f"  Matsubara sum: T={T:.4f}, n_max={n_max}, "
              f"ω_max={omega_ns[-1]:.2f}")

        def _compute(meth):
            currents = np.zeros(len(phases))
            for p_idx, phi in enumerate(phases):
                I_phi = 0.0

                if meth == 'fin':
                    H_sns, _ = myf.build_sns_junction_sliced(p.t, p.mu_sc, p.mu_n, p.alpha, p.B, p.Delta, phi, p.SL, p.SR, p.SM, symmetric=p.symmetric)
                if meth == 'inf':
                    g_L0, g_R0 = self.get_surface_gfs(0, phi)

                for omega_n in omega_ns:
                    z_m = 1j * omega_n
                    
                    # ── INF CASE ─────────────────────────────
                    if meth == 'inf':
                        G_diag, GL, GR = myf.get_rgf_sns(self.H_mid, self.V, g_L0, g_R0, 0, p.eta, return_full=False, matsubara=z_m)
                        i, j = 0, 1
                    
                    # ── FIN CASE ─────────────────────────────
                    else: 
                        G_diag, GL, GR = myf.get_rgf_finite_system(H_sns, self.V, 0, p.eta, return_full=False, matsubara=z_m)

                        i, j = p.SL, p.SL - 1
                    
                    # ── CURRENT CONTRIBUTION ─────────────────
                    G01 = G_diag[i] @ self.V @ GR[j]
                    G10 = GL[i] @ self.V_dag @ G_diag[j]
                    I_phi += 2 * T * np.imag(np.trace(self.V_dag @ G01 - self.V @ G10))

                currents[p_idx] = 2 * I_phi  # factor 2 for negative ωn
                
            return currents

        if method == 'both':
            return {
                'inf': _compute('inf'),
                'fin': _compute('fin')
            }
        return _compute(method)

    def critical_current(self, phases, T, n_max=None, method='inf'):
        """
        Critical current Ic = max_φ |I(φ)|.
        Returns scalar or dict if method='both'.
        """
        currents = self.josephson_current(phases, T, n_max=n_max,
                                          method=method)
        if isinstance(currents, dict):
            return {k: np.max(np.abs(v)) for k, v in currents.items()}
        return np.max(np.abs(currents))

    # ── Checks ────────────────────────────────────────────────────────────────

    def ph_symmetry_report(self):
        """Print particle-hole symmetry checks for Hamiltonians."""
        print("\n══ PH Symmetry Checks ══")
        for name, ok in self._ph_checks.items():
            print(f"  {name:12s}: {'✓ PASS' if ok else '✗ FAIL'}")

    def convergence_check(self, energies, phi=np.pi, sizes=None):
        """
        Check sub-gap LDOS convergence vs SC lead length.
        Compares finite RGF (varying SL/SR) vs Sancho-López.

        Parameters
        ----------
        energies : energy array
        phi      : phase (default π)
        sizes    : list of SL=SR values to test
        """
        p = self.p
        if sizes is None:
            sizes = [50, 100, 150, 200, 300]

        subgap = np.abs(energies) < p.Delta
        site   = p.SM // 2  # centre of N region

        print(f"\n══ Convergence Check (sub-gap, centre of N, φ=π) ══")
        print(f"  {'SL=SR':>8} | {'sub-gap max|ΔLDOS|':>20} | {'verdict':>10}")
        print(f"  {'─'*50}")

        # Reference: Sancho-López (infinite leads)
        ldos_inf = np.zeros(len(energies))
        for e_idx, E in enumerate(energies):
            G_diag, _, _ = self.get_gf(E, phi, method='inf')
            ldos_inf[e_idx] = _get_ldos(G_diag[site]) 

        for sl_sr in sizes:
            # Temporary solver with different lead size
            p_test = SNSParams(
                t=p.t, mu_sc=p.mu_sc, mu_n=p.mu_n,
                alpha=p.alpha, B=p.B, Delta=p.Delta,
                SM=p.SM, SL=sl_sr, SR=sl_sr,
                symmetric=p.symmetric, eta=p.eta)
            solver_test = SNSSolver(params=p_test)

            ldos_fin = np.zeros(len(energies))
            for e_idx, E in enumerate(energies):
                G_diag, _, _ = solver_test.get_gf(E, phi, method='fin')
                ldos_fin[e_idx] = _get_ldos(G_diag[sl_sr + site])

            err_sub = np.max(np.abs(ldos_fin[subgap] - ldos_inf[subgap]))
            ok      = err_sub < 0.1
            print(f"  {sl_sr:>8} | {err_sub:>20.3e} | "
                  f"{'✓ PASS' if ok else '✗ FAIL':>10}")

#%%
# ═════════════════════════════════════════════════════════════════════════════
# Example usage
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ── Initialize ────────────────────────────────────────────────────────────
    solver = SNSSolver(
        t=1.0, mu_sc=0.0025, mu_n=0.01,
        alpha=0.6, B=0.4, Delta=0.1,
        SM=50, SL=200, SR=200,
        symmetric=True, eta=1e-4
    )
    solver.p.summary()
    solver.ph_symmetry_report()

    energies = np.linspace(-0.8, 0.8, 121)
    phases   = np.linspace(0, 2 * np.pi, 61)
    sites    = [1, 25, 48]

    # ── LDOS vs energy ────────────────────────────────────────────────────────
    print("Computing LDOS vs energy...")
    ldos_E = solver.ldos_vs_energy(
        phi=np.pi, energies=energies, sites=sites, method='both')

    fig, axes = plt.subplots(len(sites), 1, figsize=(8, 6), sharex=True)
    labels = ['Left interface', 'Centre', 'Right interface']
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(energies, ldos_E['inf'][i], label='Sancho-López', lw=2) # type: ignore
        ax.plot(energies, ldos_E['fin'][i], label='Finite RGF', # type: ignore
                ls='--', lw=1.5)
        ax.axvline(-solver.p.Delta, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.axvline(+solver.p.Delta, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel('LDOS')
        ax.set_title(lbl, loc='left', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
    axes[-1].set_xlabel('Energy (E/t)')
    fig.suptitle(f'LDOS vs Energy  φ=π  B={solver.p.B}', fontweight='bold')
    fig.tight_layout()
    #fig.savefig('ldos_energy.png', dpi=150)

    # ── Josephson current ─────────────────────────────────────────────────────
    print("Computing Josephson current...")
    T        = 0.01 * solver.p.Delta
    currents = solver.josephson_current(phases, T=T, method='both')
    Ic       = solver.critical_current(phases, T=T, method='both')

    print(f"  Ic (inf) = {Ic['inf']:.6f}")
    print(f"  Ic (fin) = {Ic['fin']:.6f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(phases / np.pi, currents['inf'], # type: ignore
             label=f"Sancho-López  Ic={Ic['inf']:.4f}", lw=2)
    ax2.plot(phases / np.pi, currents['fin'], # type: ignore
             label=f"Finite RGF    Ic={Ic['fin']:.4f}", ls='--', lw=1.5)
    ax2.axhline(0,  color='k', lw=0.5, ls=':')
    ax2.axvline(1,  color='gray', lw=0.8, ls='--', alpha=0.5)
    ax2.set_xlabel('φ / π')
    ax2.set_ylabel('I(φ)  [2e/ħ]')
    ax2.set_title(f'Current-phase relation  B={solver.p.B}  T={T:.4f}',
                  fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    #fig2.savefig('current_phase.png', dpi=150)

    plt.show()

# %%
