"""
transport.py - Multi-terminal transport: self-energies, broadening, and
Landauer/Meir-Wingreen current formulas.
=========================================================================

Built for a 4-terminal SNS-type Josephson junction:
  - two semi-infinite SC leads, attached along the top and bottom
    transverse (y) edges of EVERY x-slice, inducing the proximity gap
    (see matrices.onsite_matrix_sc / hopping_y for the SC lead's
    own onsite/hopping blocks)
  - two semi-infinite normal-metal leads, attached at the left/right ends
    in the x-direction, carrying the current (same role as g_L/g_R
    already used by solvers.get_rgf_sns / get_rgf_phi_sweep)

All four leads are represented via Sancho-Rubio surface Green's functions
(solvers.get_surface_gf), so the same self-energy machinery below applies
uniformly to all of them -- SC or metal.

NOTE: this module is newly written scaffolding, not yet validated against
a known current-phase-relation result. The building blocks (self-energy,
broadening matrix, transmission formula) are standard NEGF/Landauer-
Buttiker definitions, but you should sanity-check the full pipeline
against a known limit (e.g. Delta -> 0, normal 4-terminal conductance
should reduce to a familiar Landauer result) before trusting numbers
out of it. Energy/charge prefactors (e, hbar/h) are left as explicit
keyword arguments rather than hardcoded, since your t-unit convention
(Params.tunits) will determine what "1" means physically.
"""

import numpy as np
from typing import List, Tuple, Optional
from .helpers import embed_block


# ============================================================================
# SELF-ENERGIES AND BROADENING
# ============================================================================

def self_energy(V_couple: np.ndarray, g_surface: np.ndarray) -> np.ndarray:
    """
    Lead self-energy Sigma = V_couple @ g_surface @ V_couple^dagger.

    Args:
        V_couple  : coupling matrix from the system to the lead's surface
                    site(s) (e.g. hopping_x/hopping_y block, or the
                    full x-hopping matrix V for an x-direction lead)
        g_surface : lead surface Green's function (from solvers.get_surface_gf)

    Returns:
        Self-energy matrix, same shape as g_surface (assuming V_couple
        couples system and lead subspaces of equal dimension, as is the
        case for both the SC edge leads and the x-direction metal leads
        here)
    """
    return V_couple @ g_surface @ V_couple.conj().T


def broadening_matrix(Sigma: np.ndarray) -> np.ndarray:
    """
    Broadening (linewidth) matrix Gamma = i(Sigma - Sigma^dagger) = -2 Im[Sigma],
    for a retarded self-energy Sigma. Hermitian and positive semi-definite
    for a physical (dissipative) lead.
    """
    return 1j * (Sigma - Sigma.conj().T)


# ============================================================================
# EMBEDDING EDGE (SC) LEADS INTO SLICE HAMILTONIANS
# ============================================================================

def attach_edge_lead(
    H_slice: np.ndarray,
    V_couple: np.ndarray,
    g_surface: np.ndarray,
    edge_idx: np.ndarray,
) -> np.ndarray:
    """
    Dress a slice Hamiltonian with a lead self-energy attached only at a
    subset of basis indices — e.g. the top or bottom transverse edge of a
    y-chain slice (from ``helpers.get_edge_indices`` with n_edge=1 for a
    single attaching site, matching the nearest-neighbour y-hopping used
    to build the slice itself).

    Args:
        H_slice   : (dim, dim) slice Hamiltonian to dress (not modified in place)
        V_couple  : (len(edge_idx), len(edge_idx)) coupling matrix (e.g.
                    matrices.hopping_y block for an SC lead continuing
                    the y-chain)
        g_surface : (len(edge_idx), len(edge_idx)) lead surface Green's function
        edge_idx  : basis indices of H_slice where the lead attaches

    Returns:
        (dim, dim) dressed slice Hamiltonian
    """
    Sigma = self_energy(V_couple, g_surface)
    return embed_block(H_slice, Sigma, edge_idx, add=True)


def build_dressed_slices(
    H_slices: List[np.ndarray],
    Sigma_top: Optional[np.ndarray],
    top_idx: Optional[np.ndarray],
    Sigma_bottom: Optional[np.ndarray],
    bottom_idx: Optional[np.ndarray],
) -> List[np.ndarray]:
    """
    Apply uniform top/bottom SC-lead self-energies to every slice in a
    list of normal-region Hamiltonians (e.g. ``nx`` copies of
    ``matrices.make_slice_normal``), producing the dressed H_slices to
    feed into the x-direction RGF sweep (``solvers.get_rgf_sns`` /
    ``get_rgf_phi_sweep``) together with the metal leads' surface GFs as
    g_L, g_R.

    Pass Sigma_top/top_idx as None to skip the top lead (e.g. while
    testing with only one SC edge), and likewise for the bottom.

    Args:
        H_slices     : list of N bare (dim, dim) normal-region slice Hamiltonians
        Sigma_top    : (len(top_idx), len(top_idx)) top SC lead self-energy, or None
        top_idx      : indices where the top lead attaches, or None
        Sigma_bottom : bottom SC lead self-energy, or None
        bottom_idx   : indices where the bottom lead attaches, or None

    Returns:
        list of N dressed (dim, dim) slice Hamiltonians
    """
    dressed = []
    for H in H_slices:
        Hd = H
        if Sigma_top is not None:
            Hd = embed_block(Hd, Sigma_top, top_idx, add=True)
        if Sigma_bottom is not None:
            Hd = embed_block(Hd, Sigma_bottom, bottom_idx, add=True)
        dressed.append(Hd)
    return dressed


# ============================================================================
# LANDAUER-BUTTIKER TRANSMISSION AND CURRENT
# ============================================================================

def transmission(G_pq: np.ndarray, Gamma_p: np.ndarray, Gamma_q: np.ndarray) -> float:
    """
    Landauer transmission function T_pq(E) = Tr[Gamma_p @ G_pq @ Gamma_q @ G_pq^dagger].

    G_pq is the retarded Green's function BLOCK connecting the attachment
    site of lead p to the attachment site of lead q — e.g. the (0, N-1)
    block from ``solvers.get_rgf_sns(..., return_full=True)`` for the two
    metal leads sitting at the two ends of the x-chain.

    Args:
        G_pq    : (dof, dof) off-diagonal retarded GF block between lead p and lead q sites
        Gamma_p : (dof, dof) broadening matrix of lead p (at its attachment site)
        Gamma_q : (dof, dof) broadening matrix of lead q (at its attachment site)

    Returns:
        float, transmission (should be real and non-negative for a
        physical retarded GF; small negative values at the 1e-10 level
        are numerical noise)
    """
    return float(np.real(np.trace(Gamma_p @ G_pq @ Gamma_q @ G_pq.conj().T)))


def fermi(energy: np.ndarray, mu: float, kT: float) -> np.ndarray:
    """
    Fermi-Dirac occupation f(E) = 1 / (exp((E-mu)/kT) + 1).

    Args:
        energy : energy array or scalar (same units as mu, kT)
        mu     : chemical potential of the terminal
        kT     : temperature in energy units (k_B * T); kT=0 gives a
                  sharp step function

    Returns:
        occupation, same shape as `energy`
    """
    energy = np.asarray(energy, dtype=float)
    if kT <= 0:
        return (energy < mu).astype(float)
    x = np.clip((energy - mu) / kT, -700, 700)
    return 1.0 / (np.exp(x) + 1.0)


def landauer_current(
    energies: np.ndarray,
    T_pq_of_E: np.ndarray,
    mu_p: float,
    mu_q: float,
    kT_p: float = 0.0,
    kT_q: float = 0.0,
    e_charge: float = 1.602176634e-19,
    h_planck: float = 6.62607015e-34,
) -> float:
    """
    Two-terminal Landauer-Buttiker current from lead q into lead p:

        I_p = (e / h) * integral dE  T_pq(E) * [f_p(E) - f_q(E)]

    For a genuinely multi-terminal setup, call this once per (p, q) pair
    and sum over q != p to get the net current into terminal p.

    Args:
        energies   : 1D array of energies the transmission was evaluated at
        T_pq_of_E  : 1D array, T_pq(E) at each energy (from `transmission`)
        mu_p, mu_q : chemical potentials (electrochemical potentials) of
                     the two terminals
        kT_p, kT_q : temperatures (energy units) of the two terminals
        e_charge   : charge unit (set to your convention; often 1 in
                     natural/t-units, or the physical electron charge)
        h_planck   : Planck constant in your unit system (default 2*pi,
                     i.e. hbar=1 convention -- override to match your
                     Params.tunits energy scale if you're not using
                     natural units)

    Returns:
        float, current I_p (units of e_charge / h_planck * energy)
    """
    f_p = fermi(energies, mu_p, kT_p)
    f_q = fermi(energies, mu_q, kT_q)
    integrand = np.asarray(T_pq_of_E) * (f_p - f_q)
    return float((e_charge / h_planck) * np.trapezoid(integrand, energies))

# ============================================================================
# ELECTRON-HOLE RESOLVED TRANSMISSION AND ONSAGER THERMOELECTRIC FORMALISM
#
# Follows Klees, Gresta, Sturm, Molenkamp, Hankiewicz, "Majorana-mediated
# thermoelectric transport in multiterminal junctions", arXiv:2306.17845.
# That paper's central region is a single-channel double quantum dot, so
# their transmission function (Eq. 1) is a scalar built from single matrix
# elements G^r_{l1 tau1, l2 tau2}. Our central region is a full 2D lattice
# slice, so every quantity below is the natural MATRIX generalization:
# Gamma_p and the (tau1,tau2) block of G_pq are now (dim/2, dim/2)
# sub-matrices (dim = 4*ny), and a trace replaces the bare product.
# Setting ny=1 (single-site leads) recovers their scalar formula exactly.
# ============================================================================

def resolved_transmission(
    G_pq: np.ndarray,
    Gamma_p: np.ndarray,
    Gamma_q: np.ndarray,
    electron_idx: np.ndarray,
    hole_idx: np.ndarray,
) -> dict:
    """
    Electron-hole resolved Landauer transmission functions between the
    attachment sites of lead p and lead q:

        T^{tau1 tau2}_{pq} = Tr[ Gamma_p^{tau1 tau1}  G_pq^{tau1 tau2}
                                  Gamma_q^{tau2 tau2}  (G_pq^{tau1 tau2})^dagger ]

    which generalizes Klees et al. Eq. (1) / (C8),
        T^{tau1 tau2}_{l1 l2}(eps) = 4 Gamma_l1 Gamma_l2
                                     G^r_{l1 tau1, l2 tau2} G^a_{l2 tau2, l1 tau1},
    to matrix-valued (multi-site) leads and self-energies (their factor of
    4 was specific to their scalar Gamma = t^2/t normalization; here it's
    absorbed automatically since Gamma_p/Gamma_q are already the full
    broadening matrices from `broadening_matrix`).

    Args:
        G_pq         : (dim, dim) retarded GF block connecting lead p's
                       and lead q's attachment sites (e.g.
                       G_blocks[0, -1] for two x-direction leads, or
                       G_diag[i] for the LOCAL GF at a single lead's own
                       attachment site, i.e. p=q)
        Gamma_p, Gamma_q : (dim, dim) broadening matrices of leads p, q
        electron_idx, hole_idx : from helpers.get_electron_hole_indices(dim)

    Returns:
        dict with keys 'ee', 'eh', 'he', 'hh' -> float transmission value.
        All should be real and non-negative up to numerical noise.
    """
    idx_map = {'e': electron_idx, 'h': hole_idx}
    T = {}
    for t1 in ('e', 'h'):
        for t2 in ('e', 'h'):
            i1, i2 = idx_map[t1], idx_map[t2]
            Gp = Gamma_p[np.ix_(i1, i1)]
            Gq = Gamma_q[np.ix_(i2, i2)]
            G12 = G_pq[np.ix_(i1, i2)]
            T[t1 + t2] = float(np.real(np.trace(Gp @ G12 @ Gq @ G12.conj().T)))
    return T


def total_transmissions_biased_right(T_ee_RL: np.ndarray, T_eh_RL: np.ndarray, T_eh_RR: np.ndarray):
    """
    Combine electron-hole resolved transmissions into the total
    transmission functions T11, T12, T21, T22 entering the Onsager matrix,
    for the case where only the RIGHT metal lead is biased (voltage and/or
    temperature) while the LEFT stays at the reference mu=0, T (Klees et
    al. Eq. 4a/4b — a consequence of their Appendix C derivation, not a
    generic multi-terminal formula, so it only applies to this particular
    two-normal-lead, bias-on-one-side setup):

        T11 = T21 = T^ee_RL + T^eh_RL + 2 T^eh_RR
        T12 = T22 = T^ee_RL + T^eh_RL

    Args:
        T_ee_RL, T_eh_RL, T_eh_RR : 1D arrays, function of energy (each
            from `resolved_transmission`, evaluated at every energy in a sweep)

    Returns:
        T11, T12, T21, T22 : 1D arrays (T11==T21, T12==T22 by construction)
    """
    T_ee_RL = np.asarray(T_ee_RL); T_eh_RL = np.asarray(T_eh_RL); T_eh_RR = np.asarray(T_eh_RR)
    T11 = T_ee_RL + T_eh_RL + 2 * T_eh_RR
    T12 = T_ee_RL + T_eh_RL
    return T11, T12, T11.copy(), T12.copy()


def onsager_matrix(energies: np.ndarray, T11: np.ndarray, T12: np.ndarray,
                    T21: np.ndarray, T22: np.ndarray, kT: float) -> np.ndarray:
    """
    Onsager matrix L_mn = integral (eps/kT)^(m+n-2) T_mn(eps) (-df/deps) deps,
    evaluated at equilibrium (mu=0, temperature kT) — Klees et al. Eq. (3).

    Uses -df/deps = f(eps)(1-f(eps))/kT for the equilibrium Fermi function
    at mu=0.

    Args:
        energies : 1D energy array the transmissions were evaluated at
        T11, T12, T21, T22 : 1D arrays, total transmission functions (see
                              `total_transmissions_biased_right`)
        kT : temperature in energy units (k_B * T)

    Returns:
        (2, 2) Onsager matrix L = [[L11, L12], [L21, L22]]
    """
    energies = np.asarray(energies, dtype=float)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    f = fermi(energies, 0.0, kT)
    minus_dfde = f * (1 - f) / kT if kT > 0 else np.zeros_like(energies)

    T_mn = {(1, 1): np.asarray(T11), (1, 2): np.asarray(T12),
            (2, 1): np.asarray(T21), (2, 2): np.asarray(T22)}
    L = np.zeros((2, 2))
    for m in (1, 2):
        for n in (1, 2):
            weight = (energies / kT) ** (m + n - 2) if kT > 0 else (energies == 0).astype(float)
            integrand = weight * T_mn[(m, n)] * minus_dfde
            L[m - 1, n - 1] = _trapz(integrand, energies)
    return L


def thermoelectric_coefficients(L: np.ndarray, kT: float, e_charge: float = 1.602176634e-19, h_planck: float = 6.62607015e-34, kB: float = 1.380649e-23) -> dict:
    """
    Electrical conductance G, Peltier coefficient Pi, thermal conductance K,
    Seebeck coefficient S, and the Wiedemann-Franz Lorenz ratio from the
    Onsager matrix L (Klees et al., text below Eq. 3, and Fig. 3 caption
    for the Lorenz ratio L/L0):

        G  = G0 L11,                    G0 = e^2/h
        Pi = kT L21 / (e L11)
        K  = 3 K0 det(L) / (pi^2 L11),  K0 = pi^2 kB^2 T / (3h)
        S  = kB L12 / (e L11)
        Lorenz ratio = K / (G T) / L0,  L0 = pi^2 kB^2 / (3 e^2)
        (= 1 exactly when the Wiedemann-Franz law holds)

    Args:
        L        : (2,2) Onsager matrix from `onsager_matrix`
        kT       : temperature in energy units (k_B * T)
        e_charge : charge unit (match your unit convention)
        h_planck : Planck constant in your unit system (default 2*pi,
                   hbar=1 convention)
        kB       : Boltzmann constant in your unit system (default 1;
                   then kT IS k_B*T already, and "T" below means kT/kB)

    Returns:
        dict with keys 'G','Pi','K','S','lorenz_ratio','G0','K0','L0'
    """
    L11, L12, L21, L22 = L[0, 0], L[0, 1], L[1, 0], L[1, 1]
    detL = L11 * L22 - L12 * L21
    T_abs = kT / kB

    G0 = e_charge**2 / h_planck
    K0 = (np.pi**2 * kB**2 * T_abs) / (3 * h_planck)
    L0 = (np.pi**2 * kB**2) / (3 * e_charge**2)

    G = G0 * L11
    Pi = (kT * L21) / (e_charge * L11) if L11 != 0 else np.nan
    K = (3 * K0 * detL) / (np.pi**2 * L11) if L11 != 0 else np.nan
    S = (kB * L12) / (e_charge * L11) if L11 != 0 else np.nan
    lorenz_ratio = (K / (G * T_abs)) / L0 if (G != 0 and T_abs != 0) else np.nan

    return {'G': G, 'Pi': Pi, 'K': K, 'S': S, 'lorenz_ratio': lorenz_ratio,
            'G0': G0, 'K0': K0, 'L0': L0}