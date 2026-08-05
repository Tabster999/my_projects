"""
helpers.py - Helper Functions & Static Objects
===============================================

This is Step 1 of your workflow: Define helper functions and static objects

Simple utility functions that are used everywhere. No classes, just functions.
"""

import numpy as np
from scipy.linalg import inv
from typing import Tuple, Optional


def phase_matrix(phi, N_y=1):
    """
    U(1) gauge rotation in Nambu basis (up,dn,dn†,up†).

    Args: phi:float, SC phase difference
          N_y: int, number of sites in y-direction (for 2D systems, this will be the width of the junction)

    Returns: np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)]).astype(np.complex128)
    """

    U_4 = np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)]).astype(np.complex128)
    return np.kron(np.eye(N_y, dtype=np.complex128), U_4)


def get_z(energy: float, eta: float, ra: str = 'r', matsubara: Optional[complex] = None) -> complex:
    """
    Get complex frequency z = E ± i*η

    Args:
        energy (float): Real energy
        eta (float): Broadening parameter
        ra (str): 'r' for retarded, 'a' for advanced
        matsubara (complex, optional): If provided, use this instead (for Matsubara frequencies)

    Returns:
        complex: z value
    """
    if matsubara is not None:
        return matsubara

    if ra == 'r':
        return energy + 1j * eta
    elif ra == 'a':
        return energy - 1j * eta
    else:
        raise ValueError("ra must be 'r' or 'a'")


def get_ldos_site(G, y, dof=4):
    idx = slice(y*dof, (y+1)*dof)
    return -np.imag(np.trace(G[idx, idx])) / np.pi


def get_edge_indices(N_y: int, n_edge: int, dof: int = 4, side: str = 'top',
                      electron_only: bool = False) -> np.ndarray:
    """
    Basis indices for a transverse edge-probe region of width n_edge, within
    a single slice of transverse width N_y (dof degrees of freedom per site).

    Generalizes the old script-local ``_edge_indices``: adds a ``side``
    argument so the SAME function can probe the top edge (e.g. a top SC
    lead / probe in a 4-terminal geometry) or the bottom edge (bottom SC
    lead), instead of only ever taking the last n_edge sites.

    Args:
        N_y           : transverse width (sites)
        n_edge        : probe width (sites), n_edge <= N_y
        dof           : degrees of freedom per site (4 for Nambu)
        side          : 'top' (last n_edge y-sites) or 'bottom' (first n_edge y-sites)
        electron_only : if True, restrict to electron-sector indices only
                        (first two dof entries of each site's block); use
                        this for basis conventions where dof=4 splits as
                        (e_up, e_dn, h_dn, h_up)-like ordering.

    Returns:
        1D array of basis indices into a (dof*N_y, dof*N_y) slice matrix.
    """
    if side == 'top':
        idx = np.arange(dof * (N_y - n_edge), dof * N_y)
    elif side == 'bottom':
        idx = np.arange(0, dof * n_edge)
    else:
        raise ValueError("side must be 'top' or 'bottom'")

    if electron_only:
        idx = idx[idx % dof < dof // 2]
    return idx


def ldos_trace(G: np.ndarray, idx: np.ndarray) -> float:
    """
    LDOS contribution from a Green's function block, traced over a subset
    of basis indices (e.g. an edge-probe region from ``get_edge_indices``).

    Generalizes the old script-local ``_ldos_trace`` (same formula, just
    promoted out of the script so it can be reused for any probe region —
    edge, single site, lead-attachment region, etc.).

    Args:
        G   : (dim, dim) Green's function block
        idx : 1D array of basis indices to trace over

    Returns:
        float: -Im Tr[G[idx, idx]] / pi
    """
    return -np.imag(np.trace(G[np.ix_(idx, idx)])) / np.pi


def embed_block(M: np.ndarray, block: np.ndarray, idx: np.ndarray, add: bool = True) -> np.ndarray:
    """
    Place a small block into a subset of rows/columns of a larger matrix.

    Used to embed a lead self-energy (computed on a small edge region,
    e.g. one y-site's worth of degrees of freedom) into a full slice
    Hamiltonian at the indices where that lead physically attaches
    (typically from ``get_edge_indices``).

    Args:
        M     : (dim, dim) matrix to embed into (NOT modified in place)
        block : (len(idx), len(idx)) block to embed
        idx   : 1D array of row/column indices in M where block goes
        add   : if True, adds block to the existing entries (use this for
                accumulating multiple lead self-energies onto the same
                Hamiltonian); if False, overwrites them

    Returns:
        (dim, dim) copy of M with block embedded at idx
    """
    out = M.copy()
    if add:
        out[np.ix_(idx, idx)] += block
    else:
        out[np.ix_(idx, idx)] = block
    return out


def get_electron_hole_indices(dim: int, dof: int = 4):
    """
    Partition a Nambu-space matrix's basis indices into electron and hole
    subspaces, assuming every dof-sized site block orders its first dof//2
    entries as electron-like and its last dof//2 as hole-like. This holds
    for both the plain Nambu basis (c up, c down, c down dagger, c up
    dagger) and the Scharf-Pientka basis (c up, c down, c down dagger,
    -c up dagger) used in matrices.py -- the minus sign on the last
    component doesn't change which slots are electron vs hole.

    Needed to build electron-hole resolved transmission functions
    T^{tau1 tau2}_{p q} (Klees et al., Majorana-mediated thermoelectric
    transport in multiterminal junctions, arXiv:2306.17845, Eq. 1) from a
    multi-site retarded GF block, generalizing their single-channel
    quantum-dot formula.

    Args:
        dim : total dimension (must be a multiple of dof)
        dof : degrees of freedom per site (4 for Nambu)

    Returns:
        electron_idx, hole_idx : 1D index arrays, each of length dim//2
    """
    idx = np.arange(dim)
    electron_idx = idx[idx % dof < dof // 2]
    hole_idx = idx[idx % dof >= dof // 2]
    return electron_idx, hole_idx


def get_pairing_amplitude(G_block: np.ndarray) -> float:
    """
    Extract anomalous pairing amplitude from Green's function.

    For Nambu basis (c↑, c↓, c↓†, c↑†), the anomalous component is G[0,3].

    Args:
        G_block (np.ndarray): Green's function block (typically 4×4)

    Returns:
        float: |G[0,3]| - magnitude of anomalous pair amplitude
    """
    return np.abs(G_block[0, 3])


def get_pairing_phase(G: np.ndarray, y, dof=4) -> float:
    """
    Extract phase of anomalous pairing amplitude.

    Args:
        G_block (np.ndarray): Green's function block (typically 4×4)

    Returns:
        float: Phase angle of G[0,3] in radians
    """
    from cmath import phase
    idx = slice(y*dof, (y+1)*dof)
    Gyy = G[idx, idx]
    F = Gyy[2:4, 0:2]
    return np.abs(np.trace(F))


def check_particle_hole_symmetry(G: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if LDOS(E) ≈ LDOS(-E) for particle-hole symmetry.

    Args:
        G (np.ndarray): Green's function, shape (2, N_E, 4, 4)
                       where the two slices are G(E) and G(-E)
        tolerance (float): Numerical tolerance

    Returns:
        bool: True if symmetry holds
    """
    ldos_plus = -np.imag(np.trace(G[0])) / np.pi
    ldos_minus = -np.imag(np.trace(G[1])) / np.pi

    return np.allclose(ldos_plus, ldos_minus, atol=tolerance)


def check_ldos_positivity(G: np.ndarray, tolerance: float = -1e-10) -> bool:
    """
    Check if LDOS(E) ≥ 0 (within numerical tolerance).

    Args:
        G (np.ndarray): Green's function array, shape (..., 4, 4)
        tolerance (float): Allow small negative values up to this

    Returns:
        bool: True if all LDOS values are non-negative
    """
    ldos = -np.imag(np.trace(G, axis1=-2, axis2=-1)) / np.pi
    return bool(np.all(ldos >= tolerance))


def get_colorbar_label(observable: str) -> str:
    """
    Get appropriate colorbar label for an observable.

    Args:
        observable (str): Name of observable (e.g., 'LDOS', 'pairing')

    Returns:
        str: Formatted label
    """
    labels = {
        "ldos": "LDOS (1/π) [arb. units]",
        "pairing": "|G₀₃| [arb. units]",
        "phase": "Phase(G₀₃) [rad]",
        "conductance": "G [arb. units]",
    }
    return labels.get(observable.lower(), observable)