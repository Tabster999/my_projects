"""
helpers.py - Helper Functions & Static Objects
===============================================

This is Step 1 of your workflow: Define helper functions and static objects

Simple utility functions that are used everywhere. No classes, just functions.
"""

import numpy as np
from scipy.linalg import inv
from typing import Tuple, Optional



def phase_matrix(phi):
    """
    U(1) gauge rotation in Nambu basis (up,dn,dn†,up†).

    Args: phi:float, SC phase differenece
    
    Returns: np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)]).astype(np.complex128)
    """
    return np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)]).astype(np.complex128)


# ============================================================================
# GREEN'S FUNCTION HELPERS
# ============================================================================

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


# ============================================================================
# OBSERVABLE EXTRACTION HELPERS
# ============================================================================

def get_ldos_from_gf(G_block: np.ndarray) -> float:
    """
    Extract Local Density of States (LDOS) from a Green's function block.
    
    LDOS(E) = -(1/π) * Im[Tr(G(E))]
    
    Args:
        G_block (np.ndarray): Green's function block (typically 4×4)
    
    Returns:
        float: LDOS at this energy
    """
    return -np.imag(np.trace(G_block)) / np.pi


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


def get_pairing_phase(G_block: np.ndarray) -> float:
    """
    Extract phase of anomalous pairing amplitude.
    
    Args:
        G_block (np.ndarray): Green's function block (typically 4×4)
    
    Returns:
        float: Phase angle of G[0,3] in radians
    """
    from cmath import phase
    return phase(G_block[0, 3])


# ============================================================================
# DIAGNOSTIC CHECKS
# ============================================================================

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


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

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
