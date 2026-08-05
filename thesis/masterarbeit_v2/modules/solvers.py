"""
solvers.py - Green's function solvers
======================================

This module contains the Green's function solver functions copied from the
original `my_functions.py`.
"""

import numpy as np
from scipy.linalg import inv, norm
from typing import Tuple, Optional, List
from .helpers import phase_matrix


def get_surface_gf(
    energy: float,
    eps: np.ndarray,
    t_matrix: np.ndarray,
    eta: float = 1e-4,
    max_iter: int = 400,
    tol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sancho-Lopez retarded surface Green's function for a semi-infinite lead.

    This is the single canonical implementation of the Sancho-Rubio/Lopez
    decimation algorithm (previously also duplicated as a script-local
    ``sancho()`` with a different call signature; that version has been
    folded into this one — see convention note below).

    Convention (IMPORTANT, verified numerically against the previous
    script-local ``sancho(H, V_right, V_left, ...)``):
        get_surface_gf(energy, eps, t_matrix) is equivalent to
        sancho(eps, V_right=t_matrix.conj().T, V_left=t_matrix, ...).
    i.e. ``t_matrix`` plays the role of "V_left" / alpha_0 in the
    decimation recursion; beta_0 is derived automatically as its
    conjugate transpose. There is deliberately no separate "V_left"
    argument: since beta is always t_matrix.conj().T, there's no way to
    accidentally swap left/right hopping arguments the way the old
    two-argument ``sancho()`` allowed.

    Which physical t_matrix to pass for "the left lead" vs "the right
    lead" depends on your Hamiltonian's hopping-direction convention and
    is NOT universal across bases — check against a known limit (e.g.
    alpha=beta=0, where left/right leads must coincide) before trusting
    a new geometry.

    Args:
        energy   : real energy
        eps      : (dof, dof) onsite Hamiltonian block of the bulk lead unit cell
        t_matrix : (dof, dof) inter-cell hopping matrix (see convention note)
        eta      : broadening
        max_iter : maximum decimation iterations (was hardcoded to 600)
        tol      : convergence tolerance on alpha/beta (was hardcoded to 1e-12)

    Returns:
        Gs : (dof, dof) surface Green's function
        Gb : (dof, dof) bulk Green's function
    """
    dof = eps.shape[0]
    z = (energy + 1j * eta) * np.eye(dof, dtype=np.complex128)
    alpha_0 = t_matrix
    beta_0 = np.transpose(np.conjugate(t_matrix))
    g0 = inv(z - eps)

    Epsilon_surf = eps + alpha_0 @ g0 @ beta_0
    Epsilon_bulk = eps + alpha_0 @ g0 @ beta_0 + beta_0 @ g0 @ alpha_0
    alpha        = alpha_0 @ g0 @ alpha_0
    beta         = beta_0  @ g0 @ beta_0

    for i in range(max_iter):
        aux = inv(z - Epsilon_bulk)
        Epsilon_surf = Epsilon_surf + alpha @ aux @ beta
        Epsilon_bulk = Epsilon_bulk + alpha @ aux @ beta + beta @ aux @ alpha
        alpha = alpha @ aux @ alpha
        beta = beta @ aux @ beta
        if norm(alpha, np.inf) < tol and norm(beta, np.inf) < tol:
            break

    Gs = inv(z - Epsilon_surf)
    Gb = inv(z - Epsilon_bulk)
    return Gs, Gb


def apply_phase_gauge(G: np.ndarray, phi: float, N_y: int) -> np.ndarray:
    """
    Apply a U(1) SC-phase gauge rotation to a Green's function (or any
    operator in the same Nambu basis) via U†(φ) G U(φ), U = phase_matrix(φ, N_y).

    Generalizes the old script-local ``_apply_phase`` (verified numerically
    identical for the (c↑,c↓,c↓†,±c↑†)-type bases, since the phase rotation
    only cares about which indices are "hole-like" mod 4, not the overall
    sign convention on the last component).

    Args:
        G   : (4*N_y, 4*N_y) operator (e.g. a surface Green's function)
        phi : SC phase to apply
        N_y : transverse width

    Returns:
        (4*N_y, 4*N_y) gauge-rotated operator
    """
    U = phase_matrix(phi, N_y=N_y)
    return U.conj().T @ G @ U


def calc_G(energy: float, hamiltonian: np.ndarray, eta: float = 1e-4, ra: str = 'r') -> np.ndarray:
    """Calculate the retarded/advanced Green's function by matrix inversion."""
    if ra == 'r':
        zenr = energy + 1j * eta
    elif ra == 'a':
        zenr = energy - 1j * eta
    else:
        raise ValueError("ra must be 'r' or 'a'")

    rows = hamiltonian.shape[0]
    Idmatrix = np.eye(rows, dtype=hamiltonian.dtype)
    Gf = np.linalg.inv(zenr * Idmatrix - hamiltonian)
    return Gf


def get_G_energy(energy_array: np.ndarray, hamiltonian: np.ndarray, eta: float = 1e-4, ra: str = 'r') -> np.ndarray:
    """
    Calculates retarded or advanced Green functions G(E) over the specified energy range.

    Args:
        energy_array : np.ndarray, energy range over which G_r(E) is calculated
        oniste_matrix:np.ndarray, onsite H_0 in a tight binding model
        hopping_matrix:np.ndarray, hopping T in a tight binding model
        eta: float:default 1e-4, broadening parameter IT MUST BE REAL
        ra: str:default 'r', specifies if retarded ('r') or advanced ('a') Green function is calculated.        
    Returns:
        G(E) as an array with dimensions [energy_steps, sites, sites]
        
    """
    if ra == 'r':
        zenr_array = energy_array + 1j * eta
    elif ra == 'a':
        zenr_array = energy_array - 1j * eta
    else:
        raise ValueError("ra must be 'r' or 'a'")

    rows = hamiltonian.shape[0]
    Idmatrix = np.eye(rows, dtype=hamiltonian.dtype)
    matrix_stack = zenr_array[:, np.newaxis, np.newaxis] * Idmatrix - hamiltonian
    Gf_stack = np.linalg.inv(matrix_stack)
    return Gf_stack


def get_G_lehmann(energy_array: np.ndarray, evals: np.ndarray, evecs: np.ndarray, eta: float, ra: str = 'r') -> np.ndarray:
    """
    Calculates the retarded Green's function via the Lehmann representation.

    Args:
        energy_array (np.ndarray): 1D array of energies. Shape: (E_steps,).
        evals (np.ndarray): 1D array of eigenvalues. Shape: (N,).
        evecs (np.ndarray): 2D array of eigenvectors. Shape: (N, N).
        eta (float): Small broadening parameter.
        ra: str:default 'r', specifies if retarded ('r') or advanced ('a') Green function is calculated.

    Returns:
        np.ndarray: 3D complex array of G(E), shape (E_steps, N, N).
    """
    if ra == 'r':
        zenr = 1j * eta
    elif ra == 'a':
        zenr = -1j * eta
    else:
        raise ValueError("ra must be 'r' or 'a'")

    numerator = np.einsum('ik,jk->kij', evecs, np.conjugate(evecs))
    denominator = 1 / (energy_array[:, np.newaxis] - evals + zenr)
    Gf = np.einsum('Ek,kij->Eij', denominator, numerator)
    return np.array(Gf)


def get_G_k_energy(
    k_array: np.ndarray,
    energy_array: np.ndarray,
    h0_matrix: np.ndarray,
    hopping_matrix: np.ndarray,
    eta: float = 1e-4,
    ra: str = 'r'
) -> np.ndarray:
    """
    Calculates G(k, E) over a grid of k-vectors and energies.

    Args:
        k_array (np.ndarray): 1D array of k-vectors. Shape: (k_steps,).
        energy_array (np.ndarray): 1D array of energies. Shape: (E_steps,).
        h0_matrix (np.ndarray): Onsite Hamiltonian matrix. Shape: (N, N).
        hopping_matrix (np.ndarray): Hopping matrix. Shape: (N, N).
        eta (float): Broadening parameter.
        ra (str): Retarded ('r') or advanced ('a').

    Returns:
        np.ndarray: 4D complex array G(k, E), shape (k_steps, E_steps, N, N).
    """
    if ra == 'r':
        zenr_array = energy_array + 1j * eta
    elif ra == 'a':
        zenr_array = energy_array - 1j * eta
    else:
        raise ValueError("ra must be 'r' or 'a'")

    exp_ik = np.exp(1j * k_array)
    H_k_stack = (
        h0_matrix[np.newaxis, :, :] +
        hopping_matrix[np.newaxis, :, :] * exp_ik[:, np.newaxis, np.newaxis] +
        hopping_matrix.T.conj()[np.newaxis, :, :] * np.conj(exp_ik)[:, np.newaxis, np.newaxis]
    )

    N = h0_matrix.shape[0]
    Idmatrix = np.eye(N, dtype=h0_matrix.dtype)

    matrix_to_invert = (
        zenr_array[np.newaxis, :, np.newaxis, np.newaxis] * Idmatrix -
        H_k_stack[:, np.newaxis, :, :]
    )

    Gf_stack = np.linalg.inv(matrix_to_invert)
    return Gf_stack


def get_rgf_sns(H_slices, V, g_L, g_R, energy, eta: float = 1e-4, ra: str = 'r', return_full: bool = False, return_dense: bool = False):
    """
    RGF for SNS junction with infinite leads via recursive algorithm.
    The infinite leads are represented by surface Green's functions computed via Sancho-López.
    This algorithm is based on the paper from Lewenkopf and Mucciolo 2013 "The recursive Green's function method for graphene" (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.155426) and adapted to the SNS junction geometry.

    Algorithm:
    1. LEFT SWEEP: Propagate from left lead boundary forward
       GL[i] = (z*I - H_i - Sigma_L[i])^-1, where Sigma_L[i] = V† GL[i-1] V

    2. RIGHT SWEEP: Propagate from right lead boundary backward  
       GR[i] = (z*I - H_i - Sigma_R[i])^-1, where Sigma_R[i] = V GR[i+1] V†

    3. COMBINE: Full local GF = (z*I - H_i - Sigma_L[i] - Sigma_R[i])^-1

    Args:    
        middle_hamiltonian: np.ndarray, Hamiltonian for finite middle region (includes internal hopping)
                           Dimensions: (sites_mid * dof, sites_mid * dof)
        V: np.ndarray, hopping matrix between adjacent sites (dof x dof)
        g_L: np.ndarray, surface Green's function for left lead (dof x dof)
        g_R: np.ndarray, surface Green's function for right lead (dof x dof)
        energy: float, energy at which to calculate Green's function
        eta: float, imaginary broadening parameter
        ra: str, 'r' for retarded or 'a' for advanced Green's function
        return_full: bool, if True returns full matrix; if False returns diagonal blocks only
        return_dense: bool, if True it returns the complete NxN GF as the 5th output

    Returns:
        G_out: np.ndarray, Green's function (diagonal blocks or full matrix)
        GL: np.ndarray, local GFs from left sweep (sites x dof x dof)
        GR: np.ndarray, local GFs from right sweep (sites x dof x dof)
    """
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    I = np.eye(dof, dtype=np.complex128)
    z = energy + 1j * eta if ra == 'r' else energy - 1j * eta

    # 1.RIGHT sweep: GR[i] = (z - H[i] - V GR[i+1] V†)^{-1}    
    GR = np.zeros((N, dof, dof), dtype=np.complex128)
    GR[-1] = inv(z * I - H_slices[-1] - V @ g_R @ V.conj().T)
    for i in range(N-2, -1, -1):
        GR[i] = inv(z * I - H_slices[i] - V @ GR[i+1] @ V.conj().T)
    
    # 2.LEFT sweep: GL[i] = (z - H[i] - V† GL[i-1] V)^{-1}
    GL = np.zeros((N, dof, dof), dtype=np.complex128)
    GL[0] = inv(z * I - H_slices[0] - V.conj().T @ g_L @ V)
    for i in range(1, N):
        GL[i] = inv(z * I - H_slices[i] - V.conj().T @ GL[i-1] @ V)

    # 3. COMBINE: G[i,i] = (z - H[i] - Σ_L[i] - Σ_R[i])^{-1}
    G_diag = np.zeros((N, dof, dof), dtype=np.complex128)
    for i in range(N):
        S_L = V.conj().T @ GL[i-1] @ V if i > 0 else V.conj().T @ g_L @ V 
        S_R = V @ GR[i+1] @ V.conj().T if i < N - 1 else V @ g_R @ V.conj().T
        G_diag[i] = inv(z * I - H_slices[i] - S_L - S_R)

    if not return_full:
        return G_diag, GL, GR, None, None

    # 4. FULL MATRIX
    G_blocks = np.zeros((N, N, dof, dof), dtype=np.complex128)
    for i in range(N):
        G_blocks[i, i] = G_diag[i]
    for i in range(N):
        for j in range(i+1, N):
            G_blocks[i, j] = G_blocks[i, j-1] @ V @ GR[j]
    for i in range(N):
        for j in range(i-1, -1, -1):
            G_blocks[i, j] = G_blocks[i, j+1] @ V.conj().T @ GL[j]

    if not return_dense:
        return G_diag, GL, GR, G_blocks, None
    
    # 5. CONVERT TO DENSE NxN MATRIX
    dim = N * dof
    G_dense = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            G_dense[i*dof:(i+1)*dof, j*dof:(j+1)*dof] = G_blocks[i, j]
    return G_diag, GL, GR, G_blocks, G_dense


def get_rgf_finite_system(H_slices, V, energy, eta: float = 1e-4, ra: str = 'r', return_full: bool = False, return_dense: bool = False):
    """
    Computes the RGF for a strictly finite system.
    
    Args:
        H_slices: List of Hamiltonian slices for each site.
        V: The hopping matrix connecting the blocks.
        energy: Calculation energy.
        eta: Imaginary broadening.
        return_full: If True, returns the full N*dof x N*dof matrix.
        return_dense: bool, if True it returns the complete NxN GF as the 5th output.


    Returns: 
        G_out: The Green's function (either diagonal blocks or full matrix).
        GL: Local GFs from left sweep (sites x dof x dof).
        GR: Local GFs from right sweep (sites x dof x dof).
    """
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    I = np.eye(dof, dtype=np.complex128)
    z = energy + 1j * eta if ra == 'r' else energy - 1j * eta

    # 1. LEFT-TO-RIGHT SWEEP
    GL = np.zeros((N, dof, dof), dtype=np.complex128)
    GL[0] = inv(z * I - H_slices[0])
    for i in range(1, N):
        GL[i] = inv(z * I - H_slices[i] - V.conj().T @ GL[i-1] @ V)

    # 2. RIGHT-TO-LEFT SWEEP
    GR = np.zeros((N, dof, dof), dtype=np.complex128)
    GR[N-1] = inv(z * I - H_slices[N-1])
    for i in range(N-2, -1, -1):
        GR[i] = inv(z * I - H_slices[i] - V @ GR[i+1] @ V.conj().T)

    # 3. CONSTRUCT DIAGONAL BLOCKS
    G_diag = np.zeros((N, dof, dof), dtype=np.complex128)
    for i in range(N):
        S_L = V.conj().T @ GL[i-1] @ V if i > 0 else 0
        S_R = V @ GR[i+1] @ V.conj().T if i < N-1 else 0
        G_diag[i] = inv(z * I - H_slices[i] - S_L - S_R)

    if not return_full:
        return G_diag, GL, GR, None, None

    # 4. FULL MATRIX RECONSTRUCTION
    G_blocks = np.zeros((N, N, dof, dof), dtype=np.complex128)
    for i in range(N):
        G_blocks[i, i] = G_diag[i]
    for i in range(N):
        for j in range(i+1, N):
            G_blocks[i, j] = G_blocks[i, j-1] @ V @ GR[j]
    for i in range(N):
        for j in range(i-1, -1, -1):
            G_blocks[i, j] = G_blocks[i, j+1] @ V.conj().T @ GL[j]

    if not return_dense:
        return G_diag, GL, GR, G_blocks, None

    # 5. CONVERT TO DENSE NxN MATRIX
    dim = N * dof
    G_dense = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            G_dense[i*dof:(i+1)*dof, j*dof:(j+1)*dof] = G_blocks[i, j]
    return G_diag, GL, GR, G_blocks, G_dense


def get_rgf_phi_sweep(
    H_slices: List[np.ndarray],
    V: np.ndarray,
    g_L_bare: np.ndarray,
    g_R_bare: np.ndarray,
    phi_vals: np.ndarray,
    N_y: int,
    energy: float,
    eta: float = 1e-4,
    ra: str = 'r',
    symmetric: bool = False,
    return_full: bool = False,
):
    """
    RGF sweep for an SNS-type middle region at a SINGLE energy but over MANY
    SC phase differences phi, exploiting the fact that the middle-region
    Hamiltonian (H_slices, V) is gauge-independent w.r.t. phi — only the
    lead surface Green's functions need to be phase-rotated (Scharf-Pientka
    Eq. structure; see also build_sns_junction_sliced's gauge convention).

    Gauge conventions:
        symmetric=False 
            the LEFT lead stays at phi=0 for every phi in phi_vals, so its sweep (glr) is
            computed ONCE and reused — this is the actual speed win. Only
            the RIGHT lead's bare surface GF is gauge-rotated per phi.
        symmetric=True: 
            BOTH leads carry ±phi/2, so the left sweep also
            depends on phi and must be recomputed per phi (no reuse
            possible).
    Args:
        H_slices  : list of N (dof, dof) on-site blocks for the MIDDLE
                    region only (no leads)
        V         : (dof, dof) inter-slice hopping, H_slices[i] -> H_slices[i+1]
        g_L_bare  : (dof, dof) LEFT lead surface GF at phi=0
        g_R_bare  : (dof, dof) RIGHT lead surface GF at phi=0
        phi_vals  : 1D array of SC phase differences to sweep
        N_y       : transverse width (dof = 4*N_y expected by apply_phase_gauge)
        energy    : fixed energy for this sweep
        eta, ra   : as elsewhere
        symmetric : gauge choice, see above
        return_full : if True, also returns full (N, N, dof, dof) blocks per
                      phi (memory-heavy; needed for off-diagonal/current
                      quantities between different slices)

    Returns:
        G_diag : (nphi, N, dof, dof) diagonal-block Green's functions
        G_blocks : (nphi, N, N, dof, dof), only if return_full=True
    """
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    I = np.eye(dof, dtype=np.complex128)
    z = energy + 1j * eta if ra == 'r' else energy - 1j * eta
    nphi = len(phi_vals)

    G_diag = np.zeros((nphi, N, dof, dof), dtype=np.complex128)
    if return_full:
        G_blocks = np.zeros((nphi, N, N, dof, dof), dtype=np.complex128)

    if not symmetric:
        # phi-independent left sweep, computed once
        glr = np.empty((N, dof, dof), dtype=np.complex128)
        glr[0] = inv(z * I - H_slices[0] - V.conj().T @ g_L_bare @ V)
        for i in range(1, N):
            glr[i] = inv(z * I - H_slices[i] - V.conj().T @ glr[i-1] @ V)

    for iphi, phi in enumerate(phi_vals):
        if symmetric:
            g_L = apply_phase_gauge(g_L_bare, -phi / 2, N_y)
            g_R = apply_phase_gauge(g_R_bare,  phi / 2, N_y)
            glr_phi = np.empty((N, dof, dof), dtype=np.complex128)
            glr_phi[0] = inv(z * I - H_slices[0] - V.conj().T @ g_L @ V)
            for i in range(1, N):
                glr_phi[i] = inv(z * I - H_slices[i] - V.conj().T @ glr_phi[i-1] @ V)
            left_env = glr_phi
            g_L_used = g_L
        else:
            g_R = apply_phase_gauge(g_R_bare, phi, N_y)
            left_env = glr
            g_L_used = g_L_bare

        grl = np.empty((N, dof, dof), dtype=np.complex128)
        grl[-1] = inv(z * I - H_slices[-1] - V @ g_R @ V.conj().T)
        for i in range(N - 2, -1, -1):
            grl[i] = inv(z * I - H_slices[i] - V @ grl[i+1] @ V.conj().T)

        for i in range(N):
            S_L = V.conj().T @ g_L_used @ V if i == 0 else V.conj().T @ left_env[i-1] @ V
            S_R = V @ g_R @ V.conj().T if i == N - 1 else V @ grl[i+1] @ V.conj().T
            G_diag[iphi, i] = inv(z * I - H_slices[i] - S_L - S_R)

        if return_full:
            G_blocks[iphi, 0, 0] = G_diag[iphi, 0]
            for i in range(N):
                G_blocks[iphi, i, i] = G_diag[iphi, i]
            for i in range(N):
                for j in range(i+1, N):
                    G_blocks[iphi, i, j] = G_blocks[iphi, i, j-1] @ V @ grl[j]
            for i in range(N):
                for j in range(i-1, -1, -1):
                    G_blocks[iphi, i, j] = G_blocks[iphi, i, j+1] @ V.conj().T @ left_env[j]

    if return_full:
        return G_diag, G_blocks
    return G_diag, None


def get_surface_gfs_phased(
    E: float,
    H_slice: np.ndarray,
    V_x: np.ndarray,
    N_y: int,
    phi: float,
    symmetric: bool = False,
    eta: float = 1e-4,
    max_iter: int = 400,
    tol: float = 1e-14,
):
    """
    Returns the (left, right) surface Green's functions of a lead pair with
    SC phase difference phi applied as a Nambu gauge rotation.

    Args:
        E:float, Energy at which the surface gfs are calculated.
        H_slice:np.ndarray, Hamiltonian slice for the lead.
        V_x:np.ndarray, Hopping matrix connecting the blocks.
        N_y:int, Number of sites in the y-direction.
        phi:float, Phase difference of the SC leads.
        symmetric:bool, True = symmetric gauge, False = antisymmetric gauge

    Convention (matches build_sns_junction_sliced):
      U(theta)† @ H(Delta) @ U(theta) = H(Delta * exp(i*theta))

    Symmetric gauge:
      Delta_L = Delta*exp(i*phi/2)  →  UL = phase_matrix(-phi/2)
      Delta_R = Delta*exp(-i*phi/2)  →  UR = phase_matrix(+phi/2)

    Asymmetric gauge:
      Delta_L = Delta (real)          →  UL = identity
      Delta_R = Delta*exp(+i*phi)     →  UR = phase_matrix(phi)
    """
    gL, _ = get_surface_gf(E, H_slice, V_x.conj().T, eta=eta, max_iter=max_iter, tol=tol)
    gR, _ = get_surface_gf(E, H_slice, V_x, eta=eta, max_iter=max_iter, tol=tol)

    if symmetric:
        g_L_phased = apply_phase_gauge(gL, -phi / 2, N_y)
        g_R_phased = apply_phase_gauge(gR,  phi / 2, N_y)
    else:
        g_L_phased = gL
        g_R_phased = apply_phase_gauge(gR, phi, N_y)

    return g_L_phased, g_R_phased


# Backward-compatible alias: previously a separate, identical function.
get_surface_gfs_2d_phased = get_surface_gfs_phased

def rgf_one_energy_spatial(w: float, phi: float, Hn: np.ndarray, V0: np.ndarray, Vd: np.ndarray, gSL: np.ndarray, gSR: np.ndarray, Id: np.ndarray, eta: float, nx: int, ny: int) -> np.ndarray:
    """
    Spatially resolved RGF calculation at one energy w and one phase phi.
    The phase is applied to the right lead surface GF gSR and the unphased surface GFs gSL, gSR are used for the leads.
    
    Args: 
        w (float): Energy.
        phi (float): Phase.
        Hn (numpy.ndarray): Hamiltonian for the central region.
        V0 (numpy.ndarray): Hopping matrix for the left lead.
        Vd (numpy.ndarray): Hopping matrix for the right lead.
        gSL (numpy.ndarray): Surface Green's function for the left lead.
        gSR (numpy.ndarray): Surface Green's function for the right lead.
        Id (numpy.ndarray): Identity matrix.
        eta (float): Imaginary part of the energy.
        nx (int): Number of sites in the x-direction.
        ny (int): Number of sites in the y-direction.
    """
    z   = w + 1j * eta
    zI  = z * Id
    ny  = ny
    nx  = nx
    dim = 4 * ny

    glr    = np.empty((nx, dim, dim), dtype=np.complex128)
    glr[0] = gSL
    for i in range(1, nx):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)

    grl_phi    = np.empty((nx, dim, dim), dtype=np.complex128)
    grl_phi[-1] = apply_phase_gauge(gSR, phi, ny)
    for i in range(nx - 2, -1, -1):
        grl_phi[i] = inv(zI - Hn - V0 @ grl_phi[i+1] @ Vd)

    result = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        left_dressed = zI - Hn - Vd @ glr[i] @ V0
        G    = inv(left_dressed - V0 @ grl_phi[i] @ Vd)
        diag = np.diagonal(G).reshape(ny, 4)
        result[i, :] = -np.imag(diag.sum(axis=1)) / np.pi

    return result


