"""
solvers.py - Green's function solvers
======================================

This module contains the Green's function solver functions copied from the
original `my_functions.py`.
"""

import numpy as np
from scipy.linalg import inv
from typing import Tuple, Optional
from .helpers import phase_matrix


def get_surface_gf(energy: float, eps: np.ndarray, t_matrix: np.ndarray, eta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Sancho-López retarded surface Green's function for a semi-infinite lead."""
    dof = eps.shape[0]
    z = (energy + 1j * eta) * np.eye(dof, dtype=np.complex128)
    alpha = t_matrix
    beta = np.transpose(np.conjugate(t_matrix))
    Epsilon_surf = eps
    Epsilon_bulk = eps

    for i in range(600):
        aux = inv(z - Epsilon_bulk)
        Epsilon_surf = Epsilon_surf + alpha @ aux @ beta
        Epsilon_bulk = Epsilon_bulk + alpha @ aux @ beta + beta @ aux @ alpha
        alpha = alpha @ aux @ alpha
        beta = beta @ aux @ beta
        if max(np.linalg.norm(alpha), np.linalg.norm(beta)) < 1e-14:
            break

    Gs = inv(z - Epsilon_surf)
    Gb = inv(z - Epsilon_bulk)
    return Gs, Gb


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
    Sigma_R0 = V @ g_R @ V.conj().T
    GR[N-1] = inv(z * I - H_slices[N-1] - Sigma_R0)
    for i in range(N-2, -1, -1):
        GR[i] = inv(z * I - H_slices[i] - V @ GR[i+1] @ V.conj().T)
    
    # 2.LEFT sweep: GL[i] = (z - H[i] - V† GL[i-1] V)^{-1}
    GL = np.zeros((N, dof, dof), dtype=np.complex128)
    Sigma_L0 = V.conj().T @ g_L @ V
    GL[0] = inv(z * I - H_slices[0] - Sigma_L0)
    for i in range(1, N):
        GL[i] = inv(z * I - H_slices[i] - V.conj().T @ GL[i-1] @ V)

    # 3. COMBINE: G[i,i] = (z - H[i] - Σ_L[i] - Σ_R[i])^{-1}
    G_diag = np.zeros((N, dof, dof), dtype=np.complex128)
    for i in range(N):
        if i == 0:
            S_L = V.conj().T @ g_L @ V
        else:
            S_L = V.conj().T @ GL[i-1] @ V
        if i == N-1:
            S_R = V @ g_R @ V.conj().T
        else:
            S_R = V @ GR[i+1] @ V.conj().T
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


def get_surface_gfs_phased(E: float, H_slice: np.ndarray, V_x: np.ndarray, N_y: int, phi: float, symmetric: bool = False, eta: float = 1e-4):
    """
    Returns the Surface GFs of a left and right lead with SC phase difference phi applied as Nambu gauge rotation.
    
    Args:
        E:float, Energy at which the surface gfs are calculated.
        H_slice:np.ndarray, Hamiltonian slice for the lead.
        V_x:np.ndarray, Hopping matrix connecting the blocks.
        N_y:int, Number of sites in the y-direction.
        phi:float, Phase difference of the SC leads.
        symmetric:bool, True = symmetric gauge, False = antisymmetric gauge
    
    Convention (matches build_sns_junction_sliced):
      U(theta) @ H(Delta) @ U(theta)† = H(Delta * exp(-i*theta))
    
    Symmetric gauge:
      Delta_L = Delta*exp(i*phi/2)  →  UL = phase_matrix(-phi/2)
      Delta_R = Delta*exp(-i*phi/2)  →  UR = phase_matrix(+phi/2)
    
    Asymmetric gauge:
      Delta_L = Delta (real)          →  UL = identity
      Delta_R = Delta*exp(+i*phi)     →  UR = phase_matrix(phi)

    
    """
    gL, _ = get_surface_gf(E, H_slice, V_x.conj().T, eta=eta)
    gR, _ = get_surface_gf(E, H_slice, V_x, eta=eta)
    
    if symmetric:
        U_L = phase_matrix(phi/2, N_y=N_y)
        U_R = phase_matrix(-phi/2, N_y=N_y)
    else:
        U_L = np.eye(4 * N_y, dtype=np.complex128)
        U_R = phase_matrix(-phi, N_y=N_y)
    
    g_L_phased = U_L @ gL @ U_L.conj().T
    g_R_phased = U_R @ gR @ U_R.conj().T
    
    return g_L_phased, g_R_phased


def get_surface_gfs_2d_phased(energy, H_slice, V_x, N_y, phi, symmetric=False, eta=1e-4):
    """2D surface GF with phase rotation applied."""
    gL_1d, _ = get_surface_gf(energy, H_slice, V_x.conj().T, eta=eta)
    gR_1d, _ = get_surface_gf(energy, H_slice, V_x, eta=eta)
    if symmetric:
        U_L = phase_matrix(phi/2, N_y=N_y)
        U_R = phase_matrix(-phi/2, N_y=N_y)
    else:
        U_L = np.eye(4 * N_y, dtype=np.complex128)
        U_R = phase_matrix(-phi, N_y=N_y)
    
    g_L = U_L @ gL_1d @ U_L.conj().T
    g_R = U_R @ gR_1d @ U_R.conj().T    
    
    return g_L, g_R
