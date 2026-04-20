#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy import stats as stats
from scipy import optimize as optimize
from typing import Callable, List, Tuple, Union, Any, Optional, Dict
from scipy.linalg import inv
from scipy.linalg import block_diag

def get_surface_gf(energy:float, eps:np.ndarray(4), t_matrix:np.ndarray(4), eta:float) -> np.ndarray(4,dtype=complex): # type: ignore
    '''
    Sancho Lopéz algorithm to compute the surface and bulk greens functions for a semi-infinite system. Returns: Gs (surface Green-function) and Gb (retarded bulk Green-function) as complex np.ndarrays of dimensions 4x4.
    -energy: Onsite energy 
    -eps: onsite Hamiltonian
    -t_matrix: hoppping matrix
    '''
    z = (energy + 1j*eta) * np.eye(4, dtype=np.complex128)   
    alpha = t_matrix
    beta =  np.transpose(np.conjugate(t_matrix))
    Epsilon_surf = eps
    Epsilon_bulk = eps

    for i in range(500):
        aux = np.linalg.inv((z - Epsilon_bulk))
        Epsilon_surf = Epsilon_surf + alpha@aux@beta
        Epsilon_bulk = Epsilon_bulk + alpha@aux@beta + beta@aux@alpha
        alpha = alpha@aux@alpha
        beta = beta@aux@beta
        if np.linalg.norm(alpha) < 1e-10 :
            break
    Gs = np.linalg.inv((z - Epsilon_surf))
    Gb = np.linalg.inv((z - Epsilon_bulk))

    return Gs,Gb

def onsite_matrix(t:float, mu:float, h:float, delta:float):
    '''    up, down, down^dagger, up^dagger
    Returns the matrix [[2*t - mu-h, 0, delta, 0], [0, 2*t - mu+h, 0, -delta], [delta, 0, -2*t + mu-h,0], [0, -delta, 0, -2*t + mu + h]]
    alpha: Rashba soc
    h: Zeeman energy 
    onsite: 2t - mu
    delta: SC order parameter
    t: hopping parameter
    mu: chemical potential
    '''
    matrix = np.array([[2*t - mu-h, 0, delta, 0], 
                       [0, 2*t - mu+h, 0, -delta], 
                       [delta.conjugate(), 0, -2*t + mu-h,0], 
                       [0, -delta.conjugate(), 0, -2*t + mu + h]], dtype=np.complex128)
    return matrix

def t_matrix(t:float, alpha:float):
    '''
    Returns the matrix [[-t, alpha, 0, 0],[-alpha, -t, 0, 0],[0, 0, t, alpha],[0, 0, -alpha, t]]
    t: hopping parameter
    alpha: rashba soc 
    '''
    matrix = np.array([[-t, alpha, 0, 0],[-alpha, -t, 0, 0],[0, 0, t, alpha],[0, 0, -alpha, t]], dtype=np.complex128)
    return matrix

def tb_hamiltonian_1d(sites:int, t:float, mu:float, h:float, alpha:float, delta:float):
    """
    Create a #degrees of freedom x N square matrix representing the tight binding hamiltonian for a 1d lattice with N sites.
    
    Args:
        sites:int, number of sites N
        t:float, hopping parameter
        mu:float, chemical potential
        h:float, Zeeman energy 
        alpha:float, Rashba soc
        delta:float, superconducting energy gap
    
    Returns:
        np.ndarray of dimensions #degrees of freedom x N.
    
    """
    N = sites
    id_matrix = np.eye(N, dtype=np.complex128)
    off_diag_1 = np.eye(N, k=1, dtype=np.complex128)
    off_diag_2 = np.eye(N, k=-1, dtype=np.complex128)
    h_0 = onsite_matrix(t, mu, h, delta)
    V = t_matrix(t, alpha)
    
    H_0 = np.kron(id_matrix, h_0)
    
    T_1 = np.kron(off_diag_1, V)
    
    T_2 = np.kron(off_diag_2, np.conjugate(V.T))
    
    H = H_0 + T_1 + T_2 
    
    return H

def calc_G(energy:float, hamiltonian:np.ndarray, eta:float = 1e-4,ra:str='r') -> np.ndarray:
    """
    Calculates the retarded/advanced (str:'r'/'a') Green-function G(E_0) for a given input hamiltonian and single energy E_0.
    
    Args:
        energy:float, energy for which G(E) is computed
        hamiltonian_matrix: np.ndarray, the hamiltonian matrix for which Gf_r is calculated
        eta: float:default 1e-4, broadening parameter
        ra: str:default 'r', specifies if retarded ('r') or advanced ('a') Green function is calculated
    
    Returns:
        Gf : np.ndarray with same dimensionalities as the input hamiltonian_matrix.
        
    """
    if ra=='r':
        zenr = energy+ 1j*eta
    elif ra=='a':
        zenr = energy- 1j*eta
    else:
        raise ValueError("ra must be 'r' or 'a'")
    
    rows = hamiltonian.shape[0]
    Idmatrix = np.eye(rows, dtype=hamiltonian.dtype)
    #energy_plus_eta = energy + 1j*eta
    
    Gf = np.linalg.inv(zenr * Idmatrix - hamiltonian)
    return Gf

def get_G_energy(
    energy_array:np.ndarray,
    hamiltonian:np.ndarray,
    eta:float = 1e-4,
    ra:str = 'r')-> np.ndarray:
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

def quick_plot(
    x_values: np.ndarray, y_list: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[str, List[str]]] = None, # Allow a single string or a list
    h_lines: Optional[Union[List[float], float]] = None,
    v_lines: Optional[Union[List[float], float]] = None,
    h_line_styles: Optional[dict] = None,
    v_line_styles: Optional[dict] = None,
    xlabel='', ylabel='', title='',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize=(10, 7), dpi=150, save_path=None,
    grid=False, legend=None, show=True,
    **plot_kwargs
):
    """
    Creates a plot with pyplot.matplotlib plotting functions.
    ...
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    single_color_mode = isinstance(colors, str)

    for i, y_values in enumerate(y_list):
        current_kwargs = plot_kwargs.copy()
        
        # Set label
        if labels and i < len(labels):
            current_kwargs['label'] = labels[i]

        # Set color
        if single_color_mode:
            current_kwargs['color'] = colors
        elif isinstance(colors, list) and i < len(colors):
            current_kwargs['color'] = colors[i]

        ax.plot(x_values, y_values, **current_kwargs)
    
    # ... (rest of the function is identical)
    if h_line_styles is None:
        h_line_styles = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5}
    if v_line_styles is None:
        v_line_styles = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5}

    if h_lines is not None:
        if not isinstance(h_lines, list):
            h_lines = [h_lines]
        for y_pos in h_lines:
            ax.axhline(y_pos, **h_line_styles)

    if v_lines is not None:
        if not isinstance(v_lines, list):
            v_lines = [v_lines]
        for x_pos in v_lines:
            ax.axvline(x_pos, **v_line_styles)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend or (labels is not None and legend is not False):
        ax.legend()
    if grid:
        ax.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
        return None
    else:
        return fig, ax
    
def get_tb_hamiltonian(h0_matrix:np.ndarray, hopping_matrix:np.ndarray, sites:int):
    """
    Creates a tight binding hamiltonian in the usual form using the kronecker produkt (np.kron) with the onsite 
    and hopping matrices input by oniste_matrix and hopping_matrix.
    
    Args:
        h0_matrix: NxN np.array with N being the number of degrees of freedom
        t_matrix: NxN np.array
        sites: number of sites for which the hamiltonmian is created
    
    Returns:
        np.ndarray of dimensions #degrees of freedom x sites.
    """
    dtype = h0_matrix.dtype
    id_matrix = np.eye(sites, dtype=dtype)
    off_diag_1 = np.eye(sites, k=1, dtype=dtype)
    off_diag_2 = np.eye(sites, k=-1, dtype=dtype)
    
    H_0 = np.kron(id_matrix, h0_matrix)
    T_1 = np.kron(off_diag_1, hopping_matrix)
    T_2 = np.kron(off_diag_2, np.conjugate(hopping_matrix.T))
    
    H = H_0 + T_1 + T_2 
    
    return H
        
def calc_G_lehmann(energy_array:np.ndarray, evals:np.ndarray, evecs:np.ndarray, eta:float, ra:str='r'):
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
    N = len(evals)
    e_steps = len(energy_array)
    if ra=='r':
        zenr = 1j*eta
    elif ra=='a':
        zenr = -1j*eta
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

    # 1. Construct the k-dependent Hamiltonian stack H(k)
    # Shape: (k_steps, N, N)
    exp_ik = np.exp(1j * k_array)
    H_k_stack = (
        h0_matrix[np.newaxis, :, :] +
        hopping_matrix[np.newaxis, :, :] * exp_ik[:, np.newaxis, np.newaxis] +
        hopping_matrix.T.conj()[np.newaxis, :, :] * np.conj(exp_ik)[:, np.newaxis, np.newaxis]
    )

    # 2. Prepare for inversion using broadcasting
    N = h0_matrix.shape[0]
    Idmatrix = np.eye(N, dtype=h0_matrix.dtype)
    
    # Reshape arrays to enable broadcasting for (E, k, N, N)
    # zenr_array -> (1, E_steps, 1, 1)
    # H_k_stack  -> (k_steps, 1, N, N)
    # The result of the subtraction will be broadcast to (k_steps, E_steps, N, N)
    matrix_to_invert = (
        zenr_array[np.newaxis, :, np.newaxis, np.newaxis] * Idmatrix -
        H_k_stack[:, np.newaxis, :, :]
    )

    # 3. Perform the inversion over the stack
    Gf_stack = np.linalg.inv(matrix_to_invert)

    return Gf_stack

def build_sns_junction(t, mu_sc, mu_m, alpha, h, delta, phi, sites_left, sites_right, sites_mid, dof):

    r"""
    This function is used to build the full Hamiltonian for a finite size SNS junction. This can not be used for RGF calculations!
    Builds a 1d SNS-junction Hamiltonian using following matrices
    -onsite: 
        H_0 = [[2*t - mu-h, 0, delta, 0]
        [0, 2*t - mu+h, 0, -delta]
        [delta, 0, -2*t + mu-h,0]
        [0, -delta, 0, -2*t + mu + h]]
        
    hopping: 
        [[-t, alpha, 0, 0]
        [-alpha, -t, 0, 0]
        [0, 0, t, alpha]
        [0, 0, -alpha, t]]

    Args:
        t:float, hopping
        mu:float, chemical potential
        alpha:float, spin-orbit strenght
        h:float, zeeman energy
        delta:float, SC pairing 
        phi:float, phase difference between the SC leads
        sites_left:int, number of sites left lead
        sites_right:int, number of sites right lead
        sites_mid:int, number of sites middle lead
        dof:int, degrees of freedom per site (here Nambu $\Psi^\dagger = (up, down, down^dagger, up^dagger)^T$
    
    Returns: np.ndarray of dimensions d = dof * (sites_left + sites_right + site_mid).
    """
    sites_tot = sites_left + sites_right + sites_mid
    N_tot = dof * sites_tot
    H_tot = np.zeros((N_tot, N_tot), dtype=np.complex128)
    V = t_matrix(t, alpha)

    for i in range(sites_tot):
        idx = i*dof
        if i < sites_left: # left lead
            H_site = onsite_matrix(t, mu_sc, h, delta*np.exp(-1j * phi/2))
        elif i < sites_left + sites_mid: # middle region
            H_site = onsite_matrix(t, mu_m, h, 0)
        else: # right lead
            H_site = onsite_matrix(t, mu_sc, h, delta*np.exp(1j*phi/2))
    
        H_tot[idx:idx+dof, idx:idx+dof] = H_site

        if i < sites_tot - 1:
            aux_idx = (i+1) * dof
            H_tot[idx:idx+dof, aux_idx:aux_idx+dof] = V
            H_tot[aux_idx:aux_idx+dof, idx:idx+dof] = V.conj().T
    return H_tot


def get_dos_region(G_full, block_size, start_site, end_site):
    """
    Extracts the density of states (LDOS) for a specific region of sites.
    
    Args:
        G_full: np.ndarray, full Green's function
        block_size: int, degrees of freedom per site
        start_site: int, starting site index (0-indexed)
        end_site: int, ending site index (inclusive)
    
    Returns:
        float, the local density of states for the region
    """

    idx_start = start_site * block_size
    idx_end = (end_site + 1) * block_size
    G_region = G_full[idx_start:idx_end, idx_start:idx_end]
    dos = (-1/np.pi) * np.imag(np.trace(G_region))

    return dos


def self_energy(g, V, direction='left'):
    """Calculates the self-energy contribution from a lead using the surface Green's function and the coupling matrix.
    Args:
        g: np.ndarray, surface Green's function of the lead
        V: np.ndarray, coupling matrix between the lead and the central region
        direction: str, 'left' or 'right' for the lead direction
    Returns:
        np.ndarray, the self-energy contribution from the lead
    """
    if direction == 'left':
        Sigma = V.conj().T @ g @ V
    elif direction == 'right':
        Sigma = V @ g @ V.conj().T
    else:        
        raise ValueError("direction must be 'left' or 'right'")
    return Sigma


def build_middle_region(t, mu_m, alpha, h, sites_mid, dof):
    h0 = onsite_matrix(t, mu_m, h, 0)
    V = t_matrix(t, alpha)

    H = np.zeros((sites_mid * dof, sites_mid * dof), dtype=np.complex128)

    for i in range(sites_mid):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = h0

    for i in range(sites_mid - 1):
        idx_i = i * dof
        idx_j = (i + 1) * dof

        H[idx_i:idx_i+dof, idx_j:idx_j+dof] = V
        H[idx_j:idx_j+dof, idx_i:idx_i+dof] = V.conj().T

    return H
  

def get_rgf_sns(middle_hamiltonian, V, g_L, g_R, energy, eta=1e-5, ra:str='r', return_full=False):
    """
    RGF for SNS junction with infinite leads via recursive algorithm.
    The infinite leads are represented by surface Green's functions computed via Sancho-López.
    
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
    
    Returns:
        G_out: np.ndarray, Green's function (diagonal blocks or full matrix)
        GL: np.ndarray, local GFs from left sweep (sites x dof x dof)
        GR: np.ndarray, local GFs from right sweep (sites x dof x dof)
    """
    dim = middle_hamiltonian.shape[0]
    dof = V.shape[0]
    N = dim // dof    
    I = np.eye(dof, dtype=np.complex128)

    if ra == 'r':
        z = energy + 1j*eta
    elif ra == 'a':
        z = energy - 1j*eta
    else:
        raise ValueError("ra must be 'r' or 'a' for retarded or advanced Green's function")

    # 1.RIGHT-TO-LEFT SWEEP 
    GR = np.zeros((N, dof, dof), dtype=np.complex128)
    
    Sigma_R0 = V @ g_R @ V.conj().T
    h_last = middle_hamiltonian[(N-1)*dof:N*dof, (N-1)*dof:N*dof]
    GR[N-1] = inv(z*I - h_last - Sigma_R0)
    
    for i in range(N-2, -1, -1):
        h_i = middle_hamiltonian[i*dof:(i+1)*dof, i*dof:(i+1)*dof]
        Sigma_R = V @ GR[i+1] @ V.conj().T
        GR[i] = inv(z*I - h_i - Sigma_R)

    # 2. RECONSTRUCT FULL GF
    G_out = np.zeros((N*dof, N*dof), dtype=np.complex128)

    Sigma_L0 = V.conj().T @ g_L @ V
    G_out[0:dof, 0:dof] = inv(inv(GR[0]) - Sigma_L0)

    # Do Left -> Right sweep to fill diagonal blocks
    for i in range(1, N):
        G_prev = G_out[(i-1)*dof:i*dof, (i-1)*dof:i*dof]
        G_out[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = GR[i] + GR[i] @ V.conj().T @ G_prev @ V @ GR[i]

    # Off-diagonal blocks can be reconstructed if return_full is True
    if return_full:
        for i in range(N):
            # Fill upper triangle G[i, j] for j > i 
            for j in range(i+1, N):
                G_aux_1 = G_out[i*dof:(i+1)*dof, (j-1)*dof:j*dof]
                G_out[i*dof:(i+1)*dof, j*dof:(j+1)*dof] = G_aux_1 @ V @ GR[j]
            # Fill lower triangle G[j, i] for j > i
            for j in range(i+1, N):
                G_aux_2 = G_out[(j-1)*dof:j*dof, i*dof:(i+1)*dof]
                G_out[j*dof:(j+1)*dof, i*dof:(i+1)*dof] = GR[j] @ V.conj().T @ G_aux_2
    
    return G_out
    

def get_rgf_finite_system(full_hamiltonian, V, energy, eta=1e-5, ra:str='r', return_full=False):
    """
    Computes the RGF for a strictly finite system.
    
    Args:
        full_hamiltonian: The Hamiltonian of the entire finite system (L + Mid + R).
        V: The hopping matrix connecting the blocks.
        energy: Calculation energy.
        eta: Imaginary broadening.
        return_full: If True, returns the full N*dof x N*dof matrix.
    """
    dim = full_hamiltonian.shape[0]
    dof = V.shape[0]
    N = dim // dof  
    I = np.eye(dof, dtype=np.complex128)
    z = energy + 1j*eta if ra == 'r' else energy - 1j*eta

    # 1. LEFT-TO-RIGHT SWEEP
    GL = np.zeros((N, dof, dof), dtype=np.complex128)
    
    H00 = full_hamiltonian[0:dof, 0:dof]
    GL[0] = inv(z*I - H00)

    for i in range(1, N):
        Hii = full_hamiltonian[i*dof:(i+1)*dof, i*dof:(i+1)*dof]
        Sigma_L = V.conj().T @ GL[i-1] @ V
        GL[i] = inv(z*I - Hii - Sigma_L)

    # 2. RIGHT-TO-LEFT SWEEP
    GR = np.zeros((N, dof, dof), dtype=np.complex128)
    
    HNN = full_hamiltonian[(N-1)*dof : N*dof, (N-1)*dof : N*dof]
    GR[N-1] = inv(z*I - HNN)

    for i in range(N-2, -1, -1):
        Hii = full_hamiltonian[i*dof:(i+1)*dof, i*dof:(i+1)*dof]
        Sigma_R = V @ GR[i+1] @ V.conj().T
        GR[i] = inv(z*I - Hii - Sigma_R)

    # 3. CONSTRUCT DIAGONAL BLOCKS
    G_diag_blocks = np.zeros((N, dof, dof), dtype=np.complex128)
    for i in range(N):
        Hii = full_hamiltonian[i*dof:(i+1)*dof, i*dof:(i+1)*dof]
        
        if i == 0:
            S_L = np.zeros((dof, dof))
            S_R = V @ GR[i+1] @ V.conj().T
        elif i == N-1:
            S_L = V.conj().T @ GL[i-1] @ V
            S_R = np.zeros((dof, dof))
        else:
            S_L = V.conj().T @ GL[i-1] @ V
            S_R = V @ GR[i+1] @ V.conj().T
            
        G_diag_blocks[i] = inv(z*I - Hii - S_L - S_R)
    
    if not return_full:
        return G_diag_blocks, GL, GR

    # 4. FULL MATRIX RECONSTRUCTION
    G_full = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        G_full[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = G_diag_blocks[i]
    
    # Off-diagonals using the Recursive identity: G_ij = G_ii * V * GR_jj (or similar)
    for i in range(N-1):
        G_full[i*dof:(i+1)*dof, (i+1)*dof:(i+2)*dof] = G_diag_blocks[i] @ V @ GR[i+1]
        G_full[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof] = GR[i+1] @ V.conj().T @ G_diag_blocks[i]
        
    return G_full, GL, GR

        


# %%
