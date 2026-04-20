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

def build_sns_junction(t, mu, alpha, h, delta, phi, sites_left, sites_right, sites_mid, dof):

    r"""
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
    N_tot = dof * (sites_left + sites_right + sites_mid)
    H_tot = np.zeros((N_tot, N_tot), dtype=np.complex128)
    V = t_matrix(t, alpha)

    # indices labeling the blocks for L,M,R
    idx_L_end = (sites_left -1) * dof
    idx_M_start = sites_left * dof 
    idx_M_end = (sites_left + sites_mid - 1) * dof
    idx_R_start = (sites_left + sites_mid) * dof

    # Get left, middle and right hamiltonias
    H_L = get_tb_hamiltonian(onsite_matrix(t, mu, h, delta), t_matrix(t, alpha), sites_left) #creates left lead (\Delta = const)
    H_M = get_tb_hamiltonian(onsite_matrix(t, mu, h, 0), t_matrix(t, alpha), sites_mid) #creates middle lead (\Delta = 0)
    H_R = get_tb_hamiltonian(onsite_matrix(t, mu, h, delta*np.exp(1j*phi)), t_matrix(t, alpha), sites_right)
    #creates right lead (\Delta ~ exp(i*\phi))

    H = block_diag(H_L, H_M, H_R) #create the system of uncoupled leads

    #add coupling bwteen the leads via the hopping matrix V 
    #coupling between left lead and middle
    H[idx_L_end:idx_L_end + dof, idx_M_start:idx_M_start+dof] = V 
    H[idx_M_start: idx_M_start + dof, idx_L_end:idx_L_end + dof] = V.conj().T

    #coupling between middle lead and right lead
    H[idx_M_end:idx_M_end + dof, idx_R_start:idx_R_start+dof] = V 
    H[idx_R_start: idx_R_start + dof, idx_M_end:idx_M_end + dof] = V.conj().T

    return H

def get_rgf(block_size, hamiltonian_full, energy, eta=1e-4, ra='r', sweep='both'):
    """
    Calculates the full retarded / advanced Green's function G(E) using the Recursive Green's Function (RGF) method.
    Args:
        block_size:int, degrees of freedom per site
        hamiltonian_full:np.ndarray, full H of dim [sites * block_size, sites * block_size]
        energy:float, energy at which the RGF is calculated
        eta:float, small broadening parameter
        ra:str='r', specifies if retarded ('r') or advanced ('a') Green function is calculated.
        sweep:str='both', specifies sweep direction(s): 'left' for left sweep only, 'right' for right sweep only, or 'both' for both sweeps.

    Returns:
        G_full:np.ndarray, full Green's function of dim [sites * block_size, sites * block_size]
    """
    if ra=='r':
        z = energy + 1j*eta
    elif ra=='a':
        z = energy - 1j*eta
    else:
        raise ValueError("ra must be 'r' or 'a'")
    
    if sweep not in ['left', 'right', 'both']:
        raise ValueError("sweep must be 'left', 'right', or 'both'")
    
    dim_full = hamiltonian_full.shape[0]
    N = block_size 
    sites = dim_full // block_size
    I = np.eye(N, dtype=np.complex128)

    H_diag = [hamiltonian_full[i*N:(i+1)*N, i*N:(i+1)*N] for i in range(sites)] 
    V = [hamiltonian_full[i*N:(i+1)*N, (i+1)*N:(i+2)*N] for i in range(sites - 1)]

    g_L = [None] * sites
    Sigma_L_list = [np.zeros((N,N), dtype=np.complex128) for _ in range(sites)]
    
    if sweep in ['left', 'both']:
        g_L[0] = inv(z*I - H_diag[0])
        for i in range(1, sites):
            Sigma_L_list[i] = V[i-1].conj().T @ g_L[i-1] @ V[i-1]
            g_L[i] = inv(z*I - H_diag[i] - Sigma_L_list[i])

    g_R = [None] * sites
    
    if sweep in ['right', 'both']:
        g_R[sites - 1] = inv(z*I - H_diag[sites - 1])
        for i in range(sites - 2, -1, -1):
            Sigma_R = V[i] @ g_R[i+1] @ V[i].conj().T
            g_R[i] = inv(z*I - H_diag[i] - Sigma_R)

    G_full = np.zeros_like(hamiltonian_full, dtype=np.complex128)
    for i in range(sites):
        aux_idx = i * N 
        Sigma_L = Sigma_L_list[i] if sweep in ['left', 'both'] else np.zeros((N,N), dtype=np.complex128)
        Sigma_R = V[i] @ g_R[i+1] @ V[i].conj().T if (sweep in ['right', 'both'] and i < sites - 1) else np.zeros((N,N), dtype=np.complex128)
        G_ii = inv(z*I - H_diag[i] - Sigma_L - Sigma_R)
        G_full[aux_idx:aux_idx+N,aux_idx:aux_idx+N] = G_ii

    if sweep in ['left', 'both'] and sweep in ['right', 'both']:
        for d in range(1, sites):
            for i in range(sites - d):
                j = i + d
                idx_i = i * N
                idx_j = j * N
                idx_jm1 = (j - 1) * N

                G_full[idx_i:idx_i+N, idx_j:idx_j+N] = (G_full[idx_i:idx_i+N, idx_jm1:idx_jm1+N] @ V[j-1] @ g_R[j])
                G_full[idx_j:idx_j+N, idx_i:idx_i+N] = (g_R[j] @ V[j-1].conj().T @ G_full[idx_jm1:idx_jm1+N, idx_i:idx_i+N])        
    return G_full

def get_sns_rgf(energy, eta, dof, sweep='both', ra='r', t=None, mu=None, alpha=None, h=None, delta=None, phi=None, sites_left=None, sites_right=None, sites_mid=None, hamiltonian=None):
    r"""
    Calculates the full Green's function for the SNS junction using RGF method.
    
    Can be called in two ways:
    
    1. With Hamiltonian (faster for repeated energy evaluations):
        get_sns_rgf(energy, eta, dof, hamiltonian=H_sns)
    
    2. With parameters (builds Hamiltonian internally):
        get_sns_rgf(energy, eta, dof, t=t, mu=mu, alpha=alpha, h=h, delta=delta, phi=phi, 
                    sites_left=sites_l, sites_right=sites_r, sites_mid=sites_m)
    
    Args:
        energy:float, energy at which to evaluate the Green's function
        eta:float, broadening parameter
        dof:int, degrees of freedom per site
        sweep:str, sweep direction: 'left', 'right', or 'both' (default)
        ra:str, retarded ('r') or advanced ('a') Green's function
        
        Either provide hamiltonian OR all of [t, mu, alpha, h, delta, phi, sites_left, sites_right, sites_mid]:
        t:float, hopping (required if hamiltonian is None)
        mu:float, chemical potential (required if hamiltonian is None)
        alpha:float, spin-orbit strength (required if hamiltonian is None)
        h:float, zeeman energy (required if hamiltonian is None)
        delta:float, SC pairing (required if hamiltonian is None)
        phi:float, phase difference between the SC leads (required if hamiltonian is None)
        sites_left:int, number of sites left lead (required if hamiltonian is None)
        sites_right:int, number of sites right lead (required if hamiltonian is None)
        sites_mid:int, number of sites middle lead (required if hamiltonian is None)
        hamiltonian:np.ndarray, pre-built SNS junction Hamiltonian (optional)
    
    Returns: Green's function as np.ndarray
    """
    if hamiltonian is None:
        if any(x is None for x in [t, mu, alpha, h, delta, phi, sites_left, sites_right, sites_mid]):
            raise ValueError("Either provide hamiltonian OR all SNS parameters (t, mu, alpha, h, delta, phi, sites_left, sites_right, sites_mid)")
        H_sns = build_sns_junction(t, mu, alpha, h, delta, phi, sites_left, sites_right, sites_mid, dof)
    else:
        H_sns = hamiltonian
    
    G = get_rgf(dof, H_sns, energy=energy, eta=eta, ra=ra, sweep=sweep)
    return G
# %%
