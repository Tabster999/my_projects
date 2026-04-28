#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy import stats as stats
from scipy import optimize as optimize
from typing import Callable, List, Tuple, Union, Any, Optional, Dict
from scipy.linalg import inv


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
                       [delta, 0, -2*t + mu-h,0], 
                       [0, -delta, 0, -2*t + mu + h]], dtype=np.complex128)
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
    Gf_stack = np.linalg.solve(matrix_stack,np.eye(rows, dtype=hamiltonian.dtype))
    
    return Gf_stack

def quick_plot(
    x_values: np.array, y_list: np.array,
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
    denominator = 1 / (energy_array[:, None] - evals[None, :] + zenr)
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


def load_parameters(t=1.0, mu=0.025, alpha=.4, delta = .1, h = .2, sites = 150, steps=81, eta=1e-4):
    """
    Initializes and returns a dictionary of simulation parameters.

    Args:
        t (float): Base hopping energy, typically set to 1.0 to define the unit of energy.
        mu (float): Chemical potential (mu).
        alpha (float): Spin-orbit coupling strength.
        delta (float): Induced superconducting gap.
        h (float): Zeeman field strength .
        sites (int): Number of lattice sites in the system (L).
        steps (int): Number of points in the arrays (energy steps, h steps).

    Returns:
        dict: A dictionary containing all initialized parameters, including:
            - 'sites' (int): Number of sites.
            - 't', 'mu', 'alpha', 'delta', 'h' (float): Absolute energy values.
            - 'eta' (float): Small broadening parameter (1e-4 * delta).
            - 'phase_transition' (float): sqrt{\Delta^2 + \mu^2}.
            - 'e_min', 'e_max' (float): Energy array limits.
            - 'steps' (int): Number of energy steps.
            - 'energy_array' (np.ndarray): 1D array of energies. Shape: (steps,).
    """
    phase_transition = np.sqrt(mu**2 + delta**2)

    energy_array = np.linspace(-t,t,steps)
    params = {
        'sites' : sites,
        't' : t,
        'mu' : mu,
        'alpha' : alpha,
        'h' : h,
        'delta' : delta,
        'eta' : eta,
        'phase_transition' : phase_transition,
        'energy_array' : energy_array
    }

    return params 


def Iterator_retarded(energy:float, eps:np.ndarray(4), t_matrix:np.ndarray(4), eta:float) -> np.ndarray(4,dtype=complex): # type: ignore
    '''
    Iteration that returns the Gs(retarded surface Green-function) and Gb(retarded bulk Green-function) as a complex 4x4 matrix
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

def Iterator_advanced(energy:float, eps:np.ndarray(4), t_matrix:np.ndarray(4), eta:float) -> np.ndarray(4,dtype=complex): # type: ignore
    '''
    Iteration that returns the Gs(advanced surface Green-function) and Gb(advanced bulk Green-function) as a complex 4x4 matrix
    -energy: Onsite energy 
    -eps: onsite Hamiltonian
    -t_matrix: hoppping matrix
    '''
    z = (energy - 1j*eta) * np.eye(4)   
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
        if np.linalg.norm(alpha) < 1e-10:
            break
    Gs = np.linalg.inv((z - Epsilon_surf))
    Gb = np.linalg.inv((z - Epsilon_bulk))

    return Gs,Gb
# %%
