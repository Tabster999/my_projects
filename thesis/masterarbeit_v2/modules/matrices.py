"""
hamiltonians.py - Hamiltonian builders and matrix helpers
=========================================================

This module contains the Hamiltonian-related functions copied from the original
`my_functions.py` in a minimal and reusable form.
"""

import numpy as np
from typing import List, Tuple


def onsite_matrix(t, mu, h, delta, twod=False):

    z = 4*t - mu if twod else 2*t - mu

    return np.array([
        [ z - h,                  0,              delta,                 0],
        [0,                       z + h,          0,                    -delta],

        [delta.conjugate(),       0,             -z - h,                0],
        [0,               -delta.conjugate(),     0,                   -z + h]
    ], dtype=np.complex128)



def t_matrix_x(t: float, alpha: float) -> np.ndarray:
    '''
    Rashba hopping in y-direction (intra-slice).
    Returns the matrix [[-t, alpha, 0, 0],[-alpha, -t, 0, 0],[0, 0, t, alpha],[0, 0, -alpha, t]]
    t: hopping parameter
    alpha: rashba soc 
    '''
    matrix = np.array([
        [-t,    alpha,  0,      0],
        [-alpha, -t,    0,      0],
        [0,      0,     t,      alpha],
        [0,      0,    -alpha,  t]
    ], dtype=np.complex128)
    return matrix

def t_matrix_y(t: float, alpha: float):
    '''
    Rashba hopping in x-direction (inter-slice).
    Returns the matrix [[-t, -1j*alpha, 0, 0],[1j*alpha, -t, 0, 0],[0, 0, t, -1j*alpha],[0, 0, 1j*alpha, t]]
    t: hopping parameter
    alpha: rashba soc 
    '''
    matrix = np.array([
        [-t,    1j*alpha,  0,      0],
        [1j*alpha, -t,    0,      0],
        [0,      0,     t,      1j*alpha],
        [0,      0,    1j*alpha,  t]
    ], dtype=np.complex128)
    return matrix

def get_tb_hamiltonian(h0_matrix: np.ndarray, hopping_matrix: np.ndarray, sites: int) -> np.ndarray:
    """Build a general tight-binding Hamiltonian from onsite and hopping blocks."""
    dtype = h0_matrix.dtype
    id_matrix = np.eye(sites, dtype=dtype)
    off_diag_1 = np.eye(sites, k=1, dtype=dtype)
    off_diag_2 = np.eye(sites, k=-1, dtype=dtype)

    H_0 = np.kron(id_matrix, h0_matrix)
    T_1 = np.kron(off_diag_1, hopping_matrix)
    T_2 = np.kron(off_diag_2, np.conjugate(hopping_matrix.T))
    H = H_0 + T_1 + T_2
    return H


def build_sns_junction(
    t: float,
    mu_sc: float,
    mu_m: float,
    alpha: float,
    h: float,
    delta: complex,
    phi: float,
    sites_left: int,
    sites_right: int,
    sites_mid: int,
    dof: int,
    symmetric: bool = False
) -> np.ndarray:
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
    V = t_matrix_x(t, alpha)

    phi_L = phi / 2 if symmetric else 0.0
    phi_R = -phi / 2 if symmetric else -phi

    for i in range(sites_tot):
        idx = i * dof
        if i < sites_left:
            H_site = onsite_matrix(t, mu_sc, h, delta * np.exp(1j * phi_L))
        elif i < sites_left + sites_mid:
            H_site = onsite_matrix(t, mu_m, h, 0)
        else:
            H_site = onsite_matrix(t, mu_sc, h, delta * np.exp(1j * phi_R))

        H_tot[idx:idx + dof, idx:idx + dof] = H_site

        if i < sites_tot - 1:
            aux_idx = (i + 1) * dof
            H_tot[idx:idx + dof, aux_idx:aux_idx + dof] = V
            H_tot[aux_idx:aux_idx + dof, idx:idx + dof] = V.conj().T

    return H_tot


def build_sns_junction_sliced(
    t: float,
    mu_sc: float,
    mu_m: float,
    alpha: float,
    h: float,
    delta: complex,
    phi: float,
    sites_left: int,
    sites_right: int,
    sites_mid: int,
    symmetric: bool = False
) -> Tuple[List[np.ndarray], np.ndarray]:
    r"""

    Builds a 1d SNS-junction Hamiltonian using following matrices
    -onsite: 
        H_0 = [[2*t - mu-h, 0, delta, 0]
        [0, 2*t - mu+h, 0, -delta]
        [delta, 0, -2*t + mu-h,0]
        [0, -delta, 0, -2*t + mu + h]]
        
    hopping in y-direction: 
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
        symmetric:bool, chooses gauge    
    Returns: np.ndarray of dimensions d = dof * (sites_left + sites_right + site_mid).
    """
    phi_L = phi / 2 if symmetric else 0.0
    phi_R = -phi / 2 if symmetric else -phi

    H_L = onsite_matrix(t, mu_sc, h, delta * np.exp(1j * phi_L))
    H_R = onsite_matrix(t, mu_sc, h, delta * np.exp(1j * phi_R))
    H_M = onsite_matrix(t, mu_m, h, 0)
    V = t_matrix_y(t, alpha)

    H_slices = [H_L for _ in range(sites_left)] + [H_M for _ in range(sites_mid)] + [H_R for _ in range(sites_right)]
    return H_slices, V


def build_middle_region(t: float, mu_m: float, alpha: float, h: float, sites_mid: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build the middle-region Hamiltonian slices for RGF approaches."""
    H_N = onsite_matrix(t, mu_m, h, 0)
    V = t_matrix_y(t, alpha)
    H_slices = [H_N.copy() for _ in range(sites_mid)]
    return H_slices, V



