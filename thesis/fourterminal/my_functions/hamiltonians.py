"""
Building blocks for the 4x4 Nambu-spin basis Hamiltonians.

Basis: Psi = (c_up, c_down, c_down^dagger, -c_up^dagger)  (Scharf-Pientka convention)
"""

import numpy as np


def onsite_block(t, mu, delta=0.0, phi=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0,
                  alpha=0.0, beta=0.0, twod=False):
    """4x4 onsite block: kinetic energy, SC pairing, and Zeeman terms."""
    ons = np.zeros((4, 4), dtype=np.complex128)
    onsite_val = ((4 * t - mu) if twod else (2 * t - mu)) + (alpha**2 + beta**2) / 4
    pairing = delta * np.exp(1j * phi)
    Bxy_c = Bxy * np.exp(1j * theta_z)

    ons[0, 0] = onsite_val + Bz
    ons[1, 1] = onsite_val - Bz
    ons[2, 2] = -onsite_val + Bz
    ons[3, 3] = -onsite_val - Bz

    ons[0, 1] = Bxy_c
    ons[1, 0] = np.conj(Bxy_c)
    ons[2, 3] = Bxy_c
    ons[3, 2] = np.conj(Bxy_c)

    ons[0, 2] = pairing
    ons[1, 3] = pairing
    ons[2, 0] = pairing.conj()
    ons[3, 1] = pairing.conj()
    return ons


def Vx(t, alpha=0.0, beta=0.0):
    """4x4 hopping block along x (includes Rashba/Dresselhaus SOC)."""
    s = 0.5 * (alpha - beta)
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    hop[0, 1] = -s
    hop[1, 0] = s
    hop[2, 3] = s
    hop[3, 2] = -s
    return hop


def Vy(t, alpha=0.0, beta=0.0):
    """4x4 hopping block along y (includes Rashba/Dresselhaus SOC)."""
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array(
        [[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
        dtype=np.complex128,
    ) * (alpha + beta) / 2
    return hop + soc


def make_row_hamiltonian(n, onsite, hop_x):
    """Build a 1D chain of n sites (dim = 4n) with the given onsite and x-hopping block."""
    dim = 4 * n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = onsite
    for i in range(n - 1):
        H[4 * i:4 * i + 4, 4 * i + 4:4 * i + 8] = hop_x
        H[4 * i + 4:4 * i + 8, 4 * i:4 * i + 4] = hop_x.conj().T
    return H


def get_2d_hamiltonian(nx, ny, onsite, Vx_block, Vy_block, dof=4):
    """Build a finite nx x ny 2D lattice Hamiltonian (dim = dof*nx*ny)."""
    dim = dof * nx * ny
    H = np.zeros((dim, dim), dtype=np.complex128)

    def idx(ix, iy):
        return ix + nx * iy

    for iy in range(ny):
        for ix in range(nx):
            j = idx(ix, iy)
            H[dof * j:dof * (j + 1), dof * j:dof * (j + 1)] = onsite

    for iy in range(ny):
        for ix in range(nx - 1):
            i, j = idx(ix, iy), idx(ix + 1, iy)
            H[dof * i:dof * (i + 1), dof * j:dof * (j + 1)] = Vx_block
            H[dof * j:dof * (j + 1), dof * i:dof * (i + 1)] = Vx_block.conj().T

    for iy in range(ny - 1):
        for ix in range(nx):
            i, j = idx(ix, iy), idx(ix, iy + 1)
            H[dof * i:dof * (i + 1), dof * j:dof * (j + 1)] = Vy_block
            H[dof * j:dof * (j + 1), dof * i:dof * (i + 1)] = Vy_block.conj().T

    return H


def flat_site_idx(sites, dof=4):
    """
    Expand a list of physical site indices (convention: site = ix + nx*iy)
    into the corresponding flat matrix-element indices.
    """
    return np.concatenate([np.arange(dof * s, dof * (s + 1)) for s in sites])
