"""
matrices.py - Hamiltonian builders and matrix helpers
======================================================

Contains Hamiltonian-related functions for both 1D and 2D SNS junctions.


Default basis: Ψ = (c↑, c↓, c↓†, −c↑†)
----------------------------------------
This is the standard basis convention for this project going forward.
The builders below carrying this basis have no special suffix:
onsite_energy, onsite_matrix_sc, onsite_matrix_normal, hopping_y,
hopping_x, tile_block_diagonal, tile_transverse_hopping, make_slice_sc,
make_slice_normal, make_x_hopping.


Legacy plain-Nambu basis (suffix ``_plain``)
---------------------------------------------
An earlier, different basis convention Ψ = (c↑, c↓, c↓†, c↑†) (no minus
sign on the last component) is used by an older family of builders, kept
around for backward compatibility and suffixed ``_plain`` throughout:


------------
1D functions:
    onsite_matrix_plain          : on-site Nambu Hamiltonian block
    t_matrix_x_plain             : x-direction Rashba hopping (σ_y, p_x)
    t_matrix_y_plain             : y-direction Rashba hopping (σ_x, p_y)
    get_tb_hamiltonian           : generic tight-binding assembler (basis-agnostic)
    build_sns_junction_plain     : full finite 1D SNS Hamiltonian (dense)
    build_sns_junction_sliced_plain : 1D SNS as slice list + hopping (for RGF)
    build_middle_region_plain    : middle-region slices only (1D)

    
------------
2D functions:
    build_sns_slice_2d_plain         : within-slice Hamiltonian (y-chain, σ_x, p_y hops)
    build_sns_junction_sliced_2d_plain : full 2D SNS slice list + x-hopping (σ_y, p_x)
    build_sns_normal_only_2d_plain   : normal-region slices only (2D, for RGF with surface GFs)
    get_lead_slice_2d_plain          : unphased SC lead slice (real delta)

    
-------------------
Geometry convention:
    x-direction : transport / RGF-recursive direction (slice index)
    y-direction : transverse direction within each slice

    within a slice  →  t_matrix_y_plain  (σ_x, p_y Rashba)
    between slices  →  t_matrix_x_plain  (σ_y, p_x Rashba)
"""

import numpy as np
from typing import List, Tuple


# ============================================================================
# ELEMENTARY BUILDING BLOCKS (plain Nambu basis)
# ============================================================================

def onsite_matrix_plain(t: float, mu: float, h: float, delta: complex, twod: bool = False) -> np.ndarray:
    """
    On-site Nambu Hamiltonian block in basis (c↑, c↓, c↓†, c↑†).

    Args:
        t     : hopping amplitude (sets band offset 2t − μ or 4t − μ in 2D)
        mu    : chemical potential
        h     : Zeeman energy
        delta : complex superconducting pairing potential
        twod  : if True, use 4t − μ offset (2D square lattice); else 2t − μ (1D chain)

    Returns:
        (4, 4) complex Hamiltonian block
    """
    z = 4 * t - mu if twod else 2 * t - mu

    return np.array([
        [ z + h,               0,              delta,              0       ],
        [ 0,                   z - h,          0,                 delta   ],
        [ delta.conjugate(),   0,             -z + h,             0       ],
        [ 0,                  delta.conjugate(), 0,              -z - h  ]
    ], dtype=np.complex128)


def t_matrix_x_plain(t: float, alpha: float) -> np.ndarray:
    """
    Rashba hopping in the x-direction (inter-slice).

    Corresponds to σ_y,p_x spin-orbit coupling:
        T_x = -t·τ_z + α p_x·σ_y·τ_z

    Args:
        t     : hopping amplitude
        alpha : Rashba spin-orbit coupling strength

    Returns:
        (4, 4) complex hopping matrix
    """
    return np.array([
        [-t,     -alpha,  0,      0    ],
        [alpha, -t,     0,      0    ],
        [ 0,     0,      t,      alpha],
        [ 0,     0,     -alpha,  t    ]
    ], dtype=np.complex128)


def t_matrix_y_plain(t: float, alpha: float) -> np.ndarray:
    """
    Rashba hopping in the y-direction (intra-slice / transverse).

    Corresponds to σ_y,p_y spin-orbit coupling:
        T_y = -t·τ_z + i·α p_y·σ_x·τ_z

    Args:
        t     : hopping amplitude
        alpha : Rashba spin-orbit coupling strength

    Returns:
        (4, 4) complex hopping matrix
    """
    return np.array([
        [-t,        1j * alpha, 0,          0         ],
        [-1j * alpha, -t,        0,          0         ],
        [ 0,         0,         t,          -1j * alpha],
        [ 0,         0,        1j * alpha, t         ]
    ], dtype=np.complex128)


def get_tb_hamiltonian(h0_matrix: np.ndarray, hopping_matrix: np.ndarray, sites: int) -> np.ndarray:
    """
    Build a general tight-binding Hamiltonian from onsite and hopping blocks.

    H = Σ_i h0 ⊗ |i><i| + T ⊗ |i><i+1| + T† ⊗ |i+1><i|

    Args:
        h0_matrix      : (dof, dof) on-site block
        hopping_matrix : (dof, dof) nearest-neighbour hopping block
        sites          : number of sites

    Returns:
        (dof*sites, dof*sites) dense Hamiltonian
    """
    dtype = h0_matrix.dtype
    id_matrix  = np.eye(sites, dtype=dtype)
    off_diag_1 = np.eye(sites, k=1,  dtype=dtype)
    off_diag_2 = np.eye(sites, k=-1, dtype=dtype)

    H_0 = np.kron(id_matrix,  h0_matrix)
    T_1 = np.kron(off_diag_1, hopping_matrix)
    T_2 = np.kron(off_diag_2, hopping_matrix.conj().T)
    return H_0 + T_1 + T_2


def build_sns_junction_plain(
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
    dof: int = 4,
    symmetric: bool = False,
) -> np.ndarray:
    r"""
    Build the full dense 1D SNS-junction Hamiltonian.

    Ordering: [left SC | normal region | right SC]

    .. warning::
        This returns a single dense matrix; it cannot be used directly with
        the RGF routines.  Use ``build_sns_junction_sliced_plain`` for RGF.

    Args:
        t           : hopping amplitude
        mu_sc       : chemical potential in SC leads
        mu_m        : chemical potential in normal region
        alpha       : Rashba spin-orbit coupling strength
        h           : Zeeman energy
        delta       : SC pairing amplitude
        phi         : phase difference between the two SC leads
        sites_left  : number of sites in the left lead
        sites_right : number of sites in the right lead
        sites_mid   : number of sites in the normal region
        dof         : degrees of freedom per site (default 4, Nambu basis)
        symmetric   : if True use symmetric gauge ±φ/2; if False use 0/−φ

    Returns:
        (dof*(sites_left+sites_mid+sites_right), …) dense Hamiltonian
    """
    sites_tot = sites_left + sites_right + sites_mid
    N_tot = dof * sites_tot
    H_tot = np.zeros((N_tot, N_tot), dtype=np.complex128)
    V = t_matrix_x_plain(t, alpha)

    phi_L = -phi / 2  if symmetric else 0.0
    phi_R = phi / 2 if symmetric else phi

    for i in range(sites_tot):
        idx = i * dof
        if i < sites_left:
            H_site = onsite_matrix_plain(t, mu_sc, h, delta * np.exp(1j * phi_L))
        elif i < sites_left + sites_mid:
            H_site = onsite_matrix_plain(t, mu_m,  h, 0)
        else:
            H_site = onsite_matrix_plain(t, mu_sc, h, delta * np.exp(1j * phi_R))

        H_tot[idx:idx + dof, idx:idx + dof] = H_site

        if i < sites_tot - 1:
            aux = (i + 1) * dof
            H_tot[idx:idx + dof, aux:aux + dof] = V
            H_tot[aux:aux + dof, idx:idx + dof] = V.conj().T

    return H_tot


def build_sns_junction_sliced_plain(
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
    Build 1D SNS-junction as a list of on-site slices + the hopping matrix.

    Suitable for use with the RGF solvers.

    Args:
        t           : hopping amplitude
        mu_sc       : chemical potential in SC leads
        mu_m        : chemical potential in normal region
        alpha       : Rashba spin-orbit coupling strength
        h           : Zeeman energy
        delta       : SC pairing amplitude
        phi         : phase difference between the two SC leads
        sites_left  : number of sites in the left lead
        sites_right : number of sites in the right lead
        sites_mid   : number of sites in the normal region
        symmetric   : if True use symmetric gauge ±φ/2; if False use 0/−φ

    Returns:
        H_slices : list of (4, 4) on-site Hamiltonian blocks, length = total sites
        V        : (4, 4) inter-site hopping matrix
    """
    phi_L = -phi / 2  if symmetric else 0.0
    phi_R = phi / 2 if symmetric else phi

    H_L = onsite_matrix_plain(t, mu_sc, h, delta * np.exp(1j * phi_L))
    H_R = onsite_matrix_plain(t, mu_sc, h, delta * np.exp(1j * phi_R))
    H_M = onsite_matrix_plain(t, mu_m,  h, 0)
    V   = t_matrix_x_plain(t, alpha)

    H_slices = (
        [H_L for _ in range(sites_left)] +
        [H_M for _ in range(sites_mid)] +
        [H_R for _ in range(sites_right)]
    )
    return H_slices, V


def build_middle_region_plain(
    t: float,
    mu_m: float,
    alpha: float,
    h: float,
    sites_mid: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build the normal-region slices only (1D), for RGF with external surface GFs.

    Args:
        t         : hopping amplitude
        mu_m      : chemical potential in normal region
        alpha     : Rashba spin-orbit coupling strength
        h         : Zeeman energy
        sites_mid : number of sites in the normal region

    Returns:
        H_slices : list of (4, 4) on-site blocks
        V        : (4, 4) inter-site hopping matrix
    """
    H_N = onsite_matrix_plain(t, mu_m, h, 0)
    V   = t_matrix_x_plain(t, alpha)
    return [H_N.copy() for _ in range(sites_mid)], V



def build_sns_slice_2d_plain(
    N_y: int,
    t: float,
    mu: float,
    h: float,
    alpha: float,
    delta: complex
) -> np.ndarray:
    """
    Build the within-slice Hamiltonian for a 2D SNS junction.

    Each slice is a chain of N_y sites running in the y-direction, coupled
    by σ_y Rashba hops (``t_matrix_y_plain``).  The result is a
    (4·N_y) × (4·N_y) matrix.

    Args:
        N_y   : number of sites in the y-direction (transverse width)
        t     : hopping amplitude
        mu    : chemical potential
        h     : Zeeman energy
        alpha : Rashba spin-orbit coupling strength
        delta : complex SC pairing potential (can include SC phase)

    Returns:
        (4*N_y, 4*N_y) complex Hamiltonian slice
    """
    h_0     = onsite_matrix_plain(t, mu, h, delta)
    V_y     = t_matrix_y_plain(t, alpha)
    I_y     = np.eye(N_y, dtype=np.complex128)
    off_y   = np.eye(N_y, k=1, dtype=np.complex128)
    return (
        np.kron(I_y,      h_0)
      + np.kron(off_y,    V_y)
      + np.kron(off_y.T,  V_y.conj().T)
    )


def build_sns_junction_sliced_2d_plain(
    N_y: int,
    t: float,
    mu_sc: float,
    mu_m: float,
    h: float,
    alpha: float,
    delta: complex,
    phi: float,
    sites_left: int,
    sites_right: int,
    sites_mid: int,
    symmetric: bool = False
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build the 2D SNS junction as slice Hamiltonians + inter-slice hopping.

    Each slice is a (4·N_y) × (4·N_y) block representing a column of N_y
    sites.  Inter-slice hopping uses σ_y Rashba (t_matrix_x_plain), tiled
    over N_y sites. Intra-slice hopping uses σ_x Rashba (t_matrix_y_plain), included in the slice Hamiltonians.

    Gauge conventions (consistent with ``build_sns_junction_sliced_plain``):
        symmetric  → Δ_L = Δ·exp(−iφ/2),  Δ_R = Δ·exp(+iφ/2)
        asymmetric → Δ_L = Δ (real),       Δ_R = Δ·exp(+iφ)

    Args:
        N_y         : transverse width (y-direction sites per slice)
        t           : hopping amplitude
        mu_sc       : chemical potential in SC leads
        mu_m        : chemical potential in normal region
        h           : Zeeman energy
        alpha       : Rashba spin-orbit coupling strength
        delta       : SC pairing amplitude
        phi         : SC phase difference
        sites_left  : number of slices in the left SC lead
        sites_right : number of slices in the right SC lead
        sites_mid   : number of slices in the normal region
        symmetric   : gauge choice (see above)

    Returns:
        H_slices : list of (4*N_y, 4*N_y) slice Hamiltonians
        V_x_2d   : (4*N_y, 4*N_y) inter-slice hopping matrix
    """
    phi_L = -phi / 2 if symmetric else 0.0
    phi_R =  phi / 2 if symmetric else phi

    H_L = build_sns_slice_2d_plain(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_L))
    H_R = build_sns_slice_2d_plain(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_R))
    H_M = build_sns_slice_2d_plain(N_y, t, mu_m,  h, alpha, 0.0)

    V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), t_matrix_x_plain(t, alpha))

    H_slices = (
        [H_L.copy() for _ in range(sites_left)] +
        [H_M.copy() for _ in range(sites_mid)] +
        [H_R.copy() for _ in range(sites_right)]
    )
    return H_slices, V_x_2d


def build_sns_normal_only_2d_plain(
    N_y: int,
    t: float,
    mu_m: float,
    h: float,
    alpha: float,
    sites_mid: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build only the normal-region slices for a 2D SNS junction.

    Use this when the SC leads are represented by surface Green's functions
    (infinite leads via Sancho-López) rather than explicit slices.

    Args:
        N_y       : transverse width
        t         : hopping amplitude
        mu_m      : chemical potential in normal region
        h         : Zeeman energy
        alpha     : Rashba spin-orbit coupling strength
        sites_mid : number of slices in the normal region

    Returns:
        H_slices : list of (4*N_y, 4*N_y) normal-region slice Hamiltonians
        V_x_2d   : (4*N_y, 4*N_y) inter-slice hopping matrix
    """
    H_M    = build_sns_slice_2d_plain(N_y, t, mu_m, h, alpha, 0.0)
    V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), t_matrix_x_plain(t, alpha))
    return [H_M.copy() for _ in range(sites_mid)], V_x_2d


def get_lead_slice_2d_plain(
    N_y: int,
    t: float,
    mu_sc: float,
    h: float,
    alpha: float,
    delta: float
) -> np.ndarray:
    """
    Return an unphased SC lead slice (real delta, zero phase).

    Used as the template slice for ``get_surface_gfs_phased``, which
    applies the U(1) gauge rotation internally.

    Args:
        N_y   : transverse width
        t     : hopping amplitude
        mu_sc : chemical potential in SC lead
        h     : Zeeman energy
        alpha : Rashba spin-orbit coupling strength
        delta : SC pairing amplitude (real)

    Returns:
        (4*N_y, 4*N_y) lead Hamiltonian slice
    """
    return build_sns_slice_2d_plain(N_y, t, mu_sc, h, alpha, delta)


# ============================================================================
# DEFAULT BASIS BUILDERS
# Basis: Ψ = (c↑, c↓, c↓†, −c↑†)
#
# This is the standard basis for this project going forward. Decomposed
# into small reusable pieces (raw 4x4 blocks + generic tiling) instead of
# a monolithic make_H_SC/make_H_N, so the same pieces serve SC leads and
# normal regions, and the tiling utilities are reusable for the 4-terminal
# geometry (e.g. metal leads will need their own onsite block tiled with
# the same hopping_y/hopping_x, just mu -> mu_lead and no Zeeman/pairing).
#
# Verified numerically identical (bit-for-bit within 1e-15) to the
# original make_H_SC / make_H_N / _make_x_hopping / _make_transverse_hopping.
# ============================================================================

def onsite_energy(t: float, mu: float, alpha_t: float, beta_t: float, twod: bool = True) -> float:
    """
    Common SOC-renormalized band offset appearing in every SP-basis onsite
    block: (4t − μ) + (α² + β²)/4 in 2D, or (2t − μ) + (α² + β²)/4 in 1D.
    """
    base = 4 * t - mu if twod else 2 * t - mu
    return base + (alpha_t**2 + beta_t**2) / 4


def onsite_matrix_sc(t: float, mu: float, delta: complex, alpha_t: float, beta_t: float,
                         twod: bool = True) -> np.ndarray:
    """
    SC-lead on-site 4x4 block in the default basis (c↑, c↓, c↓†, −c↑†).

    Pairing enters at [0,2]/[1,3] (not [0,3]/[1,2] as in the plain Nambu
    basis) — this is the direct consequence of the minus sign on the last
    basis component.
    """
    val = onsite_energy(t, mu, alpha_t, beta_t, twod=twod)
    H = np.zeros((4, 4), dtype=np.complex128)
    H[0, 0] = val;  H[1, 1] = val
    H[2, 2] = -val; H[3, 3] = -val
    H[0, 2] = delta; H[1, 3] = delta
    H[2, 0] = np.conjugate(delta); H[3, 1] = np.conjugate(delta)
    return H


def onsite_matrix_normal(t: float, mu: float, E_par: float, theta_z: float, E_z: float,
                             alpha_t: float, beta_t: float, twod: bool = True) -> np.ndarray:
    """
    Normal-region on-site 4x4 block in the default basis (c↑, c↓, c↓†, −c↑†),
    with an in-plane Zeeman field E_par at azimuthal angle theta_z and an
    out-of-plane Zeeman field E_z.
    """
    val = onsite_energy(t, mu, alpha_t, beta_t, twod=twod)
    Bxy = E_par * np.exp(1j * theta_z)
    H = np.zeros((4, 4), dtype=np.complex128)
    H[0, 0] = val + E_z;  H[1, 1] = val - E_z
    H[2, 2] = -val + E_z; H[3, 3] = -val - E_z
    H[0, 1] = Bxy; H[1, 0] = np.conjugate(Bxy)
    H[2, 3] = Bxy; H[3, 2] = np.conjugate(Bxy)
    return H


def hopping_y(alpha_t: float, beta_t: float, t: float = 1.0) -> np.ndarray:
    """
    Raw 4x4 transverse (y-direction) hopping block, default basis.

    Tile along y with ``tile_transverse_hopping`` to build a full slice's
    intra-slice hopping.
    """
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([
        [0,   1j,  0,   0  ],
        [1j,  0,   0,   0  ],
        [0,   0,   0,  -1j ],
        [0,   0,  -1j,  0  ],
    ], dtype=np.complex128) * (alpha_t + beta_t) / 2
    return hop + soc


def hopping_x(alpha_t: float, beta_t: float, t: float = 1.0) -> np.ndarray:
    """
    Raw 4x4 longitudinal (x-direction, inter-slice) hopping block, default basis.

    Tile with ``tile_block_diagonal`` (block-diagonal over y-sites, since
    x-hopping doesn't mix different y-sites) to get the full inter-slice
    hopping matrix.
    """
    s = 0.5 * (alpha_t - beta_t)
    v = np.zeros((4, 4), dtype=np.complex128)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t
    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return v


def tile_block_diagonal(block: np.ndarray, N: int) -> np.ndarray:
    """N copies of a (dof, dof) block placed block-diagonally (no inter-site coupling)."""
    return np.kron(np.eye(N, dtype=np.complex128), block)


def tile_transverse_hopping(hop_block: np.ndarray, N_y: int) -> np.ndarray:
    """
    Build a full (dof*N_y, dof*N_y) nearest-neighbour y-hopping matrix
    (open boundary chain) from a raw (dof, dof) hopping block.
    """
    dof = hop_block.shape[0]
    dim = dof * N_y
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N_y - 1):
        H[dof*i:dof*(i+1), dof*(i+1):dof*(i+2)] = hop_block
        H[dof*(i+1):dof*(i+2), dof*i:dof*(i+1)] = hop_block.conj().T
    return H


def make_slice_sc(N_y: int, t: float, mu_sc: float, delta: complex,
                      alpha_t: float, beta_t: float) -> np.ndarray:
    """
    Full (4*N_y, 4*N_y) SC-lead slice Hamiltonian in the default basis
    (onsite + transverse hopping), for use as the bulk unit cell fed into
    ``solvers.get_surface_gf``.
    """
    onsite = onsite_matrix_sc(t, mu_sc, delta, alpha_t, beta_t, twod=True)
    H = tile_block_diagonal(onsite, N_y)
    H += tile_transverse_hopping(hopping_y(alpha_t, beta_t, t), N_y)
    return H


def make_slice_normal(N_y: int, t: float, mu_n: float, E_par: float, theta_z: float,
                          E_z: float, alpha_t: float, beta_t: float) -> np.ndarray:
    """
    Full (4*N_y, 4*N_y) normal-region slice Hamiltonian in the default basis
    (onsite + transverse hopping), for use as an ``H_slices[i]`` entry in
    the RGF solvers.
    """
    onsite = onsite_matrix_normal(t, mu_n, E_par, theta_z, E_z, alpha_t, beta_t, twod=True)
    H = tile_block_diagonal(onsite, N_y)
    H += tile_transverse_hopping(hopping_y(alpha_t, beta_t, t), N_y)
    return H


def make_x_hopping(N_y: int, alpha_t: float, beta_t: float, t: float = 1.0) -> np.ndarray:
    """Full (4*N_y, 4*N_y) inter-slice (x-direction) hopping matrix, default basis."""
    return tile_block_diagonal(hopping_x(alpha_t, beta_t, t), N_y)