"""
Building blocks for the 4x4 Nambu-spin basis Hamiltonians.

Basis: Psi = (c_up, c_down, c_down^dagger, -c_up^dagger)  (Scharf-Pientka convention)
In this basis the BdG Hamiltonian reads  H = [[h, D], [D^dag, -s_y h^*(-k) s_y]],
so a term X (x) tau_z flips sign in the hole block and X (x) tau_0 does not.

Real-space convention (both models): a hopping block V is H_{j, j+1}, i.e. the
matrix element from site j to its +x (+y) neighbour, so that
    H(k) = onsite + V e^{ik} + V^dag e^{-ik}.

Two models share this basis, so geometry, leads and solvers are model-agnostic:
  * 'rashba' : onsite_block / Vx / Vy            (2DEG + SOC + Zeeman + s-wave)
  * 'dirac'  : dirac_onsite_block / dirac_Vx / dirac_Vy
               (lattice Dirac-BdG with Wilson mass, Gresta et al. Eq. (1))
Use model_onsite / model_hop to get the right block for a Params instance.
"""

import numpy as np


# =============================================================================
# Model 1: Rashba 2DEG
# =============================================================================
def onsite_block(t, mu, delta=0.0, phi=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0,
                  alpha=0.0, beta=0.0, twod=False):
    """4x4 onsite block: kinetic energy, SC pairing, and Zeeman terms."""
    ons = np.zeros((4, 4), dtype=complex)
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


_SX2 = np.array([[0, 1], [1, 0]], dtype=complex)
_SY2 = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _soc_hop(t, cx, cy):
    """
    4x4 hopping block whose continuum limit is H_soc(k) = sin(k) * (cx s_x + cy s_y).
    Electron SOC block M = -(i/2)(cx s_x + cy s_y); the hole block is -M (BdG basis
    (c_up, c_dn, c_dn^dag, -c_up^dag)).
    """
    M = -0.5j * (cx * _SX2 + cy * _SY2)
    hop = np.diag([-t, -t, t, t]).astype(complex)
    hop[:2, :2] += M
    hop[2:, 2:] -= M
    return hop


def Vx(t, alpha=0.0, beta=0.0, soc_axis='110'):
    """
    4x4 hopping block along x (along the S/N interfaces), including SOC.

    soc_axis='110' (default, unchanged):  H_soc = (alpha - beta) sin(kx) s_y
    soc_axis='100':                       H_soc = -(alpha s_x + beta s_y) sin(kx)

    These are the two crystallographic setups of Scharf et al. (PRB 99, 214503), Eqs. (17)
    and (1), written in this package's geometry (x along the interfaces, y across them).
    """
    if soc_axis == '110':
        return _soc_hop(t, 0.0, alpha - beta)
    if soc_axis == '100':
        return _soc_hop(t, -alpha, -beta)
    raise ValueError(f"soc_axis must be '100' or '110', got {soc_axis!r}")


def Vy(t, alpha=0.0, beta=0.0, soc_axis='110'):
    """
    4x4 hopping block along y (across the junction), including SOC.

    soc_axis='110' (default, unchanged):  H_soc = -(alpha + beta) sin(ky) s_x
    soc_axis='100':                       H_soc = (alpha s_y + beta s_x) sin(ky)
    """
    if soc_axis == '110':
        return _soc_hop(t, -(alpha + beta), 0.0)
    if soc_axis == '100':
        return _soc_hop(t, beta, alpha)
    raise ValueError(f"soc_axis must be '100' or '110', got {soc_axis!r}")


# =============================================================================
# Model 2: lattice Dirac-BdG with Wilson mass (Gresta et al., Eq. (1))
#   H(k) = sum_{a=x,y} t sin(k_a) s_a tz + [Z + m(k)] sz t0 - mu s0 tz + Delta s0 t_phi
#   m(k) = m0 (2 - cos kx - cos ky)
# Fourier-decomposed with the convention above:
#   onsite : (Z + 2 m0) sz t0 - mu tz + Delta t_phi   (+ optional in-plane Zeeman)
#   V_a    : -(i t / 2) s_a tz - (m0 / 2) sz t0
# =============================================================================
_S0 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_T0 = _S0
_TZ = _SZ


def dirac_onsite_block(mu, m0, delta=0.0, phi=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0, twod=True):
    """
    4x4 onsite block of the Dirac-BdG model.  Bz is the Zeeman term Z (s_z t_0),
    identical in convention to onsite_block.  twod=False gives the 1D Wilson
    mass m0 (1 - cos k) (onsite m0 instead of 2 m0), used for single-chain leads.
    """
    mass = (2.0 if twod else 1.0) * m0 + Bz
    pairing = delta * np.exp(1j * phi)
    Bxy_c = Bxy * np.exp(1j * theta_z)

    ons = np.zeros((4, 4), dtype=complex)
    ons[0, 0] = mass - mu
    ons[1, 1] = -mass - mu
    ons[2, 2] = mass + mu
    ons[3, 3] = -mass + mu

    ons[0, 1] = Bxy_c
    ons[1, 0] = np.conj(Bxy_c)
    ons[2, 3] = Bxy_c
    ons[3, 2] = np.conj(Bxy_c)

    ons[0, 2] = pairing
    ons[1, 3] = pairing
    ons[2, 0] = pairing.conj()
    ons[3, 1] = pairing.conj()
    return ons


def dirac_Vx(t, m0):
    """H_{j,j+1} along x:  -(i t/2) s_x tz - (m0/2) s_z t0."""
    return -0.5j * t * np.kron(_TZ, _SX) - 0.5 * m0 * np.kron(_T0, _SZ)


def dirac_Vy(t, m0):
    """H_{j,j+1} along y:  -(i t/2) s_y tz - (m0/2) s_z t0."""
    return -0.5j * t * np.kron(_TZ, _SY) - 0.5 * m0 * np.kron(_T0, _SZ)


# =============================================================================
# Model dispatch: the ONLY place that knows which model is active
# =============================================================================
def region_alpha(p, region):
    """Rashba coupling of region 'c', 'n' or 's' (per-region override or the global alpha)."""
    a = {'c': p.alpha_c, 'n': p.alpha_n, 's': p.alpha_s}[region]
    return p.alpha if a is None else a


def bond_alpha(p, ra, rb):
    """Rashba coupling on a bond joining two regions (arithmetic mean, as for the Wilson mass)."""
    return 0.5 * (region_alpha(p, ra) + region_alpha(p, rb))


def region_params(p, region):
    """Physical parameters of region 'c' (central), 'n' (normal leads) or 's' (SC ribbons)."""
    if region == 'c':
        return dict(t=p.t_c, mu=p.mu_c, delta=0.0, Bz=p.Bz, Bxy=p.Bxy if p.Bxy_c is None else p.Bxy_c,
                    theta_z=p.theta_z, m0=p.m0_c)
    if region == 'n':
        return dict(t=p.t_n, mu=p.mu_n, delta=0.0, Bz=p.Bz if p.Bz_n is None else p.Bz_n, Bxy=p.Bxy if p.Bxy_n is None else p.Bxy_n,
                    theta_z=p.theta_z_n, m0=p.m0_n)
    if region == 's':
        return dict(t=p.t_s, mu=p.mu_s, delta=p.delta, Bz=p.Bz_s,
                    Bxy=p.Bxy if p.Bxy_s is None else p.Bxy_s, theta_z=p.theta_z, m0=p.m0)
    raise ValueError(f"unknown region {region!r}")


def model_onsite(p, region, phi=0.0, twod=True, **override):
    """Onsite 4x4 block of `region` for the model selected by p.model."""
    r = region_params(p, region)
    r.update(override)
    if p.model == 'rashba':
        return onsite_block(r['t'], r['mu'], delta=r['delta'], phi=phi, Bz=r['Bz'], Bxy=r['Bxy'],
                            theta_z=r['theta_z'], alpha=region_alpha(p, region), beta=p.beta, twod=twod)
    return dirac_onsite_block(r['mu'], r['m0'], delta=r['delta'], phi=phi, Bz=r['Bz'], Bxy=r['Bxy'],
                              theta_z=r['theta_z'], twod=twod)


def model_hop(p, direction, t, m0, alpha=None):
    """
    Hopping block H_{j,j+1} along `direction` ('x' or 'y') with amplitude t.
    m0 is the Wilson mass of the bond (ignored by the Rashba model).
    """
    if p.model == 'rashba':
        a = p.alpha if alpha is None else alpha
        return (Vx(t, a, p.beta, p.soc_axis) if direction == 'x'
                else Vy(t, a, p.beta, p.soc_axis))
    return dirac_Vx(t, m0) if direction == 'x' else dirac_Vy(t, m0)


def bond_m0(m0_a, m0_b):
    """Wilson mass on a bond joining two regions (arithmetic mean; exact for uniform m0)."""
    return 0.5 * (m0_a + m0_b)


# =============================================================================
# Lattice assembly (model-agnostic)
# =============================================================================
def make_row_hamiltonian(n, onsite, hop_x):
    """Build a 1D chain of n sites (dim = 4n) with the given onsite and x-hopping block."""
    dim = 4 * n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = onsite
    for i in range(n - 1):
        H[4 * i:4 * i + 4, 4 * i + 4:4 * i + 8] = hop_x
        H[4 * i + 4:4 * i + 8, 4 * i:4 * i + 4] = hop_x.conj().T
    return H


def get_2d_hamiltonian(nx, ny, onsite, Vx_block, Vy_block, dof=4):
    """Build a finite nx x ny 2D lattice Hamiltonian (dim = dof*nx*ny)."""
    dim = dof * nx * ny
    H = np.zeros((dim, dim), dtype=complex)

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
