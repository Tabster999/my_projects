#%% Imports
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.constants as const
from joblib import Parallel, delayed
from tqdm import tqdm

#%% Parameters

ny     = 10
a      = 20e-9
normal = 10
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2       

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it    = 800
Delta = 0.25 / tunits
t     = 1.0

mu      = 2   / tunits
mu_n    = 0.1
mu_barr = 0.0 / tunits
theta   = 0.3 * np.pi / 2

eta  = 0.02 * Delta
nphi = 41
nw   = 41

phi_vals = np.linspace(0, 2*np.pi, nphi)
energies = np.linspace(-Delta, Delta, nw)

_I4 = np.eye(4 * ny, dtype=np.complex128)

#%% Hamiltonian Builders

def make_transverse_hopping(alpha, beta):
    """y-direction hopping + SOC block, shape (4*ny, 4*ny)."""
    hop_block = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([
        [0,  -1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,   1j ],
        [0,   0, -1j,  0  ]
    ], dtype=np.complex128) * (alpha + beta) / 2

    block = hop_block + soc
    blockH = hop_block + soc.conj().T

    hop = np.zeros((4*ny, 4*ny), dtype=np.complex128)

    for i in range(ny - 1):
        hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = block
        hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = blockH

    return hop


def make_block_diagonal(block_4x4):
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)


def Hloc_SC(alpha, beta):

    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.array([
        [ onsite_val,  0,          Delta, 0    ],
        [ 0,           onsite_val, 0,     Delta],
        [ Delta,       0,         -onsite_val, 0],
        [ 0,           Delta,      0,    -onsite_val]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)

    return H


def Hloc_N(Bz, Bxy, alpha, beta):

    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.array([
        [ onsite_val + Bz,              Bxy * np.exp( 1j*theta), 0, 0],
        [ Bxy * np.exp(-1j*theta),      onsite_val - Bz,         0, 0],
        [ 0, 0,  -(onsite_val + Bz),              Bxy * np.exp( 1j*theta)],
        [ 0, 0,   Bxy * np.exp(-1j*theta),  -(onsite_val - Bz)          ]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)

    return H


def Hloc_B(alpha, beta):

    onsite_val = 4*t - mu_barr
    onsite = np.diag([onsite_val, onsite_val, -onsite_val, -onsite_val]).astype(np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)

    return H


def V_mat(alpha, beta):

    s = 0.5 * (alpha - beta)
    v = np.array([
        [-t,  -s,  0,  0],
        [ s,  -t,  0,  0],
        [ 0,   0,  t,  s],
        [ 0,   0, -s,  t]
    ], dtype=np.complex128)

    return make_block_diagonal(v)


def apply_phase(G, phi):

    ep = np.exp( 1j * phi)
    em = np.exp(-1j * phi)
    G_out = G.copy()
    hole_idx = np.array([i for i in range(4*ny) if i % 4 >= 2])
    elec_idx = np.array([i for i in range(4*ny) if i % 4 < 2])
    G_out[np.ix_(hole_idx, elec_idx)] *= em
    G_out[np.ix_(hole_idx, hole_idx)] *= 1.0 
    G_out[np.ix_(elec_idx, hole_idx)] *= ep
    return G_out


def sancho(H, alpha_0, beta_0, w, eta):

    z     = (w + 1j*eta) * np.eye(H.shape[0], dtype=np.complex128)
    eps   = H.copy()
    eps_s = H.copy()
    alpha = alpha_0.copy()
    beta  = beta_0.copy()
    for _ in range(it):
        gi    = inv(z - eps)
        ab    = alpha @ gi @ beta
        ba    = beta  @ gi @ alpha
        eps_s = eps_s + ab
        eps   = eps   + ab + ba
        alpha = alpha @ gi @ alpha
        beta  = beta  @ gi @ beta
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < 1e-12:
            break

    return inv(z - eps_s)

#%% Green's Function Helpers

def _build_surface_greens(Hs, Hb, Hn, V0, Vd, w):
    """
    Compute gSL, gSR, glr given fixed w. These don't depend on phi,
    so we host them out of the phi loop.
    """
    z    = w + 1j*eta
    zI   = z * _I4
    gSL0 = sancho(Hs, Vd, V0, w, eta)
    gSR0 = sancho(Hs, V0, Vd, w, eta)
    gSL  = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR  = inv(zI - Hb - V0 @ gSR0 @ Vd)
    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    glr    = [None] * normal
    glr[0] = inv(zI - Hn - Vd @ gSL @ V0)
    for i in range(1, normal):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)

    return gSL, gSR, glr


def _phi_sweep_ldos(gSL, gSR, Hn, V0, Vd, w, spatial=False):
    """
    Inner phi loop. Returns:
      spatial=False : 1D array (nphi,) of total LDOS
      spatial=True  : 2D array (nphi, normal, ny) of site-resolved LDOS
    """
    z  = w + 1j*eta
    zI = z * _I4

    if spatial:
        result = np.zeros((nphi, normal, ny))
    else:
        result = np.zeros(nphi)

    for iphi, phi in enumerate(phi_vals):
        gSRp = apply_phase(gSR, phi)

        grl       = [None] * normal
        grl[-1]   = inv(zI - Hn - V0 @ gSRp @ Vd)
        for j in range(normal - 2, -1, -1):
            grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

        if spatial:
            for j in range(normal):
                SL = Vd @ grl[j-1] @ V0 if j > 0 else Vd @ gSL @ V0
                SR = V0 @ grl[j+1] @ Vd if j < normal-1 else V0 @ gSRp @ Vd
                G  = inv(zI - Hn - SL - SR)
                for iy in range(ny):
                    blk = G[4*iy:4*iy+4, 4*iy:4*iy+4]
                    result[iphi, j, iy] = -np.imag(np.trace(blk)) / np.pi
        else:
            LD = 0.0 + 0j
            for j in range(normal):
                SL = Vd @ grl[j-1] @ V0 if j > 0 else Vd @ gSL @ V0
                SR = V0 @ grl[j+1] @ Vd if j < normal-1 else V0 @ gSRp @ Vd
                G  = inv(zI - Hn - SL - SR)
                LD += np.trace(G)
            result[iphi] = -np.imag(LD) / np.pi

    return result

#%% Parallelized LDOS Sweeps

def _worker_w(w, Hs, Hb, Hn, V0, Vd):
    """One energy slice → row of shape (nphi,). Used by calculate_ldos_parallel."""
    gSL, gSR, glr = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w)
    return _phi_sweep_ldos(gSL, gSR, Hn, V0, Vd, w, spatial=False)


def calculate_ldos_parallel(B_field, alpha_raw, beta_raw, energy_array=None, n_jobs=-1):
    """
    LDOS(E, phi) — parallelized over energy.
    Returns array of shape (nw, nphi).
    """
    if energy_array is None:
        energy_array = energies
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = (0.5 * muB / tunits) * B_field
    Bxy   = (0.5 * muB * theta / tunits) * B_field 

    V0 = V_mat(alpha, beta);  Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta)
    Hn = Hloc_N(Bz, Bxy, alpha, beta)
    Hb = Hloc_B(alpha, beta)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_worker_w)(w, Hs, Hb, Hn, V0, Vd)
        for w in tqdm(energy_array, desc='LDOS(E,φ)  ')
    )
    return np.array(rows)   # (nw, nphi)


def _worker_B(B_field, alpha, beta, w):
    """One B-field slice at fixed w → row of shape (nphi,)."""
    Bz  = (0.5 * muB / tunits) * B_field
    Bxy = (0.5 * muB * theta / tunits) * B_field 
    V0  = V_mat(alpha, beta);  Vd = V0.conj().T
    Hs  = Hloc_SC(alpha, beta)
    Hn  = Hloc_N(Bz, Bxy, alpha, beta)
    Hb  = Hloc_B(alpha, beta)
    gSL, gSR, glr = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w)
    return _phi_sweep_ldos(gSL, gSR, Hn, V0, Vd, w, spatial=False)


def calculate_ldos_phi_B_parallel(B_vals, alpha_raw, beta_raw, target_energy=0.0, n_jobs=-1):
    """
    LDOS(B, phi) at fixed energy — parallelized over B.
    Returns array of shape (nB, nphi).
    """
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_worker_B)(B, alpha, beta, target_energy)
        for B in tqdm(B_vals, desc='LDOS(B,φ)  ')
    )
    return np.array(rows)   # (nB, nphi)


def calculate_ldos_spatial(B_field, alpha_raw, beta_raw, phi, target_energy=0.0):
    """
    LDOS(x, y) at fixed (phi, energy).
    Returns array of shape (normal, ny).
    """
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = (0.5 * muB / tunits) * B_field
    Bxy   = (0.5 * muB * theta / tunits) * B_field 

    V0 = V_mat(alpha, beta);  Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta)
    Hn = Hloc_N(Bz, Bxy, alpha, beta)
    Hb = Hloc_B(alpha, beta)

    gSL, gSR, _ = _build_surface_greens(Hs, Hb, Hn, V0, Vd, target_energy)
    # spatial=True returns (nphi, normal, ny); take the single phi slice
    phi_idx = np.argmin(np.abs(phi_vals - phi))
    full = _phi_sweep_ldos(gSL, gSR, Hn, V0, Vd, target_energy, spatial=True)
    return full[phi_idx]    # (normal, ny)

#%% Band Structure (normal state)

def build_H_normal(k, ny_bands, t_b, mu_b, alpha_b, beta_b):
    dim = 2 * ny_bands
    H   = np.zeros((dim, dim), dtype=np.complex128)
    diag = 4*t_b - mu_b + (alpha_b**2 + beta_b**2)/4 - 2*t_b*np.cos(k)
    for i in range(ny_bands):
        H[2*i,   2*i  ] = diag
        H[2*i+1, 2*i+1] = diag

    soc_y = 1j * (beta_b + alpha_b) / 2
    for i in range(ny_bands - 1):
        ui, di = 2*i,   2*i+1
        uj, dj = 2*i+2, 2*i+3
        H[ui, uj] += -t_b;  H[uj, ui] += -t_b
        H[di, dj] += -t_b;  H[dj, di] += -t_b
        H[ui, dj] +=  soc_y; H[dj, ui] += -soc_y
        H[di, uj] +=  soc_y; H[uj, di] += -soc_y

    soc_x = (alpha_b - beta_b) / 2
    for i in range(ny_bands):
        up   = 2 * i
        down = 2 * i + 1
        H[up,   down] += -soc_x * np.exp( 1j * k)  
        H[down, up  ] +=  soc_x * np.exp( 1j * k)   
        H[up,   down] +=  soc_x * np.exp(-1j * k)  
        H[down, up  ] += -soc_x * np.exp(-1j * k)  

    return (H + H.conj().T) / 2


def compute_bands(ny_b=6, a_b=10e-9, m_eff=0.05):
    tunits_b = (1e3 / e) * hbar**2 / (2 * m_eff * m * a_b**2)
    t_b      = 1.0
    mu_b     = 10.0 / tunits_b
    alpha_b  =  10.0e-9 / (a_b * tunits_b)
    beta_b   = -10.0e-9 / (a_b * tunits_b)

    nk    = 201
    k_arr = np.linspace(-np.pi, np.pi, nk)
    dim   = 2 * ny_b
    bands = np.zeros((nk, dim))
    for i, k in enumerate(k_arr):
        bands[i] = np.linalg.eigvalsh(build_H_normal(k, ny_b, t_b, mu_b, alpha_b, beta_b))
    return k_arr, bands, tunits_b

#%% Plotting Helpers

def _style_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos, title=''):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(energies / Delta, phi_vals / np.pi, ldos.T,
                     levels=100, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega / \Delta$', loc='right')
    ax.set_ylabel(r'$\phi / \pi$', loc='top', rotation=0, labelpad=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    _style_ax(ax)
    if title:
        ax.set_title(title, loc='right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, title=''):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(phi_vals / np.pi, B_vals, ldos,
                     levels=100, cmap='magma')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\phi / \pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    _style_ax(ax)
    if title:
        ax.set_title(title, loc='right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi_val=None, energy_val=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    cf = ax.contourf(np.arange(normal), np.arange(ny), ldos_xy.T,
                     levels=100, cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel('x (junction site)', loc='right')
    ax.set_ylabel('y (transverse site)', loc='top', rotation=0, labelpad=10)
    _style_ax(ax)
    parts = []
    if phi_val   is not None: parts.append(rf'$\phi={phi_val/np.pi:.2f}\pi$')
    if energy_val is not None: parts.append(rf'$\omega={energy_val:.3f}$')
    if parts:
        ax.set_title(', '.join(parts), loc='right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_bands(k_arr, bands, tunits_b):
    colors = ['tab:orange', 'tab:blue']
    dim    = bands.shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(dim):
        ax.plot(k_arr / np.pi, bands[:, j] * tunits_b,
                c=colors[j % 2], lw=1.0, ls='-')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-5, 5)
    ax.set_xticks([-1, -0.5, 0.5, 1])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.4))
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$k / \pi$', loc='right')
    ax.set_ylabel(r'$E$', loc='top', rotation=0, labelpad=3)
    ax.tick_params(which='both', direction='in')
    ax.grid(alpha=0.3, which='major')
    plt.tight_layout()
    return fig, ax

#%% Run

if __name__ == '__main__':

    alpha_1 = 14.3e-9
    beta_1  =  7.3e-9
    B_field = 0.0
    B_vals  = np.linspace(0, 2.0, 41)   

    # 1. LDOS(E, phi) 
    print("Computing LDOS(E, φ)...")
    ldos_E_phi = calculate_ldos_parallel(B_field, alpha_1, beta_1)
    plot_ldos_E_phi(ldos_E_phi,
        title=rf'$\alpha={alpha_1/1e-9:.1f}$ nm, $\beta={beta_1/1e-9:.1f}$ nm, $B={B_field:.1f}$ T')

    # 2. LDOS(B, phi) 
    print("Computing LDOS(B, φ)...")
    ldos_B_phi = calculate_ldos_phi_B_parallel(B_vals, alpha_1, beta_1, target_energy=0.0)
    plot_ldos_B_phi(ldos_B_phi, B_vals,
        title=rf'$\omega=0$, $\alpha={alpha_1/1e-9:.1f}$ nm, $\beta={beta_1/1e-9:.1f}$ nm')

    # 3. LDOS(x, y) 
    print("Computing LDOS(x, y)...")
    ldos_xy = calculate_ldos_spatial(B_field, alpha_1, beta_1, phi=np.pi, target_energy=0.0)
    plot_ldos_spatial(ldos_xy, phi_val=np.pi, energy_val=0.0)

    # 4. Normal-state band structure 
    print("Computing band structure...")
    k_arr, bands, tunits_b = compute_bands(ny_b=6, a_b=10e-9, m_eff=0.05)
    plot_bands(k_arr, bands, tunits_b)

    plt.show()
# %%
hbar = 1.05e-34
e = 1.602e-19
m0 = 9.1e-31
muB = 5.78e-2         

ny = 6
a  = 10e-9               

tunits = (1e3 / e) * hbar**2 / (2 * 0.05 * m0 * a**2)

t      = 1.0
mu     = 10  / tunits
alpha  =  10.0e-9 / (a * tunits)
beta   = -10.0e-9 / (a * tunits)  

B      = 0.0
Bz     = 0.0   
Bxy    = 0.0

def compute_majorana_modes(H, tol=1e-3, n_modes=4):
    """
    Compute eigenvectors of BdG Hamiltonian corresponding to near-zero-energy modes.

    Parameters
    ----------
    H : (N,N) np.complex128 ndarray
        Full BdG Hamiltonian
    tol : float
        Energy tolerance for "zero mode"
    n_modes : int
        Maximum number of modes to return

    Returns
    -------
    energies : array
        Energies of selected modes
    modes : list of arrays
        Corresponding eigenvectors ψ_n
    """

    E, psi = np.linalg.eigh(H)

    idx = np.where(np.abs(E) < tol)[0]

    idx = idx[np.argsort(np.abs(E[idx]))]

    idx = idx[:n_modes]

    energies = E[idx]
    modes = [psi[:, i] for i in idx]

    return energies, modes

def majorana_density(psi, normal, ny):
    """
    Convert eigenvector into spatial density |psi(x,y)|^2
    (summing spin + Nambu)
    """
    psi_r = psi.reshape(normal * ny, 4)

    density = np.sum(np.abs(psi_r)**2, axis=1)  # sum internal DOF
    return density.reshape(normal, ny)

def build_full_H(regions, V):
    block_dim = regions[0].shape[0]
    N = len(regions) * block_dim

    H = np.zeros((N, N), dtype=np.complex128)

    for i, H_i in enumerate(regions):
        sl = slice(i*block_dim, (i+1)*block_dim)

        # onsite block
        H[sl, sl] = H_i

        # hopping to next slice
        if i < len(regions) - 1:
            sl_next = slice((i+1)*block_dim, (i+2)*block_dim)

            H[sl, sl_next] = V
            H[sl_next, sl] = V.conj().T

    return H

regions = (
    [Hloc_SC(alpha_1, beta_1)] * 200 +
    [Hloc_B(alpha_1, beta_1)] * 1 +
    [Hloc_N(Bz, Bxy, alpha_1, beta_1)] * 50 +
    [Hloc_B(alpha_1, beta_1)] * 1 +
    [Hloc_SC(alpha_1, beta_1)] * 200
)

V = V_mat(alpha_1, beta_1)

H_total = build_full_H(regions, V)

E0, modes = compute_majorana_modes(H_total, tol=1e-3)

for n, psi in enumerate(modes):
    rho_xy = majorana_density(psi, normal, ny)

    plt.figure()
    plt.title(f"Majorana mode n={n}, E={E0[n]:.3e}")
    plt.imshow(rho_xy.T, origin='lower', aspect='auto')
    plt.colorbar(label="|ψ|²")
    plt.show()
    
    
#%% 
import numpy as np
from numpy.linalg import inv

def majorana_profile_from_gf(Hs, Hb, Hn, V0, Vd, phi, E0=0.0, eta=1e-4):
    z  = E0 + 1j * eta
    zI = z * _I4

    gSL0 = sancho(Hs, Vd, V0, E0, eta)
    gSR0 = sancho(Hs, V0, Vd, E0, eta)

    gSL = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR = inv(zI - Hb - V0 @ gSR0 @ Vd)

    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    gSR = apply_phase(gSR, phi)

    grl = [None] * normal
    grl[-1] = inv(zI - Hn - V0 @ gSR @ Vd)

    for j in range(normal - 2, -1, -1):
        grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

    psi2 = np.zeros((normal, ny), dtype=float)

    for j in range(normal):
        SL = Vd @ grl[j-1] @ V0 if j > 0 else Vd @ gSL @ V0
        SR = V0 @ grl[j+1] @ Vd if j < normal - 1 else V0 @ gSR @ Vd

        G = inv(zI - Hn - SL - SR)

        for iy in range(ny):
            blk = G[4*iy:4*iy+4, 4*iy:4*iy+4]
            psi2[j, iy] = -np.imag(np.trace(blk)) / np.pi

    psi2 /= np.max(psi2)
    return psi2

Hs = Hloc_SC(alpha_1, beta_1)
Hb = Hloc_B(alpha_1, beta_1)
Hn = Hloc_N(Bz, Bxy, alpha_1, beta_1)

V0 = V_mat(alpha_1, beta_1)
Vd = V0.conj().T

maj = majorana_profile_from_gf(Hs, Hb, Hn, V0, Vd, phi=np.pi, E0=0.0)


plt.imshow(maj.T, origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="Majorana weight")
plt.show()
# %%

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
from tqdm import tqdm

#%% Global Configuration Scale Parameters
ny      = 20
a       = 20e-9
normal  = 30
nbar    = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
m0 = m * 0.038
muB  = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it    = 800
Delta = 0.25 / tunits
t     = 1.0

mu      = 1   / tunits
mu_n    = 0.7 / tunits
mu_barr = 0.0 / tunits
theta   = 0.3 * np.pi / 2
eta     = 0.01 * Delta

nphi = 41
nw   = 41

phi_vals = np.linspace(0, 2*np.pi, nphi)
energies = np.linspace(-Delta, Delta, nw)

#%% Dynamic Hamiltonian & Phase Builders

def make_transverse_hopping(alpha, beta, current_ny):
    """y-direction hopping + SOC block dynamically scaled to current_ny."""
    hop_block = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([
        [0,  -1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,   1j ],
        [0,   0, -1j,  0  ]
    ], dtype=np.complex128) * (alpha + beta) / 2

    block  = hop_block + soc
    blockH = hop_block + soc.conj().T

    hop = np.zeros((4*current_ny, 4*current_ny), dtype=np.complex128)
    for i in range(current_ny - 1):
        hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = block
        hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = blockH
    return hop


def make_block_diagonal(block_4x4, current_ny):
    return np.kron(np.eye(current_ny, dtype=np.complex128), block_4x4)


def Hloc_SC(alpha, beta, current_ny=ny):
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.array([
        [ onsite_val,  0,          Delta, 0          ],
        [ 0,           onsite_val, 0,     Delta      ],
        [ Delta,       0,         -onsite_val, 0      ],
        [ 0,           Delta,      0,    -onsite_val  ]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite, current_ny)
    H += make_transverse_hopping(alpha, beta, current_ny)
    return H


def Hloc_N(Bz, Bxy, alpha, beta, current_ny=ny):
    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.array([
        [ onsite_val + Bz,           Bxy * np.exp( 1j*theta), 0, 0                          ],
        [ Bxy * np.exp(-1j*theta),   onsite_val - Bz,         0, 0                          ],
        [ 0, 0,  -(onsite_val + Bz),           Bxy * np.exp( 1j*theta)                  ],
        [ 0, 0,   Bxy * np.exp(-1j*theta),  -(onsite_val - Bz)                          ]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite, current_ny)
    H += make_transverse_hopping(alpha, beta, current_ny)
    return H


def Hloc_B(alpha, beta, current_ny=ny):
    onsite_val = 4*t - mu_barr
    onsite = np.diag([onsite_val, onsite_val, -onsite_val, -onsite_val]).astype(np.complex128)
    H = make_block_diagonal(onsite, current_ny)
    H += make_transverse_hopping(alpha, beta, current_ny)
    return H


def V_mat(alpha, beta, current_ny=ny):
    s = 0.5 * (alpha - beta)
    v = np.array([
        [-t,  -s,  0,  0],
        [ s,  -t,  0,  0],
        [ 0,   0,  t,  s],
        [ 0,   0, -s,  t]
    ], dtype=np.complex128)
    return make_block_diagonal(v, current_ny)

def U(phi, ny_val=None):
    u = np.eye(4, dtype=np.complex128)
    u[2, 2] = np.exp(1j * phi)
    u[3, 3] = np.exp(1j * phi)
    return make_block_diagonal(u, ny_val)


def apply_phase(G, phi):
    dim = G.shape[0]
    ny_val = dim // 4
    return U(-phi, ny_val) @ G @ U(phi, ny_val)


def sancho(H, alpha_0, beta_0, w, eta):
    z     = (w + 1j*eta) * np.eye(H.shape[0], dtype=np.complex128)
    eps   = H.copy()
    eps_s = H.copy()
    alpha = alpha_0.copy()
    beta  = beta_0.copy()
    for _ in range(it):
        gi    = inv(z - eps)
        ab    = alpha @ gi @ beta
        ba    = beta  @ gi @ alpha
        eps_s = eps_s + ab
        eps   = eps   + ab + ba
        alpha = alpha @ gi @ alpha
        beta  = beta  @ gi @ beta
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < 1e-12:
            break
    return inv(z - eps_s)

#%% Green's Function Subroutines

def _build_surface_greens(Hs, Hb, Hn, V0, Vd, w, current_normal):
    dim = Hn.shape[0]
    zI  = (w + 1j*eta) * np.eye(dim, dtype=np.complex128)
    
    gSL0 = sancho(Hs, Vd, V0, w, eta)
    gSR0 = sancho(Hs, V0, Vd, w, eta)
    gSL  = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR  = inv(zI - Hb - V0 @ gSR0 @ Vd)
    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    Vd_gSL_V0 = Vd @ gSL @ V0

    glr    = [None] * current_normal
    glr[0] = inv(zI - Hn - Vd_gSL_V0)
    for i in range(1, current_normal):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)

    Vd_glr_V0 = [Vd @ glr[j] @ V0 for j in range(current_normal)]
    return gSL, gSR, Vd_gSL_V0, Vd_glr_V0


def _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w, current_normal,
                    spatial=False, phi_single=None):
    dim = Hn.shape[0]
    current_ny = dim // 4
    zI = (w + 1j*eta) * np.eye(dim, dtype=np.complex128)

    phi_arr = [phi_single] if (spatial and phi_single is not None) else phi_vals
    result = np.zeros((current_normal, current_ny)) if spatial else np.zeros(len(phi_vals))

    for iphi, phi in enumerate(phi_arr):
        gSRp = apply_phase(gSR, phi)

        grl      = [None] * current_normal
        grl[-1]  = inv(zI - Hn - V0 @ gSRp @ Vd)
        for j in range(current_normal - 2, -1, -1):
            grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

        V0_gSRp_Vd = V0 @ gSRp @ Vd

        if spatial:
            for j in range(current_normal):
                SL = Vd_glr_V0[j-1] if j > 0       else Vd_gSL_V0
                SR = V0 @ grl[j+1] @ Vd if j < current_normal-1 else V0_gSRp_Vd
                G  = inv(zI - Hn - SL - SR)
                diag_G = np.diag(G).reshape(current_ny, 4)
                result[j, :] += -np.imag(diag_G.sum(axis=1)) / np.pi
        else:
            LD = 0.0 + 0j
            for j in range(current_normal):
                SL = Vd_glr_V0[j-1] if j > 0       else Vd_gSL_V0
                SR = V0 @ grl[j+1] @ Vd if j < current_normal-1 else V0_gSRp_Vd
                G  = inv(zI - Hn - SL - SR)
                LD += np.trace(G)
            result[iphi] = -np.imag(LD) / np.pi

    return result

#%% Parallel Sweep Managers

def _worker_w(w, Hs, Hb, Hn, V0, Vd, current_normal):
    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w, current_normal)
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w, current_normal, spatial=False)


def calculate_ldos_parallel(B_field, alpha_raw, beta_raw, energy_array=None, n_jobs=-1):
    if energy_array is None:
        energy_array = energies
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits

    V0 = V_mat(alpha, beta, ny);  Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta, ny)
    Hn = Hloc_N(Bz, Bxy, alpha, beta, ny)
    Hb = Hloc_B(alpha, beta, ny)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_worker_w)(w, Hs, Hb, Hn, V0, Vd, normal)
        for w in tqdm(energy_array, desc='LDOS(E,φ)  ')
    )
    return np.array(rows)


def _worker_B(B_field, alpha, beta, w, current_normal):
    Bz  = 0.5 * muB * theta * B_field / tunits
    Bxy = 0.5 * muB         * B_field / tunits
    V0  = V_mat(alpha, beta, ny);  Vd = V0.conj().T
    Hs  = Hloc_SC(alpha, beta, ny)
    Hn  = Hloc_N(Bz, Bxy, alpha, beta, ny)
    Hb  = Hloc_B(alpha, beta, ny)
    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w, current_normal)
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w, current_normal, spatial=False)


def calculate_ldos_phi_B_parallel(B_vals, alpha_raw, beta_raw, target_energy=0.0, n_jobs=-1):
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_worker_B)(B, alpha, beta, target_energy, normal)
        for B in tqdm(B_vals, desc='LDOS(B,φ)  ')
    )
    return np.array(rows)


def calculate_ldos_spatial(B_field, alpha_raw, beta_raw, phi, target_energy=0.0):
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits

    V0 = V_mat(alpha, beta, ny);  Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta, ny)
    Hn = Hloc_N(Bz, Bxy, alpha, beta, ny)
    Hb = Hloc_B(alpha, beta, ny)

    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, target_energy, normal)
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd,
                           target_energy, normal, spatial=True, phi_single=phi)

#%% Diag, Band Structure & Profile Tools

def compute_majorana_modes(H, tol=1e-3, n_modes=4):
    E, psi = np.linalg.eigh(H)
    idx = np.where(np.abs(E) < tol)[0]
    idx = idx[np.argsort(np.abs(E[idx]))][:n_modes]
    return E[idx], [psi[:, i] for i in idx]


def majorana_density(psi, total_slices, current_ny):
    psi_r   = psi.reshape(total_slices * current_ny, 4)
    density = np.sum(np.abs(psi_r)**2, axis=1)
    return density.reshape(total_slices, current_ny)


def build_full_H(regions, V):
    block_dim = regions[0].shape[0]
    N = len(regions) * block_dim
    H = np.zeros((N, N), dtype=np.complex128)
    for i, H_i in enumerate(regions):
        sl = slice(i*block_dim, (i+1)*block_dim)
        H[sl, sl] = H_i
        if i < len(regions) - 1:
            sl_next = slice((i+1)*block_dim, (i+2)*block_dim)
            H[sl,       sl_next] = V
            H[sl_next,  sl     ] = V.conj().T
    return H


def build_H_normal(k, ny_bands, t_b, mu_b, alpha_b, beta_b):
    dim  = 2 * ny_bands
    H    = np.zeros((dim, dim), dtype=np.complex128)
    diag = 4*t_b - mu_b + (alpha_b**2 + beta_b**2)/4 - 2*t_b*np.cos(k)

    for i in range(ny_bands):
        H[2*i,   2*i  ] = diag
        H[2*i+1, 2*i+1] = diag

    soc_y = 1j * (beta_b + alpha_b) / 2
    for i in range(ny_bands - 1):
        ui, di = 2*i,   2*i+1
        uj, dj = 2*i+2, 2*i+3
        H[ui, uj] += -t_b;   H[uj, ui] += -t_b
        H[di, dj] += -t_b;   H[dj, di] += -t_b
        H[ui, dj] +=  soc_y; H[dj, ui] += -soc_y
        H[di, uj] +=  soc_y; H[uj, di] += -soc_y

    soc_x = (alpha_b - beta_b) / 2
    for i in range(ny_bands):
        up   = 2 * i
        down = 2 * i + 1
        H[up,   down] += -soc_x * np.exp( 1j * k)
        H[down, up  ] +=  soc_x * np.exp( 1j * k)
        H[up,   down] +=  soc_x * np.exp(-1j * k)
        H[down, up  ] += -soc_x * np.exp(-1j * k)

    return (H + H.conj().T) / 2


def majorana_profile_from_gf(Hs, Hb, Hn, V0, Vd, phi, current_normal, E0=0.0, eta=1e-4):
    dim = Hn.shape[0]
    current_ny = dim // 4
    zI = (E0 + 1j * eta) * np.eye(dim, dtype=np.complex128)

    gSL0 = sancho(Hs, Vd, V0, E0, eta)
    gSR0 = sancho(Hs, V0, Vd, E0, eta)
    gSL  = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR  = inv(zI - Hb - V0 @ gSR0 @ Vd)

    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    gSR = apply_phase(gSR, phi)

    grl      = [None] * current_normal
    grl[-1]  = inv(zI - Hn - V0 @ gSR @ Vd)
    for j in range(current_normal - 2, -1, -1):
        grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

    psi2 = np.zeros((current_normal, current_ny), dtype=float)

    for j in range(current_normal):
        SL = Vd @ grl[j-1] @ V0 if j > 0       else Vd @ gSL @ V0
        SR = V0 @ grl[j+1] @ Vd if j < current_normal-1 else V0 @ gSR @ Vd
        G  = inv(zI - Hn - SL - SR)
        for iy in range(current_ny):
            blk = G[4*iy:4*iy+4, 4*iy:4*iy+4]
            psi2[j, iy] = -np.imag(np.trace(blk)) / np.pi

    psi2 /= np.max(psi2)
    return psi2

#%% Plotting Directives

def _style_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')

def plot_ldos_E_phi(ldos, title=''):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(energies / Delta, phi_vals / np.pi, ldos.T, levels=100, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega / \Delta$', loc='right')
    ax.set_ylabel(r'$\phi / \pi$', loc='top', rotation=0, labelpad=10)
    _style_ax(ax)
    if title: ax.set_title(title, loc='right', fontsize=9)
    plt.tight_layout()

def plot_ldos_B_phi(ldos, B_vals, title=''):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(phi_vals / np.pi, B_vals, ldos, levels=100, cmap='magma')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\phi / \pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=10)
    _style_ax(ax)
    if title: ax.set_title(title, loc='right', fontsize=9)
    plt.tight_layout()

#%% Execution Entrypoint
if __name__ == '__main__':
    alpha_1, beta_1 = 14.3e-9, 7.3e-9
    B_field = 0.5
    B_vals  = np.linspace(0, 2.0, 51)

    # 1. Transport Sweeps
    print("Computing Transport Sweeps...")
    ldos_E_phi = calculate_ldos_parallel(B_field, alpha_1, beta_1)
    plot_ldos_E_phi(ldos_E_phi, title="Energy-Phase Spectrum")

    ldos_B_phi = calculate_ldos_phi_B_parallel(B_vals, alpha_1, beta_1, target_energy=0.0)
    plot_ldos_B_phi(ldos_B_phi, B_vals, title="Critical Field Sweeps")

    # # 2. Majorana Full Real-Space Diagonalization
    # print("Computing Majorana Real-Space Modes...")
    ny_maj = 6  # Isolated scale space to prevent global collision
    tunits_maj = (1e3 / e) * hbar**2 / (2 * 0.05 * m0 * a**2)
    
    Bz_maj, Bxy_maj = 1.2, 0.0
    
    # # Generate local matrices using custom target size dimensions
    H_sc   = Hloc_SC(alpha_1, beta_1, current_ny=ny_maj)
    H_barr = Hloc_B(alpha_1, beta_1, current_ny=ny_maj)
    H_norm = Hloc_N(Bz_maj, Bxy_maj, alpha_1, beta_1, current_ny=ny_maj)
    V_hop  = V_mat(alpha_1, beta_1, current_ny=ny_maj)

    # regions = [H_sc]*200 + [H_barr]*1 + [H_norm]*50 + [H_barr]*1 + [H_sc]*200
    # H_total = build_full_H(regions, V_hop)
    
    # E0, modes = compute_majorana_modes(H_total, tol=1e-3)
    # total_slices = len(regions)

    # for n, psi in enumerate(modes[:2]): # view lowest primary modes
    #     rho_xy = majorana_density(psi, total_slices, ny_maj)
    #     plt.figure(figsize=(6, 3))
    #     plt.title(f"Majorana mode n={n}, E={E0[n]:.3e}")
    #     plt.imshow(rho_xy.T, origin='lower', aspect='auto', cmap='plasma')
    #     plt.colorbar(label=r"$|\psi|^2$")
    #     plt.xlabel("X Spatial Coordinates")
    #     plt.ylabel("Y Transverse Sites")
    #     plt.tight_layout()

    # 3. Localized Green's Function Profile Mapping
    print("Computing Green's Function Boundary Profiles...")
    maj_profile = majorana_profile_from_gf(H_sc, H_barr, H_norm, V_hop, V_hop.conj().T, 
                                           phi=np.pi, current_normal=normal, E0=0.0)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(maj_profile.T, origin='lower', aspect='auto', cmap='inferno')
    plt.title("Green's Function Zero-Bias Peak Profile Mapping")
    plt.colorbar(label="Majorana Local Weight")
    plt.xlabel("Junction Slice Index")
    plt.ylabel("Transverse Index")
    plt.tight_layout()
    plt.show()


# %%
