#%% Imports
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
from tqdm import tqdm

#%% Parameters

ny     = 100
a      = 20e-9
normal = 2
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it    = 800
Delta = 0.25 / tunits
t     = 1.0

mu      = 2 / tunits
mu_n    = 0.1
mu_barr = 0.0 / tunits
theta   = 0.15 * np.pi

eta  = 0.025 * Delta
nphi = 61
nw   = 61

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

    block  = hop_block + soc
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
        [ onsite_val,  0,          Delta, 0          ],
        [ 0,           onsite_val, 0,     Delta      ],
        [ Delta,       0,         -onsite_val, 0      ],
        [ 0,           Delta,      0,    -onsite_val  ]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_N(Bz, Bxy, alpha, beta):
    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.array([
        [ onsite_val + Bz,           Bxy * np.exp( 1j*theta), 0, 0                      ],
        [ Bxy * np.exp(-1j*theta),   onsite_val - Bz,         0, 0                      ],
        [ 0, 0,  -(onsite_val - Bz),           Bxy * np.exp( 1j*theta)                  ],
        [ 0, 0,   Bxy * np.exp(-1j*theta),  -(onsite_val + Bz)                          ]
    ], dtype=np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_B(alpha, beta):
    onsite_val = 4*t - mu_barr
    onsite = np.diag([onsite_val, onsite_val, -onsite_val, -onsite_val]).astype(np.complex128)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(0, 0)
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
    """Applies the exact U[phi] . G . U[-phi] gauge transformation from Mathematica."""
    dim = G.shape[0]
    elec_idx = np.arange(dim)[np.arange(dim) % 4 < 2]
    hole_idx = np.arange(dim)[np.arange(dim) % 4 >= 2]

    Gp = G.copy()
    # U[phi] puts exp(1j*phi) on holes; U[-phi] puts exp(-1j*phi) on holes
    Gp[np.ix_(elec_idx, hole_idx)] *= np.exp(-1j * phi)
    Gp[np.ix_(hole_idx, elec_idx)] *= np.exp(1j * phi)
    return Gp


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
    so we hoist them out of the phi loop.
    """
    z    = w + 1j*eta
    zI   = z * _I4
    gSL0 = sancho(Hs, Vd, V0, w, eta)
    gSR0 = sancho(Hs, V0, Vd, w, eta)
    gSL  = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR  = inv(zI - Hb - V0 @ gSR0 @ Vd)
    for _ in range(nbar - 1):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    # Precompute left boundary self-energy — phi-independent, reused every phi
    Vd_gSL_V0 = Vd @ gSL @ V0

    # Left-to-right propagator chain — phi-independent
    glr    = [None] * (normal + 1)
    glr[0] = gSL
    for i in range(1, normal + 1):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)

    # Precompute Vd @ glr[j] @ V0 for all j — reused as SL in every phi iteration
    Vd_glr_V0 = [Vd @ glr[j] @ V0 for j in range(normal)]

    return gSL, gSR, Vd_gSL_V0, Vd_glr_V0


def _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w,
                    spatial=False, phi_single=None):
    """
    Inner phi loop. Returns:
      spatial=False, phi_single=None  : 1D array (nphi,)      total LDOS per phi
      spatial=True,  phi_single=float : 2D array (normal, ny) site-resolved LDOS

    phi_single skips the full phi sweep when only one value is needed (spatial case).
    """
    z  = w + 1j*eta
    zI = z * _I4

    phi_arr = [phi_single] if (spatial and phi_single is not None) else phi_vals

    if spatial:
        result = np.zeros((normal, ny))
    else:
        result = np.zeros(nphi)

    for iphi, phi in enumerate(phi_arr):
        gSRp = apply_phase(gSR, phi)

        # Right-to-left propagator chain — phi-dependent
        grl      = [None] * (normal + 1)
        grl[normal]  = gSRp
        for j in range(normal - 1, -1, -1):
            grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

        if spatial:
            for j in range(normal):
                SL = Vd_glr_V0[j-1] 
                SR = V0 @ grl[j+1] @ Vd 
                G  = inv(zI - Hn - SL - SR)
                # Vectorized diagonal trace over all ny sites at once
                diag_G = np.diag(G).reshape(ny, 4)
                result[j, :] += -np.imag(diag_G.sum(axis=1)) / np.pi
        else:
            LD = 0.0 + 0j
            for j in range(normal):
                SL = Vd_glr_V0[j-1]  
                SR = V0 @ grl[j+1] @ Vd 
                G  = inv(zI - Hn - SL - SR)
                LD += np.trace(G)
            result[iphi] = -np.imag(LD) / np.pi

    return result

#%% Parallelized LDOS Sweeps

def _worker_w(w, Hs, Hb, Hn, V0, Vd):
    """One energy slice → row of shape (nphi,). Used by calculate_ldos_parallel."""
    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w)
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w, spatial=False)


def calculate_ldos_parallel(B_field, alpha_raw, beta_raw, energy_array=None, n_jobs=-1):
    """
    LDOS(E, phi) — parallelized over energy.
    Returns array of shape (nw, nphi).
    """
    if energy_array is None:
        energy_array = energies
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits

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
    Bz  = 0.5 * muB * theta * B_field / tunits
    Bxy = 0.5 * muB         * B_field / tunits
    V0  = V_mat(alpha, beta);  Vd = V0.conj().T
    Hs  = Hloc_SC(alpha, beta)
    Hn  = Hloc_N(Bz, Bxy, alpha, beta)
    Hb  = Hloc_B(alpha, beta)
    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, w)
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd, w, spatial=False)


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
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits

    V0 = V_mat(alpha, beta);  Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta)
    Hn = Hloc_N(Bz, Bxy, alpha, beta)
    Hb = Hloc_B(alpha, beta)

    gSL, gSR, Vd_gSL_V0, Vd_glr_V0 = _build_surface_greens(Hs, Hb, Hn, V0, Vd, target_energy)
    # phi_single avoids sweeping all nphi values when only one is needed
    return _phi_sweep_ldos(gSL, gSR, Vd_gSL_V0, Vd_glr_V0, Hn, V0, Vd,
                           target_energy, spatial=True, phi_single=phi)  # (normal, ny)

#%% Majorana Modes — full diagonalization

def compute_majorana_modes(H, tol=1e-3, n_modes=4):
    """
    Compute eigenvectors of BdG Hamiltonian corresponding to near-zero-energy modes.
    """
    E, psi = np.linalg.eigh(H)
    idx = np.where(np.abs(E) < tol)[0]
    idx = idx[np.argsort(np.abs(E[idx]))]
    idx = idx[:n_modes]
    energies = E[idx]
    modes    = [psi[:, i] for i in idx]
    return energies, modes


def majorana_density(psi, total_slices, current_ny):
    """Convert eigenvector into spatial density |psi(x,y)|^2."""
    psi_r   = psi.reshape(total_slices * current_ny, 4)
    density = np.sum(np.abs(psi_r)**2, axis=1)
    return density.reshape(total_slices, current_ny)


def build_full_H(regions, V):
    """Assemble full real-space BdG Hamiltonian."""
    block_dim = regions[0].shape[0]
    N = len(regions) * block_dim
    H = np.zeros((N, N), dtype=np.complex128)

    for i, H_i in enumerate(regions):
        sl = slice(i*block_dim, (i+1)*block_dim)
        H[sl, sl] = H_i
        if i < len(regions) - 1:
            sl_next = slice((i+1)*block_dim, (i+2)*block_dim)
            H[sl,      sl_next] = V
            H[sl_next, sl     ] = V.conj().T
    return H

#%% Band Structure (normal state)

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
    if phi_val    is not None: parts.append(rf'$\phi={phi_val/np.pi:.2f}\pi$')
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

    alpha_1 = 14.3e-9 / (a * tunits)
    beta_1  =  7.3e-9 / (a * tunits)
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

# %%  Majorana modes — full real-space diagonalization

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def majorana_profile_with_leads(Hs, Hb, Hn, V0, normal, num_S=20):
    Vd = V0.conj().T
    block_dim = Hs.shape[0]
    ny = block_dim // 4
    
    diag_blocks = [Hs] * num_S + [Hb] + [Hn] * normal + [Hb] + [Hs] * num_S
    N_total = len(diag_blocks)
    total_dim = N_total * block_dim
    
    H_full = sp.lil_matrix((total_dim, total_dim), dtype=np.complex128)
    
    # Fill diagonal and off-diagonal blocks
    for i in range(N_total):
        # Diagonal block
        i_start = i * block_dim
        i_end = (i + 1) * block_dim
        H_full[i_start:i_end, i_start:i_end] = diag_blocks[i]
        
        # Right hopping (if not last block)
        if i < N_total - 1:
            j_start = (i + 1) * block_dim
            j_end = (i + 2) * block_dim
            H_full[i_start:i_end, j_start:j_end] = V0
            H_full[j_start:j_end, i_start:i_end] = Vd
    
    # Convert to CSC for efficient eigensolving
    H_full = H_full.tocsc()
    
    energies, wavefunctions = spla.eigsh(H_full, k=2, sigma=0.0, which='LM')
    print("Lowest system energies found:", energies)
    
    idx_min = np.argmin(np.abs(energies))
    psi = wavefunctions[:, idx_min]
    psi_reshaped = psi.reshape(N_total, ny, 4)
    
    prob_density_full = np.abs(psi_reshaped)**2
    prob_density_spatial = np.sum(prob_density_full, axis=2)
    prob_density_spatial /= np.max(prob_density_spatial) if np.max(prob_density_spatial) > 0 else 1.0
    
    return prob_density_spatial, np.sum(prob_density_full[:, :, :2], axis=2), \
           np.sum(prob_density_full[:, :, 2:], axis=2), psi_reshaped, energies[idx_min]


alpha_1 = 16e-9
beta_1  = 7e-9
B_field = 2
theta_maj = 0
Bz_maj    = 1.5
Bxy_maj   = 1.5

Hs = Hloc_SC(alpha_1 / (a * tunits), beta_1 / (a * tunits))
Hb = Hloc_B(alpha_1 / (a * tunits), beta_1 / (a * tunits))
Hn = Hloc_N(Bz_maj, Bxy_maj, alpha_1 / (a * tunits), beta_1 / (a * tunits))

V0 = V_mat(alpha_1 / (a * tunits), beta_1 / (a * tunits))
Vd = V0.conj().T


# --- Run with diagnostics ---
maj_spatial, maj_elec, maj_hole, psi_full, _ = majorana_profile_with_leads(
    Hs, Hb, Hn, V0, normal=50, num_S=200
)

# Plot all three to diagnose the structure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total LDOS(x, y)
im0 = axes[0].imshow(maj_spatial.T, origin='lower', aspect='auto', cmap='inferno')
axes[0].set_title("Total LDOS(x, y)")
axes[0].set_xlabel("Junction slice x")
axes[0].set_ylabel("Transverse site y")
plt.colorbar(im0, ax=axes[0])

# Electron contribution only
im1 = axes[1].imshow(maj_elec.T, origin='lower', aspect='auto', cmap='inferno')
axes[1].set_title("Electron density (c↑, c↓)")
axes[1].set_xlabel("Junction slice x")
plt.colorbar(im1, ax=axes[1])

# Hole contribution only
im2 = axes[2].imshow(maj_hole.T, origin='lower', aspect='auto', cmap='inferno')
axes[2].set_title("Hole density (c↓†, -c↑†)")
axes[2].set_xlabel("Junction slice x")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()


# Zoom into the junction region for clarity
fig, ax = plt.subplots(figsize=(10, 5))
x_start = 200  # num_S
x_end = 250   # num_S + normal
junction_dat = maj_spatial[x_start:x_end].T
im = ax.imshow(junction_dat, origin='lower', aspect='auto', cmap='inferno')
ax.set_title("Majorana density at the junction")
ax.set_xlabel("x (relative to junction start)")
ax.set_ylabel("y (transverse)")
plt.colorbar(im, ax=ax, label="|ψ|²")
plt.tight_layout()
plt.show()

# %%