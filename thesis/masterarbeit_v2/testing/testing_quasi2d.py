#%% Imports
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
from tqdm import tqdm
#%% Parameters
ny     = 30
a      = 20e-9
normal = 50 
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2        # meV/T

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it    = 1000
Delta = 0.25 / tunits
t     = 1.0

mu      = 2.0 / tunits
mu_n    = 0.1
mu_barr = 0.0 / tunits
theta   = 0.3 * np.pi / 2

eta  = 0.01 * Delta
nphi = 51
nw   = 51

phi_vals = np.linspace(0, 2*np.pi, nphi)
energies = np.linspace(-Delta, Delta, nw)

_I4 = np.eye(4 * ny, dtype=complex)

# Precompute hole indices once — used by the fast phase application
_hole_idx = np.array([i for i in range(4*ny) if i % 4 >= 2])
_elec_idx = np.array([i for i in range(4*ny) if i % 4 <  2])

#%% Parametrized Hamiltonian Generators

def make_transverse_hopping(alpha, beta):
    hop_block = np.diag([-t, -t, t, t]).astype(complex)
    soc = np.array([
        [0,  -1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,   1j ],
        [0,   0, -1j,  0  ]
    ], dtype=complex) * (alpha + beta) / 2

    hop = np.zeros((4*ny, 4*ny), dtype=complex)
    for i in range(ny - 1):
        hop[4*i   :4*(i+1), 4*(i+1):4*(i+2)] = hop_block + soc
        hop[4*(i+1):4*(i+2), 4*i   :4*(i+1)] = hop_block + soc.conj().T
    return hop


def make_block_diagonal(block_4x4):
    return np.kron(np.eye(ny, dtype=complex), block_4x4)


def Hloc_SC(alpha, beta):
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val;  onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val;  onsite[3, 3] = -onsite_val
    onsite[0, 2] =  Delta;       onsite[1, 3] =  Delta
    onsite[2, 0] =  Delta;       onsite[3, 1] =  Delta
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_N(Bz, Bxy, alpha, beta):
    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val + Bz;  onsite[1, 1] =  onsite_val - Bz
    onsite[2, 2] = -onsite_val - Bz;  onsite[3, 3] = -onsite_val + Bz
    onsite[0, 1] =  Bxy * np.exp( 1j*theta)
    onsite[1, 0] =  Bxy * np.exp(-1j*theta)
    onsite[2, 3] =  Bxy * np.exp( 1j*theta)
    onsite[3, 2] =  Bxy * np.exp(-1j*theta)
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_B(alpha, beta):
    onsite_val = 4*t - mu_barr
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val;  onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val;  onsite[3, 3] = -onsite_val
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def V_mat(alpha, beta):
    s = 0.5 * (alpha - beta)
    v = np.zeros((4, 4), dtype=complex)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t
    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return make_block_diagonal(v)


def _apply_phase_fast(G, phi):
    """
    Computes U(-phi) @ G @ U(phi) without building the full diagonal matrix.
    U(phi) = diag([1,1,e^{i*phi},e^{i*phi}]) ⊗ I_ny  (holes get e^{i*phi}).

    Effect on blocks:
      G[elec, elec]  →  unchanged          (e^{-i*phi} * e^{+i*phi} = 1 cancel)
      G[elec, hole]  →  G * e^{+i*phi}     (only right U acts)
      G[hole, elec]  →  G * e^{-i*phi}     (only left  U acts)
      G[hole, hole]  →  unchanged          (e^{-i*phi} * e^{+i*phi} = 1 cancel)
    """
    ep = np.exp( 1j * phi)
    em = np.exp(-1j * phi)
    G_out = G.copy()
    G_out[np.ix_(_elec_idx, _hole_idx)] *= ep
    G_out[np.ix_(_hole_idx, _elec_idx)] *= em
    # elec-elec and hole-hole blocks are unchanged
    return G_out


def sancho(H, alpha_0, beta_0, w, eta):
    z     = (w + 1j*eta) * np.eye(H.shape[0], dtype=complex)
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


#%% Core Sweep Engines

def _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, spatial=False, phi_single=None):
    """
    Single energy slice. Sweeps phi and returns:
      spatial=False, phi_single=None  : array (nphi,)       total LDOS per phi
      spatial=True,  phi_single=float : array (normal, ny)  site LDOS at that phi

    phi_single avoids sweeping all phi values when only one is needed (spatial case).
    """
    z  = w + 1j*eta
    zI = z * _I4

    # ── surface Green's functions — computed once, phi-independent ────────────
    gSL0 = sancho(Hs, Vd, V0, w, eta)
    gSR0 = sancho(Hs, V0, Vd, w, eta)

    gSL = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR = inv(zI - Hb - V0 @ gSR0 @ Vd)
    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)

    # Precompute these products once — reused in every j-iteration of every phi
    Vd_gSL_V0 = Vd @ gSL @ V0   # left boundary self-energy (phi-independent)

    # ── left-to-right chain — phi-independent ────────────────────────────────
    glr    = [None] * normal
    glr[0] = inv(zI - Hn - Vd_gSL_V0)
    for i in range(1, normal):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)

    # Precompute Vd @ glr[j-1] @ V0 for j=1..normal-1 — reused every phi
    Vd_glr_V0 = [Vd @ glr[j] @ V0 for j in range(normal)]

    # ── phi loop ──────────────────────────────────────────────────────────────
    phi_arr = [phi_single] if (spatial and phi_single is not None) else phi_vals

    if spatial:
        result = np.zeros((normal, ny))
    else:
        result = np.zeros(nphi)

    for iphi, phi in enumerate(phi_arr):
        # Fast O(N²) phase application instead of two O(N³) matrix multiplies
        gSRp = _apply_phase_fast(gSR, phi)

        # right-to-left chain — phi-dependent
        grl      = [None] * normal
        grl[-1]  = inv(zI - Hn - V0 @ gSRp @ Vd)
        for j in range(normal - 2, -1, -1):
            grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)

        # Precompute right boundary self-energy for last site
        V0_gSRp_Vd = V0 @ gSRp @ Vd

        if spatial:
            for j in range(normal):
                SL = Vd_glr_V0[j-1] if j > 0       else Vd_gSL_V0
                SR = V0 @ grl[j+1] @ Vd if j < normal-1 else V0_gSRp_Vd
                G  = inv(zI - Hn - SL - SR)
                # Vectorized trace over all ny sites — no inner Python loop
                # diag of G at positions [4*iy, 4*iy+1, 4*iy+2, 4*iy+3] for each iy
                diag_G = np.diag(G).reshape(ny, 4)
                result[j, :] += -np.imag(diag_G.sum(axis=1)) / np.pi
        else:
            LD = 0.0 + 0j
            for j in range(normal):
                SL = Vd_glr_V0[j-1] if j > 0       else Vd_gSL_V0
                SR = V0 @ grl[j+1] @ Vd if j < normal-1 else V0_gSRp_Vd
                G  = inv(zI - Hn - SL - SR)
                LD += np.trace(G)
            result[iphi] = -np.imag(LD) / np.pi

    return result
def _build_hamiltonians(alpha_raw, beta_raw, B_field):
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits
    V0    = V_mat(alpha, beta)
    Vd    = V0.conj().T
    Hs    = Hloc_SC(alpha, beta)
    Hn    = Hloc_N(Bz, Bxy, alpha, beta)
    Hb    = Hloc_B(alpha, beta)
    return Hs, Hb, Hn, V0, Vd


def calculate_ldos_parallel(alpha_raw, beta_raw, energy_array=None, n_jobs=-1):
    """LDOS(E, phi) at B=0. Returns (nw, nphi)."""
    if energy_array is None:
        energy_array = energies
    Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B_field=1.2)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_ldos_one_slice)(w, Hs, Hb, Hn, V0, Vd, spatial=False)
        for w in tqdm(energy_array, desc='LDOS(E,φ)')
    )
    return np.array(rows)   # (nw, nphi)


def calculate_ldos_phi_B_parallel(B_vals, alpha_raw, beta_raw, target_energy=0.0, n_jobs=-1):
    """LDOS(B, phi) at fixed energy. Returns (nB, nphi)."""
    def _one_B(B):
        Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B)
        return _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, spatial=False)

    rows = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_one_B)(B)
        for B in tqdm(B_vals, desc='LDOS(B,φ)')
    )
    return np.array(rows)   # (nB, nphi)


def calculate_ldos_spatial(alpha_raw, beta_raw, phi, target_energy=0.0):
    """LDOS(x, y) at B=0, single phi and energy. Returns (normal, ny)."""
    Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B_field=1.2)
    # phi_single avoids computing all nphi values unnecessarily
    return _ldos_one_slice(
        target_energy, Hs, Hb, Hn, V0, Vd, spatial=True, phi_single=phi
    )   # (normal, ny)

def build_H_total(alpha_raw, beta_raw, B_field, phi, NS=20):
    """
    Build the full real-space BdG Hamiltonian for the junction:
    NS superconductor sites | normal sites | NS superconductor sites.
    Total sites: 2*NS + normal, each with 4 DOF → (4*(2*NS+normal))² matrix.

    phi: superconducting phase difference applied to the right SC.
    Returns H of shape (4*Ntot*ny, 4*Ntot*ny)  [if ny>1 use kron structure].
    """
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 0.5 * muB * theta * B_field / tunits
    Bxy   = 0.5 * muB         * B_field / tunits

    Ntot  = 2*NS + normal
    dim   = 4 * ny * Ntot
    H     = np.zeros((dim, dim), dtype=complex)

    V0 = V_mat(alpha, beta)   # (4*ny, 4*ny) inter-site hopping
    Vd = V0.conj().T

    Hs_L  = Hloc_SC(alpha, beta)                # left SC onsite
    Hn_bl = Hloc_N(Bz, Bxy, alpha, beta)        # normal onsite
    Hb_bl = Hloc_B(alpha, beta)                 # barrier onsite (nbar=0 here)

    # Right SC: apply phase phi to the pairing (hole sector)
    # U(phi/2) H_SC U(-phi/2) is the standard gauge choice
    # Equivalent: multiply Delta → Delta*e^{i*phi} in the onsite block
    onsite_sc = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite_R  = np.zeros((4, 4), dtype=complex)
    onsite_R[0, 0] =  onsite_sc;  onsite_R[1, 1] =  onsite_sc
    onsite_R[2, 2] = -onsite_sc;  onsite_R[3, 3] = -onsite_sc
    onsite_R[0, 2] =  Delta * np.exp( 1j*phi)
    onsite_R[1, 3] =  Delta * np.exp( 1j*phi)
    onsite_R[2, 0] =  Delta * np.exp(-1j*phi)
    onsite_R[3, 1] =  Delta * np.exp(-1j*phi)
    Hs_R = make_block_diagonal(onsite_R) + make_transverse_hopping(alpha, beta)

    def _site_type(ix):
        if ix < NS:             return Hs_L
        elif ix < NS + normal:  return Hn_bl
        else:                   return Hs_R

    bk = 4 * ny   # block size per x-site
    for ix in range(Ntot):
        sl = slice(ix*bk, (ix+1)*bk)
        H[sl, sl] = _site_type(ix)
        if ix < Ntot - 1:
            sl2 = slice((ix+1)*bk, (ix+2)*bk)
            H[sl,  sl2] = V0
            H[sl2, sl ] = Vd

    return H


def majorana_density(alpha_raw, beta_raw, B_field, phi, NS=20):
    """
    Diagonalize the full BdG Hamiltonian and return the probability density
    of the eigenstate closest to E=0 (the putative Majorana / ABS).

    Returns:
      rho   : (Ntot, ny) array — |psi|² summed over 4 internal DOF per site
      E_min : float — energy of the selected state
    """
    H = build_H_total(alpha_raw, beta_raw, B_field, phi, NS=NS)
    E, psi = np.linalg.eigh(H)

    idx  = np.argmin(np.abs(E))
    E_min = E[idx]
    psi0 = psi[:, idx]

    Ntot  = 2*NS + normal
    # reshape: (Ntot*ny, 4) then sum |psi|² over the 4 BdG DOF per (x,y) site
    # basis order: site ix, transverse iy, DOF d  →  flat index ix*(ny*4) + iy*4 + d
    rho_flat = np.abs(psi0.reshape(Ntot * ny, 4))**2   # (Ntot*ny, 4)
    rho = rho_flat.sum(axis=1).reshape(Ntot, ny)        # (Ntot, ny)

    return rho, E_min

def build_H_normal(k, ny_b, t_b, mu_b, alpha_b, beta_b):
    dim  = 2 * ny_b
    H    = np.zeros((dim, dim), dtype=complex)
    diag = 4*t_b - mu_b + (alpha_b**2 + beta_b**2)/4 - 2*t_b*np.cos(k)

    for i in range(ny_b):
        H[2*i,   2*i  ] = diag
        H[2*i+1, 2*i+1] = diag

    soc_y = 1j * (beta_b + alpha_b) / 2
    for i in range(ny_b - 1):
        ui, di = 2*i,   2*i+1
        uj, dj = 2*i+2, 2*i+3
        H[ui, uj] += -t_b;    H[uj, ui] += -t_b
        H[di, dj] += -t_b;    H[dj, di] += -t_b
        H[ui, dj] +=  soc_y;  H[dj, ui] += -soc_y
        H[di, uj] +=  soc_y;  H[uj, di] += -soc_y

    soc_x = (alpha_b - beta_b) / 2
    val   = -2j * np.sin(k) * soc_x
    for i in range(ny_b):
        H[2*i,   2*i+1] += val
        H[2*i+1, 2*i  ] -= val

    return (H + H.conj().T) / 2


def compute_bands(ny_b=6, a_b=10e-9, m_eff=0.05):
    tunits_b = (1e3 / e) * hbar**2 / (2 * m_eff * m * a_b**2)
    t_b      = 1.0
    mu_b     =  10.0 / tunits_b
    alpha_b  =  10.0e-9 / (a_b * tunits_b)
    beta_b   = -10.0e-9 / (a_b * tunits_b)
    nk       = 201
    k_arr    = np.linspace(-np.pi, np.pi, nk)
    bands    = np.array([
        np.linalg.eigvalsh(build_H_normal(k, ny_b, t_b, mu_b, alpha_b, beta_b))
        for k in k_arr
    ])
    return k_arr, bands, tunits_b

def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos, alpha_raw, beta_raw):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(energies / Delta, phi_vals / np.pi, ldos.T,
                     levels=100, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm, $B=0$',
        loc='right', fontsize=9)
    _style(ax);  plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, alpha_raw, beta_raw, target_energy=0.0):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(phi_vals / np.pi, B_vals, ldos,
                     levels=100, cmap='magma')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\phi\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\omega={target_energy:.3f}$, $\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm',
        loc='right', fontsize=9)
    _style(ax);  plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi, target_energy=0.0):
    fig, ax = plt.subplots(figsize=(7, 4))
    cf = ax.contourf(np.arange(normal), np.arange(ny), ldos_xy.T,
                     levels=100, cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel('x (junction site)', loc='right')
    ax.set_ylabel('y (transverse site)', loc='top', rotation=0, labelpad=12)
    ax.set_title(
        rf'$\phi={phi/np.pi:.2f}\pi$, $\omega={target_energy:.3f}$, $B=0$',
        loc='right', fontsize=9)
    _style(ax);  plt.tight_layout()
    return fig, ax


def plot_majorana_density(rho, E_min, phi, B_field, NS=20):
    Ntot = 2*NS + normal
    fig, ax = plt.subplots(figsize=(8, 4))
    cf = ax.contourf(np.arange(normal + 2), np.arange(ny), rho.T,
                     levels=100, cmap='hot')
    fig.colorbar(cf, ax=ax, label=r'$|\psi|^2$')
    # Mark SC/N interfaces
    ax.axvline(NS - 0.5,          color='cyan', lw=1.2, ls='--', label='SC|N')
    ax.axvline(NS + normal - 0.5, color='cyan', lw=1.2, ls='--')
    ax.set_xlabel('x (site)', loc='right')
    ax.set_ylabel('y (transverse)', loc='top', rotation=0, labelpad=12)
    ax.set_title(
        rf'$|\psi_0|^2$, $E_0={E_min:.4f}$, $\phi={phi/np.pi:.2f}\pi$, $B={B_field:.2f}$ T',
        loc='right', fontsize=9)
    ax.legend(fontsize=8)
    _style(ax);  plt.tight_layout()
    return fig, ax


def plot_bands(k_arr, bands, tunits_b):
    colors = ['tab:orange', 'tab:blue']
    fig, ax = plt.subplots(figsize=(7, 5))
    for j in range(bands.shape[1]):
        ax.plot(k_arr / np.pi, bands[:, j] * tunits_b, c=colors[j % 2], lw=1.0)
    ax.set_xlim(-1.05, 1.05);  ax.set_ylim(-5, 5)
    ax.set_xticks([-1, -0.5, 0.5, 1])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.4))
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$k\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$E$ (eV)', loc='top', rotation=0, labelpad=10)
    ax.tick_params(which='both', direction='in')
    ax.grid(alpha=0.3, which='major')
    plt.tight_layout()
    return fig, ax

#%% Computation and Plots LDOS(B, phi)
if __name__ == '__main__':

    alpha = 14.3e-9
    beta  =  7.3e-9
    B_vals  = np.linspace(0, 3, 41)

    # 1. LDOS(E, phi)  —  B=0
    print("Computing LDOS(E, φ)...")
    ldos_E_phi = calculate_ldos_parallel(alpha, beta)
    plot_ldos_E_phi(ldos_E_phi, alpha, beta)

    # 2. LDOS(B, phi)  —  sweep B at omega=0
    print("Computing LDOS(B, φ)...")
    ldos_B_phi = calculate_ldos_phi_B_parallel(B_vals, alpha, beta, target_energy=0.0)
    plot_ldos_B_phi(ldos_B_phi, B_vals, alpha, beta, target_energy=0.0)

    # 3. LDOS(x, y)  —  B=0, phi=pi, omega=0
    print("Computing LDOS(x, y)...")
    ldos_xy = calculate_ldos_spatial(alpha, beta, phi=np.pi, target_energy=0.0)
    plot_ldos_spatial(ldos_xy, phi=np.pi, target_energy=0.0)

    # 4. Majorana probability density  —  eigh approach
    print("Computing Majorana density...")
    rho, E_min = majorana_density(alpha, beta, B_field=1.2, phi=np.pi, NS=20)
    plot_majorana_density(rho, E_min, phi=np.pi, B_field=1.2, NS=20)

    # 5. Normal-state band structure
    print("Computing band structure...")
    k_arr, bands, tunits_b = compute_bands(ny_b=6, a_b=10e-9, m_eff=0.05)
    plot_bands(k_arr, bands, tunits_b)

    plt.show()
        #%%