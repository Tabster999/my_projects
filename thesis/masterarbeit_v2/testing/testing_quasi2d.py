#%% IMPORTS
from IPython.display import display, Math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
from tqdm import tqdm

#%% PARAMETERS

ny     = 50
a      = 20e-9
normal = 10
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it    = 1000

delta_val = 0.25
Delta = delta_val / tunits
t     = 1.0 

B_field = 0
mu      = 1.0 / tunits
mu_n    = 0.7 / tunits  
mu_barr = 0.0 / tunits
theta   = 0.15 * np.pi

eta  = 0.025 * Delta
nphi = 41
nw   = 71

start_w = -1.01*Delta 
end_w = 1.01*Delta
phi_vals = np.linspace(0, 2*np.pi, nphi)
energies = np.linspace(start_w, end_w, nw)

_I4 = np.eye(4 * ny, dtype=np.complex128)


_hole_idx = np.array([i for i in range(4*ny) if i % 4 >= 2])
_elec_idx = np.array([i for i in range(4*ny) if i % 4 <  2])


#%% HAMILTONIAN BUILDERS

def make_transverse_hopping(alpha, beta):
    """Build 4x4 transverse hopping + SOC block."""
    hop_block = np.diag([-t, -t, t, t]).astype(complex)
    soc = np.array([
        [0,  1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,   -1j ],
        [0,   0, -1j,  0  ]
    ], dtype=complex) * (alpha + beta) / 2
    
    hop_4x4 = hop_block 
    
    # Build full hopping matrix
    dim_total = 4 * ny
    H_hop = np.zeros((dim_total, dim_total), dtype=np.complex128)
    for i in range(ny - 1):
        H_hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = hop_4x4 + soc 
        H_hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = hop_4x4 + soc.conj().T
    
    return H_hop


def make_block_diagonal(block_4x4):
    """Make block diagonal matrix with ny copies."""
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)


def Hloc_SC(alpha, beta):
    """Superconductor onsite + hopping."""
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val
    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val
    onsite[3, 3] = -onsite_val
    onsite[0, 2] =  Delta
    onsite[1, 3] =  np.conjugate(Delta)
    onsite[2, 0] =  Delta
    onsite[3, 1] =  np.conjugate(Delta)
    
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H



def Hloc_N(Bz, Bxy, alpha, beta):
    """Normal region onsite + hopping."""
    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val + Bz
    onsite[1, 1] =  onsite_val - Bz
    onsite[2, 2] = -onsite_val + Bz
    onsite[3, 3] = -onsite_val - Bz
    onsite[0, 1] =  Bxy * np.exp( 1j*theta)
    onsite[1, 0] =  Bxy * np.exp(-1j*theta)
    onsite[2, 3] =  Bxy * np.exp(1j*theta)
    onsite[3, 2] =  Bxy * np.exp(-1j*theta)
    
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_B(alpha, beta):
    """Barrier onsite + hopping."""
    onsite_val = 4*t - mu_barr
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val
    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val
    onsite[3, 3] = -onsite_val
    
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def V_mat(alpha, beta):
    s = 0.5 * (alpha - beta)
    v = np.zeros((4, 4), dtype=np.complex128)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t
    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return make_block_diagonal(v)

# def build_U_phase(phi):
#     """Build U(phi) = diag([1,1,e^(i*phi),e^(i*phi)]) ⊗ I_ny.
    
#     SIMPLE: just full diagonal matrix, no optimization.
#     """
#     U = np.eye(4*ny, dtype=complex)
#     ep = np.exp(1j * phi)
    
#     for i in range(ny):
#         # Hole indices get e^(i*phi)
#         U[4*i + 2, 4*i + 2] = ep
#         U[4*i + 3, 4*i + 3] = ep
    
#    return U
def build_U_phase(phi):

    U = np.diag((1 , 1 , np.exp(1j*phi) , np.exp(1j*phi)))

    return make_block_diagonal(U)

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

#%% SANCHO RECURSION

def sancho(H, alpha_0, beta_0, w, eta):
    """Sancho-Rubio recursion for surface Green's function.
    
    Computes: g = (ω·I - H - Σ)^(-1)
    where Σ is built recursively from alpha, beta.
    """
    z = w + 1j*eta
    dim = H.shape[0]
    zI = z * np.eye(dim, dtype=complex)
    
    eps = H.copy()
    eps_s = H.copy()
    alpha = alpha_0.copy()
    beta = beta_0.copy()
    
    for iteration in range(it):
        gi = inv(zI - eps)
        ab = alpha @ gi @ beta
        ba = beta @ gi @ alpha
        
        eps = eps + ab + ba
        eps_s = eps_s + ab
        alpha = alpha @ gi @ alpha
        beta = beta @ gi @ beta
        
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < 1e-14:
            break
    
    return inv(zI - eps_s)


#%% MAIN LDOS CALCULATION

def _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, spatial=False, phi_single=None):
    """Calculate LDOS at one energy point, sweep all phi (or single phi if spatial)."""
    z  = w + 1j*eta
    zI = z * _I4
    
    # ── Surface Green's functions ────────────────────────────────────────
    gSL0 = sancho(Hs, Vd, V0, w, eta)
    gSR0 = sancho(Hs, V0, Vd, w, eta)
    
    gSL = inv(zI - Hb - Vd @ gSL0 @ V0)
    gSR = inv(zI - Hb - V0 @ gSR0 @ Vd)
    for _ in range(nbar):
        gSL = inv(zI - Hb - Vd @ gSL @ V0)
        gSR = inv(zI - Hb - V0 @ gSR @ Vd)
    
    # ── Left-to-right chain (phi-independent) ────────────────────────────
    glr = [None] * normal
    glr[0] = inv(zI - Hn - Vd @ gSL @ V0)
    for i in range(1, normal):
        glr[i] = inv(zI - Hn - Vd @ glr[i-1] @ V0)
    
    phi_arr = [phi_single] if (spatial and phi_single is not None) else phi_vals
    
    if spatial:
        result = np.zeros((normal, ny))
    else:
        result = np.zeros(nphi)
    
    for iphi, phi in enumerate(phi_arr):
        # ── Phase application (SIMPLE: full matrix multiplication) ────────
        U = build_U_phase(-phi)
        Ud = build_U_phase(phi)
        gSRp = Ud @ gSR @ U
        #gSRp = _apply_phase_fast(gSR, phi)
        # ── Right-to-left chain (phi-dependent) ──────────────────────────
        grl = [None] * normal
        grl[-1] = inv(zI - Hn - V0 @ gSRp @ Vd)
        for j in range(normal - 2, -1, -1):
            grl[j] = inv(zI - Hn - V0 @ grl[j+1] @ Vd)
        
        # ── Full Green's functions ───────────────────────────────────────
        if spatial:
            for j in range(normal):
                if j > 0:
                    SL = Vd @ glr[j-1] @ V0
                else:
                    SL = Vd @ gSL @ V0

                if j < normal-1:
                    SR = V0 @ grl[j+1] @ Vd
                else:
                    SR = V0 @ gSRp @ Vd
                G = inv(zI - Hn - SL - SR)
                # Trace over the 4 BdG DOF at each transverse site
                diag_G = np.diag(G).reshape(ny, 4)
                result[j, :] += -np.imag(diag_G.sum(axis=1)) / np.pi
        else:
            LD = 0.0 + 0j
            for j in range(normal):
                if j > 0:
                    SL = Vd @ glr[j-1] @ V0
                else:
                    SL = Vd @ gSL @ V0


                if j < normal-1:
                    SR = V0 @ grl[j+1] @ Vd
                else:
                    SR = V0 @ gSRp @ Vd
                G = inv(zI - Hn - SL - SR)
                LD += np.trace(G)
            result[iphi] = -np.imag(LD) / np.pi
    
    return result


def _build_hamiltonians(alpha_raw, beta_raw, B_field):
    """Build all Hamiltonians for given parameters."""
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = 10 * B_field * theta * muB / tunits 
    Bxy   = 10 * B_field  * muB / tunits 
    
    V0    = V_mat(alpha, beta)
    Vd    = V0.conj().T
    Hs    = Hloc_SC(alpha, beta)
    Hn    = Hloc_N(Bz, Bxy, alpha, beta)
    Hb    = Hloc_B(alpha, beta)
    
    return Hs, Hb, Hn, V0, Vd


def calculate_ldos_parallel(alpha_raw, beta_raw, B_val, energy_array=None, n_jobs=-1):
    if energy_array is None:
        energy_array = energies
    Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B_field=B_val)
    
    rows = Parallel(n_jobs=1, prefer='threads')(
        delayed(_ldos_one_slice)(w, Hs, Hb, Hn, V0, Vd, spatial=False)
        for w in tqdm(energy_array, desc='LDOS(E,φ)')
    )
    return np.array(rows)


def calculate_ldos_phi_B_parallel(B_vals, alpha_raw, beta_raw, target_energy=0.0, n_jobs=1):
    """LDOS(B, phi) at fixed energy. Returns (nB, nphi)."""
    def _one_B(B):
        Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B)
        return _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, spatial=False)
    
    rows = Parallel(n_jobs=1, prefer='threads')(
        delayed(_one_B)(B)
        for B in tqdm(B_vals, desc='LDOS(B,φ)')
    )
    return np.array(rows)


def calculate_ldos_spatial(alpha_raw, beta_raw, B_val, phi, target_energy=0.0):
    Hs, Hb, Hn, V0, Vd = _build_hamiltonians(alpha_raw, beta_raw, B_field=B_val)
    return _ldos_one_slice(
        target_energy, Hs, Hb, Hn, V0, Vd, spatial=True, phi_single=phi
    )

#%% PLOTTING FUNCTIONS
def _style(ax):
    """Style axis."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos, alpha_raw, beta_raw):
    """Plot LDOS(E, phi)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.pcolormesh(energies / Delta, phi_vals / np.pi, ldos.T,
                     shading='auto', cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm, $B={B_field:.2f}$',
        loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, alpha_raw, beta_raw, target_energy=0.0):
    """Plot LDOS(B, phi)."""
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
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi, target_energy=0.0):
    """Plot LDOS(x, y)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cf = ax.contourf(np.arange(normal), np.arange(ny), ldos_xy.T,
                     levels=100, cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel('x (junction site)', loc='right')
    ax.set_ylabel('y (transverse site)', loc='top', rotation=0, labelpad=12)
    ax.set_title(
        rf'$\phi={phi/np.pi:.2f}\pi$, $\omega={target_energy:.3f}$, $B={B_field:.2f}$',
        loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


#%% MAIN

if __name__ == '__main__':
    
    alpha_1 = 14.3e-9
    beta_1  = 7.3e-9
    B_vals  = np.linspace(0, 2.0, 50)
    B_field = 2
    print("Computing LDOS(E, φ)...")
    display(Math(rf"\alpha = {alpha_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\beta = {beta_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\theta = {theta / np.pi:.2f} \pi"))
    display(Math(rf"\mu_S = {mu:.2f}"))
    display(Math(rf"\mu_N = {mu_n:.2f}"))

    ldos_E_phi = calculate_ldos_parallel(alpha_1, beta_1, B_val=B_field)
    plot_ldos_E_phi(ldos_E_phi, alpha_1, beta_1)
    
    # print("Computing LDOS(B, φ)...")
    # ldos_B_phi = calculate_ldos_phi_B_parallel(B_vals, alpha_1, beta_1, target_energy=0.0)
    # plot_ldos_B_phi(ldos_B_phi, B_vals, alpha_1, beta_1, target_energy=0.0)
    
    print("Computing LDOS(x, y)...")
    ldos_xy = calculate_ldos_spatial(alpha_1, beta_1, B_val=B_field, phi=np.pi, target_energy=0.0)
    plot_ldos_spatial(ldos_xy, phi=np.pi, target_energy=0.0)
    
    plt.show()
    print("Done!")


    # %%
