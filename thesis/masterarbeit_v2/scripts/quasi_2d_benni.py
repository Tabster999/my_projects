#%% IMPORTS
import os, multiprocessing
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
from IPython.display import display, Math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import cProfile
import pstats

# Allow BLAS/MKL to use all allocated cores for tensor-batched math operations
os.environ['MKL_NUM_THREADS'] = slurm_cpus
os.environ['OMP_NUM_THREADS'] = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

#%% PARAMETERS
ny     = 100
a      = 20e-9
normal = 5
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))

it = 100

delta_val = 0.25
Delta = delta_val / tunits
t     = 1.0 

B_field = 0.0
mu      = 2.0 / tunits
mu_n    = .1 #/ tunits 
mu_barr = -0.0 / tunits
theta_z   = np.pi / 2 / 0.15 * np.pi

eta  = 0.025 * Delta
nphi = 31
nw   = 61

n_edge = 5
start_w = -1.05 * Delta 
end_w   = 1.05 * Delta
phi_vals = np.linspace(0, 2 * np.pi, nphi)
energies = np.linspace(start_w, end_w, nw)

_I4 = np.eye(4 * ny, dtype=complex)

_elec_idx = np.array([i for i in range(4 * ny) if i % 4 <  2])
_hole_idx = np.array([i for i in range(4 * ny) if i % 4 >= 2])


#%% HAMILTONIAN BUILDERS
def make_transverse_hopping(alpha, beta):
    """Build dense 4x4 transverse hopping + SOC block."""
    hop_block = np.diag([-t, -t, t, t]).astype(complex)
    soc = np.array([
        [0,  1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,  -1j ],
        [0,   0, -1j,  0  ]
    ], dtype=complex) * (alpha + beta) / 2
    
    hop_4x4 = hop_block + soc
    
    dim_total = 4 * ny
    H_hop = np.zeros((dim_total, dim_total), dtype=complex)
    for i in range(ny - 1):
        H_hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = hop_4x4
        H_hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = hop_4x4.conj().T
    
    return H_hop


def make_block_diagonal(block_4x4):
    """Make dense block diagonal matrix with ny copies."""
    return np.kron(np.eye(ny, dtype=complex), block_4x4)


def Hloc_SC(alpha, beta):
    """Superconductor onsite + hopping."""
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val
    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val
    onsite[3, 3] = -onsite_val
    onsite[0, 2] =  Delta
    onsite[1, 3] =  Delta
    onsite[2, 0] =  np.conjugate(Delta)
    onsite[3, 1] =  np.conjugate(Delta)
    
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_N(Bz, Bxy, alpha, beta):
    """Normal region onsite + hopping."""
    onsite_val = 4*t - mu_n + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val + Bz
    onsite[1, 1] =  onsite_val - Bz
    onsite[2, 2] = -onsite_val + Bz
    onsite[3, 3] = -onsite_val - Bz
    onsite[0, 1] =  Bxy * np.exp( 1j*theta_z)
    onsite[1, 0] =  Bxy * np.exp(-1j*theta_z)
    onsite[2, 3] =  Bxy * np.exp(1j*theta_z)
    onsite[3, 2] =  Bxy * np.exp(-1j*theta_z)
    
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_B(alpha, beta):
    """Barrier onsite + hopping."""
    onsite_val = 4*t - mu_barr
    onsite = np.zeros((4, 4), dtype=complex)
    onsite[0, 0] =  onsite_val
    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val
    onsite[3, 3] = -onsite_val
    
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



#%% SYSTEM INITIALIZATION LOGIC
def get_static_system(alpha_raw, beta_raw):
    """Builds the parts of the system that don't depend on B or Energy."""    
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    
    V0 = V_mat(alpha, beta)
    Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta)
    Hb = Hloc_B(alpha, beta)

    dim = Hs.shape[0]
    Id = np.eye(dim, dtype=complex)
    return Hs, Hb, V0, Vd, Id


def get_Hn(B_field, alpha_raw, beta_raw):
    """Builds ONLY the normal region Hamiltonian which depends on B."""
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
        
    Bxy = 0.5 * B_field * muB / tunits
    Bz  = 0.5 * B_field * muB * theta_z / tunits
    

    return Hloc_N(Bz, Bxy, alpha, beta)


#%% CORE RECURSIVE GREEN FUNCTION SLICE
def _apply_phase_vectorized(gSR, phi, dim, hole_idx=_hole_idx):
    """Explicitly builds Nambu transformation matrix U[-φ] @ gSR @ U[φ]"""
    U = np.eye(dim, dtype=complex)
    U[hole_idx, hole_idx] = np.exp(1j * phi)
    return U.conj().T @ gSR @ U


def sancho(H, alpha_0, beta_0, w, eta, Id):
    """Optimized Dense Sancho-Rubio surface Green function."""
    z = w + 1j * eta
    zI = z * Id

    eps   = H.copy()
    eps_s = H.copy()
    alpha = alpha_0.copy()
    beta  = beta_0.copy()
    tol = 1e-14

    for _ in range(it):
        # g = la.solve(zI - eps, Id, assume_a='gen')
        g = la.inv(zI - eps)
        ab = alpha @ g @ beta
        ba = beta @ g @ alpha

        eps   += ab + ba
        eps_s += ab

        alpha = alpha @ g @ alpha
        beta  = beta  @ g @ beta

        #if max(np.linalg.norm(alpha), np.linalg.norm(beta)) < 1e-14:
        if la.norm(alpha, np.inf) < tol and la.norm(beta, np.inf) < tol:
            break

    return np.linalg.inv(zI - eps_s) # la.solve(zI - eps_s, Id, assume_a='gen')


import numpy as np

def _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, Id, phi_vals, eta, nbar=nbar, normal=normal, ny=ny, spatial=False, phi_single=None, n_edge=n_edge):
    """
    Calculates the Local Density of States (LDOS) for a single energy slice 'w'.
    Supports both total integrated trace or spatially resolved LDOS across 'normal' layers and 'ny' channels.
    """
    z = w + 1j * eta
    dim = Hs.shape[0]
    zI = z * Id
    hole_idx = np.array([i for i in range(dim) if i % 4 >= 2])

    gSL = sancho(Hs, Vd, V0, w, eta, Id)
    gSR = sancho(Hs, V0, Vd, w, eta, Id)

    if nbar > 0:
        for _ in range(nbar):
            gSL = np.linalg.inv(zI - Hs - Vd @ gSL @ V0)
            gSR = np.linalg.inv(zI - Hs - V0 @ gSR @ Vd)

    glr = np.empty((normal, dim, dim), dtype=complex)
    glr[0] = gSL
    for i in range(1, normal):
        glr[i] = np.linalg.inv(zI - Hn - Vd @ glr[i-1] @ V0)
        
    if spatial and phi_single is not None:
        phases = np.array([phi_single])
    else:
        phases = phi_vals

    nphi = len(phases)
    
    if spatial:
        # Resolves shape: (phases, layers, y_coordinates)
        result = np.zeros((nphi, normal, ny), dtype=np.float64)
    else:
        # Integrated trace format matching calculate_ldos_nested
        result = np.zeros(nphi, dtype=np.float64)

    grl = np.empty((normal, nphi, dim, dim), dtype=complex)
    for iphi, phi in enumerate(phases):
        grl[-1, iphi] = _apply_phase_vectorized(gSR, phi, dim, hole_idx)   

        # Right-to-Left layer propagation
        for i in range(normal - 2, -1, -1):
            G_inv = zI - Hn - V0 @ grl[i+1, iphi] @ Vd
            grl[i, iphi] = np.linalg.inv(G_inv)
            
    for i in range(normal):
        # Pre-compute phase-independent term once per layer
        left_dressed = zI - Hn - Vd @ glr[i] @ V0
        
        for iphi in range(nphi):
            G_inv = left_dressed - V0 @ grl[i, iphi] @ Vd
            G = np.linalg.inv(G_inv)
            
            if spatial:
                diag = np.diagonal(G)
                diag_spatial = diag.reshape(ny, 4)
                result[iphi, i, :] = -np.imag(diag_spatial.sum(axis=1)) / np.pi
            else:
                result[iphi] += -np.imag(np.trace(G[:20, :20])) / np.pi

    return result[0] if (spatial and phi_single is not None) else result


#%% HIGH-LEVEL SEQUENTIAL DRIVERS (CLUSTER-SAFE)
def calculate_ldos_sequential(Hs, Hb, V0, Vd, Id, alpha_raw, beta_raw, B_val, energy_array=None):
    if energy_array is None:
        energy_array = energies
    
    # Build normal matrix once for the chosen B field
    Hn = get_Hn(B_val, alpha_raw, beta_raw)
    
    results = [
        _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, Id, phi_vals=phi_vals, eta=eta, 
                        nbar=nbar, normal=normal, ny=ny, spatial=False)
        for w in tqdm(energy_array, desc='LDOS(E,φ)')
    ]
    return np.array(results)


def calculate_ldos_phi_B_sequential(Hs, Hb, V0, Vd, Id, B_vals, alpha_raw, beta_raw, target_energy=0.0):
    results = []
    # Loop updates Hn purely when B changes, reusing all static matrices
    for B in tqdm(B_vals, desc='LDOS(B,φ)'):
        Hn = get_Hn(B, alpha_raw, beta_raw)
        res = _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, Id, phi_vals=phi_vals, 
                              eta=eta, nbar=nbar, normal=normal, ny=ny, spatial=False)
        results.append(res)
    return np.array(results)


def calculate_ldos_spatial(Hs, Hb, V0, Vd, Id, alpha_raw, beta_raw, B_val, phi, target_energy=0.0):
    Hn = get_Hn(B_val, alpha_raw, beta_raw)
    # spatial=True and phi_single=phi isolates a single slice array of shape (normal, ny)
    return _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, Id, phi_vals=phi_vals, 
                          eta=eta, nbar=nbar, normal=normal, ny=ny, 
                          spatial=True, phi_single=phi)


#%% PLOTTING FUNCTIONS
def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos, alpha_raw, beta_raw, ny=ny):
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = cf = ax.imshow(
    ldos.T,
    aspect='auto',
    origin='lower',
    extent=[energies[0]/Delta, energies[-1]/Delta, phi_vals[0]/np.pi, phi_vals[-1]/np.pi], #type: ignore
    cmap='magma',
    interpolation='gaussian'  # or 'bilinear', 'bicubic'
)
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(rf'$\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm, $B={B_field:.2f}$, $n_y={ny:}$, $n_x={normal}$, $\mu_n={mu_n:.2f}$, $\mu_s={mu:.2f}$', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, alpha_raw, beta_raw, target_energy=0.0):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(phi_vals / np.pi, B_vals, ldos, levels=100, cmap='magma')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\phi\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(rf'$\omega={target_energy:.3f}$, $\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi, target_energy=0.0):
    fig, ax = plt.subplots(figsize=(7, 4))
    max = np.max(ldos_xy)
    cf = ax.contourf(np.arange(normal), np.arange(ny), ldos_xy.T / max, cmap='magma', levels=100)#shading='auto', cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel('x (junction site)', loc='right')
    ax.set_ylabel('y (transverse site)', loc='top', rotation=0, labelpad=12)
    ax.set_title(rf'$\phi={phi/np.pi:.2f}\pi$, $\omega={target_energy:.3f}$, $B={B_field:.2f}$', loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


#%% MAIN RUNNER
if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    alpha_1 = 14.3e-9
    beta_1  = 7.3e-9


    B_vals  = np.linspace(0, 2.0, 50)
    B_field = 5.0
    
    print("=" * 70)
    print(f"Native Thread-Pool Allocation: {slurm_cpus} cores")
    print("=" * 70)
    display(Math(rf"\alpha = {alpha_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\beta = {beta_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\mu_S = {mu:.2f}"))
    display(Math(rf"\mu_N = {mu_n:.2f}"))
    print()

    print("Pre-computing static Hamiltonians...")
    Hs, Hb, V0, Vd, Id = get_static_system(alpha_1, beta_1)

    print("Computing LDOS(E, φ) sequentially...")
    ldos_E_phi = calculate_ldos_sequential(Hs, Hb, V0, Vd, Id, alpha_1, beta_1, B_val=B_field)
    plot_ldos_E_phi(ldos_E_phi, alpha_1, beta_1)
    print(f"✓ LDOS(E,φ) complete. Shape: {ldos_E_phi.shape}")
    print()
    
    # print("Computing LDOS(B, φ) sequentially...")
    # ldos_B_phi = calculate_ldos_phi_B_sequential(Hs, Hb, V0, Vd, Id, B_vals, alpha_1, beta_1, target_energy=0.0)
    # plot_ldos_B_phi(ldos_B_phi, B_vals, alpha_1, beta_1, target_energy=0.0)
    # print(f"✓ LDOS(B,φ) complete. Shape: {ldos_B_phi.shape}")
    # print()
    
    # print("Computing LDOS(x, y)...")
    # ldos_xy = calculate_ldos_spatial(Hs, Hb, V0, Vd, Id, alpha_1, beta_1, B_val=B_field, phi=np.pi, target_energy=0.0)
    # plot_ldos_spatial(ldos_xy, phi=np.pi, target_energy=0.0)
    # print(f"✓ LDOS(x,y) complete. Shape: {ldos_xy.shape}")
    # print()
    
    plt.show()
    print("=" * 70)
    print("Execution Finished Successfully!")
    print("=" * 70)
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats(20)
# %%
