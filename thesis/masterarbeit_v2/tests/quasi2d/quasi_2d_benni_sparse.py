# %% ============================================================================
#   COMPLETE WORKING SCRIPT: PHYSICALLY CORRECT NESTED vs STEP-BY-STEP LDOS
# ============================================================================

import os
import multiprocessing
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import time
from IPython.display import display, Math
# Core settings
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
os.environ['MKL_NUM_THREADS'] = slurm_cpus
os.environ['OMP_NUM_THREADS'] = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

# %% PARAMETERS
ny     = 100
a      = 20e-9
normal = 5
nbar   = 1

hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))
it = 50

delta_val = 0.25
Delta = delta_val / tunits
t     = 1.0 

B_field = 0.0
mu      = 1.0 / tunits
mu_n    = .7 / tunits
mu_barr = 0.0 / tunits
theta   = 0.15 * np.pi

eta  = 0.025 * Delta
nphi = 51
nw   = 51

ny_trace = 20
start_w = -1.1 * Delta 
end_w   = 1.1 * Delta
phi_vals = np.linspace(0, 2 * np.pi, nphi)
energies = np.linspace(start_w, end_w, nw)

_I4 = np.eye(4 * ny, dtype=np.complex128)
_hole_idx = np.array([i for i in range(4 * ny) if i % 4 >= 2])

# %% HAMILTONIAN BUILDERS
def make_transverse_hopping(alpha, beta):
    hop_block = np.diag([-t, -t, t, t]).astype(complex)
    soc = np.array([
        [0,  1j, 0,   0  ],
        [1j,  0,  0,   0  ],
        [0,   0,  0,  -1j ],
        [0,   0, -1j,  0  ]
    ], dtype=complex) * (alpha + beta) / 2
    
    hop_4x4 = hop_block + soc
    dim_total = 4 * ny
    H_hop = np.zeros((dim_total, dim_total), dtype=np.complex128)
    for i in range(ny - 1):
        H_hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = hop_4x4
        H_hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = hop_4x4.conj().T
    return H_hop

def make_block_diagonal(block_4x4):
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)

def Hloc_SC(alpha, beta):
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=np.complex128)
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
    onsite_val = 4*t - mu_barr
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val
    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val
    onsite[3, 3] = -onsite_val
    H = make_block_diagonal(onsite)
    H += make_transverse_hopping(0, 0)
    return H

def V_mat(alpha, beta):
    s = 0.5 * (alpha - beta)
    v = np.zeros((4, 4), dtype=np.complex128)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t
    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return make_block_diagonal(v)

def get_static_system(alpha_raw, beta_raw):
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
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    Bz    = B_field * theta * muB / tunits 
    Bxy   = B_field * muB / tunits
    return Hloc_N(Bz, Bxy, alpha, beta)

# %% SANCHO-RUBIO SURFACE GREEN'S FUNCTION
def sancho_gf(H, V_left, V_right, w, eta, Id, max_iter=50, tol=1e-12):
    zI = (w + 1j * eta) * Id
    g0 = np.linalg.inv(zI - H)
    eps = H + (V_left @ g0 @ V_right) + (V_right @ g0 @ V_left)
    eps_s = H + (V_left @ g0 @ V_right)
    alpha = V_left @ g0 @ V_left
    beta = V_right @ g0 @ V_right
    
    for iteration in range(max_iter):
        g = np.linalg.inv(zI - eps)
        ab = alpha @ g @ beta
        ba = beta @ g @ alpha
        eps += ab + ba
        eps_s += ab
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if la.norm(alpha, np.inf) < tol:
            break
    return np.linalg.inv(zI - eps_s)

def _apply_phase_explicit(gSR, phi, dim, hole_idx):
    """Explicitly builds Nambu transformation matrix U[-φ] @ gSR @ U[φ]"""
    U = np.eye(dim, dtype=np.complex128)
    U[hole_idx, hole_idx] = np.exp(1j * phi)
    return U @ gSR @ U.conj().T

# %% VERSION A: EXPLICIT PHYSICAL NESTED LOOPS
def calculate_ldos_nested(Hs, Hb, V0, Vd, Id, Hn, phi_vals, energies, 
                          nbar=1, ny_trace=20, normal=1):
    nw = len(energies)
    nphi = len(phi_vals)
    dim = Hs.shape[0]
    hole_idx = np.array([i for i in range(dim) if i % 4 >= 2])
    ldos_result = np.zeros((nw, nphi), dtype=np.float64)
    
    for iw, w in enumerate(tqdm(energies, desc='nested loops')):
        zI = (w + 1j * eta) * Id
        
        gSL = sancho_gf(Hs, Vd, V0, w, eta, Id)
        gSR = sancho_gf(Hs, V0, Vd, w, eta, Id)
        
        if nbar > 0:
            for _ in range(nbar):
                gSL = np.linalg.inv(zI - Hb - Vd @ gSL @ V0)
                gSR = np.linalg.inv(zI - Hb - V0 @ gSR @ Vd)
        
        glr = np.empty((normal, dim, dim), dtype=np.complex128)
        glr[0] = gSL 
        for i in range(1, normal):
            glr[i] = np.linalg.inv(zI - Hn - Vd @ glr[i-1] @ V0)
        
        for iphi, phi in enumerate(phi_vals):
            gSR_phased = _apply_phase_explicit(gSR, phi, dim, hole_idx)          
            
            grl = np.empty((normal, dim, dim), dtype=np.complex128)
            grl[-1] = gSR_phased
            for i in range(normal - 2, -1, -1):
                grl[i] = np.linalg.inv(zI - Hn - V0 @ grl[i+1] @ Vd)
            
            ldos_sum = 0.0
            for i in range(normal):
                G_inv = (zI - Hn - Vd @ glr[i] @ V0) - V0 @ grl[i] @ Vd
                G = np.linalg.inv(G_inv)
                
                tr_G = np.trace(G)
                ldos_sum += -np.imag(tr_G) / np.pi 
                
            ldos_result[iw, iphi] = ldos_sum
    
    return ldos_result

# %% VERSION B: PHYSICALLY TRANSPARENT STEP-BY-STEP ARRAYS (NO CRYPTIC BROADCASTING)
def calculate_ldos_step_by_step(Hs, Hb, V0, Vd, Id, Hn, phi_vals, energies, 
                                 nbar=1, normal=1):
    nw = len(energies)
    nphi = len(phi_vals)
    dim = Hs.shape[0]
    hole_idx = np.array([i for i in range(dim) if i % 4 >= 2])
    ldos_result = np.zeros((nw, nphi), dtype=np.float64)
    
    for iw, w in enumerate(tqdm(energies, desc='phase isolated')):
        zI = (w + 1j * eta) * Id
        gSL = sancho_gf(Hs, Vd, V0, w, eta, Id)
        gSR = sancho_gf(Hs, V0, Vd, w, eta, Id)
        
        if nbar > 0:
            for _ in range(nbar):
                gSL = np.linalg.inv(zI - Hb - Vd @ gSL @ V0)
                gSR = np.linalg.inv(zI - Hb - V0 @ gSR @ Vd)
        
        glr = np.empty((normal, dim, dim), dtype=np.complex128)
        for i in range(1, normal):
            glr[i] = np.linalg.inv(zI - Hn - Vd @ glr[i-1] @ V0)
        
        # Pre-allocate phase dimension explicitly to avoid any python garbage collection drops
        grl = np.zeros((normal, nphi, dim, dim), dtype=np.complex128)
        
        # Populate right lead phase boundaries cleanly
        for iphi, phi in enumerate(phi_vals):
            grl[-1, iphi] = _apply_phase_explicit(gSR, phi, dim, hole_idx)
        
        # Recurrence down through spatial slices for each phase coordinate independently
        for i in range(normal - 2, -1, -1):
            for iphi in range(nphi):
                G_inv = zI - Hn - V0 @ grl[i+1, iphi] @ Vd
                grl[i, iphi] = np.linalg.inv(G_inv)
        
        # Final physical trace synthesis loop
        for i in range(normal):
            for iphi in range(nphi):
                G_inv = (zI - Hn - Vd @ glr[i] @ V0) - V0 @ grl[i, iphi] @ Vd
                G = np.linalg.inv(G_inv)
                tr_G = np.trace(G)
                ldos_result[iw, iphi] += -np.imag(tr_G) / np.pi
    
    return ldos_result

# %% EXECUTION AND PHYSICAL VERIFICATION
if __name__ == '__main__':
    alpha_1 = 14.3e-9
    beta_1  = 7.3e-9
    
    Hs, Hb, V0, Vd, Id = get_static_system(alpha_1, beta_1)
    Hn = get_Hn(B_field, alpha_1, beta_1)
    
    test_energies = energies
    test_phi_vals = phi_vals
    
    ldos_nested = calculate_ldos_nested(
        Hs, Hb, V0, Vd, Id, Hn, test_phi_vals, test_energies,
        nbar=nbar, normal=normal
    )
    
    ldos_step = calculate_ldos_step_by_step(
        Hs, Hb, V0, Vd, Id, Hn, test_phi_vals, test_energies,
        nbar=nbar, normal=normal
    )
    
    display(Math(rf"\alpha = {alpha_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\beta = {beta_1*1e9:.2f}\text{{ nm}}"))
    display(Math(rf"\mu_S = {mu:.2f}"))
    display(Math(rf"\mu_N = {mu_n:.2f}"))
    print()

    # Numerical validation test
    diff = np.abs(ldos_nested - ldos_step)
    print("\n" + "=" * 80)
    print("PHYSICS CONSISTENCY CHECK")
    print("=" * 80)
    print(f"Max numerical delta between layouts: {diff.max():.2e}")
    
    if diff.max() < 1e-11:
        print("MATRICES ARE IDENTICAL\n")
    else:
        print("Discrepancy detected.\n")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].pcolormesh(test_energies / Delta, test_phi_vals / np.pi, 
                            ldos_nested.T, cmap='inferno', shading='auto')
    axes[0].set_title('Calculated LDOS Spectrum', fontweight='bold')
    axes[0].set_ylabel(r'$\phi / \pi$')
    axes[0].set_xlabel(r'$\omega / \Delta$')
    fig.colorbar(im0, ax=axes[0])
    
    axes[1].plot(test_energies / Delta, ldos_nested[:, len(test_phi_vals)//4], label='Slice at $\phi/2$')
    axes[1].set_title('Energy Line Cut', fontweight='bold')
    axes[1].set_xlabel(r'$\omega / \Delta$')
    axes[1].set_ylabel('Raw Trace Value')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
#%%