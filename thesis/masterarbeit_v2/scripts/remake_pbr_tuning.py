#%% IMPORTS
import os, multiprocessing
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
from IPython.display import display, Math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import PowerNorm
from tqdm import tqdm

os.environ['MKL_NUM_THREADS']     = slurm_cpus
os.environ['OMP_NUM_THREADS']     = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

#%% PHYSICAL CONSTANTS & UNITS
hbar = 1.05e-34
e    = 1.602e-19
m    = 9.1e-31
muB  = 5.78e-2   # meV/T

a      = 20e-9
tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))   # meV → t conversion

#%% SIMULATION PARAMETERS
ny     = 100
normal = 5
nbar   = 0
it     = 100
t      = 1.0

delta_val = 0.25             
Delta     = delta_val / tunits
mu        = 1.0   / tunits       
mu_n      = 0.7   / tunits           
mu_barr   = 0.0  / tunits       

E_z = 0.0
theta_z =  0.35 * np.pi       
g_factor = 10

eta  = 0.025 * Delta
nphi = 41
nw   = 61

n_edge  = 3
start_w = -1.01 * Delta
end_w   =  1.01 * Delta

phi_vals = np.linspace(0, 2 * np.pi, nphi)
energies = np.linspace(start_w, end_w, nw)

_hole_idx = np.array([i for i in range(4 * ny) if i % 4 >= 2])


#%% HAMILTONIAN BUILDERS
def make_transverse_hopping(alpha, beta):
    """(4*ny)x(4*ny) y-direction hopping matrix with SOC."""
    hop_block = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([
        [0,   1j,  0,   0  ],
        [1j,  0,   0,   0  ],
        [0,   0,   0,  -1j ],
        [0,   0,  -1j,  0  ],
    ], dtype=np.complex128) * (alpha + beta) / 2
    hop_4x4  = hop_block + soc
    dim      = 4 * ny
    H_hop    = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(ny - 1):
        H_hop[4*i:4*(i+1), 4*(i+1):4*(i+2)] = hop_4x4
        H_hop[4*(i+1):4*(i+2), 4*i:4*(i+1)] = hop_4x4.conj().T
    return H_hop


def make_block_diagonal(block_4x4):
    """ny copies of a 4x4 block on the diagonal."""
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)


def Hloc_SC(alpha, beta):
    """Superconductor onsite + transverse hopping."""
    onsite_val = 4*t - mu + (alpha**2 + beta**2) / 4
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val;    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val;    onsite[3, 3] = -onsite_val
    onsite[0, 2] =  Delta;         onsite[1, 3] =  Delta
    onsite[2, 0] =  np.conj(Delta); onsite[3, 1] = np.conj(Delta)
    H  = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def Hloc_N_eff(Bz, Ez_x, Ez_y, alpha_eff, beta_eff, theta_z):
    """
    Normal region Hamiltonian in the Scharf-Pientka rotated frame.
      alpha_eff = lambda_soc,  beta_eff = 0
      Bz        = out-of-plane Zeeman (t-units)
      Ez_x, Ez_y = rotated in-plane Zeeman components (t-units)
    """
    onsite_val = 4*t - mu_n + (alpha_eff**2 + beta_eff**2) / 4
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val + Bz
    onsite[1, 1] =  onsite_val - Bz
    onsite[2, 2] = -onsite_val + Bz
    onsite[3, 3] = -onsite_val - Bz
    Bxy_c        = Ez_x + 1j * Ez_y
    onsite[0, 1] =  Bxy_c;           onsite[1, 0] =  np.conj(Bxy_c)
    onsite[2, 3] =  Bxy_c;           onsite[3, 2] =  np.conj(Bxy_c)
    H  = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha_eff, beta_eff)
    return H


def Hloc_B(alpha, beta):
    """Barrier onsite + transverse hopping (no pairing, no Zeeman)."""
    onsite_val = 4*t - mu_barr
    onsite = np.zeros((4, 4), dtype=np.complex128)
    onsite[0, 0] =  onsite_val;    onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val;    onsite[3, 3] = -onsite_val
    H  = make_block_diagonal(onsite)
    H += make_transverse_hopping(alpha, beta)
    return H


def V_mat(alpha, beta):
    """x-direction hopping matrix with SOC."""
    s = 0.5 * (alpha - beta)
    v = np.zeros((4, 4), dtype=np.complex128)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t
    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return make_block_diagonal(v)


#%% SYSTEM INITIALIZATION
def get_static_system(alpha_raw, beta_raw):
    """Build matrices that don't depend on B or energy."""
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)
    V0 = V_mat(alpha, beta)
    Vd = V0.conj().T
    Hs = Hloc_SC(alpha, beta)
    Hb = Hloc_B(alpha, beta)
    Id = np.eye(Hs.shape[0], dtype=np.complex128)
    return Hs, Hb, V0, Vd, Id


def get_Hn(B_field, E_z, theta_z, alpha_raw, beta_raw):
    """
    Normal region Hamiltonian.

    Parameters (all physical, set freely by the user)
    ----------
    E_par   : in-plane Zeeman magnitude (meV),  E_∥ = sqrt(Ez_x² + Ez_y²)
    E_z     : out-of-plane Zeeman magnitude (meV)
    theta_z : in-plane Zeeman azimuthal angle (rad), matches paper Eq. (5)
    alpha_raw, beta_raw : Rashba/Dresselhaus in SI (m·eV / hbar, i.e. nm units)

    The Scharf-Pientka unitary Û(θ_soc) [paper Eq. (4)] is applied internally:
      α → λ_soc,  β → 0,  θ_z → θ_z + θ_soc
    so the caller never needs to worry about the SOC rotation.
    """
    alpha = alpha_raw / (a * tunits)
    beta  = beta_raw  / (a * tunits)

    # Eq. (4): SOC magnitude and rotation angle
    lambda_soc = np.sqrt(alpha**2 + beta**2)
    theta_soc  = np.arctan2(beta, alpha)

    # Eq. (5): effective in-plane angle after Û(θ_soc)
    theta_eff  = theta_z + theta_soc

    # Effective SOC parameters in rotated frame
    alpha_eff  = lambda_soc
    beta_eff   = 0.0
    B_t = B_field * muB * g_factor / tunits
    E_par = np.abs(B_t) 
    # Convert Zeeman magnitudes to t-units and decompose in-plane vector
    Bz   = E_z    / tunits
    E_tu = E_par  / tunits
    Ez_x = E_tu * np.cos(theta_eff)   # E_∥ cos(θ_z + θ_soc)
    Ez_y = E_tu * np.sin(theta_eff)   # E_∥ sin(θ_z + θ_soc)

    return Hloc_N_eff(Bz, Ez_x, Ez_y, alpha_eff, beta_eff, theta_z)


#%% CORE RGF
def _apply_phase_vectorized(gSR, phi, dim, hole_idx=_hole_idx):
    """U[-φ] @ gSR @ U[+φ]  (Mathematica second-block convention)."""
    U = np.eye(dim, dtype=np.complex128)
    U[hole_idx, hole_idx] = np.exp(1j * phi)
    return U.conj().T @ gSR @ U


def sancho(H, alpha_0, beta_0, w, eta, Id):
    """Sancho-Rubio iterative surface Green function."""
    z     = w + 1j * eta
    zI    = z * Id
    eps   = H.copy();  eps_s = H.copy()
    alpha = alpha_0.copy();  beta = beta_0.copy()
    tol   = 1e-14
    for _ in range(it):
        g   = la.inv(zI - eps)
        ab  = alpha @ g @ beta
        ba  = beta  @ g @ alpha
        eps   += ab + ba
        eps_s += ab
        alpha  = alpha @ g @ alpha
        beta   = beta  @ g @ beta
        if la.norm(alpha, np.inf) < tol and la.norm(beta, np.inf) < tol:
            break
    return np.linalg.inv(zI - eps_s)


def _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, Id, phi_vals, eta,
                    nbar=nbar, normal=normal, ny=ny,
                    spatial=False, phi_single=None, n_edge=n_edge):
    z        = w + 1j * eta
    dim      = Hs.shape[0]
    zI       = z * Id
    hole_idx = np.array([i for i in range(dim) if i % 4 >= 2])
    edge_idx = np.arange(4*(ny - n_edge), 4*ny)
    # SC surface Green functions
    gSL = sancho(Hs, Vd, V0, w, eta, Id)
    gSR = sancho(Hs, V0, Vd, w, eta, Id)

    # Advance through nbar barrier/SC layers
    if nbar > 0:
        for _ in range(nbar):
            gSL = np.linalg.inv(zI - Hb - Vd @ gSL @ V0)
            gSR = np.linalg.inv(zI - Hb - V0 @ gSR @ Vd)
    else: 
            gSL = np.linalg.inv(zI - Hs - Vd @ gSL @ V0)
            gSR = np.linalg.inv(zI - Hs - V0 @ gSR @ Vd)   

    # Left-to-right propagation through normal region
    glr    = np.empty((normal, dim, dim), dtype=np.complex128)
    glr[0] = gSL
    for i in range(1, normal):
        glr[i] = np.linalg.inv(zI - Hn - Vd @ glr[i-1] @ V0)

    phases = np.array([phi_single]) if (spatial and phi_single is not None) else phi_vals
    nphi_  = len(phases)

    result = (np.zeros((nphi_, normal, ny), dtype=np.float64) if spatial
              else np.zeros(nphi_, dtype=np.float64))

    # Right-to-left propagation
    grl = np.empty((normal, nphi_, dim, dim), dtype=np.complex128)
    for iphi, phi in enumerate(phases):
        grl[-1, iphi] = _apply_phase_vectorized(gSR, phi, dim, hole_idx)
        for i in range(normal - 2, -1, -1):
            grl[i, iphi] = np.linalg.inv(zI - Hn - V0 @ grl[i+1, iphi] @ Vd)

    # Dress each layer and accumulate LDOS
    for i in range(normal):
        left_dressed = zI - Hn - Vd @ glr[i] @ V0
        for iphi in range(nphi_):
            G = np.linalg.inv(left_dressed - V0 @ grl[i, iphi] @ Vd)
            if spatial:
                diag = np.diagonal(G).reshape(ny, 4)
                result[iphi, i, :] = -np.imag(diag.sum(axis=1)) / np.pi
            else:
                # Trace over first n_edge y-sites — matches Mathematica [[1;;4*n_edge,1;;4*n_edge]]
                G_edge = G[np.ix_(edge_idx, edge_idx)]
                result[iphi] += -np.imag(np.trace(G_edge)) / np.pi

    return result[0] if (spatial and phi_single is not None) else result


#%% DRIVERS
def calculate_ldos_sequential(Hs, Hb, V0, Vd, Id,
                               alpha_raw, beta_raw,
                               B_field, E_z, theta_z,
                               energy_array=None):
    if energy_array is None:
        energy_array = energies
    Hn = get_Hn(B_field, E_z, theta_z, alpha_raw, beta_raw)
    results = [
        _ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, Id,
                        phi_vals=phi_vals, eta=eta,
                        nbar=nbar, normal=normal, ny=ny, spatial=False)
        for w in tqdm(energy_array, desc='LDOS(E,φ)')
    ]
    return np.array(results)


def calculate_ldos_phi_B_sequential(Hs, Hb, V0, Vd, Id,
                                     B_vals, alpha_raw, beta_raw,
                                     E_z, theta_z,
                                     target_energy=0.0):
    """Sweeps B magnitude; E_par is recomputed from B at each step."""
    results = []
    for B in tqdm(B_vals, desc='LDOS(B,φ)'):
        Hn  = get_Hn(B, E_z, alpha_raw, beta_raw, theta_z)
        res = _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, Id,
                              phi_vals=phi_vals, eta=eta,
                              nbar=nbar, normal=normal, ny=ny, spatial=False)
        results.append(res)
    return np.array(results)


def calculate_ldos_spatial(Hs, Hb, V0, Vd, Id,
                            alpha_raw, beta_raw,
                            E_z, theta_z, B_field,
                            phi=np.pi, target_energy=0.0):
    Hn = get_Hn(B_field, E_z, theta_z, alpha_raw, beta_raw)
    return _ldos_one_slice(target_energy, Hs, Hb, Hn, V0, Vd, Id,
                           phi_vals=phi_vals, eta=eta,
                           nbar=nbar, normal=normal, ny=ny,
                           spatial=True, phi_single=phi)


#%% PLOTTING
def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')

def plot_ldos_E_phi(ldos, alpha_raw, beta_raw):
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.imshow(ldos.T, aspect='auto', origin='lower', 
        extent=[energies[0]/Delta, energies[-1]/Delta,
               phi_vals[0]/np.pi,  phi_vals[-1]/np.pi], cmap='magma', interpolation='gaussian',#type: ignore
               norm=PowerNorm(gamma=0.4)
    )
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\alpha={alpha_raw/1e-9:.1f}$ nm, $\beta={beta_raw/1e-9:.1f}$ nm, '
        rf'$E_\parallel={B_field:.3f}$ meV, $E_z={E_z:.3f}$ meV, '
        rf'$\theta_z={theta_z/np.pi:.2f}\pi$, '
        rf'$n_y={ny}$, $n_x={normal}$, $\mu_n={mu_n:.2f}$',
        loc='right', fontsize=8)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, alpha_raw, beta_raw, target_energy=0.0):
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.imshow(
        ldos,
        aspect='auto', origin='lower',
        extent=[phi_vals[0]/np.pi, phi_vals[-1]/np.pi, B_vals[0], B_vals[-1]], #type: ignore
        cmap='magma', interpolation='gaussian',
    )
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\phi\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\omega={target_energy:.3f}$, $\alpha={alpha_raw/1e-9:.1f}$ nm, '
        rf'$\beta={beta_raw/1e-9:.1f}$ nm, $\theta_z={theta_z/np.pi:.2f}\pi$',
        loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi, target_energy=0.0):
    """
    ldos_xy : shape (normal, ny)
    imshow expects (rows, cols) = (ny, normal), so we transpose.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.max(ldos_xy)
    cf = ax.imshow(
        ldos_xy.T / vmax,
        aspect='auto', origin='lower',
        extent=[-0.5, normal - 0.5, -0.5, ny - 0.5], #type: ignore
        cmap='magma', interpolation='gaussian',
    )
    fig.colorbar(cf, ax=ax, label='LDOS (norm.)')
    ax.set_xlabel('x (junction site)', loc='right')
    ax.set_ylabel('y (transverse site)', loc='top', rotation=0, labelpad=12)
    ax.set_title(
        rf'$\phi={phi/np.pi:.2f}\pi$, $\omega={target_energy:.4f}$, '
        rf'$E_\parallel={B_field:.3f}$ meV, $E_z={E_z:.3f}$ meV',
        loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


#%% MAIN
if __name__ == '__main__':
    alpha_1 = 14.3e-9   # m (Rashba)
    beta_1  = 7.3e-9    # m (Dresselhaus)
    lambda_soc = np.sqrt(alpha_1**2 + beta_1**2)

    B_vals  = np.linspace(0, 2.0, 50)
    B_field = 1

    # Print optimal theta_z for SOC ⊥ Zeeman
    _a = alpha_1 / (a * tunits)
    _b = beta_1  / (a * tunits)

    theta_soc_val   = np.arctan2(_b, _a)
    theta_z_optimal = np.pi / 2 - theta_soc_val

    display(Math(fr"\theta_{{soc}} = {theta_soc_val/np.pi:.2f} \, \pi"))
    display(Math(fr"E_{{\parallel}} = {B_field*muB*g_factor/tunits:.3f} \, \text{{meV}}"))
    display(Math(fr"E_z = {E_z:.3f} \, \text{{meV}}"))
    display(Math(fr"\theta_z = {theta_z_optimal/np.pi:.2f} \, \pi"))
    print("_" * 70)
    print(f"Cores: {slurm_cpus}  |  ny={ny} ; normal={normal} ; nbar={nbar}")
    print("_" * 70)

    Hs, Hb, V0, Vd, Id = get_static_system(lambda_soc, 0.0)

    print(r"Computing LDOS$(E, $\phi$)...")
    ldos_E_phi = calculate_ldos_sequential(Hs, Hb, V0, Vd, Id, lambda_soc, 0,
                                            B_field, E_z, theta_z= theta_z_optimal)
    plot_ldos_E_phi(ldos_E_phi, alpha_1, beta_1)
    print(f"✓ Shape: {ldos_E_phi.shape}")

    # print("Computing LDOS(B, φ)...")
    # ldos_B_phi = calculate_ldos_phi_B_sequential(Hs, Hb, V0, Vd, Id, B_vals, alpha_1, beta_1,
    #                                               E_z=E_z, theta_z=theta_z)
    # plot_ldos_B_phi(ldos_B_phi, B_vals, alpha_1, beta_1)
    # print(f"✓ Shape: {ldos_B_phi.shape}")

    print("Computing LDOS(x, y)...")
    ldos_xy = calculate_ldos_spatial(Hs, Hb, V0, Vd, Id, alpha_1, beta_1, E_z=E_z, theta_z=theta_z_optimal, B_field=B_field, phi=np.pi, target_energy=0.0)
    plot_ldos_spatial(ldos_xy, phi=np.pi, target_energy=0.0)
    print(f"✓ Shape: {ldos_xy.shape}")

    plt.show()
    print("_" * 70)
    print("Done.")
    print("_" * 70)
#%%