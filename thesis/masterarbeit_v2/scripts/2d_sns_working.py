#%% IMPORTS
import os
import multiprocessing

slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
os.environ['MKL_NUM_THREADS']      = slurm_cpus
os.environ['OMP_NUM_THREADS']      = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

from IPython.display import display, Math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from scipy.linalg import norm

#%%  PHYSICAL CONSTANTS  (fixed, never touch)
_hbar  = 1.05e-34        # J·s
_e     = 1.602e-19       # C
_m0    = 9.1e-31         # kg
_muB   = 5.78e-2         # meV / T   (Bohr magneton)
#symmetric and antisymmetric coupling, compute conductance from left and right 

#%% PARAMETERS
#  All inputs in physical units; one conversion block produces t-units.
class Params:
    """
    All physical parameters in one place.
    Physical units throughout:
      energies / gaps / chemical potentials  →  meV
      SOC strengths                          →  meV·nm
      lattice constant                       →  nm
      magnetic field                         →  T
      angles                                 →  rad
    """
    def __init__(self):
        # --- lattice & effective mass ---
        self.a       = 20.0          # nm,  lattice constant
        self.m_eff   = 0.038         # m_eff / m0

        # --- superconductor ---
        self.Delta   = 0.25          # meV,  pairing amplitude
        self.mu_S    = 1.0           # meV,  SC chemical potential

        # --- normal region ---
        self.mu_N    = 0.7           # meV,  N chemical potential
        self.alpha   = 14.3          # meV·nm,  Rashba SOC
        self.beta    = 7.3           # meV·nm,  Dresselhaus SOC

        # --- Zeeman (in-plane) ---
        self.g       = 1           # g-factor
        self.B       = 0.398           # T,  magnetic field magnitude
        # theta_z: azimuthal angle of E_∥ in the xy-plane (rad).
        # Set to theta_z_optimal() for the topological phase.
        self.theta_z = 0.35*np.pi          # rad

        # --- Zeeman (out-of-plane, usually 0) ---
        self.E_z     = 0.0           # meV

        # --- junction geometry ---
        self.ny      = 50            # transverse sites  (W = ny * a)
        self.nx      = 5             # normal-region x-slices
        self.n_edge  = 5             # probe width in y  (Lp = n_edge * a)

        # --- numerics ---
        self.eta_rel = 0.025      # broadening η = eta_rel * Delta (dimensionless)
        self.it      = 450       # max Sancho-Rubio iterations
        self.tol     = 1e-15         # Sancho-Rubio convergence tolerance

        # --- scan ranges ---
        self.nphi    = 41
        self.nw      = 51
        self.w_range = 1.0           # scan ±w_range * Delta

    @property
    def a_m(self):
        """Lattice constant in metres."""
        return self.a * 1e-9

    @property
    def tunits(self):
        """Energy scale: t = ħ²/(2 m_eff a²)  in meV."""
        return (_hbar**2 / (2 * self.m_eff * _m0 * self.a_m**2)) * (1e3 / _e)

    @property
    def t(self):
        return 1.0   # hopping in t-units (by definition)

    # --- dimensionless (t-unit) versions of all energy parameters ---
    @property
    def Delta_t(self):   return self.Delta  / self.tunits
    @property
    def mu_S_t(self):    return self.mu_S   / self.tunits
    @property
    def mu_N_t(self):    return self.mu_N   / self.tunits
    @property
    def alpha_t(self):   return self.alpha  / (self.a * self.tunits)
    @property
    def beta_t(self):    return self.beta   / (self.a * self.tunits)
    @property
    def E_par_meV(self): return self.B * _muB * self.g
    @property
    def E_par_t(self):   return self.E_par_meV / self.tunits
    @property
    def E_z_t(self):     return self.E_z / self.tunits
    @property
    def eta(self):       return self.eta_rel * self.Delta_t

    # --- scan arrays ---
    @property
    def phi_vals(self):
        return np.linspace(0, 2*np.pi, self.nphi)
    @property
    def energies(self):
        return np.linspace(-self.w_range * self.Delta_t,
                            self.w_range * self.Delta_t, self.nw)

    # --- optimal Zeeman angle for E_∥ ⊥ n_soc  (Sec. III B) ---
    def theta_z_optimal(self):
        """
        n_soc = (alpha, beta, 0)  →  perpendicular direction = arctan2(alpha, -beta).
        Returns angle in radians.
        """
        return np.arctan2(self.alpha_t, -self.beta_t)

    def summary(self):
        t = self.tunits
        lines = [
            ("Δ",        self.Delta,      "meV", self.Delta_t,   "t"),
            ("μ_S",      self.mu_S,       "meV", self.mu_S_t,    "t"),
            ("μ_N",      self.mu_N,       "meV", self.mu_N_t,    "t"),
            ("α",        self.alpha,      "meV·nm", self.alpha_t, "t"),
            ("β",        self.beta,       "meV·nm", self.beta_t,  "t"),
            ("E_∥",       self.E_par_meV,  "meV", self.E_par_t,   "t"),
            ("E_z",      self.E_z,        "meV", self.E_z_t,     "t"),
        ]

        print("─" * 60)
        print(f"  t_units        = {t:.2f} meV")
        for sym, val, unit, val_t, unit_t in lines:
            print(f"  {sym:<12} = {val:>8.3f} {unit:<8} = {val_t:>8.5f} {unit_t}")
        print(f"  θ_z            = {self.theta_z/np.pi:.2f}π")
        print(f"  θ_z,opt        = {self.theta_z_optimal()/np.pi:.2f}π")
        print(f"  η              = {self.eta:.6f} t  ({self.eta_rel} × Δ)")
        print(f"  n_y={self.ny}, n_x={self.nx}, n_edge={self.n_edge}")
        print(f"  n_φ={self.nphi}, n_w={self.nw}")
        print("─" * 60)


#%%  FUNCTIONS 
#  HAMILTONIAN BUILDERS
#  All functions take explicit t-unit parameters — no globals.

def _make_block_diagonal(block_4x4, ny):
    """ny copies of a 4×4 block on the diagonal."""
    return np.kron(np.eye(ny, dtype=np.complex128), block_4x4)


def _make_transverse_hopping(alpha_t, beta_t, ny, t=1.0):
    """
    (4*ny)×(4*ny) y-direction hopping matrix.
    Basis: (c↑, c↓, c↓†, −c↑†)   [paper Eq. (B6)]
    """
    hop = np.diag([-t, -t, t, t]).astype(np.complex128)
    soc = np.array([
        [0,   1j,  0,   0  ],
        [1j,  0,   0,   0  ],
        [0,   0,   0,  -1j ],
        [0,   0,  -1j,  0  ],
    ], dtype=np.complex128) * (alpha_t + beta_t) / 2

    hop_4x4 = hop + soc
    dim     = 4 * ny
    H       = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(ny - 1):
        H[4*i:4*(i+1), 4*(i+1):4*(i+2)] = hop_4x4
        H[4*(i+1):4*(i+2), 4*i:4*(i+1)] = hop_4x4.conj().T
    return H


def _make_x_hopping(alpha_t, beta_t, ny, t=1.0):
    """
    (4*ny)×(4*ny) x-direction hopping matrix V0.
    Basis: (c↑, c↓, c↓†, −c↑†)   [paper Eq. (B6)]
    """
    s = 0.5 * (alpha_t - beta_t)
    v = np.zeros((4, 4), dtype=np.complex128)
    v[0, 0] = -t;  v[1, 1] = -t
    v[2, 2] =  t;  v[3, 3] =  t

    v[0, 1] = -s;  v[1, 0] =  s
    v[2, 3] =  s;  v[3, 2] = -s
    return _make_block_diagonal(v, ny)


def make_H_SC(p: Params):
    """
    Superconductor onsite + y-hopping.
    Basis: (c↑, c↓, c↓†, −c↑†)
    """
    a_t, b_t = p.alpha_t, p.beta_t
    onsite_val = 4*p.t - p.mu_S_t + (a_t**2 + b_t**2) / 4

    onsite = np.zeros((4, 4), dtype=np.complex128)

    onsite[0, 0] =  onsite_val;       onsite[1, 1] =  onsite_val
    onsite[2, 2] = -onsite_val;       onsite[3, 3] = -onsite_val

    onsite[0, 2] =  p.Delta_t;          onsite[1, 3] = p.Delta_t 
    onsite[2, 0] =  np.conj(p.Delta_t);        onsite[3, 1] = np.conj(p.Delta_t)

    H  = _make_block_diagonal(onsite, p.ny)
    H += _make_transverse_hopping(a_t, b_t, p.ny)
    return H


def make_H_N(p: Params):
    """
    Normal region onsite + y-hopping.
    Zeeman: E_∥ at angle theta_z, plus out-of-plane E_z.
    Basis: (c↑, c↓, c↓†, −c↑†)
    """
    a_t, b_t  = p.alpha_t, p.beta_t
    onsite_val = 4*p.t - p.mu_N_t + (a_t**2 + b_t**2) / 4

    Bz    = p.E_z_t
    Bxy = p.E_par_t * np.exp(1j * p.theta_z)
    onsite = np.zeros((4, 4), dtype=np.complex128)

    onsite[0, 0] =  onsite_val + Bz
    onsite[1, 1] =  onsite_val - Bz
    onsite[2, 2] = -onsite_val + Bz
    onsite[3, 3] = -onsite_val - Bz
    
    # off-diagonal: in-plane Zeeman (sx, sy terms)
    onsite[0, 1] =  Bxy;            onsite[1, 0] =  np.conjugate(Bxy)
    onsite[2, 3] =  Bxy;            onsite[3, 2] =  np.conjugate(Bxy)

    H  = _make_block_diagonal(onsite, p.ny)
    H += _make_transverse_hopping(a_t, b_t, p.ny)
    return H


def make_static_matrices(p: Params):
    """
    Build all energy/phase-independent matrices once.
    Returns: Hs, V0, Vd, Id
      Hs  : SC onsite + y-hopping
      V0  : x-hopping  (right-hop)
      Vd  : x-hopping† (left-hop)
      Id  : identity
    """
    Hs = make_H_SC(p)
    V0 = _make_x_hopping(p.alpha_t, p.beta_t, p.ny)
    Vd = V0.conj().T
    Id = np.eye(4 * p.ny, dtype=np.complex128)
    return Hs, V0, Vd, Id


#  CORE RGF

def _apply_phase(gSR, ny, phi=np.pi):
    """
    U†(φ) @ gSR @ U(φ)
    U acts on hole indices (positions 2,3 mod 4) with e^{iφ}.
    """
    dim      = 4 * ny
    hole_idx = np.array([i for i in range(dim) if i % 4 >= 2])
    U        = np.eye(dim, dtype=np.complex128)
    U[hole_idx, hole_idx] = np.exp(1j * phi)
    return U.conj().T @ gSR @ U


def _edge_indices(ny, n_edge, electron_only=False):
    edge_idx = np.arange(4 * (ny - n_edge), 4 * ny)
    if electron_only:
        return edge_idx[edge_idx % 4 < 2]
    return edge_idx


def _ldos_trace(G, idx):
    return -np.imag(np.trace(G[np.ix_(idx, idx)])) / np.pi


def sancho(H, V_right, V_left, w, eta, Id, it, tol, label=None):
    """
    Sancho-Rubio surface Green's function of a semi-infinite lead.

    Convention: the lead is stacked in the +x direction using V_right
    (hopping from a bulk cell to the NEXT cell further from the junction),
    and V_left = V_right† (hopping back toward the junction).

    For a LEFT lead (extends toward -x, attaches to the junction on its
    right side): call sancho(H, V0, Vd, ...)   → returns gSL
    For a RIGHT lead (extends toward +x, attaches to the junction on its
    left side): call sancho(H, Vd, V0, ...)    → returns gSR

    i.e. V_right is "hopping deeper into the lead", V_left is its dagger.
    This is intentionally opposite of "which physical side" — don't infer
    handedness from the argument name, only from which one you pass first.

    label: optional string ('L' or 'R') — if given, asserts V_left == V_right.conj().T
    to catch accidental swaps.
    """
    if label is not None:
        assert np.allclose(V_left, V_right.conj().T), (
            f"sancho({label}): V_left is not V_right† — arguments are probably swapped"
        )

    z   = w + 1j * eta
    zI  = z * Id
    eps   = H.copy();  eps_s = H.copy()
    alpha = V_left.copy();  beta = V_right.copy()
    for _ in range(it):
        g  = la.inv(zI - eps)
        ab = alpha @ g @ beta
        ba = beta @ g @ alpha
        eps_s += ab
        eps  += ab + ba
        alpha  = alpha @ g @ alpha
        beta   = beta  @ g @ beta
        if norm(alpha, np.inf) < tol and norm(beta, np.inf) < tol:
            break
    return la.inv(zI - eps_s)


def calculate_surface_gfs(H, V_forward, V_backward, w, eta, Id, iterations):
    """
    Direct recursive decimation (Matches DOSBenni.nb).
    V_forward corresponds to V10 (Mathematica)
    V_backward corresponds to V01 (Mathematica)
    """
    zI = (w + 1j * eta) * Id
    # Initialize gSL and gSR
    gSL = la.inv(zI - H)
    gSR = la.inv(zI - H)
    # Recursive loop
    for _ in range(iterations):
        gSR = la.inv(zI - H - V_forward @ gSR @ V_backward)
        gSL = la.inv(zI - H - V_backward @ gSL @ V_forward)
        if norm(V_forward @ gSR @ V_backward, np.inf) < 1e-15 and norm(V_backward @ gSL @ V_forward, np.inf) < 1e-15:
            break
    return gSL, gSR


def _ldos_one_energy(w, Hs, Hn, V0, Vd, Id, p: Params, phi_vals, electron_only=True):
    """
    LDOS at one energy w, for all phi in phi_vals.
    Probes the last n_edge y-sites (top edge of N region), all nx x-slices.
    Returns array of shape (nphi,).
    """
    z   = w + 1j * p.eta
    zI  = z * Id
    ny  = p.ny
    nx  = p.nx
    dim = 4 * ny
    edge_idx = _edge_indices(ny, p.n_edge, electron_only=electron_only)

    gSL0 = sancho(Hs, Vd, V0, w, p.eta, Id, p.it, p.tol, label='L')
    gSR0 = sancho(Hs, V0, Vd, w, p.eta, Id, p.it, p.tol, label='R')

    #gSL0, gSR0 = calculate_surface_gfs(Hs, V0, Vd, w, p.eta, Id, p.it)
    glr    = np.empty((nx, dim, dim), dtype=np.complex128)
    glr[0] = la.inv(zI - Hn - Vd @ gSL0 @ V0)  # Slice 0 connected to left lead
    for i in range(1, nx):
        glr[i] = la.inv(zI - Hn - Vd @ glr[i-1] @ V0)

    # --- right-to-left propagation for each phi ---
    nphi   = len(phi_vals)
    grl    = np.empty((nx, nphi, dim, dim), dtype=np.complex128)
    for iphi, phi in enumerate(phi_vals):
        gSR_phase = _apply_phase(gSR0, ny, phi)
        grl[-1, iphi] = la.inv(zI - Hn - V0 @ gSR_phase @ Vd)        
        for i in range(nx - 2, -1, -1):
            grl[i, iphi] = la.inv(zI - Hn - V0 @ grl[i+1, iphi] @ Vd)

    # --- dress each x-slice and accumulate edge LDOS ---
    result = np.zeros(nphi, dtype=np.float64)
    for i in range(nx):
        if i == 0:
            # Determine left environment self-energy
            Sigma_L = Vd @ gSL0 @ V0
        else:
            Sigma_L = Vd @ glr[i-1] @ V0
        
        for iphi in range(nphi):
            # Determine right environment self-energy
            if i == nx - 1:
                gSR_phased = _apply_phase(gSR0, ny, phi_vals[iphi])
                Sigma_R = V0 @ gSR_phased @ Vd
            else:
                Sigma_R = V0 @ grl[i+1, iphi] @ Vd
                
            # Full Green's Function at slice i
            G = la.inv(zI - Hn - Sigma_L - Sigma_R)
            result[iphi] += _ldos_trace(G, edge_idx)

    return result


def _ldos_one_energy_spatial(w, phi, Hs, Hn, V0, Vd, Id, p: Params):
    """
    Spatial LDOS at one energy w and one phi.
    Returns array of shape (nx, ny): LDOS per x-slice per y-site.
    """
    z   = w + 1j * p.eta
    zI  = z * Id
    ny  = p.ny
    nx  = p.nx
    dim = 4 * ny

    gSL = sancho(Hs, Vd, V0, w, p.eta, Id, p.it, p.tol)
    gSR = sancho(Hs, V0, Vd, w, p.eta, Id, p.it, p.tol)

    glr    = np.empty((nx, dim, dim), dtype=np.complex128)
    glr[0] = gSL
    for i in range(1, nx):
        glr[i] = la.inv(zI - Hn - Vd @ glr[i-1] @ V0)

    grl_phi    = np.empty((nx, dim, dim), dtype=np.complex128)
    grl_phi[-1] = _apply_phase(gSR, ny, phi)
    for i in range(nx - 2, -1, -1):
        grl_phi[i] = la.inv(zI - Hn - V0 @ grl_phi[i+1] @ Vd)

    result = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        left_dressed = zI - Hn - Vd @ glr[i] @ V0
        G    = la.inv(left_dressed - V0 @ grl_phi[i] @ Vd)
        diag = np.diagonal(G).reshape(ny, 4)
        result[i, :] = -np.imag(diag.sum(axis=1)) / np.pi

    return result


#  DRIVERS

def compute_ldos_E_phi(p: Params, electron_only=True):
    """
    Main LDOS(E, φ) sweep for Fig. 8-style plots.
    Returns array of shape (nw, nphi).
    """
    Hs, V0, Vd, Id = make_static_matrices(p)
    Hn = make_H_N(p)

    results = [
        _ldos_one_energy(w, Hs, Hn, V0, Vd, Id, p, p.phi_vals, electron_only=electron_only)
        for w in tqdm(p.energies, desc='LDOS(E,φ)')
    ]
    return np.array(results)   # shape (nw, nphi)


def compute_ldos_B_phi(p: Params, B_vals, target_energy=0.0, electron_only=False):
    """
    LDOS(B, φ) at fixed energy. Sweeps B, rebuilds Hn each step.
    Returns array of shape (nB, nphi).
    """
    Hs, V0, Vd, Id = make_static_matrices(p)
    results = []
    for B in tqdm(B_vals, desc='LDOS(B,φ)'):
        p.B  = B
        Hn   = make_H_N(p)
        res  = _ldos_one_energy(target_energy, Hs, Hn, V0, Vd, Id, p, p.phi_vals, electron_only=electron_only)
        results.append(res)
    return np.array(results)   # shape (nB, nphi)


def compute_ldos_spatial(p: Params, phi=np.pi, target_energy=0.0):
    """
    Spatial LDOS(x, y) at fixed phi and energy.
    Returns array of shape (nx, ny).
    """
    Hs, V0, Vd, Id = make_static_matrices(p)
    Hn = make_H_N(p)
    return _ldos_one_energy_spatial(target_energy, phi, Hs, Hn, V0, Vd, Id, p)

#  PLOTTING

def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='out', top=False, right=False)


def plot_ldos_E_phi(ldos, p: Params):
    """ldos: shape (nw, nphi)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    energies = p.energies / p.Delta_t
    phases = p.phi_vals / np.pi
    cf = ax.imshow(ldos.T,
        extent=[energies[0] , energies[-1] , phases[0] , phases[-1] ], #type: ignore
        cmap='jet', 
        aspect='auto',
        interpolation='bilinear'
    )
    fig.colorbar(cf, ax=ax, label='LDOS (a.u.)')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\alpha={p.alpha:.1f}$ nm·meV, $\beta={p.beta:.1f}$ nm·meV, '
        rf'$E_\parallel={p.E_par_meV:.3f}$ meV, $E_z={p.E_z:.3f}$ meV, '
        rf'$\theta_z={p.theta_z/np.pi:.3f}\pi$, '
        rf'$n_y={p.ny}$, $n_x={p.nx}$, $\mu_N={p.mu_N:.2f}$ meV, $\mu_S={p.mu_S:.2f}$ meV',
        loc='right', fontsize=8
    )
    _style(ax)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    return fig, ax


def plot_ldos_B_phi(ldos, B_vals, p: Params, target_energy=0.0):
    """ldos: shape (nB, nphi)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.imshow(
        ldos, aspect='auto', origin='lower',
        extent=[p.phi_vals[0]/np.pi, p.phi_vals[-1]/np.pi, B_vals[0], B_vals[-1]], #type: ignore
        cmap='magma',
        interpolation='gaussian' 
    )
    fig.colorbar(cf, ax=ax, label='LDOS (a.u.)')
    ax.set_xlabel(r'$\phi\,/\,\pi$', loc='right')
    ax.set_ylabel(r'$B$ (T)', loc='top', rotation=0, labelpad=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_title(
        rf'$\omega={target_energy:.3f}$ t, '
        rf'$\alpha={p.alpha:.1f}$, $\beta={p.beta:.1f}$ nm·meV, '
        rf'$\theta_z={p.theta_z/np.pi:.3f}\pi$',
        loc='right', fontsize=9
    )
    _style(ax)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, p: Params, phi, target_energy=0.0):
    """ldos_xy: shape (nx, ny)"""
    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = ldos_xy.max()
    cf = ax.imshow(
        ldos_xy.T / vmax, aspect='auto', origin='lower',
        extent=[-0.5, p.nx - 0.5, -0.5, p.ny - 0.5], #type: ignore
        cmap='magma', 
        interpolation='gaussian',
    )
    fig.colorbar(cf, ax=ax, label='LDOS (normalized to max value)')
    ax.set_xlabel('x-site', loc='right')
    ax.set_ylabel('y-site', loc='top', rotation=0, labelpad=12)
    ax.set_title(
        rf'$\phi={phi/np.pi:.2f}\pi$, $\omega={target_energy:.4f}$ t, '
        rf'$E_\parallel={p.E_par_meV:.3f}$ meV',
        loc='right', fontsize=9
    )
    _style(ax)
    ax.set_xticks(np.arange(1, p.nx , step=max(1, p.nx // 5)))
    ax.set_yticks(np.arange(1, p.ny + 1, step=max(1, p.ny // 5)))
    plt.tight_layout()
    return fig, ax


#%%  MAIN 
if __name__ == '__main__':

    # --- base parameters ---
    p = Params()
    p.alpha   = 14.3    # meV·nm
    p.beta    = 7.3     # meV·nm
    p.Delta   = 0.25    # meV
    p.mu_S    = 1.0     # meV
    p.mu_N    = 0.7    # meV
    p.g       = 1
    p.ny      = 100
    p.nx      = 5
    p.n_edge  = 5
    p.eta_rel = 0.01
    p.nphi    = 41
    p.nw      = 41
    p.w_range = 1.0
    p.E_z     = 0.0
 
    # set theta_z to the optimal angle (E_∥ ⊥ n_soc)
    p.theta_z = p.theta_z_optimal()

    p.summary()

    # --- Fig. 8 panels (a)-(d): E_∥ = 0, 0.23, 0.46, 0.69 meV ---
    E_per = [0.0, 0.23, 0.46, 0.69]   # meV

    for E_per_val in E_per:
        p.B = E_per_val / (_muB * p.g)   # convert meV → T
        print(f"\nPanel E_∥ = {p.E_par_meV:.3f} meV  (B = {p.B:.3f} T)")

        ldos = compute_ldos_E_phi(p, electron_only=False)
        fig, ax = plot_ldos_E_phi(ldos, p)
        print(f"  ✓ shape {ldos.shape},  max = {ldos.max():.3f}")

        # --- spatial LDOS at phi=pi, E=0 ---
        ldos_xy = compute_ldos_spatial(p, phi=np.pi, target_energy=0.0)
        plot_ldos_spatial(ldos_xy, p, phi=np.pi, target_energy=0.0)
        print(f"\n✓ Spatial LDOS shape: {ldos_xy.shape}")

        plt.show()
        print("\nDone.")
# %%
