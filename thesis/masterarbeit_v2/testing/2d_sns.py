"""
RGF-based LDOS calculation for a 2D SNS junction — optimized.

Key optimizations vs original:
  - MKL/OpenBLAS thread pinning replaces joblib parallelism
  - Surface GFs computed once per energy (not once per phi)
  - Phase vectorization: single GL/GR sweep through normal region,
    phase applied only at the combination step across all phi simultaneously
  - Static Hamiltonians (H_M, V_x_2d, H_lead_slice) built once at startup
  - Finite and infinite paths separated into independent functions

Geometry convention
-------------------
x-direction : transport / recursive direction (slice index)
y-direction : transverse direction inside each slice

  within a slice  →  t_matrix_y  (σ_x, k_y Rashba)
  between slices  →  t_matrix_x  (σ_y, k_x Rashba)
"""
#%% ── Imports & thread pinning ───────────────────────────────────────────────
import os
import multiprocessing

slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
os.environ['MKL_NUM_THREADS']     = slurm_cpus
os.environ['OMP_NUM_THREADS']     = slurm_cpus
os.environ['OPENBLAS_NUM_THREADS'] = slurm_cpus

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent
sys.path.insert(0, str(module_root))
import modules

plt.rcParams['font.size']        = 14
plt.rcParams['axes.titlesize']   = 18
plt.rcParams['axes.labelsize']   = 16
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 10
plt.rcParams['figure.titlesize'] = 20

#%% ── Parameters ─────────────────────────────────────────────────────────────
SYMMETRIC = False

SL, SM, SR = 100, 5, 100
N_y = 20

Delta  = 0.1
mu_sc  = .7
mu_n   = 0.1
t      = 1.0
alpha  = 0.6
B      = 1.2
eta    = 1e-3

E0        = 0.0
phi_fixed = np.pi

N_E   = 61
N_PHI = 61

range_e  = -1.05 * Delta
energies = np.linspace(range_e, -range_e, N_E)
phases   = np.linspace(0, 2 * np.pi, N_PHI)

probe_x = [SM // 2, SM - 1]
probe_y = [N_y // 2, N_y - 1]

#%% ── Electron/hole index masks for phase vectorization ──────────────────────
# Each site has dof=4 components: (c↑, c↓, c↓†, -c↑†)
# Electron indices: components 0,1 (mod 4);  hole indices: 2,3 (mod 4)
_dof     = 4 * N_y
_elec_ix = np.array([i for i in range(_dof) if i % 4 < 2])
_hole_ix = np.array([i for i in range(_dof) if i % 4 >= 2])


#%% ── Phase vectorization helper ─────────────────────────────────────────────
def _apply_phase_batch(g_unphased, phi_array):
    """
    Apply Nambu gauge rotation U(φ) g U†(φ) for every φ in phi_array.

    U(φ) = diag(e^{iφ/2} on electron, e^{-iφ/2} on hole).
    U g U† multiplies the (elec, hole) block by e^{iφ} and (hole, elec) by e^{-iφ}.
    The (elec,elec) and (hole,hole) blocks are invariant.

    Args:
        g_unphased : (dof, dof) unphased surface GF
        phi_array  : (nphi,) array of phase values

    Returns:
        (nphi, dof, dof) array of phased GFs
    """
    nphi   = len(phi_array)
    G_out  = np.empty((nphi, _dof, _dof), dtype=np.complex128)
    G_out[:] = g_unphased  # broadcast copy

    ep = np.exp( 1j * phi_array)[:, None, None]
    em = ep.conj()

    G_out[:, _elec_ix[:, None], _hole_ix] *= ep
    G_out[:, _hole_ix[:, None], _elec_ix] *= em
    return G_out


#%% ── Static system (built once) ─────────────────────────────────────────────
print("Pre-computing static Hamiltonians...")

H_lead_slice = modules.get_lead_slice_2d(N_y, t, mu_sc, B, alpha, Delta)
H_normal_slices, V_x_2d = modules.build_sns_normal_only_2d(N_y, t, mu_n, B, alpha, SM)
Vd = V_x_2d.conj().T
I  = np.eye(_dof, dtype=np.complex128)

print(f"  Normal region: {SM} slices of ({_dof}×{_dof})")
print(f"  Lead slice:    ({_dof}×{_dof})")


#%% ── Core RGF: infinite leads, vectorized over phi ─────────────────────────
def _rgf_inf_phi_batch(gL_unphased, gR_unphased, z, phi_array):
    """
    RGF through normal region for all phases simultaneously.

    Exploits gauge invariance of H_M (delta=0): the GL/GR sweeps are
    phase-independent and run once; the phase enters only at the
    boundary self-energies in the combination step.

    Args:
        gL_unphased : (dof, dof) unphased left surface GF
        gR_unphased : (dof, dof) unphased right surface GF
        z           : complex energy (w + i*eta)
        phi_array   : (nphi,) phases

    Returns:
        ldos : (nphi, SM) LDOS summed over y and dof per x-slice
    """
    N    = SM
    nphi = len(phi_array)
    zI   = z * I

    # 1. RIGHT sweep — unphased seed, runs once
    GR = np.empty((N, _dof, _dof), dtype=np.complex128)
    GR[-1] = la.inv(zI - H_normal_slices[-1] - V_x_2d @ gR_unphased @ Vd)
    for i in range(N - 2, -1, -1):
        GR[i] = la.inv(zI - H_normal_slices[i] - V_x_2d @ GR[i + 1] @ Vd)

    # 2. LEFT sweep — unphased seed (asymmetric gauge: phi_L = 0 → g_L unchanged)
    GL = np.empty((N, _dof, _dof), dtype=np.complex128)
    GL[0] = la.inv(zI - H_normal_slices[0] - Vd @ gL_unphased @ V_x_2d)
    for i in range(1, N):
        GL[i] = la.inv(zI - H_normal_slices[i] - Vd @ GL[i - 1] @ V_x_2d)

    # 3. Phased right boundary self-energy: Σ_R^boundary = V gR(φ) V†
    #    Only the rightmost slice uses gR directly; all others use GR[i+1].
    #    Apply phase only to the gR_unphased boundary term.
    gR_phased = _apply_phase_batch(gR_unphased, phi_array)  # (nphi, dof, dof)
    # Precompute the phased boundary SE for the last slice: (nphi, dof, dof)
    SR_boundary = np.einsum('ij,pjk,kl->pil', V_x_2d, gR_phased, Vd)

    # 4. Combination step — all phi simultaneously
    ldos = np.zeros((nphi, N))
    for i in range(N):
        SL_i = Vd @ (GL[i - 1] if i > 0 else gL_unphased) @ V_x_2d  # (dof, dof), phase-free

        if i < N - 1:
            SR_i = V_x_2d @ GR[i + 1] @ Vd                            # (dof, dof), phase-free
            M    = zI - H_normal_slices[i] - SL_i - SR_i               # (dof, dof)
            G_i  = la.inv(M)                                            # (dof, dof)
            # Same G for all phi since no phase here
            ldos[:, i] = -np.imag(np.trace(G_i)) / np.pi
        else:
            # Last slice: SR depends on phi
            M_base = zI - H_normal_slices[i] - SL_i                    # (dof, dof)
            # Broadcast: M_base - SR_boundary[p] for each p
            M_phi  = M_base[None] - SR_boundary                         # (nphi, dof, dof)
            G_phi  = np.linalg.inv(M_phi)                              # (nphi, dof, dof)
            ldos[:, i] = -np.imag(
                np.einsum('pii->p', G_phi)
            ) / np.pi

    return ldos


def _rgf_inf_phi_batch_spatial(gL_unphased, gR_unphased, z, phi_array):
    """
    Same as _rgf_inf_phi_batch but returns full (nphi, SM, N_y) spatial LDOS.
    """
    N    = SM
    nphi = len(phi_array)
    zI   = z * I

    GR = np.empty((N, _dof, _dof), dtype=np.complex128)
    GR[-1] = la.inv(zI - H_normal_slices[-1] - V_x_2d @ gR_unphased @ Vd)
    for i in range(N - 2, -1, -1):
        GR[i] = la.inv(zI - H_normal_slices[i] - V_x_2d @ GR[i + 1] @ Vd)

    GL = np.empty((N, _dof, _dof), dtype=np.complex128)
    GL[0] = la.inv(zI - H_normal_slices[0] - Vd @ gL_unphased @ V_x_2d)
    for i in range(1, N):
        GL[i] = la.inv(zI - H_normal_slices[i] - Vd @ GL[i - 1] @ V_x_2d)

    gR_phased    = _apply_phase_batch(gR_unphased, phi_array)
    SR_boundary  = np.einsum('ij,pjk,kl->pil', V_x_2d, gR_phased, Vd)

    ldos = np.zeros((nphi, N, N_y))
    for i in range(N):
        SL_i = Vd @ (GL[i - 1] if i > 0 else gL_unphased) @ V_x_2d

        if i < N - 1:
            G_i = la.inv(zI - H_normal_slices[i] - SL_i - V_x_2d @ GR[i + 1] @ Vd)
            # Same for all phi — tile across phi axis
            diag_i = np.diagonal(G_i).reshape(N_y, 4).sum(axis=1)
            ldos[:, i, :] = (-np.imag(diag_i) / np.pi)[None, :]
        else:
            M_base = zI - H_normal_slices[i] - SL_i
            M_phi  = M_base[None] - SR_boundary
            G_phi  = np.linalg.inv(M_phi)
            diag_phi = np.diagonal(G_phi, axis1=1, axis2=2)            # (nphi, dof)
            ldos[:, i, :] = -np.imag(
                diag_phi.reshape(nphi, N_y, 4).sum(axis=2)
            ) / np.pi

    return ldos


#%% ── Core RGF: finite system (explicit SL+SM+SR slices) ────────────────────
def _rgf_finite_one_phi(H_slices_full, z):
    """
    Single-phi finite-system RGF. Returns (SM, N_y) spatial LDOS.
    Phase is already baked into H_slices_full.
    """
    N   = len(H_slices_full)
    zI  = z * I

    GL = np.empty((N, _dof, _dof), dtype=np.complex128)
    GL[0] = la.inv(zI - H_slices_full[0])
    for i in range(1, N):
        GL[i] = la.inv(zI - H_slices_full[i] - Vd @ GL[i - 1] @ V_x_2d)

    GR = np.empty((N, _dof, _dof), dtype=np.complex128)
    GR[-1] = la.inv(zI - H_slices_full[-1])
    for i in range(N - 2, -1, -1):
        GR[i] = la.inv(zI - H_slices_full[i] - V_x_2d @ GR[i + 1] @ Vd)

    ldos = np.zeros((SM, N_y))
    for ix in range(SM):
        i    = SL + ix
        SL_i = Vd @ GL[i - 1] @ V_x_2d if i > 0 else 0
        SR_i = V_x_2d @ GR[i + 1] @ Vd if i < N - 1 else 0
        G_i  = la.inv(zI - H_slices_full[i] - SL_i - SR_i)
        diag = np.diagonal(G_i).reshape(N_y, 4).sum(axis=1)
        ldos[ix, :] = -np.imag(diag) / np.pi

    return ldos


def _build_full_slices(phi):
    """Build SL+SM+SR slice list for a given phi."""
    phi_L = -phi / 2 if SYMMETRIC else 0.0
    phi_R =  phi / 2 if SYMMETRIC else phi
    H_L   = modules.build_sns_slice_2d(N_y, t, mu_sc, B, alpha, Delta * np.exp(1j * phi_L))
    H_R   = modules.build_sns_slice_2d(N_y, t, mu_sc, B, alpha, Delta * np.exp(1j * phi_R))
    return (
        [H_L] * SL +
        [s.copy() for s in H_normal_slices] +
        [H_R] * SR
    )


#%% ── High-level drivers ─────────────────────────────────────────────────────

# ── INFINITE LEADS ────────────────────────────────────────────────────────────

def run_inf_energy_sweep():
    """
    LDOS(E, phi) — infinite leads.
    Returns (N_E, N_PHI, SM) array.
    """
    out = np.zeros((N_E, N_PHI, SM))
    for ie, E in enumerate(tqdm(energies, desc='inf E-sweep')):
        z = E + 1j * eta
        gL, _ = modules.get_surface_gf(E, H_lead_slice, Vd, eta=eta)
        gR, _ = modules.get_surface_gf(E, H_lead_slice, V_x_2d, eta=eta)
        out[ie] = _rgf_inf_phi_batch(gL, gR, z, phases)
    return out


def run_inf_energy_sweep_spatial():
    """
    Spatial LDOS(E, phi, x, y) — infinite leads.
    Returns (N_E, N_PHI, SM, N_y) array.
    """
    out = np.zeros((N_E, N_PHI, SM, N_y))
    for ie, E in enumerate(tqdm(energies, desc='inf E-sweep spatial')):
        z = E + 1j * eta
        gL, _ = modules.get_surface_gf(E, H_lead_slice, Vd, eta=eta)
        gR, _ = modules.get_surface_gf(E, H_lead_slice, V_x_2d, eta=eta)
        out[ie] = _rgf_inf_phi_batch_spatial(gL, gR, z, phases)
    return out


def run_inf_phi_sweep(target_energy=E0):
    """
    LDOS(phi, x, y) at fixed energy — infinite leads.
    Returns (N_PHI, SM, N_y) array.
    Computes surface GFs once then batches all phi.
    """
    z = target_energy + 1j * eta
    gL, _ = modules.get_surface_gf(target_energy, H_lead_slice, Vd, eta=eta)
    gR, _ = modules.get_surface_gf(target_energy, H_lead_slice, V_x_2d, eta=eta)
    return _rgf_inf_phi_batch_spatial(gL, gR, z, phases)


# ── FINITE SYSTEM ─────────────────────────────────────────────────────────────

def run_finite_energy_sweep():
    """
    Spatial LDOS(E, phi, x, y) — finite system (explicit SL+SM+SR slices).
    Returns (N_E, N_PHI, SM, N_y).
    Phase must be baked into slices so no vectorization across phi here.
    """
    out = np.zeros((N_E, N_PHI, SM, N_y))
    for ip, phi in enumerate(tqdm(phases, desc='finite phi')):
        H_full = _build_full_slices(phi)
        for ie, E in enumerate(energies):
            z = E + 1j * eta
            out[ie, ip] = _rgf_finite_one_phi(H_full, z)
    return out


def run_finite_phi_sweep(target_energy=E0):
    """
    Spatial LDOS(phi, x, y) at fixed energy — finite system.
    Returns (N_PHI, SM, N_y).
    """
    out = np.zeros((N_PHI, SM, N_y))
    for ip, phi in enumerate(tqdm(phases, desc='finite phi-sweep')):
        H_full = _build_full_slices(phi)
        z      = target_energy + 1j * eta
        out[ip] = _rgf_finite_one_phi(H_full, z)
    return out


#%% ── Plotting helpers ───────────────────────────────────────────────────────
def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')


def plot_ldos_E_phi(ldos_E_phi_x, title=''):
    """ldos_E_phi_x: (N_E, N_PHI) — already summed/selected over x."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.pcolormesh(energies / Delta, phases / np.pi, ldos_E_phi_x.T,
                       shading='auto', cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel(r'$\omega\,/\,\Delta$', loc='right')
    ax.set_ylabel(r'$\phi\,/\,\pi$',      loc='top', rotation=0, labelpad=12)
    ax.set_title(title, loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


def plot_ldos_spatial(ldos_xy, phi_val=None, E_val=None):
    """ldos_xy: (SM, N_y)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cf = ax.contourf(np.arange(SM), np.arange(N_y), ldos_xy.T,
                     levels=100, cmap='inferno')
    fig.colorbar(cf, ax=ax, label='LDOS')
    ax.set_xlabel('x (slice)', loc='right')
    ax.set_ylabel('y (site)', loc='top', rotation=0, labelpad=12)
    title = ''
    if phi_val is not None:
        title += rf'$\phi={phi_val/np.pi:.2f}\pi$'
    if E_val is not None:
        title += rf'  $\omega={E_val:.3f}$'
    ax.set_title(title, loc='right', fontsize=9)
    _style(ax)
    plt.tight_layout()
    return fig, ax


#%% ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("RGF 2D SNS — OPTIMIZED")
    print(f"Threads: {slurm_cpus}  |  N_y={N_y}  SM={SM}  SL={SL}  SR={SR}")
    print(f"N_E={N_E}  N_PHI={N_PHI}  eta={eta}")
    print("=" * 70)

    # ── Infinite-lead sweeps ──────────────────────────────────────────────────
    print("\nRunning infinite-lead energy sweep (all phi batched per energy)...")
    ldos_inf_Ephi_xy = run_inf_energy_sweep_spatial()     # (N_E, N_PHI, SM, N_y)
    ldos_inf_Ephi    = ldos_inf_Ephi_xy.sum(axis=(2, 3))  # (N_E, N_PHI) total LDOS
    print(f"  Done. Shape: {ldos_inf_Ephi_xy.shape}")

    # Pick probe site from the full array
    px, py = probe_x[0], probe_y[1]
    plot_ldos_E_phi(ldos_inf_Ephi, title=f'Inf leads — total LDOS')
    plot_ldos_E_phi(ldos_inf_Ephi_xy[:, :, px, py],
                    title=f'Inf leads — x={px}, y={py}')

    # Zero-energy spatial map (phi=pi)
    phi_pi_idx = np.argmin(np.abs(phases - np.pi))
    E0_idx     = np.argmin(np.abs(energies))
    plot_ldos_spatial(ldos_inf_Ephi_xy[E0_idx, phi_pi_idx],
                      phi_val=np.pi, E_val=energies[E0_idx])

    # ── Finite-system phi sweep at E=0 (example) ─────────────────────────────
    # print("\nRunning finite-system phi sweep at E=0...")
    # ldos_fin_phi_xy = run_finite_phi_sweep(target_energy=E0)   # (N_PHI, SM, N_y)
    # print(f"  Done. Shape: {ldos_fin_phi_xy.shape}")

    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax.plot(phases / np.pi, ldos_fin_phi_xy[:, px, py], label=f'finite x={px},y={py}')
    # ax.plot(phases / np.pi,
    #         run_inf_phi_sweep(target_energy=E0)[:, px, py],
    #         '--', label=f'inf x={px},y={py}')
    # ax.set_xlabel(r'$\phi\,/\,\pi$')
    # ax.set_ylabel('LDOS')
    # ax.legend()
    # ax.set_title(f'LDOS vs phase at E=0, x={px}, y={py}', fontsize=9)
    # _style(ax)
    plt.tight_layout()

    plt.show()
    print("=" * 70)
    print("Finished.")
    print("=" * 70)
# %%
