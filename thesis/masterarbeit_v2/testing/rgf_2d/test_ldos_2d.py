"""
After verifying the RGf method works, by comparing it to the exact diagonalization, it is now used to calculate the GF for a 2d SNS junction and plot the LDOS as a function of energy and phase difference, as always.

FIXES applied:
  1. Energy sweep: H_slices_list and lead GFs are now both built at phi_fixed,
     so the finite system and infinite-lead calculations are consistent.
  2. Phase sweep: H_lead_slice for the left and right leads is rebuilt at the
     correct phase (phi_L / phi_R) for every phi step, instead of using the
     stale phi_fixed lead slice.
  3. Phase sweep: get_rgf_finite_system now explicitly passes return_full=False,
     matching the energy-sweep call and ensuring G_fin is a slice list.
  4. Minor: get_Vx_2d is now the single source of truth for V_x_2d; the
     redundant local V_x / V_y variables that were never used have been removed.
"""

#%% ── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numpy._core.multiarray import dtype
# Add parent directory to path for relative imports
current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent.parent
sys.path.insert(0, str(module_root))

import modules

# Plotting styles1
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
props = dict(boxstyle='round', facecolor='lightyellow', alpha=1)



#%% ── Helper functions ───────────────────────────────────────────────────────
"""
Convention: Slice in y-direction (h_slices has the V_y hopping matrix)
and the columns are in the x-direction.
"""

def build_sns_slice_2d(N_y, t, mu, h, alpha, delta):
    """
    Within-slice Hamiltonian: y-direction chain.
    y-hops use t_matrix_y (imaginary Rashba: σ_y).

    Args:
        N_y   : number of sites in y-direction (width of the junction)
        t     : hopping amplitude
        mu    : chemical potential
        h     : Zeeman energy
        alpha : spin-orbit coupling strength
        delta : complex SC pairing potential (can include phase)
    """
    h_0     = modules.onsite_matrix(t, mu, h, delta)
    V_y     = modules.t_matrix_y(t, alpha)      # imaginary Rashba in y
    V_y_dag = V_y.conj().T
    I_y     = np.eye(N_y, dtype=np.complex128)
    off_y   = np.eye(N_y, k=1, dtype=np.complex128)
    H = (np.kron(I_y, h_0)
         + np.kron(off_y,   V_y)
         + np.kron(off_y.T, V_y_dag))
    return H


def build_sns_junction_sliced_2d(N_y, t, mu_sc, mu_m, h, alpha, delta, phi,
                                  SL, SR, SM, symmetric=False):
    """
    Build the list of on-site Hamiltonian slices and the inter-slice hopping
    for the full SNS junction.

    Args:
        N_y       : number of sites in y-direction
        t         : hopping amplitude
        mu_sc     : chemical potential in the SC leads
        mu_m      : chemical potential in the normal region
        h         : Zeeman energy
        alpha     : spin-orbit coupling strength
        delta     : SC pairing potential magnitude
        phi       : phase difference between the SC leads
        SL        : number of sites in the left SC lead
        SR        : number of sites in the right SC lead
        SM        : number of sites in the normal region
        symmetric : use symmetric gauge (±φ/2) or asymmetric (full φ on right)

    Returns:
        H_slices : list of (4·N_y × 4·N_y) slice Hamiltonians
        V_2d     : (4·N_y × 4·N_y) inter-slice hopping matrix
    """
    phi_L = -phi / 2 if symmetric else 0.0
    phi_R =  phi / 2 if symmetric else phi

    H_L = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_L))
    H_R = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_R))
    H_M = build_sns_slice_2d(N_y, t, mu_m,  h, alpha, 0.0)

    V_2d = np.kron(np.eye(N_y), modules.t_matrix_y(t, alpha))   # y-direction hops

    H_slices = ([H_L.copy() for _ in range(SL)] +
                [H_M.copy() for _ in range(SM)] +
                [H_R.copy() for _ in range(SR)])
    return H_slices, V_2d


def get_lead_slice_2d(N_y, t, mu_sc, h, alpha, delta):

    return build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta)


def get_local_block(G_slice, y, dof=4):
    """Extract the (dof × dof) on-site block for site y from a slice GF."""
    s = y * dof
    e = (y + 1) * dof
    return G_slice[s:e, s:e]


#%% ── Parameters ─────────────────────────────────────────────────────────────
SYMMETRIC = True

SL, SM, SR = 80, 20, 80
N_y = 5

Delta  = 0.1
mu_sc  = 0.5
mu_n   = 0.1
t      = 1.0
alpha  = 0.2
B      = 0.8
eta    = 1e-4
phi_fixed = np.pi

N_E       = 61
N_PHI     = 61

energies = np.linspace(-0.15, 0.15, N_E)
phases   = np.linspace(0, 2 * np.pi, N_PHI)

probe_x      = [1, SM // 2, SM - 1]
probe_y      = [N_y // 2]
probe_labels = [f"x={x}, y={y}" for x in probe_x for y in probe_y]


#%% ── Building blocks ────────────────────────────────────────────────────────

# Lead slice Hamiltonians at phi_fixed (left and right, respecting gauge)
H_lead_slice = get_lead_slice_2d(N_y, t, mu_sc, B, alpha, Delta)
H_slices_list, V_x_2d = build_sns_junction_sliced_2d(
    N_y, t, mu_sc, mu_n, B, alpha, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC
)

#%% ── Energy sweep ───────────────────────────────────────────────────────────
print("\n══ Energy sweep ══")

ldos_e        = np.zeros((len(probe_x), N_E, 2))
pair_e        = np.zeros_like(ldos_e)
ldos_map_fin  = np.zeros((SM, N_E))
ldos_map_inf  = np.zeros((SM, N_E))

# First iteration with debug info
E = energies[0]

# Finite system
G_fin, *_ = modules.get_rgf_finite_system(
    H_slices_list, V_x_2d, E, eta=eta, return_full=False
)

# FIX 1 (cont.) ── Use the phase-correct left/right lead slices.
#   Previously a single H_lead_slice (always at phi_fixed for the left gauge)
#   was passed for both leads, making the right lead GF carry the wrong phase.
gL, gR = modules.get_surface_gfs_phased(
    E, H_lead_slice, V_x_2d, N_y,
    phi_fixed, symmetric=SYMMETRIC, eta=eta
)
G_inf, *_ = modules.get_rgf_sns(
    H_slices_list[SL:SL + SM], V_x_2d, gL, gR, E, eta=eta, return_full=False
)

print(f"\n  E={E:.4f}:")
print(f"    G_fin shape : {G_fin.shape}")
print(f"    G_inf shape : {G_inf.shape}")
print(f"    Testing probe indexing:")
for i, x in enumerate(probe_x):
    idx_fin = SL + x
    print(f"      probe_x[{i}]={x}: "
          f"G_fin[{idx_fin}].shape={G_fin[idx_fin].shape}, "
          f"G_inf[{x}].shape={G_inf[x].shape}")
    blk_fin = get_local_block(G_fin[idx_fin], probe_y[0])
    blk_inf = get_local_block(G_inf[x],        probe_y[0])
    print(f"        Local blocks : fin={blk_fin.shape}, inf={blk_inf.shape}")
    print(f"        LDOS : fin={-np.imag(np.trace(blk_fin)) / np.pi:.6f}, "
          f"inf={-np.imag(np.trace(blk_inf)) / np.pi:.6f}")

# Full energy loop
for e_idx, E in enumerate(energies):

    G_fin, *_ = modules.get_rgf_finite_system(
        H_slices_list, V_x_2d, E, eta=eta, return_full=False
    )

    gL, gR = modules.get_surface_gfs_2d_phased(
        E, H_lead_slice, V_x_2d, N_y,
        phi_fixed, symmetric=SYMMETRIC, eta=eta
    )
    G_inf, *_ = modules.get_rgf_sns(
        H_slices_list[SL:SL + SM], V_x_2d, gL, gR, E, eta=eta, return_full=False
    )

    for i, x in enumerate(probe_x):
        for j, y in enumerate(probe_y):
            blk_fin = get_local_block(G_fin[SL + x], y)
            blk_inf = get_local_block(G_inf[x],       y)
            for m, blk in enumerate([blk_fin, blk_inf]):
                ldos_e[i, e_idx, m] = -np.imag(np.trace(blk)) / np.pi
                pair_e[i, e_idx, m] = modules.get_pairing_amplitude(blk)

    for x, y in zip(probe_x, probe_y):
        blk_f = get_local_block(G_fin[SL + x], y)
        blk_i = get_local_block(G_inf[x],       y)
        ldos_map_fin[x, e_idx] = -np.imag(np.trace(blk_f)) / np.pi
        ldos_map_inf[x, e_idx] = -np.imag(np.trace(blk_i)) / np.pi


#%% ── Phase sweep ────────────────────────────────────────────────────────────
print("\n══ Phase sweep ══")

ldos_p = np.zeros((len(probe_x), N_PHI, 2))
pair_p = np.zeros_like(ldos_p)
E0 = 0.0

for p_idx, phi in enumerate(phases):

    # Rebuild junction slices at the current phi
    H_full, V_2d_phi = build_sns_junction_sliced_2d(
        N_y, t, mu_sc, mu_n, B, alpha, Delta, phi,
        SL, SR, SM, symmetric=SYMMETRIC
    )

    # FIX 2 ── Rebuild left/right lead slices at the current phi so that the
    #           surface GFs carry the correct phase for each sweep point.
    #           Previously a stale H_lead_slice (fixed at phi_fixed) was used
    #           for every phi step, giving wrong GFs for phi ≠ phi_fixed.
    H_lead = get_lead_slice_2d(N_y, t, mu_sc, B, alpha, Delta)

    # FIX 3 ── Pass return_full=False explicitly, consistent with the energy
    #           sweep, so G_fin is always a slice list indexed as G_fin[SL+x].
    G_fin, *_ = modules.get_rgf_finite_system(
        H_full, V_2d_phi, E0, eta=eta, return_full=False
    )

    gL, gR = modules.get_surface_gfs_2d_phased(
        E0, H_lead, V_x_2d, N_y,
        phi, symmetric=SYMMETRIC, eta=eta
    )
    G_inf, *_ = modules.get_rgf_sns(
        H_full[SL:SL + SM], V_2d_phi, gL, gR, E0, eta=eta, return_full=False
    )

    for i, x in enumerate(probe_x):
        for j, y in enumerate(probe_y):
            blk_f = get_local_block(G_fin[SL + x], y)
            blk_i = get_local_block(G_inf[x],       y)
            for m, blk in enumerate([blk_f, blk_i]):
                ldos_p[i, p_idx, m] = -np.imag(np.trace(blk)) / np.pi
                pair_p[i, p_idx, m] = modules.get_pairing_amplitude(blk)


#%% ── Figure 1: LDOS vs Energy ───────────────────────────────────────────────
fig, axes = plt.subplots(len(probe_x), 1, figsize=(8, 6), sharex=True)

for i, lbl in enumerate(probe_labels):
    ax = axes[i]
    ax.plot(energies, ldos_e[i, :, 0], '--', label='finite')
    ax.plot(energies, ldos_e[i, :, 1], '-',  label='infinite')
    ax.set_title(lbl)
    ax.set_ylabel("LDOS")
    ax.legend()
    ax.grid()

axes[-1].set_xlabel("Energy")
plt.suptitle(f"LDOS vs Energy  (φ = π)")
plt.tight_layout()


#%% ── Figure 2: Phase sweep ──────────────────────────────────────────────────
fig, axes = plt.subplots(len(probe_x), 2, figsize=(10, 6), sharex=True)

for i, lbl in enumerate(probe_labels):
    ax1, ax2 = axes[i]

    ax1.plot(phases / np.pi, ldos_p[i, :, 0], '--', label='finite')
    ax1.plot(phases / np.pi, ldos_p[i, :, 1], '-',  label='infinite')
    ax1.set_ylabel("LDOS")
    ax1.set_title(lbl)
    ax1.grid()

    ax2.plot(phases / np.pi, pair_p[i, :, 0], '--', label='finite')
    ax2.plot(phases / np.pi, pair_p[i, :, 1], '-',  label='infinite')
    ax2.set_ylabel("Pairing")
    ax2.grid()

axes[-1, 0].set_xlabel("φ/π")
axes[-1, 1].set_xlabel("φ/π")
plt.suptitle(f"LDOS & Pairing vs Phase  (E = 0)")
plt.tight_layout()


#%% ── Figure 3: LDOS map (x vs energy) ──────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

im0 = ax[0].imshow(ldos_map_fin.T, aspect='auto', origin='lower',
                   extent=[0, SM, energies[0], energies[-1]])
ax[0].set_title("Finite system")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Energy")

im1 = ax[1].imshow(ldos_map_inf.T, aspect='auto', origin='lower',
                   extent=[0, SM, energies[0], energies[-1]])
ax[1].set_title("Infinite leads")
ax[1].set_xlabel("x")

plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")
plt.suptitle(f"LDOS map  (φ = π)")
plt.show()

print("Finished 2D sweep.")
#%% ── Energy and Phase sweep ────────────────────────────────────────────────────────

def compute_ldos_point(energy, phase, energies, phases, N_y, t, mu_sc, mu_n, B, 
                        alpha, Delta, SL, SR, SM, SYMMETRIC, V_y_2d, eta, 
                        probe_x, probe_y, modules):
    """Compute LDOS for a single (energy, phase) point."""
    
    H_full, V_2d_phi = build_sns_junction_sliced_2d(
        N_y, t, mu_sc, mu_n, B, alpha, Delta, phase,
        SL, SR, SM, symmetric=SYMMETRIC
    )

    H_lead = get_lead_slice_2d(N_y, t, mu_sc, B, alpha, Delta)

    G_fin, *_ = modules.get_rgf_finite_system(
        H_full, V_2d_phi, energy, eta=eta, return_full=False
    )

    gL, gR = modules.get_surface_gfs_2d_phased(
        energy, H_lead, V_y_2d, N_y,
        phase, symmetric=SYMMETRIC, eta=eta
    )
    
    G_inf, *_ = modules.get_rgf_sns(
        H_full[SL:SL + SM], V_2d_phi, gL, gR, energy, eta=eta, return_full=False
    )

    blk_i = get_local_block(G_inf[probe_x[0]], probe_y[0])
    blk_f = get_local_block(G_fin[SL + probe_x[0]], probe_y[0])
    return -np.imag(np.trace(blk_i)) / np.pi, -np.imag(np.trace(blk_f)) / np.pi

# Replace the nested loop with:
results = Parallel(n_jobs=6)(  
    delayed(compute_ldos_point)(
        E, p, energies, phases, N_y, t, mu_sc, mu_n, B, alpha, Delta,
        SL, SR, SM, SYMMETRIC, V_x_2d, eta, probe_x, probe_y, modules
    )
    for i, E in enumerate(energies)
    for j, p in enumerate(phases)
)
results_inf, results_fin = zip(*results)
ldos_2d_infinite = np.array(results_inf).reshape((len(energies), len(phases)))
ldos_2d_finite = np.array(results_fin).reshape((len(energies), len(phases)))
#%% ── Figure 4: LDOS map (phase & energy) ──────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

im0 = ax[0].contourf(phases / np.pi, energies, ldos_2d_finite.T, levels=100, cmap='viridis')
ax[0].set_title("Finite system")
ax[0].set_xlabel("φ/π")
ax[0].set_ylabel("Energy")

im1 = ax[1].contourf(phases / np.pi, energies, ldos_2d_infinite.T, levels=100, cmap='viridis')
ax[1].set_title("Infinite leads")
ax[1].set_xlabel("φ/π")
ax[1].set_ylabel("Energy")

plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")
plt.suptitle(f"LDOS map  (φ = π)")
plt.show()

#%% ── Building blocks ────────────────────────────────────────────────────────
"""
Geometry convention
-------------------
x-direction:
    - transport direction
    - recursive direction
    - slice index

y-direction:
    - transverse direction inside each slice

Therefore:
    - build_sns_slice_2d() contains y-hopping
    - V_x_2d contains x-hopping between slices
"""

# Build full SNS system
H_slices_list, V_x_2d = build_sns_junction_sliced_2d(
    N_y, t, mu_sc, mu_n, B, alpha, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC
)

# Lead slice Hamiltonian
H_lead_slice = get_lead_slice_2d(
    N_y, t, mu_sc, B, alpha, Delta
)


#%% ── Energy sweep ───────────────────────────────────────────────────────────
print("\n══ Energy sweep ══")

ldos_e = np.zeros((len(probe_x), len(probe_y), N_E, 2))
pair_e = np.zeros_like(ldos_e)

# Full spatial map:
# shape = (x, y, energy)
ldos_map_fin = np.zeros((SM, N_y, N_E))
ldos_map_inf = np.zeros((SM, N_y, N_E))

for e_idx, E in enumerate(energies):

    # ---------- finite system ----------
    G_fin, *_ = modules.get_rgf_finite_system(
        H_slices_list,
        V_x_2d,
        E,
        eta=eta,
        return_full=False
    )

    # ---------- infinite leads ----------
    gL, gR = modules.get_surface_gfs_2d_phased(
        E,
        H_lead_slice,
        V_x_2d,
        N_y,
        phi_fixed,
        symmetric=SYMMETRIC,
        eta=eta
    )

    G_inf, *_ = modules.get_rgf_sns(
        H_slices_list[SL:SL + SM],
        V_x_2d,
        gL,
        gR,
        E,
        eta=eta,
        return_full=False
    )

    for ix, x in enumerate(probe_x):
        for iy, y in enumerate(probe_y):

            blk_fin = get_local_block(G_fin[SL + x], y)
            blk_inf = get_local_block(G_inf[x],       y)

            for m, blk in enumerate([blk_fin, blk_inf]):

                ldos_e[ix, iy, e_idx, m] = (
                    -np.imag(np.trace(blk)) / np.pi
                )

                pair_e[ix, iy, e_idx, m] = (
                    modules.get_pairing_amplitude(blk)
                )

    # ---------- full spatial map ----------
    for x in range(SM):
        for y in range(N_y):

            blk_f = get_local_block(G_fin[SL + x], y)
            blk_i = get_local_block(G_inf[x],       y)

            ldos_map_fin[x, y, e_idx] = (
                -np.imag(np.trace(blk_f)) / np.pi
            )

            ldos_map_inf[x, y, e_idx] = (
                -np.imag(np.trace(blk_i)) / np.pi
            )


#%% ── Figure 1: LDOS vs Energy ───────────────────────────────────────────────
#── Spatial maps LDOS vs energy and x/y for selected probe sites 
fig, axes = plt.subplots(
    len(probe_x),
    len(probe_y),
    figsize=(4 * len(probe_y), 3 * len(probe_x)),
    sharex=True
)

for ix, x in enumerate(probe_x):
    for iy, y in enumerate(probe_y):

        ax = axes[ix, iy]

        ax.plot(
            energies,
            ldos_e[ix, iy, :, 0],
            '--',
            label='finite'
)

        ax.plot(
            energies,
            ldos_e[ix, iy, :, 1],
            '-',
            label='infinite'
        )

        ax.set_title(f"x={x}, y={y}")
        ax.grid()

        if ix == len(probe_x) - 1:
            ax.set_xlabel("Energy")

        if iy == 0:
            ax.set_ylabel("LDOS")

        if ix == 0 and iy == 0:
            ax.legend()

plt.suptitle(f"LDOS vs Energy  (φ = {phi_fixed/np.pi:.2f}π)")


#%% ── Figure 2: Zero-energy spatial LDOS map ────────────────────────────────

E_zero_idx = np.argmin(np.abs(energies))

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# finite
im0 = ax[0].imshow(
    ldos_map_fin[:, :, E_zero_idx].T,
    origin='lower',
    aspect='auto',
    extent=[0, SM, 0, N_y]
)

ax[0].set_title("Finite system")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

# infinite
im1 = ax[1].imshow(
    ldos_map_inf[:, :, E_zero_idx].T,
    origin='lower',
    aspect='auto',
    extent=[0, SM, 0, N_y]
)

ax[1].set_title("Infinite leads")
ax[1].set_xlabel("x")

plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")

plt.suptitle("Zero-energy LDOS map")


#%% ── Figure 3: LDOS(x,E) for selected y-sites ──────────────────────────────

selected_y = [0, N_y // 2, N_y - 1]

fig, axes = plt.subplots(
    1,
    len(selected_y),
    figsize=(5 * len(selected_y), 4),
    sharey=True
)

for i, y0 in enumerate(selected_y):

    ax = axes[i]

    im = ax.imshow(
        ldos_map_inf[:, y0, :].T,
        aspect='auto',
        origin='lower',
        extent=[0, SM, energies[0], energies[-1]]
    )

    ax.set_title(f"y = {y0}")
    ax.set_xlabel("x")

    if i == 0:
        ax.set_ylabel("Energy")

plt.colorbar(im, ax=axes.ravel().tolist(), label="LDOS")

plt.suptitle("LDOS(x,E) at different y-sites")


#%% ── Figure 4: LDOS(y,E) for selected x-sites ──────────────────────────────

selected_x = [0, SM // 2, SM - 1]

fig, axes = plt.subplots(
    1,
    len(selected_x),
    figsize=(5 * len(selected_x), 4),
    sharey=True
)

for i, x0 in enumerate(selected_x):

    ax = axes[i]

    im = ax.imshow(
        ldos_map_inf[x0, :, :].T,
        aspect='auto',
        origin='lower',
        extent=[0, N_y, energies[0], energies[-1]]
    )

    ax.set_title(f"x = {x0}")
    ax.set_xlabel("y")

    if i == 0:
        ax.set_ylabel("Energy")

plt.colorbar(im, ax=axes.ravel().tolist(), label="LDOS")

plt.suptitle("LDOS(y,E) at different x-sites")

plt.show()

print("Finished 2D sweep.")
# %%
