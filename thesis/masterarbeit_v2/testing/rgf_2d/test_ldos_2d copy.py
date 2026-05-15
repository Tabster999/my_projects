"""
RGF-based LDOS calculation for a 2D SNS junction.
Plots LDOS as a function of energy, phase difference, and spatial position.

Geometry convention
-------------------
x-direction : transport / recursive direction (slice index)
y-direction : transverse direction inside each slice

Therefore:
  - build_sns_slice_2d()          uses t_matrix_y  (σ_y Rashba) for y-hops WITHIN a slice
  - build_sns_junction_sliced_2d  uses t_matrix_x    (σ_x Rashba) for x-hops BETWEEN slices → V_x_2d
"""

#%% ── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent.parent
sys.path.insert(0, str(module_root))
import modules

plt.rcParams['font.size']        = 14
plt.rcParams['axes.titlesize']   = 18
plt.rcParams['axes.labelsize']   = 16
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 10
plt.rcParams['figure.titlesize'] = 20

#%% ── Helper functions ───────────────────────────────────────────────────────

def build_sns_slice_2d(N_y, t, mu, h, alpha, delta):
    """
    Within-slice Hamiltonian: y-direction chain.
    y-hops use t_matrix_y (σ_y Rashba).

    Args:
        N_y   : number of sites in y-direction
        t     : hopping amplitude
        mu    : chemical potential
        h     : Zeeman energy
        alpha : spin-orbit coupling strength
        delta : complex SC pairing potential (can include phase)
    """
    h_0     = modules.onsite_matrix(t, mu, h, delta)
    V_y     = modules.t_matrix_y(t, alpha)
    V_y_dag = V_y.conj().T
    I_y     = np.eye(N_y, dtype=np.complex128)
    off_y   = np.eye(N_y, k=1, dtype=np.complex128)
    H = (np.kron(I_y,     h_0)
       + np.kron(off_y,   V_y)
       + np.kron(off_y.T, V_y_dag))
    return H


def build_sns_junction_sliced_2d(N_y, t, mu_sc, mu_m, h, alpha, delta, phi,
                                  SL, SR, SM, symmetric=False):
    """
    Build slice Hamiltonians and inter-slice hopping for the full SNS junction.
    Args:
        N_y       : number of sites in y-direction
        t         : hopping amplitude
        mu_sc     : chemical potential in SC leads
        mu_m      : chemical potential in normal region
        h         : Zeeman energy
        alpha     : spin-orbit coupling strength
        delta     : SC pairing potential magnitude
        phi       : phase difference between SC leads
        SL        : number of slices in left SC lead
        SR        : number of slices in right SC lead
        SM        : number of slices in normal region
        symmetric : symmetric gauge (±φ/2) if True, asymmetric (0/φ) if False

    Returns:
        H_slices : list of (4·N_y × 4·N_y) slice Hamiltonians
        V_x_2d   : (4·N_y × 4·N_y) inter-slice x-hopping matrix (σ_x Rashba)
    """
    phi_L = -phi / 2 if symmetric else 0.0
    phi_R = phi / 2 if symmetric else phi

    H_L = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_L))
    H_R = build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta * np.exp(1j * phi_R))
    H_M = build_sns_slice_2d(N_y, t, mu_m,  h, alpha, 0.0)

    V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), modules.t_matrix_x(t, alpha))

    H_slices = ([H_L.copy() for _ in range(SL)] +
                [H_M.copy() for _ in range(SM)] +
                [H_R.copy() for _ in range(SR)])
    return H_slices, V_x_2d


def get_lead_slice_2d(N_y, t, mu_sc, h, alpha, delta):
    """Unphased lead slice Hamiltonian (real delta)."""
    return build_sns_slice_2d(N_y, t, mu_sc, h, alpha, delta)


def get_local_block(G_slice, y, dof=4):
    """Extract the (dof × dof) on-site block for site y from a slice GF."""
    s = y * dof
    e = (y + 1) * dof
    return G_slice[s:e, s:e]


def build_sns_normal_only(N_y, t, mu_m, h, alpha, SM):
    """Just the normal region slices — for use in 2D sweep where leads are surface GFs."""
    H_M    = build_sns_slice_2d(N_y, t, mu_m, h, alpha, 0.0)
    V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), modules.t_matrix_x(t, alpha))
    return [H_M.copy() for _ in range(SM)], V_x_2d


#%% ── Parameters ─────────────────────────────────────────────────────────────
SYMMETRIC = False

SL, SM, SR = 200, 100, 200
N_y = 10

Delta  = 250e-6
mu_sc  = 0.1e-3
mu_n   = 0.7e-3
t      = 1.0
alpha  = 16e-3
B      = 0.5e-3
eta    = 1e-5

E0 = 0.0
phi_fixed = np.pi

N_E   = 151
N_PHI = 151

energies = np.linspace(-0.2, 0.2, N_E)
phases   = np.linspace(-1e-2, 2 * np.pi + 1e-2, N_PHI)

probe_x = [0, SM // 2, SM - 1]
probe_y = [0, N_y // 2, N_y - 1]

#%% ── Building blocks ────────────────────────────────────────────────────────

# Full SNS junction at phi_fixed    ;   len(H_slice_list) = SL + SM + SR
H_slices_list, V_x_2d = build_sns_junction_sliced_2d(
    N_y, t, mu_sc, mu_n, B, alpha, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC
)

# Unphased lead slice (phase applied inside get_surface_gfs_phased via U·g·U†)
H_lead_slice = get_lead_slice_2d(N_y, t, mu_sc, B, alpha, Delta)

H_normal_slices, _ = build_sns_normal_only(N_y, t, mu_n, B, alpha, SM)


#%% ── Energy sweep ───────────────────────────────────────────────────────────
print("\n══ Energy sweep ══")

def compute_energy_slice(E, H_slices_list, V_x_2d, H_lead_slice, probe_x_set, probe_x_idx, probe_y_set, probe_y_idx):
    G_fin, *_ = modules.get_rgf_finite_system(
        H_slices_list, V_x_2d, E, eta=eta, return_full=False
    )
    gL, gR = modules.get_surface_gfs_phased(
        E, H_lead_slice, V_x_2d, N_y,
        phi_fixed, symmetric=SYMMETRIC, eta=eta
    )
    G_inf, *_ = modules.get_rgf_sns(
        H_normal_slices, V_x_2d, gL, gR, E, eta=eta, return_full=False
    )

    map_fin = np.zeros((SM, N_y))
    map_inf = np.zeros((SM, N_y))
    pair_fin = np.zeros((len(probe_x_idx), len(probe_y_idx)))
    pair_inf = np.zeros_like(pair_fin)

    for x in range(SM):
        for y in range(N_y):
            blk_f = get_local_block(G_fin[SL + x], y)
            blk_i = get_local_block(G_inf[x],       y)
            map_fin[x, y] = -np.imag(np.trace(blk_f)) / np.pi
            map_inf[x, y] = -np.imag(np.trace(blk_i)) / np.pi
            if x in probe_x_set and y in probe_y_set:
                ix, iy = probe_x_idx[x], probe_y_idx[y]
                pair_fin[ix, iy] = modules.get_pairing_amplitude(blk_f)
                pair_inf[ix, iy] = modules.get_pairing_amplitude(blk_i)

    return map_fin, map_inf, pair_fin, pair_inf


# Precompute lookup dicts
probe_x_set = set(probe_x)
probe_y_set = set(probe_y)
probe_x_idx = {x: i for i, x in enumerate(probe_x)}
probe_y_idx = {y: i for i, y in enumerate(probe_y)}

results = Parallel(n_jobs=-1)(
    delayed(compute_energy_slice)(
        E, H_slices_list, V_x_2d, H_lead_slice,
        probe_x_set, probe_x_idx, probe_y_set, probe_y_idx
    )
    for E in energies
)

# Unpack
ldos_map_fin = np.stack([r[0] for r in results], axis=2)   # (SM, N_y, N_E)
ldos_map_inf = np.stack([r[1] for r in results], axis=2)
pair_fin_all = np.stack([r[2] for r in results], axis=2)   # (n_probe_x, n_probe_y, N_E)
pair_inf_all = np.stack([r[3] for r in results], axis=2)

# Slice ldos_e from maps
ldos_e = np.zeros((len(probe_x), len(probe_y), N_E, 2))
for ix, x in enumerate(probe_x):
    for iy, y in enumerate(probe_y):
        ldos_e[ix, iy, :, 0] = ldos_map_fin[x, y, :]
        ldos_e[ix, iy, :, 1] = ldos_map_inf[x, y, :]

pair_e = np.stack([pair_fin_all, pair_inf_all], axis=3)    # (n_probe_x, n_probe_y, N_E, 2)

#%% ── Phase sweep ────────────────────────────────────────────────────────────
print("\n══ Phase sweep ══")

def compute_phi(phi, E0, H_lead_slice, V_x_2d, N_y, SL, SM, SR, eta, SYMMETRIC, probe_x, probe_y):
    H_full, _ = build_sns_junction_sliced_2d(
        N_y, t, mu_sc, mu_n, B, alpha, Delta, phi,
        SL, SR, SM, symmetric=SYMMETRIC
    )
    G_fin, *_ = modules.get_rgf_finite_system(
        H_full, V_x_2d, E0, eta=eta, return_full=False
    )
    gL, gR = modules.get_surface_gfs_phased(
        E0, H_lead_slice, V_x_2d, N_y,
        phi, symmetric=SYMMETRIC, eta=eta
    )
    G_inf, *_ = modules.get_rgf_sns(
        H_normal_slices, V_x_2d, gL, gR, E0, eta=eta, return_full=False
    )

    ldos_out = np.zeros((len(probe_x), len(probe_y), 2))
    pair_out = np.zeros_like(ldos_out)

    for ix, x in enumerate(probe_x):
        for iy, y in enumerate(probe_y):
            blk_f = get_local_block(G_fin[SL + x], y)
            blk_i = get_local_block(G_inf[x],       y)
            for m, blk in enumerate([blk_f, blk_i]):
                ldos_out[ix, iy, m] = -np.imag(np.trace(blk)) / np.pi
                pair_out[ix, iy, m] =  modules.get_pairing_amplitude(blk)

    return ldos_out, pair_out


results = Parallel(n_jobs=-1)(
    delayed(compute_phi)(phi, E0, H_lead_slice, V_x_2d, N_y, SL, SM, SR, eta, SYMMETRIC, probe_x, probe_y)
    for phi in phases
)

# Unpack results
ldos_p = np.stack([r[0] for r in results], axis=2)  # (n_probe_x, n_probe_y, N_PHI, 2)
pair_p = np.stack([r[1] for r in results], axis=2)
#%% ── 2D (energy × phase) sweep ─────────────────────────────────────────────
print("\n══ 2D sweep (energy × phase) ══")

def compute_ldos_point(energy, phase, H_normal_slices, H_lead_slice, V_x_2d):
    gL, gR = modules.get_surface_gfs_phased(
        energy, H_lead_slice, V_x_2d, N_y,
        phase, symmetric=SYMMETRIC, eta=eta
    )

    # Infinite — already have normal slices
    G_inf, *_ = modules.get_rgf_sns(
        H_normal_slices, V_x_2d, gL, gR, energy, eta=eta, return_full=False
    )

    # Finite — stitch prebuilt SC slices + normal slices, no full rebuild
    phi_L = -phase / 2 if SYMMETRIC else 0.0
    phi_R = phase / 2 if SYMMETRIC else phase
    H_L = build_sns_slice_2d(N_y, t, mu_sc, B, alpha, Delta * np.exp(1j * phi_L))
    H_R = build_sns_slice_2d(N_y, t, mu_sc, B, alpha, Delta * np.exp(1j * phi_R))
    H_full = ([H_L.copy() for _ in range(SL)] + 
          [s.copy() for s in H_normal_slices] + 
          [H_R.copy() for _ in range(SR)])

    G_fin, *_ = modules.get_rgf_finite_system(
        H_full, V_x_2d, energy, eta=eta, return_full=False
    )

    blk_i = get_local_block(G_inf[probe_x[0]],      probe_y[0])
    blk_f = get_local_block(G_fin[SL + probe_x[0]], probe_y[0])
    return (
        -np.imag(np.trace(blk_i)) / np.pi,
        -np.imag(np.trace(blk_f)) / np.pi,
    )


def compute_ldos_row(energy, phases, H_normal_slices, H_lead_slice, V_x_2d):
    """Compute all phases for a single energy — one job per energy."""
    row_inf = np.zeros(len(phases))
    row_fin = np.zeros(len(phases))
    for p_idx, phase in enumerate(phases):
        inf_val, fin_val = compute_ldos_point(energy, phase, H_normal_slices, H_lead_slice, V_x_2d)
        row_inf[p_idx] = inf_val
        row_fin[p_idx] = fin_val
    return row_inf, row_fin

results = Parallel(n_jobs=-1)(
    delayed(compute_ldos_row)(E, phases, H_normal_slices, H_lead_slice, V_x_2d)
    for E in energies
)

ldos_2d_inf = np.array([r[0] for r in results])  # (N_E, N_PHI)
ldos_2d_fin = np.array([r[1] for r in results])

#%% ── Figure 1: LDOS vs Energy (probe sites) ────────────────────────────────
fig, axes = plt.subplots(
    len(probe_x), len(probe_y),
    figsize=(4 * len(probe_y), 3 * len(probe_x)),
    sharex=True, squeeze=False
)
for ix, x in enumerate(probe_x):
    for iy, y in enumerate(probe_y):
        ax = axes[ix, iy]
        ax.plot(energies, ldos_e[ix, iy, :, 0], '--', label='finite')
        ax.plot(energies, ldos_e[ix, iy, :, 1], '-',  label='infinite')
        ax.set_title(f"x={x}, y={y}")
        ax.grid()
        if ix == len(probe_x) - 1:
            ax.set_xlabel("Energy")
        if iy == 0:
            ax.set_ylabel("LDOS")
        if ix == 0 and iy == 0:
            ax.legend()
plt.suptitle(f"LDOS vs Energy  (φ = {phi_fixed/np.pi:.2f}π)")


# ── Figure 2: Zero-energy spatial LDOS map (x, y) ────────────────────────
E_zero_idx = np.argmin(np.abs(energies))

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
im0 = ax[0].imshow(ldos_map_fin[:, :, E_zero_idx].T, origin='lower',
                   aspect='auto', extent=[0, SM, 0, N_y])
ax[0].set_title("Finite system")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
im1 = ax[1].imshow(ldos_map_inf[:, :, E_zero_idx].T, origin='lower',
                   aspect='auto', extent=[0, SM, 0, N_y])
ax[1].set_title("Infinite leads")
ax[1].set_xlabel("x")
plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")
plt.suptitle(f"Zero-energy LDOS map  (φ = {phi_fixed/np.pi:.2f}π)")


# ── Figure 3: LDOS(x, E) for selected y-sites ────────────────────────────
selected_y = [0, N_y // 2, N_y - 1]

fig, axes = plt.subplots(1, len(selected_y),
                         figsize=(5 * len(selected_y), 4), sharey=True)
for i, y0 in enumerate(selected_y):
    im = axes[i].imshow(ldos_map_inf[:, y0, :].T, aspect='auto', origin='lower',
                        extent=[0, SM, energies[0], energies[-1]])
    axes[i].set_title(f"y = {y0}")
    axes[i].set_xlabel("x")
    if i == 0:
        axes[i].set_ylabel("Energy")
plt.colorbar(im, ax=axes.ravel().tolist(), label="LDOS")
plt.suptitle("LDOS(x, E) — infinite leads — selected y-sites")


# ── Figure 4: LDOS(y, E) for selected x-sites ────────────────────────────
selected_x = [0, SM // 2, SM - 1]

fig, axes = plt.subplots(1, len(selected_x),
                         figsize=(5 * len(selected_x), 4), sharey=True)
for i, x0 in enumerate(selected_x):
    im = axes[i].imshow(ldos_map_inf[x0, :, :].T, aspect='auto', origin='lower',
                        extent=[0, N_y, energies[0], energies[-1]])
    axes[i].set_title(f"x = {x0}")
    axes[i].set_xlabel("y")
    if i == 0:
        axes[i].set_ylabel("Energy")
plt.colorbar(im, ax=axes.ravel().tolist(), label="LDOS")
plt.suptitle("LDOS(y, E) — infinite leads — selected x-sites")


# ── Figure 5: LDOS & Pairing vs Phase (probe sites) ──────────────────────
fig, axes = plt.subplots(len(probe_x), len(probe_y) * 2,
                         figsize=(10, 3 * len(probe_x)), sharex=True)
axes = np.atleast_2d(axes)
for ix, x in enumerate(probe_x):
    for iy, y in enumerate(probe_y):
        ax_l = axes[ix, 2 * iy]
        ax_p = axes[ix, 2 * iy + 1]
        ax_l.plot(phases / np.pi, ldos_p[ix, iy, :, 0], '--', label='finite')
        ax_l.plot(phases / np.pi, ldos_p[ix, iy, :, 1], '-',  label='infinite')
        ax_l.set_title(f"x={x}, y={y}")
        ax_l.set_ylabel("LDOS")
        ax_l.grid()
        ax_p.plot(phases / np.pi, pair_p[ix, iy, :, 0], '--', label='finite')
        ax_p.plot(phases / np.pi, pair_p[ix, iy, :, 1], '-',  label='infinite')
        ax_p.set_ylabel("Pairing")
        ax_p.grid()
        if ix == len(probe_x) - 1:
            ax_l.set_xlabel("φ/π")
            ax_p.set_xlabel("φ/π")
        if ix == 0 and iy == 0:
            ax_l.legend()
plt.suptitle("LDOS & Pairing vs Phase  (E = 0)")


# ── Figure 6: LDOS map (energy × phase) ──────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
im0 = ax[0].contourf(phases / np.pi, energies, ldos_2d_fin, levels=100, cmap='viridis')
ax[0].set_title("Finite system")
ax[0].set_xlabel("φ/π")
ax[0].set_ylabel("Energy")
im1 = ax[1].contourf(phases / np.pi, energies, ldos_2d_inf, levels=100, cmap='viridis')
ax[1].set_title("Infinite leads")
ax[1].set_xlabel("φ/π")
plt.colorbar(im1, ax=ax.ravel().tolist(), label="LDOS")
plt.suptitle(f"LDOS(E, φ) — probe x={probe_x[0]}, y={probe_y[0]}")

plt.show()
print("Finished.")
# %%
