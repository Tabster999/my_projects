# %% ── Imports ──────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from joblib import Parallel, delayed
import matplotlib.colors as mcolors

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
import modules

plt.rcParams['font.size']        = 12
plt.rcParams['axes.titlesize']   = 13
plt.rcParams['axes.labelsize']   = 12
plt.rcParams['xtick.labelsize']  = 11
plt.rcParams['ytick.labelsize']  = 11
plt.rcParams['legend.fontsize']  = 10
plt.rcParams['figure.titlesize'] = 14

# %% ── Units & Hopping Energy ──────────────────────────────────────────────
hbar_sq_over_2m0 = 38.1      # meV·nm²
m_eff_ratio      = 0.038     # m_eff / m₀ (HgTe)
E_kin            = hbar_sq_over_2m0 / m_eff_ratio   # ℏ²/(2 m_eff) in meV·nm²

a_x = 10.0               # nm
t   = E_kin / a_x**2         # meV 

# %% ── Pauli Matrices ──────────────────────────────────────────────────────
sx = np.array([[0, 1],  [1,  0]], dtype=np.complex128)
sy = np.array([[0,-1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
s0 = np.eye(2, dtype=np.complex128)
tx = np.array([[0, 1],  [1,  0]], dtype=np.complex128)
ty = np.array([[0,-1j], [1j, 0]], dtype=np.complex128)
tz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
t0 = np.eye(2, dtype=np.complex128)

def kron(A, B): return np.kron(A, B)

TAU_X  = kron(tx, s0);  TAU_Y  = kron(ty, s0);  TAU_Z  = kron(tz, s0)
SIG_X  = kron(t0, sx);  SIG_Y  = kron(t0, sy);  SIG_Z  = kron(t0, sz)
TZ_SX  = kron(tz, sx);  TZ_SY  = kron(tz, sy);  TZ_SZ  = kron(tz, sz)

# %% ── Hamiltonian Building Blocks ─────────────────────────────────────────
def onsite_scharf(t: float, mu: float, h: float, delta: complex, twod: bool = True) -> np.ndarray:
    z = 4 * t - mu if twod else 2 * t - mu
    
    return np.array([
        [ z - h,               0,              delta,              0       ],
        [ 0,                   z + h,          0,                 -delta   ],
        [ delta.conjugate(),   0,             -z - h,             0       ],
        [ 0,                  -delta.conjugate(), 0,              -z + h  ]
    ], dtype=np.complex128)


def hopping_scharf(alpha, beta):

    soc_hop = (1j / (2.0 * a_x)) * (alpha * TZ_SY + beta * TZ_SX)
    return -t * TAU_Z + soc_hop 

def build_slices_scharf(ky, mu_sc, mu_n, EZ_vec, alpha, beta, Delta, phi, SL, SM, SR):
    T     = hopping_scharf(alpha, beta)
    phi_L = -phi / 2.0
    phi_R = +phi / 2.0

    H_L = onsite_scharf(ky, mu_sc, EZ_vec, alpha, beta, Delta, phi_L, True)
    H_R = onsite_scharf(ky, mu_sc, EZ_vec, alpha, beta, Delta, phi_R, True)
    H_M = onsite_scharf(ky, mu_n,  EZ_vec, alpha, beta, 0.0,   0.0,  False)

    H_slices = ([H_L.copy() for _ in range(SL)] +
                [H_M.copy() for _ in range(SM)] +
                [H_R.copy() for _ in range(SR)])
    return H_slices, T


def t_matrix_x_scharf(alpha, beta):
    soc_hop = (1j / (2.0 * a_x)) * (alpha * TZ_SY + beta * TZ_SX)
    return -t * TAU_Z + soc_hop


def t_matrix_y_scharf(alpha, beta):
    soc_hop = (-1j / (2.0 * a_x)) * (alpha * TZ_SX + beta * TZ_SY)
    return -t * TAU_Z + soc_hop


def build_sns_slice_2d_scharf(N_y, mu, EZ_vec, alpha, beta, Delta, phi_site, is_SC):
    """Build a single y-chain slice for the finite y-width 2D Hamiltonian (Eq. B6)."""
    H_onsite = onsite_scharf(0.0, mu, EZ_vec, alpha, beta, Delta, phi_site, is_SC)
    T_y = t_matrix_y_scharf(alpha, beta)

    I_y   = np.eye(N_y, dtype=np.complex128)
    off_y = np.eye(N_y, k=1, dtype=np.complex128)

    return (
        np.kron(I_y,      H_onsite)
      + np.kron(off_y,    T_y)
      + np.kron(off_y.T,  T_y.conj().T)
    )


def build_slices_scharf_2d(N_y, mu_sc, mu_n, EZ_vec, alpha, beta, Delta, phi, SL, SM, SR):
    """Build the 2D finite-y-width SNS slice list for RGF."""
    phi_L = -phi / 2.0
    phi_R = +phi / 2.0

    H_L = build_sns_slice_2d_scharf(N_y, mu_sc, EZ_vec, alpha, beta, Delta, phi_L, True)
    H_R = build_sns_slice_2d_scharf(N_y, mu_sc, EZ_vec, alpha, beta, Delta, phi_R, True)
    H_M = build_sns_slice_2d_scharf(N_y, mu_n,  EZ_vec, alpha, beta, 0.0,   0.0,  False)
    V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), t_matrix_x_scharf(alpha, beta))

    H_slices = ([H_L.copy() for _ in range(SL)] +
                [H_M.copy() for _ in range(SM)] +
                [H_R.copy() for _ in range(SR)])
    return H_slices, V_x_2d


def build_full_hamiltonian_matrix_2d(N_y, mu_sc, mu_n, EZ_vec, alpha, beta,
                                     Delta, phi, SL, SM, SR):
    """Constructs the dense finite-y-width Hamiltonian for direct diagonalization."""
    H_slices, V_x_2d = build_slices_scharf_2d(N_y, mu_sc, mu_n, EZ_vec,
    alpha, beta, Delta, phi,
    SL, SM, SR
    )

    N_slices = len(H_slices)
    dof = 4 * N_y
    H_full = np.zeros((dof * N_slices, dof * N_slices), dtype=np.complex128)

    for idx in range(N_slices):
        start = idx * dof
        end = (idx + 1) * dof
        H_full[start:end, start:end] = H_slices[idx]
        if idx < N_slices - 1:
            H_full[start:end, end:end + dof] = V_x_2d
            H_full[end:end + dof, start:end] = V_x_2d.conj().T
    return H_full


def build_full_hamiltonian_matrix(ky, mu_sc, mu_n, EZ_vec, alpha, beta, Delta, phi, SL, SM, SR):
    H_slices, T = build_slices_scharf(ky, mu_sc, mu_n, EZ_vec, alpha, beta, Delta, phi, SL, SM, SR)
    N_slices = len(H_slices)
    H_full = np.zeros((4 * N_slices, 4 * N_slices), dtype=np.complex128)
    
    for idx in range(N_slices):
        H_full[4*idx:4*(idx+1), 4*idx:4*(idx+1)] = H_slices[idx]
        if idx < N_slices - 1:
            H_full[4*idx:4*(idx+1), 4*(idx+1):4*(idx+2)] = T
            H_full[4*(idx+1):4*(idx+2), 4*idx:4*(idx+1)] = T.conj().T
    return H_full

def apply_phase_scharf(gL_raw, gR_raw, phi):
    eL = np.exp(-1j * phi / 4.0)
    eR = np.exp(+1j * phi / 4.0)

    U_L = np.diag([eL, eL, eL.conj(), eL.conj()]).astype(np.complex128)
    U_R = np.diag([eR, eR, eR.conj(), eR.conj()]).astype(np.complex128)
    return U_L @ gL_raw @ U_L.conj().T, U_R @ gR_raw @ U_R.conj().T


def apply_phase_scharf_2d(gL_raw, gR_raw, phi, N_y):
    eR = np.exp(+1j * phi / 2.0)

    U_L = np.eye(4 * N_y, dtype=np.complex128)
    U_R = np.kron(np.eye(N_y, dtype=np.complex128),
                  np.diag([eR, eR, eR.conj(), eR.conj()]).astype(np.complex128))
    return U_L @ gL_raw @ U_L.conj().T, U_R @ gR_raw @ U_R.conj().T


def ldos_from_block(blk):
    return -np.imag(np.trace(blk)) / np.pi

# %% ── Physical Parameters ─────────────────────────────────────────────────
theta_soc  = np.pi * 0.15 
lambda_soc = 16.0 
alpha      = lambda_soc * np.cos(theta_soc)
beta       = lambda_soc * np.sin(theta_soc)

mu_sc, mu_n, Delta = 1.0, 0.7, 0.25
SM, SL, SR = 60, 120, 120
N_slices = SL + SM + SR
N_y = 6

EZ_mag_default = 1.1
EZ_dir         = np.array([-np.sin(theta_soc), np.cos(theta_soc), 0.0]) 
EZ_default     = EZ_mag_default * EZ_dir
eta            = 1e-3

# Grids Setup
N_E, N_PHI, N_EZ = 71, 71, 71
energies = np.linspace(-1.05 * Delta, 1.05 * Delta, N_E)        
phases   = np.linspace(0, 2 * np.pi, N_PHI)
EZ_array = np.linspace(0.0, 1.2, N_EZ) 

phi_fixed = np.pi
EZ_topo   = 1.1

# %% ── SWEEP 1: ABS Dispersion from Diagonalization E(k_y) ──────────────────
print("\n══ Sweep 1: Direct Diagonalization E(k_y) at φ=π ══")

def diag_ky(ky):
    H_mat = build_full_hamiltonian_matrix(ky, mu_sc, mu_n, EZ_default, alpha, beta, Delta, phi_fixed, SL, SM, SR)
    return la.eigvalsh(H_mat)

H_full_2d = build_full_hamiltonian_matrix_2d(N_y, mu_sc, mu_n, EZ_default,
                                             alpha, beta, Delta, phi_fixed,
                                             SL, SM, SR)
eigenvalues_2d = la.eigvalsh(H_full_2d)

# %% ── SWEEP 2: Phase Spectrum LDOS(E, φ) with finite-y-width leads ───────────
print("══ Sweep 2: LDOS(E, φ) for finite-y-width 2D geometry ══")
V_x_2d = np.kron(np.eye(N_y, dtype=np.complex128), t_matrix_x_scharf(alpha, beta))

print(" -> Precomputing 2D energy lead GFs...")
gL_E_cache, gR_E_cache = [], []
H_lead_2d = build_sns_slice_2d_scharf(N_y, mu_sc, EZ_default, alpha, beta, Delta, 0.0, True)
for e in energies:
    gL, gR = modules.get_surface_gfs_2d_phased(e, H_lead_2d, V_x_2d, N_y, 0.0, symmetric=False, eta=eta)
    gL_E_cache.append(gL)
    gR_E_cache.append(gR)

H_M_2d = [build_sns_slice_2d_scharf(N_y, mu_n, EZ_default, alpha, beta, 0.0, 0.0, False)
          for _ in range(SM)]

phiE_pairs = [(i_phi, i_e, phi, e) for i_phi, phi in enumerate(phases) for i_e, e in enumerate(energies)]

def compute_phi_E(i_phi, i_e, phi, e):
    g_L, g_R = apply_phase_scharf_2d(gL_E_cache[i_e], gR_E_cache[i_e], phi, N_y)
    G_inf, *_ = modules.get_rgf_sns(H_M_2d, V_x_2d, g_L, g_R, e, eta=eta, return_full=False)

    H_full, V = build_slices_scharf_2d(N_y, mu_sc, mu_n, EZ_default, alpha, beta,
                                      Delta, phi, SL, SM, SR)
    G_fin, *_ = modules.get_rgf_finite_system(H_full, V, e, eta=eta, return_full=False)
    return i_phi, i_e, ldos_from_block(G_inf[SM // 2]), ldos_from_block(G_fin[SL + SM // 2])

results_phiE = Parallel(n_jobs=-1)(delayed(compute_phi_E)(*p) for p in phiE_pairs)

ldos_phi_E_inf = np.zeros((N_PHI, N_E))
ldos_phi_E_fin = np.zeros((N_PHI, N_E))
for i_phi, i_e, li, lf in results_phiE:
    ldos_phi_E_inf[i_phi, i_e] = li
    ldos_phi_E_fin[i_phi, i_e] = lf

# %% ── SWEEP 3: Spatial LDOS(x) inside the Topological Regime ───────────────
print("══ Sweep 3: Spatial LDOS(x) Profile at E=0 for finite-y-width 2D geometry ══")
EZ_topo_vec = EZ_topo * EZ_dir

H_lead_t = build_sns_slice_2d_scharf(N_y, mu_sc, EZ_topo_vec, alpha, beta, Delta, 0.0, True)

print(" -> Computing 2D surface Green's functions for the topological regime...")
gL_ph, gR_ph = modules.get_surface_gfs_2d_phased(0.0, H_lead_t, V_x_2d, N_y, phi_fixed, symmetric=False, eta=eta)

H_M_t   = [build_sns_slice_2d_scharf(N_y, mu_n, EZ_topo_vec, alpha, beta, 0.0, 0.0, False)
           for _ in range(SM)]
G_inf_t, *_ = modules.get_rgf_sns(H_M_t, V_x_2d, gL_ph, gR_ph, 0.0, eta=eta, return_full=True)

H_full_t, V_t = build_slices_scharf_2d(N_y, mu_sc, mu_n, EZ_topo_vec, alpha,
                                      beta, Delta, phi_fixed, SL, SM, SR)
G_fin_t, *_   = modules.get_rgf_finite_system(H_full_t, V_t, 0.0, eta=eta, return_full=True)

ldos_x_inf = np.array([ldos_from_block(G_inf_t[x]) for x in range(SM)])
ldos_x_fin = np.array([ldos_from_block(G_fin_t[x]) for x in range(N_slices)])

# %% ── SWEEP 4: Majorana Probability Density in the Middle Region ──────────
print("══ Sweep 4: Majorana Probability Density in the Middle Region ══")
H_full_2d_topo = build_full_hamiltonian_matrix_2d(N_y, mu_sc, mu_n, EZ_topo_vec,
                                                alpha, beta, Delta, phi_fixed,
                                                SL, SM, SR)
evals_2d, evecs_2d = la.eigh(H_full_2d_topo)
idx_2d = np.argsort(np.abs(evals_2d))[:2]
psi_2d = evecs_2d[:, idx_2d]
print(" -> Lowest finite-width energies:", np.round(np.real(evals_2d[idx_2d]), 8))

density_xy = np.zeros((SL + SM + SR, N_y), dtype=np.float64)
for state in psi_2d.T:
    psi_xy = state.reshape(N_slices, N_y, 4)
    density_xy += np.sum(np.abs(psi_xy)**2, axis=2)

x_positions = np.arange(SL + SM + SR) * a_x
mid_x_start = SL
mid_x_end = SL + SM
x_middle = x_positions[mid_x_start:mid_x_end]
density_xy = density_xy[mid_x_start:mid_x_end, :]
y_array = np.arange(N_y) * a_x

if np.max(density_xy) > 0:
    density_xy /= np.max(density_xy)

# %% ── Visualizations ───────────────────────────────────────────────────────
print("══ Generating Clean Figures ══")
import matplotlib.cm as cm
viridis = cm.get_cmap('viridis')
E_plot   = energies / Delta
phi_plot = phases   / np.pi
NUM_CENTRAL_BANDS = 4

# ── FIGURE 1: ABS Spectrum via Diagonalization
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

center_idx = len(eigenvalues_2d) // 2
n_sel = NUM_CENTRAL_BANDS // 2
start_idx = max(0, center_idx - n_sel)
end_idx = min(len(eigenvalues_2d), center_idx + n_sel)
selected_energies = eigenvalues_2d[start_idx:end_idx]

ax.plot(np.arange(len(selected_energies)), selected_energies / Delta,
        marker='o', color=viridis(0.65), lw=2.0)
ax.set_xlabel('State index')
ax.set_ylabel(r'$E / \Delta$')
ax.set_title(rf'Selected low-energy states of the finite-y-width 2D Hamiltonian at $\phi=\pi$, $E_Z={EZ_mag_default:.2f}\ \mathrm{{meV}}$')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)
plt.show()

# ── FIGURE 2: LDOS(E, φ) Panels
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)
PP, EE = np.meshgrid(phases / np.pi, energies / Delta, indexing="ij")
vmax = max(np.percentile(ldos_phi_E_inf, 99), np.percentile(ldos_phi_E_fin, 99))

for ax, data, title in zip(axes, [ldos_phi_E_inf, ldos_phi_E_fin], ["Infinite Leads (RGF)", "Finite Leads"]):
    cf = ax.contourf(PP, EE, data, levels=150, cmap="magma", vmax=vmax, extend="max")
    ax.set_xlabel(r"$\phi / \pi$")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.colorbar(cf, ax=ax, label="LDOS")
axes[0].set_ylabel(r"$E / \Delta$")
fig.suptitle(r"Fig 2: Phase Spectrum $LDOS(\phi, E)$ for the finite-y-width 2D junction")
plt.show()


# ── FIGURE 3: LDOS(x, y) Color Plot in the Middle Normal Region
plt.figure(figsize=(8, 5), constrained_layout=True)
plt.imshow(density_xy, aspect='auto', origin='lower', extent=(y_array[0], y_array[-1], x_middle[0], x_middle[-1]), cmap='inferno')
plt.colorbar(label=r'Normalized $|\Psi|^2$')
plt.xlabel(r'$y$ [nm]')
plt.ylabel(r'$x$ [nm]')
plt.title(r'Fig 3: LDOS$(x, y)$ Color Plot for the Middle Region')
plt.show()


# ── FIGURE 4: Spatial LDOS Profile
plt.figure(figsize=(8, 4.5), constrained_layout=True)
x_inf = np.arange(SM) * a_x + (SL * a_x)
x_fin = np.arange(N_slices) * a_x

plt.plot(x_fin, ldos_x_fin, "-", label="Finite Leads", color="black", alpha=0.85, lw=1.5)
plt.plot(x_inf, ldos_x_inf, "--", label="Infinite Leads (Scattering Region)", color="orange", lw=2)

plt.axvspan(0, SL*a_x, color="grey", alpha=0.12, label="Left Lead")
plt.axvspan(SL*a_x, (SL+SM)*a_x, color="blue", alpha=0.05, label="Normal Region")
plt.axvspan((SL+SM)*a_x, N_slices*a_x, color="grey", alpha=0.12, label="Right Lead")

plt.xlabel(r"Spatial Coordinate $x$ [nm]")
plt.ylabel("Local Density of States (LDOS)")
plt.title(r"Fig 4: Spatial $LDOS(x)$ Profile at $E=0,\ \phi=\pi$")
plt.xlim(0, N_slices * a_x)
plt.grid(alpha=0.2)
plt.legend(loc="upper right")
plt.xlim(SL*a_x, (SL+SM) * a_x)
plt.show()

# %% ── MAJORANA & PHASE DIAGNOSTICS SUITE ────────────────────────────────────
print("\n" + "═"*60)
print(" 🧬 EXPLICIT MAJORANA & PHASE TRANSITION DIAGNOSTICS 🧬")
print("═"*60)

print("\n🔍 Diagnostic 1: Numerical Bulk Phase Transition Boundary")
ez_scan = np.linspace(0.0, 2.0, 400)
bulk_gaps = []

T_hop = hopping_scharf(alpha, beta)
for ez in ez_scan:
    ez_v = ez * EZ_dir
    H_on = onsite_scharf(0.0, mu_sc, ez_v, alpha, beta, Delta, phi_site=0.0, is_SC=True)
    H_bulk = H_on + T_hop + T_hop.conj().T
    egs = la.eigvalsh(H_bulk)
    bulk_gaps.append(np.min(np.abs(egs)))

critical_EZ = ez_scan[np.argmin(bulk_gaps)]
print(f"   ℹ️ Calculated Critical Zeeman Field: E_Z,c ≈ {critical_EZ:.4f} meV")

if EZ_default[0]**2 + EZ_default[1]**2 + EZ_default[2]**2 > critical_EZ**2:
    print(f"   ✅ STATUS: Your current E_Z ({EZ_mag_default} meV) is safely TOPOLOGICAL.")
else:
    print(f"   ❌ STATUS: Your current E_Z ({EZ_mag_default} meV) is TRIVIAL (Below transition).")


# ── 2. ELECTRON-HOLE SYMMETRIC LOCKING ───────────────────────────────────────
print("\n🔍 Diagnostic 2: Electron-Hole Balance (Majorana Self-Conjugacy)")

m_state = psi_2d[:, 0]
state_reshaped = m_state.reshape(N_slices, N_y, 4) 

u_prob = np.sum(np.abs(state_reshaped[:, :, 0])**2 + np.abs(state_reshaped[:, :, 1])**2)
v_prob = np.sum(np.abs(state_reshaped[:, :, 2])**2 + np.abs(state_reshaped[:, :, 3])**2)
eh_ratio = min(u_prob, v_prob) / max(u_prob, v_prob)

print(f"   ℹ️ Total Electron Weight (|u|^2): {u_prob:.4f}")
print(f"   ℹ️ Total Hole Weight     (|v|^2): {v_prob:.4f}")

if eh_ratio > 0.99:
    print("   ✅ STATUS: Perfect Electron-Hole Balance. Highly indicative of an MBS.")
else:
    print(f"   ⚠️ WARNING: Unbalanced charge distribution (Ratio: {eh_ratio:.2f}).")
    print("      This state may be a trivial Andreev Bound State trapped by potential walls.")


print("\n🔍 Diagnostic 3: Spatial Allocation & Hybridization Risk")

prob_per_slice = np.sum(np.abs(state_reshaped)**2, axis=1)

left_lead_weight  = np.sum(prob_per_slice[:SL])
mid_region_weight = np.sum(prob_per_slice[SL:SL+SM])
right_lead_weight = np.sum(prob_per_slice[SL+SM:])

print(f"   ℹ️ Left Lead Allocation:  {left_lead_weight*100:.1f}%")
print(f"   ℹ️ Normal Junction Core:  {mid_region_weight*100:.1f}%")
print(f"   ℹ️ Right Lead Allocation: {right_lead_weight*100:.1f}%")

edge_leakage = np.sum(prob_per_slice[:5]) + np.sum(prob_per_slice[-5:])
if edge_leakage > 0.05:
    print(f"   ⚠️ WARNING: High boundary leakage detected ({edge_leakage*100:.1f}% at outer edges).")
    print("      Your leads (SL/SR) are too short! The Majorana is hitting the physical walls,")
    print("      which causes artificial energy splitting. Increase SL and SR.")
else:
    print("   ✅ STATUS: Well-isolated. Wavefunction decays fully before reaching outer boundaries.")

print("═"*60 + "\n")
# %%
