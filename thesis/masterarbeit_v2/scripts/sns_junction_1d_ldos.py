#%% ── Imports ──────────────────────────────────────────────────────────────────
import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from joblib import Parallel, delayed

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent
sys.path.insert(0, str(module_root))
import modules as myf
"""
=============================================================================
SURFACE GREEN'S FUNCTION CONVENTION AND PHASE CORRECTION

get_surface_gf(E, onsite, hopping, eta) runs the Sancho-López algorithm for
a semi-infinite chain where each unit cell hops to the NEXT cell via `hopping`.
It returns the GF of the SURFACE (outermost) site.

For the SNS junction, the two leads face each other:

  ... [L bulk] --V†--> [L surface] --V--> [N region] <--V†-- [R surface] <--V-- [R bulk] ...

The RIGHT lead's surface GF (g_R):
  - Chain propagates rightward into the bulk via V
  - Call: get_surface_gf(E, onsite_sc, V)
  - Couples to N region via: Sigma_R = V @ g_R @ V†
    (N site hops RIGHT into lead via V, returns via V†)

The LEFT lead's surface GF (g_L):
  - Chain propagates leftward into the bulk via V†
  - Call: get_surface_gf(E, onsite_sc, V†)
  - Couples to N region via: Sigma_L = V† @ g_L @ V
    (N site hops LEFT into lead via V†, returns via V)


SUMMARY OF SELF-ENERGY CONTRACTIONS (must match get_rgf_sns internals):
  Sigma_R = V  @ g_R @ V†   (used at right boundary, i = N-1)
  Sigma_L = V† @ g_L @ V    (used at left  boundary, i = 0  )
=============================================================================
"""

# ── Colour palette (colour-blind friendly) ─────────────────────────────────
C_FIN  = "#d6604d"   # red   – finite RGF
C_INF  = "#4dac26"   # green – infinite leads
ALPHA  = 0.85

#%% ── Helpers ───────────────────────────────────────────────────────────────────

def get_ldos(G_block):
    """LDOS from the local retarded GF block: -Im Tr G / pi."""
    return -np.imag(np.trace(G_block)) / np.pi


def get_pairing(G_block):
    """Anomalous amplitude proxy |G[0,3]|."""
    return np.abs(G_block[0, 3])


def get_surface_gfs(E, phi, onsite_sc, symmetric=False, eta=1e-3):
    """
    Surface GFs with SC phase applied as Nambu gauge rotation.
    
    Convention (matches build_sns_junction_sliced):
      U(theta)† @ H(Delta) @ U(theta) = H(Delta * exp(i*theta))
    
    Symmetric gauge:
      Delta_L = Delta*exp(-i*phi/2)  →  UL = phase_matrix(-phi/2)
      Delta_R = Delta*exp(+i*phi/2)  →  UR = phase_matrix(+phi/2)
    
    Asymmetric gauge:
      Delta_L = Delta (real)          →  UL = identity
      Delta_R = Delta*exp(+i*phi)     →  UR = phase_matrix(phi)
    """
    if symmetric:
        UL = myf.phase_matrix(-phi / 2.0)
        UR = myf.phase_matrix(phi / 2.0)
    else:
        UL = np.eye(4, dtype=np.complex128)
        UR = myf.phase_matrix(phi)

    g_L, _ = myf.get_surface_gf(E, onsite_sc, V.conj().T, eta=eta)
    g_R, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)

    return UL.conj().T @ g_L @ UL, UR.conj().T @ g_R @ UR

def ph_check(name, M):
    ok = np.allclose(M, -C_ph @ M.conj() @ C_ph, atol=1e-12)
    print(f"  PH symmetry [{name}]: {'✓ PASS' if ok else '✗ FAIL'}")

def compute_energy_slice(e_idx, E):
    """Compute LDOS for all phases at a fixed energy."""
    row = np.zeros(N_PHI)

    for p_idx, phi in enumerate(phases):
        g_L, g_R = get_surface_gfs(E, phi, onsite_sc, SYMMETRIC, eta=eta)

        G_inf, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False
        )

        row[p_idx] = get_ldos(G_inf[probe])

    return e_idx, row

def compute_energy_slice_fin(e_idx, E):
    """Compute LDOS for all phases at fixed energy (finite system)."""
    row = np.zeros(N_PHI)

    for p_idx, phi in enumerate(phases):
        # rebuild full SNS Hamiltonian with phase
        H_full_p, _ = myf.build_sns_junction_sliced(
            t, mu_sc, mu_n, alpha, B, Delta, phi,
            SL, SR, SM, symmetric=SYMMETRIC
        )

        G_fin, *_ = myf.get_rgf_finite_system(
            H_full_p, V, E, eta=eta, return_full=False
        )

        row[p_idx] = get_ldos(G_fin[SL + probe])

    return e_idx, row
#%% ── Parameters ────────────────────────────────────────────────────────────────
SYMMETRIC = False     # True  → ±phi/2 on left/right
                     # False → full phi on right only

SL, SM, SR = 100, 40, 100        # sites: left SC | normal | right SC
DOF        = 4
Delta      = 0.1
mu_sc      = 0.1
mu_n       = 0.025
t          = 1.0
alpha      = 0.6
B          = 0.3
eta        = 1e-4
phi_fixed  = np.pi

N_E   = 121                     # energy points
N_PHI = 121                   # phase points
energies = np.linspace(-1.05*Delta, 1.05*Delta, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

probe_sites  = [0, SM // 2, SM - 1]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# ── Fixed objects ─────────────────────────────────────────────────────────────
V      = myf.t_matrix_x(t, alpha)
V_dag  = V.conj().T
onsite_sc    = myf.onsite_matrix(t, mu_sc, B, Delta)   # real Delta
H_mid_slices = [myf.onsite_matrix(t, mu_n, 0, 0.0) for _ in range(SM)]

H_full_slices, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC)

# ── Particle-hole symmetry check on Hamiltonians ──────────────────────────────
C_ph = np.array([[0,0,0,-1], [0,0,1,0], [0,1,0,0], [-1,0,0,0]])   # antidiag identity – PH conjugation matrix


print("\n══ Hamiltonian PH checks ══")
ph_check("onsite_sc", onsite_sc)
ph_check("V hopping", V)
ph_check("onsite_N",  myf.onsite_matrix(t, mu_n, B, 0.0))

# ── Topological phase check ───────────────────────────────────────────────────
topo_threshold = np.sqrt(Delta**2 + mu_sc**2)
in_topo = B > topo_threshold
print(f"\n══ Topological phase ══")
print(f"  B = {B:.3f},  sqrt(Delta^2+mu^2) = {topo_threshold:.4f}")
print(f"  In topological phase: {'YES ✓' if in_topo else 'NO  ✗'}")
print(f"\n══ Gauge: {'symmetric ±phi/2' if SYMMETRIC else 'asymmetric, full phi on right'} ══")

# ── Quick phase-rotation sanity check 
print("\n══ Gauge-rotation self-test ══")
for phi_test in [0.0, np.pi/2, np.pi]:
    U = myf.phase_matrix(phi_test)   # U(-phi) gives Delta*exp(+i*phi)
    onsite_rot = U.conj().T @ onsite_sc @ U
    onsite_ref = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(1j * phi_test))
    err = np.max(np.abs(onsite_rot - onsite_ref))
    print(f"  phi={phi_test/np.pi:.2f}pi: err={err:.2e}  {'✓' if err < 1e-12 else '✗'}")

#%% energy sweep 
# ENERGY SWEEP  (phi fixed = pi)
print("\n══ Running energy sweep ══")

# shape: (probe, energy, method)  methods: 0=finite RGF  1=Sancho-López
ldos_e  = np.zeros((len(probe_sites), N_E, 2))
pair_e  = np.zeros((len(probe_sites), N_E, 2))
Gblk_e  = np.zeros((len(probe_sites), N_E, 2, DOF, DOF), dtype=np.complex128)
ldos_all_fin = np.zeros((SM, N_E))
ldos_all_inf = np.zeros((SM, N_E))

for e_idx, E in enumerate(energies):
    G_fin, *_ = myf.get_rgf_finite_system(
        H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, phi_fixed, onsite_sc, SYMMETRIC, eta=eta)
    G_inf, *_ = myf.get_rgf_sns(
        H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_fin = G_fin[SL + s]
        blk_inf = G_inf[s]
        for m, blk in enumerate([blk_fin, blk_inf]):
            ldos_e[s_idx, e_idx, m] = get_ldos(blk)
            pair_e[s_idx, e_idx, m] = get_pairing(blk)
            Gblk_e[s_idx, e_idx, m] = blk

    for s in range(SM):
        ldos_all_fin[s, e_idx] = get_ldos(G_fin[SL + s])
        ldos_all_inf[s, e_idx] = get_ldos(G_inf[s])

#%% phase sweep 
# PHASE SWEEP  (E = 0)
print("══ Running phase sweep ══")

energy_fixed = 0.0
ldos_p = np.zeros((len(probe_sites), N_PHI, 2))
pair_p = np.zeros((len(probe_sites), N_PHI, 2))

for p_idx, phi in enumerate(phases):
    H_full_p, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta, phi,
        SL, SR, SM, symmetric=SYMMETRIC)

    G_fin_p, *_ = myf.get_rgf_finite_system(
        H_full_p, V, energy_fixed, eta=eta, return_full=False)

    g_L_p, g_R_p = get_surface_gfs(energy_fixed, phi, onsite_sc, SYMMETRIC, eta=eta)
    G_inf_p, *_ = myf.get_rgf_sns(
        H_mid_slices, V, g_L_p, g_R_p, energy_fixed, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_fin = G_fin_p[SL + s]
        blk_inf = G_inf_p[s]
        for m, blk in enumerate([blk_fin, blk_inf]):
            ldos_p[s_idx, p_idx, m] = get_ldos(blk)
            pair_p[s_idx, p_idx, m] = get_pairing(blk)

#%% figures 
# FIGURE 1 – Energy sweep: LDOS comparison per probe site
method_styles = [
    dict(color=C_FIN, lw=1.5, ls="dashed",
         label=f"Finite RGF ({SL}+{SM}+{SR})", zorder=3, alpha=ALPHA),
    dict(color=C_INF, lw=2,   ls="solid",
         label="Sancho-López (∞ leads)",       zorder=2, alpha=ALPHA),
]

fig1, axes1 = plt.subplots(len(probe_sites), 1, figsize=(8, 6), sharex=True)
fig1.suptitle(
    f"LDOS vs Energy  (φ = π,  B = {B}, Δ = {Delta}, α = {alpha})",
    fontsize=14, fontweight="bold", y=1.01)

for s_idx, (s, lbl) in enumerate(zip(probe_sites, probe_labels)):
    ax = axes1[s_idx]
    for m in range(2):
        ax.plot(energies / Delta, ldos_e[s_idx, :, m], **method_styles[m])
    ax.axvline(-Delta, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(+Delta, color="gray", lw=0.8, ls="--", alpha=0.5, label="±Δ")
    ax.axvline(0,      color="k",    lw=0.5, ls=":",  alpha=0.3)
    ax.set_ylabel("LDOS  (arb.)", fontsize=11)
    ax.set_title(lbl, fontsize=11, loc="left", pad=4)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(bottom=0)

axes1[-1].set_xlabel(r"Energy  (E / $\Delta$)", fontsize=12)
fig1.tight_layout()

# FIGURE 2 – Phase sweep: LDOS and pairing at E = 0
fig2, axes2 = plt.subplots(len(probe_sites), 2, figsize=(10, 8), sharex=True)
fig2.suptitle(
    f"Phase sweep  (E = 0,  B = {B}, Δ = {Delta}, α = {alpha})",
    fontsize=14, fontweight="bold", y=1.01)

for s_idx, (s, lbl) in enumerate(zip(probe_sites, probe_labels)):
    ax_l = axes2[s_idx, 0]
    ax_r = axes2[s_idx, 1]
    for m in range(2):
        ax_l.plot(phases / np.pi, ldos_p[s_idx, :, m], **method_styles[m])
        ax_r.plot(phases / np.pi, pair_p[s_idx, :, m], **method_styles[m])
    ax_l.set_ylabel("LDOS  (E=0)", fontsize=11)
    ax_r.set_ylabel("|G₀₃|  anomalous", fontsize=11)
    ax_l.set_title(lbl, fontsize=11, loc="left", pad=4)
    for ax in (ax_l, ax_r):
        ax.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

axes2[-1, 0].set_xlabel("φ / π", fontsize=12)
axes2[-1, 1].set_xlabel("φ / π", fontsize=12)
fig2.tight_layout()

# FIGURE 3 – Method difference map |LDOS_inf - LDOS_fin|
err_inf = np.abs(ldos_all_inf - ldos_all_fin)
vmax = err_inf.max() * 1.05
norm = Normalize(vmin=0, vmax=vmax)

fig3, ax3 = plt.subplots(figsize=(8, 6))
fig3.suptitle("Absolute error: |LDOS(Sancho-López) - LDOS(Finite RGF)|  (all N-region sites)",
              fontsize=12, fontweight="bold")
im = ax3.imshow(err_inf, aspect="auto", origin="lower",
                extent=[energies[0], energies[-1], 0, SM], #type: ignore
                cmap="hot_r", norm=norm)
ax3.axvline(-Delta, color="cyan",  lw=1,   ls="--", alpha=0.7, label="±Δ")
ax3.axvline(+Delta, color="cyan",  lw=1,   ls="--", alpha=0.7)
ax3.axvline(0,      color="white", lw=0.5, ls=":",  alpha=0.5)
ax3.set_xlabel("Energy", fontsize=11)
ax3.set_ylabel("Site index (N region)", fontsize=11)
ax3.legend(fontsize=9)
ax3.set_yticks([0, (SM // 2) / 2, SM // 2, SM])
fig3.colorbar(im, ax=ax3, label="|ΔLDOS|")
fig3.tight_layout()

# FIGURE 4 – Spatial LDOS map (site × energy, finite RGF ground truth)
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig4.suptitle(f"Spatial LDOS map  (φ=π, B={B})", fontsize=13, fontweight="bold")

for ax, data, title in zip(axes4,
                            [ldos_all_fin, ldos_all_inf],
                            [f"Finite RGF ({SL}+{SM}+{SR})", "Sancho-López (∞ leads)"]):
    im4 = ax.imshow(data, aspect="auto", origin="lower",
                    extent=[energies[0], energies[-1], 0, SM], cmap="inferno")
    ax.axvline(-Delta, color="cyan",  lw=1,   ls="--", alpha=0.7, label="±Δ")
    ax.axvline(+Delta, color="cyan",  lw=1,   ls="--", alpha=0.7)
    ax.axvline(0,      color="white", lw=0.8, ls=":",  alpha=0.6, label="E=0")
    ax.set_xlabel("Energy", fontsize=12)
    ax.set_ylabel("Site index (N region)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig4.colorbar(im4, ax=ax, label="LDOS  (arb.)")
fig4.tight_layout()

# FIGURE 5 – |G_ij| matrix structure at E≈0, phi=pi
e0_idx = np.argmin(np.abs(energies))
s_fig5  = 0   # 0 - site 1;     1 - centre probe (SM // 2);    2 - end of the probe (SM)
fig5, axes5 = plt.subplots(1, 2, figsize=(9, 4.5))
fig5.suptitle(f"|G_ij| at E≈0, site={probe_labels[s_fig5]}  (φ=π)",
              fontsize=12, fontweight="bold")

nambu_labels = [r"$\uparrow$", r"$\downarrow$", r"$\downarrow^\dagger$", r"$-\uparrow^\dagger$"]

for m, (ax, ttl) in enumerate(zip(axes5, ["Finite RGF", "Sancho-López"])):
    mat = np.abs(Gblk_e[s_fig5, e0_idx, m])
    im5 = ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set_title(ttl, fontsize=11)
    
    # Set the basis labels on both axes
    ax.set_xticks(range(DOF))
    ax.set_xticklabels(nambu_labels, fontsize=11)
    ax.set_yticks(range(DOF))
    ax.set_yticklabels(nambu_labels, fontsize=11)
    
    # Overlay text values inside the matrix blocks
    for i in range(DOF):
        for j in range(DOF):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if mat[i,j] > mat.max()*0.5 else "black")
                    
    fig5.colorbar(im5, ax=ax, shrink=0.75)

fig5.tight_layout()
plt.show()
#%% Fig 7 ldos fin vs inf computation
print("══ Running 2D Phase×Energy sweep (Finite RGF) ══")
# Parallel execution

probe = probe_sites[0]

results = Parallel(n_jobs=-2)(
    delayed(compute_energy_slice)(e_idx, E)
    for e_idx, E in enumerate(energies)
)

ldos_2d = np.zeros((N_PHI, N_E))
for e_idx, row in results:
    ldos_2d[:, e_idx] = row

results_fin = Parallel(n_jobs=-2)(
    delayed(compute_energy_slice_fin)(e_idx, E)
    for e_idx, E in enumerate(energies)
)

# Assemble
ldos_2d_fin = np.zeros((N_PHI, N_E))
for e_idx, row in results_fin:
    ldos_2d_fin[:, e_idx] = row
fig7, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, data, title in zip(
    axes,
    [ldos_2d_fin, ldos_2d],
    ["Finite RGF", "Sancho–López (∞ leads)"]
):
    pcm = ax.pcolormesh(
        phases / np.pi,
        energies / Delta,
        data.T,
        #np.clip(data.T, 0, 10),
        shading='auto',
        #levels=50,
        cmap='inferno'
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\phi / \pi$")
axes[0].set_ylabel(r"$ E / \Delta$")
fig7.suptitle(r'LDOS(E, $\phi$)')
fig7.colorbar(pcm, ax=axes, label="LDOS")
plt.show()


#%% Spatial Wavefunction at E=0 from Infinite Leads
print("\n Extracting zero-energy wavefunction from infinite leads")

E_zero = 0.0
g_L, g_R = get_surface_gfs(E_zero, phi_fixed, onsite_sc, SYMMETRIC)

G_inf, *_ = myf.get_rgf_sns(H_mid_slices, V, g_L, g_R, E_zero, eta=1e-4, return_full=False)

psi = np.array([get_ldos(G_inf[s]) for s in range(SM)])
rho = np.abs(np.sqrt(psi*np.conjugate(psi)))
plt.figure(figsize=(10, 4))
plt.plot(rho, 'o-', color=C_INF, markersize=4, label='Infinite Leads (RGF)')
plt.ylabel(r'$|\psi(x)|$')
plt.xlabel('Middle site index')
plt.title(r'Spatial propability amplitude')
plt.legend(loc=8)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

#%% Fig. 8 spatial density of eigenvectors 
from scipy.linalg import eigh
H_full_slices, V = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC) # Get full SNS Hamiltonian

H_full = np.zeros((DOF * (SL + SM + SR), DOF * (SL + SM + SR)), dtype=np.complex128)
for i, H_i in enumerate(H_full_slices):
    H_full[i*DOF:(i+1)*DOF, i*DOF:(i+1)*DOF] = H_i
for i in range(len(H_full_slices) - 1):
    H_full[i*DOF:(i+1)*DOF, (i+1)*DOF:(i+2)*DOF] = V
    H_full[(i+1)*DOF:(i+2)*DOF, i*DOF:(i+1)*DOF] = V.conj().T

evals, evecs = eigh(H_full)

# Find E≈0
e0_idx = np.argmin(np.abs(evals))
print(f"Eigenvalue closest to E=0: {evals[e0_idx]:.6e}")

# Extract spatial wavefunction (trace over Nambu DoF)
psi = evecs[:, e0_idx]
psi_spatial = np.zeros(SL + SM + SR)
for site_idx in range(SL + SM + SR):
    psi_spatial[site_idx] = np.linalg.norm(psi[site_idx*DOF:(site_idx+1)*DOF])

plt.figure(figsize=(12, 4))
plt.plot(psi_spatial[SL:SM+SL+1], '-', markersize=3)
plt.ylabel(r'$|\psi(x)|$')
plt.xlabel('Site index')
plt.title(f'MZM wavefunction at E={evals[e0_idx]:.2e}')
plt.legend()
plt.show()
#%% Evolution of lowest energy eval vs. magnetic field 
#%% MZM splitting vs normal region length

from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np

SM_values = [50, 100, 150, 200, 300, 400, 500]

lowest_E = []

for SM_test in SM_values:

    # build finite SNS
    H_slices, V_test = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta,
        phi_fixed,
        SL, SR, SM_test,
        symmetric=SYMMETRIC
    )

    Ntot = len(H_slices)
    H = np.zeros((DOF*Ntot, DOF*Ntot), dtype=np.complex128)

    # onsite blocks
    for i, Hi in enumerate(H_slices):
        H[i*DOF:(i+1)*DOF,
          i*DOF:(i+1)*DOF] = Hi

    # hopping blocks
    for i in range(Ntot-1):
        H[i*DOF:(i+1)*DOF,
          (i+1)*DOF:(i+2)*DOF] = V_test

        H[(i+1)*DOF:(i+2)*DOF,
          i*DOF:(i+1)*DOF] = V_test.conj().T

    # only need eigenvalues near zero
    evals = eigh(H, eigvals_only=True)

    E0 = evals[np.argmin(np.abs(evals))]
    lowest_E.append(abs(E0))

    print(f"SM={SM_test:4d}   |E0|={abs(E0):.5e}")


# plot
plt.figure(figsize=(7,4))

plt.plot(
    SM_values,
    lowest_E,
    "o-"
)

plt.xlabel("Normal region length $S_M$")
plt.ylabel(r"$|E_0|$")
plt.title("Lowest BdG energy vs junction length")
plt.grid(True)

plt.show()
#%% SANITY CHECKS  

SEP  = "═" * 60
SEP2 = "─" * 60

def verdict(ok, tol=None):
    s = "✓ PASS" if ok else "✗ FAIL"
    if tol is not None:
        s += f"  (tol={tol:.0e})"
    return s

method_names = ["Finite RGF", "Sancho-López"]

print(f"\n{SEP}")
print("   SANITY CHECK REPORT")
print(f"   Gauge: {'symmetric ±φ/2' if SYMMETRIC else 'asymmetric full φ on right'}")
print(SEP)

# [1]. Particle-Hole Symmetry
print("\n[1] Particle-Hole Symmetry  LDOS(E) = LDOS(-E)")
print(SEP2)
e_sym = np.allclose(energies, -energies[::-1], atol=1e-10)
if not e_sym:
    print("  ⚠ Energy grid not symmetric — skipping")
else:
    tol_phs = 5e-3
    for s_idx, lbl in enumerate(probe_labels):
        for m, mname in enumerate(method_names):
            diff = np.max(np.abs(ldos_e[s_idx, :, m] - ldos_e[s_idx, ::-1, m]))
            print(f"  {lbl:22s} | {mname:16s} | max|Δ|={diff:.2e}  {verdict(diff < tol_phs, tol_phs)}")

# [2]. LDOS positivity
print(f"\n[2] LDOS Positivity  (should be >= 0 everywhere)")
print(SEP2)
tol_pos = -1e-5
for m, mname in enumerate(method_names):
    mn_e = ldos_e[:, :, m].min()
    mn_p = ldos_p[:, :, m].min()
    ok   = (mn_e >= tol_pos) and (mn_p >= tol_pos)
    print(f"  {mname:16s} | min(E-sweep)={mn_e:+.2e}, min(φ-sweep)={mn_p:+.2e}  {verdict(ok)}")

# [3]. Method agreement (sub-gap only — above-gap differs due to finite SC bandwidth)
print(f"\n[3] Method Agreement  |LDOS(SL) - LDOS(FinRGF)|  (sub-gap only: |E| < Δ)")
print(SEP2)
tol_agree = 5e-2
subgap = np.abs(energies) < Delta
diffs_sub = [np.max(np.abs(ldos_e[s_idx, subgap, 1] - ldos_e[s_idx, subgap, 0]))
             for s_idx in range(len(probe_sites))]
diffs_full = [np.max(np.abs(ldos_e[s_idx, :, 1] - ldos_e[s_idx, :, 0]))
              for s_idx in range(len(probe_sites))]
print(f"  Overall | sub-gap max={max(diffs_sub):.2e}  {verdict(max(diffs_sub) < tol_agree, tol_agree)}"
      f"  (full max={max(diffs_full):.2e})")
for s_idx, lbl in enumerate(probe_labels):
    print(f"    {lbl:22s}: sub-gap={diffs_sub[s_idx]:.2e}  full={diffs_full[s_idx]:.2e}")

# [4]. Sub-gap spectral weight
print(f"\n[4] Sub-gap spectral weight  (|E| < Δ={Delta})")
print(SEP2)
for m, mname in enumerate(method_names):
    fracs = []
    for s_idx in range(len(probe_sites)):
        tot = np.trapezoid(ldos_e[s_idx, :, m], energies)
        sub = np.trapezoid(ldos_e[s_idx, subgap, m], energies[subgap])
        fracs.append(sub / tot if tot > 1e-12 else 0)
    print(f"  {mname:16s} | " + ", ".join(f"{f:.3f}" for f in fracs))

# [5]. Zero-energy peak at phi=pi
print(f"\n[5] Zero-energy peak at φ=π  (topological: {'YES' if in_topo else 'NO'})")
print(SEP2)
e0_idx = np.argmin(np.abs(energies))
for s_idx, lbl in enumerate(probe_labels):
    for m, mname in enumerate(method_names):
        ld0  = ldos_e[s_idx, e0_idx, m]
        ldmx = ldos_e[s_idx, :, m].max()
        rat  = ld0 / ldmx if ldmx > 1e-12 else 0
        print(f"  {lbl:22s} | {mname:16s} | LDOS(0)/max={rat:.3f}  {verdict(rat > 0.4)}")

# [6]. Phase periodicity
print(f"\n[6] Phase Periodicity  (φ=0 ≈ φ=2π)")
print(SEP2)
tol_per = 1e-3
for s_idx, lbl in enumerate(probe_labels):
    for m, mname in enumerate(method_names):
        diff = abs(ldos_p[s_idx, 0, m] - ldos_p[s_idx, -1, m])
        print(f"  {lbl:22s} | {mname:16s} | Δ={diff:.2e}  {verdict(diff < tol_per, tol_per)}")

# [7]. Proximity effect
print(f"\n[7] Proximity effect  |G₀₃| edge > centre  (φ=π)")
print(SEP2)
phi_pi_idx = np.argmin(np.abs(phases - np.pi))
for m, mname in enumerate(method_names):
    edge  = pair_p[0, phi_pi_idx, m]
    mid   = pair_p[1, phi_pi_idx, m]
    ratio = mid / edge if edge > 1e-12 else float("nan")
    print(f"  {mname:16s} | edge={edge:.4f}, centre={mid:.4f}, ratio={ratio:.3f}  {verdict(ratio < 0.99)}")

# [8]. BdG integral consistency between methods
print(f"\n[8] BdG integral consistency  ∫LDOS dE  (FinRGF vs Sancho-López)")
print(SEP2)
for s_idx, lbl in enumerate(probe_labels):
    integrals = [np.trapezoid(ldos_e[s_idx, :, m], energies) for m in range(2)]
    print(f"  {lbl:22s} | " +
          " | ".join(f"{mn}: {iv:.4f}" for mn, iv in zip(method_names, integrals)))
    delta_methods = abs(integrals[0] - integrals[1])
    print(f"  {'':22s}   |FinRGF - SL|={delta_methods:.2e}  {verdict(delta_methods < 1e-2)}")

# [9]. Finite-size convergence (sub-gap only)
print(f"\n[9] Finite-size convergence  (sub-gap |E|<Δ, centre probe vs Sancho-López)")
print(f"    Note: above-gap errors are physical (finite vs continuous SC spectrum)")
print(SEP2)
tol_conv = 1e-1
sizes_to_test = [50, 100, 150, 300]

for sl_sr in sizes_to_test:
    H_test_slices, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta,
        phi_fixed, sl_sr, sl_sr, SM,
        symmetric=SYMMETRIC)

    ldos_fin_test = np.zeros(N_E)
    ldos_inf_test = np.zeros(N_E)

    for e_idx, E in enumerate(energies):
        G_fin_test, *_ = myf.get_rgf_finite_system(
            H_test_slices, V, E, eta=eta, return_full=False)
        ldos_fin_test[e_idx] = get_ldos(G_fin_test[sl_sr + probe_sites[1]])

        g_L, g_R = get_surface_gfs(E, phi_fixed, onsite_sc, SYMMETRIC, eta=eta)
        G_inf_test, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        ldos_inf_test[e_idx] = get_ldos(G_inf_test[probe_sites[1]])

    max_err_sub = np.max(np.abs(ldos_fin_test[subgap] - ldos_inf_test[subgap]))
    max_err_all = np.max(np.abs(ldos_fin_test - ldos_inf_test))
    print(f"  SL=SR={sl_sr:4d} | sub-gap max|Δ|={max_err_sub:.3e}  "
          f"{verdict(max_err_sub < tol_conv, tol_conv)}  (full={max_err_all:.3e})")


# %%
