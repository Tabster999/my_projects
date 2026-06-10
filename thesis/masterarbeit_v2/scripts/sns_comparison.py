"""
SNS Junction Green's Function Comparison
=========================================
Compares three methods for computing the local Green's function of a 1D
Kitaev-style SNS junction in the BdG Nambu basis (c↑, c↓, c↓†, c↑†):

  Method 1 – Direct matrix inversion  (ground truth)
  Method 2 – Recursive GF, finite leads
  Method 3 – Recursive GF, Sancho-López semi-infinite leads

Checks performed
----------------
  A. Particle-hole symmetry  LDOS(E) = LDOS(-E)
  B. LDOS positivity
  C. Method agreement vs inversion (max |Δ LDOS|)
  D. Sub-gap spectral-weight fraction
  E. Zero-energy peak present at φ = π  (topological phase)
  F. 2π phase periodicity
  G. Proximity effect: |G₀₃| decays edge → centre

Usage
-----
  python sns_comparison.py

The script imports `my_functions` from the parent directory exactly as your
existing code does.  Adjust `sys.path` if needed.
"""

#%% ── Imports ──────────────────────────────────────────────────────────────────
from cmath import phase
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.linalg import inv
from joblib import Parallel, delayed

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
module_root = parent_dir.parent
sys.path.insert(0, str(module_root))

import modules as myf

# ── Colour palette (colour-blind friendly) ────────────────────────────────────
C_INV  = "#2166ac"   # blue  – inversion
C_FIN  = "#d6604d"   # red   – finite RGF
C_INF  = "#4dac26"   # green – infinite leads
ALPHA  = 0.85

#%% ── Helpers # Define helper functions

def phase_matrix(phi):
    """U(1) gauge rotation in Nambu basis (up,dn,dn†,up†)."""
    return np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)]).astype(np.complex128)

def get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def get_pairing(G_block):
    """Anomalous amplitude |G[0,3]| = |⟨c↑ c↑†⟩| proxy."""
    return np.abs(G_block[0, 3])


def get_surface_gfs(E, phi, symmetric=True):
    """
    Surface GFs with SC phase applied as Nambu gauge rotation.
    
    Convention (matches build_sns_junction_sliced):
      U(theta) @ H(Delta) @ U(theta)† = H(Delta * exp(-i*theta))
    
    Symmetric gauge:
      Delta_L = Delta*exp(-i*phi/2)  →  UL = phase_matrix(+phi/2)
      Delta_R = Delta*exp(+i*phi/2)  →  UR = phase_matrix(-phi/2)
    
    Asymmetric gauge:
      Delta_L = Delta (real)          →  UL = identity
      Delta_R = Delta*exp(+i*phi)     →  UR = phase_matrix(-phi)
    """
    onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)
    if symmetric:
        UL = phase_matrix(-phi / 2.0)
        UR = phase_matrix(+phi / 2.0)
    else:
        UL = np.eye(4, dtype=np.complex128)
        UR = phase_matrix(-phi)

    g_L, _ = myf.get_surface_gf(E, onsite_sc, V.conj().T, eta=eta)
    g_R, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)

    return UL @ g_L @ UL.conj().T, UR @ g_R @ UR.conj().T

#%% ── Parameters 
# Define parameters
SYMMETRIC = True    # True  → ±φ/2 on left/right
                    # False → full φ on right only

SL, SM, SR = 100, 50, 100        # sites: left SC | normal | right SC
DOF        = 4
Delta      = 0.1
mu_sc      = 0.1
mu_n       = 0.000
t          = 1.0
alpha      = 0.15
B          = 0.3     
eta        = 1e-3
phi_fixed  = np.pi

N_E   = 61                    # energy points
N_PHI = 61                    # phase points
energies = np.linspace(-.051, .051, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

probe_sites = [1, SM // 2, SM - 2]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# ── Fixed objects ───────────────────────────────────────────────────────
V      = myf.t_matrix_y(t, alpha)
V_dag  = V.conj().T
onsite_sc   = myf.onsite_matrix(t, mu_sc, B, Delta)
H_mid_slices, _ = myf.build_middle_region(t, mu_n, alpha, B, SM)

H_full_slices, _ = myf.build_sns_junction_sliced(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, SL, SR, SM, symmetric=SYMMETRIC)
H_full_matrix = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, SL, SR, SM, 4, symmetric=SYMMETRIC)
dim_full = H_full_matrix.shape[0]
I_full = np.eye(dim_full, dtype=np.complex128)

# ── Particle-hole symmetry check on Hamiltonians ─────────────────────────────
C = np.fliplr(np.eye(DOF))   # antidiag identity — PH matrix

def ph_check(name, M):
    ok = np.allclose(M, -C @ M.conj() @ C, atol=1e-12)
    print(f"  PH symmetry [{name}]: {'✓ PASS' if ok else '✗ FAIL'}")

print("\n══ Hamiltonian PH checks ══")
ph_check("onsite_sc", onsite_sc)
ph_check("V hopping", V)
ph_check("onsite_N",  myf.onsite_matrix(t, mu_n, B, 0.0))

# ── Topological phase check ───────────────────────────────────────────────────
topo_threshold = np.sqrt(Delta**2 + mu_sc**2)
in_topo = B > topo_threshold
print(f"\n══ Topological phase ══")
print(f"  B = {B:.3f},  √(Δ²+μ²) = {topo_threshold:.4f}")
print(f"  In topological phase: {'YES ✓' if in_topo else 'NO  ✗'}")
print(f"\n══ Gauge convention: {'symmetric ±φ/2' if SYMMETRIC else 'asymmetric, full φ on right'} ══")


#%% ENERGY SWEEP  (φ = pi)
# ═════════════════════════════════════════════════════════════════════════════
# ENERGY SWEEP  (φ fixed)
# ═════════════════════════════════════════════════════════════════════════════
print("\n══ Running energy sweep ══")

# shape: (probe, energy, method)  methods: 0=inv 1=fin 2=inf
ldos_e  = np.zeros((len(probe_sites), N_E, 3))
pair_e  = np.zeros((len(probe_sites), N_E, 3))
Gblk_e  = np.zeros((len(probe_sites), N_E, 3, DOF, DOF), dtype=np.complex128)
ldos_map     = np.zeros((SM, N_E))
ldos_all_fin = np.zeros((SM, N_E))
ldos_all_inf = np.zeros((SM, N_E))

for e_idx, E in enumerate(energies):
    z = E + 1j * eta

    G_inv = inv(z * I_full - H_full_matrix)
    G_fin, *_, = myf.get_rgf_finite_system(H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC)
    G_inf, *_ = myf.get_rgf_sns(H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        idx_start = (SL + s) * 4
        blk_inv = G_inv[idx_start : idx_start + 4, idx_start : idx_start + 4]
        
        blk_fin = G_fin[SL + s]
        
        blk_inf = G_inf[s]

        # Store results for all 3 methods
        methods = [blk_inv, blk_fin, blk_inf]
        for m, blk in enumerate(methods):
            ldos_e[s_idx, e_idx, m] = get_ldos(blk)
            Gblk_e[s_idx, e_idx, m] = blk  # Needed for Figure 5
# %% PHASE SWEEP  (E = 0)
# PHASE SWEEP  (E = 0)
print("══ Running phase sweep  ══")

energy_fixed = 0 
ldos_p = np.zeros((len(probe_sites), N_PHI, 3))
pair_p = np.zeros((len(probe_sites), N_PHI, 3))


for p_idx, phi in enumerate(phases):
    H_full_p, _ = myf.build_sns_junction_sliced(t, mu_sc, mu_n, alpha, B, Delta, phi, SL, SR, SM, symmetric=SYMMETRIC)
    H_mat_p = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi, SL, SR, SM, DOF, symmetric=SYMMETRIC)
    
    I_p = np.eye(H_mat_p.shape[0], dtype=np.complex128)
    z   = energy_fixed + 1j * eta

    G_inv_p = inv(z * I_p - H_mat_p)
    G_fin_p, *_ = myf.get_rgf_finite_system(
        H_full_p, V, energy_fixed, eta=eta, return_full=False)

    g_L_p, g_R_p = get_surface_gfs(energy_fixed, phi, SYMMETRIC)
    G_inf_p, *_ = myf.get_rgf_sns(H_mid_slices, V, g_L_p, g_R_p, energy_fixed, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
            # Extract blocks for all three methods
            idx_start = (SL + s) * DOF
            blk_inv = G_inv_p[idx_start : idx_start + DOF, idx_start : idx_start + DOF]
            blk_fin = G_fin_p[SL + s]
            blk_inf = G_inf_p[s]
            
            methods = [blk_inv, blk_fin, blk_inf]
            for m, blk in enumerate(methods):
                ldos_p[s_idx, p_idx, m] = get_ldos(blk)
                pair_p[s_idx, p_idx, m] = get_pairing(blk)


#%% FIGURES 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Energy sweep: LDOS comparison per probe site
# ═════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(len(probe_sites), 1,
                            figsize=(8, 6),
                            sharex=True)
fig1.suptitle(
    f"LDOS vs Energy  (φ = π,  B = {B}, Δ = {Delta}, α = {alpha})",
    fontsize=14, fontweight="bold", y=1.01)

method_styles = [
    dict(color=C_INV, marker="o", ms=3, ls="none",  label="Direct inversion", zorder=4),
    dict(color=C_FIN, lw=1.5, ls="dashed", label=f"Finite RGF ({SL}+{SM}+{SR})", zorder=3, alpha=ALPHA),
    dict(color=C_INF, lw=2, ls="solid",  label="Sancho-López (∞ leads)",  zorder=2, alpha=ALPHA),
]

for s_idx, (s, lbl) in enumerate(zip(probe_sites, probe_labels)):
    ax = axes1[s_idx]
    for m in range(3):
        ax.plot(energies, ldos_e[s_idx, :, m], **method_styles[m])
    ax.axvline(-Delta, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(+Delta, color="gray", lw=0.8, ls="--", alpha=0.5, label="±Δ")
    ax.axvline(0,      color="k",    lw=0.5, ls=":",  alpha=0.3)
    ax.set_ylabel("LDOS  (arb.)", fontsize=11)
    ax.set_title(lbl, fontsize=11, loc="left", pad=4)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(energies[0], energies[-1])
    ax.set_ylim(bottom=0)

axes1[-1].set_xlabel("Energy  (E / t)", fontsize=12)
fig1.tight_layout()
#fig1.savefig("fig1_energy_sweep.png", dpi=150, bbox_inches="tight")
print("  saved fig1_energy_sweep.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Phase sweep: LDOS and pairing at E = 0
# ═════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(len(probe_sites), 2,
                            figsize=(10, 8),
                            sharex=True)
fig2.suptitle(
    f"Phase sweep  (E = 0,  B = {B}, Δ = {Delta}, α = {alpha})",
    fontsize=14, fontweight="bold", y=1.01)

for s_idx, (s, lbl) in enumerate(zip(probe_sites, probe_labels)):
    ax_l = axes2[s_idx, 0]
    ax_r = axes2[s_idx, 1]

    for m in range(3):
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
#fig2.savefig("fig2_phase_sweep.png", dpi=150, bbox_inches="tight")
print("  saved fig2_phase_sweep.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Method difference maps |LDOS_method - LDOS_inv|
# ═════════════════════════════════════════════════════════════════════════════
err_fin = np.abs(ldos_all_fin - ldos_map)
err_inf = np.abs(ldos_all_inf - ldos_map)
vmax = max(err_fin.max(), err_inf.max()) * 1.05
norm = Normalize(vmin=0, vmax=vmax)
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
fig3.suptitle("Absolute error vs direct inversion  (all N-region sites)",
              fontsize=13, fontweight="bold")
for ax, err, title in zip(axes3, [err_fin, err_inf],
                           ["Finite RGF", "Sancho-López (∞ leads)"]):
    im = ax.imshow(err, aspect="auto", origin="lower",
                   extent=[energies[0], energies[-1], 0, SM],
                   cmap="hot_r", norm=norm)
    ax.axvline(-Delta, color="cyan",  lw=1,   ls="--", alpha=0.7)
    ax.axvline(+Delta, color="cyan",  lw=1,   ls="--", alpha=0.7)
    ax.axvline(0,      color="white", lw=0.5, ls=":",  alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Energy  (E / t)", fontsize=11)
    ax.set_ylabel("Site index (N region)", fontsize=11)
fig3.colorbar(ScalarMappable(norm=norm, cmap="hot_r"),
              ax=axes3, label="|ΔLDOS|", shrink=0.8)
#fig3.savefig("fig3_error_maps.png", dpi=150, bbox_inches="tight")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Spatial LDOS map  (site × energy, inversion ground truth)
# ═════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(10, 8))
im4 = ax4.imshow(ldos_map, aspect="auto", origin="lower", extent=[energies[0], energies[-1], 0, SM], cmap="inferno") # type: ignore
ax4.axvline(-Delta, color="cyan",  lw=1,   ls="--", alpha=0.7, label="±Δ")
ax4.axvline(+Delta, color="cyan",  lw=1,   ls="--", alpha=0.7)
ax4.axvline(0,      color="white", lw=0.8, ls=":",  alpha=0.6, label="E=0")
ax4.set_xlabel("Energy  (E / t)", fontsize=12)
ax4.set_ylabel("Site index (N region)", fontsize=12)
ax4.set_title(f"Spatial LDOS map — direct inversion  (φ=π, B={B})", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10, loc="upper right")
fig4.colorbar(im4, ax=ax4, label="LDOS  (arb.)")
fig4.tight_layout()
#fig4.savefig("fig4_spatial_ldos.png", dpi=150, bbox_inches="tight")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – G matrix element magnitudes at one probe site, one energy
#             (visual check that full 4×4 structure is correct)
# ═════════════════════════════════════════════════════════════════════════════
e0_idx = np.argmin(np.abs(energies))
s_mid  = 1
fig5, axes5 = plt.subplots(1, 3, figsize=(10, 8))
fig5.suptitle(f"|G_ij| at E≈0, site={probe_labels[s_mid]}  (φ=π)",
              fontsize=13, fontweight="bold")
for m, (ax, ttl) in enumerate(zip(axes5, ["Direct inversion", "Finite RGF", "Sancho-López"])):
    mat = np.abs(Gblk_e[s_mid, e0_idx, m])
    im5 = ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set_title(ttl, fontsize=11)
    ax.set_xticks(range(DOF)); ax.set_xticklabels([r"$↑$", r"↓", r"$↓^†$", r"$↑^†$"], fontsize=10)
    ax.set_yticks(range(DOF)); ax.set_yticklabels([r"↑", r"↓", r"$↓^†$", r"$↑^†$"], fontsize=10)
    for i in range(DOF):
        for j in range(DOF):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i,j] > mat.max()*0.5 else "black")
    fig5.colorbar(im5, ax=ax, shrink=0.75)
fig5.tight_layout()
#fig5.savefig("fig5_Gmatrix.png", dpi=150, bbox_inches="tight")
print("  saved all figures")

#%% SANITY CHECKS 
# ═════════════════════════════════════════════════════════════════════════════
SEP  = "═" * 55
SEP2 = "─" * 55

def verdict(ok, tol=None):
    s = "✓ PASS" if ok else "✗ FAIL"
    if tol is not None: s += f"  (tol={tol:.0e})"
    return s

method_names = ["Inversion", "Finite RGF", "Sancho-López"]

print(f"\n{SEP}")
print("   SANITY CHECK REPORT")
print(f"   Gauge: {'symmetric ±φ/2' if SYMMETRIC else 'asymmetric full φ on right'}")
print(SEP)

# A. Particle-Hole Symmetry
print("\n[A] Particle-Hole Symmetry  LDOS(E) = LDOS(-E)")
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

# B. LDOS positivity
print(f"\n[B] LDOS Positivity")
print(SEP2)
tol_pos = -1e-5
for m, mname in enumerate(method_names):
    mn_e = ldos_e[:, :, m].min()
    mn_p = ldos_p[:, :, m].min()
    ok   = (mn_e >= tol_pos) and (mn_p >= tol_pos)
    print(f"  {mname:16s} | min(E-sweep)={mn_e:+.2e}, min(φ-sweep)={mn_p:+.2e}  {verdict(ok)}")

# C. Method agreement
print(f"\n[C] Method Agreement  |LDOS_method − LDOS_inv|")
print(SEP2)
tol_fin = 5e-2
tol_inf = 1e-1
for m, (mname, tol) in enumerate(zip(method_names[1:], [tol_fin, tol_inf]), 1):
    diffs = [np.max(np.abs(ldos_e[s_idx, :, m] - ldos_e[s_idx, :, 0]))
             for s_idx in range(len(probe_sites))]
    print(f"  {mname:16s} | max={max(diffs):.2e}  {verdict(max(diffs) < tol, tol)}")
    for s_idx, lbl in enumerate(probe_labels):
        print(f"    {lbl:22s}: {diffs[s_idx]:.2e}")

# D. Sub-gap spectral weight
print(f"\n[D] Sub-gap spectral weight  (|E| < Δ={Delta})")
print(SEP2)
in_gap = np.abs(energies) < Delta
for m, mname in enumerate(method_names):
    fracs = []
    for s_idx in range(len(probe_sites)):
        tot = np.trapezoid(ldos_e[s_idx, :, m], energies)
        sub = np.trapezoid(ldos_e[s_idx, in_gap, m], energies[in_gap])
        fracs.append(sub / tot if tot > 1e-12 else 0)
    print(f"  {mname:16s} | " + ", ".join(f"{f:.3f}" for f in fracs))

# E. Zero-energy peak at φ=π
print(f"\n[E] Zero-energy peak at φ=π  (topological: {'YES' if in_topo else 'NO'})")
print(SEP2)
e0_idx = np.argmin(np.abs(energies))
for s_idx, lbl in enumerate(probe_labels):
    for m, mname in enumerate(method_names):
        ld0  = ldos_e[s_idx, e0_idx, m]
        ldmx = ldos_e[s_idx, :, m].max()
        rat  = ld0 / ldmx if ldmx > 1e-12 else 0
        print(f"  {lbl:22s} | {mname:16s} | LDOS(0)/max={rat:.3f}  {verdict(rat > 0.4)}")

# F. Phase periodicity
print(f"\n[F] Phase Periodicity  (φ=0 ≈ φ=2π)")
print(SEP2)
tol_per = 1e-3
for s_idx, lbl in enumerate(probe_labels):
    for m, mname in enumerate(method_names):
        diff = abs(ldos_p[s_idx, 0, m] - ldos_p[s_idx, -1, m])
        print(f"  {lbl:22s} | {mname:16s} | Δ={diff:.2e}  {verdict(diff < tol_per, tol_per)}")

# G. Proximity effect
print(f"\n[G] Proximity effect  |G₀₃| edge > centre  (φ=π)")
print(SEP2)
phi_pi = np.argmin(np.abs(phases - np.pi))
for m, mname in enumerate(method_names):
    edge  = pair_p[0, phi_pi, m]
    mid   = pair_p[1, phi_pi, m]
    ratio = mid / edge if edge > 1e-12 else float("nan")
    print(f"  {mname:16s} | edge={edge:.4f}, centre={mid:.4f}, ratio={ratio:.3f}  {verdict(ratio < 0.99)}")

# H. BdG sum rule
print(f"\n[H] BdG sum rule  ∫LDOS dE  (truncated window, Inv vs FinRGF must match)")
print(SEP2)
for s_idx, lbl in enumerate(probe_labels):
    integrals = [np.trapezoid(ldos_e[s_idx, :, m], energies) for m in range(3)]
    print(f"  {lbl:22s} | " +
          " | ".join(f"{mn}: {iv:.4f}" for mn, iv in zip(method_names, integrals)))
    delta_inv_fin = abs(integrals[0] - integrals[1])
    print(f"  {'':22s}   |Inv-FinRGF|={delta_inv_fin:.2e}  {verdict(delta_inv_fin < 1e-3)}")

# I. Finite-size convergence
print(f"\n[I] Finite-size convergence  (Sancho-López vs inversion, centre probe)")
print(SEP2)

sizes_to_test = [100, 200, 400, 500]

for sl_sr in sizes_to_test:
    # Finite system via RGF (not inversion — much faster)
    H_test_slices, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta,
        phi_fixed, sl_sr, sl_sr, SM,
        symmetric=SYMMETRIC)

    ldos_inv_test = np.zeros(N_E)
    ldos_inf_test = np.zeros(N_E)

    for e_idx, E in enumerate(energies):
        # Finite: use the full SNS slices with finite leads
        G_fin_test, *_ = myf.get_rgf_finite_system(
            H_test_slices, V, E, eta=eta, return_full=False)
        ldos_inv_test[e_idx] = get_ldos(G_fin_test[sl_sr + probe_sites[1]])

        # Infinite: always uses H_mid_slices (N region only)
        g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC)
        G_inf_test, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        ldos_inf_test[e_idx] = get_ldos(G_inf_test[probe_sites[1]])

    max_err = np.max(np.abs(ldos_inv_test - ldos_inf_test))
    print(f"  SL=SR={sl_sr:4d} | max|Δ LDOS|={max_err:.3e}  {verdict(max_err < tol_inf, tol_inf)}")
print(f"\n{SEP}")
print("  Done.")
print(SEP)

plt.show()
# %% Compare LDOS at centre of N for symmetric vs asymmetric at same sl_sr
E_test = 0.0
z_test = E_test + 1j * eta

for sl_sr in [50, 100, 200, 700]:
    for sym in [True, False]:
        H_test, _ = myf.build_sns_junction_sliced(
            t, mu_sc, mu_n, alpha, B, Delta,
            phi_fixed, sl_sr, sl_sr, SM, symmetric=sym)
        G_fin, *_ = myf.get_rgf_finite_system(
            H_test, V, E_test, eta=eta, return_full=False)
        ldos_fin = get_ldos(G_fin[sl_sr + probe_sites[1]])

        g_L, g_R = get_surface_gfs(E_test, phi_fixed, SYMMETRIC)
        G_inf, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E_test, eta=eta, return_full=False)
        ldos_inf = get_ldos(G_inf[probe_sites[1]])

        print(f"  sl_sr={sl_sr}, sym={sym} | fin={ldos_fin:.4f}, inf={ldos_inf:.4f}, diff={abs(ldos_fin-ldos_inf):.2e}")

# %% I. Finite-size convergence (sub-gap only)

print(f"\n[I] Finite-size convergence  (sub-gap |E| < Δ, centre probe)")
print(f"    Note: above-gap errors are physical (finite vs continuous SC spectrum)")
print(SEP2)

subgap = np.abs(energies) < Delta
sizes_to_test = [50, 100, 150, 200, 300]

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

        g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC)
        G_inf_test, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        ldos_inf_test[e_idx] = get_ldos(G_inf_test[probe_sites[1]])

    max_err_subgap = np.max(np.abs(ldos_fin_test[subgap] - ldos_inf_test[subgap]))
    print(f"  SL=SR={sl_sr:4d} | sub-gap max|Δ LDOS|={max_err_subgap:.3e}  "
          f"{verdict(max_err_subgap < tol_inf, tol_inf)}")

# C. Method agreement
print(f"\n[C] Method Agreement  |LDOS_method − LDOS_inv|")
print(SEP2)
tol_fin = 5e-2
tol_inf_subgap = 1e-1
subgap = np.abs(energies) < Delta

for m, (mname, tol) in enumerate(zip(method_names[1:], [tol_fin, tol_inf_subgap]), 1):
    diffs_full   = [np.max(np.abs(ldos_e[s_idx, :, m] - ldos_e[s_idx, :, 0]))
                    for s_idx in range(len(probe_sites))]
    diffs_subgap = [np.max(np.abs(ldos_e[s_idx, subgap, m] - ldos_e[s_idx, subgap, 0]))
                    for s_idx in range(len(probe_sites))]
    print(f"  {mname:16s} | sub-gap max={max(diffs_subgap):.2e}  {verdict(max(diffs_subgap) < tol, tol)}"
          f"  (full max={max(diffs_full):.2e})")
    for s_idx, lbl in enumerate(probe_labels):
        print(f"    {lbl:22s}: sub-gap={diffs_subgap[s_idx]:.2e}  full={diffs_full[s_idx]:.2e}")

# %% Contourplot LDOS vs energy and phase computation
# --- New 2D Arrays for Contour ---
ldos_2d = np.zeros((N_PHI, N_E))

print("══ Running 2D Sweep (Phase vs Energy) ══")

for p_idx, phi in enumerate(phases):    
    for e_idx, E in enumerate(energies):
        # Using Sancho-Lopez (Method 3) as it is fastest for 2D sweeps
        g_L, g_R = get_surface_gfs(E, phi, SYMMETRIC)
        G_inf, *_ = myf.get_rgf_sns(H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        
        # Store LDOS for the middle probe site
        ldos_2d[p_idx, e_idx] = get_ldos(G_inf[probe_sites[1]])
#%% plotting of the contour plot computed above
fig, ax = plt.subplots(figsize=(8, 8))
pcm = ax.pcolormesh(
    phases / np.pi,
    energies,
    ldos_2d.T,
    shading='auto',
    cmap='magma'
)
ax.set_xlabel(r"$\phi / \pi$", fontsize=12)
ax.set_ylabel("Energy (E / t)", fontsize=12)
ax.axhline(-Delta, color="white", lw=1, ls="--", alpha=0.6)
ax.axhline(+Delta, color="white", lw=1, ls="--", alpha=0.6)
ax.axhline(0,      color="white", lw=0.5, ls=":", alpha=0.4)
ax.axvline(1,      color="gray",  lw=0.8, ls="--", alpha=0.5)
fig.colorbar(pcm, ax=ax, label="LDOS")
plt.tight_layout()
plt.show()
# ABS should be symmetric: ldos_2d[phi] ≈ ldos_2d[2π - phi]
sym_err = np.max(np.abs(ldos_2d - ldos_2d[::-1, :]))
print(f"Phase-symmetry error: {sym_err:.3e}")
# %% quick check with small system 
# quick check with small system
SL_t, SM_t, SR_t = 30, 10, 30
phi_t  = np.pi
E_t    = 0.0
eta_t  = 1e-3

H_slices_t, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi_t, SL_t, SR_t, SM_t, symmetric=True)

N_t  = len(H_slices_t)
dim  = N_t * 4

# direct inversion ground truth
H_mat = np.zeros((dim, dim), dtype=np.complex128)
for i, h in enumerate(H_slices_t):
    H_mat[i*4:(i+1)*4, i*4:(i+1)*4] = h
    if i < N_t-1:
        H_mat[i*4:(i+1)*4,     (i+1)*4:(i+2)*4] = V
        H_mat[(i+1)*4:(i+2)*4, i*4:(i+1)*4]     = V.conj().T
G_dir = np.linalg.inv((E_t + 1j*eta_t)*np.eye(dim) - H_mat)

# RGF
G_diag, GL, GR, G_blocks, _ = myf.get_rgf_finite_system(
    H_slices_t, V, E_t, eta=eta_t, return_full=True, return_dense=False)

print("=== Block comparison: RGF vs direct ===")
if G_blocks is not None:
    print(f"  N_t={N_t}, dim={dim}, G_blocks.shape={G_blocks.shape}, G_dir.shape={G_dir.shape}")
else:
    print(f"  N_t={N_t}, dim={dim}, G_blocks=None, G_dir.shape={G_dir.shape}")
max_err = 0
for i in range(N_t):
    for j in range(N_t):
        G_rgf_ij  = G_blocks[i, j, :, :] #type: ignore
        G_dir_ij  = G_dir[i*4:(i+1)*4, j*4:(j+1)*4]
        err = np.max(np.abs(G_rgf_ij - G_dir_ij))
        max_err = max(max_err, err)
        if err > 1e-10:
            print(f"  FAIL [{i},{j}]: err={err:.2e}")

if max_err < 1e-10:
    print(f"  All blocks match — max err={max_err:.2e}")
else:
    print(f"  Max error across all blocks: {max_err:.2e}")

# Dense matrix comparison
print("\n=== Dense matrix comparison: RGF vs direct ===")
G_diag_dense, GL_dense, GR_dense, G_blocks_dense, G_dense = myf.get_rgf_finite_system(
    H_slices_t, V, E_t, eta=eta_t, return_full=True, return_dense=True)

if G_dense is not None:
    max_abs_err = np.max(np.abs(G_dense - G_dir))
    print(f"  Max absolute error: {max_abs_err:.2e}")

    # Element-wise comparison with np.isclose
    close_rtol = 1e-9
    close_atol = 1e-11
    is_close = np.isclose(G_dense, G_dir, rtol=close_rtol, atol=close_atol)
    frac_close = np.sum(is_close) / is_close.size
    print(f"  Fraction of elements close (rtol={close_rtol}, atol={close_atol}): {frac_close:.4f}")

    if frac_close == 1.0:
        print(f"  ✓ All matrix elements match within tolerance")
    else:
        mismatches = np.sum(~is_close)
        print(f"  ✗ {mismatches} elements exceed tolerance")
        # Find and report largest mismatches
        err_matrix = np.abs(G_dense - G_dir)
        top_errors_idx = np.argsort(err_matrix.flatten())[-5:][::-1]
        for idx in top_errors_idx:
            i, j = np.unravel_index(idx, err_matrix.shape)
            print(f"    [{i},{j}]: |err|={err_matrix[i,j]:.2e}, dense={G_dense[i,j]:.4e}, direct={G_dir[i,j]:.4e}")
else:
    print("  ⚠ return_dense=True did not return a dense matrix")

# %%
