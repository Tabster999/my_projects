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
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.linalg import inv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import my_functions as myf

# ── Colour palette (colour-blind friendly) ────────────────────────────────────
C_INV  = "#2166ac"   # blue  – inversion
C_FIN  = "#d6604d"   # red   – finite RGF
C_INF  = "#4dac26"   # green – infinite leads
ALPHA  = 0.85

# ── Helpers ───────────────────────────────────────────────────────────────────
def slices_to_matrix(H_slices, V):
    N   = len(H_slices)
    dof = H_slices[0].shape[0]
    dim = N * dof
    H   = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = H_slices[i]
    for i in range(N - 1):
        H[i*dof:(i+1)*dof,   (i+1)*dof:(i+2)*dof] = V
        H[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof]   = V.conj().T
    return H

def phase_matrix(phi):
    """U(1) gauge rotation in Nambu basis (up,dn,dn†,up†)."""
    h = phi / 2
    return np.diag([np.exp(1j*h), np.exp(1j*h), np.exp(-1j*h), np.exp(-1j*h)]).astype(np.complex128)

def get_ldos(G_block):
    return -np.imag(np.trace(G_block)) / np.pi

def get_pairing(G_block):
    """Anomalous amplitude |G[0,3]| = |⟨c↑ c↑†⟩| proxy."""
    return np.abs(G_block[0, 3])

def get_symmetric_phases(phi):
    """Returns phase matrices for Left (-phi/2) and Right (+phi/2) leads."""
    return phase_matrix(-phi/2), phase_matrix(phi/2)

def get_surface_gfs(E, phi, symmetric):
    """
    Returns (g_L, g_R) surface GFs in the correct gauge.
    symmetric=True  → ±φ/2 on left/right leads
    symmetric=False → full φ on right lead only
    """
    onsite_L = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(-1j * phi / 2 if symmetric else 0.0))
    onsite_R = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(+1j * phi / 2 if symmetric else 1j * phi))
    g_L, _ = myf.get_surface_gf(E, onsite_L, V, eta=eta)
    g_R, _ = myf.get_surface_gf(E, onsite_R, V,     eta=eta)
    g_L = g_L.conj().T
    return g_L, g_R

# ── Parameters ────────────────────────────────────────────────────────────────
SYMMETRIC = True    # True  → ±φ/2 on left/right
                    # False → full φ on right only

SL, SM, SR = 80, 30, 80        # sites: left SC | normal | right SC
DOF        = 4
Delta      = 0.10
mu_sc      = 0.0025
mu_n       = 0.01
t          = 1.0
alpha      = 0.40
B          = 0.50              # Zeeman; topological when B > sqrt(Delta²+mu_sc²)
eta        = 1e-3
phi_fixed  = np.pi

N_E   = 101                    # energy points
N_PHI = 101                    # phase points
energies = np.linspace(-.8, .8, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

probe_sites = [1, SM // 2, SM - 2]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# ── Fixed objects ───────────────────────────────────────────────────────
V      = myf.t_matrix(t, alpha)
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
#%%
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
    G_fin, _, _ = myf.get_rgf_finite_system(H_full_slices, V, E, eta=eta, return_full=False)

    g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC)
    G_inf = myf.get_rgf_sns(H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_inv = G_inv[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF]
        blk_fin = G_fin[SL + s]
        blk_inf = G_inf[s]
        for m, blk in enumerate([blk_inv, blk_fin, blk_inf]):
            ldos_e[s_idx, e_idx, m] = get_ldos(blk)
            pair_e[s_idx, e_idx, m] = get_pairing(blk)
            Gblk_e[s_idx, e_idx, m] = blk

    for s in range(SM):
        ldos_map[s, e_idx]     = get_ldos(G_inv[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF])
        ldos_all_fin[s, e_idx] = get_ldos(G_fin[SL + s])
        ldos_all_inf[s, e_idx] = get_ldos(G_inf[s])
#%%
# ═════════════════════════════════════════════════════════════════════════════
# PHASE SWEEP  (E = 0)
# ═════════════════════════════════════════════════════════════════════════════
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
    G_fin_p, _, _ = myf.get_rgf_finite_system(
        H_full_p, V, energy_fixed, eta=eta, return_full=False)

    g_L_p, g_R_p = get_surface_gfs(energy_fixed, phi, SYMMETRIC)
    G_inf_p = myf.get_rgf_sns(H_mid_slices, V, g_L_p, g_R_p, energy_fixed, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_inv = G_inv_p[(SL+s)*DOF:(SL+s+1)*DOF, (SL+s)*DOF:(SL+s+1)*DOF]
        blk_fin = G_fin_p[SL + s]
        blk_inf = G_inf_p[s]
        for m, blk in enumerate([blk_inv, blk_fin, blk_inf]):
            ldos_p[s_idx, p_idx, m] = get_ldos(blk)
            pair_p[s_idx, p_idx, m] = get_pairing(blk)

#%%
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
fig1.savefig("fig1_energy_sweep.png", dpi=150, bbox_inches="tight")
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
fig2.savefig("fig2_phase_sweep.png", dpi=150, bbox_inches="tight")
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
fig3.savefig("fig3_error_maps.png", dpi=150, bbox_inches="tight")


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
fig4.savefig("fig4_spatial_ldos.png", dpi=150, bbox_inches="tight")

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
    ax.set_xticks(range(DOF)); ax.set_xticklabels([r"↑", r"↓", r"↓†", r"↑†"], fontsize=10)
    ax.set_yticks(range(DOF)); ax.set_yticklabels([r"↑", r"↓", r"↓†", r"↑†"], fontsize=10)
    for i in range(DOF):
        for j in range(DOF):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i,j] > mat.max()*0.5 else "black")
    fig5.colorbar(im5, ax=ax, shrink=0.75)
fig5.tight_layout()
fig5.savefig("fig5_Gmatrix.png", dpi=150, bbox_inches="tight")
print("  saved all figures")
#%%
# ═════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS  (printed report)
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
sizes_to_test = [20, 50, 100, 150, 200]
for sl_sr in sizes_to_test:
    H_mat_test = myf.build_sns_junction(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, sl_sr, sl_sr, SM, DOF, symmetric=SYMMETRIC)
    I_test = np.eye(H_mat_test.shape[0], dtype=np.complex128)

    ldos_inv_test = np.zeros(N_E)
    ldos_inf_test = np.zeros(N_E)
    for e_idx, E in enumerate(energies):
        z = E + 1j * eta
        G_inv_test = inv(z * I_test - H_mat_test)
        blk = G_inv_test[(sl_sr + probe_sites[1])*DOF:(sl_sr + probe_sites[1]+1)*DOF,
                          (sl_sr + probe_sites[1])*DOF:(sl_sr + probe_sites[1]+1)*DOF]
        ldos_inv_test[e_idx] = get_ldos(blk)

        g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC)
        G_inf_test = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        ldos_inf_test[e_idx] = get_ldos(G_inf_test[probe_sites[1]])

    max_err = np.max(np.abs(ldos_inv_test - ldos_inf_test))
    print(f"  SL=SR={sl_sr:4d} | max|Δ LDOS|={max_err:.3e}  {verdict(max_err < tol_inf, tol_inf)}")

print(f"\n{SEP}")
print("  Done.")
print(SEP)

plt.show()
# %%
