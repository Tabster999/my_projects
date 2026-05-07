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

These two bare GFs are NOT the same object because V is not unitary in general
(spin-orbit coupling makes V complex with non-trivial phase structure).
Specifically, the anomalous component g[0,3] acquires an INTRINSIC phase from
the accumulated hopping phases in the Sancho-López iteration:

  angle(g_bare_R[0,3]) ~ +pi  (from V  iterations)
  angle(g_bare_L[0,3]) ~  0   (from V† iterations)
  intrinsic difference  = pi

This pi offset is NOT a physical superconducting phase — it is a gauge artifact
of the chain direction. If uncorrected, the N region sees an effective phase
difference of pi even at phi=0, breaking the periodicity entirely.

CORRECTION STRATEGY:
Before applying the physical gauge rotation U(±phi/2), we rotate g_bare_R by
U(-pi/4) to strip the intrinsic pi phase offset (since U(theta) shifts Delta
by 2*theta, we need theta = -pi/4 to remove a pi phase from g_bare_R[0,3]):

  U_fix = phase_matrix(-pi/4)
  g_bare_R_fixed = U_fix @ g_bare_R @ U_fix†

After this, both bare GFs have the same intrinsic anomalous phase (~0), and
the physical rotation U(±phi/2) correctly produces:

  angle(Sigma_R[0,3]) = +phi/2 * 2 = +phi/2  ... wait, no:
  angle(Sigma_R[0,3]) = +phi    (since U(phi/2) shifts Delta by phi)
  angle(Sigma_L[0,3]) = -phi
  phase difference across junction = phi  ✓

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


def get_surface_gfs(E, phi, symmetric=True, eta=1e-3):
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
        UL = myf.phase_matrix(-phi / 2.0)
        UR = myf.phase_matrix(+phi / 2.0)
    else:
        UL = np.eye(4, dtype=np.complex128)
        UR = myf.phase_matrix(-phi)

    g_L, _ = myf.get_surface_gf(E, onsite_sc, V.conj().T, eta=eta)
    g_R, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)

    return UL @ g_L @ UL.conj().T, UR @ g_R @ UR.conj().T

def ph_check(name, M):
    ok = np.allclose(M, -C_ph @ M.conj() @ C_ph, atol=1e-12)
    print(f"  PH symmetry [{name}]: {'✓ PASS' if ok else '✗ FAIL'}")

def compute_energy_slice(e_idx, E):
    """Compute LDOS for all phases at a fixed energy."""
    row = np.zeros(N_PHI)

    for p_idx, phi in enumerate(phases):
        g_L, g_R = get_surface_gfs(E, phi, SYMMETRIC, eta=eta)

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
SYMMETRIC = True     # True  → ±phi/2 on left/right
                     # False → full phi on right only

SL, SM, SR = 200, 101, 200        # sites: left SC | normal | right SC
DOF        = 4
Delta      = 0.1
mu_sc      = 0.0025
mu_n       = 0.01
t          = 1.0
alpha      = 0.4
B          = 0.3
eta        = 1e-3
phi_fixed  = np.pi

N_E   = 151                     # energy points
N_PHI = 151                     # phase points
energies = np.linspace(-.15, .15, N_E)
phases   = np.linspace(0, 2*np.pi, N_PHI)

probe_sites  = [1, SM // 2, SM - 2]
probe_labels = ["Left interface", "Centre of N", "Right interface"]

# ── Fixed objects ─────────────────────────────────────────────────────────────
V      = myf.t_matrix(t, alpha)
V_dag  = V.conj().T
onsite_sc    = myf.onsite_matrix(t, mu_sc, B, Delta)   # real Delta
H_mid_slices, _ = myf.build_middle_region(t, mu_n, alpha, B, SM)

H_full_slices, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, phi_fixed,
    SL, SR, SM, symmetric=SYMMETRIC)

# ── Particle-hole symmetry check on Hamiltonians ──────────────────────────────
C_ph = np.fliplr(np.eye(DOF))   # antidiag identity – PH conjugation matrix


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

# ── Quick phase-rotation sanity check ─────────────────────────────────────────
# Verify that rotating a real-Delta onsite by U(phi/2) gives the correct
# complex-Delta onsite.
print("\n══ Gauge-rotation self-test ══")
for phi_test in [0.0, np.pi/2, np.pi]:
    U = myf.phase_matrix(-phi_test)   # U(-phi) gives Delta*exp(+i*phi)
    onsite_rot = U @ onsite_sc @ U.conj().T
    onsite_ref = myf.onsite_matrix(t, mu_sc, B, Delta * np.exp(1j * phi_test))
    err = np.max(np.abs(onsite_rot - onsite_ref))
    print(f"  phi={phi_test/np.pi:.2f}pi: err={err:.2e}  {'✓' if err < 1e-12 else '✗'}")

#%% energy sweep 
# ═════════════════════════════════════════════════════════════════════════════
# ENERGY SWEEP  (phi fixed = pi)
# ═════════════════════════════════════════════════════════════════════════════
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

    g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC, eta=eta)
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
# ═════════════════════════════════════════════════════════════════════════════
# PHASE SWEEP  (E = 0)
# ═════════════════════════════════════════════════════════════════════════════
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

    g_L_p, g_R_p = get_surface_gfs(energy_fixed, phi, SYMMETRIC, eta=eta)
    G_inf_p, *_ = myf.get_rgf_sns(
        H_mid_slices, V, g_L_p, g_R_p, energy_fixed, eta=eta, return_full=False)

    for s_idx, s in enumerate(probe_sites):
        blk_fin = G_fin_p[SL + s]
        blk_inf = G_inf_p[s]
        for m, blk in enumerate([blk_fin, blk_inf]):
            ldos_p[s_idx, p_idx, m] = get_ldos(blk)
            pair_p[s_idx, p_idx, m] = get_pairing(blk)

#%% figures 
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Energy sweep: LDOS comparison per probe site
# ═════════════════════════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Phase sweep: LDOS and pairing at E = 0
# ═════════════════════════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Method difference map |LDOS_inf - LDOS_fin|
# ═════════════════════════════════════════════════════════════════════════════
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
ax3.set_xlabel("Energy  (E / t)", fontsize=11)
ax3.set_ylabel("Site index (N region)", fontsize=11)
ax3.legend(fontsize=9)
fig3.colorbar(im, ax=ax3, label="|ΔLDOS|")
fig3.tight_layout()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Spatial LDOS map (site × energy, finite RGF ground truth)
# ═════════════════════════════════════════════════════════════════════════════
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
    ax.set_xlabel("Energy  (E / t)", fontsize=12)
    ax.set_ylabel("Site index (N region)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig4.colorbar(im4, ax=ax, label="LDOS  (arb.)")
fig4.tight_layout()

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – |G_ij| matrix structure at E≈0, phi=pi
# ═════════════════════════════════════════════════════════════════════════════
e0_idx = np.argmin(np.abs(energies))
s_mid  = 1   # centre probe
fig5, axes5 = plt.subplots(1, 2, figsize=(8, 4))
fig5.suptitle(f"|G_ij| at E≈0, site={probe_labels[s_mid]}  (φ=π)",
              fontsize=12, fontweight="bold")
for m, (ax, ttl) in enumerate(zip(axes5, ["Finite RGF", "Sancho-López"])):
    mat = np.abs(Gblk_e[s_mid, e0_idx, m])
    im5 = ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set_title(ttl, fontsize=11)
    ax.set_xticks(range(DOF))
    ax.set_xticklabels([r"$↑$", r"↓", r"$↓^\dagger$", r"$↑^\dagger$"], fontsize=10)
    ax.set_yticks(range(DOF))
    ax.set_yticklabels([r"↑", r"↓", r"$↓^\dagger$", r"$↑^\dagger$"], fontsize=10)
    for i in range(DOF):
        for j in range(DOF):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i,j] > mat.max()*0.5 else "black")
    fig5.colorbar(im5, ax=ax, shrink=0.75)
fig5.tight_layout()
#%% fig 6 and 7: 2D contour: LDOS vs Phase vs Energy
# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – 2D contour: LDOS vs Phase vs Energy  (Sancho-López)
# ═════════════════════════════════════════════════════════════════════════════
print("══ Running 2D Phase×Energy sweep (Sancho-López) ══")

probe = probe_sites[0]

# Run in parallel over all energies
results = Parallel(n_jobs=-2)(
    delayed(compute_energy_slice)(e_idx, E)
    for e_idx, E in enumerate(energies)
)

# Assemble result
ldos_2d = np.zeros((N_PHI, N_E))
for e_idx, row in results:
    ldos_2d[:, e_idx] = row
fig6, ax6 = plt.subplots(figsize=(8, 6))

pcm = ax6.pcolormesh(
    phases / np.pi,
    energies,
    ldos_2d.T,      # transpose!
    shading='auto',
    cmap='magma'
)

cbar = fig6.colorbar(pcm, ax=ax6)
cbar.set_label("LDOS")

ax6.set_xlabel(r"$\phi / \pi$")
ax6.set_ylabel("Energy")

plt.tight_layout()
plt.show()
#%% Fig 7 ldos fin vs inf computation
print("══ Running 2D Phase×Energy sweep (Finite RGF) ══")
# Parallel execution
results_fin = Parallel(n_jobs=-2)(
    delayed(compute_energy_slice_fin)(e_idx, E)
    for e_idx, E in enumerate(energies)
)

# Assemble
ldos_2d_fin = np.zeros((N_PHI, N_E))
for e_idx, row in results_fin:
    ldos_2d_fin[:, e_idx] = row
#%% show fig7
fig7, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, data, title in zip(
    axes,
    [ldos_2d_fin, ldos_2d],
    ["Finite RGF", "Sancho–López (∞ leads)"]
):
    pcm = ax.pcolormesh(
        phases / np.pi,
        energies,
        data.T,
        shading='auto',
        cmap='magma'
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\phi / \pi$")
    ax.axhline(0, color='white', linestyle='--', linewidth=1)
axes[0].set_ylabel("Energy")

fig7.colorbar(pcm, ax=axes, label="LDOS")
plt.show()
#%% SANITY CHECKS  (printed report)

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
sizes_to_test = [50, 100, 150, 200, 800]

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

        g_L, g_R = get_surface_gfs(E, phi_fixed, SYMMETRIC, eta=eta)
        G_inf_test, *_ = myf.get_rgf_sns(
            H_mid_slices, V, g_L, g_R, E, eta=eta, return_full=False)
        ldos_inf_test[e_idx] = get_ldos(G_inf_test[probe_sites[1]])

    max_err_sub = np.max(np.abs(ldos_fin_test[subgap] - ldos_inf_test[subgap]))
    max_err_all = np.max(np.abs(ldos_fin_test - ldos_inf_test))
    print(f"  SL=SR={sl_sr:4d} | sub-gap max|Δ|={max_err_sub:.3e}  "
          f"{verdict(max_err_sub < tol_conv, tol_conv)}  (full={max_err_all:.3e})")
    
# [10] Phase symmetry check on 2D map
sym_err = np.max(np.abs(ldos_2d - ldos_2d[::-1, :]))
print(f"\n[10]  Phase-symmetry of 2D map  ldos(phi) == ldos(2pi-phi): "
      f"max err={sym_err:.3e}  {verdict(sym_err < 5e-3)}")

# [11] Surface GF PH symmetry (left vs right)
print("\n[11] Surface GF PH symmetry")
C = np.fliplr(np.eye(4))

errs = []
for E in energies:
    g, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)
    g_m, _ = myf.get_surface_gf(-E, onsite_sc, V, eta=eta)
    
    err = np.max(np.abs(g + C @ g_m.conj() @ C))
    errs.append(err)

print(f"  max error = {max(errs):.2e}")
print(f"  mean error = {np.mean(errs):.2e}")
print(f"  verdict: {'✓ PASS' if max(errs) < 1e-6 else '✗ FAIL'}")

# [12] self energy consistency 
print("\n[12] Self-energy causality + symmetry")

for E in energies[::len(energies)//5]:
    g, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)
    Sigma = V.conj().T @ g @ V
    
    # causality: Im Σ ≤ 0
    eig_im = np.linalg.eigvals(0.5j * (Sigma - Sigma.conj().T))
    max_im = np.max(eig_im.real)
    
    print(f"  E={E:+.3f}  max Im Σ eigenvalue = {max_im:.2e}  "
            f"{'✓' if max_im < 1e-8 else '✗'}")
plt.show()

def eta_stability_test(E, phi, site):
    etas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2] 
    vals_fin = []
    vals_inf = [] 
    for eta_test in etas:
        G_fin, *_ = myf.get_rgf_finite_system( H_full_slices, V, E, eta=eta_test) 
        gL, gR = get_surface_gfs(E, phi, SYMMETRIC, eta=eta_test) 
        G_inf, *_ = myf.get_rgf_sns( H_mid_slices, V, gL, gR, E, eta=eta_test)
        vals_fin.append(get_ldos(G_fin[SL + site]))
        vals_inf.append(get_ldos(G_inf[site])) 
    return etas, vals_fin, vals_inf

def run_eta_diagnostic(E, phi, site_idx, site_label="probe"):
    print("\n" + "═"*70)
    print(" ETA STABILITY DIAGNOSTIC")
    print("═"*70)
    print(f"Energy  E = {E:.4f}")
    print(f"Phase   φ = {phi/np.pi:.2f} π")
    print(f"Site    = {site_label} (index {site_idx})")
    print("─"*70)
    
    etas, vals_fin, vals_inf = eta_stability_test(E, phi, site_idx)

    vals_fin = np.array(vals_fin)
    vals_inf = np.array(vals_inf)

    # ---- diagnostics ----
    fin_var = np.max(vals_fin) - np.min(vals_fin)
    inf_var = np.max(vals_inf) - np.min(vals_inf)

    fin_rel = fin_var / (np.mean(vals_fin) + 1e-15)
    inf_rel = inf_var / (np.mean(vals_inf) + 1e-15)

    print(f"Finite system LDOS:")
    print(f"  min = {vals_fin.min():.4e}, max = {vals_fin.max():.4e}")
    print(f"  relative variation = {fin_rel:.2e}")

    print(f"\nInfinite leads LDOS:")
    print(f"  min = {vals_inf.min():.4e}, max = {vals_inf.max():.4e}")
    print(f"  relative variation = {inf_rel:.2e}")

    # stability verdict
    tol = 5e-2
    stable = (fin_rel < tol) and (inf_rel < tol)

    print("\nVerdict:")
    if stable:
        print("  ✓ ETA-STABLE regime (results reliable)")
    else:
        print("  ✗ ETA-SENSITIVE regime (results not converged)")
    print("═"*70)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(etas, vals_fin, "o-", label="Finite RGF", color="tab:red")
    ax.plot(etas, vals_inf, "s-", label="Sancho–López", color="tab:green")

    ax.set_xscale("log")
    ax.set_xlabel("η")
    ax.set_ylabel("LDOS")
    ax.set_title(f"η stability test | E={E:.3f}, φ={phi/np.pi:.2f}π")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return stable

# choose a physically interesting point
E_test = 0.0
phi_test = np.pi
site_test = probe_sites[1]   # centre of N region

stable = run_eta_diagnostic(
    E=E_test,
    phi=phi_test,
    site_idx=site_test,
    site_label="N-centre"
)
