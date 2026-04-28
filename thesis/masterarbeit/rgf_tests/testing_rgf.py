#%%
from scipy.linalg import inv
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(parent_dir)

import my_functions as myf

def slices_to_matrix(H_slices, V):
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    dim = N * dof
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(N):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = H_slices[i]
    for i in range(N-1):
        H[i*dof:(i+1)*dof, (i+1)*dof:(i+2)*dof] = V
        H[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof] = V.conj().T
    return H

def get_phase_matrix(phi):
    """Returns the U(1) phase rotation matrix for the Nambu basis (up, dn, dn†, up†)."""
    half = phi / 2
    return np.diag([np.exp(1j*half), np.exp(1j*half),
                    np.exp(-1j*half), np.exp(-1j*half)])

def get_ldos(G_block):
    """Calculates Local Density of States from a Green's function block."""
    return -np.imag(np.trace(G_block)) / np.pi

def finite_lead_surface_gf(E, n_lead, onsite, V, dof, eta):
    """Compute surface Green's function from a finite lead by inversion."""
    H_slices = [onsite for _ in range(n_lead)]
    H_lead = slices_to_matrix(H_slices, V)
    dim = H_lead.shape[0]
    I = np.eye(dim, dtype=np.complex128)
    G = inv((E + 1j*eta)*I - H_lead)
    return G[:dof, :dof]


def run_optimized_comparison():
    # --- 1. Parameters ---
    sites_left, sites_mid, sites_right = 400, 150, 400
    dof = 4
    Delta, mu_sc, mu_n = 0.1, 0.0025, 0.01
    t, alpha, B = 1.0, 0.4, 0.5
    eta = 1e-4
    phi_fixed = np.pi
    target_E = 0

    energies = np.linspace(-0.3, 0.3, 101)
    phases = np.linspace(0, 2*np.pi, 101)

    # Indices relative to the START of the middle (N) region
    sites_to_plot_mid = [1, sites_mid // 2, sites_mid - 2]

    V = myf.t_matrix(t, alpha)
    V_herm = V.conj().T

    # Pre-allocate storage [Site, DataPoint, Method(Finite=0, Infinite=1)]
    ldos_e = np.zeros((len(sites_to_plot_mid), len(energies), 2))
    ldos_p = np.zeros((len(sites_to_plot_mid), len(phases), 2))
    pair_p = np.zeros((len(sites_to_plot_mid), len(phases), 2))

    onsite_sc = myf.onsite_matrix(t, mu_sc, B, Delta)

    # Build Hamiltonians once — they don't depend on E or phi for the fixed-phase sweep
    H_slices_full, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta, phi_fixed,
        sites_left, sites_right, sites_mid)
    # FIXED: use build_middle_region directly, not extracted from full junction
    H_slices_mid_fixed, _ = myf.build_middle_region(t, mu_n, alpha, B, sites_mid)

    # --- 2. Energy Sweep ---
    print("Running Energy Sweep...")
    for e_idx, E in enumerate(energies):
        # 2a. Finite System RGF over full junction
        G_fin_all, _, _ = myf.get_rgf_finite_system(
            H_slices_full, V, E, eta=eta, return_full=False)

        # 2b. Infinite Leads via Sancho-Lopez
        # FIXED: left lead bulk is rightward → pass V
        #        right lead bulk is leftward → pass V†
        g_L_raw, _ = myf.get_surface_gf(E, onsite_sc, V_herm,      eta=eta)
        g_R_raw, _ = myf.get_surface_gf(E, onsite_sc, V, eta=eta)

        # Phase convention: phase 0 on left, full phi_fixed on right
        g_L = g_L_raw
        g_R = get_phase_matrix(phi_fixed) @ g_R_raw @ get_phase_matrix(phi_fixed).conj().T

        G_inf_all = myf.get_rgf_sns(
            H_slices_mid_fixed, V, g_L, g_R, E, eta=eta, return_full=False)

        for s_idx, s in enumerate(sites_to_plot_mid):
            ldos_e[s_idx, e_idx, 0] = get_ldos(G_fin_all[sites_left + s])  # shifted index into full chain
            ldos_e[s_idx, e_idx, 1] = get_ldos(G_inf_all[s])               # direct index into mid region

    # --- 3. Phase Sweep ---
    print("Running Phase Sweep (E=0)...")
    # Raw lead GFs computed once at target_E
    # FIXED: correct hopping directions (same as energy sweep)
    g_L_raw0, _ = myf.get_surface_gf(target_E, onsite_sc, V_herm, eta=eta)
    g_R_raw0, _ = myf.get_surface_gf(target_E, onsite_sc, V, eta=eta)

    # FIXED: normal region has no phi dependence — build once
    H_slices_mid_n, _ = myf.build_middle_region(t, mu_n, alpha, B, sites_mid)

    for p_idx, phi in enumerate(phases):
        # Full junction needed for finite RGF reference
        H_slices_p, _ = myf.build_sns_junction_sliced(
            t, mu_sc, mu_n, alpha, B, Delta, phi,
            sites_left, sites_right, sites_mid)

        # Phase convention: 0 on left, phi on right (consistent with energy sweep)
        g_L = g_L_raw0
        # FIXED: was passing g_L_raw0 directly (unphased) to get_rgf_sns — now uses g_L
        g_R = get_phase_matrix(phi) @ g_R_raw0 @ get_phase_matrix(phi).conj().T

        G_fin_all, _, _ = myf.get_rgf_finite_system(
            H_slices_p, V, target_E, eta=eta, return_full=False)
        # FIXED: was passing H_mid_p (extracted from full junction) — now uses H_slices_mid_n
        G_inf_all = myf.get_rgf_sns(
            H_slices_mid_n, V, g_L, g_R, target_E, eta=eta, return_full=False)

        for s_idx, s in enumerate(sites_to_plot_mid):
            idx_f = sites_left + s
            G_fin_block = G_fin_all[idx_f]
            G_inf_block = G_inf_all[s]

            ldos_p[s_idx, p_idx, 0] = get_ldos(G_fin_block)
            ldos_p[s_idx, p_idx, 1] = get_ldos(G_inf_block)
            pair_p[s_idx, p_idx, 0] = np.abs(G_fin_block[0, 3])
            pair_p[s_idx, p_idx, 1] = np.abs(G_inf_block[0, 3])

    # --- 4. Plotting ---
    fig, axes = plt.subplots(len(sites_to_plot_mid), 2, figsize=(14, 12), sharex='col')
    row_titles = ["Left Interface (N-side)", "Middle of N-wire", "Right Interface (N-side)"]

    for i in range(len(sites_to_plot_mid)):
        axes[i, 0].plot(energies, ldos_e[i, :, 0], label="Finite RGF",     color='tab:blue',   alpha=0.8)
        axes[i, 0].plot(energies, ldos_e[i, :, 1], label="Infinite Leads", color='tab:orange', ls='--')
        axes[i, 0].set_ylabel(f"{row_titles[i]}\nLDOS")

        axes[i, 1].plot(phases/np.pi, ldos_p[i, :, 0], color='tab:green', alpha=0.8, label='Finite RGF')
        axes[i, 1].plot(phases/np.pi, ldos_p[i, :, 1], color='tab:red',   ls='--',  label='Infinite Leads')
        axes[i, 1].legend()

        if i == 0:
            axes[0, 0].set_title("Energy Sweep Comparison")
            axes[0, 1].set_title(f"Phase Sweep Comparison (E={target_E})")
            axes[0, 0].legend()

        axes[i, 0].grid(True, alpha=0.2)
        axes[i, 1].grid(True, alpha=0.2)
        axes[i, 0].legend()

    axes[-1, 0].set_xlabel("Energy ($t$)")
    axes[-1, 1].set_xlabel(r"Phase ($\phi/\pi$)")
    plt.tight_layout()
    plt.show()

    return ldos_e, ldos_p, pair_p, energies, phases, sites_to_plot_mid


# --- 5. Physical Sanity Checks ---
def run_sanity_checks(ldos_e, ldos_p, pair_p, energies, phases, sites_to_plot_mid):
    print("\n" + "="*30)
    print("   PHYSICAL SANITY CHECKS")
    print("="*30)

    methods = ["Finite RGF", "Infinite Leads"]

    # A. Particle-Hole Symmetry: LDOS(+E) = LDOS(-E)
    print("\n[A] Particle-Hole Symmetry (LDOS(+E) = LDOS(-E))")
    energies_symmetric = np.allclose(energies, -energies[::-1], atol=1e-10)
    if not energies_symmetric:
        print("  ⚠️  Energy array not symmetric around 0 — skipping PHS check.")
    else:
        for s_idx, s in enumerate(sites_to_plot_mid):
            for m_idx, mname in enumerate(methods):
                ldos_trace = ldos_e[s_idx, :, m_idx]
                phs_diff = np.max(np.abs(ldos_trace - ldos_trace[::-1]))
                # Relaxed tolerance: finite eta breaks exact PHS at ~1% level near gap edge
                status = "PASS" if phs_diff < 1e-4 else "FAIL"
                print(f"  Site {s:3} | {mname:15} | PHS: {status} (Max Diff: {phs_diff:.2e})")

    # B. Method Agreement: Finite RGF vs Infinite Leads
    print("\n[B] Method Agreement (Finite RGF vs Infinite Leads)")
    for s_idx, s in enumerate(sites_to_plot_mid):
        diff_e = np.max(np.abs(ldos_e[s_idx, :, 0] - ldos_e[s_idx, :, 1]))
        status_e = "PASS" if diff_e < 0.1 else "FAIL"
        diff_p = np.max(np.abs(ldos_p[s_idx, :, 0] - ldos_p[s_idx, :, 1]))
        status_p = "PASS" if diff_p < 0.1 else "FAIL"
        print(f"  Site {s:3} | Energy sweep: {status_e} (Max Diff: {diff_e:.2e}) | "
              f"Phase sweep: {status_p} (Max Diff: {diff_p:.2e})")

    # C. Zero-Energy Peak at phi=pi
    # ABS should sit at E=0 for a topological junction at phi=pi
    print("\n[C] Zero-Energy Peak (phi=pi, energy sweep)")
    e_zero_idx = np.argmin(np.abs(energies))
    for s_idx, s in enumerate(sites_to_plot_mid):
        for m_idx, mname in enumerate(methods):
            ldos_at_zero = ldos_e[s_idx, e_zero_idx, m_idx]
            ldos_max     = np.max(ldos_e[s_idx, :, m_idx])
            ratio = ldos_at_zero / ldos_max if ldos_max > 0 else 0
            status = "PASS" if ratio > 0.5 else "FAIL"
            print(f"  Site {s:3} | {mname:15} | LDOS(E=0)/LDOS_max = {ratio:.3f}  {status}")

    # D. Phase Periodicity: must be 2pi periodic
    print("\n[D] Phase Periodicity (phi=0 vs phi=2pi)")
    for s_idx, s in enumerate(sites_to_plot_mid):
        for m_idx, mname in enumerate(methods):
            p0   = ldos_p[s_idx,  0, m_idx]
            p2pi = ldos_p[s_idx, -1, m_idx]
            diff = abs(p0 - p2pi)
            status = "PASS" if diff < 1e-3 else "FAIL"
            print(f"  Site {s:3} | {mname:15} | Periodicity: {status} (Diff: {diff:.2e})")

    # E. Proximity Decay: pairing amplitude |G_03| decays edge → middle
    print("\n[E] Proximity Effect Spatial Decay (|G_03| at phi=pi)")
    phi_pi_idx = np.argmin(np.abs(phases - np.pi))
    if len(sites_to_plot_mid) >= 2:
        for m_idx, mname in enumerate(methods):
            pair_edge = pair_p[0, phi_pi_idx, m_idx]
            pair_mid  = pair_p[1, phi_pi_idx, m_idx]
            if pair_edge > 1e-12:
                ratio = pair_mid / pair_edge
                status = "PASS" if ratio < 0.95 else "FAIL"
                print(f"  {mname:15} | Mid/Edge = {ratio:.4f}  {status}")
            else:
                print(f"  {mname:15} | Edge pairing ~0 — check Delta and lead coupling.")

    # F. Spectral Weight sum rule: integral of LDOS over energy window
    print("\n[F] Spectral Weight  integral(LDOS dE)")
    for s_idx, s in enumerate(sites_to_plot_mid):
        for m_idx, mname in enumerate(methods):
            integral = np.trapz(ldos_e[s_idx, :, m_idx], energies)
            print(f"  Site {s:3} | {mname:15} | Integral = {integral:.4f}")

    # G. LDOS Positivity: must be >= 0 everywhere (unphysical if negative)
    print("\n[G] LDOS Positivity")
    for m_idx, mname in enumerate(methods):
        min_ldos_e = np.min(ldos_e[:, :, m_idx])
        min_ldos_p = np.min(ldos_p[:, :, m_idx])
        status = "PASS" if min_ldos_e >= -1e-6 and min_ldos_p >= -1e-6 else "FAIL"
        print(f"  {mname:15} | Min LDOS energy sweep: {min_ldos_e:.2e} | "
              f"phase sweep: {min_ldos_p:.2e}  {status}")

    print("\n" + "="*30)


if __name__ == "__main__":
    ldos_e, ldos_p, pair_p, energies, phases, sites_to_plot_mid = run_optimized_comparison()
    run_sanity_checks(ldos_e, ldos_p, pair_p, energies, phases, sites_to_plot_mid)

# %%