#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import my_functions as myf
from scipy.linalg import inv

#%% Helper functions

def slices_to_matrix(H_slices, V):
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    H = np.zeros((N*dof, N*dof), dtype=np.complex128)

    for i in range(N):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = H_slices[i]

    for i in range(N-1):
        H[i*dof:(i+1)*dof, (i+1)*dof:(i+2)*dof] = V
        H[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof] = V.conj().T

    return H


def finite_lead_surface_gf(E, n, onsite, V, dof, eta):
    H = slices_to_matrix([onsite]*n, V)
    dim = H.shape[0]
    G = inv((E+1j*eta)*np.eye(dim) - H)
    return G[:dof, :dof]


#%% Parameters

sites_left, sites_mid, sites_right = 120, 40, 120
dof = 4

t = 1.0
mu_sc = 0.02
mu_n = 0.025
alpha = 0.5
B = 0.3
Delta = 0.1

eta = 1e-4
V = myf.t_matrix(t, alpha)

n_lead = 200
sites_to_plot = [0, sites_mid//2, sites_mid-1]

energies = np.linspace(-0.8, 0.8, 81)
phases = np.linspace(0, 2*np.pi, 101)

#%% Build system

H_full, _ = myf.build_sns_junction_sliced(
    t, mu_sc, mu_n, alpha, B, Delta, np.pi,
    sites_left, sites_right, sites_mid
)

H_mid, _ = myf.build_middle_region(
    t, mu_n, alpha, B, sites_mid
)

H_full_mat = slices_to_matrix(H_full, V)
I_full = np.eye(H_full_mat.shape[0])

#%% CLEAN LEAD MODEL (ONLY PHASE HERE)

def lead_surface(E, phi, side, method="finite"):

    if side == "L":
        onsite = myf.onsite_matrix(
            t, mu_sc, B, Delta*np.exp(-1j*phi/2)
        )
    else:
        onsite = myf.onsite_matrix(
            t, mu_sc, B, Delta*np.exp(+1j*phi/2)
        )

    if method == "finite":
        return finite_lead_surface_gf(E, n_lead, onsite, V, dof, eta)

    elif method == "sancho":
        g, _ = myf.get_surface_gf(
            E, onsite, V, eta=eta
        )
        return g

#%% STORAGE

results = {
    s: {"inv": [], "fin": [], "sancho": [], "finiteLead": []}
    for s in sites_to_plot
}

#%% ENERGY SWEEP (Sancho vs finite lead vs inversion)

phi_fixed = np.pi

for E in energies:

    G_inv = inv((E+1j*eta)*I_full - H_full_mat)

    G_fin = myf.get_rgf_finite_system(
        H_full, V, E, eta=eta, return_full=True
    )

    gL_f = lead_surface(E, phi_fixed, "L", "finite")
    gR_f = lead_surface(E, phi_fixed, "R", "finite")

    gL_s = lead_surface(E, phi_fixed, "L", "sancho")
    gR_s = lead_surface(E, phi_fixed, "R", "sancho")

    G_sancho = myf.get_rgf_sns(H_mid, V, gL_s, gR_s, E, eta=eta, return_full=True)
    G_finiteLead = myf.get_rgf_sns(H_mid, V, gL_f, gR_f, E, eta=eta, return_full=True)

    for s in sites_to_plot:

        i_m = s*dof
        i_f = (sites_left+s)*dof

        results[s]["inv"].append(
            -np.imag(np.trace(G_inv[i_f:i_f+dof, i_f:i_f+dof]))/np.pi
        )

        results[s]["fin"].append(
            -np.imag(np.trace(G_fin[i_f:i_f+dof, i_f:i_f+dof]))/np.pi
        )

        results[s]["sancho"].append(
            -np.imag(np.trace(G_sancho[i_m:i_m+dof, i_m:i_m+dof]))/np.pi
        )

        results[s]["finiteLead"].append(
            -np.imag(np.trace(G_finiteLead[i_m:i_m+dof, i_m:i_m+dof]))/np.pi
        )

#%% PHASE SWEEP (critical sanity test)

results_phase = {
    s: {"sancho": [], "finiteLead": []}
    for s in sites_to_plot
}

target_E = 0

for phi in phases:

    H_full, _ = myf.build_sns_junction_sliced(
        t, mu_sc, mu_n, alpha, B, Delta, phi,
        sites_left, sites_right, sites_mid
    )

    H_full_mat = slices_to_matrix(H_full, V)

    gL_s = lead_surface(target_E, phi, "L", "sancho")
    gR_s = lead_surface(target_E, phi, "R", "sancho")

    gL_f = lead_surface(target_E, phi, "L", "finite")
    gR_f = lead_surface(target_E, phi, "R", "finite")

    G_sancho = myf.get_rgf_sns(H_mid, V, gL_s, gR_s, target_E, eta=eta, return_full=True)
    G_finiteLead = myf.get_rgf_sns(H_mid, V, gL_f, gR_f, target_E, eta=eta, return_full=True)

    for s in sites_to_plot:

        i_m = s*dof

        results_phase[s]["sancho"].append(
            -np.imag(np.trace(G_sancho[i_m:i_m+dof, i_m:i_m+dof]))/np.pi
        )

        results_phase[s]["finiteLead"].append(
            -np.imag(np.trace(G_finiteLead[i_m:i_m+dof, i_m:i_m+dof]))/np.pi
        )

#%% PLOTS

for s in sites_to_plot:

    plt.figure(figsize=(10,5))
    plt.title(f"Sanity check site {s}")

    plt.plot(energies, results[s]["sancho"], label="Sancho")
    plt.plot(energies, results[s]["finiteLead"], label="Finite lead GF")
    plt.plot(energies, results[s]["inv"], "--", label="Direct inversion")

    plt.legend()
    plt.grid()
    plt.show()

# phase
    plt.figure(figsize=(10,5))
    plt.title(f"Phase dependence site {s}")

    plt.plot(phases/np.pi, results_phase[s]["sancho"], label="Sancho")
    plt.plot(phases/np.pi, results_phase[s]["finiteLead"], label="Finite lead")

    plt.legend()
    plt.grid()
    plt.show()

#%% google version 
import numpy as np
import matplotlib.pyplot as plt
import my_functions as myf
from scipy.linalg import inv

#%% Helper functions

def slices_to_matrix(H_slices, V):
    N = len(H_slices)
    dof = H_slices[0].shape[0]
    H = np.zeros((N*dof, N*dof), dtype=np.complex128)
    for i in range(N):
        H[i*dof:(i+1)*dof, i*dof:(i+1)*dof] = H_slices[i]
    for i in range(N-1):
        H[i*dof:(i+1)*dof, (i+1)*dof:(i+2)*dof] = V
        H[(i+1)*dof:(i+2)*dof, i*dof:(i+1)*dof] = V.conj().T
    return H

def finite_lead_surface_gf(E, n, onsite, V, dof, eta):
    H = slices_to_matrix([onsite]*n, V)
    dim = H.shape[0]
    G = inv((E+1j*eta)*np.eye(dim) - H)
    return G[:dof, :dof]

def apply_phase_unitary(g, phi):
    """ Applies U = diag(1, 1, e^{i*phi}, e^{i*phi}) to Green's function """
    U = np.diag([1, 1, np.exp(1j*phi), np.exp(1j*phi)])
    return U @ g @ U.conj().T

#%% Parameters
sites_left, sites_mid, sites_right = 100, 30, 100
dof = 4
t, mu_sc, mu_n, alpha, B, Delta = 1.0, 0.02, 0.025, 0.5, 0.3, 0.1
eta = 1e-4
V = myf.t_matrix(t, alpha)

n_lead = 200 # For the finite lead GF comparison
sites_to_plot = [0, sites_mid//2, sites_mid-1]

# Higher resolution for Energy to resolve Andreev levels
energies = np.linspace(-0.5, 0.5, 201) 
phases = np.linspace(0, 2*np.pi, 101)
phi_fixed = np.pi

#%% 1. ENERGY SWEEP: The Triple-Check
results = {s: {"inv": [], "sancho_Ham": [], "sancho_U": [], "finiteLead": []} for s in sites_to_plot}

# Pre-build Hamiltonian for inversion
H_full, _ = myf.build_sns_junction_sliced(t, mu_sc, mu_n, alpha, B, Delta, phi_fixed, sites_left, sites_right, sites_mid)
H_full_mat = slices_to_matrix(H_full, V)
I_full = np.eye(H_full_mat.shape[0])

# Middle region for RGF
H_mid, _ = myf.build_middle_region(t, mu_n, alpha, B, sites_mid)

print("Starting Energy Sweep...")
for E in energies:
    # A. Inversion (The baseline)
    G_inv = inv((E+1j*eta)*I_full - H_full_mat)

    # B. Sancho-Lopez (Hamiltonian Phase Injection)
    onsite_L = myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(-1j*phi_fixed/2))
    onsite_R = myf.onsite_matrix(t, mu_sc, B, Delta*np.exp(+1j*phi_fixed/2))
    gL_h, _ = myf.get_surface_gf(E, onsite_L, V, eta=eta)
    gR_h, _ = myf.get_surface_gf(E, onsite_R, V, eta=eta)
    G_sancho_H = myf.get_rgf_sns(H_mid, V, gL_h, gR_h, E, eta=eta, return_full=True)

    # C. Sancho-Lopez (Unitary Phase Injection)
    onsite_0 = myf.onsite_matrix(t, mu_sc, B, Delta)
    g0, _ = myf.get_surface_gf(E, onsite_0, V, eta=eta)
    gL_u = apply_phase_unitary(g0, -phi_fixed/2)
    gR_u = apply_phase_unitary(g0, +phi_fixed/2)
    G_sancho_U = myf.get_rgf_sns(H_mid, V, gL_u, gR_u, E, eta=eta, return_full=True)

    # D. Finite Lead GF (Sanity check for the RGF itself)
    gL_f = finite_lead_surface_gf(E, n_lead, onsite_L, V, dof, eta)
    gR_f = finite_lead_surface_gf(E, n_lead, onsite_R, V, dof, eta)
    G_fLead = myf.get_rgf_sns(H_mid, V, gL_f, gR_f, E, eta=eta, return_full=True)

    for s in sites_to_plot:
        i_m, i_f = s*dof, (sites_left+s)*dof
        results[s]["inv"].append(-np.imag(np.trace(G_inv[i_f:i_f+dof, i_f:i_f+dof]))/np.pi)
        results[s]["sancho_Ham"].append(-np.imag(np.trace(G_sancho_H[i_m:i_m+dof, i_m:i_m+dof]))/np.pi)
        results[s]["sancho_U"].append(-np.imag(np.trace(G_sancho_U[i_m:i_m+dof, i_m:i_m+dof]))/np.pi)
        results[s]["finiteLead"].append(-np.imag(np.trace(G_fLead[i_m:i_m+dof, i_m:i_m+dof]))/np.pi)

#%% 2. PHASE SWEEP
results_p = {s: {"sancho": []} for s in sites_to_plot}
target_E = 0

print("Starting Phase Sweep...")
# Optimized: Only compute Sancho lead once for E=0
onsite_0 = myf.onsite_matrix(t, mu_sc, B, Delta)
g0_E0, _ = myf.get_surface_gf(target_E, onsite_0, V, eta=eta)

for phi in phases:
    gL = apply_phase_unitary(g0_E0, -phi/2)
    gR = apply_phase_unitary(g0_E0, +phi/2)
    G_s = myf.get_rgf_sns(H_mid, V, gL, gR, target_E, eta=eta, return_full=True)

    for s in sites_to_plot:
        i_m = s*dof
        results_p[s]["sancho"].append(-np.imag(np.trace(G_s[i_m:i_m+dof, i_m:i_m+dof]))/np.pi)

#%% 3. FINAL PLOTTING
for s in sites_to_plot:
    # Energy Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Site {s} Energy LDOS")
    plt.plot(energies, results[s]["inv"], 'k-', lw=4, alpha=0.2, label="Direct Inversion")
    plt.plot(energies, results[s]["sancho_Ham"], 'r-', label="Sancho (Ham Phase)")
    plt.plot(energies, results[s]["sancho_U"], 'b--', label="Sancho (Unitary U)")
    plt.plot(energies, results[s]["finiteLead"], 'g:', label="Finite Lead RGF")
    plt.xlabel("Energy"); plt.ylabel("LDOS"); plt.legend(); plt.grid()

    # Phase Plot
    plt.subplot(1, 2, 2)
    plt.title(f"Site {s} Phase LDOS (E=0)")
    plt.plot(phases/np.pi, results_p[s]["sancho"], color='purple', lw=2)
    plt.xlabel(r"Phase ($\phi/\pi$)"); plt.ylabel("LDOS"); plt.grid()
    
    plt.tight_layout()
    plt.show()
# %%
