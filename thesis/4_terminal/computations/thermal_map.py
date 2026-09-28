#%% --- IMPORTS ---
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import my_functions as myf
except ModuleNotFoundError:  
    import sys
    _start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    _root = next(p for p in (_start, *_start.parents) if (p / "my_functions" / "__init__.py").is_file())
    sys.path.insert(0, str(_root))
    import my_functions as myf
from my_functions import leads

#%% --- PARAMETERS ---
p_base = myf.Params(
    model='rashba', nx=80, ny=10, t_n=1.0, mu_n=0.5, t_c=1.0, mu_c=-0.01, t_s=1.0, mu_s=0.05,
    delta=0.25, phi=np.pi, tc_top=1.2, tc_bot=1.2, tc_barr=1.0, 
    alpha=0.8,
    Bz=0.0, Bxy=0.0, Bz_s=1.2, m0=0.8, m0_c=0.8, m0_n=0.8, eta=1e-4, kT=2e-3
)
E_fixed = np.array([0.0])

param1_name = 'Bz_s'                          # x-axis
param1_vals = np.linspace(2.0, 4.0, 41)
param2_name = 'mu_s'                          # y-axis
param2_vals = np.linspace(0.0, 2.0, 41)
param1_label = r"$B_{z,s}$"
param2_label = r"$\mu_s$"

out_file = Path("results") / f"thermal_map_{param1_name}_{param2_name}.npz"
out_file.parent.mkdir(exist_ok=True)

#%% --- COMPUTE T_th AND T_el ON THE GRID ---
T_th_grid = np.zeros((len(param2_vals), len(param1_vals)))
T_el_grid = np.zeros_like(T_th_grid)
unreliable = np.zeros(T_th_grid.shape, dtype=bool)

def store(j, i, ch, n_bad_before):
    T_th_grid[j, i] = ch['ee'][0] + ch['he_cross'][0]      # kappa / kappa0
    T_el_grid[j, i] = ch['ee'][0] - ch['he_cross'][0]      # G / G0
    unreliable[j, i] = leads.N_UNRELIABLE > n_bad_before
t0 = time.perf_counter()
for j, val2 in enumerate(param2_vals):
    if param1_name == 'phi':                         
        n_bad = leads.N_UNRELIABLE
        solver = myf.RGFFourTerminal(myf.FourTerminalJunction(replace(p_base, **{param2_name: val2})), E_fixed)
        for i, val1 in enumerate(param1_vals):
            store(j, i, solver.channels_at_phi(val1, side_name='right'), n_bad)
    else:
        for i, val1 in enumerate(param1_vals):
            p_curr = replace(p_base, **{param1_name: val1, param2_name: val2})
            n_bad = leads.N_UNRELIABLE
            solver = myf.RGFFourTerminal(myf.FourTerminalJunction(p_curr), E_fixed)
            store(j, i, solver.channels_at_phi(p_curr.phi, side_name='right'), n_bad)
    np.savez(out_file, T_th=T_th_grid, T_el=T_el_grid, unreliable=unreliable, rows_done=j + 1,
             param1_vals=param1_vals, param2_vals=param2_vals)       # saved after every row
    print(f"row {j + 1}/{len(param2_vals)} done ({time.perf_counter() - t0:.0f} s), "
          f"unreliable points so far: {unreliable.sum()}")

#%% --- PLOT ---
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
X, Y = np.meshgrid(param1_vals, param2_vals)

c0 = ax[0].contourf(X, Y, T_th_grid, levels=50, cmap='inferno')
c_aux = ax[0].contour(X, Y, T_th_grid, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')
print("Contour at T_th = 0.5:")
print(f"For     {param1_name}           {param2_name}")
for segment in c_aux.allsegs[0]:
    for p1, p2 in segment:
        
        print(f"    {p1:12.6f}   {p2:12.6f}")
fig.colorbar(c0, ax=ax[0]).set_label(r"$\kappa/\kappa_0 = T_{ee} + T_{he}$")
ax[0].set_title(f"Thermal conductance ($E = {E_fixed[0]:.2f}$)")
lim = max(abs(T_el_grid.min()), abs(T_el_grid.max()), 1e-6)
c1 = ax[1].contourf(X, Y, T_el_grid, levels=50, cmap='RdBu_r')
fig.colorbar(c1, ax=ax[1]).set_label(r"$G/G_0 = T_{ee} - T_{he}$")
ax[1].set_title("Nonlocal electrical conductance")

for a in ax:
    a.plot(X[unreliable], Y[unreliable], 'x', color='cyan', ms=4, label='lead GF unreliable')
    a.set_xlabel(param1_label)
    a.set_ylabel(param2_label)
if unreliable.any():
    ax[0].legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(out_file.with_suffix(".png"), dpi=150)
plt.show()
# %% --- PLOT T_th AND T_el LINE CUTS ---
which = 'p1' # From contourplot: x-axis is p1, y-axis is p2

t1, t2 = 1.2, -0.2  # fixed value for either param1 or param2

i, j = np.argmin(np.abs(param1_vals - t1)), np.argmin(np.abs(param2_vals - t2))
if which == 'p1':
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(param2_vals, T_th_grid[:, i], label=f"{param1_name} = {param1_vals[i]:.3f}")
    ax[1].plot(param2_vals, T_el_grid[:, i], label=f"{param1_name} = {param1_vals[i]:.3f}")
    ax[0].axhline(0.5, color='gray', lw=1.0, ls='--', label=r'$\kappa/\kappa_0$ = 0.5')

    ax[0].set_xlabel(param2_label)
    ax[1].set_xlabel(param2_label)
    fig.suptitle(f"Line cuts at {param1_label} = {t1:.3f}")

elif which == 'p2':
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(param1_vals, T_th_grid[j], label=f"{param2_name} = {param2_vals[j]:.3f}")
    ax[1].plot(param1_vals, T_el_grid[j], label=f"{param2_name} = {param2_vals[j]:.3f}")
    ax[0].set_xlabel(param1_label)
    ax[0].axhline(0.5, color='gray', lw=1.0, ls='--', label=r'$\kappa/\kappa_0$ = 0.5')
    ax[1].set_xlabel(param1_label)
    fig.suptitle(f"Line cuts at {param2_label} = {t2:.3f}")

ax[0].grid(alpha=0.5)
ax[1].grid(alpha=0.5)
ax[0].set_ylabel(r"$\kappa/\kappa_0$")
ax[1].set_ylabel(r"$G/G_0$")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
#%% --- DIAGNOSTICS 1: SC-RIBBON TOPOLOGY ON THE MAP GRID ---
# Bulk Chern number of the SC ribbons on a coarse version of the map grid, with the
# topological phase boundaries drawn over T_th.  Only meaningful if param1/param2
# change the ribbons (Bz_s, mu_s, alpha, beta, delta, t_s, Bxy, theta_z, m0).

step = 2                                            # use every 2nd grid point (Chern: ~0.05 s per point)
p1_c, p2_c = param1_vals[::step], param2_vals[::step]
C_grid = np.array([[myf.chern_number(replace(p_base, **{param1_name: v1, param2_name: v2}))
                    for v1 in p1_c] for v2 in p2_c])
print("Chern numbers found on the grid:", sorted(set(np.rint(C_grid).astype(int).ravel())))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
im0 = ax[0].pcolormesh(param1_vals, param2_vals, T_th_grid, cmap='inferno', shading='nearest')
ax[0].contour(p1_c, p2_c, C_grid, levels=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], colors='cyan', linewidths=1)
fig.colorbar(im0, ax=ax[0]).set_label(r"$\kappa/\kappa_0$")
ax[0].set_title("T_th (no interpolation) + topological phase boundaries (cyan)")
im1 = ax[1].pcolormesh(p1_c, p2_c, np.rint(C_grid), cmap='RdBu_r', vmin=-2.5, vmax=2.5, shading='nearest')
fig.colorbar(im1, ax=ax[1], ticks=[-2, -1, 0, 1, 2]).set_label(r"Chern number $\mathcal{C}$ of the SC ribbons")
ax[1].set_title("Bulk topology of the SC ribbons")
for a in ax:
    a.set_xlabel(param1_label)
    a.set_ylabel(param2_label)
plt.tight_layout()
plt.show()

#%% --- DIAGNOSTICS 2: ONE PARAMETER POINT (topology, ribbon edge gap, junction spectrum) ---
p_diag = replace(p_base, **{param1_name: 2.5, param2_name: 2.2})     # <- point to inspect

C = myf.chern_number(p_diag)
gap = myf.ribbon_gap(p_diag)                        # ~1-2 s at nx = 40
print(f"{param1_name} = {getattr(p_diag, param1_name)}, {param2_name} = {getattr(p_diag, param2_name)}:")
print(f"  Chern number of the SC ribbons : {C:+.2f}   ({'topological' if abs(C) > 0.5 else 'trivial'})")
print(f"  ribbon gap at E=0 (width nx={p_diag.nx}) : {gap:.1e}   "
      f"({'edge modes propagate' if gap < 10 * p_diag.eta else 'edge modes gapped by the finite width'})")

# junction spectrum with periodic boundary conditions in x (no normal leads); eta only broadens the lines
kx_vals = np.linspace(-np.pi, np.pi, 181)
E_vals = np.linspace(-1.0, 1.0, 201) * p_diag.delta
A = myf.junction_spectrum_kx(replace(p_diag, eta=2e-3), kx_vals, E_vals)     # ~10-30 s

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.pcolormesh(kx_vals / np.pi, E_vals / p_diag.delta, A, norm=LogNorm(vmin=A.max() * 1e-3, vmax=A.max()),
              cmap='inferno', shading='nearest')
ax.axhline(0, color='cyan', lw=0.7, ls='--')
ax.set_xlabel(r"$k_x a/\pi$")
ax.set_ylabel(r"$E/\Delta$")
ax.set_title(f"Junction spectrum, PBC in x, $\\phi$ = {p_diag.phi / np.pi:.2f}$\\pi$  (lines crossing E=0 carry heat)")
plt.tight_layout()
plt.show()
# %%