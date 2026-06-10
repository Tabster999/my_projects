"""
example_ldos_2d.py
==================
Computes and plots the 2D LDOS(E, φ) map for a 2D SNS junction,
comparing the finite and infinite-lead cases.

Reproduces the last panel of the original script using SNSJunction.
"""
#%% ── imports ──────────────────────────────────────────────────────────
import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
module_root = current_dir.parent
sys.path.insert(0, str(module_root))
import modules

from modules import SNSJunction

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "figure.titlesize": 18,
})

#%% ── junction parameters ───────────────────────────────────────────────────────
junc = SNSJunction(
    t         = 1.0,
    mu_sc     = 0.0025,
    mu_n      = 0.01,
    alpha     = .4,
    h         = 0.3,
    delta     = .1,
    SL        = 70,
    SM        = 20,
    SR        = 70,
    N_y       = 10,
    eta       = 1e-3,
    symmetric = False,
    n_jobs    = -1,        # use all available cores
)

# ── sweep parameters ──────────────────────────────────────────────────────────
energies = np.linspace(-0.15, 0.15, 81)
phases   = np.linspace(-1e-2, 2 * np.pi + 1e-2, 81)

# probe site: middle of normal region, middle of transverse width
probe_x = 1     # 50
probe_y = 0 # 5

#%% ── compute ───────────────────────────────────────────────────────────────────
print("Computing 2D LDOS sweep (energy × phase) ...")
print(f"  probe site: x={probe_x}, y={probe_y}")
print(f"  grid: {len(energies)} energies × {len(phases)} phases\n")

ldos_2d = junc.ldos_2d_sweep(
    energies, phases,
    probe_x = probe_x,
    probe_y = probe_y,
    modes   = ["infinite", "finite"],
)
# ldos_2d.shape → (N_E, N_PHI, 2)
#                              ↑ index 0 = infinite, 1 = finite

ldos_inf = ldos_2d[:, :, 0]   # (N_E, N_PHI)
ldos_fin = ldos_2d[:, :, 1]

#%% ── plot ──────────────────────────────────────────────────────────────────────
vmin = min(ldos_inf.min(), ldos_fin.min())
vmax = 5

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, data, title in zip(
    axes,
    [ldos_inf, ldos_fin],
    ["Infinite leads", "Finite system"],
):
    cf = ax.contourf(
        phases / np.pi, energies, data,
        levels = 309,
        cmap   = "coolwarm",
        vmin   = vmin,
        vmax   = vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("φ / π")
    ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.5)   # E = 0 guide

axes[0].set_ylabel("Energy ")

cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.02)
cbar.set_label("LDOS (arb. units)")

fig.suptitle(
    f"LDOS(E, φ)  —  probe x={probe_x}, y={probe_y}  "
    f"(N_y={junc.N_y}, SM={junc.SM}, Δ={junc.delta*1e6:.0f} μeV, "
    f"h={junc.h:.1f})"
)
plt.show()
# %%
