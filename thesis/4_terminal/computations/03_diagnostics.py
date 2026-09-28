"""SC-ribbon diagnostics vs field: Chern number, Majorana decay length xi, edge gap; edge-state profile."""
#%% --- IMPORTS ---
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
try:
    import my_functions as myf
except ModuleNotFoundError:                     # not pip-installed: find the package by walking up
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(next(p for p in (_here, *_here.parents) if (p / "my_functions").is_dir())))
    import my_functions as myf

#%% --- PARAMETERS ---
p0 = myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
Bs = np.linspace(0.3, 1.6, 27)

#%% --- COMPUTE ---
C = np.array([myf.chern_number(replace(p0, Bz_s=b), N=40) for b in Bs])
xi = np.array([myf.coherence_length(replace(p0, Bz_s=b)) for b in Bs])
gap = np.array([myf.ribbon_gap(replace(p0, Bz_s=b), nk=41) for b in Bs])
x, dens, xi_fit, _ = myf.edge_state_profile(replace(p0, nx=200))
np.savez("diagnostics.npz", Bs=Bs, C=C, xi=xi, gap=gap, x=x, dens=dens)

#%% --- PLOT ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(Bs, C, "o-"); ax[0].set_ylabel("C")
ax[1].plot(Bs, xi, "o-"); ax[1].axhline(p0.nx / 12, ls="--", c="gray"); ax[1].set_ylabel("xi (sites); dashed: nx/12")
ax[2].semilogy(x, dens); ax[2].set_xlabel("x"); ax[2].set_title(f"edge state, xi = {xi_fit:.1f}")
for a in ax[:2]: a.set_xlabel("Bz_s")
plt.tight_layout(); plt.show()
