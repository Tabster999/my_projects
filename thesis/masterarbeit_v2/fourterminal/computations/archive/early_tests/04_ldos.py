"""LDOS at E = 0 at phi = pi and phi = 0: where the Majorana channel lives."""
#%% --- IMPORTS ---
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
p = myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
cols = list(range(0, p.nx, 2))

#%% --- COMPUTE ---
A = {phi: myf.ldos(p, [0.0], phi, cols)[0] for phi in (np.pi, 0.0)}      # (n_cols, ny) each
np.savez("ldos.npz", cols=cols, A_pi=A[np.pi], A_0=A[0.0])

#%% --- PLOT ---
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for a, (phi, M) in zip(ax, A.items()):
    im = a.pcolormesh(cols, np.arange(p.ny), M.T, shading="nearest"); fig.colorbar(im, ax=a)
    a.set_title(f"LDOS(E=0), phi = {phi/np.pi:.0f} pi"); a.set_xlabel("x"); a.set_ylabel("y")
plt.tight_layout(); plt.show()
