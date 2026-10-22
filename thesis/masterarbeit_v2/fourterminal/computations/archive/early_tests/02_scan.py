"""2D (or 1D) parameter scan of kappa and G: parallel, resumable, returns arrays. Ribbon parameters last."""
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
p0 = myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
KT = 0.0                                   # 0: T -> 0; e.g. 1e-3 for finite temperature
GRID = {"mu_s": np.linspace(-0.15, 0.55, 15), "Bz_s": np.linspace(0.6, 1.4, 17)}
N_WORKERS = 1                              # e.g. 8 on a multi-core machine
SAVE = f"scan_mu_Bz_KT{KT:g}.npz"          # rerun to continue an interrupted scan

#%% --- COMPUTE ---
if __name__ == "__main__":
    res = myf.scan(myf.ThermalPoint(p0, KT, phis=(np.pi, 0.0)), GRID, n_workers=N_WORKERS, save=SAVE)
    kappa, G, kappa_0 = res["values"][..., 0], res["values"][..., 1], res["values"][..., 2]   # (len(mu_s), len(Bz_s))
    print(f"failed lead energies: {int(np.nansum(res['values'][..., -1]))}")

#%% --- PLOT ---
    mu, Bz = res["axes"]["mu_s"], res["axes"]["Bz_s"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, Z, t in zip(ax, (kappa, kappa_0, G), ("kappa/kappa0 (pi)", "kappa/kappa0 (0)", "G/G0 (pi)")):
        im = a.pcolormesh(Bz, mu, Z, shading="nearest"); fig.colorbar(im, ax=a); a.set_title(t)
        a.set_xlabel("Bz_s"); a.set_ylabel("mu_s")
    fig.suptitle(myf.param_title(p0), fontsize=8); plt.tight_layout(); plt.show()
