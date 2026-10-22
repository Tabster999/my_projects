"""Fig. 8 (b,c,d) of Gresta et al.: kappa, G vs mu_c and vs the SC Zeeman field at two mu_s.
MODEL = "dirac" reproduces the paper, MODEL = "rashba" gives the Rashba analogue."""
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
MODEL, KT, N_WORKERS = "rashba", 0.0, 1
if MODEL == "dirac":
    p0 = myf.Params(model="dirac", nx=60, ny=20, t_n=1.0, t_c=1.0, t_s=1.0, mu_n=1.0, mu_c=0.0, delta=0.35, phi=np.pi,
                    tc_top=1.0, tc_bot=1.0, tc_barr=1.0, m0=0.8, m0_c=0.8, m0_n=0.8, eta=1e-5)
    PANELS = {"b": {"mu_s": [0.8], "Bz_s": [-2.0], "mu_c": np.linspace(0, 1, 101)},
              "c": {"mu_s": [0.77], "Bz_s": np.linspace(-2.5, 2.0, 181)},
              "d": {"mu_s": [0.27], "Bz_s": np.linspace(-2.5, 2.0, 181)}}
else:
    p0 = myf.Params(model="rashba", nx=80, ny=20, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.02, t_s=1.0, delta=0.35, phi=np.pi,
                    tc_top=1.0, tc_bot=1.0, tc_barr=1.0, alpha=1.2, eta=1e-5)
    PANELS = {"b": {"mu_s": [0.15], "Bz_s": [0.94], "mu_c": np.linspace(-0.1, 0.5, 121)},
              "c": {"mu_s": [0.15], "Bz_s": np.linspace(0.0, 1.6, 81)},
              "d": {"mu_s": [0.40], "Bz_s": np.linspace(0.0, 1.6, 81)}}

#%% --- COMPUTE ---
if __name__ == "__main__":
    f = myf.ThermalPoint(p0, KT, phis=(np.pi,))
    res = {n: myf.scan(f, g, n_workers=N_WORKERS, save=f"fig8_{MODEL}_{n}_KT{KT:g}.npz") for n, g in PANELS.items()}

#%% --- PLOT ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (n, r) in zip(ax, res.items()):
        key = list(PANELS[n])[-1]                       # the scanned axis
        v = r["values"].reshape(-1, r["values"].shape[-1])
        a.plot(r["axes"][key], v[:, 0], "b", label="kappa/kappa0"); a.plot(r["axes"][key], v[:, 1], "r", label="G/G0")
        a.axhline(0.5, c="gray", ls=":"); a.set_xlabel(key); a.set_title(f"({n})")
    ax[0].legend(); fig.suptitle(myf.param_title(p0), fontsize=8); plt.tight_layout(); plt.show()
