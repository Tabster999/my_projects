"""Finite-bias transport of the left lead: current (EC/CAR/LAR) and differential conductances."""
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
p = myf.Params(model="rashba", nx=40, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
kT = 2e-3
E = np.linspace(-0.6, 0.6, 481)            # must cover the bias window plus a few kT
V = np.linspace(-0.4, 0.4, 81)             # V_L = +V, V_R = -V

#%% --- COMPUTE ---
ch = myf.transmissions(p, E, np.pi, side="left")                 # once; reused for all biases
I = myf.current(ch, E, V, -V, kT, parts=True)
G_loc, G_nl = myf.differential_conductance(ch, E, V, -V, kT)
np.savez("finite_bias.npz", V=V, G_loc=G_loc, G_nl=G_nl, **I)

#%% --- PLOT ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in ("EC", "CAR", "LAR", "total"): ax[0].plot(V, I[k], label=k)
ax[0].set_xlabel("V"); ax[0].set_ylabel("I_L [e/h]"); ax[0].legend()
ax[1].plot(V, G_loc, label="dI_L/dV_L"); ax[1].plot(V, G_nl, label="dI_L/dV_R"); ax[1].set_xlabel("V"); ax[1].legend()
plt.tight_layout(); plt.show()
