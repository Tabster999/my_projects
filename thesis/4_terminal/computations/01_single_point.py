"""One parameter point: diagnostics, linear response (T -> 0 and finite T), phase dependence, T(E)."""
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

#%% --- PARAMETERS (Rashba working point) ---
p = myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
kT = 1e-3

#%% --- COMPUTE ---
print(f"C = {myf.chern_number(p):+.0f}, edge gap = {myf.ribbon_gap(p, nk=61):.1e}, xi = {myf.coherence_length(p):.2f} sites")
kappa0, G0 = myf.linear_response(p, np.pi, 0.0)
kappa, G, err_k, err_G = myf.linear_response(p, np.pi, kT, error=True)
print(f"T->0: kappa={kappa0:.4f} G={G0:+.4f} | kT={kT:g}: kappa={kappa:.4f}+-{err_k:.0e} G={G:+.4f}+-{err_G:.0e}")
phis = np.linspace(0, 2 * np.pi, 121)
kappa_phi, G_phi = myf.linear_response(p, phis, 0.0)            # all phases share one solver
E = np.linspace(-0.01, 0.01, 81)
T = myf.transmissions(p, E, np.pi)
np.savez("single_point.npz", phis=phis, kappa_phi=kappa_phi, G_phi=G_phi, E=E, **T)

#%% --- PLOT ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(phis / np.pi, kappa_phi, label="kappa/kappa0"); ax[0].plot(phis / np.pi, G_phi, label="G/G0")
ax[0].set_xlabel("phi / pi"); ax[0].legend()
ax[1].plot(E, T["ee"] + T["he_cross"], label="T_ee + T_he"); ax[1].plot(E, T["ee"], label="T_ee"); ax[1].plot(E, T["he_cross"], label="T_he")
ax[1].set_xlabel("E"); ax[1].legend()
fig.suptitle(myf.param_title(p), fontsize=8); plt.tight_layout(); plt.show()
