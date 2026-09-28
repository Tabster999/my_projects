"""
map_working_point.py -- kappa/kappa0 and G/G0 at E = 0 over (Bz_s, mu_s) at the Rashba
working point.  Resumable: finished rows are stored in results/, rerun to continue.
Set MAX_SECONDS to stop after a while and continue later.
"""
#%% --- IMPORTS ---
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

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
    model='rashba', nx=60, ny=20, t_n=1.0, mu_n=0.5, t_c=1.0, mu_c=-0.01, t_s=1.0, mu_s=0.05,
    delta=0.25, phi=np.pi, tc_top=1.2, tc_bot=1.2, tc_barr=1.0,
    alpha=0.8, Bz=0.0, Bxy=0.0, Bz_s=0.6, eta=1e-6,
)
Bz_vals = np.linspace(0.30, 1.00, 36)          # x axis
mu_vals = np.linspace(-0.20, 0.40, 31)         # y axis
MAX_SECONDS = 250                               # e.g. 250 to stop and resume later

out = Path("results"); out.mkdir(exist_ok=True)
f_map = out / f"map_Bz_mu_nx{p_base.nx}_ny{p_base.ny}.npz"

#%% --- COMPUTE (resumable) ---
if f_map.exists():
    d = np.load(f_map)
    T_pi, T_0, G_pi, done = d['T_pi'].copy(), d['T_0'].copy(), d['G_pi'].copy(), d['done'].copy()
else:
    T_pi, T_0, G_pi = (np.full((len(mu_vals), len(Bz_vals)), np.nan) for _ in range(3))
    done = np.zeros(len(mu_vals), dtype=bool)
t0 = time.perf_counter()
for j, mu in enumerate(mu_vals):
    if done[j]:
        continue
    if MAX_SECONDS is not None and time.perf_counter() - t0 > MAX_SECONDS:
        break
    for i, Bz in enumerate(Bz_vals):
        rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(replace(p_base, Bz_s=Bz, mu_s=mu)), [0.0])
        a, b = rgf.channels_at_phi(np.pi, 'right'), rgf.channels_at_phi(0.0, 'right')
        T_pi[j, i] = a['ee'][0] + a['he_cross'][0]
        G_pi[j, i] = a['ee'][0] - a['he_cross'][0]
        T_0[j, i] = b['ee'][0] + b['he_cross'][0]
    done[j] = True
    np.savez(f_map, T_pi=T_pi, T_0=T_0, G_pi=G_pi, done=done, Bz_vals=Bz_vals, mu_vals=mu_vals)
    print(f"row {done.sum()}/{len(mu_vals)} (mu_s={mu:+.3f}) done, {time.perf_counter()-t0:.0f} s, "
          f"unreliable lead energies so far: {leads.N_UNRELIABLE}")

#%% --- PLOT ---
if done.all():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a_, Z, ttl, kw in ((ax[0], T_pi, r"$\kappa/\kappa_0$ at $\phi=\pi$", dict(cmap='inferno', vmin=0, vmax=1.2)),
                           (ax[1], T_0, r"$\kappa/\kappa_0$ at $\phi=0$", dict(cmap='inferno', vmin=0, vmax=1.2)),
                           (ax[2], G_pi, r"$G/G_0$ at $\phi=\pi$", dict(cmap='RdBu_r', vmin=-0.5, vmax=0.5))):
        im = a_.pcolormesh(Bz_vals, mu_vals, Z, shading='nearest', **kw)
        fig.colorbar(im, ax=a_)
        a_.set_title(ttl); a_.set_xlabel(r"$B_{z,s}$"); a_.set_ylabel(r"$\mu_s$")
    ax[0].contour(Bz_vals, mu_vals, np.abs(T_pi - 0.5) < 0.05, levels=[0.5], colors='cyan', linewidths=1.2)
    fig.suptitle(myf.param_title(p_base, extra=r"E = 0, cyan: $|\kappa/\kappa_0-0.5|<0.05$"), fontsize=8, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(f_map.with_suffix('.png'), dpi=130); plt.show()
# %%
