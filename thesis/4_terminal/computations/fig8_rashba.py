"""
fig8_rashba.py -- Fig. 8 of Gresta et al. in the RASHBA model, at the working point found
                  for half-integer quantization (kappa/kappa0 = 0.5 with G/G0 = 0 at phi = pi).

  (a) kappa/kappa0 vs (Bz_s, mu_s)        at the working-point mu_c
  (b) kappa/kappa0 and G/G0 vs mu_c       at the working-point (Bz_s, mu_s)
  (c) ... vs Bz_s                          at the working-point mu_s
  (d) ... vs mu_s                          at the working-point Bz_s

KT = 0 evaluates the T -> 0 limit (kappa/kappa0 = T_th(E=0)); KT > 0 integrates Eqs. (7)-(8)
with myf.linear_response_nodes (16 energies per point).

Requirements for the plateau (all checkable before running, see the printout):
  * ribbons topological:            myf.chern_number(p) != 0
  * edge modes not gapped by width: myf.ribbon_gap(p) << eta (KT = 0) or << kT (KT > 0)
  * centre below its first subband: mu_c inside the confinement window

Runtimes (single core): panels (b)-(d) a few minutes; panel (a) ~30 min at nx=100, KT=0.
The map is resumable -- rerun the script (or set MAX_SECONDS) until it finishes.
Switching p_base to the Dirac parameters of the paper reproduces the Dirac version.
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

#%% --- PARAMETERS (Rashba working point) ---
p_base = myf.Params(
    model='rashba', nx=85, ny=20, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.05, t_s=1.0, mu_s=0.05,
    delta=0.25, phi=np.pi, tc_top=1.0, tc_bot=1.0, tc_barr=1.0,
    alpha=1.2, Bz=0.0, Bxy=0.0, Bz_s=0.7, eta=2e-5, kT = 0.0       # nx 120 -> 80, Bz_s 0.7 -> 0.6, eta 1e-5 -> 1e-6
)
KT = 0.0                     # 0 -> T -> 0 limit; e.g. 1e-3 -> finite temperature (16 energies/point)
MAX_SECONDS = None             # e.g. 250 to stop the map early and resume on the next run
param1_name = 'Bz_s'                          # x-axis
param1_vals = np.linspace(0.50, 2.20, 31)
param2_name = 'mu_s'                          # y-axis
param2_vals = np.linspace(-0.05, 0.4, 31)
param1_label = r"$B_{z,s}$"
param2_label = r"$\mu_s$"
PANELS = {   # name: (x-parameter, x-values, y-parameter or None, y-values)
    'b': ('mu_c', np.linspace(-0.06, 0.05, 41), None, None),
    'c': ('Bz_s', np.linspace(0.50, 2.00, 41), None, None),
    'd': ('t_c', np.linspace(0.5, 2.0, 41), None, None),
    'a': (param1_name, param1_vals, param2_name, param2_vals),
}
out = Path("results"); out.mkdir(exist_ok=True)
E_nodes, w_th, w_el = myf.linear_response_nodes(KT, n=16)

print(f"working point: C = {myf.chern_number(p_base):+.2f}, ribbon edge gap = {myf.ribbon_gap(p_base, nk=61):.1e}, "
      f"eta = {p_base.eta:g}" + (f", kT = {KT:g}" if KT else " (T -> 0)"))
#%% --- COMPUTE (resumable) ---

def point(p):
    """(kappa/kappa0, G/G0) at phi = p.phi, and the same at phi = 0."""
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(p), E_nodes)
    out_ = []
    for ph in (p.phi, 0.0):
        ch = rgf.channels_at_phi(ph, 'right')
        out_ += [np.dot(w_th, ch['ee'] + ch['he_cross']), np.dot(w_el, ch['ee'] - ch['he_cross'])] #type: ignore
    return out_          # [kappa(phi), G(phi), kappa(0), G(0)]

def compute(name):
    f = out / f"fig8r_{name}_KT{KT:g}.npz"
    n1, v1, n2, v2 = PANELS[name]
    if f.exists():
        d = np.load(f)
        if d['done'].all():
            return d['data'], d['done']
        data, done = d['data'].copy(), d['done'].copy()
    else:
        shape = (len(v1), 4) if n2 is None else (len(v2), len(v1), 4)
        data, done = np.full(shape, np.nan), np.zeros(shape[:-1][0] if n2 is None else len(v2), dtype=bool)
    t0 = time.perf_counter()
    if n2 is None:                                   # 1D scan
        for i, a in enumerate(v1):
            if done[i]:
                continue
            data[i] = point(replace(p_base, **{n1: a}))
            done[i] = True
    else:                                            # 2D map, row by row
        for j, b in enumerate(v2):
            if done[j]:
                continue
            if MAX_SECONDS is not None and time.perf_counter() - t0 > MAX_SECONDS:
                break
            for i, a in enumerate(v1):
                data[j, i] = point(replace(p_base, **{n1: a, n2: b}))
            done[j] = True
            time_min = (time.perf_counter() - t0) // 60
            time_sec = (time.perf_counter() - t0) % 60
            print(f"  panel a: row {done.sum()}/{len(v2)} ({n2}={b:+.3f}), {time_min:.0f} min {time_sec:.0f} s, "
                  f"unreliable lead energies: {leads.N_UNRELIABLE}")
    return data, done

if __name__ == "__main__":
    results = {}
    for name in PANELS:                              # cheap panels first, map last
        results[name], done = compute(name)
        print(f"panel {name}: {'done' if done.all() else 'INCOMPLETE - rerun to continue'}")

#%% --- PLOT ---
def plot_fig8_rashba(results):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _, v1, _, v2 = PANELS['a']
    d = results['a']
    im = axs[0, 0].pcolormesh(v1, v2, d[:, :, 0], cmap='inferno', vmin=0, vmax=1.2, shading='nearest')
    fig.colorbar(im, ax=axs[0, 0]).set_label(r"$\kappa/\kappa_0$")
    axs[0, 0].contour(v1, v2, np.abs(d[:, :, 0] - 0.5) < 0.05, levels=[0.5], colors='cyan', linewidths=1)
    axs[0, 0].set_xlabel(r"$B_{z,s}$"); axs[0, 0].set_ylabel(r"$\mu_s$")
    axs[0, 0].set_title(r"(a) $\kappa/\kappa_0$ (cyan: within 0.05 of 0.5)", fontsize=10)
    for ax, name, lab in ((axs[0, 1], 'b', r"$\mu_c$"), (axs[1, 0], 'c', r"$B_{z,s}$"), (axs[1, 1], 'd', r"$\mu_s$")):
        x, r = PANELS[name][1], results[name]
        ax.plot(x, r[:, 0], '-', color='tab:blue', label=r'$\kappa/\kappa_0$, $\phi=\pi$')
        ax.plot(x, r[:, 2], ':', color='tab:cyan', label=r'$\kappa/\kappa_0$, $\phi=0$')
        ax.plot(x, r[:, 1], '-', color='tab:red', label=r'$G/G_0$, $\phi=\pi$')
        ax.axhline(0.5, color='gray', lw=0.6, ls=':'); ax.axhline(0, color='gray', lw=0.6)
        ax.set_xlabel(lab); ax.set_title(f"({name}) scan of {lab}", fontsize=10); ax.legend(fontsize=7)
    fig.suptitle(myf.param_title(p_base, extra=("T -> 0 limit (E = 0)" if KT == 0 else f"$k_BT$ = {KT:g}")), fontsize=8, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.93]) #type: ignore
    fig.savefig(out / f"fig8_rashba_KT{KT:g}.png", dpi=130)
    return fig

if __name__ == "__main__":
    plot_fig8_rashba(results)
    plt.show()
# %%
which = 'p2' # From contourplot: x-axis is p1, y-axis is p2

t1, t2 = 1.2, 0.2  # fixed value for either param1 or param2

T_th_grid = results['a'][:, :, 0]
T_el_grid = results['a'][:, :, 1]
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
# %%
