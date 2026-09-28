#%% --- IMPORTS ---
import time
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

try:
    import my_functions as myf
except ModuleNotFoundError:   # not pip-installed -> find <parent_folder>/my_functions by walking up
    import sys
    from pathlib import Path
    _start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    _root = next(p for p in (_start, *_start.parents) if (p / "my_functions" / "__init__.py").is_file())
    sys.path.insert(0, str(_root))
    import my_functions as myf

#%% --- COMPARISON OF FAST_PHASE_SWEEP AND RGF ---
p = myf.Params(
    model='rashba',
    nx=12, ny=6, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi, tc_top=1.00, tc_bot=1.00, tc_barr=1.00, alpha=0.2, beta=0.0,
    Bz=0.4, Bxy=0.0, theta_z=0.00*np.pi, Bz_s=0.4,
    eta=1e-5, kT=5e-6,
)

junction = myf.FourTerminalJunction(p)
E_sweep = np.linspace(-1.5*p.delta, 1.5*p.delta, 41)

phi_test = 0.37
t0 = time.perf_counter()
fast = myf.FastPhaseSweep(junction, E_sweep, phi_ref=0.0)
t_fast = time.perf_counter() - t0
print(f'System size: nx={p.nx}, ny={p.ny}, N_E={len(E_sweep)}')
print(f"FastPhaseSweep energy sweep: {t_fast:.4f} s")

t0 = time.perf_counter()
rgf = myf.RGFFourTerminal(junction, E_sweep, phi_ref=0.0)
t_rgf = time.perf_counter() - t0

print(f"RGF energy sweep:            {t_rgf:.4f} s")
print(f"Energy speedup: {t_fast:.1f}s / {t_rgf:.1f}s = {(t_fast)/(t_rgf):.2f}x")
a = fast.channels_at_phi(phi_test, "left")
b = rgf.channels_at_phi(phi_test, "left")
print(f"Max abs. error at phi={phi_test:.3f}:")
for key in a:
    print(f"  {key:9s} {np.max(np.abs(a[key] - b[key])):.2e}")   # type: ignore

t0 = time.perf_counter()
for ph in np.linspace(0, 2*np.pi, 41):
    fast.channels_at_phi(ph, "left")
t_fast_phi = time.perf_counter() - t0

t0 = time.perf_counter()
for ph in np.linspace(0, 2*np.pi, 41):
    rgf.channels_at_phi(ph, "left")
t_rgf_phi = time.perf_counter() - t0

print(f"per-phi: dense={1000*t_fast_phi/41:.1f} ms   rgf={1000*t_rgf_phi/41:.1f} ms")
print(f"full phi sweep: dense={t_fast+t_fast_phi:.1f}s  rgf={t_rgf+t_rgf_phi:.1f}s")
print(f"Phase speedup: {(t_fast+t_fast_phi)/(t_rgf+t_rgf_phi):.2f}x")

# %% --- PHASE DEPENDENCE OF TRANSMISSION COEFFICIENTS ---
p = myf.Params(
    model='rashba',
    nx=80, ny=30, t_n=1.0, mu_n=1.00, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=0.0, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    alpha=-0.4, beta=0.0, Bz=0.0, Bxy=0.0, theta_z=0.00*np.pi, Bz_s=2.0,
    eta=1e-5, kT=2e-3,
)

E_fixed = np.array([0.0])
phi_vals = np.linspace(0.0, 2 * np.pi, 201)

junction = myf.FourTerminalJunction(p)
rgf_solver = myf.RGFFourTerminal(junction, E_fixed, phi_ref=p.phi)

T_ee, T_eh, T_he, T_lar_eh = [], [], [], []
for phi_val in phi_vals:
    ch_right = rgf_solver.channels_at_phi(phi_val, side_name='right')
    T_ee.append(ch_right['ee'])            # normal transmission (e -> e)
    T_eh.append(ch_right['eh_cross'])      # CAR (h -> e)
    T_he.append(ch_right['he_cross'])      # CAR (e -> h)
    T_lar_eh.append(ch_right['eh_local'])  # LAR

T_th = np.array(T_ee) + np.array(T_he)

fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax[0].plot(phi_vals / np.pi, np.array(T_he), label=r"$T_{he}$ (CAR)")
ax[0].plot(phi_vals / np.pi, T_ee, label=r"$T_{ee}$ ")
ax[0].set_title(fr'Transmissions at fixed E={E_fixed[0]:.2f} vs. $\phi$')
ax[0].set_ylabel("Transmission")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(phi_vals / np.pi, T_th, label=r"$T_{th}$")
ax[1].set_xlabel(r"$\phi / \pi$")
ax[1].set_ylabel("Transmission")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# %% --- DENSE REFERENCE CHECK (small system: dense inverse vs RGF) ---
p_small = replace(p, nx=8, ny=5, phi=0.83)
E_small = np.linspace(-0.5, 0.5, 11)
dense = myf.FourTerminalJunction(p_small).channels(E_small, side_name='right')
rgf_small = myf.RGFFourTerminal(myf.FourTerminalJunction(p_small), E_small).channels_at_phi(p_small.phi, 'right')
print("max |dense - RGF| per channel:", {k: f"{np.max(np.abs(dense[k] - rgf_small[k])):.1e}" for k in dense})
# %%
