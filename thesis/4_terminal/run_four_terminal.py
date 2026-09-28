#%% --- IMPORTS ---
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(r"c:\coding\my_projects\thesis\fourterminal")
import my_functions as myf

SCHEMES = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
           ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')]


def mark_gap(ax, delta):
    ax.axvline(-delta, color='gray', linestyle=':')
    ax.axvline(delta, color='gray', linestyle=':')


#%% --- BUILD JUNCTION & CHANNELS ---
# E only has to resolve T(E); the thermal window is refined automatically in current()/conductance().
p = myf.Params(
    nx=20, ny=40, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi,
    alpha=0.2, beta=0.0, Bz=0.4, Bxy=0.0, theta_z=0.0, eta=1e-4, kT=2e-3,
)
V_range = np.linspace(-1.5 * p.delta, 1.5 * p.delta, 41)
E_sweep = myf.energy_grid(V_range.max(), p.kT, 41)

print(f"Computing {len(E_sweep)} energies for {p.nx} x {p.ny} sites...")
junction = myf.FourTerminalJunction(p)
ch = junction.channels(E_sweep)          # {'left': {...}, 'right': {...}}, arrays over E_sweep
print("Finished.")
myf.print_params(p)

#%% --- CURRENTS (sym / anti bias) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
for ax, (scheme, title) in zip(axes, SCHEMES):
    VL, VR = myf.bias_line(V_range, scheme)
    I = myf.current(ch['left'], E_sweep, VL, VR, p.kT)
    for key in ('EC', 'CAR', 'LAR'):
        ax.plot(V_range, I[key], linewidth=2, label=rf'$I_{{\rm {key}}}$')
    ax.plot(V_range, I['total'], linewidth=2, color='k', linestyle='--', alpha=.5, label=r'$I_{\rm total}$')
    mark_gap(ax, p.delta)
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.5)
    ax.legend(loc='upper left')
    myf.style_axis(ax, f"Currents - {title}")
axes[0].set_ylabel(r'Current ($e/h$ units)')
plt.tight_layout()
plt.show()

#%% --- CONDUCTANCES (analytic dI/dV) ---
# G[scheme]['LL'] = dI_L/dV_L, ['LR'] = dI_L/dV_R, ['RR'], ['RL'] likewise; each {'EC','CAR','LAR','total'}
G = {}
for scheme, _ in SCHEMES:
    VL, VR = myf.bias_line(V_range, scheme)
    G_LL, G_LR = myf.conductance(ch['left'], E_sweep, VL, VR, p.kT)
    G_RR, G_RL = myf.conductance(ch['right'], E_sweep, VR, VL, p.kT)
    G[scheme] = {'LL': G_LL, 'LR': G_LR, 'RR': G_RR, 'RL': G_RL}

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for col, (scheme, title) in enumerate(SCHEMES):
    s = 1 if scheme == 'sym' else -1
    G_LL, G_LR = G[scheme]['LL'], G[scheme]['LR']
    # dI_L/dV along the bias line = G_LL +/- G_LR (chain rule), channel by channel
    ax = axes[0, col]
    for key in ('EC', 'CAR', 'LAR'):
        ax.plot(V_range, G_LL[key] + s * G_LR[key], linewidth=2, label=rf'$G_{{\rm {key}}}$')
    ax.plot(V_range, G_LL['total'] + s * G_LR['total'], linewidth=2, color='k', linestyle='--', alpha=.5,
            label=r'$G_{\rm total}$')
    mark_gap(ax, p.delta)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    myf.style_axis(ax, title)

    ax = axes[1, col]
    ax.plot(V_range, G_LL['total'], linewidth=2, color='tab:green', alpha=.7, label=r'$G_{LL}$')
    ax.plot(V_range, G_LR['total'], linewidth=2, color='tab:gray', alpha=.7, label=r'$G_{LR}$')
    ax.plot(V_range, G_LL['total'] + s * G_LR['total'], linewidth=2, color='tab:orange', alpha=.5,
            label=r'$G_{LL}+G_{LR}$' if s > 0 else r'$G_{LL}-G_{LR}$')
    mark_gap(ax, p.delta)
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right')
    myf.style_axis(ax, 'Local and non-local conductance')
axes[0, 0].set_ylabel(r'$dI_L/dV_{\rm bias}$ ($e^2/h$)')
axes[1, 0].set_ylabel(r'Differential conductance ($e^2/h$)')
plt.tight_layout()
plt.show()

#%% --- SYMMETRIZATION TEST (uses p / G from above) ---
G_LL, G_LR = G['sym']['LL']['total'], G['sym']['LR']['total']
even = lambda x: 0.5 * (x + x[::-1])      # V_range is symmetric around 0
odd = lambda x: 0.5 * (x - x[::-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(V_range, even(G_LL), linewidth=2, label=r'$G_{LL}^S$')
axes[0].plot(V_range, even(G_LR), linewidth=2, label=r'$G_{LR}^S$')
myf.style_axis(axes[0], 'Symmetric bias: even parts')
axes[1].plot(V_range, odd(G_LL), linewidth=2, label=r'$G_{LL}^A$')
axes[1].plot(V_range, odd(G_LR), linewidth=2, label=r'$G_{LR}^A$')
axes[1].plot(V_range, odd(G_LL) + odd(G_LR), linewidth=2, label=r'$G_{LL}^A+G_{LR}^A$')
myf.style_axis(axes[1], 'Symmetric bias: odd parts')
for ax in axes:
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.5)
    ax.legend()
plt.tight_layout()
plt.show()
myf.print_params(p)

#%% --- PHASE-DEPENDENT TRANSMISSION AT FIXED ENERGY ---
p_phi = myf.Params(
    nx=10, ny=40, t_n=1.0, mu_n=0.5, t_c=1.00, mu_c=-0.1, t_s=1.0, mu_s=0.8,
    delta=0.35, phi=np.pi,
    alpha=0.2, beta=0.0, Bz=0.4, Bxy=0.0, theta_z=0.0, eta=1e-3, kT=5e-3,
)
phi_values = np.linspace(0, 2 * np.pi, 41)
E_fixed, side = 0.0, 'right'

T_phi = myf.transmissions_vs_phi(myf.FourTerminalJunction(p_phi), phi_values, E_fixed, side=side)
fig, axes = myf.plot_channels_vs_phi(phi_values, T_phi)
fig.suptitle(fr'Transmission channels at fixed $E={E_fixed:.2f}$, lead={side}')
plt.tight_layout()
plt.show()
myf.print_params(p_phi)

#%% --- PHASE / BIAS CONDUCTANCE MAP ---
pc = myf.Params(
    nx=10, ny=40, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.00, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi,
    alpha=0.0, beta=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0, eta=1e-5, kT=5e-2,
)
phi_vals = np.linspace(-np.pi, np.pi, 41)
V_bias_map = np.linspace(-2.5 * pc.delta, 2.5 * pc.delta, 41)
E_map = myf.energy_grid(V_bias_map.max(), pc.kT, 81)

T_map = myf.transmissions_vs_phi(myf.FourTerminalJunction(pc), phi_vals, E_map, side='left')  # (N_phi, N_E)
maps = {}
for scheme, _ in SCHEMES:
    VL, VR = myf.bias_line(V_bias_map, scheme)
    maps[scheme] = myf.conductance(T_map, E_map, VL, VR, pc.kT)          # (G_LL, G_LR), (N_phi, N_V)

panels = [
    (maps['sym'][0]['total'], r"$G_{LL}=\partial I_L/\partial V_L$ (symmetric baseline)"),
    (maps['sym'][1]['total'], r"$G_{LR}=\partial I_L/\partial V_R$ (symmetric baseline)"),
    (maps['anti'][0]['total'], r"$G_{LL}=\partial I_L/\partial V_L$ (antisymmetric baseline)"),
    (maps['anti'][1]['total'], r"$G_{LR}=\partial I_L/\partial V_R$ (antisymmetric baseline)"),
    (maps['sym'][0]['total'] + maps['sym'][1]['total'], r"$dI_L/dV$ along symmetric line"),
    (maps['anti'][0]['total'] - maps['anti'][1]['total'], r"$dI_L/dV$ along antisymmetric line"),
]
fig, axs = plt.subplots(3, 2, figsize=(16, 12))
for ax, (Gmap, title) in zip(axs.flat, panels):
    cf = ax.contourf(phi_vals / np.pi, V_bias_map / pc.delta, Gmap.T, levels=100, cmap="inferno")
    fig.colorbar(cf, ax=ax, label=r"$e^2/h$")
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$V_{\rm bias}/\Delta$")
    myf.style_axis(ax, title)
plt.tight_layout()
plt.show()
myf.print_params(pc)

#%% --- PARAMETER SCAN ---
base = myf.Params(nx=7, ny=1, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.0, t_s=1.0, mu_s=1.0,
                  delta=0.35, eta=1e-4, kT=5e-2)
phi_vals_scan = np.linspace(-np.pi, np.pi, 21)
V_bias_scan = np.linspace(-1.1 * base.delta, 1.1 * base.delta, 31)
E_scan = myf.energy_grid(V_bias_scan.max(), base.kT, 61)

var1_name, var1_vals = 'mu_s', np.array([0.0, 1.0])
var2_name, var2_vals = 't_c', np.array([0.4, 0.9])
# G_LR(phi, V) along the symmetric line for every (var1, var2) -> shape (N1, N2, N_phi, N_V)
G_LR_maps = myf.sweep(base, lambda j: myf.phase_bias_map(j, E_scan, phi_vals_scan, V_bias_scan, 'sym')[1]['total'],
                      **{var1_name: var1_vals, var2_name: var2_vals})

myf.plot_grid_thumbnails(G_LR_maps, var1_name, var1_vals, var2_name, var2_vals, phi_vals_scan, V_bias_scan, base.delta)
myf.plot_summary_trends(G_LR_maps, var1_name, var1_vals, var2_name, var2_vals)
plt.show()

#%% --- PHASE-DEPENDENT TRANSMISSION, EXTENDED (finer phi) ---
pT = myf.Params(
    nx=20, ny=10, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=0.0, eta=2e-5, kT=2e-5,
)
E_fixed = 0.0
phi_vals_ext = np.linspace(0.0, 2 * np.pi, 81)
T_ext = myf.transmissions_vs_phi(myf.FourTerminalJunction(pT), phi_vals_ext, E_fixed, side='right')

fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax[0].plot(phi_vals_ext / np.pi, T_ext['eh_cross'], label=r"$T_{eh}$ (CAR)")
ax[0].plot(phi_vals_ext / np.pi, T_ext['he_cross'], label=r"$T_{he}$ (CAR)")
ax[0].set_title(fr'Transmissions at fixed E={E_fixed:.2f} vs. $\phi$ (lead=right)')
ax[1].plot(phi_vals_ext / np.pi, T_ext['eh_local'], label=r"$T_{eh}$ (LAR)")
ax[1].set_xlabel(r"$\phi / \pi$")
for a in ax:
    a.set_ylabel("Transmission")
    a.legend()
    a.grid(True)
plt.tight_layout()
plt.show()

#%% --- 2D SC RIBBONS vs. 1D SC CHAINS ---
p_1d2d = myf.Params(
    nx=10, ny=5, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi / 4, eta=2e-5, kT=5e-6,
)
phi_values = np.linspace(0, 2 * np.pi, 301)
E_fixed, side = 0.0, 'left'

T_ribbon = myf.transmissions_vs_phi(myf.FourTerminalJunction(p_1d2d, 'ribbon'), phi_values, E_fixed, side=side)
T_chain = myf.transmissions_vs_phi(myf.FourTerminalJunction(p_1d2d, 'chain'), phi_values, E_fixed, side=side)
fig, axes = myf.plot_channels_vs_phi(phi_values, T_ribbon, label='(2D ribbon)')
myf.plot_channels_vs_phi(phi_values, T_chain, axes=axes, label='(1D chains)', linestyle='--')
fig.suptitle(fr'Ribbon vs. chain leads at fixed $E={E_fixed:.2f}$, lead={side}')
plt.tight_layout()
plt.show()
# %%
