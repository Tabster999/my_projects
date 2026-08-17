#%% --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import replace
import os
os.chdir(r"c:\coding\my_projects\thesis\fourterminal")
import my_functions as myf
import time
#%% --- DEINFE CORE COMPUTATION FUNCTIONS --- 

def get_transmissions(params, phi_array, E_fixed=0.0, LR='left'):

    T_ee = np.zeros(len(phi_array))
    T_hh = np.zeros(len(phi_array))
    T_eh = np.zeros(len(phi_array))
    T_he = np.zeros(len(phi_array))
    T_lar_eh = np.zeros(len(phi_array))
    T_lar_he = np.zeros(len(phi_array))
    E0 = np.array([E_fixed])
    junction_phi = myf.FourTerminalJunction(params)
    fps = myf.FastPhaseSweep(junction_phi, E0, phi_ref=params.phi)
    for i, phi_val in enumerate(phi_array):
        ch = fps.channels_at_phi(phi_val, side_name=LR)

        T_ee[i]     = ch['ee'][0]           #type:ignore
        T_hh[i]     = ch['hh'][0]           #type:ignore
        T_eh[i]     = (ch['eh_cross'])[0]   #type:ignore
        T_he[i]     = (ch['he_cross'])[0]   #type:ignore
        T_lar_eh[i] = (ch['eh_local'])[0]   #type:ignore
        T_lar_he[i] = (ch['he_local'])[0]   #type:ignore

    return T_ee, T_hh, T_eh, T_he, T_lar_eh, T_lar_he
#%% --- BUILD JUNCTION & RUN CORE COMPUTATION ---
# eta should be larger than the energy step size, to smooth out numerical integration

p = myf.Params(
    nx=10, ny=5, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi / 4, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    eta=2e-5, kT=2e-3,
)

V_range = np.linspace(-1.5 * p.delta, 1.5 * p.delta, 121)
E_sweep = np.linspace(-1.5 * p.delta, 1.5 * p.delta, 1001)

print(f"Computing {len(E_sweep)} energy points for {p.nx} sites...")
junction = myf.FourTerminalJunction(p)
channel_sweeps = myf.FastPhaseSweep(junction, E_sweep=E_sweep, phi_ref=0.0).channels_at_phi(p.phi, 'left')
print("Finished compute.")
myf.print_params(p)

#%% --- CURRENTS (sym / anti bias) ---
# channel_sweeps['left'/'right']['ee'] etc. are the transmission channels vs. E_sweep
ch = channel_sweeps['left']
results = {'sym': {k: [] for k in ['EC', 'CAR', 'LAR', 'total']},
           'anti': {k: [] for k in ['EC', 'CAR', 'LAR', 'total']}}

for Vb in V_range:
    r_sym = myf.dc_current_channels(ch, E_sweep, {"left": Vb, "right": Vb}, p.kT, "left", "right")
    r_anti = myf.dc_current_channels(ch, E_sweep, {"left": Vb, "right": -Vb}, p.kT, "left", "right")
    for key in ['EC', 'CAR', 'LAR', 'total']:
        results['sym'][key].append(r_sym[key])
        results['anti'][key].append(r_anti[key])

fig_curr, axes_curr = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
schemes_info = [('sym', r'Symmetric bias ($\mu_L = \mu_R = +V$)'),
                ('anti', r'Antisymmetric bias ($\mu_L = +V,\ \mu_R = -V$)')]

for col, (scheme, title) in enumerate(schemes_info):
    ax = axes_curr[col]
    for key, lbl in [('EC', r'$I_{\rm EC}$'), ('CAR', r'$I_{\rm CAR}$'), ('LAR', r'$I_{\rm LAR}$')]:
        ax.plot(V_range, results[scheme][key], linewidth=2, label=lbl)
    ax.plot(V_range, results[scheme]['total'], linewidth=2, color='k', linestyle='--',
            label=r'$I_{\rm total}$', alpha=.5)
    ax.axvline(-p.delta, color='gray', linestyle=':')
    ax.axvline(p.delta, color='gray', linestyle=':')
    ax.set_xlabel(r'$V_{\rm bias}$')
    ax.grid(alpha=0.5)
    ax.legend(loc='upper left')
    myf.style_axis(ax, f"Currents - {title}")
    if col == 0:
        ax.set_ylabel(r'Current ($e/h$ units)')

plt.tight_layout()
plt.show()

#%% --- 2x2 CONDUCTANCE PLOTS ---
leads_names = ['left', 'right']
dV = 1e-5

G_channels = {'sym': {k: [] for k in ['EC', 'CAR', 'LAR', 'total']},
              'anti': {k: [] for k in ['EC', 'CAR', 'LAR', 'total']}}
G_matrix_elems = {'sym': {k: [] for k in ['G_LL', 'G_LR', 'G_RR', 'G_RL']},
                   'anti': {k: [] for k in ['G_LL', 'G_LR', 'G_RR', 'G_RL']}}

for Vb in V_range:
    for scheme in ['sym', 'anti']:
        sign = 1 if scheme == 'sym' else -1
        bias0 = {"left": Vb, "right": sign * Vb}
        b_plus = {"left": Vb + dV, "right": sign * (Vb + dV)}
        b_minus = {"left": Vb - dV, "right": sign * (Vb - dV)}

        G_mat = myf.conductance_matrix(bias0, leads_names, channel_sweeps, E_sweep, p.kT, dV=1e-5)
        for key, tup in [('G_LL', ('left', 'left')), ('G_LR', ('left', 'right')),
                          ('G_RR', ('right', 'right')), ('G_RL', ('right', 'left'))]:
            G_matrix_elems[scheme][key].append(G_mat[tup])

        r_plus = myf.dc_current_channels(channel_sweeps['left'], E_sweep, b_plus, p.kT, out_name="left", in_name="right")
        r_minus = myf.dc_current_channels(channel_sweeps['left'], E_sweep, b_minus, p.kT, out_name="left", in_name="right")
        for key in ['EC', 'CAR', 'LAR', 'total']:
            G_channels[scheme][key].append((r_plus[key] - r_minus[key]) / (2 * dV))

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for col, (scheme, title) in enumerate(schemes_info):
    ax_top = axes[0, col]
    for key, lbl in [('EC', r'$G_{\rm EC}$'), ('CAR', r'$G_{\rm CAR}$'), ('LAR', r'$G_{\rm LAR}$')]:
        ax_top.plot(V_range, G_channels[scheme][key], linewidth=2, label=lbl)
    ax_top.plot(V_range, G_channels[scheme]['total'], linewidth=2, color='k', linestyle='--',
                label=r'$G_{\rm total}$', alpha=.5)
    ax_top.axvline(-p.delta, color='gray', linestyle=':')
    ax_top.axvline(p.delta, color='gray', linestyle=':')
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc='upper right')
    myf.style_axis(ax_top, title)
    if col == 0:
        ax_top.set_ylabel(r'$dI_{\rm left}/dV_{\rm bias}$ (a.u.)')

    ax_bot = axes[1, col]
    G_LL = np.asarray(G_matrix_elems[scheme]['G_LL'])
    G_LR = np.asarray(G_matrix_elems[scheme]['G_LR'])
    ax_bot.plot(V_range, np.asarray(G_channels[scheme]['total']), linewidth=2, color='k', linestyle='--',
                label=r'Direct $dI_L/dV$', alpha=.5)
    ax_bot.plot(V_range, G_LL, linewidth=2, color='tab:green', label=r'$G_{LL}$', alpha=.7)
    ax_bot.plot(V_range, G_LR, linewidth=2, color='tab:gray', label=r'$G_{LR}$', alpha=.7)
    combo = (G_LL + G_LR) if scheme == 'sym' else (G_LL - G_LR)
    lbl = r'$(G_{LL}+G_{LR})$' if scheme == 'sym' else r'$(G_{LL}-G_{LR})$'
    ax_bot.plot(V_range, combo, linewidth=2, color='tab:orange', label=lbl, alpha=.5)
    ax_bot.axvline(-p.delta, color='gray', linestyle=':')
    ax_bot.axvline(p.delta, color='gray', linestyle=':')
    ax_bot.set_xlabel(r'$V_{\rm bias}$')
    ax_bot.grid(alpha=0.5)
    ax_bot.legend(loc='upper right')
    myf.style_axis(ax_bot, 'Comparison with total differential G')
    if col == 0:
        ax_bot.set_ylabel(r'Differential conductance ($e^2/h$)')

plt.tight_layout()
plt.show()
myf.print_params(p)

#%% --- PHASE-DEPENDENT TRANSMISSION AT FIXED ENERGY ---

p = myf.Params(
    nx=10, ny=5, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi / 4, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    eta=2e-5, kT=5e-6,
)

phi_values = np.linspace(0, 2 * np.pi, 301)
E_fixed = -0.00
dir = 'right'

#t0 = time.perf_counter()
T_ee, T_hh, T_eh, T_he, T_lar_eh, T_lar_he = get_transmissions(p, phi_values, E_fixed, LR=dir)
#t1 = time.perf_counter() - t0
#print(f"Computed {len(phi_values)} phase points in {t1:.2f} seconds.")

transmissions_list = [T_ee, T_eh, T_he, T_lar_eh, T_lar_he]
panels = [
    (T_ee,     r'EC ($e\to e$)',        r'$T_{ee}$'),
    (T_hh,     r'EC ($h\to h$)',        r'$T_{hh}$'),
    (T_eh,     r'CAR ($e\to h$ cross)', r'$T_{eh}$'),
    (T_he,     r'CAR ($h\to e$ cross)', r'$T_{he}$'),
    (T_lar_eh, r'LAR ($e\to h$ local)', r'$T_{LAR,eh}$'),
    (T_lar_he, r'LAR ($h\to e$ local)', r'$T_{LAR,he}$'),
]

fig_phi, axes_phi = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for ax, (data, title, lbl) in zip(axes_phi.flat, panels):
    ax.plot(phi_values, data, linewidth=2, label=lbl)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right')
    myf.style_axis(ax, title)
for ax in axes_phi[1, :]:
    ax.set_xlabel(r'Phase difference $\phi$')
for ax in axes_phi[:, 0]:
    ax.set_ylabel(r'Transmission ($e^2/h$)')

fig_phi.suptitle(fr'Transmission channels at fixed $E={E_fixed:.2f}$, lead={dir}')
plt.tight_layout()
plt.show()

#%% --- SYMMETRIZATION TEST ---
G_LL = np.asarray(G_matrix_elems['sym']['G_LL'])
G_LR = np.asarray(G_matrix_elems['sym']['G_LR'])

# Even part: [G(V) + G(-V)]/2, symmetric around V=0
G_LL_S = 0.5 * (G_LL + G_LL[::-1])
G_LR_S = 0.5 * (G_LR + G_LR[::-1])
# Odd part: [G(V) - G(-V)]/2, antisymmetric around V=0
G_LL_A = 0.5 * (G_LL - G_LL[::-1])
G_LR_A = 0.5 * (G_LR - G_LR[::-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(V_range, G_LL_S, linewidth=2, label=r'$G_{LL}^S$')
axes[0].plot(V_range, G_LR_S, linewidth=2, label=r'$G_{LR}^S$')
axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
myf.style_axis(axes[0], 'Symmetric-bias: even parts')
axes[0].legend()
axes[0].set_xlabel(r'$V_{\rm bias}$')
axes[0].grid(alpha=0.5)

axes[1].plot(V_range, G_LL_A, linewidth=2, label=r'$G_{LL}^A$')
axes[1].plot(V_range, G_LR_A, linewidth=2, label=r'$G_{LR}^A$')
axes[1].plot(V_range, G_LL_A + G_LR_A, linewidth=2, label=r'$G_{LL}^A+G_{LR}^A != 0$')
axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
myf.style_axis(axes[1], r'Symmetric-bias: odd parts')
axes[1].legend()
axes[1].set_xlabel(r'$V_{\rm bias}$')
axes[1].grid(alpha=0.5)

plt.tight_layout()
plt.show()
myf.print_params(p)

#%% --- PHASE / BIAS CONDUCTANCE MAP ---
pc = myf.Params(
    nx=10, ny=5, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.00, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=0.0, tc_top=1.0, tc_bot=1.0, tc_barr=1.0,
    alpha=0.0, beta=0.0, Bz=0.0, Bxy=0.0, theta_z=0.0, eta=2e-5, kT=5e-2,
)

phi_vals = np.linspace(-np.pi, np.pi, 101)
V_bias_map = np.linspace(-2.5 * pc.delta, 2.5 * pc.delta, 101)
dV_map = 1e-5

junction0 = myf.FourTerminalJunction(pc)
z1, _ = junction0._z_batches(E_sweep)
precomputed_normal = (
    junction0.lead_L.self_energy(z1),
    junction0.lead_R.self_energy(z1),
)

G_LL_sym_map = np.empty((len(phi_vals), len(V_bias_map)))
G_LR_sym_map = np.empty_like(G_LL_sym_map)
G_LL_anti_map = np.empty_like(G_LL_sym_map)
G_LR_anti_map = np.empty_like(G_LL_sym_map)
G_sym_map = np.empty_like(G_LL_sym_map)
G_anti_map = np.empty_like(G_LL_sym_map)

for i, phi in enumerate(tqdm(phi_vals, desc="Phase sweep")):
    p_i = replace(pc, phi=phi)
    ch_i = myf.FourTerminalJunction(p_i).channels(E_sweep, side_name="left", precompute_sigmas_n=precomputed_normal)

    # 1. Partial conductances evaluated along the symmetric baseline (V_R = +V)
    G_LL_sym_map[i], G_LR_sym_map[i] = myf.partial_G_vectorized(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="sym")
    # 2. Partial conductances evaluated along the antisymmetric baseline (V_R = -V)
    G_LL_anti_map[i], G_LR_anti_map[i] = myf.partial_G_vectorized(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="anti")
    # 3. Total directional derivatives along the bias trajectories
    G_sym_map[i] = myf.total_dIdV_map(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="sym")
    G_anti_map[i] = myf.total_dIdV_map(ch_i, E_sweep, V_bias_map, dV_map, pc.kT, scheme="anti")

#%% --- PLOT 2D MAP ---
maps = [
    (G_LL_sym_map, r"$G_{LL}=\partial I_L/\partial V_L$ (Symmetric baseline)"),
    (G_LR_sym_map, r"$G_{LR}=\partial I_L/\partial V_R$ (Symmetric baseline)"),
    (G_LL_anti_map, r"$G_{LL}=\partial I_L/\partial V_L$ (Antisymmetric baseline)"),
    (G_LR_anti_map, r"$G_{LR}=\partial I_L/\partial V_R$ (Antisymmetric baseline)"),
    (G_sym_map, r"$dI_L/dV$ Total Symmetric Directional"),
    (G_anti_map, r"$dI_L/dV$ Total Antisymmetric Directional"),
]

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
for ax, (G, title) in zip(axs.flat, maps):
    cf = ax.contourf(phi_vals / np.pi, V_bias_map / pc.delta, G.T, levels=100, cmap="inferno")
    fig.colorbar(cf, ax=ax, label=r"$e^2/h$")
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$V_{\rm bias}/\Delta$")
    myf.style_axis(ax, title)
plt.tight_layout()
plt.show()
myf.print_params(pc)

#%% --- PARAMETER SCAN ---
base = myf.Params(nx=7, t_n=1.0, mu_n=1.50, t_c=1.0, mu_c=1.0, t_s=1.0, mu_s=1.0,
              delta=0.35, tc_top=1.0, tc_bot=1.0, tc_barr=1.0, eta=1e-4, kT=5e-2)

phi_vals_scan = np.linspace(-np.pi, np.pi, 21)
V_bias_scan = np.linspace(-1.1 * base.delta, 1.1 * base.delta, 31)

var1_name, var1_vals = 'mu_s', np.array([0.0, 1.0])
var2_name, var2_vals = 't_c', np.array([0.4, 0.9])

scan_results = myf.scan_grid(base, var1_name, var1_vals, var2_name, var2_vals, E_sweep, phi_vals_scan, V_bias_scan)

# Grid of small heatmaps of the non-local conductance G_LR
myf.plot_grid_thumbnails(scan_results, var1_name, var1_vals, var2_name, var2_vals, phi_vals_scan, V_bias_scan, base.delta)

# Peak G_LR vs. var1 (zero crossing = EC-to-CAR transition)
myf.plot_summary_trends(scan_results, var1_name, var1_vals, var2_name, var2_vals)

plt.show()

#%% --- PHASE-DEPENDENT TRANSMISSION, EXTENDED (finer ny, wider phi sweep) ---
pT = myf.Params(
    nx=20, ny=10, t_n=1.0, mu_n=1.50, t_c=1.00, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=0.0, tc_top=1.00, tc_bot=1.00, tc_barr=1.00,
    eta=2e-5, kT=2e-5,
)

T_ee, T_eh, T_he, T_lar_eh, T_lar_he = [], [], [], [], []
E_fixed = np.array([0.0])
phi_vals_ext = np.linspace(0.0, 2 * np.pi, 81)

for phi_val in phi_vals_ext:
    p_phi = replace(pT, phi=phi_val)
    junction_phi = myf.FourTerminalJunction(p_phi)
    ch_left = junction_phi.channels(E_fixed)['right']

    T_ee.append(ch_left['ee'])
    T_eh.append(ch_left['eh_cross'])
    T_he.append(ch_left['he_cross'])
    T_lar_eh.append(ch_left['eh_local'])
    T_lar_he.append(ch_left['he_local'])

fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax[0].plot(phi_vals_ext / np.pi, T_eh, label=r"$T_{eh}$ (CAR)")
ax[0].set_title(fr'Transmissions at fixed E={E_fixed[0]:.2f} vs. $\phi$')
ax[0].plot(phi_vals_ext / np.pi, T_he, label=r"$T_{he}$ (CAR)")
ax[0].set_ylabel("Transmission")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(phi_vals_ext / np.pi, T_lar_eh, label=r"$T_{eh}$ (LAR)")
ax[1].set_xlabel(r"$\phi / \pi$")
ax[1].set_ylabel("Transmission")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# %%
