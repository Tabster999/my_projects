#%% --- IMPORTS ---
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(r"c:\coding\my_projects\thesis\fourterminal")
import my_functions as myf

SCHEMES = [('sym', r'Symmetric bias ($V_L = V_R = V$)'),
           ('anti', r'Antisymmetric bias ($V_L = -V_R = V$)')]
CHANNEL_STYLE = [('EC', 'tab:blue'), ('CAR', 'tab:red'), ('LAR', 'tab:green')]


def mark_gap(ax, delta, vertical=True):
    line = ax.axvline if vertical else ax.axhline
    for s in (-1, 1):
        line(s * delta, color='gray', linestyle=':', linewidth=1)


#%% --- PARAMETERS & JUNCTION ---
# All four leads are attached without a barrier (lead/centre bond = the lead's own hopping).
p = myf.Params(
    nx=10, ny=20, t_n=1.0, mu_n=1.5, t_c=1.0, mu_c=1.0, t_s=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi,
    alpha=0.2, beta=0.0, Bz=0.2, Bxy=0.0, theta_z=0.0, eta=1e-4, kT=5e-3,
)
V = np.linspace(-1.5 * p.delta, 1.5 * p.delta, 61)       # bias values
E = myf.energy_grid(V.max(), p.kT, 121)                   # covers |V| + 8 kT; must resolve T(E)

junction = myf.FourTerminalJunction(p)
probes = {                                                # LDOS probe sites (ix, iy); iy = 0 borders the top SC
    'centre': junction.site(p.nx // 2, p.ny // 2),
    'next to L lead': junction.site(0, p.ny // 2),
    'next to top SC': junction.site(p.nx // 2, 0),
}
centre = probes['centre']
myf.print_params(p)

#%% --- TRANSMISSION COEFFICIENTS vs ENERGY ---
ch = junction.channels(E)                                 # {'left': {...}, 'right': {...}}
fig, axes = myf.plot_channels(E / p.delta, ch['left'], r'$E/\Delta$', label='(into L)')
myf.plot_channels(E / p.delta, ch['right'], r'$E/\Delta$', axes=axes, label='(into R)', linestyle='--')
for ax in axes.flat:
    mark_gap(ax, 1.0)
fig.suptitle(fr'Transmission channels, $\phi = {p.phi / np.pi:.2g}\pi$')
plt.tight_layout()
plt.show()

#%% --- TRANSMISSION COEFFICIENTS vs PHASE (fixed energy) ---
phi_vals = np.linspace(0, 2 * np.pi, 81)
E_fixed = 0.0
T_phi = myf.transmissions_vs_phi(junction, phi_vals, E_fixed, side='left')
fig, axes = myf.plot_channels_vs_phi(phi_vals, T_phi)
fig.suptitle(fr'Transmission channels into L at $E = {E_fixed:.2f}$')
plt.tight_layout()
plt.show()

#%% --- CONDUCTANCE (local / non-local, channel-resolved) ---
# With bias-independent T(E), G_LL depends only on V_L and G_LR only on V_R. The bias line only
# matters for the total derivative along it: dI_L/dV = G_LL(V) + G_LR(V) (sym), G_LL(V) - G_LR(-V) (anti).
G_sym = myf.conductance(ch['left'], E, *myf.bias_line(V, 'sym'), p.kT)     # (G_LL, G_LR) at V_L = V_R = V
G_anti = myf.conductance(ch['left'], E, *myf.bias_line(V, 'anti'), p.kT)   # at V_L = V, V_R = -V
panels = [
    (G_sym[0], r'$G_{LL} = \partial I_L/\partial V_L$ vs. $V_L$'),
    (G_sym[1], r'$G_{LR} = \partial I_L/\partial V_R$ vs. $V_R$'),
    ({k: G_sym[0][k] + G_sym[1][k] for k in G_sym[0]}, 'Total $dI_L/dV$, ' + SCHEMES[0][1]),
    ({k: G_anti[0][k] - G_anti[1][k] for k in G_anti[0]}, 'Total $dI_L/dV$, ' + SCHEMES[1][1]),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (G, title) in zip(axes.flat, panels):
    for key, color in CHANNEL_STYLE:
        ax.plot(V / p.delta, G[key], linewidth=2, color=color, label=key)
    ax.plot(V / p.delta, G['total'], linewidth=2, color='k', linestyle='--', alpha=.6, label='total')
    ax.axhline(0, color='k', linewidth=0.5)
    mark_gap(ax, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    myf.style_axis(ax, title)
for ax in axes[1, :]:
    ax.set_xlabel(r'$V/\Delta$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'Conductance ($e^2/h$)')
plt.tight_layout()
plt.show()

#%% --- LDOS: energy dependence at probe sites ---
rho = junction.ldos(E, sites=list(probes.values()))       # (N_E, n_probes)
fig, ax = plt.subplots(figsize=(8, 4.5))
for k, name in enumerate(probes):
    ax.plot(E / p.delta, rho[:, k], linewidth=2, label=name)
mark_gap(ax, 1.0)
ax.set_xlabel(r'$E/\Delta$')
ax.set_ylabel(r'LDOS $\rho(\mathbf{r}, E)$ (1/t)')
ax.legend()
ax.grid(alpha=0.3)
myf.style_axis(ax, 'Electron LDOS at probe sites')
plt.tight_layout()
plt.show()

#%% --- LDOS: real-space maps ---
E_real = np.array([0.0, 0.5 * p.delta])
rho_xy = junction.ldos(E_real)                            # (N_E, ny, nx), full inverse
fig, axes = plt.subplots(1, len(E_real), figsize=(6 * len(E_real), 5))
for ax, Ek, r in zip(np.atleast_1d(axes), E_real, rho_xy):
    myf.plot_map(ax, np.arange(p.nx), np.arange(p.ny), r.T, r'$i_x$', r'$i_y$',
                 fr'LDOS at $E = {Ek / p.delta:.2g}\Delta$', 'LDOS (1/t)')
    ax.invert_yaxis()                                     # top SC (iy = 0) on top
plt.tight_layout()
plt.show()

#%% --- LDOS: phase dependence at the centre ---
rho_phi = myf.ldos_vs_phi(junction, phi_vals, E, sites=[centre])[:, :, 0]    # (N_phi, N_E)
fig, ax = plt.subplots(figsize=(8, 5))
myf.plot_map(ax, phi_vals / np.pi, E / p.delta, rho_phi, r'$\phi/\pi$', r'$E/\Delta$',
             'LDOS at the centre vs. phase', 'LDOS (1/t)')
mark_gap(ax, 1.0, vertical=False)
plt.tight_layout()
plt.show()

#%% --- 2D MAPS: one parameter vs. bias (conductance) ---
mu_vals = np.linspace(0.0, 2.0, 21)


def G_sym(j):
    G_loc, G_nl = myf.bias_conductance(j, E, V, 'sym', side='left')
    return {'G_LL': G_loc['total'], 'G_LR': G_nl['total']}


res_mu = myf.sweep(p, G_sym, mu_c=mu_vals)                # dict of (N_mu, N_V)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
myf.plot_map(axes[0], mu_vals, V / p.delta, res_mu['G_LL'], r'$\mu_c$', r'$V/\Delta$',
             r'$G_{LL}$ (symmetric bias)', r'$e^2/h$')
myf.plot_map(axes[1], mu_vals, V / p.delta, res_mu['G_LR'], r'$\mu_c$', r'$V/\Delta$',
             r'$G_{LR}$ (symmetric bias; red = CAR, blue = EC)', r'$e^2/h$', diverging=True)
plt.tight_layout()
plt.show()

#%% --- 2D MAPS: one parameter vs. energy (transmissions + LDOS) ---
Bz_vals = np.linspace(0.0, 0.6, 21)


def T_and_ldos(j):
    T, rho = j.channels_and_ldos(E, sites=[centre], side='left')     # one factorisation for both
    return {**T, 'ldos': rho[:, 0]}


res_Bz = myf.sweep(p, T_and_ldos, Bz=Bz_vals)            # dict of (N_Bz, N_E)
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, key, title in [(axes[0], 'ee', r'$T_{ee}$ (EC)'), (axes[1], 'eh_cross', r'$T_{eh}$ (CAR)'),
                       (axes[2], 'ldos', 'LDOS at the centre')]:
    myf.plot_map(ax, Bz_vals, E / p.delta, res_Bz[key], r'$B_z$', r'$E/\Delta$', title)
    mark_gap(ax, 1.0, vertical=False)
plt.tight_layout()
plt.show()

#%% --- 2D MAPS: two parameters at zero bias / zero energy ---
Bz_grid = np.linspace(0.0, 0.6, 13)
alpha_grid = np.linspace(0.0, 0.4, 11)
E0 = myf.energy_grid(0.0, p.kT, 33)                       # only the Fermi window around V = 0 is needed


def zero_bias(j):
    G_loc, G_nl = myf.bias_conductance(j, E0, 0.0, 'sym', side='left')
    return {'G_LL': G_loc['total'], 'G_LR': G_nl['total'],
            'ldos': j.ldos(0.0, sites=[centre])[0, 0]}


res_2d = myf.sweep(p, zero_bias, Bz=Bz_grid, alpha=alpha_grid)   # dict of (N_Bz, N_alpha)
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
myf.plot_map(axes[0], Bz_grid, alpha_grid, res_2d['G_LL'], r'$B_z$', r'$\alpha$', r'$G_{LL}(V=0)$', r'$e^2/h$')
myf.plot_map(axes[1], Bz_grid, alpha_grid, res_2d['G_LR'], r'$B_z$', r'$\alpha$', r'$G_{LR}(V=0)$', r'$e^2/h$',
             diverging=True)
myf.plot_map(axes[2], Bz_grid, alpha_grid, res_2d['ldos'], r'$B_z$', r'$\alpha$', r'LDOS$(E=0)$ at the centre',
             'LDOS (1/t)')
plt.tight_layout()
plt.show()
# Any Params field works as a sweep axis, e.g. myf.sweep(p, zero_bias, mu_c=..., Bz=...) or phi=..., Bxy=...

# %%
