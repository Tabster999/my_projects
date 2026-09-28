"""
plot_thermal_map.py -- T -> 0 thermal / nonlocal electrical maps for arbitrary parameter sets
==============================================================================================

Edit only the three cells PARAMETERS, SCAN DEFINITION and VARIANTS.
Everything else lives in modules.analysis:
    thermal_scan      map over (x, y) + 1D cuts          (any Params field or 'E' as axis)
    plot_thermal_scan T_th map, T_el map, cuts; gap-closing overlay if axes are (mu_s, Bz_s)
    save_scan / load_scan   .npz with arrays + parameters, re-plot without recomputing
    lead_channels     open channels of the L/R normal leads (0 -> every T is ~0)

T_th = T^ee_RL + T^he_RL = kappa/kappa0,  T_el = T^ee_RL - T^he_RL = G/G0  (Gresta Eqs. 9-10).
"""
#%% --- IMPORTS ---
import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\coding\my_projects\thesis\thermal_transport')  # set working dir to this file's folder
from modules.params import Params
from modules.analysis import (thermal_scan, plot_thermal_scan, save_scan, load_scan,
                              lead_channels, LeadsClosedError)

#%% --- PARAMETERS (base) ---
base = Params(
    nx=20, ny=8,
    t_n=1.0, t_c=1.0, t_s=1.0,
    mu_n=1.0, mu_c=1.0, mu_s=1.0,
    delta=0.35, phi=np.pi,
    alpha=0.0, beta=0.0,
    Bz=0.0, Bz_s=0.0,                        # Zeeman only in T/B
    soc_in_sc=True, soc_bonds=True,          # SOC everywhere, bonds carry SOC
    tc_top=1.0, tc_bot=1.0, tc_barr=1.0,     # tc = t_c -> no barrier
    eta=1e-5, tol=1e-14, max_iter=450,
)

#%% --- SCAN DEFINITION ---
x = ('mu_s', np.arange(-base.delta, base.delta, 0.07))         # map x-axis: any Params field, or 'E'
y = ('Bz_s', np.arange(-4.0, 4.0, 0.25))         # map y-axis
E = 0.0                                          # energy for axes that are not 'E'

cut_axis = ('mu_c', np.arange(0.0, 0.15, 0.005)) # swept in every cut, rest = base + overrides
cuts = [
    dict(phi=np.pi),                   # reference
    dict(phi=0.0),                     # phase control
]

#%% --- VARIANTS: one scan + figure per entry, overrides applied on top of base ---
variants = {
    'base': {},
    # 'mu_n=0.2': dict(mu_n=0.2),                # opens the leads at ny = 8
    # 'ny=9':     dict(ny=9),                    # keeps mu_n = 0.1, one row wider
}
OUT = 'maps'
os.makedirs(OUT, exist_ok=True)

#%% --- RUN ---
scans = {}
for tag, over in variants.items():
    p = replace(base, **over)
    n_ch, bottom = lead_channels(p, E)
    print(f"[{tag}] normal leads: {n_ch} open channels at E={E} "
          f"(lowest subband bottom {bottom:+.4f} relative to E)")
    try:
        scan = thermal_scan(p, x, y, cuts=cuts, cut_axis=cut_axis, E=E)
    except LeadsClosedError as err:
        print(f"[{tag}] skipped: {err}")
        continue
    fname = os.path.join(OUT, ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in tag))
    save_scan(fname + '.npz', scan)
    plot_thermal_scan(scan, savepath=fname + '.png')
    scans[tag] = scan
plt.show()

#%% --- RE-PLOT A SAVED SCAN (no recomputation) ---
scan = load_scan(os.path.join(OUT, 'base.npz'))
plot_thermal_scan(scan)
plt.show()
# %%