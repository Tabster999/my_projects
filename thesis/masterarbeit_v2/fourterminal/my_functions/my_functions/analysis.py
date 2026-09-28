"""Higher-level scans built on top of FourTerminalJunction.channels()."""

from ast import Param
from dataclasses import replace

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import tqdm
from .params import Params
from .fast_phase_sweep import FastPhaseSweep
from .junction import FourTerminalJunction
from .transport import partial_G_vectorized
from .plotting import style_axis

def compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map=1e-5, side='left', scheme='sym'):
    """For one parameter instance pc, sweep phi and return G_LL/G_LR(phi, V) maps."""
    junction0 = FourTerminalJunction(pc)
    G_LL_map = np.empty((len(phi_vals), len(V_bias_map)))
    G_LR_map = np.empty_like(G_LL_map)

    fps = FastPhaseSweep(junction0, E_sweep, phi_ref=0.0)
    for i, phi in enumerate(phi_vals):
        ch = fps.channels_at_phi(phi, side_name=side)
        G_LL_map[i], G_LR_map[i] = partial_G_vectorized(
            ch, E_sweep, V_bias_map, dV_map, pc.kT, scheme=scheme
        )

    return G_LL_map, G_LR_map


def summarize_map(G_LL_map, G_LR_map):
    """
    Summarize the non-local conductance map.
    Positive peak G_LR indicates CAR dominates, negative indicates EC dominates.
    """
    flat_GLR = G_LR_map.flatten()
    max_idx = np.argmax(np.abs(flat_GLR))
    peak_GLR = flat_GLR[max_idx]

    return {
        'mean_GLR': np.mean(G_LR_map),
        'peak_GLR': peak_GLR,
        'peak_GLL': np.max(np.abs(G_LL_map)),
    }


def scan_grid(base_params, name1, vals1, name2, vals2, E_sweep, phi_vals, V_bias_map, dV_map=1e-5, side='left', scheme='sym'):
    """2D parameter scan over (name1, name2); returns an object array of per-cell results."""
    results = np.empty((len(vals1), len(vals2)), dtype=object)
    for i, v1 in enumerate(tqdm(vals1, desc=f'{name1} sweep')):
        for j, v2 in enumerate(vals2):
            pc = replace(base_params, **{name1: v1, name2: v2})
            G_LL_map, G_LR_map = compute_phase_bias_maps(pc, E_sweep, phi_vals, V_bias_map, dV_map, side, scheme)
            results[i, j] = {
                'G_LL_map': G_LL_map,
                'G_LR_map': G_LR_map,
                'summary': summarize_map(G_LL_map, G_LR_map),
            }
    return results


def plot_grid_thumbnails(results, name1, vals1, name2, vals2, phi_vals, V_bias_map, delta):
    """Grid of small contour plots of the non-local conductance G_LR, one per (v1, v2) cell."""
    n1, n2 = len(vals1), len(vals2)
    fig, axs = plt.subplots(n1, n2, figsize=(2.6 * n2, 2.4 * n1), squeeze=False)

    for i in range(n1):
        for j in range(n2):
            ax = axs[i, j]
            G_LR = results[i, j]['G_LR_map']

            vmax = np.max(np.abs(G_LR))
            if vmax == 0:
                vmax = 1e-10  # avoid a singular norm

            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            ax.contourf(phi_vals / np.pi, V_bias_map / delta, G_LR.T, levels=40,
                        cmap='RdBu_r', norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{name2}={vals2[j]:.2g}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{name1}={vals1[i]:.2g}', fontsize=9)

    fig.suptitle(r'Non-local Conductance $G_{LR}$ across ' + f'{name1} x {name2} \n(Red = CAR, Blue = EC)', y=1.05)
    plt.tight_layout()
    return fig, axs


def plot_summary_trends(results, name1, vals1, name2, vals2):
    """Line plot of peak G_LR vs. name1, tracking the EC <-> CAR transition."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for j, v2 in enumerate(vals2):
        peak_glr = [results[i, j]['summary']['peak_GLR'] for i in range(len(vals1))]
        ax.plot(vals1, peak_glr, marker='o', label=f'{name2}={v2:.2g}')
    ax.set_xlabel(name1)
    ax.set_ylabel(r'Peak Non-local Conductance $G_{LR}$ ($e^2/h$)')
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.7)
    ylim = ax.get_ylim()
    ax.text(vals1[0], 0.15 * abs(ylim[1] - ylim[0]), 'CAR Dominates ($G_{LR} > 0$)',
            color='tab:red', fontsize=10, va='bottom')
    ax.text(vals1[0], -0.15 * abs(ylim[1] - ylim[0]), 'EC Dominates ($G_{LR} < 0$)',
            color='tab:blue', fontsize=10, va='top')
    ax.legend()
    style_axis(ax, r'Mechanism Tuning via $G_{LR}$')
    plt.tight_layout()
    return fig, ax
