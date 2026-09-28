"""Higher-level scans built on top of FourTerminalJunction.channels()."""

from ast import Param
import json
import warnings
from dataclasses import replace, asdict

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import block_diag
from .hamiltonians import onsite_block, Vx, Vy, make_row_hamiltonian
from .params import Params
from .fast_phase_sweep import FastPhaseSweep
from .junction import FourTerminalJunction, normal_lead_blocks
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


SC_ONLY_PARAMS = {'mu_s', 'Bz_s', 'delta', 'phi', 't_s', 'tc_top', 'tc_bot'}


def thermal_map(base_params, name1, vals1, name2, vals2, E=0.0):
    """
    T -> 0 maps T_th(name1, name2) and T_el(name1, name2) at energy E (Gresta Fig. 8a analog).
    Returns arrays of shape (len(vals1), len(vals2)).  Normal-lead self-energies are computed
    once if both scanned parameters only enter the SC ribbons (they are phi / mu_s / Bz_s blind).
    """
    reuse = {name1, name2} <= SC_ONLY_PARAMS
    sig_n = None
    if reuse:
        J0 = FourTerminalJunction(base_params)
        zN = (E + 1j * base_params.eta) * np.eye(4 * base_params.ny, dtype=complex)[None]
        sig_n = (J0.lead_L.self_energy(zN)[0], J0.lead_R.self_energy(zN)[0])

    T_th = np.empty((len(vals1), len(vals2)))
    T_el = np.empty_like(T_th)
    for i, v1 in enumerate(tqdm(vals1, desc=f'{name1} sweep')):
        for j, v2 in enumerate(vals2):
            pc = replace(base_params, **{name1: v1, name2: v2})
            T_th[i, j], T_el[i, j] = FourTerminalJunction(pc).thermal_T0(E, precompute_sigmas_n=sig_n)
    return T_th, T_el



# =====================================================================================
#  T -> 0 thermal scans: lead check, grids (any Params field or 'E'), cuts, plot, save/load
# =====================================================================================

PARAM_LABELS = {
    'mu_s': r'$\mu_s/t$', 'mu_c': r'$\mu_c/t$', 'mu_n': r'$\mu_n/t$',
    'Bz_s': r'$B_{z,s}/t$', 'Bz': r'$B_z/t$', 'delta': r'$\Delta/t$',
    'alpha': r'$\alpha/t$', 'beta': r'$\beta/t$', 'phi': r'$\phi/\pi$', 'E': r'$E/t$',
    'nx': r'$n_x$', 'ny': r'$n_y$', 'tc_top': r'$t_{c,T}/t$', 'tc_bot': r'$t_{c,B}/t$',
    'tc_barr': r'$t_{c,barr}/t$',
}


def _label(name):
    return PARAM_LABELS.get(name, name)


def _axis(name, vals):
    vals = np.asarray(vals, dtype=float)
    return vals / np.pi if name == 'phi' else vals


def _fmt(name, v):
    return rf'$\phi={v / np.pi:g}\pi$' if name == 'phi' else f'{name}={v:g}'


class LeadsClosedError(RuntimeError):
    """The L/R normal leads have no propagating electron channel -> every transmission is ~0."""


def lead_channels(p, E=0.0, Nk=2001):
    """
    Open electron channels of the L/R normal lead at energy E, counted from its Bloch bands.
    Delta = 0 in the lead, so the electron block decouples: H_e(kx) = H0 + V e^{ikx} + V^dag e^{-ikx}.
    Each band crossing E twice over kx in [-pi, pi) is one channel.
    Returns (n_channels, bottom of the lowest subband relative to E).
    """
    _, H0, V = normal_lead_blocks(p)          # same blocks the junction uses
    e = np.where(np.tile([1, 1, 0, 0], p.ny))[0]
    H0, V = H0[np.ix_(e, e)], V[np.ix_(e, e)]
    ph = np.exp(1j * np.linspace(-np.pi, np.pi, Nk))[:, None, None]
    ev = np.linalg.eigvalsh(H0[None] + V[None] * ph + V.conj().T[None] / ph) - E
    n_cross = np.sum(np.abs(np.diff(np.sign(ev), axis=0)) > 0)
    return int(n_cross // 2), float(ev.min())


def check_leads(p, energies=(0.0,)):
    """Raise LeadsClosedError if the leads are closed at every energy, warn if closed at some."""
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    n = np.array([lead_channels(p, E)[0] for E in energies])
    if np.all(n == 0):
        _, bottom = lead_channels(p, energies[0])
        raise LeadsClosedError(
            f"normal leads (ny={p.ny}, mu_n={p.mu_n}) have 0 open channels at all scanned E; at "
            f"E={energies[0]:g} the lowest subband sits {bottom:+.4f} above E -> raise mu_n or widen ny")
    if np.any(n == 0):
        warnings.warn(f"normal leads closed at {np.sum(n == 0)}/{len(n)} energies "
                      f"(E = {energies[n == 0]}) -> T = 0 there")
    return n


def gap_closing_lines(mu_s, p):
    """TRIM gap closings of the isolated SC lead: |Bz_s| = sqrt((mu_s - eps_K)^2 + Delta^2)."""
    shift = (p.alpha**2 + p.beta**2) / 4 if p.soc_in_sc else 0.0
    return {
        r'$\Gamma$': np.sqrt((mu_s - shift) ** 2 + p.delta**2),
        r'$X/Y$': np.sqrt((mu_s - shift - 4 * p.t_s) ** 2 + p.delta**2),
        r'$M$': np.sqrt((mu_s - shift - 8 * p.t_s) ** 2 + p.delta**2),
    }


def _normal_sigmas(p, E):
    zN = (E + 1j * p.eta) * np.eye(4 * p.ny, dtype=complex)[None]
    J = FourTerminalJunction(p)
    return J.lead_L.self_energy(zN)[0], J.lead_R.self_energy(zN)[0]


def thermal_grid(base, y, x=None, E=0.0):
    """
    T_th, T_el of shape (len(y_vals), len(x_vals)); x=None gives a 1D line (shape (len(y_vals), 1)).
    y, x : (name, values), name = any Params field or 'E'.
    Normal-lead self-energies are cached per energy when only SC-lead parameters are scanned.
    """
    yn, yv = y
    xn, xv = x if x is not None else (None, [None])
    lead_blind = ({yn, xn} - {'E', None}) <= SC_ONLY_PARAMS
    cache = {}
    T_th = np.empty((len(yv), len(xv)))
    T_el = np.empty_like(T_th)
    for i, yy in enumerate(tqdm(yv, desc=f'{yn} sweep', leave=False)):
        for j, xx in enumerate(xv):
            over = {yn: yy}
            if xn is not None:
                over[xn] = xx
            En = float(over.pop('E', E))
            pc = replace(base, **over)
            sig = None
            if lead_blind:
                if En not in cache:
                    cache[En] = _normal_sigmas(base, En)
                sig = cache[En]
            T_th[i, j], T_el[i, j] = FourTerminalJunction(pc).thermal_T0(En, precompute_sigmas_n=sig)
    return T_th, T_el


def thermal_scan(base, x, y, cuts=(), cut_axis=None, E=0.0, check=True):
    """
    One (x, y) map plus optional 1D cuts, all in the T -> 0 limit.

    x, y     : (name, values) -- any Params field, or 'E' for an energy axis
    cuts     : list of override dicts, each a separate line, e.g.
               [dict(phi=np.pi), dict(phi=0.0), dict(mu_c=0.1)]
    cut_axis : (name, values) swept in every cut; all other parameters = base + overrides
    E        : energy for every axis that is not 'E'
    check    : check the normal leads first (raises LeadsClosedError if closed everywhere)
    """
    x = (x[0], np.asarray(x[1]))
    y = (y[0], np.asarray(y[1]))
    cut_axis = None if cut_axis is None else (cut_axis[0], np.asarray(cut_axis[1]))

    def energies(axes):
        for name, vals in axes:
            if name == 'E':
                return vals
        return [E]

    if check:
        check_leads(base, energies([x, y]))
        for over in cuts:
            check_leads(replace(base, **over), energies([cut_axis]))

    T_th, T_el = thermal_grid(base, y, x, E)
    cut_res = []
    for over in cuts:
        th, el = thermal_grid(replace(base, **over), cut_axis, None, E)
        cut_res.append(dict(overrides=dict(over), T_th=th[:, 0], T_el=el[:, 0]))
    return dict(base=base, x=x, y=y, E=E, T_th=T_th, T_el=T_el, cut_axis=cut_axis, cuts=cut_res)


def plot_thermal_scan(scan, T_ref=0.5, savepath=None):
    """Panels: T_th map, T_el map, cuts (if any). Gap-closing overlay if the axes are (mu_s, Bz_s)."""
    p = scan['base']
    (xn, xv), (yn, yv) = scan['x'], scan['y']
    has_cuts = len(scan['cuts']) > 0
    fig, ax = plt.subplots(1, 3 if has_cuts else 2, figsize=(15.5 if has_cuts else 10.5, 4.4))

    fixed = []
    if 'phi' not in (xn, yn):
        fixed.append(rf'$\phi={p.phi / np.pi:g}\pi$')
    if 'E' not in (xn, yn):
        fixed.append(rf'$E={scan["E"]:g}$')
    overlay = {xn, yn} == {'mu_s', 'Bz_s'}

    for k, (Z, lab, cmap, vmin, vmax) in enumerate([
            (scan['T_th'], r'$T_{th}=\kappa/\kappa_0$', 'inferno', 0.0, 2.0),
            (scan['T_el'], r'$T_{el}=G/G_0$', 'RdBu_r', -1.5, 1.5)]):
        im = ax[k].pcolormesh(_axis(xn, xv), _axis(yn, yv), Z, shading='nearest',
                              cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[k], label=lab)
        if overlay:
            xl, yl = ax[k].get_xlim(), ax[k].get_ylim()
            mu_vals = xv if xn == 'mu_s' else yv
            mu_fine = np.linspace(np.min(mu_vals), np.max(mu_vals), 400)
            for (name, B), ls in zip(gap_closing_lines(mu_fine, p).items(), ['-', '--', ':']):
                for sgn in (+1, -1):
                    xs, ys = (mu_fine, sgn * B) if xn == 'mu_s' else (sgn * B, mu_fine)
                    ax[k].plot(xs, ys, color='c', ls=ls, lw=1.1,
                               label=f'gap closing at {name}' if sgn > 0 else None)
            ax[k].set_xlim(xl)
            ax[k].set_ylim(yl)
        ax[k].set_xlabel(_label(xn))
        ax[k].set_ylabel(_label(yn))
        ax[k].set_title(lab + '  at ' + ', '.join(fixed))
    if overlay:
        ax[0].legend(fontsize=7, loc='upper right')

    if has_cuts:
        cn, cv = scan['cut_axis']
        for c, col in zip(scan['cuts'], plt.cm.tab10.colors):
            lab = ', '.join(_fmt(k_, v_) for k_, v_ in c['overrides'].items()) or 'base'
            ax[2].plot(_axis(cn, cv), c['T_th'], color=col, label=lab)
            ax[2].plot(_axis(cn, cv), c['T_el'], color=col, ls='--', lw=1)
        ax[2].axhline(T_ref, color='k', lw=0.6)
        if cn == 'Bz_s':
            for B in gap_closing_lines(np.array(p.mu_s), p).values():
                for sgn in (+1, -1):
                    if np.min(cv) <= sgn * B <= np.max(cv):
                        ax[2].axvline(sgn * B, color='c', lw=0.8)
        at = [_fmt(n, getattr(p, n)) for n in (xn, yn) if n not in (cn, 'E')]
        ax[2].set_xlabel(_label(cn))
        ax[2].set_title(f'cut along {cn}' + (' at ' + ', '.join(at) if at else '')
                        + '\n(solid $T_{th}$, dashed $T_{el}$)', fontsize=10)
        ax[2].legend(fontsize=8)

    keys = [k_ for k_ in ('nx', 'ny', 'alpha', 'beta', 'delta', 'mu_c', 'mu_n', 'mu_s', 'Bz_s',
                          'tc_barr', 'tc_top', 'tc_bot') if k_ not in (xn, yn)]
    fig.suptitle(', '.join(f'{k_}={getattr(p, k_):g}' for k_ in keys), y=1.02)
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig, ax


def save_scan(path, scan):
    """Store a thermal_scan result (arrays + parameters as JSON) in one .npz."""
    to_json = lambda o: json.dumps(o, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
    meta = dict(base=asdict(scan['base']), x=scan['x'][0], y=scan['y'][0], E=scan['E'],
                cut_axis=None if scan['cut_axis'] is None else scan['cut_axis'][0],
                cuts=[c['overrides'] for c in scan['cuts']])
    arrays = dict(T_th=scan['T_th'], T_el=scan['T_el'], x_vals=scan['x'][1], y_vals=scan['y'][1])
    if scan['cut_axis'] is not None:
        arrays['cut_vals'] = scan['cut_axis'][1]
    for k_, c in enumerate(scan['cuts']):
        arrays[f'cut{k_}_T_th'] = c['T_th']
        arrays[f'cut{k_}_T_el'] = c['T_el']
    np.savez(path, meta=to_json(meta), **arrays)


def load_scan(path):
    """Inverse of save_scan -> same dict as thermal_scan returns (re-plot without recomputing)."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    cut_axis = None if meta['cut_axis'] is None else (meta['cut_axis'], d['cut_vals'])
    cuts = [dict(overrides=over, T_th=d[f'cut{k_}_T_th'], T_el=d[f'cut{k_}_T_el'])
            for k_, over in enumerate(meta['cuts'])]
    return dict(base=Params(**meta['base']), x=(meta['x'], d['x_vals']), y=(meta['y'], d['y_vals']),
                E=meta['E'], T_th=d['T_th'], T_el=d['T_el'], cut_axis=cut_axis, cuts=cuts)



# =====================================================================================
#  Andreev bound-state spectrum: DOS(E, phi) of the central region (HWBC, finite nx)
# =====================================================================================

def dos_map(base, E_vals, phi_vals, rows=None, phi_ref=0.0):
    """
    DOS(E, phi) = -(1/pi) Im Tr G^r(E, phi) of the central region -- Gresta Figs. 4(c,d)/5(c).

    One dense inverse per energy; every phase then follows from a Woodbury update on the
    top-ribbon block (dimension 4*nx), using the exact U(1) gauge rotation of Sigma_T
    (the same one FastPhaseSweep uses):

        A(phi) = A0 - P dSigma_T P^T,
        G(phi) = G0 + G0 P (1 - dSigma_T G0_TT)^{-1} dSigma_T P^T G0 .

    rows  : y-rows to trace over (None = whole central region).  Row 0 touches the TOP
            ribbon, row ny-1 the BOTTOM one, so rows=[ny-1] gives the bottom-interface DOS.
    Returns (len(phi_vals), len(E_vals)).
    """
    p = base
    nx, ny = p.nx, p.ny
    dim = 4 * nx * ny
    J = FourTerminalJunction(replace(p, phi=phi_ref))
    rows = range(ny) if rows is None else rows
    sel = np.concatenate([np.arange(4 * nx * iy, 4 * nx * (iy + 1)) for iy in rows])
    T = np.arange(4 * nx)                                   # top-ribbon block of the central region
    iL, iR = J.idx_L, J.idx_R
    eyeN = np.eye(4 * ny, dtype=complex)[None]
    eyeS = np.eye(4 * nx, dtype=complex)[None]
    Id_T = np.eye(4 * nx, dtype=complex)

    out = np.empty((len(phi_vals), len(E_vals)))
    for j, E in enumerate(tqdm(E_vals, desc='DOS(E, phi)', leave=False)):
        z = E + 1j * p.eta
        SL, SR = J.lead_L.self_energy(z * eyeN)[0], J.lead_R.self_energy(z * eyeN)[0]
        ST0, SB = J.ribbon_top.self_energy(z * eyeS)[0], J.ribbon_bot.self_energy(z * eyeS)[0]
        A0 = -J.H_C.astype(complex)
        A0[np.diag_indices(dim)] += z
        A0[:4 * nx, :4 * nx] -= ST0
        A0[-4 * nx:, -4 * nx:] -= SB
        A0[iL[:, None], iL[None, :]] -= SL
        A0[iR[:, None], iR[None, :]] -= SR
        G0 = np.linalg.inv(A0)
        d0 = np.diag(G0)[sel]
        C = G0[sel][:, T]                                   # rows we trace over, top columns
        R = G0[T][:, sel]
        W = G0[T][:, T]
        for i, phi in enumerate(phi_vals):
            u = np.tile([1.0, 1.0, np.exp(1j * (phi - phi_ref)), np.exp(1j * (phi - phi_ref))], nx)
            dS = u.conj()[:, None] * ST0 * u[None, :] - ST0
            X = np.linalg.solve(Id_T - dS @ W, dS)
            d = d0 + np.einsum('ia,ab,bi->i', C, X, R, optimize=True)
            out[i, j] = -np.sum(d.imag) / np.pi
    return out



def ldos_map(base, E_vals, phi_vals, phi_ref=0.0):
    """
    Site-resolved LDOS(E, phi) of the central region -- Gresta Eq. (13):
        LDOS(E, x) = -(1/pi) Im Tr G^r_{x,x}(E),  trace over spin and particle-hole.
    All four leads are attached (L, R via Sigma_L/R, T/B via the ribbon self-energies),
    i.e. the same G^r that channels()/thermal_T0() use.

    One dense inverse per energy; every phase follows from a Woodbury update on the
    top-ribbon block (dimension 4*nx) with the exact U(1) gauge rotation of Sigma_T:
        A(phi) = A0 - P dSigma_T P^T
        G(phi) = G0 + G0 P (1 - dSigma_T G0_TT)^{-1} dSigma_T P^T G0
    so only 4*nx columns/rows of G0 and its diagonal are needed.

    Returns (len(phi_vals), len(E_vals), ny, nx); row iy=0 touches the TOP ribbon,
    iy=ny-1 the BOTTOM one.  Sum over the last two axes to get the total DOS.
    """
    p = base
    nx, ny = p.nx, p.ny
    dim = 4 * nx * ny
    J = FourTerminalJunction(replace(p, phi=phi_ref))
    T = np.arange(4 * nx)                                  # top-ribbon block
    iL, iR = J.idx_L, J.idx_R
    eyeN = np.eye(4 * ny, dtype=complex)[None]
    eyeS = np.eye(4 * nx, dtype=complex)[None]
    Id_T = np.eye(4 * nx, dtype=complex)
    out = np.empty((len(phi_vals), len(E_vals), ny, nx))

    for j, E in enumerate(tqdm(E_vals, desc='LDOS(E, phi)', leave=False)):
        z = E + 1j * p.eta
        SL, SR = J.lead_L.self_energy(z * eyeN)[0], J.lead_R.self_energy(z * eyeN)[0]
        ST0, SB = J.ribbon_top.self_energy(z * eyeS)[0], J.ribbon_bot.self_energy(z * eyeS)[0]
        A0 = -J.H_C.astype(complex)
        A0[np.diag_indices(dim)] += z
        A0[:4 * nx, :4 * nx] -= ST0
        A0[-4 * nx:, -4 * nx:] -= SB
        A0[iL[:, None], iL[None, :]] -= SL
        A0[iR[:, None], iR[None, :]] -= SR
        G0 = np.linalg.inv(A0)
        d0, C, R, W = np.diag(G0), G0[:, T], G0[T, :], G0[T][:, T]
        for i, phi in enumerate(phi_vals):
            u = np.tile([1.0, 1.0, np.exp(1j * (phi - phi_ref)), np.exp(1j * (phi - phi_ref))], nx)
            dS = u.conj()[:, None] * ST0 * u[None, :] - ST0
            X = np.linalg.solve(Id_T - dS @ W, dS)
            d = d0 + np.einsum('ia,ab,bi->i', C, X, R, optimize=True)
            out[i, j] = (-d.imag / np.pi).reshape(ny, nx, 4).sum(-1)
    return out