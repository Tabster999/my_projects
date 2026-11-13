"""
Slide figures for the November 2026 advisor meeting (four-terminal Scharf junction, beyond the papers).
One PNG per slide, sized for a 16:9 slide: FULL fills the content area under a title (12.5 x 5.6 in);
insert at 100 % and the fonts are ~14-16 pt.  Same colours on every slide: kappa black, |G| red,
local Andreev blue.

Everything is read from existing caches in plot_npz/ (nothing is recomputed) except the E(k) band of
slide 6, which is diagonalised here once (~20 s) and cached as plot_npz/slides_band_EZ0.8.npz.
Delta_top on slide 6 is min(grid gap, refined outer dip): the cached grid gap misses the narrow dip
(up to ~20 % too large, see CLAUDE.md, oscillation section), the k ~ 0 minimum is wide and resolved.
Output: plots/slides/slide<N>_<what>.png
"""
# %% --- IMPORTS AND STYLE ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tools as st
from study_oscillation import EZ as EZ_O, HBAR_VF, CASE, fit_A, zeros_of
from study_oscillation import EZ_F, smooth_gamma          # part fingerprint (EZ there = EZ_F here)

OUT = st.HERE / 'plots' / 'slides'
FULL = (12.5, 5.6)
L_NM = 2000.0
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#9a9893', '#e4e3df'
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
GAPLESS_COL = '#f5c9b3'                          # Q = -1 but Delta_top = 0 (slide9)
GAPLESS_UEV = 0.01                               # mixed deck maps: Delta_top is < 1e-3 or > 0.08 ueV, nothing between
plt.rcParams.update({
    'font.size': 15, 'axes.labelsize': 16, 'axes.titlesize': 16, 'legend.fontsize': 14,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'lines.linewidth': 2.2,
    'savefig.dpi': 200, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def load(name):
    """(values, {axis name: axis}) of a run_scan cache."""
    d = np.load(st.DATA / f'{name}.npz', allow_pickle=True)
    return d['values'], {k[5:]: d[k] for k in d.files if k.startswith('axis_')}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png')
    plt.close(fig)
    print('saved', OUT / f'{name}.png')


def half_line(ax):
    ax.axhline(0.5, color=INK2, lw=1, ls='--', zorder=1)


# %% --- SLIDE 4: kappa = 1/2 are resonances, neutral; Andreev high where kappa is low ---
def slide4():
    V, ax_ = load('scharf_absF_tr_L2')                     # W 100, L 2 um, a 20, tc_barr 0.2, phi pi
    EZ = ax_['E_Z']
    fig, ax = plt.subplots(figsize=FULL, layout='constrained')
    ax.plot(EZ, V[:, 2] / 2, color=BLUE, lw=1.8, ls=(0, (1, 1.2)), label=r'local Andreev  [$2e^2/h$]')
    ax.plot(EZ, V[:, 0], color=INK, label=r'thermal  $\kappa/\kappa_0$')
    ax.plot(EZ, np.abs(V[:, 1]), color=RED, lw=1.6, label=r'charge  $|G|/G_0$')
    half_line(ax)
    ax.set_xlim(EZ[0], EZ[-1]); ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(r'Zeeman energy $E_Z$ [meV]'); ax.set_ylabel('transport (normalised)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, 'slide4_resonances')


# %% --- SLIDE 5: eps oscillates; eps = 0 -> kappa = 0, eps = Gamma -> kappa = 1/2 ---
def slide5():
    EZ = EZ_F
    eps = load('scharf_absF_box_L2')[0][:, 0]              # ueV, closed box
    V = load('scharf_absF_tr_L2')[0]                       # NEGF, same fields
    ev, dip = load('scharf_absO_evanescent_W100')[0], load('scharf_absO_dip_W100')[0]
    xi, k_osc = np.interp(EZ, EZ_O, ev[:, 0]), np.interp(EZ, EZ_O, ev[:, 1])
    env = np.interp(EZ, EZ_O, dip[:, 1]) * np.exp(-L_NM / xi) * 1e3
    A, ph, rms = fit_A(eps, env, k_osc, L_NM)
    Gam = smooth_gamma(eps, V[:, 3])
    z = zeros_of(EZ, eps, half=0.03)
    print(f'slide 5: A = {A:.2f}, phi0 = {ph / np.pi:.2f} pi, rms {rms:.2f} ueV ({rms / eps.max():.1%}), '
          f'Gamma {Gam.min():.2f}-{Gam.max():.2f} ueV, zeros {np.round(z, 3)}')

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FULL, sharex=True, layout='constrained', height_ratios=(1.2, 1))
    a1.plot(EZ, A * env, color=MUTED, lw=1.3, ls='--', label=r'envelope $\propto e^{-L/\xi_M}$')
    a1.plot(EZ, A * env * np.abs(np.cos(k_osc * L_NM + ph)), color=ORANGE, lw=2,
            label=r'$A\,\Delta_{out}\,e^{-L/\xi_M}\,|\cos(k_{osc}L+\varphi_0)|$')
    a1.plot(EZ[::2], eps[::2], 'o', ms=4.5, color=BLUE, mec='white', mew=0.5, label=r'splitting $\varepsilon$')
    a1.plot(EZ, Gam, color=AQUA, lw=2.4, label=r'probe broadening $\Gamma$')
    a1.set_ylabel(r'energy [$\mu$eV]'); a1.set_ylim(0, None)
    a1.legend(loc='upper left', ncol=2, fontsize=13)
    a2.plot(EZ, V[:, 0], color=INK, label=r'$\kappa/\kappa_0$')
    a2.plot(EZ, np.abs(V[:, 1]), color=RED, lw=1.4, label=r'$|G|/G_0$')
    half_line(a2)
    a2.set_ylim(0, 0.6); a2.set_yticks([0, 0.25, 0.5])
    a2.set_ylabel(r'$\kappa/\kappa_0$,  $|G|/G_0$')
    a2.set_xlabel(r'Zeeman energy $E_Z$ [meV]'); a2.set_xlim(EZ[0], EZ[-1])
    a2.legend(loc='upper left', ncol=2, fontsize=13)
    for x in z:
        for a in (a1, a2):
            a.axvline(x, color=MUTED, lw=1, ls=':', zorder=1)
    save(fig, 'slide5_oscillation')


# %% --- SLIDE 6: the lengths -- band with the outer dip, which k, which xi ---
def band(EZ_val=0.8, k0=None, half=0.006, cache='slides_band_EZ0.8', q=None):
    """Three lowest positive BdG levels (ueV) of the cross-section on a coarse k grid and around k0 (q: CASE)."""
    q = CASE if q is None else q
    f = st.DATA / f'{cache}.npz'
    if f.exists():
        d = np.load(f)
        return d['k'], d['E'], d['kz'], d['Ez']
    p, nS = q.params(E_Z=EZ_val, phi=np.pi), st.n_S_of(q)
    lev = lambda kn: np.sort(np.abs(np.linalg.eigvalsh(st.H_cross(p, kn * q.a, nS))))[::2][:3] * q.t * 1e3
    k = np.linspace(0.0, 0.07, 701)
    kz = np.linspace(k0 - half, k0 + half, 241)
    E, Ez = np.array([lev(x) for x in k]), np.array([lev(x) for x in kz])
    np.savez(f, k=k, E=E, kz=kz, Ez=Ez)
    return k, E, kz, Ez


def slide6(EZ_val=0.8):
    ev, dip = load('scharf_absO_evanescent_W100')[0], load('scharf_absO_dip_W100')[0]
    xi, k_osc, D_grid, Q, k_gap = ev.T
    k0, D_out, hv_A, xi_hyp, xi_poly = dip.T
    topo = Q < 0
    D_top = np.minimum(D_grid, D_out)                     # the grid misses the narrow outer dip
    k_glob = np.where(D_grid < D_out, k_gap, k0)
    j = int(np.argmin(np.abs(EZ_O - EZ_val)))
    k, E, kz, Ez = band(EZ_val, k0=k0[j])
    print(f'slide 6 at E_Z {EZ_O[j]}: k0 {k0[j]:.4f}, k_osc {k_osc[j]:.4f} 1/nm, Delta_out {D_out[j] * 1e3:.1f} ueV, '
          f'hbar v_A {hv_A[j]:.1f} meV nm, xi_hyp {xi_hyp[j]:.0f}, xi_M {xi[j]:.0f} nm, '
          f'naive {HBAR_VF / D_top[j]:.0f} nm; max |k0 - k_osc| {np.max(np.abs(k0 - k_osc)[topo]):.5f} 1/nm')

    fig, (a, b, c) = plt.subplots(1, 3, figsize=FULL, layout='constrained', width_ratios=(1.2, 1, 1))
    # (a) the band
    a.plot(k, E[:, 0], color=INK, lw=2)
    a.plot(k0[j], D_out[j] * 1e3, 'o', color=BLUE, ms=8, zorder=5)
    a.annotate(r'outer dip $k_0$:' + '\n' + r'$\Delta_{out}$, velocity $v_A$', xy=(k0[j], D_out[j] * 1e3),
               xytext=(0.012, 25), color=BLUE, fontsize=14, arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    a.set_xlim(0, 0.07); a.set_ylim(0, 260)
    a.set_xlabel(r'$k$ along the junction [1/nm]'); a.set_ylabel(r'$E$ [$\mu$eV]')
    a.set_title(rf'lowest Andreev band, $E_Z$ = {EZ_val:g} meV', fontsize=15)
    ins = a.inset_axes([0.64, 0.06, 0.34, 0.44])                 # empty corner: the band is above 260 there
    ins.plot(kz, Ez[:, 0], color=INK, lw=2)
    dk = kz - k0[j]
    ins.plot(kz, np.sqrt((D_out[j] * 1e3) ** 2 + (hv_A[j] * 1e3 * dk) ** 2), color=ORANGE, lw=1.6, ls='--')
    ins.plot(k0[j], D_out[j] * 1e3, 'o', color=BLUE, ms=6, zorder=5)
    ins.set_xlim(kz[0], kz[-1]); ins.set_ylim(0, 4 * D_out[j] * 1e3)
    ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    a.text(0.81, 0.525,r'$\sqrt{\Delta_{out}^2 + (\hbar v_A\,\delta k)^2}$', transform=a.transAxes, ha='center',
           va='bottom', fontsize=13, color=ORANGE)
    # (b) which k
    b.plot(EZ_O, np.where(topo, k_osc, np.nan), color=INK, label=r'$k_{osc}$ (Majorana tail)')
    b.plot(EZ_O[::8], np.where(topo, k0, np.nan)[::8], 'o', color=BLUE, ms=6, mfc='none', mew=1.6,
           label=r'outer dip $k_0$')
    b.plot(EZ_O, np.where(topo, k_glob, np.nan), color=ORANGE, lw=2, ls='--', label='smallest gap')
    b.set_xlabel(r'$E_Z$ [meV]'); b.set_ylabel(r'$k$ [1/nm]'); b.set_ylim(-0.002, 0.05)
    b.legend(loc='lower left', fontsize=13)
    b.set_title('which k?', fontsize=15)
    # (c) which xi
    c.plot(EZ_O, np.where(topo, HBAR_VF / D_top, np.nan), color=MUTED, lw=2, ls=':',
           label=r'naive $\hbar v_F/\Delta_{top}$')
    c.plot(EZ_O, np.where(topo, xi_hyp, np.nan), color=BLUE, lw=2, ls='--', label=r'$\hbar v_A/\Delta_{out}$ (dip)')
    c.plot(EZ_O, np.where(topo, xi, np.nan), color=INK, label=r'$\xi_M$ (exact)')
    c.axhline(L_NM, color=INK2, lw=1, ls='-.')
    c.text(0.51, L_NM + 60, r'$L$ = 2 $\mu$m', color=INK2, fontsize=13)
    c.set_xlabel(r'$E_Z$ [meV]'); c.set_ylabel(r'length [nm]'); c.set_ylim(0, 2700)
    c.legend(loc='lower right', fontsize=13)
    c.set_title(r'which $\xi$?', fontsize=15)
    save(fig, 'slide6_lengths')


# %% --- SLIDE 9: S/N coupling buys gap and short xi_M, not area (a = 10, cross-section) ---
def slide9(TCS=(0.5, 0.75, 1.0), mu0=0.7, topo='scharf_absTopo_muC_a10_tc{tc:g}', xi0='scharf_absTopo_xi0_a10_tc{tc:g}',
           name='slide9_SN_coupling'):
    """Rashba 16 (study_tuning [topology]); the mixed SOC via mixed_sn (study_mixed [deck] cell 4)."""
    M = {tc: load(topo.format(tc=tc)) for tc in TCS}
    MU, EZ = M[1.0][1]['mu_C'], M[1.0][1]['E_Z']
    vmax = max(np.nanmax(M[tc][0][..., 1]) for tc in TCS) * 1e3
    cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.12, 1.0, 256)))
    cmap.set_bad(GRID)
    fig = plt.figure(figsize=FULL, layout='constrained')
    left, right = fig.subfigures(1, 2, width_ratios=(2.35, 1), wspace=0.02)
    axs = left.subplots(1, 3, sharey=True)
    gl_cmap = ListedColormap([GAPLESS_COL])                                     # NaN cells stay transparent
    for ax, tc in zip(axs, TCS):
        Qm, g = M[tc][0][..., 0], M[tc][0][..., 1] * 1e3
        gapped = (Qm < 0) & (g >= GAPLESS_UEV)
        im = ax.pcolormesh(MU, EZ, np.where(gapped, g, np.nan).T, shading='nearest', cmap=cmap, vmin=0, vmax=vmax)
        ax.pcolormesh(MU, EZ, np.where((Qm < 0) & ~gapped, 1.0, np.nan).T, shading='nearest', cmap=gl_cmap, vmin=0, vmax=1)
        ax.contour(MU, EZ, Qm.T, levels=[0], colors=INK2, linewidths=1.2)
        ax.axvline(mu0, color=RED, lw=1.4, ls='--')
        ax.grid(False)
        ax.set_title(rf'$t_{{SN}}$ = {tc:g}')
        ax.set_xlabel(r'$\mu_C$ [meV]')
        txt = (f'gapped {np.mean(gapped):.0%}' + (f' (Q=−1 {np.mean(Qm < 0):.0%})' if (~gapped & (Qm < 0)).any() else '')
               + f'\nmax {np.nanmax(np.where(gapped, g, np.nan)) if gapped.any() else 0:.0f} µeV')
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
    axs[0].set_ylabel(r'$E_Z$ [meV]')
    left.colorbar(im, ax=axs, location='bottom', shrink=0.7, aspect=35,
                  label=r'topological gap $\Delta_{top}$ [$\mu$eV]   (grey: trivial; orange: $Q = -1$ but gapless)')
    ax = right.subplots()
    i0 = int(np.argmin(np.abs(MU - mu0)))
    for tc, col in zip(TCS, (BLUE, ORANGE, AQUA)):
        line = np.flatnonzero(M[tc][0][i0, :, 0] < 0)
        ok = M[tc][0][i0, line, 1] * 1e3 >= GAPLESS_UEV                     # gapless: slowest_mode returns a bulk-SC mode
        xi = np.where(ok, load(xi0.format(tc=tc))[0][:, 1], np.nan)
        ax.semilogy(EZ[line], xi, 'o-', color=col, ms=5, lw=2, label=f'{tc:g}' + ('' if ok.any() else ' (gapless)'))
    ax.axhline(L_NM, color=INK2, lw=1, ls='-.')
    ax.set_xlim(0, EZ.max() + 0.05)                                         # same E_Z range as the maps
    ax.text(0.97, L_NM * 1.12, r'$L$ = 2 $\mu$m', color=INK2, fontsize=13, ha='right',
            transform=ax.get_yaxis_transform())
    ax.set_xlabel(r'$E_Z$ [meV]'); ax.set_ylabel(r'$\xi_M$ [nm]')
    ax.set_title(rf'$\xi_M$ at $\mu_C$ = {mu0:g} meV', fontsize=15)
    ax.legend(loc='lower right', fontsize=12, title=r'$t_{SN}$', title_fontsize=12)
    save(fig, name)


# %% --- SLIDE 10: the probe coupling sets Gamma -- a temperature optimum ---
def slide10(TAUS=(0.2, 0.4, 0.7), TS=(0.0, 10.0, 20.0, 50.0), win=(0.70, 1.10), case='W100_L2',
            name='slide10_temperature'):
    """Pure Rashba, a 20 (study_rashba [T]); case='W100_L2_mixed' is the mixed SOC (mixed_temperature)."""
    TCOL = ('#1e1757', '#4a3bb8', '#8a7ee0', '#c6c0f2')              # one hue, cold = dark
    tags = {0.2: 'too weak', 0.4: 'optimum', 0.7: 'too strong'}
    fig, axs = plt.subplots(1, 3, figsize=FULL, sharey=True, layout='constrained')
    for ax, tau in zip(axs, TAUS):
        V, ax_ = load(f'scharf_absTtc_{case}_tc{tau:g}')          # a 20, W 100, L 2 um, phi pi
        EZ, E = ax_['E_Z'], ax_['E']
        R = {T: st.thermal_line(V, st.kT_of_mK(T), E) for T in TS}
        for T, col in zip(TS[::-1], TCOL[::-1]):
            ax.plot(EZ, R[T][:, 0], color=col, lw=2.2, label=f'{T:g} mK')
        ax.plot(EZ, np.abs(R[0.0][:, 1]), color=RED, lw=1.4, ls=(0, (1, 1.2)), label=r'$|G|/G_0$ (0 mK)')
        half_line(ax)
        k20, G20 = R[20.0][:, 0], R[20.0][:, 1]
        neu = np.abs(G20) < 0.05 * np.maximum(k20, 1e-12)
        best = np.max(np.where(neu, k20, 0.0))
        ax.text(0.03, 0.97, f'best neutral $\\kappa$ at 20 mK: {best:.2f}' if best > 0.05 else
                'no neutral peak', transform=ax.transAxes, va='top', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
        ax.set_title(rf'probe $\tau$ = {tau:g}:  {tags[tau]}', fontsize=15)
        ax.set_xlim(*win); ax.set_xlabel(r'$E_Z$ [meV]')
        print(f'slide 10: tau {tau}: best neutral kappa at 20 mK over the whole scan {best:.3f}')
    axs[0].set_ylim(0, 0.62); axs[0].set_yticks([0, 0.25, 0.5]); axs[0].set_ylabel(r'$\kappa/\kappa_0$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[3::-1] + h[4:], l[3::-1] + l[4:], loc='outside upper center', ncol=5)
    save(fig, name)


# %% --- OVERLAP: how much the two MZMs must overlap to carry heat (eps = Gamma) ---
def majorana_profiles(q, E_Z, phi=np.pi):
    """
    The two end Majoranas of the closed box (study_oscillation.box_point geometry): the +-eps pair, rotated
    into its left- and right-localised combinations (eigenvectors of the position operator in the pair's
    2D subspace).  -> x (nm, 0 = centre), density of gamma_L and gamma_R per slice (each normalised), eps (ueV)
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from study_oscillation import bloch_blocks
    p = q.params(E_Z=float(E_Z), phi=phi)
    H0, H1 = bloch_blocks(p, st.n_S_of(q))
    n, m = q.n_along, len(H0)
    off, H1s = sp.diags(np.ones(n - 1), 1), sp.csr_matrix(H1)
    H = (sp.kron(sp.eye(n), sp.csr_matrix(H0)) + sp.kron(off, H1s) + sp.kron(off.T, H1s.conj().T)).tocsc()
    E, V = spla.eigsh(H, k=2, sigma=0.0, which='LM')
    X = np.repeat(np.arange(n), m).astype(float)                    # slice index of every basis state
    _, U = np.linalg.eigh(V.conj().T @ (X[:, None] * V))
    G = V @ U                                                       # columns: left-, then right-localised
    dens = (np.abs(G) ** 2).reshape(n, m, 2).sum(axis=1)
    return (np.arange(n) + 0.5) * q.a - q.L / 2, dens[:, 0], dens[:, 1], np.abs(E).mean() * q.t * 1e3


def overlap(fields=(0.55, None, 1.05), cache='slides_mzm_profiles'):
    """Left: MZM densities at three fields (L 2 um).  Right: kappa vs eps/Gamma, NEGF for L 1.5 / 2 / 2.5 um."""
    from study_oscillation import kappa_model, LS
    # right panel data: eps_box and NEGF on the fingerprint grid, Gamma(E_Z) from det r (smoothed)
    pts = []
    for L in LS:
        tag = f'L{L / 1000:g}'
        eps = load(f'scharf_absF_box_{tag}')[0][:, 0]
        V = load(f'scharf_absF_tr_{tag}')[0]
        x = eps / smooth_gamma(eps, V[:, 3])
        pts.append(np.c_[x, V[:, 0], np.abs(V[:, 1])])
        if L == 2000.0:
            x2, k2, Gam2 = x, V[:, 0], smooth_gamma(eps, V[:, 3])
    pts = np.vstack(pts)
    # the three fields: too little overlap, the first kappa = 1/2 peak, too much
    fields = list(fields)
    if fields[1] is None:
        w = (EZ_F > 0.70) & (EZ_F < 0.75)
        fields[1] = float(EZ_F[w][np.argmax(k2[w])])
    idx = [int(np.argmin(np.abs(EZ_F - f))) for f in fields]
    f = st.DATA / f'{cache}.npz'
    if f.exists():
        d = np.load(f)
        xs, prof = d['x'], d['prof']
    else:
        prof = []
        for E_Z in fields:
            xs, dL, dR, e = majorana_profiles(CASE, E_Z)
            prof.append([dL, dR])
            print(f'  profile E_Z {E_Z}: eps {e:.3f} ueV')
        prof = np.array(prof)
        np.savez(f, x=xs, prof=prof, fields=fields)

    fig = plt.figure(figsize=FULL, layout='constrained')
    left, right = fig.subfigures(1, 2, width_ratios=(1, 1.15), wspace=0.03)
    axl = left.subplots(3, 1, sharex=True, sharey=True)
    tags = ('too little overlap', r'just right: $\varepsilon = \Gamma$', 'too much overlap')
    for r, (ax, i) in enumerate(zip(axl, idx)):
        dL, dR = prof[r]
        ax.semilogy(xs / 1000, dL / dL.max(), color=BLUE, lw=1.6, label=r'$\gamma_L$')
        ax.semilogy(xs / 1000, dR / dR.max(), color=ORANGE, lw=1.6, label=r'$\gamma_R$')
        ax.set_ylim(1e-7, 300); ax.set_yticks([1e-6, 1e-3, 1])
        ax.text(0.5, 0.97, f'({"i" * (r + 1) if r < 2 else "iii"})  {tags[r]}   '
                rf'$E_Z$ = {EZ_F[i]:.2f},  $\varepsilon/\Gamma$ = {x2[i]:.2g}',
                transform=ax.transAxes, ha='center', va='top', fontsize=12.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
        print(f'overlap ({r + 1}): E_Z {EZ_F[i]:.3f}, eps/Gamma {x2[i]:.3f}, Gamma {Gam2[i]:.2f} ueV, kappa {k2[i]:.3f}')
    left.legend(*axl[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=13)
    axl[1].set_ylabel('Majorana density (each normalised)', fontsize=14)
    axl[-1].set_xlabel(r'position along the junction [$\mu$m]   (probes at $\pm$1)', fontsize=14)
    axl[-1].set_xlim(-1, 1)

    ax = right.subplots()
    xx = np.geomspace(1e-2, 1e2, 400)
    neutral = pts[:, 2] < 0.1 * np.maximum(pts[:, 1], 1e-12)
    ax.semilogx(pts[neutral, 0], pts[neutral, 1], 'o', ms=3, color=MUTED, alpha=0.6,
                label=r'full model, neutral ($L$ = 1.5, 2, 2.5 $\mu$m)')
    ax.semilogx(pts[~neutral, 0], pts[~neutral, 1], 'o', ms=4, mfc='none', mec=RED, mew=0.9,
                label=r'full model, charged ($|G| > 0.1\,\kappa$)')
    ax.semilogx(xx, kappa_model(xx), color=INK, lw=2.4, label=r'two Majoranas: $2x^2/(1+x^2)^2$')
    for r, i in enumerate(idx):
        ax.plot(x2[i], k2[i], 'o', ms=13, mfc='white', mec=INK, mew=2, zorder=6)
        ax.text(x2[i], k2[i], ('i', 'ii', 'iii')[r], ha='center', va='center', fontsize=10, zorder=7)
    half_line(ax)
    ax.text(0.024, 0.09, 'each MZM\nstays with\nits probe:\nno heat', ha='center', fontsize=12.5, color=INK2)
    ax.text(30, 0.25, 'ordinary fermion\nlevel at $\\pm\\varepsilon$,\noff resonance', ha='center',
            fontsize=13, color=INK2)
    ax.set_xlim(1e-2, 1e2); ax.set_ylim(0, 0.62); ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel(r'overlap energy / probe broadening   $x = \varepsilon/\Gamma$')
    ax.set_ylabel(r'$\kappa/\kappa_0$')
    ax.legend(loc='upper left', fontsize=11.5, bbox_to_anchor=(0, 1.0))
    save(fig, 'overlap_kappa')


# %% --- OVERLAP vs L: kappa at the top of each oscillation lobe, against L / xi_M ---
def overlap_L(EZ_ref=0.8):
    """
    At a lobe top |cos| = 1, so x = eps_max/Gamma = c e^{-L/xi_M} with c = A Delta_out/Gamma, and the
    two-Majorana kappa = 2x^2/(1+x^2)^2 is a single bump in L: 1/2 at L/xi_M = ln c.  Dots: NEGF kappa at the
    eps_box maximum of every full lobe for L = 1.5 / 2 / 2.5 um (study_fingerprint caches).  Grey band: the
    model for the spread of c over those lobes; line: its median.  Top axis: L at EZ_ref.
    """
    from study_oscillation import kappa_model, LS, XI_M
    rows = []
    for L in LS:
        tag = f'L{L / 1000:g}'
        eps = load(f'scharf_absF_box_{tag}')[0][:, 0]
        V = load(f'scharf_absF_tr_{tag}')[0]
        Gam = smooth_gamma(eps, V[:, 3])
        z = zeros_of(EZ_F, eps, half=0.03)
        for lo, hi in zip(z[:-1], z[1:]):
            w = np.flatnonzero((EZ_F > lo) & (EZ_F < hi))
            i = w[np.argmax(eps[w])]
            rows.append([L, L / XI_M[i], eps[i] / Gam[i], V[i, 0], abs(V[i, 1]), EZ_F[i]])
    R = np.array(rows)
    c = R[:, 2] * np.exp(R[:, 1])
    print('  L     E_Z    L/xi_M   eps/Gamma   kappa NEGF   model    |G|/kappa')
    for r in R:
        print(f'  {r[0] / 1000:g}   {r[5]:.3f}   {r[1]:5.2f}    {r[2]:6.2f}      {r[3]:.3f}      '
              f'{kappa_model(r[2]):.3f}    {r[4] / r[3]:.2f}')
    print(f'  c = A Delta_out/Gamma: {c.min():.0f} .. {c.max():.0f} (median {np.median(c):.0f}), '
          f'peak at L/xi_M = {np.log(c.min()):.2f} .. {np.log(c.max()):.2f}; '
          f'rms(NEGF - model at own x) {np.sqrt(np.mean((R[:, 3] - kappa_model(R[:, 2])) ** 2)):.3f}')

    s = np.linspace(0, 7.5, 600)
    fam = np.array([kappa_model(cc * np.exp(-s)) for cc in np.geomspace(c.min(), c.max(), 60)])
    xi_ref = float(np.interp(EZ_ref, EZ_F, XI_M))
    fig, ax = plt.subplots(figsize=FULL, layout='constrained')
    ax.fill_between(s, fam.min(0), fam.max(0), color=GRID, lw=0)
    ax.plot(s, kappa_model(np.median(c) * np.exp(-s)), color=INK, lw=2.4,
            label=r'two Majoranas, $\varepsilon = A\,\Delta_{out}\,e^{-L/\xi_M}$')
    for L, col in zip(LS, (BLUE, ORANGE, AQUA)):
        m = R[:, 0] == L
        ax.plot(R[m, 1], R[m, 3], 'o', ms=10, color=col, mec='white', mew=1.2, zorder=5,
                label=rf'full model, $L$ = {L / 1000:g} $\mu$m')
    half_line(ax)
    ax.annotate(r'$\varepsilon = \Gamma$', xy=(np.log(np.median(c)), 0.5), xytext=(np.log(np.median(c)), 0.56),
                ha='center', fontsize=15)
    ax.text(1.2, 0.30, 'too much overlap\n' + r'$\varepsilon \gg \Gamma$', ha='center', fontsize=15, color=INK2)
    ax.text(6.3, 0.30, 'too little overlap\n' + r'$\varepsilon \ll \Gamma$', ha='center', fontsize=15, color=INK2)
    ax.set_xlim(0, 7.5); ax.set_ylim(0, 0.62); ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel(r'junction length in units of the Majorana length   $L/\xi_M$')
    ax.set_ylabel(r'$\kappa/\kappa_0$  at the top of a lobe')
    top = ax.secondary_xaxis('top', functions=(lambda v: v * xi_ref / 1000, lambda v: v * 1000 / xi_ref))
    top.set_xlabel(rf'$L$ [$\mu$m]  at $E_Z$ = {EZ_ref:g} meV  ($\xi_M$ = {xi_ref / 1000:.2f} $\mu$m)', fontsize=14)
    top.tick_params(colors=INK2)
    ax.legend(loc='upper right', fontsize=13)
    save(fig, 'overlap_vs_L')


# %% --- kappa(E_Z) FOR L = 2, 4, 8 um: the right length depends on the field (xi_M grows with E_Z) ---
def kappa_L(LS_K=(2000.0, 4000.0, 8000.0), n_workers=4):
    """NEGF kappa, |G| at E = 0 on the fingerprint field grid (tc_barr 0.2, a 20).  L = 2 um is the
    study_fingerprint cache; other L are computed here and cached as scharf_absF_tr_L<L>."""
    from study_oscillation import XI_M
    R = {L: st.transport_line(st.case(100.0, L=L), EZ_F, cache=f'scharf_absF_tr_L{L / 1000:g}', n_workers=n_workers)
         for L in LS_K}
    fig, axs = plt.subplots(len(LS_K), 1, figsize=FULL, sharex=True, sharey=True, layout='constrained')
    for ax, L in zip(axs, LS_K):
        k, G = R[L]['kappa'], np.abs(R[L]['G'])
        ax.plot(EZ_F, G, color=RED, lw=1.2)
        ax.plot(EZ_F, k, color=INK, lw=1.8)
        half_line(ax)
        ax.text(0.005, 0.93, rf'$L$ = {L / 1000:g} $\mu$m', transform=ax.transAxes, va='top', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
        neu = G < 0.1 * np.maximum(k, 1e-12)
        j = int(np.argmax(np.where(neu, k, -1)))
        s = L / XI_M
        print(f'  L {L / 1000:g} um: best neutral kappa {k[j]:.3f} at E_Z {EZ_F[j]:.3f} (L/xi_M {s[j]:.2f}); '
              f'neutral kappa > 0.45 at {np.round(EZ_F[neu & (k > 0.45)][[0, -1]], 3) if (neu & (k > 0.45)).any() else "-"}; '
              f'L/xi_M in 2.5-3.8 for E_Z {np.round(EZ_F[(s > 2.5) & (s < 3.8)][[0, -1]], 3) if ((s > 2.5) & (s < 3.8)).any() else "-"}')
    axs[0].set_ylim(0, 0.6); axs[0].set_yticks([0, 0.5])
    axs[1].set_ylabel(r'$\kappa/\kappa_0$  (black),   $|G|/G_0$  (red)')
    axs[-1].set_xlabel(r'Zeeman energy $E_Z$ [meV]'); axs[-1].set_xlim(EZ_F[0], EZ_F[-1])
    save(fig, 'kappa_vs_L')


# %% --- MIXED SOC: kappa depends on eps/Gamma alone -- a resonance, not a topological plateau ---
def collapse_mixed(EZ_PAIR=0.94, L=2000.0):
    """
    Mixed SOC (st.mixed_soc(16, 0.15 pi)), mu_C 0.7, W 100, L 2 um, a 10, phi pi; caches of study_oscillation_mixed.
    Left: NEGF kappa(E_Z) at two probe bonds -- the kappa = 1/2 points sit at DIFFERENT fields.
    Right: the same points against x = eps_box / Gamma.  eps from the closed box (no leads), Gamma measured directly
    (half width of the local Andreev Lorentzian at an eps zero, mean of two zeros per tau): nothing fitted.  Two
    Majoranas: kappa = 2x^2/(1+x^2)^2, = 1/2 only at x = 1.  Constant Gamma: Gamma_eff rises ~1.1 -> 1.6 ueV over
    E_Z 0.45-0.93 (CLAUDE.md), which spreads x by up to ~1.4.  E_Z > EZ_PAIR: L/xi_M small, no separated end pair
    (two-Majorana points end at 0.94), shown grey.
    """
    M, am = load('scharf_absOM_modes_mixed_a10')
    EZ, xi = am['E_Z'], M[:, 0]
    eps = load('scharf_absOM_box_mixed_a10_L2')[0][:, 0]           # ueV
    GD = load('scharf_absOM_gamma_mixed')[0][:, 0]                 # ueV, cases (0.3387, 0.5) x two eps zeros
    TAU = {0.3387: GD[:2].mean(), 0.5: GD[2:].mean()}
    s_cut = L / np.interp(EZ_PAIR, EZ, xi)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FULL, layout='constrained', width_ratios=(1, 1.1))
    xx = np.logspace(-3, 1.5, 500)
    a2.plot(xx, 2 * xx**2 / (1 + xx**2) ** 2, color=INK, lw=2.6, zorder=2)
    a2.text(3.3, 0.43, 'two Majoranas\n' + r'$2x^2/(1+x^2)^2$', fontsize=13, color=INK)
    grey = []
    for (t, g), c in zip(TAU.items(), (BLUE, ORANGE)):
        V, at = load(f'scharf_absOM_line_mixed_tau{t:g}')
        ez = at['E_Z']
        k, G = V[:, 0], V[:, 1]
        x = eps[np.searchsorted(EZ, ez)] / g
        pair, neu = ez <= EZ_PAIR + 1e-9, np.abs(G) < 0.1 * np.maximum(k, 1e-12)
        lab = rf'$\tau$ = {t:.2f}  ($\Gamma$ = {g:.1f} $\mu$eV)'
        a1.plot(ez, k, color=c, lw=2, label=lab)
        a2.plot(x[pair & neu], k[pair & neu], 'o', ms=7.5, color=c, mec='white', mew=0.7, zorder=3, label=lab)
        if (pair & ~neu).any():
            a2.plot(x[pair & ~neu], k[pair & ~neu], 'o', ms=7.5, mfc='none', mec=c, mew=1.4, zorder=3)
        grey.append((x[~pair], k[~pair]))
        pred = 2 * x**2 / (1 + x**2) ** 2
        r = np.abs(k - pred)[pair]
        big = pair & (k > 0.3)
        print(f'collapse tau {t}: Gamma {g:.2f} ueV, {pair.sum()} pts E_Z <= {EZ_PAIR}: |kappa - model| median '
              f'{np.median(r):.3f}, 90 % {np.percentile(r, 90):.3f}; kappa > 0.3: {big.sum()} pts at x '
              f'{np.round(np.sort(x[big]), 2)}; max kappa {k[pair].max():.3f} (neutral {k[pair & neu].max():.3f}); '
              f'charged in pair range: {(pair & ~neu).sum()}')
    a2.plot([], [], 'o', ms=7.5, mfc='none', mec=INK2, mew=1.4, label=r'open: charged, $|G| > 0.1\,\kappa$')
    a2.plot(np.concatenate([g[0] for g in grey]), np.concatenate([g[1] for g in grey]), 'o', ms=5, color=MUTED,
            alpha=0.45, mec='none', zorder=1, label=rf'$E_Z > {EZ_PAIR}$ meV  ($L/\xi_M < {s_cut:.1f}$)')
    a1.text(1.145, 0.70, 'no separated end pair', ha='center', fontsize=12, color=INK2)
    half_line(a1); half_line(a2)
    a2.axvline(1, color=INK2, lw=1, ls=':', zorder=1)
    a1.set_xlim(EZ[0], EZ[-1]); a1.set_ylim(0, 0.78); a1.set_yticks([0, 0.25, 0.5])
    a1.axvspan(EZ_PAIR, EZ[-1], color=GRID, alpha=0.6, lw=0, zorder=0)
    a1.set_xlabel(r'Zeeman energy $E_Z$ [meV]  ($\phi = \pi$)'); a1.set_ylabel(r'$\kappa/\kappa_0$  (NEGF)')
    a1.legend(loc='upper left', fontsize=13)
    a2.set_xscale('log'); a2.set_xlim(2e-3, 30); a2.set_ylim(0, 0.78); a2.set_yticks([0, 0.25, 0.5])
    a2.set_xlabel(r'$x = \varepsilon_{box}/\Gamma$   (splitting / measured broadening)')
    a2.legend(loc='upper left', fontsize=12)
    save(fig, 'mixed_collapse')


# %% --- MIXED SOC: longer junction = better Majoranas -> kappa vanishes, det r -> -1 (not a plateau) ---
def length_detr_mixed(tau=0.5, LS=(1000.0, 1500.0, 2000.0, 3000.0)):
    """
    Mixed SOC, mu_C 0.7, W 100, a 10, phi pi, probe bond tau: phi = pi column of the study_tuning cell 7 coarse maps
    (E_Z step 0.05 -- the kappa peaks are under-sampled; the point is the trend).  Separating the Majoranas (longer L)
    makes kappa vanish over a growing low-field range, where det r -> -1 (Fulga: each lead sees one isolated
    Majorana), and pushes the kappa peaks up to where xi_M ~ L.  A topological plateau would grow with L instead.
    Shaded: bulk Q = -1 (infinite junction, independent of L and tau).
    """
    Qm, aq = load('scharf_absTune_Q_mixed_mu0.7')
    EZq, jq = aq['E_Z'], int(np.argmin(np.abs(aq['phi'] - np.pi)))
    topo = EZq[Qm[:, jq, 0] < 0]
    ramp = ('#86b6ef', '#3987e5', '#1c5cab', '#0d366b')            # ordinal blue, steps 250/400/550/700
    fig, (a1, a2) = plt.subplots(2, 1, figsize=FULL, sharex=True, layout='constrained')
    for a in (a1, a2):
        a.axvspan(topo.min(), topo.max(), color=GRID, alpha=0.6, lw=0, zorder=0)
    for L, c in zip(LS, ramp):
        V, a = load(f'scharf_absTune_tr_mixed_mu0.7_tau{tau:g}' + ('' if L == 2000.0 else f'_L{L / 1000:g}'))
        EZ = a['E_Z']
        j = int(np.argmin(np.abs(np.linspace(0, 2 * np.pi, V.shape[1]) - np.pi)))   # study_tuning.PHI
        k, G, d = V[:, j, 0], V[:, j, 1], V[:, j, 3]
        lab = rf'$L$ = {L / 1000:g} $\mu$m'
        a1.plot(EZ, k, 'o-', color=c, lw=2, ms=5, label=lab)
        a2.plot(EZ, d, 'o-', color=c, lw=2, ms=5, label=lab)
        low = (EZ > 0) & (EZ < topo.max()) & (d < -0.9)
        run = EZ[low]
        print(f'L {L / 1000:g} um, tau {tau}: det r < -0.9 for E_Z {run.min():.2f}-{run.max():.2f} meV '
              f'(kappa there <= {k[low].max():.3f}); first neutral kappa > 0.3 at E_Z '
              f'{EZ[(k > 0.3) & (np.abs(G) < 0.1 * k)].min():.2f}' if low.any() else f'L {L / 1000:g}: no det r < -0.9')
    print(f'bulk Q = -1 at phi = pi for E_Z {topo.min():.2f}-{topo.max():.2f} meV (grid step {EZq[1] - EZq[0]:.2f})')
    half_line(a1)
    a1.set_ylim(0, 1.0); a1.set_yticks([0, 0.5, 1]); a1.set_ylabel(r'$\kappa/\kappa_0$')
    a1.legend(loc='upper left', ncol=4, fontsize=13)
    a2.axhline(-1, color=INK2, lw=1, ls='--', zorder=1)
    a2.set_ylim(-1.1, 1.1); a2.set_yticks([-1, 0, 1]); a2.set_ylabel(r'det $r$')
    a2.text(0.99, 0.08, r'det $r = -1$: each probe sees one isolated Majorana', transform=a2.transAxes, fontsize=13,
            color=INK2, va='bottom', ha='right')
    a1.text(1.38, 0.78, r'trivial ($Q = +1$)', ha='left', fontsize=13, color=INK2)
    a2.set_xlabel(r'Zeeman energy $E_Z$ [meV]  ($\phi = \pi$, $\tau$ = ' + f'{tau:g}' + r';  grey: bulk $Q = -1$)')
    a2.set_xlim(0, 2)
    save(fig, 'mixed_length_detr')


# %% --- MIXED SOC versions of the pure-Rashba slides (2026-11-11) ---
# The thesis junction study_tuning.junction('mixed', 0.7): a 10, W 100, L 2 um, t_SN 1, mu_C 0.7, phi = pi.
# Data: study_mixed [oscillation] (scharf_absOM_*), [amplitude] (scharf_absAmp_*), [deck] (scharf_absDeck_*).
HBAR_VF_2DEG = 52.98            # meV nm, hbar^2 k_mu/m at mu_C 0.7 (continuum; the a = 20 lattice gives 51.4)


def mixed_case(tau=0.3387, L=2000.0):
    from study_tuning import junction
    return junction('mixed', 0.7, tau, L=L)


def om_modes():
    """Infinite junction at phi = pi on the [oscillation] grid: dict of the osc_point columns.  The dip columns
    (k0 .. xi_poly) for E_Z >= 0.9 come from scharf_absOM_dip_mixed_a10_fix (dip_point FD step fixed 2026-11-11,
    recomputed by study_mixed [oscillation] cell 1), as in study_mixed."""
    V, a = load('scharf_absOM_modes_mixed_a10')
    V = V.copy()
    V[a['E_Z'] >= 0.9, 9:] = load('scharf_absOM_dip_mixed_a10_fix')[0]
    names =('xi', 'k_osc', 'xi2', 'k2', 'xi3', 'k3', 'D_top', 'Q', 'k_gap', 'k0', 'D_out', 'hv_A', 'xi_hyp', 'xi_poly')
    return a['E_Z'], dict(zip(names, V.T))


def mixed_lengths(EZ_val=0.8):
    """Slide 6 in the mixed SOC: the band with the outer dip, which k sets the oscillation, which xi the decay."""
    EZ, m = om_modes()
    topo = m['Q'] < 0
    j = int(np.argmin(np.abs(EZ - EZ_val)))
    k, E, kz, Ez = band(EZ_val, k0=m['k0'][j], half=0.004, cache=f'slides_band_mixed_a10_EZ{EZ_val:g}', q=mixed_case())
    naive = HBAR_VF_2DEG / (m['D_top'] * 1e3) * 1e3                 # nm
    hi = EZ > 0.5                                                    # the slowest mode switches branch below ~0.49
    print(f"mixed lengths at E_Z {EZ[j]:g}: k0 {m['k0'][j]:.4f}, k_osc {m['k_osc'][j]:.4f} 1/nm, Delta_out "
          f"{m['D_out'][j] * 1e3:.1f} ueV (Delta_top {m['D_top'][j] * 1e3:.1f}), hbar v_A {m['hv_A'][j]:.1f} meV nm, "
          f"xi_hyp {m['xi_hyp'][j]:.0f}, xi_M {m['xi'][j]:.0f}, naive {naive[j]:.0f} nm; E_Z > 0.5: "
          f"max |k_osc - k0| {np.max(np.abs(m['k_osc'] - m['k0'])[hi]):.5f} 1/nm, naive/xi_M "
          f"{np.min((naive / m['xi'])[hi]):.2f}-{np.max((naive / m['xi'])[hi]):.2f}, xi_hyp/xi_M "
          f"{np.min((m['xi_hyp'] / m['xi'])[hi]):.2f}-{np.max((m['xi_hyp'] / m['xi'])[hi]):.2f}")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=FULL, layout='constrained', width_ratios=(1.2, 1, 1))
    D, kk = m['D_out'][j] * 1e3, m['k0'][j]
    a.plot(k, E[:, 0], color=INK, lw=2)
    a.plot(kk, D, 'o', color=BLUE, ms=8, zorder=5)
    a.annotate(r'outer dip $k_0$:' + '\n' + r'$\Delta_{out}$, velocity $v_A$', xy=(kk, D), xytext=(0.003, 30),
               color=BLUE, fontsize=14, arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    a.set_xlim(0, 0.07); a.set_ylim(0, 160)
    a.set_xlabel(r'$k$ along the junction [1/nm]'); a.set_ylabel(r'$E$ [$\mu$eV]')
    a.set_title(rf'lowest Andreev band, $E_Z$ = {EZ_val:g} meV', fontsize=15)
    ins = a.inset_axes([0.62, 0.06, 0.36, 0.42])
    ins.plot(kz, Ez[:, 0], color=INK, lw=2)
    ins.plot(kz, np.sqrt(D ** 2 + (m['hv_A'][j] * 1e3 * (kz - kk)) ** 2), color=ORANGE, lw=1.6, ls='--')
    ins.plot(kk, D, 'o', color=BLUE, ms=6, zorder=5)
    ins.set_xlim(kz[0], kz[-1]); ins.set_ylim(0, 4 * D)
    ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    a.text(0.80, 0.50, r'$\sqrt{\Delta_{out}^2 + (\hbar v_A\,\delta k)^2}$', transform=a.transAxes, ha='center',
           va='bottom', fontsize=13, color=ORANGE)
    b.plot(EZ, np.where(topo, m['k_osc'], np.nan), color=INK, label=r'$k_{osc}$ (Majorana tail)')
    b.plot(EZ[::10], np.where(topo, m['k0'], np.nan)[::10], 'o', color=BLUE, ms=6, mfc='none', mew=1.6,
           label=r'outer dip $k_0$ = smallest gap')
    b.set_xlabel(r'$E_Z$ [meV]'); b.set_ylabel(r'$k$ [1/nm]'); b.set_ylim(0.018, 0.046)
    b.legend(loc='lower right', fontsize=13)
    b.set_title('which k?', fontsize=15)
    c.semilogy(EZ, np.where(topo, naive, np.nan), color=MUTED, lw=2, ls=':', label=r'naive $\hbar v_F/\Delta_{top}$')
    c.semilogy(EZ, np.where(topo, m['xi_hyp'], np.nan), color=BLUE, lw=2, ls='--', label=r'$\hbar v_A/\Delta_{out}$ (dip)')
    c.semilogy(EZ, np.where(topo, m['xi'], np.nan), color=INK, label=r'$\xi_M$ (exact)')
    c.axhline(2000.0, color=INK2, lw=1, ls='-.')
    c.text(0.55, 2000.0 * 1.12, r'$L$ = 2 $\mu$m', color=INK2, fontsize=13)
    c.set_xlabel(r'$E_Z$ [meV]'); c.set_ylabel('length [nm]'); c.set_ylim(150, 3e4)
    c.legend(loc='upper left', fontsize=13)
    c.set_title(r'which $\xi$?', fontsize=15)
    save(fig, 'mixed_lengths')


def mixed_transport_line(L=2000.0, tau=0.3387):
    """(E_Z, values) of the finest cached phi = pi line: study_mixed [deck] (step 0.002) if complete, else [oscillation]."""
    try:
        V, a = load(f'scharf_absDeck_tr_mixed_tau{tau:g}_L{L / 1000:g}')
        if np.all(np.isfinite(V[:, 0])):
            return a['E_Z'], V
    except Exception:                        # missing, or being written by a running scan
        pass
    V, a = load(f'scharf_absOM_line_mixed_tau{tau:g}')
    return a['E_Z'], V


def mixed_oscillation(tau=0.3387):
    """
    Slide 5 in the mixed SOC.  Top: eps of the closed box (exact) and the overlap of the two semi-infinite end
    Majoranas (Rayleigh-Ritz in span{gamma_L, gamma_R}, no fit), log scale, against the measured probe broadening
    Gamma.  Bottom: NEGF kappa, |G| (L 2 um, phi = pi).  Dotted: zeros of eps.  The single envelope A Delta_out e^{-L/xi_M}
    is NOT drawn: in the mixed SOC its prefactor drifts ~4x (CLAUDE.md, MIXED SOC); the zeros (k_osc) and the overlap hold.
    """
    EZ, m = om_modes()
    eps = load('scharf_absOM_box_mixed_a10_L2')[0][:, 0]
    RR, aa = load('scharf_absAmp_mixed_a10_L2')
    ez_rr, eps_rr = aa['E_Z'], RR[:, 0]
    GD = load('scharf_absOM_gamma_mixed')[0][:, 0]
    Gam = {0.3387: GD[:2].mean(), 0.5: GD[2:].mean()}
    ez_t, V = mixed_transport_line(tau=tau)
    z = zeros_of(EZ, eps, half=0.03)
    z = z[(z > 0.45) & (z < 1.3)]
    Phi = np.interp(z, EZ, m['k_osc']) * 2000.0 / np.pi
    print(f"mixed oscillation: zeros {np.round(z, 3)}; k_osc L/pi steps {np.round(np.diff(Phi), 2)}; transport grid "
          f"step {ez_t[1] - ez_t[0]:.3f}; Gamma {Gam}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=FULL, sharex=True, layout='constrained', height_ratios=(1.25, 1))
    a1.semilogy(EZ[::2], eps[::2], 'o', ms=4.2, color=BLUE, mec='white', mew=0.4, label=r'splitting $\varepsilon$ (closed box)')
    a1.semilogy(ez_rr, eps_rr, color=ORANGE, lw=1.8, label=r'overlap of the two end Majoranas (no fit)')
    for (t, g), ls in zip(Gam.items(), ('-', '--')):
        a1.axhline(g, color=AQUA, lw=2, ls=ls, label=rf'probe broadening $\Gamma$ ($\tau$ = {t:.2f})')
    a1.set_ylim(1e-3, 3e3); a1.set_ylabel(r'energy [$\mu$eV]')
    a1.legend(loc='upper left', ncol=2, fontsize=12.5)
    a2.plot(ez_t, V[:, 0], color=INK, lw=1.8, label=rf'$\kappa/\kappa_0$  ($\tau$ = {tau:.2f})')
    a2.plot(ez_t, np.abs(V[:, 1]), color=RED, lw=1.4, label=r'$|G|/G_0$')
    half_line(a2)
    a2.set_ylim(0, 0.65); a2.set_yticks([0, 0.25, 0.5]); a2.set_ylabel(r'$\kappa/\kappa_0$,  $|G|/G_0$')
    a2.set_xlabel(r'Zeeman energy $E_Z$ [meV]  ($\phi = \pi$, $L$ = 2 $\mu$m)'); a2.set_xlim(0.1, 1.35)
    a2.legend(loc='upper left', ncol=2, fontsize=13)
    top = a1.secondary_xaxis('top', functions=(lambda v: v, lambda v: v))
    ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    top.set_xticks(ticks)
    top.set_xticklabels([f'{2000.0 / np.interp(t, EZ, m["xi"]):.1f}' for t in ticks])
    top.set_xlabel(r'$L/\xi_M$', fontsize=14); top.tick_params(colors=INK2)
    for x in z:
        for a in (a1, a2):
            a.axvline(x, color=MUTED, lw=1, ls=':', zorder=1)
    save(fig, 'mixed_oscillation')


def mixed_overlap(fields=(0.37, None, 1.15), tau=0.3387, EZ_PAIR=0.94):
    """
    The overlap picture in the mixed SOC.  Left: densities of the two end Majoranas of the closed box (L 2 um) at three
    fields.  Right: NEGF kappa against x = eps_box/Gamma (Gamma measured, nothing fitted) for both probe bonds.
    """
    EZ, m = om_modes()
    eps = load('scharf_absOM_box_mixed_a10_L2')[0][:, 0]
    GD = load('scharf_absOM_gamma_mixed')[0][:, 0]
    Gam = {0.3387: GD[:2].mean(), 0.5: GD[2:].mean()}
    lines = {t: load(f'scharf_absOM_line_mixed_tau{t:g}') for t in Gam}
    ez_l = lines[tau][1]['E_Z']
    k_l = lines[tau][0][:, 0]
    fields = list(fields)
    if fields[1] is None:
        w = (ez_l > 0.55) & (ez_l < 0.68)
        fields[1] = float(ez_l[w][np.argmax(k_l[w])])
    q = mixed_case(tau)
    f = st.DATA / 'slides_mzm_profiles_mixed_a10.npz'
    if f.exists() and np.allclose(np.load(f)['fields'], fields):
        d = np.load(f)
        xs, prof, ee = d['x'], d['prof'], d['eps']
    else:
        prof, ee = [], []
        for E_Z in fields:
            xs, dL, dR, e = majorana_profiles(q, E_Z)
            prof.append([dL, dR]); ee.append(e)
            print(f'  profile E_Z {E_Z}: eps {e:.4f} ueV', flush=True)
        prof, ee = np.array(prof), np.array(ee)
        np.savez(f, x=xs, prof=prof, fields=fields, eps=ee)

    fig = plt.figure(figsize=FULL, layout='constrained')
    left, right = fig.subfigures(1, 2, width_ratios=(1, 1.15), wspace=0.03)
    axl = left.subplots(3, 1, sharex=True, sharey=True)
    tags = ('too little overlap', r'just right: $\varepsilon \approx \Gamma$', 'too much overlap')
    for r, ax in enumerate(axl):
        dL, dR = prof[r]
        ax.semilogy(xs / 1000, dL / dL.max(), color=BLUE, lw=1.6, label=r'$\gamma_L$')
        ax.semilogy(xs / 1000, dR / dR.max(), color=ORANGE, lw=1.6, label=r'$\gamma_R$')
        ax.set_ylim(1e-7, 300); ax.set_yticks([1e-6, 1e-3, 1])
        x_r = ee[r] / Gam[tau]
        ax.text(0.5, 0.97, f'({"i" * (r + 1)})  {tags[r]}   ' + rf'$E_Z$ = {fields[r]:.2f},  $\varepsilon/\Gamma$ = {x_r:.2g}',
                transform=ax.transAxes, ha='center', va='top', fontsize=12.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
        print(f'mixed overlap ({r + 1}): E_Z {fields[r]:.3f}, eps {ee[r]:.4f} ueV, eps/Gamma {x_r:.3f}, '
              f'L/xi_M {2000.0 / np.interp(fields[r], EZ, m["xi"]):.2f}')
    left.legend(*axl[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=13)
    axl[1].set_ylabel('Majorana density (each normalised)', fontsize=14)
    axl[-1].set_xlabel(r'position along the junction [$\mu$m]   (probes at $\pm$1)', fontsize=14)
    axl[-1].set_xlim(-1, 1)

    ax = right.subplots()
    xx = np.geomspace(1e-3, 1e2, 400)
    ax.semilogx(xx, 2 * xx ** 2 / (1 + xx ** 2) ** 2, color=INK, lw=2.4, label=r'two Majoranas: $2x^2/(1+x^2)^2$')
    for (t, g), col in zip(Gam.items(), (BLUE, ORANGE)):
        V, at = lines[t]
        ez = at['E_Z']
        k, G = V[:, 0], np.abs(V[:, 1])
        x = eps[np.searchsorted(EZ, ez)] / g
        pair, neu = ez <= EZ_PAIR + 1e-9, G < 0.1 * np.maximum(k, 1e-12)
        ax.semilogx(x[pair & neu], k[pair & neu], 'o', ms=6, color=col, mec='white', mew=0.6,
                    label=rf'full model, $\tau$ = {t:.2f} ($\Gamma$ = {g:.1f} $\mu$eV)')
        ax.semilogx(x[pair & ~neu], k[pair & ~neu], 'o', ms=6, mfc='none', mec=col, mew=1.2)
    for r in range(3):
        i = int(np.argmin(np.abs(ez_l - fields[r])))
        xr = eps[np.argmin(np.abs(EZ - fields[r]))] / Gam[tau]
        ax.plot(xr, k_l[i], 'o', ms=13, mfc='white', mec=INK, mew=2, zorder=6)
        ax.text(xr, k_l[i], ('i', 'ii', 'iii')[r], ha='center', va='center', fontsize=10, zorder=7)
    half_line(ax)
    ax.plot([], [], 'o', ms=6, mfc='none', mec=INK2, mew=1.2, label=r'open: charged, $|G| > 0.1\,\kappa$')
    ax.set_xlim(1e-3, 1e2); ax.set_ylim(0, 0.62); ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel(r'overlap energy / probe broadening   $x = \varepsilon/\Gamma$')
    ax.set_ylabel(r'$\kappa/\kappa_0$')
    ax.legend(loc='upper left', fontsize=11.5)
    save(fig, 'mixed_overlap')


def mixed_kappa_L(LS=(2000.0, 3000.0, 4000.0), tau=0.3387, s_on=3.8):
    """kappa_L in the mixed SOC (study_mixed [deck] cell 1): does neutral kappa ~ 1/2 switch on at L/xi_M ~ 3.8?
    Shaded: 2.5 < L/xi_M < 3.8 (the Rashba window); dashed: L/xi_M = 3.8."""
    EZ, m = om_modes()
    fig, axs = plt.subplots(len(LS), 1, figsize=FULL, sharex=True, sharey=True, layout='constrained')
    for ax, L in zip(axs, LS):
        V, a = load(f'scharf_absDeck_tr_mixed_tau{tau:g}_L{L / 1000:g}')
        e, k, G = a['E_Z'], V[:, 0], np.abs(V[:, 1])
        s = L / np.interp(EZ, EZ, m['xi'])
        win = (s > 2.5) & (s < s_on)
        if win.any():
            ax.axvspan(EZ[win][0], EZ[win][-1], color=GRID, alpha=0.7, lw=0, zorder=0)
        e_on = EZ[np.flatnonzero(s < s_on)[0]]
        ax.axvline(e_on, color=INK2, lw=1.2, ls='--', zorder=1)
        ax.plot(e, G, color=RED, lw=1.2)
        ax.plot(e, k, color=INK, lw=1.6)
        half_line(ax)
        ax.text(0.005, 0.93, rf'$L$ = {L / 1000:g} $\mu$m', transform=ax.transAxes, va='top', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
        neu = G < 0.1 * np.maximum(k, 1e-12)
        on = e[neu & (k > 0.45)]
        print(f'mixed kappa_L: L {L / 1000:g} um: neutral kappa > 0.45 from E_Z {on[0] if on.size else np.nan:.3f} '
              f'(L/xi_M {L / np.interp(on[0], EZ, m["xi"]) if on.size else np.nan:.2f}); L/xi_M = {s_on} at {e_on:.3f}; '
              f'best neutral kappa {np.max(np.where(neu, k, 0)):.3f}; charged kappa > 0.3 from '
              f'{e[~neu & (k > 0.3)][0] if (~neu & (k > 0.3)).any() else np.nan:.3f}')
    axs[0].set_ylim(0, 0.65); axs[0].set_yticks([0, 0.5])
    axs[1].set_ylabel(r'$\kappa/\kappa_0$  (black),   $|G|/G_0$  (red)')
    axs[-1].set_xlabel(r'Zeeman energy $E_Z$ [meV]  ($\phi = \pi$, $\tau$ = ' + f'{tau:.2f}'
                       + r';  grey: $2.5 < L/\xi_M < 3.8$, dashed: $L/\xi_M = 3.8$)')
    axs[-1].set_xlim(0.3, 1.35)
    save(fig, 'mixed_kappa_vs_L')


def mixed_sn():
    """Slide 9 in the mixed SOC (study_mixed [deck] cell 4)."""
    slide9(topo='scharf_absDeck_topo_mixed_tc{tc:g}', xi0='scharf_absDeck_xi0_mixed_tc{tc:g}', name='mixed_SN_coupling')


def mixed_temperature():
    """Slide 10 in the mixed SOC, a = 20 (study_rashba [T] mixed case; tc 0.7 from study_mixed [deck] cell 2)."""
    slide10(case='W100_L2_mixed', name='mixed_temperature')


def mixed_best_T(TS=(0.0, 10.0, 20.0, 50.0), EZ0=0.930, phi_c=1.7417):
    """kappa(T) at the best mixed kappa = 1/2 region (a 10, tau 0.5, L 2 um; study_mixed [deck] cell 3).
    Left: along phi at E_Z 0.930.  Right: along E_Z at phi 1.742 pi.  |G| dotted at 20 mK."""
    from study_rashba import E_SCAN
    TCOL = ('#1e1757', '#4a3bb8', '#8a7ee0', '#c6c0f2')
    VP = load(f'scharf_absDeck_Tphi_mixed_tau0.5_L2_EZ{EZ0:g}')[0][None]           # (1, n_E, 6 x 241)
    VE, ae = load('scharf_absDeck_T_mixed_tau0.5_L2_EZ0.85-1.05')                 # (n_EZ, n_E, 6 x 3)
    ez = ae['E_Z']
    phis = np.linspace(0.0, 2.0, 241)[:-1]                                          # study_mixed.PHI_F / pi
    order = np.argsort((phis - 0.75) % 2.0)                                         # start at 0.75 pi, as plot_best
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FULL, sharey=True, layout='constrained')
    for T, col in zip(TS[::-1], TCOL[::-1]):
        kT = st.kT_of_mK(T)
        rp = np.array([st.thermal_line(VP, kT, E_SCAN, phase=n)[0] for n in range(len(phis))])
        re = st.thermal_line(VE, kT, E_SCAN, phase=0)
        a1.plot(np.arange(len(phis)), rp[order, 0], color=col, lw=2.2, label=f'{T:g} mK')
        a2.plot(ez, re[:, 0], color=col, lw=2.2)
        if T == 20.0:
            a1.plot(np.arange(len(phis)), np.abs(rp[order, 1]), color=RED, lw=1.3, ls=(0, (1, 1.2)), label=r'$|G|/G_0$ (20 mK)')
            a2.plot(ez, np.abs(re[:, 1]), color=RED, lw=1.3, ls=(0, (1, 1.2)))
        ok = (np.abs(rp[:, 0] - 0.5) < 0.03) & (np.abs(rp[:, 1]) < 0.1 * rp[:, 0])
        oke = (np.abs(re[:, 0] - 0.5) < 0.03) & (np.abs(re[:, 1]) < 0.1 * re[:, 0])
        print(f'mixed best T {T:g} mK: along phi {ok.sum() * 2 / len(phis):.2f} pi within 1/2 +- 0.03 (max kappa '
              f'{rp[:, 0].max():.3f}); along E_Z {oke.sum() * (ez[1] - ez[0]) * 1e3:.0f} ueV (max {re[:, 0].max():.3f})')
    for a in (a1, a2):
        half_line(a)
    ticks = [0.75, 1.0, 1.5, 0.0, 0.5]
    a1.set_xticks([int(np.argmin(np.abs(phis[order] - t))) for t in ticks])
    a1.set_xticklabels([f'{t:g}' for t in ticks])
    a1.set_xlabel(rf'phase $\phi/\pi$  (mod 2) at $E_Z$ = {EZ0:g} meV')
    a1.set_ylabel(r'$\kappa/\kappa_0$'); a1.set_ylim(0, 0.62); a1.set_yticks([0, 0.25, 0.5])
    a2.set_xlabel(rf'Zeeman energy $E_Z$ [meV] at $\phi$ = {phi_c:.2f}$\pi$')
    a2.axvline(EZ0, color=MUTED, lw=1, ls=':')
    h, l = a1.get_legend_handles_labels()                                   # drawn hot -> cold, |G| after 20 mK
    order_l = [l.index(f'{T:g} mK') for T in TS] + [i for i, s in enumerate(l) if 'G' in s]
    fig.legend([h[i] for i in order_l], [l[i] for i in order_l], loc='outside upper center', ncol=5)
    save(fig, 'mixed_best_temperature')


def mixed_detr(L=2000.0, TAUS=(0.3387, 0.5, 0.7)):
    """kappa on (phi, E_Z) with det r = 0 (white) and the bulk Q boundary (dashed), L 2 um, three probe bonds (study_tuning
    cell 7/8 coarse maps).  Boundary det r lines stay put when tau changes (bulk transitions), interior ones move (eps = Gamma)."""
    Qm, aq = load('scharf_absTune_Q_mixed_mu0.7')
    EZq, phq = aq['E_Z'], aq['phi'] / np.pi
    fig, axs = plt.subplots(1, len(TAUS), figsize=FULL, sharey=True, layout='constrained')
    for ax, tau in zip(axs, TAUS):
        V, a = load(f'scharf_absTune_tr_mixed_mu0.7_tau{tau:g}' + ('' if L == 2000.0 else f'_L{L / 1000:g}'))
        EZ, ph = a['E_Z'], np.linspace(0, 2, V.shape[1])
        im = ax.pcolormesh(ph, EZ, V[..., 0], shading='nearest', cmap='viridis', vmin=0, vmax=1, rasterized=True)
        ax.contour(ph, EZ, V[..., 3], levels=[0], colors='white', linewidths=1.3)
        ax.contour(phq, EZq, Qm[..., 0], levels=[0], colors=RED, linewidths=1.6, linestyles='--')
        neu = (np.abs(V[..., 1]) < 0.1 * V[..., 0]) & (V[..., 0] > 0.25)
        ax.set_title(rf'probe $\tau$ = {tau:.2f}', fontsize=15)
        ax.set_xlabel(r'$\phi/\pi$'); ax.grid(False)
        print(f'mixed detr L {L / 1000:g} tau {tau}: kappa > 0.25 on {np.mean(V[..., 0] > 0.25):.1%}, neutral share '
              f'{neu.sum() / max((V[..., 0] > 0.25).sum(), 1):.0%}')
    axs[0].set_ylabel(r'$E_Z$ [meV]')
    fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$', shrink=0.9)
    fig.suptitle(rf'$L$ = {L / 1000:g} $\mu$m.   white: det $r$ = 0     red dashed: boundary of the bulk $Q = -1$ region',
                 fontsize=14, color=INK2)
    save(fig, 'mixed_detr_maps')


def mixed_detr_x(phi_c=1.7417):
    """det r as a readout of x = eps/Gamma along the E_Z cut through the best mixed plateau (study_mixed [plateau] cell 4,
    tau 0.5, L 2 um, phi_c).  Two-Majorana: det r = (x^2-1)/(x^2+1) is monotonic in x, kappa = 2x^2/(1+x^2)^2 folds at x = 1
    -> lobes with eps_max < Gamma give a kappa peak < 1/2 at a det r maximum (det r < 0, kappa and det r move together);
    lobes with eps_max > Gamma give kappa = 1/2 at each det r = 0 and a kappa dip at the eps top."""
    V, a = load(f'scharf_absPlat_cut_mu0.7_tau0.5_L2_phi{phi_c:.4f}pi')
    Q = load(f'scharf_absPlat_Qcut_mu0.7_phi{phi_c:.4f}pi')[0][:, 0]
    ez, k, G, d = a['E_Z'], V[:, 0], V[:, 1], V[:, 3]
    h, triv = ez[1] - ez[0], Q > 0
    x = np.sqrt((1 + np.clip(d, -1 + 1e-12, 1 - 1e-12)) / (1 - np.clip(d, -1 + 1e-12, 1 - 1e-12)))
    tops = [i for i in range(1, len(ez) - 1) if x[i] >= x[i - 1] and x[i] > x[i + 1] and 0.05 < x[i] < 1 and not triv[i]]
    cross = [i for i in range(len(ez) - 1) if np.sign(d[i]) != np.sign(d[i + 1]) and not (triv[i] or triv[i + 1])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, layout='constrained',
                                 gridspec_kw=dict(height_ratios=(1, 1.25)))
    edges = np.flatnonzero(np.diff(np.r_[0, triv.astype(int), 0]))
    for ax in (a1, a2):
        for i0, i1 in zip(edges[::2], edges[1::2]):
            ax.axvspan(ez[i0] - h / 2, ez[i1 - 1] + h / 2, color=GRID, lw=0, zorder=0)
        for i in cross:
            ax.axvline((ez[i] + ez[i + 1]) / 2, color=MUTED, lw=1, ls=':', zorder=1)
    a1.plot(ez, k, color=BLUE, label=r'$\kappa/\kappa_0$ (NEGF)')
    a1.plot(ez, (1 - d**2) / 2, color=INK, lw=1.3, ls='--', label=r'$(1-\det r^2)/2$')
    half_line(a1)
    for i in tops:
        a1.plot(ez[i], k[i], 'o', ms=8, color=BLUE, mec='white', mew=1.5, zorder=4)
        a1.annotate(f'{k[i]:.3f}', (ez[i], k[i]), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=12)
    if edges.size:
        a1.text(ez[edges[0]] + 0.005, 0.66, 'Q = +1\n(trivial)', fontsize=12, color=INK2, va='top')
    a1.set_ylim(0, 0.75); a1.set_yticks([0, 0.25, 0.5]); a1.set_ylabel(r'$\kappa/\kappa_0$')
    a1.legend(loc='upper left', ncol=2)
    a2.plot(ez, np.where(triv, np.nan, x), color=AQUA, label=r'$x = \varepsilon/\Gamma$')
    a2.plot(ez, np.where(triv, x, np.nan), color=MUTED, lw=1.5, ls=(0, (2, 2)))
    a2.axhline(1.0, color=INK2, lw=1, ls='--')
    a2.text(ez[0] + 0.005, 1.12, r'$\varepsilon = \Gamma$:  det $r$ = 0,  $\kappa$ = 1/2', fontsize=12, color=INK2)
    for i in tops:
        a2.plot(ez[i], x[i], 'o', ms=8, color=AQUA, mec='white', mew=1.5, zorder=4)
        a2.annotate(f'top {x[i]:.2f}', (ez[i], x[i]), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=12)
    a2.set_yscale('log'); a2.set_ylim(5e-3, 8)
    a2.set_ylabel(r'$x = \sqrt{(1+\det r)/(1-\det r)}$')
    sec = a2.twinx()                                        # the same axis, labelled in det r (x <-> det r is one-to-one)
    sec.set_yscale('log'); sec.set_ylim(a2.get_ylim()); sec.grid(False); sec.spines['right'].set_visible(True)
    dt = np.array([-0.9999, -0.999, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9])
    sec.set_yticks(np.sqrt((1 + dt) / (1 - dt)), [f'{v:+g}' if v else '0' for v in dt])
    sec.yaxis.set_minor_locator(plt.NullLocator()); sec.set_ylabel(r'det $r$')
    a2.set_xlabel(rf'Zeeman energy $E_Z$ [meV] at $\phi$ = {phi_c:.2f}$\pi$   ($\tau$ 0.5, L 2 $\mu$m, a 10)')
    a2.set_xlim(ez[0], ez[-1])
    save(fig, 'mixed_detr_x')
    res = np.abs(k - (1 - d**2) / 2)
    print(f'mixed_detr_x: two-Majorana residual max {res[~triv].max():.4f} in Q = -1, {res[triv].max():.3f} in Q = +1')
    print('  lobe tops below Gamma: ' + ', '.join(f'E_Z {ez[i]:.3f} x {x[i]:.2f} kappa {k[i]:.3f} det r {d[i]:+.3f}' for i in tops))
    print('  det r = 0 (Q = -1): ' + ', '.join(f'{(ez[i] + ez[i + 1]) / 2:.4f} (kappa {max(k[i], k[i + 1]):.3f})' for i in cross))


# %% --- RUN ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:                    # python slides_nov.py mixed_lengths mixed_oscillation ...
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    overlap_L()
    overlap()
    slide4()
    slide5()
    slide6()
    slide9()
    slide10()
    collapse_mixed()
    length_detr_mixed()
