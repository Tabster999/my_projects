"""
Topology, couplings and tuning at a = 10 nm: the topological regime, the S/N and probe couplings, and the mu_C / SOC
/ L / tau survey (incl. the mixed SOC maps).

PARTS (one per former script; the cells of each part are tagged [part]):
  topology     (was study_topology.py)  STUDY: can the topological regime be reached with the coupling-study parameters (study_coupling.py)?
  coupling     (was study_coupling.py)  STUDY: how the S/N interface coupling tc (tc_top = tc_bot; 1 = bulk bond, Scharf) and the probe coupling
  tuning       (was study_tuning.py)  STUDY: can mu_C and the spin-orbit coupling make the kappa = 1/2 regions larger?

Run one part:   python study_tuning.py topology      (several: list them; no argument: every part, in order)
In the interactive window the cells run as usual.  run_part(part) replaces the old `if __name__ == "__main__":`
guard: the parallel workers re-import this file and must not re-run the cells.
Names that collided between parts were renamed (see each PART header); cache names are unchanged.
"""
# %% --- IMPORTS AND THE PART SELECTOR ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tools as st
from study_oscillation import slowest_mode

PARTS = ('topology', 'coupling', 'tuning')
_SELECTED = {a for a in sys.argv[1:] if a in PARTS}


def run_part(part):
    """True for the cells of `part` when this file runs as a script: every part, or only those named on
    the command line.  False in the parallel workers (they import this file)."""
    return __name__ == "__main__" and (not _SELECTED or part in _SELECTED)


# ====================================================================================================================
# PART 'topology'  (was study_topology.py)   renamed: EZ -> EZ_TOPO, junction -> junction_topo, MUS -> MUS_MAP, BEST -> BEST_TOPO
# ====================================================================================================================
# STUDY: can the topological regime be reached with the coupling-study parameters (study_coupling.py)?
# W 100, L 2 um, pure Rashba alpha 16, Delta 0.25, a = 10, phi = pi, infinite junction (cross-section, W_S 900).
# "Reached" needs three things, all from the full cross-section model:
#   Q = -1                 invariant (tools.invariant)
#   Delta_top              gap over all k along the junction (tools.gap_refined: grid + 1D refinement)
#   L / xi_M >> 1          xi_M = decay length of the slowest E = 0 solution at complex k
#                          (study_oscillation.slowest_mode, exact); at L = 2 um the two ends overlap ~ e^{-L/xi_M}
# Scans: (mu_C, E_Z) at mu_S = 1 and (mu_S, E_Z) at mu_C = 0.7, for the S/N couplings tc 0.5 / 0.75 / 1.
# Working point of every study so far: mu_C = 0.7, mu_S = 1.0.  The probe leads (mu_N) are not part of
# the cross-section and cannot change Q.

# %% [topology] --- PARAMETERS ---

A = 10.0
L_NM = 2000.0
TCS = (0.5, 0.75, 1.0)
EZ_TOPO = np.round(np.arange(0.0, 2.0001, 0.1), 3)
MU_C = np.round(np.arange(0.0, 2.4001, 0.1), 3)
MU_S = np.round(np.arange(0.5, 3.0001, 0.25), 3)
MU_C0, MU_S0 = 0.7, 1.0                            # the working point


def junction_topo(tc):
    return st.replace(st.case(100.0, a=A), tc=tc)


def topo_point(q, E_Z, mu_C=None, mu_S=None, phi=np.pi):
    """[Q, Delta_top (meV), k of the gap minimum (1/nm)]; the gap only inside Q = -1 (NaN outside)."""
    q = st.replace(q, **{k: v for k, v in (('mu_C', mu_C), ('mu_S', mu_S)) if v is not None})
    p, n_S = q.params(E_Z=float(E_Z), phi=float(phi)), st.n_S_of(q)
    Q = st.invariant(p, n_S)
    if Q > 0:
        return np.array([Q, np.nan, np.nan])
    g, k = st.gap_refined(p, n_S, st.k_grid(q.a))
    return np.array([Q, g * q.t, k / q.a])


def xi_point(q, i, mu_C, E_Z, phi=np.pi):
    """[Q, xi_M (nm)] at (mu_C[i], E_Z[i])."""
    q = st.replace(q, mu_C=float(mu_C[i]))
    p, n_S = q.params(E_Z=float(E_Z[i]), phi=float(phi)), st.n_S_of(q)
    return np.array([st.invariant(p, n_S), -q.a / np.log(np.abs(slowest_mode(p, n_S)))])


def xi_scan(q, mu_C, E_Z, cache):
    """xi_point at the listed (mu_C, E_Z) pairs: (len, 2)."""
    if not len(mu_C):
        return np.empty((0, 2))
    return st.run_scan(partial(xi_point, q, mu_C=np.asarray(mu_C), E_Z=np.asarray(E_Z)),
                       {'i': np.arange(len(mu_C))}, cache)


# %% [topology] --- 1. SCANS: Q and Delta_top on (mu_C, E_Z) and (mu_S, E_Z) ---
if run_part('topology'):
    MUC, MUS_MAP = {}, {}
    for tc in TCS:
        MUC[tc] = st.run_scan(partial(topo_point, junction_topo(tc)), {'mu_C': MU_C, 'E_Z': EZ_TOPO},
                              f'scharf_absTopo_muC_a{A:g}_tc{tc:g}')
        print(f"tc {tc:g}  (mu_C, E_Z) done", flush=True)
    for tc in TCS:
        MUS_MAP[tc] = st.run_scan(partial(topo_point, st.replace(junction_topo(tc), mu_C=MU_C0)), {'mu_S': MU_S, 'E_Z': EZ_TOPO},
                              f'scharf_absTopo_muS_a{A:g}_tc{tc:g}')
        print(f"tc {tc:g}  (mu_S, E_Z) done", flush=True)


# %% [topology] --- 2. xi_M: along E_Z at the working point, and at the largest gap of every mu_C ---
if run_part('topology'):
    XI0, XIB, BEST_TOPO = {}, {}, {}
    i0 = int(np.argmin(np.abs(MU_C - MU_C0)))
    for tc in TCS:
        line = np.flatnonzero(MUC[tc][i0, :, 0] < 0)               # topological fields at mu_C = 0.7
        XI0[tc] = (EZ_TOPO[line], xi_scan(junction_topo(tc), np.full(len(line), MU_C0), EZ_TOPO[line],
                                     f'scharf_absTopo_xi0_a{A:g}_tc{tc:g}'))
        g = np.where(MUC[tc][..., 0] < 0, MUC[tc][..., 1], -1.0)
        rows = np.flatnonzero(g.max(axis=1) > 0)                   # mu_C with any topological field
        BEST_TOPO[tc] = (MU_C[rows], EZ_TOPO[np.argmax(g[rows], axis=1)], g[rows].max(axis=1))
        XIB[tc] = xi_scan(junction_topo(tc), *BEST_TOPO[tc][:2], f'scharf_absTopo_xibest_a{A:g}_tc{tc:g}')
        print(f"tc {tc:g}  xi_M done", flush=True)


# %% [topology] --- 3. NUMBERS ---
if run_part('topology'):
    i0 = int(np.argmin(np.abs(MU_C - MU_C0)))
    print(f"\nWORKING POINT mu_C = {MU_C0}, mu_S = {MU_S0}, phi = pi, a = {A:g}, L = {L_NM:g} nm")
    print("  tc    Q=-1 at E_Z [meV]     max Delta_top [ueV] at E_Z    xi_M [nm] (min .. max)   L/xi_M (max)")
    for tc in TCS:
        Qr, gr = MUC[tc][i0, :, 0], MUC[tc][i0, :, 1]
        ez_t, xi = XI0[tc][0], XI0[tc][1][:, 1]
        if not (Qr < 0).any():
            print(f"  {tc:4.2f}  none"); continue
        j = int(np.nanargmax(gr))
        print(f"  {tc:4.2f}  {EZ_TOPO[Qr < 0].min():.1f} .. {EZ_TOPO[Qr < 0].max():.1f}  ({(Qr < 0).sum()} of {len(EZ_TOPO)})"
              f"      {gr[j] * 1e3:6.1f}  at {EZ_TOPO[j]:.1f}            {xi.min():6.0f} .. {xi.max():6.0f}"
              f"          {L_NM / xi.min():.2f}")
    print("\nOVER mu_C (mu_S = 1): Q = -1 fraction of the map, and the best point of each tc")
    print("  tc    Q=-1 frac   mu_C with Q=-1        best Delta_top [ueV] at (mu_C, E_Z)   its xi_M [nm]  L/xi_M")
    for tc in TCS:
        Qm = MUC[tc][..., 0]
        mu, ez, g = BEST_TOPO[tc]
        b = int(np.argmax(g))
        print(f"  {tc:4.2f}  {np.mean(Qm < 0):.3f}       {mu.min():.1f} .. {mu.max():.1f}  ({len(mu)} of {len(MU_C)})"
              f"     {g[b] * 1e3:6.1f}  at ({mu[b]:.1f}, {ez[b]:.1f})             {XIB[tc][b, 1]:6.0f}"
              f"        {L_NM / XIB[tc][b, 1]:.2f}")
        print(f"        largest L/xi_M over mu_C: {L_NM / XIB[tc][:, 1].min():.2f} at mu_C = "
              f"{mu[np.argmin(XIB[tc][:, 1])]:.1f} (E_Z {ez[np.argmin(XIB[tc][:, 1])]:.1f}, "
              f"Delta_top {g[np.argmin(XIB[tc][:, 1])] * 1e3:.1f} ueV)")
    print("\nOVER mu_S (mu_C = 0.7): Q = -1 fraction and max Delta_top")
    for tc in TCS:
        Qm, gm = MUS_MAP[tc][..., 0], MUS_MAP[tc][..., 1]
        print(f"  tc {tc:4.2f}: Q=-1 frac {np.mean(Qm < 0):.3f}   max Delta_top {np.nanmax(gm) * 1e3:.1f} ueV at mu_S "
              f"{MU_S[np.unravel_index(np.nanargmax(gm), gm.shape)[0]]:.2f}, E_Z "
              f"{EZ_TOPO[np.unravel_index(np.nanargmax(gm), gm.shape)[1]]:.1f}")


# %% [topology] --- FIGURES ---
if run_part('topology'):
    head = rf'W 100, a = {A:g}, $\alpha$ = 16, $\Delta$ = 0.25, $\phi = \pi$; grey: Q = +1 (trivial)'
    for name, D, axis, x0, lab in (('muC', MUC, MU_C, MU_C0, r'$\mu_C$ [meV] (junction)'),
                                   ('muS', MUS_MAP, MU_S, MU_S0, r'$\mu_S$ [meV] (superconductor)')):
        fig, axs = plt.subplots(1, len(TCS), figsize=(4.2 * len(TCS), 4.4), sharey=True, squeeze=False)
        fig.subplots_adjust(top=0.8, bottom=0.14)
        for c, tc in enumerate(TCS):
            ax = axs[0, c]
            ax.pcolormesh(axis, EZ_TOPO, (D[tc][..., 0] > 0).T.astype(float), shading='nearest', cmap='Greys',
                          vmin=0, vmax=2.5)
            im = ax.pcolormesh(axis, EZ_TOPO, D[tc][..., 1].T * 1e3, shading='nearest', cmap='viridis', vmin=0,
                               vmax=max(np.nanmax(D[t][..., 1]) for t in TCS) * 1e3)
            ax.axvline(x0, color='r', lw=0.8, ls='--')
            ax.set_title(rf'$t_c$ = {tc:g}', fontsize=10)
            ax.set_xlabel(lab)
        axs[0, 0].set_ylabel(r'$E_Z$ [meV]')
        fig.colorbar(im, ax=axs, label=r'$\Delta_{top}$ [$\mu$eV] (Q = $-1$)', pad=0.01)
        fig.suptitle(f'topological gap over {lab.split(" [")[0]} and $E_Z$; red dashed: working point\n' + head,
                     fontsize=10)
        st.save_fig(fig, f'topology_{name}'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for tc in TCS:
        ez_t, xi = XI0[tc][0], XI0[tc][1][:, 1]
        axs[0].semilogy(ez_t, xi, 'o-', ms=3, label=rf'$t_c$ = {tc:g}')
        mu, ez, g = BEST_TOPO[tc]
        axs[1].semilogy(mu, XIB[tc][:, 1], 'o-', ms=3, label=rf'$t_c$ = {tc:g}')
    for ax in axs:
        ax.axhline(L_NM, color='k', lw=0.8, ls='--')
        ax.set_ylabel(r'$\xi_M$ [nm]  (dashed: L)')
        ax.legend(fontsize=8)
    axs[0].set_xlabel(r'$E_Z$ [meV]'); axs[0].set_title(rf'$\mu_C$ = {MU_C0}: every topological field', fontsize=10)
    axs[1].set_xlabel(r'$\mu_C$ [meV]'); axs[1].set_title(r'at the largest $\Delta_{top}$ of each $\mu_C$', fontsize=10)
    fig.suptitle('Majorana decay length (slowest E = 0 solution at complex k)\n' + head, fontsize=10)
    fig.tight_layout()
    st.save_fig(fig, 'topology_xi'); plt.close(fig)


# ====================================================================================================================
# PART 'coupling'  (was study_coupling.py)   renamed: TAUS -> TAUS_SN, junction -> junction_sn, EZ -> EZ_SN, PHI -> PHI_SN
# ====================================================================================================================
# STUDY: how the S/N interface coupling tc (tc_top = tc_bot; 1 = bulk bond, Scharf) and the probe coupling
# tc_barr change the (E_Z, phi) maps of the total DOS at E = 0, kappa and G -- and how both couplings change
# when the lattice constant goes from a = 10 to a = 5 nm (the SAME bond values at both a, on purpose: a bond's
# transparency depends on a, and this shows by how much).
# Prediction (advisor): the topological regime grows a lot with tc -- meant as the extent of the kappa = 1/2
# PLATEAUS (flat tops where eps ~ Gamma over a range of E_Z), not Q.
# Plateau point: |kappa - 1/2| < 0.03 and |G|/kappa < 0.1.
#
# W 100, L 2 um, pure Rashba alpha 16, E = 0, T = 0.  Probe contact: one weak bond tc_barr_transport (decision
# 2026-11-02; barrier models live only in study_a).  Per field ONE set-up serves every phase:
#   DOS   tools.dos_total_phis: probes decoupled, eta = 0.05 Delta (Scharf), summed over ALL centre sites
#   kappa, G, local Andreev, det r   tools.transport_phis
#   Q     invariant of the infinite cross-section (same tc, same a), white dashed line on the maps
# DOS for 4 S/N couplings, transport for 2 S/N x 4 probe couplings, each at a = 10 and a = 5.
# a = 5 is trimmed (~5 min of set-up per field point): coarser E_Z and phi grids (GRID), two probe couplings,
# transport before DOS, Q only for TCS_TR.

# %% [coupling] --- PARAMETERS ---

AS = (10.0, 5.0)
TCS_DOS = (0.5, 0.75, 0.9, 1.0)                    # S/N bond, DOS maps
TCS_TR = (0.75, 1.0)                               # S/N bond, transport maps
TAUS_SN = {10.0: (0.2, 0.3387, 0.5, 0.7), 5.0: (0.3387, 0.7)}   # probe bond tc_barr_transport
GRID = {10.0: (np.round(np.arange(0.0, 2.0001, 0.025), 3), np.linspace(0.0, 2 * np.pi, 61)),   # (E_Z, phi)
        5.0: (np.round(np.arange(0.0, 2.0001, 0.05), 3), np.linspace(0.0, 2 * np.pi, 31))}
Q_GRID = {10.0: (np.round(np.linspace(0, 2.4, 49), 3), np.linspace(0, 2 * np.pi, 25)),   # ~5 s / point
          5.0: (np.round(np.linspace(0, 2.4, 25), 3), np.linspace(0, 2 * np.pi, 13))}    # ~8x dearer
ETA_DOS = 0.05 * 0.25                              # meV
WORKERS = {10.0: 14, 5.0: 8}                       # a = 5: ~2.6 GB per worker


def junction_sn(a, tc, tau=0.3387):
    """S/N bond tc, probe bond tau, lattice constant a."""
    return st.replace(st.case(100.0, a=a), tc=tc, tc_barr_transport=tau)


# %% [coupling] --- 1. SCANS (cached in plot_npz/scharf_absSN_*); a = 10 first, the Q maps last ---
if run_part('coupling'):
    DOS, TR, Q = {}, {}, {}
    for a in AS:
        EZ_SN, PHI_SN = GRID[a]
        for tc in TCS_TR:
            for tau in TAUS_SN[a]:
                TR[(a, tc, tau)] = st.run_scan(partial(st.transport_phis, junction_sn(a, tc, tau), phis=PHI_SN),
                                               {'E_Z': EZ_SN}, f'scharf_absSN_tr_a{a:g}_tc{tc:g}_tau{tau:g}',
                                               n_workers=WORKERS[a])
                print(f"a {a:g}  transport tc {tc:g} tau {tau:g} done", flush=True)
        for tc in TCS_DOS:
            DOS[(a, tc)] = st.run_scan(partial(st.dos_total_phis, junction_sn(a, tc), phis=PHI_SN, eta=ETA_DOS),
                                       {'E_Z': EZ_SN}, f'scharf_absSN_dosT_a{a:g}_tc{tc:g}', n_workers=WORKERS[a])
            print(f"a {a:g}  DOS tc {tc:g} done", flush=True)
    for a in AS:
        for tc in (TCS_DOS if a == 10.0 else TCS_TR):
            Q[(a, tc)] = st.q_map(junction_sn(a, tc), *Q_GRID[a], f'scharf_absSN_Qcoarse_a{a:g}_tc{tc:g}')
            print(f"a {a:g}  Q tc {tc:g} done", flush=True)


# %% [coupling] --- 2. NUMBERS: plateau size per coupling and lattice constant ---
if run_part('coupling'):
    print("\n  a    tc    tau     plateau frac   plateau E_Z length at phi=pi [meV]   kappa>0.25 frac   "
          "neutral of those   failed solves")
    for (a, tc, tau), tr in TR.items():
        EZ_SN, PHI_SN = GRID[a]
        ip = np.argmin(np.abs(PHI_SN - np.pi))
        k, G = tr[..., 0], tr[..., 1]
        P = (np.abs(k - 0.5) < 0.03) & (np.abs(G) < 0.1 * k)
        hot = k > 0.25
        print(f"  {a:3g}  {tc:4.2f}  {tau:.4f}    {P.mean():.3f}          {P[:, ip].sum() * (EZ_SN[1] - EZ_SN[0]):.3f}"
              f"                            {hot.mean():.3f}             "
              f"{np.mean(np.abs(G[hot]) < 0.1 * k[hot]) if hot.any() else np.nan:.2f}"
              f"              {int(np.nansum(tr[..., 4]))}")
    # two-Majorana test without fitting: det r = (x^2-1)/(x^2+1), kappa = 2x^2/(1+x^2)^2  =>  kappa = (1 - det r^2)/2.
    # Only meaningful at large kappa (weak normal transmission satisfies it to O(kappa^2)); kappa = 1/2 by a
    # plain electron channel would miss it by +0.125.
    print("\n  a    tc    tau      kappa > 0.3: neutral n, |kappa - (1-det^2)/2| median    charged n, median")
    for (a, tc, tau), tr in TR.items():
        k, G, dr = (tr[..., j].ravel() for j in (0, 1, 3))
        res, big, neu = k - (1 - dr**2) / 2, k > 0.3, np.abs(G) < 0.1 * k
        print(f"  {a:3g}  {tc:4.2f}  {tau:.4f}   {(big & neu).sum():5d}  {np.median(np.abs(res[big & neu])):.4f}"
              f"                         {(big & ~neu).sum():5d}  {np.median(res[big & ~neu]) if (big & ~neu).any() else np.nan:+.3f}")
    print("\n  a    tc    Q=-1 fraction (E_Z <= 2)   Q=-1 at phi=pi (E_Z range)")
    for (a, tc), Qm in Q.items():
        ez, ph = Q_GRID[a]
        line = ez[Qm[:, np.argmin(np.abs(ph - np.pi))] < 0]
        print(f"  {a:3g}  {tc:4.2f}   {np.mean(Qm[ez <= GRID[a][0][-1]] < 0):.3f}                     "
              f"{line.min() if line.size else np.nan:.3f} .. {line.max() if line.size else np.nan:.3f}")


# %% [coupling] --- FIGURES ---
def plot_2M_test(TR, name='coupling_2M_test'):
    """kappa against the two-Majorana value (1 - det r^2)/2 for every map in TR (rows a), colour |G|/kappa."""
    from matplotlib.colors import LogNorm
    keys = {a: [k for k in TR if k[0] == a] for a in AS}
    ncol = max(len(v) for v in keys.values())
    fig, axs = plt.subplots(len(AS), ncol, figsize=(3.1 * ncol + 1.2, 3.1 * len(AS) + 0.6), sharex=True, sharey=True,
                            squeeze=False, layout='constrained')
    for r, a in enumerate(AS):
        for c in range(ncol):
            ax = axs[r, c]
            if c >= len(keys[a]):
                ax.set_visible(False); continue
            _, tc, tau = keys[a][c]
            k, G, dr = (TR[keys[a][c]][..., j].ravel() for j in (0, 1, 3))
            m = k > 0.01
            sc = ax.scatter((1 - dr[m]**2) / 2, k[m], c=np.clip(np.abs(G[m]) / k[m], 1e-4, 1), s=2, cmap='plasma',
                            norm=LogNorm(1e-4, 1), rasterized=True)
            ax.plot([0, 0.5], [0, 0.5], 'k--', lw=0.7)
            ax.axhline(0.5, color='g', lw=0.5)
            ax.set_title(f'a {a:g}, $t_c$ {tc:g}, tc_barr {tau:g}', fontsize=8)
        axs[r, 0].set_ylabel(r'$\kappa/\kappa_0$ (full model)')
    for ax in axs[-1]:
        ax.set_xlabel(r'$(1-\det r^2)/2$')
    fig.colorbar(sc, ax=axs, label=r'$|G|/\kappa$')
    fig.suptitle(r'two-Majorana test: on the dashed line $\kappa = (1-\det r^2)/2$; only meaningful at large $\kappa$',
                 fontsize=10)
    st.save_fig(fig, name); plt.close(fig)


def _panel(ax, Z, a, tc, title, **kw):
    EZ_SN, PHI_SN = GRID[a]
    im = ax.pcolormesh(PHI_SN / np.pi, EZ_SN, Z, shading='nearest', rasterized=True, **kw)
    if (a, tc) in Q:
        ez, ph = Q_GRID[a]
        ax.contour(ph / np.pi, ez, Q[(a, tc)], levels=[0], colors='w', linewidths=0.8, linestyles='--')
    ax.set_ylim(EZ_SN[0], EZ_SN[-1])
    ax.set_title(title, fontsize=9)
    return im


if run_part('coupling'):
    from matplotlib.colors import LogNorm
    head = r'W 100, L 2 $\mu$m, $\alpha$ = 16, E = 0, T = 0; white dashed: Q = $-1$ boundary (infinite junction); red: plateau points'

    # DOS: rows a, columns S/N coupling, one colour scale
    lo, hi = min(np.nanmin(v) for v in DOS.values()), max(np.nanmax(v) for v in DOS.values())
    fig, axs = plt.subplots(len(AS), len(TCS_DOS), figsize=(3.8 * len(TCS_DOS), 3.6 * len(AS)),
                            sharex=True, sharey=True, squeeze=False)
    for r, a in enumerate(AS):
        for c, tc in enumerate(TCS_DOS):
            im = _panel(axs[r, c], DOS[(a, tc)], a, tc, rf'a = {a:g} nm, $t_c$ = {tc:g}', norm=LogNorm(lo, hi),
                        cmap='magma')
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi/\pi$')
    fig.colorbar(im, ax=axs, label='total DOS($E=0$) [1/meV], probes decoupled', pad=0.01)
    fig.suptitle('total DOS at E = 0 vs S/N coupling $t_c$ (columns) and lattice constant (rows)\n' + head,
                 fontsize=10)
    st.save_fig(fig, 'coupling_dos'); plt.close(fig)

    # kappa and G: one figure per a and quantity, rows S/N coupling, columns probe coupling
    for a in AS:
        for name, j, kw in (('kappa', 0, dict(vmin=0, vmax=1, cmap='viridis')),
                            ('G', 1, dict(vmin=0, vmax=1, cmap='cividis'))):       # |G|: G < -0.01 on <= 3% of points
            fig, axs = plt.subplots(len(TCS_TR), len(TAUS_SN[a]), figsize=(3.8 * len(TAUS_SN[a]), 3.6 * len(TCS_TR)),
                                    sharex=True, sharey=True, squeeze=False)
            for r, tc in enumerate(TCS_TR):
                for c, tau in enumerate(TAUS_SN[a]):
                    im = _panel(axs[r, c], np.abs(TR[(a, tc, tau)][..., j]), a, tc,
                                rf'$t_c$ = {tc:g}, tc_barr = {tau:g}', **kw)
                    k_, G_ = TR[(a, tc, tau)][..., 0], TR[(a, tc, tau)][..., 1]
                    P = (np.abs(k_ - 0.5) < 0.03) & (np.abs(G_) < 0.1 * k_)
                    if j == 0 and P.any():                   # red: plateau points
                        axs[r, c].contour(GRID[a][1] / np.pi, GRID[a][0], P.astype(float), levels=[0.5],
                                          colors='r', linewidths=0.6)
                axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
            for ax in axs[-1]:
                ax.set_xlabel(r'$\phi/\pi$')
            fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$' if j == 0 else r'$|G|/G_0$', pad=0.01)
            fig.suptitle(rf'{name} at a = {a:g} nm: S/N coupling $t_c$ (rows) x probe coupling (columns)' + '\n' + head,
                         fontsize=10)
            st.save_fig(fig, f'coupling_{name}_a{a:g}'); plt.close(fig)
    plot_2M_test(TR)


# ====================================================================================================================
# PART 'tuning'  (was study_tuning.py)
# ====================================================================================================================
# STUDY: can mu_C and the spin-orbit coupling make the kappa = 1/2 regions larger?
# A plateau point needs eps ~ Gamma (0.78 < eps/Gamma < 1.28), and eps = A Delta_out e^{-L/xi_M} |cos(k_osc L + phi0)|
# oscillates, so the regions are bands around the lobe tops.  Levers: a slower envelope (xi_M, Delta_out that
# change slowly with E_Z and phi) and a non-oscillating tail (the k ~ 0 branch, study_branches: stronger alpha).
# Pure Rashba alpha 16 / 32 / 48 and mixed Rashba + Dresselhaus (amplitude 16, theta 0.15 pi, '100'; the early
# Scharf set), each at mu_C 0.3 / 0.7 / 1.2; a = 10, W 100, L 2 um, S/N bond tc = 1 (bulk), probe tau 0.3387 / 0.5.
# Coarse grid (E_Z step 0.05, 31 phases) -- a survey: confirm the best cases on the study_coupling grid.
# Per junction also Q, Delta_top along phi = pi (study_topology.topo_point).

# %% [tuning] --- PARAMETERS ---

A = 10.0
EZ = np.round(np.arange(0.0, 2.0001, 0.05), 3)
PHI = np.linspace(0.0, 2 * np.pi, 31)
EZ_Q = np.round(np.arange(0.0, 2.0001, 0.1), 3)
TAUS = (0.3387, 0.5)
MUS = (0.3, 0.7, 1.2)
SOCS = {'R16': (16.0, 0.0), 'R32': (32.0, 0.0), 'R48': (48.0, 0.0), 'mixed': st.mixed_soc(16.0, 0.15 * np.pi)}


def junction(soc, mu_C, tau=0.3387, L=2000.0):
    alpha, beta = SOCS[soc]
    return st.replace(st.case(100.0, L=L, alpha=alpha, beta=beta, a=A), tc=1.0, mu_C=mu_C, tc_barr_transport=tau)


def plateau(tr):
    """Plateau points: |kappa - 1/2| < 0.03 and |G| < 0.1 kappa."""
    return (np.abs(tr[..., 0] - 0.5) < 0.03) & (np.abs(tr[..., 1]) < 0.1 * tr[..., 0])


# %% [tuning] --- 1. SCANS (cached in plot_npz/scharf_absTune_*) ---
if run_part('tuning'):
    TR, TOP = {}, {}
    for soc in SOCS:
        for mu in MUS:
            TOP[(soc, mu)] = st.run_scan(partial(topo_point, junction(soc, mu)), {'E_Z': EZ_Q},
                                         f'scharf_absTune_topo_{soc}_mu{mu:g}')
            for tau in TAUS:
                TR[(soc, mu, tau)] = st.run_scan(partial(st.transport_phis, junction(soc, mu, tau), phis=PHI),
                                                 {'E_Z': EZ}, f'scharf_absTune_tr_{soc}_mu{mu:g}_tau{tau:g}')
                print(f"{soc} mu_C {mu:g} tau {tau:g} done", flush=True)


# %% [tuning] --- 2. NUMBERS ---
if run_part('tuning'):
    print("\n  SOC     mu_C  tau      plateau frac  kappa>0.25 (neutral)  Q=-1 at phi=pi [meV]  max Delta_top [ueV]  failed")
    for (soc, mu, tau), tr in TR.items():
        k, G = tr[..., 0], tr[..., 1]
        hot, neu = k > 0.25, np.abs(G) < 0.1 * k
        Qr, gr = TOP[(soc, mu)][:, 0], TOP[(soc, mu)][:, 1]
        qr = f"{EZ_Q[Qr < 0].min():.1f}-{EZ_Q[Qr < 0].max():.1f}" if (Qr < 0).any() else "none"
        print(f"  {soc:6s}  {mu:4.1f}  {tau:.4f}   {plateau(tr).mean():.3f}         {hot.mean():.3f} ({np.mean(neu[hot]) if hot.any() else np.nan:.2f})"
              f"         {qr:10s}            {np.nanmax(gr) * 1e3 if (Qr < 0).any() else 0:6.1f}            {int(np.nansum(tr[..., 4]))}")


# %% [tuning] --- FIGURES: kappa per tau, rows SOC, columns mu_C ---
if run_part('tuning'):
    for tau in TAUS:
        fig, axs = plt.subplots(len(SOCS), len(MUS), figsize=(3.6 * len(MUS) + 1.2, 3.0 * len(SOCS) + 0.8),
                                sharex=True, sharey=True, squeeze=False, layout='constrained')
        for r, soc in enumerate(SOCS):
            for c, mu in enumerate(MUS):
                tr = TR[(soc, mu, tau)]
                im = axs[r, c].pcolormesh(PHI / np.pi, EZ, tr[..., 0], shading='nearest', vmin=0, vmax=1,
                                          cmap='viridis', rasterized=True)
                P = plateau(tr)
                if P.any():
                    axs[r, c].contour(PHI / np.pi, EZ, P.astype(float), levels=[0.5], colors='r', linewidths=0.6)
                axs[r, c].set_title(f'{soc}, $\\mu_C$ = {mu:g}: plateau {P.mean():.1%}', fontsize=9)
            axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
        for ax in axs[-1]:
            ax.set_xlabel(r'$\phi/\pi$')
        fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$')
        fig.suptitle(rf'$\kappa$, a = {A:g}, $t_c$ = 1, tc_barr = {tau:g}: SOC (rows) x $\mu_C$ (columns); red: plateau',
                     fontsize=10)
        st.save_fig(fig, f'tuning_kappa_tau{tau:g}'); plt.close(fig)


# %% [tuning] --- 3. MAXIMISE THE AREA: choose tau from Gamma ~ tau^p, then check with the full model (2026-11-03) ---
# On two-Majorana points det r gives x = eps/Gamma; the probe bond changes only Gamma, measured Gamma ~ tau^p with
# p from the two maps per junction.  One map then predicts the plateau fraction at any tau; the optimum is checked
# with NEGF maps.  Gamma itself from the local Andreev Lorentzian Gamma^2/(E^2+Gamma^2) at an isolated Majorana:
# a smaller Gamma means the plateaus need a lower temperature (kT << Gamma).
TAU_GRID = np.round(np.arange(0.03, 0.901, 0.01), 2)
TAU_OPT = (0.08, 0.10, 0.12)
BEST = (('R32', 0.3), ('R48', 0.3))
GAMMA_JOBS = tuple((s, m, t) for s, m in BEST for t in TAUS)       # E_Z 1.0, phi pi: det r = -0.999, eps << Gamma


def x_of(tr, k_min=0.02):
    """
    x = eps/Gamma from det r, and the points where it can be trusted: neutral, two-Majorana relation to 0.01, and
    kappa > k_min.  At kappa << 1 the relation holds for ANY weak transmission, and det r ~ -1 + small says nothing
    about eps (R32 mu_C 0.3, E_Z 1, phi pi: relation gives kappa 0.0013, NEGF 0.0001) -- extrapolating those
    points to a weak probe predicted 10-16 % plateau where NEGF gives 1-2.6 %.
    """
    k, G, d = tr[..., 0], tr[..., 1], np.clip(tr[..., 3], -1 + 1e-12, 1 - 1e-12)
    ok = (np.abs(G) < 0.1 * np.maximum(k, 1e-12)) & (np.abs(k - (1 - d**2) / 2) < 0.01) & (k > k_min)
    return np.sqrt((1 + d) / (1 - d)), ok


def gamma_exponent(tr_a, tr_b, tau_a, tau_b):
    """p of Gamma ~ tau^p from x at the same points of two maps (two-Majorana, |det r| < 0.98, kappa > 0.01 in both)."""
    (xa, oa), (xb, ob) = x_of(tr_a), x_of(tr_b)
    m = oa & ob
    for tr in (tr_a, tr_b):
        m &= (np.abs(tr[..., 3]) < 0.98) & (tr[..., 0] > 0.01)
    return np.log(xa[m] / xb[m]) / np.log(tau_b / tau_a)


def predicted_fraction(tr, tau_ref, taus, p):
    """Plateau fraction expected at each tau: x -> x (tau_ref/tau)^p on the two-Majorana points of the map."""
    x, ok = x_of(tr)
    out = []
    for t in taus:
        xp = x * (tau_ref / t) ** p
        out.append(np.mean(ok & (np.abs(2 * xp**2 / (1 + xp**2) ** 2 - 0.5) < 0.03)))
    return np.array(out)


def gamma_job(k, E_Z=1.0, phi=np.pi):
    """[Gamma (ueV), andreev(E=0)/2] for GAMMA_JOBS[k]: where (eh_local + he_local)/2 falls to half (bisection in ln E)."""
    s, m, t = GAMMA_JOBS[int(k)]
    q = junction(s, m, t)
    A = lambda E: (lambda c: (c['eh_local'] + c['he_local']) / 2)(dict(zip(st.CHANNELS, st.channel_point(q, E_Z, E, (phi,)))))
    A0, lo, hi = A(0.0), 1e-5, 1e-1                                   # meV (1e-2 clipped Gamma at tau 0.5)
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if A(mid) > A0 / 2 else (lo, mid)
    return np.array([np.sqrt(lo * hi) * 1e3, A0])


if run_part('tuning'):
    P_ALL = np.concatenate([gamma_exponent(TR[(s, m, TAUS[0])], TR[(s, m, TAUS[1])], *TAUS) for s in SOCS for m in MUS])
    P_EXP = float(np.median(P_ALL))
    print(f"\nGamma ~ tau^p from x(det r): p = {P_EXP:.2f} (IQR {np.percentile(P_ALL, 25):.2f}-{np.percentile(P_ALL, 75):.2f},"
          f" {P_ALL.size} points)")
    PRED = {(s, m): predicted_fraction(TR[(s, m, TAUS[0])], TAUS[0], TAU_GRID, P_EXP) for s in SOCS for m in MUS}
    GAM = st.run_scan(gamma_job, {'k': np.arange(len(GAMMA_JOBS))}, 'scharf_absTune_gamma', n_workers=len(GAMMA_JOBS))
    for s, m in BEST:
        for t in TAU_OPT:
            TR[(s, m, t)] = st.run_scan(partial(st.transport_phis, junction(s, m, t), phis=PHI), {'E_Z': EZ},
                                        f'scharf_absTune_tr_{s}_mu{m:g}_tau{t:g}')
            print(f"{s} mu_C {m:g} tau {t:g} done", flush=True)
    print("\n  junction      tau    predicted   measured (NEGF)   kappa>0.25 (neutral)   Gamma [ueV] / [mK]")
    for s, m in BEST:
        g = {t: GAM[GAMMA_JOBS.index((s, m, t)), 0] for t in TAUS}
        p_dir = np.log(g[TAUS[1]] / g[TAUS[0]]) / np.log(TAUS[1] / TAUS[0])
        for t in TAUS + TAU_OPT:
            tr = TR[(s, m, t)]
            hot, neu = tr[..., 0] > 0.25, np.abs(tr[..., 1]) < 0.1 * tr[..., 0]
            G_t = g[TAUS[0]] * (t / TAUS[0]) ** p_dir
            print(f"  {s:5s} {m:3.1f}   {t:.4f}   {PRED[(s, m)][np.argmin(np.abs(TAU_GRID - t))]:.3f}       {plateau(tr).mean():.3f}"
                  f"            {hot.mean():.3f} ({np.mean(neu[hot]) if hot.any() else np.nan:.2f})"
                  f"            {G_t:7.3f} / {G_t * 11.6:6.2f}{'  (measured)' if t in TAUS else ''}")
        print(f"        Gamma measured {g[TAUS[0]]:.3f} / {g[TAUS[1]]:.3f} ueV at tau {TAUS[0]} / {TAUS[1]}: p = {p_dir:.2f}")


# %% [tuning] --- FIGURES of cell 3 ---
if run_part('tuning'):
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    for (s, m), pr in PRED.items():
        ln, = ax.plot(TAU_GRID, pr, lw=1, ls='-' if m == 0.3 else ('--' if m == 0.7 else ':'), label=rf'{s}, $\mu_C$ {m:g}')
        meas = [(t, plateau(TR[(s, m, t)]).mean()) for t in TAUS + TAU_OPT if (s, m, t) in TR]
        ax.plot(*zip(*meas), 'o', ms=4, color=ln.get_color())
    ax.set_xscale('log'); ax.set_xlabel('probe bond tc_barr'); ax.set_ylabel('plateau fraction of the (E_Z, phi) map')
    ax.set_title(f'lines: predicted from the tau = {TAUS[0]} maps with Gamma ~ tau^{P_EXP:.2f}; dots: NEGF', fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    st.save_fig(fig, 'tuning_tau_opt'); plt.close(fig)

    fig, axs = plt.subplots(len(BEST), len(TAU_OPT), figsize=(3.6 * len(TAU_OPT) + 1.2, 3.3 * len(BEST) + 0.6),
                            sharex=True, sharey=True, squeeze=False, layout='constrained')
    for r, (s, m) in enumerate(BEST):
        for c, t in enumerate(TAU_OPT):
            tr = TR[(s, m, t)]
            im = axs[r, c].pcolormesh(PHI / np.pi, EZ, tr[..., 0], shading='nearest', vmin=0, vmax=1, cmap='viridis',
                                      rasterized=True)
            P = plateau(tr)
            if P.any():
                axs[r, c].contour(PHI / np.pi, EZ, P.astype(float), levels=[0.5], colors='r', linewidths=0.6)
            axs[r, c].set_title(rf'{s}, $\mu_C$ {m:g}, tc_barr {t:g}: plateau {P.mean():.1%}', fontsize=9)
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi/\pi$')
    fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$')
    fig.suptitle(r'$\kappa$ at the predicted optimum (a = 10, $t_c$ = 1); red: plateau points', fontsize=10)
    st.save_fig(fig, 'tuning_kappa_best'); plt.close(fig)


# %% [tuning] --- 4. THE LENGTH: a shorter junction compresses ln eps (2026-11-03) ---
# eps ~ A Delta_out e^{-L/xi_M}: across the diamond ln eps spreads by ~ L * Delta(1/xi_M), so a fixed Gamma matches
# a band whose share grows as L shrinks -- until xi_M > L makes the transport charged.  Best junction of cell 2.
LS = (1000.0, 1500.0, 2000.0)
TAUS_L = (0.3387, 0.5, 0.7)
JL = ('R48', 0.7)

if run_part('tuning'):
    for L in LS:
        for t in TAUS_L:
            name = (f'scharf_absTune_tr_{JL[0]}_mu{JL[1]:g}_tau{t:g}' if L == 2000.0 else
                    f'scharf_absTune_tr_{JL[0]}_mu{JL[1]:g}_tau{t:g}_L{L / 1000:g}')
            TR[(JL[0], JL[1], t, L)] = st.run_scan(partial(st.transport_phis, junction(*JL, t, L), phis=PHI),
                                                   {'E_Z': EZ}, name)
            print(f"L {L / 1000:g} um tau {t:g} done", flush=True)
    print(f"\n  {JL[0]} mu_C {JL[1]:g}:  L [um]  tau      plateau frac  kappa>0.25 (neutral)  2M test neutral kappa>0.3 (n)")
    for L in LS:
        for t in TAUS_L:
            tr = TR[(JL[0], JL[1], t, L)]
            k, G, d = tr[..., 0], tr[..., 1], tr[..., 3]
            hot, neu, big = k > 0.25, np.abs(G) < 0.1 * k, k > 0.3
            res = np.abs(k - (1 - d**2) / 2)[big & neu]
            print(f"                 {L / 1000:4.1f}    {t:.4f}   {plateau(tr).mean():.3f}         {hot.mean():.3f} ({np.mean(neu[hot]):.2f})"
                  f"          {np.median(res) if res.size else np.nan:.4f} ({res.size})   failed {int(np.nansum(tr[..., 4]))}")

    fig, axs = plt.subplots(len(LS), len(TAUS_L), figsize=(3.6 * len(TAUS_L) + 1.2, 3.1 * len(LS) + 0.6),
                            sharex=True, sharey=True, squeeze=False, layout='constrained')
    for r, L in enumerate(LS):
        for c, t in enumerate(TAUS_L):
            tr = TR[(JL[0], JL[1], t, L)]
            im = axs[r, c].pcolormesh(PHI / np.pi, EZ, tr[..., 0], shading='nearest', vmin=0, vmax=1, cmap='viridis',
                                      rasterized=True)
            P = plateau(tr)
            if P.any():
                axs[r, c].contour(PHI / np.pi, EZ, P.astype(float), levels=[0.5], colors='r', linewidths=0.6)
            axs[r, c].set_title(f'L {L / 1000:g} um, tc_barr {t:g}: plateau {P.mean():.1%}', fontsize=9)
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi/\pi$')
    fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$')
    fig.suptitle(rf'$\kappa$ vs junction length, {JL[0]}, $\mu_C$ = {JL[1]:g}, a = {A:g}, $t_c$ = 1; red: plateau', fontsize=10)
    st.save_fig(fig, 'tuning_kappa_length'); plt.close(fig)


# %% [tuning] --- 5. AROUND THE LENGTH OPTIMUM: tau from the predictor (interpolation only), checked with NEGF ---
EXTRA = ((1000.0, 0.55), (1000.0, 0.6), (1250.0, 0.5), (1250.0, 0.55))

if run_part('tuning'):
    for L, t in EXTRA:
        TR[(JL[0], JL[1], t, L)] = st.run_scan(partial(st.transport_phis, junction(*JL, t, L), phis=PHI), {'E_Z': EZ},
                                               f'scharf_absTune_tr_{JL[0]}_mu{JL[1]:g}_tau{t:g}_L{L / 1000:g}')
        print(f"L {L / 1000:g} um tau {t:g} done", flush=True)
    keys = sorted(k for k in TR if len(k) == 4)
    print(f"\n  {JL[0]} mu_C {JL[1]:g}, all lengths:  L [um]  tau     plateau frac  kappa>0.25 (neutral)")
    for (_, _, t, L) in sorted(keys, key=lambda k: (k[3], k[2])):
        tr = TR[(JL[0], JL[1], t, L)]
        hot, neu = tr[..., 0] > 0.25, np.abs(tr[..., 1]) < 0.1 * tr[..., 0]
        print(f"                              {L / 1000:4.2f}    {t:.4f}  {plateau(tr).mean():.3f}         "
              f"{hot.mean():.3f} ({np.mean(neu[hot]):.2f})")
    fig, ax = plt.subplots(figsize=(7, 4.2), layout='constrained')
    for L in sorted({k[3] for k in keys}):
        pts = sorted((k[2], plateau(TR[k]).mean()) for k in keys if k[3] == L)
        ax.plot(*zip(*pts), 'o-', label=f'L = {L / 1000:g} um')
    ax.set_xlabel('probe bond tc_barr'); ax.set_ylabel('plateau fraction (NEGF)')
    ax.set_title(f'{JL[0]}, mu_C {JL[1]:g}, a {A:g}, t_c 1: plateau area against probe bond and length', fontsize=9)
    ax.legend(fontsize=8)
    st.save_fig(fig, 'tuning_length_tau'); plt.close(fig)


# %% [tuning] --- 6. THE TWO BEST CASES ON THE FINE GRID (study_coupling resolution: E_Z 0.025, 61 phases) ---
EZ_FINE, PHI_FINE = np.round(np.arange(0.0, 2.0001, 0.025), 3), np.linspace(0.0, 2 * np.pi, 61)
FINE = ((1000.0, 0.55), (1500.0, 0.5))

if run_part('tuning'):
    TRF = {c: st.run_scan(partial(st.transport_phis, junction(*JL, c[1], c[0]), phis=PHI_FINE), {'E_Z': EZ_FINE},
                          f'scharf_absTune_fine_{JL[0]}_mu{JL[1]:g}_tau{c[1]:g}_L{c[0] / 1000:g}') for c in FINE}
    print(f"\n  FINE GRID {JL[0]} mu_C {JL[1]:g}:  L [um]  tau    plateau frac  kappa>0.25 (neutral)  2M test (n)   failed")
    for (L, t), tr in TRF.items():
        k, G, d = tr[..., 0], tr[..., 1], tr[..., 3]
        hot, neu, big = k > 0.25, np.abs(G) < 0.1 * k, k > 0.3
        res = np.abs(k - (1 - d**2) / 2)[big & neu]
        print(f"                                {L / 1000:4.2f}    {t:.2f}   {plateau(tr).mean():.3f}         "
              f"{hot.mean():.3f} ({np.mean(neu[hot]):.2f})          {np.median(res):.4f} ({res.size})   {int(np.nansum(tr[..., 4]))}")
    fig, axs = plt.subplots(2, len(FINE), figsize=(5.0 * len(FINE) + 1.2, 8.2), sharex=True, sharey=True,
                            squeeze=False, layout='constrained')
    for c, ((L, t), tr) in enumerate(TRF.items()):
        P = plateau(tr)
        im0 = axs[0, c].pcolormesh(PHI_FINE / np.pi, EZ_FINE, tr[..., 0], shading='nearest', vmin=0, vmax=1,
                                   cmap='viridis', rasterized=True)
        if P.any():
            axs[0, c].contour(PHI_FINE / np.pi, EZ_FINE, P.astype(float), levels=[0.5], colors='r', linewidths=0.6)
        axs[0, c].set_title(f'kappa: L {L / 1000:g} um, tc_barr {t:g}, plateau {P.mean():.1%}', fontsize=9)
        im1 = axs[1, c].pcolormesh(PHI_FINE / np.pi, EZ_FINE, np.abs(tr[..., 1]), shading='nearest', vmin=0, vmax=1,
                                   cmap='cividis', rasterized=True)
        axs[1, c].set_title(f'|G|: L {L / 1000:g} um, tc_barr {t:g}', fontsize=9)
        axs[1, c].set_xlabel(r'$\phi/\pi$')
    for r in range(2):
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    fig.colorbar(im0, ax=axs[0], label=r'$\kappa/\kappa_0$'); fig.colorbar(im1, ax=axs[1], label=r'$|G|/G_0$')
    fig.suptitle(rf'best cases on the fine grid: {JL[0]}, $\mu_C$ = {JL[1]:g}, a = {A:g}, $t_c$ = 1, W 100; red: plateau',
                 fontsize=10)
    st.save_fig(fig, 'tuning_best_fine'); plt.close(fig)


# %% [tuning] --- 7. MIXED SOC, THE THESIS CASE (2026-11-04): the length lever with the SOC held fixed ---
# Rashba + Dresselhaus, amplitude 16 meV nm, theta 0.15 pi ('100', the early Scharf set) is the material: only the
# junction length L, the probe bond tau and the gate mu_C are tuned.  The survey (cell 2) gave 2.3 / 4.7 % at
# mu_C 0.7, L 2 um (tau 0.3387 / 0.5) with half the Rashba-16 gap, so xi_M first: the lever ends where L ~ xi_M.
MU_M = (0.3, 0.7)
JM = ('mixed', 0.7)
LS_M = (1000.0, 1500.0, 2000.0, 3000.0)
TAUS_M = (0.3387, 0.5, 0.7)


def tr_name(soc, mu, t, L):
    """Cache name of a coarse map (L = 2 um keeps the survey's name)."""
    return f'scharf_absTune_tr_{soc}_mu{mu:g}_tau{t:g}' + ('' if L == 2000.0 else f'_L{L / 1000:g}')


if run_part('tuning'):
    XIM = {}
    for mu in MU_M:
        top = st.run_scan(partial(topo_point, junction('mixed', mu)), {'E_Z': EZ_Q}, f'scharf_absTune_topo_mixed_mu{mu:g}')
        on = np.flatnonzero(top[:, 0] < 0)
        XIM[mu] = (EZ_Q[on], top[on, 1], xi_scan(junction('mixed', mu), np.full(len(on), mu), EZ_Q[on],
                                                 f'scharf_absTune_xi_mixed_mu{mu:g}')[:, 1])
    print("\n  mixed SOC, phi = pi, infinite junction:  mu_C  E_Z   Delta_top [ueV]  xi_M [um]")
    for mu, (ez, g, xi) in XIM.items():
        for e, gg, x in zip(ez, g, xi):
            print(f"                                          {mu:3.1f}  {e:4.1f}  {gg * 1e3:6.1f}           {x / 1000:6.2f}")
    for L in LS_M:
        for t in TAUS_M:
            TR[(*JM, t, L)] = st.run_scan(partial(st.transport_phis, junction(*JM, t, L), phis=PHI), {'E_Z': EZ},
                                          tr_name(*JM, t, L))
            print(f"mixed L {L / 1000:g} um tau {t:g} done", flush=True)
    print(f"\n  mixed mu_C {JM[1]:g}:  L [um]  tau     plateau frac  kappa>0.25 (neutral)  2M test neutral kappa>0.3 (n)  failed")
    for L in LS_M:
        for t in TAUS_M:
            tr = TR[(*JM, t, L)]
            k, G, d = tr[..., 0], tr[..., 1], tr[..., 3]
            hot, neu, big = k > 0.25, np.abs(G) < 0.1 * k, k > 0.3
            res = np.abs(k - (1 - d**2) / 2)[big & neu]
            print(f"                     {L / 1000:4.1f}    {t:.4f}  {plateau(tr).mean():.3f}         {hot.mean():.3f}"
                  f" ({np.mean(neu[hot]) if hot.any() else np.nan:.2f})          "
                  f"{np.median(res) if res.size else np.nan:.4f} ({res.size})          {int(np.nansum(tr[..., 4]))}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for mu, (ez, g, xi) in XIM.items():
        axs[0].plot(ez, g * 1e3, 'o-', ms=3, label=rf'$\mu_C$ = {mu:g}')
        axs[1].plot(ez, xi / 1000, 'o-', ms=3, label=rf'$\mu_C$ = {mu:g}')
    for L in LS_M:
        axs[1].axhline(L / 3000, color='0.6', lw=0.6, ls=':')
    axs[0].set_ylabel(r'$\Delta_{top}$ [$\mu$eV]'); axs[1].set_ylabel(r'$\xi_M$ [$\mu$m]  (dotted: L/3 for the scanned L)')
    axs[1].set_yscale('log')
    for ax in axs:
        ax.set_xlabel(r'$E_Z$ [meV]'); ax.legend(fontsize=8)
    fig.suptitle(r'mixed SOC (16 meV nm, $\theta$ 0.15$\pi$), infinite junction at $\phi = \pi$, a 10, $t_c$ 1', fontsize=10)
    st.save_fig(fig, 'tuning_mixed_xi'); plt.close(fig)

    fig, axs = plt.subplots(len(LS_M), len(TAUS_M), figsize=(3.6 * len(TAUS_M) + 1.2, 3.0 * len(LS_M) + 0.6),
                            sharex=True, sharey=True, squeeze=False, layout='constrained')
    for r, L in enumerate(LS_M):
        for c, t in enumerate(TAUS_M):
            tr = TR[(*JM, t, L)]
            im = axs[r, c].pcolormesh(PHI / np.pi, EZ, tr[..., 0], shading='nearest', vmin=0, vmax=1, cmap='viridis',
                                      rasterized=True)
            P = plateau(tr)
            if P.any():
                axs[r, c].contour(PHI / np.pi, EZ, P.astype(float), levels=[0.5], colors='r', linewidths=0.6)
            axs[r, c].set_title(f'L {L / 1000:g} um, tc_barr {t:g}: plateau {P.mean():.1%}', fontsize=9)
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi/\pi$')
    fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$')
    fig.suptitle(rf'$\kappa$, mixed SOC, $\mu_C$ = {JM[1]:g}, a = {A:g}, $t_c$ = 1: length (rows) x probe bond; red: plateau',
                 fontsize=10)
    st.save_fig(fig, 'tuning_mixed_length'); plt.close(fig)


# %% [tuning] --- 8. kappa PEAKS AND THE SIGN OF det r: bulk transition or eps = Gamma? (2026-11-04) ---
# det r (left lead, E = 0, real) is -1 when the lead sees an isolated Majorana (Fulga et al.), +1 when it sees none
# or a hybridised pair.  At a sign change det r = 0: r has a zero singular value, one eigenchannel is fully
# transmitted.  (a) refine det r = 0 crossings along E_Z (regula falsi on the full model), read kappa and G there;
# (b) compare the det r = 0 lines with the bulk Q boundary (infinite cross-section: independent of L and tau).  A bulk
# transition stays on the Q boundary for every L and tau; the finite-size eps = Gamma crossing lies inside Q = -1
# and moves with tau (Gamma) and L (eps).
N_PER_L = 7                                                         # refined crossings per length (tau 0.5)


def sign_edges(S):
    """Midpoints (grid-index units) of the links between neighbouring grid points where S changes sign."""
    i, j = np.nonzero(S[1:] * S[:-1] < 0)
    k, l = np.nonzero(S[:, 1:] * S[:, :-1] < 0)
    return np.r_[np.c_[i + 0.5, j], np.c_[k, l + 0.5]]


def dist_to(P, B):
    """Distance (grid steps) from each point of P to the nearest point of B."""
    from scipy.spatial import cKDTree
    return cKDTree(B).query(P)[0] if len(B) and len(P) else np.full(len(P), np.inf)


def crossing_job(k, cases):
    """
    det r = 0 between E_Z lo and hi at phase phi, cases[k] = (L, tau, lo, hi, det(lo), det(hi), phi): Illinois
    regula falsi, 5 full-model solves.  Returns [E_Z, kappa, G, det r, final bracket width].
    """
    L, t, lo, hi, dlo, dhi, ph = cases[int(k)]
    q, side = junction(*JM, t, L), 0
    for _ in range(5):
        m = (lo * dhi - hi * dlo) / (dhi - dlo)
        v = st.transport_point(q, m, ph)
        if v[3] * dlo > 0:
            lo, dlo = m, v[3]
            dhi = dhi / 2 if side == -1 else dhi
            side = -1
        else:
            hi, dhi = m, v[3]
            dlo = dlo / 2 if side == 1 else dlo
            side = 1
    return np.array([m, v[0], v[1], v[3], hi - lo])


if run_part('tuning'):
    QM = st.q_map(junction(*JM), EZ, PHI, f'scharf_absTune_Q_{JM[0]}_mu{JM[1]:g}')
    BQ = sign_edges(QM)
    dEZ = EZ[1] - EZ[0]
    print(f"\n  det r SIGN CHANGES vs BULK Q, mixed mu_C {JM[1]:g} (grid steps E_Z {dEZ:g}, phi 2pi/{len(PHI) - 1}):")
    print("  L [um]  tau     det r<0   det-edges  on Q bdry (<=1 step)  Q-edges with a det-edge  kappa peaks>0.3 at a det-edge")
    for L in LS_M:
        for t in TAUS_M:
            tr = TR[(*JM, t, L)]
            k, d = tr[..., 0], tr[..., 3]
            D = sign_edges(np.sign(d))
            pk = np.argwhere(np.vstack([np.zeros((1, k.shape[1]), bool), (k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > 0.3),
                                        np.zeros((1, k.shape[1]), bool)]))
            print(f"  {L / 1000:4.1f}    {t:.4f}  {np.mean(d < 0):.2f}      {len(D):5d}      {np.mean(dist_to(D, BQ) <= 1):.2f}"
                  f"                  {np.mean(dist_to(BQ, D) <= 1):.2f}                     "
                  f"{np.mean(dist_to(pk, D) <= 1) if len(pk) else np.nan:.2f} ({len(pk)})")
        same = [sign_edges(np.sign(TR[(*JM, t, L)][..., 3])) for t in TAUS_M]
        print(f"          det-edges at tau {TAUS_M[0]} still on the same link at tau {TAUS_M[1]} / {TAUS_M[2]}: "
              f"{np.mean(dist_to(same[0], same[1]) < 0.1):.2f} / {np.mean(dist_to(same[0], same[2]) < 0.1):.2f}")

    # (a) refine det r = 0 along E_Z: N_PER_L crossings per length at tau 0.5, evenly through the list
    CASES = []
    for L in LS_M:
        d = TR[(*JM, 0.5, L)][..., 3]
        i, j = np.nonzero(d[1:] * d[:-1] < 0)
        for n in np.linspace(0, len(i) - 1, min(N_PER_L, len(i))).astype(int):
            CASES.append((L, 0.5, EZ[i[n]], EZ[i[n] + 1], d[i[n], j[n]], d[i[n] + 1, j[n]], PHI[j[n]]))
    CASES = tuple(CASES)
    REF = st.run_scan(partial(crossing_job, cases=CASES), {'k': np.arange(len(CASES))}, 'scharf_absTune_detr0_mixed')
    print("\n  det r = 0 refined (5 regula-falsi solves), tau 0.5:   L [um]  E_Z [meV]  phi/pi  kappa   G       |det r|  "
          "to Q bdry [steps]")
    for c, (ez, kk, G, dd, w) in zip(CASES, REF):
        dq = dist_to(np.array([[ez / dEZ, np.argmin(np.abs(PHI - c[6]))]]), BQ)[0]
        print(f"                                                      {c[0] / 1000:4.1f}    {ez:7.4f}    {c[6] / np.pi:5.3f}"
              f"   {kk:.4f}  {G:+.4f}  {abs(dd):.1e}  {dq:.1f}")

    fig, axs = plt.subplots(len(LS_M), len(TAUS_M), figsize=(3.6 * len(TAUS_M) + 1.2, 3.0 * len(LS_M) + 0.6),
                            sharex=True, sharey=True, squeeze=False, layout='constrained')
    for r, L in enumerate(LS_M):
        for c, t in enumerate(TAUS_M):
            tr = TR[(*JM, t, L)]
            im = axs[r, c].pcolormesh(PHI / np.pi, EZ, tr[..., 0], shading='nearest', vmin=0, vmax=1, cmap='viridis',
                                      rasterized=True)
            axs[r, c].contour(PHI / np.pi, EZ, tr[..., 3], levels=[0.0], colors='w', linewidths=0.7)
            axs[r, c].contour(PHI / np.pi, EZ, QM, levels=[0.0], colors='r', linewidths=1.0, linestyles='--')
            D = sign_edges(np.sign(tr[..., 3]))
            axs[r, c].set_title(f'L {L / 1000:g} um, tc_barr {t:g}: {np.mean(dist_to(D, BQ) <= 1):.0%} on Q bdry',
                                fontsize=9)
        axs[r, 0].set_ylabel(r'$E_Z$ [meV]')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi/\pi$')
    fig.colorbar(im, ax=axs, label=r'$\kappa/\kappa_0$')
    fig.suptitle(rf'mixed SOC, $\mu_C$ = {JM[1]:g}: $\kappa$ with det r = 0 (white) and the bulk Q boundary (red dashed)',
                 fontsize=10)
    st.save_fig(fig, 'detr_kappa_mixed'); plt.close(fig)

    jpi = int(np.argmin(np.abs(PHI - np.pi)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), layout='constrained', gridspec_kw={'width_ratios': [1, 1.6]})
    on = np.array([dist_to(np.array([[ez / dEZ, np.argmin(np.abs(PHI - c[6]))]]), BQ)[0] <= 1
                   for c, ez in zip(CASES, REF[:, 0])])
    for m, lab, col in ((~on, 'inside Q = -1 / outside', 'C0'), (on, 'on the Q boundary', 'C3')):
        axs[0].plot(np.abs(REF[m, 2]), REF[m, 1], 'o', ms=4, color=col, label=lab)
    axs[0].axhline(0.5, color='k', lw=0.6, ls=':')
    axs[0].set_xlabel(r'$|G|/G_0$ at det r = 0'); axs[0].set_ylabel(r'$\kappa/\kappa_0$ at det r = 0')
    axs[0].set_title('refined det r = 0 crossings, tc_barr 0.5', fontsize=9); axs[0].legend(fontsize=8)
    for L, col in zip(LS_M, ('C0', 'C1', 'C2', 'C3')):
        tr = TR[(*JM, 0.5, L)]
        axs[1].plot(EZ, tr[:, jpi, 0], '-', color=col, lw=1.2, label=rf'$\kappa$, L {L / 1000:g} um')
        axs[1].plot(EZ, tr[:, jpi, 3], '--', color=col, lw=0.8)
    axs[1].fill_between(EZ, -1, 1, where=QM[:, jpi] < 0, color='0.85', step='mid', label='bulk Q = -1')
    axs[1].axhline(0, color='k', lw=0.4)
    axs[1].set_xlabel(r'$E_Z$ [meV]'); axs[1].set_ylabel(r'$\kappa/\kappa_0$ (solid), det r (dashed)')
    axs[1].set_title(rf'$\phi = \pi$ column of the tc_barr 0.5 maps (grid step {dEZ:g} meV)', fontsize=9)
    axs[1].legend(fontsize=7, ncol=2)
    st.save_fig(fig, 'detr_crossings_mixed'); plt.close(fig)
