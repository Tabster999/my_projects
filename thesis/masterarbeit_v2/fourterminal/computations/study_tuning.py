"""
Topology, couplings and tuning at a = 10 nm: the topological regime, the S/N and probe couplings, and the mu_C / SOC
/ L / tau survey (incl. the mixed SOC maps).

PARTS (one per former script; the cells of each part are tagged [part]):
  topology     (was study_topology.py)  STUDY: can the topological regime be reached with the coupling-study parameters (study_coupling.py)?
  coupling     (was study_coupling.py)  STUDY: how the S/N interface coupling tc (tc_top = tc_bot; 1 = bulk bond, Scharf) and the probe coupling
  tuning       (was study_tuning.py)  STUDY: can mu_C and the spin-orbit coupling make the kappa = 1/2 regions larger?
  maps         (2026-11-09)  thesis figures: total DOS(E = 0) and kappa on (E_Z, phi) at a = 5 nm, mixed SOC, three probe bonds
  leads        (2026-11-11)  STUDY: the mixed-SOC junction vs BOTH lead couplings -- S/N bond tc (SC leads) and probe bond tau
  spectrum     (2026-11-13)  STUDY: DOS(E, phi) at one field through a bright E = 0 DOS spot, vs the probe bond tau (a = 10)
  ep           (2026-11-13)  STUDY: exceptional points of the open junction (poles with the probe leads), asymmetric tau_L, tau_R
  ep2          (2026-11-13)  the two-Majorana reduction of [ep] from the full model (G_L, G_R, c along phi), no fit
  ep3          (2026-11-16)  [ep] follow-ups: det r / kappa with asymmetric probes, a weak pair on a fine grid, E_Z 0.70
  ep4          (2026-11-16)  STUDY: can asymmetric probe bonds improve the kappa = 1/2 plateau? (along E_Z and phi)

Run one part:   python study_tuning.py topology      (several: list them; no argument: every part, in order)
In the interactive window the cells run as usual.  run_part(part) replaces the old `if __name__ == "__main__":`
guard: the parallel workers re-import this file and must not re-run the cells.
Names that collided between parts were renamed (see each PART header); cache names are unchanged.
"""
# %% --- IMPORTS AND THE PART SELECTOR ---
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import tools as st
from study_oscillation import slowest_mode

PARTS = ('topology', 'coupling', 'tuning', 'maps', 'leads', 'spectrum', 'ep', 'ep2', 'ep3', 'ep4')
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


# ====================================================================================================================
# PART 'maps'  (2026-11-09)
# ====================================================================================================================
# THESIS FIGURES: the total DOS at E = 0 and kappa on the (E_Z, phi) plane at a = 5 nm, for three probe couplings.
# Junction: the mixed SOC (16 meV nm, theta 0.15 pi; the thesis case), mu_C 0.7, S/N bond t_c 1, W 100, L 2 um.
# Probe bond tau = tc_barr_transport 0.2 / 0.5 / 1.0 (1 = a bulk bond: the probe is fully open).
#   DOS    tools.dos_total_phis with the probes ATTACHED (bond tau), summed over all sites of the centre (BdG, e and h),
#          eta = 0.1 ueV (2.5e-6 t, above the 1e-6 t solver rule), so the probes set the line widths and the three
#          DOS maps differ by the coupling.  Checked against a dense -Im Tr G / pi of the centre to 4e-12 (2026-11-09).
#   kappa  tools.transport_phis: E = 0, T = 0, eta_rel_transport 1e-5 Delta.
#   Q      infinite cross-section at a = 5 (no probes, one map for all six figures): dashed boundary of Q = -1.
# Six single-panel figures plots/maps_mixed_a5_{dos,kappa}_tau*.png, linear colour scales; caches scharf_absMap_*.

# %% [maps] --- PARAMETERS ---
A_MAP = 5.0
TAUS_MAP = (0.2, 0.5, 1.0)
EZ_MAP = np.round(np.arange(0.0, 2.0001, 0.025), 4)              # 81 fields
PHI_MAP = np.linspace(0.0, 2 * np.pi, 121)                         # phases are nearly free: one set-up per field
EZ_MAP_Q = EZ_MAP[::2]                                             # Q on half the grid (step 0.05, 41 fields):
PHI_MAP_Q = np.linspace(0.0, 2 * np.pi, 31)                        # only the dashed line; 44 s per point (see below)
ETA_MAP = 1e-4                                                     # meV, DOS broadening
WORKERS_MAP = 8                                                    # a = 5: 2.8 GB per worker
# Timing at a = 5, one field and 121 phases: 145 s transport / 91 s DOS with all BLAS threads, but ~11 / ~7 min on a
# one-thread worker (measured: the tau 0.2 kappa scan took 2 h on 8 workers) -> ~3.2 h per tau, ~10 h in all.  The
# laptop throttles under this load (95 C, ~1 GHz): up to ~24 min per kappa field.  Q: 8 s with all threads, 44 s on one
# thread under load -> the full (81 x 61) grid would take ~4.5 h on 14 workers, the half grid ~1.1 h.
# chunksize 1 (one row per task for Q) saves every finished field, so an interrupted run resumes where it stopped.


def junction_map(tau):
    """The thesis junction at a = 5 nm with probe bond tau."""
    alpha, beta = st.mixed_soc(16.0, 0.15 * np.pi)
    return st.replace(st.case(100.0, L=2000.0, alpha=alpha, beta=beta, a=A_MAP), tc=1.0, mu_C=0.7,
                      tc_barr_transport=tau)


# %% [maps] --- 1. SCANS (cached in plot_npz/scharf_absMap_*) ---
if run_part('maps'):
    TR_MAP, DOS_MAP = {}, {}
    for tau in TAUS_MAP:
        TR_MAP[tau] = st.run_scan(partial(st.transport_phis, junction_map(tau), phis=PHI_MAP), {'E_Z': EZ_MAP},
                                  f'scharf_absMap_tr_mixed_a5_tau{tau:g}', n_workers=WORKERS_MAP, chunksize=1)
        print(f"maps: kappa tau {tau:g} done", flush=True)
        DOS_MAP[tau] = st.run_scan(partial(st.dos_total_phis, junction_map(tau), phis=PHI_MAP, eta=ETA_MAP, tau=tau),
                                   {'E_Z': EZ_MAP}, f'scharf_absMap_dos_mixed_a5_tau{tau:g}', n_workers=WORKERS_MAP,
                                   chunksize=1)
        print(f"maps: DOS tau {tau:g} done", flush=True)
    Q_MAP = st.q_map(junction_map(TAUS_MAP[0]), EZ_MAP_Q, PHI_MAP_Q, 'scharf_absMap_Q_mixed_a5',
                     chunksize=len(PHI_MAP_Q))
    print("maps: Q done", flush=True)


# %% [maps] --- 2. NUMBERS ---
if run_part('maps'):
    print(f"\n  Q = -1 on {np.mean(Q_MAP < 0):.1%} of the (E_Z <= {EZ_MAP_Q[-1]:g}, phi) map")
    print("  tau   kappa max   kappa > 0.25   neutral of those   plateau (also Q = -1)   DOS min .. max [1/meV]   failed")
    Qi = Q_MAP[np.ix_(np.abs(EZ_MAP[:, None] - EZ_MAP_Q[None, :]).argmin(1),
                      np.abs(PHI_MAP[:, None] - PHI_MAP_Q[None, :]).argmin(1))]      # nearest Q point per map point
    for tau in TAUS_MAP:
        k, G, d = TR_MAP[tau][..., 0], TR_MAP[tau][..., 1], DOS_MAP[tau]
        hot, P = k > 0.25, plateau(TR_MAP[tau])
        print(f"  {tau:3g}   {np.nanmax(k):.3f}       {hot.mean():6.1%}         "
              f"{np.mean(np.abs(G[hot]) < 0.1 * k[hot]) if hot.any() else np.nan:5.2f}             "
              f"{P.mean():5.1%} ({np.mean(P & (Qi < 0)):5.1%})          {np.nanmin(d):.3g} .. {np.nanmax(d):.3g}"
              f"          {int(np.nansum(TR_MAP[tau][..., 4]))}")


# %% [maps] --- FIGURES: one PNG per quantity and probe bond ---
def map_figure(Z, cmap, label, title, name, vmax=None):
    """One (E_Z, phi) map, linear colour scale from 0, dashed Q = -1 boundary; saved as plots/<name>.png."""
    from matplotlib import patheffects as pe
    fig, ax = plt.subplots(figsize=(5.6, 4.5), layout='constrained')
    top = np.nanmax(Z) if vmax is None else vmax
    im = ax.pcolormesh(PHI_MAP / np.pi, EZ_MAP, Z, shading='nearest', cmap=cmap, vmin=0, vmax=top, rasterized=True)
    cs = ax.contour(PHI_MAP_Q / np.pi, EZ_MAP_Q, Q_MAP, levels=[0], colors='w', linewidths=1.2, linestyles='--')
    cs.set(path_effects=[pe.withStroke(linewidth=2.6, foreground='k')])
    ax.set_xticks(np.arange(0, 2.01, 0.5))
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_ylabel(r'$E_Z$ [meV]')
    ax.set_title(title + '\n' + r'mixed SOC, $\mu_C$ = 0.7 meV, W = 100 nm, L = 2 $\mu$m, a = 5 nm'
                 + '\n' + r'dashed: boundary of the bulk $Q = -1$ region', fontsize=9)
    fig.colorbar(im, ax=ax, label=label, extend='max' if np.nanmax(Z) > top else 'neither')
    st.save_fig(fig, name, dpi=200); plt.close(fig)


if run_part('maps'):
    # Linear scales.  kappa: 0..1 (same colour for 1/2 on every map), up to the 99.9th percentile where kappa > 1
    # (tau 1: 1.32; its max 1.90 is ONE pixel, the gap closing at E_Z = 0, phi = pi, with G = kappa).
    # DOS: 0..99th percentile -- the max is set by a few pixels on the gap-closing lines, and the Majorana peak in
    # the V scales as 2/(pi Gamma): ~1000 / ~170 / ~20 per meV at tau 0.2 / 0.5 / 1, so one common scale cannot work.
    for tau in TAUS_MAP:
        k = TR_MAP[tau][..., 0]
        map_figure(k, 'viridis', r'$\kappa/\kappa_0$', rf'Thermal conductance, probe bond $\tau$ = {tau:g}',
                   f'maps_mixed_a5_kappa_tau{tau:g}', vmax=max(1.0, np.nanpercentile(k, 99.9)))
        map_figure(DOS_MAP[tau], 'magma', r'total DOS($E = 0$) [1/meV]', rf'Total DOS at $E = 0$, probe bond $\tau$ = {tau:g}',
                   f'maps_mixed_a5_dos_tau{tau:g}', vmax=np.nanpercentile(DOS_MAP[tau], 99))


# %% [maps] --- 3. IS Q = -1 GAPPED? (2026-11-12) ---
# In the mixed SOC Q = -1 does not imply a gap: the middle BdG pair can tilt across zero at finite k (study_mixed [angle]
# 1); the [deck] (mu_C, E_Z) maps are 52-56 % gapless inside Q = -1.  The Q map above counts such points as topological.
# Delta_top (tools.gaps: k_grid(5), refined) on every Q = -1 point of Q_MAP; gapless = below 0.01 ueV as in [deck].
# ~28 s per point on one thread.  Figure plots/maps_mixed_a5_gap.png.
MAP_GAPLESS = 1e-5                                                  # meV


def map_gap_case(n, pts):
    """[gap at k = 0, Delta_top (meV), k a of it] at pts[n] = (E_Z, phi) for the a = 5 junction (no invariant)."""
    q = junction_map(TAUS_MAP[0])
    p = q.params(E_Z=float(pts[int(n)][0]), phi=float(pts[int(n)][1]))
    g0, g, k = st.gaps(p, st.n_S_of(q), st.k_grid(q.a))
    return np.array([g0 * q.t, g * q.t, k])


if run_part('maps'):
    iq = np.argwhere(Q_MAP < 0)
    PTS = tuple((float(EZ_MAP_Q[a]), float(PHI_MAP_Q[b])) for a, b in iq)
    GQ = st.run_scan(partial(map_gap_case, pts=PTS), {'n': np.arange(len(PTS))}, 'scharf_absMap_gap_mixed_a5',
                     n_workers=6, chunksize=2)
    GAP_MAP = np.full(Q_MAP.shape, np.nan)
    GAP_MAP[tuple(iq.T)] = GQ[:, 1]
    gpd, gls = (Q_MAP < 0) & (GAP_MAP >= MAP_GAPLESS), (Q_MAP < 0) & ~(GAP_MAP >= MAP_GAPLESS)
    mid = np.sum((GQ[:, 1] >= 1e-6) & (GQ[:, 1] < 8e-5))
    print(f"\n  a = 5 Q map (E_Z 0-2 step 0.05 x 31 phases, all columns as for the 45.6 %): Q = -1 {np.mean(Q_MAP < 0):.1%}, "
          f"gapped (>= 0.01 ueV) {gpd.mean():.1%}, gapless {gls.mean():.1%} (= {gls.sum() / max((Q_MAP < 0).sum(), 1):.0%} "
          f"of Q = -1); Delta_top max {np.nanmax(GAP_MAP) * 1e3:.1f} ueV; values between 1e-3 and 0.08 ueV: {mid}")
    print("  gapped Q = -1 per field (phases out of 30):  " + "  ".join(
        f"{e:.2f}:{int(gpd[i, :-1].sum())}/{int((Q_MAP[i, :-1] < 0).sum())}" for i, e in enumerate(EZ_MAP_Q)
        if (Q_MAP[i, :-1] < 0).any()))
    from matplotlib.colors import ListedColormap
    from matplotlib import patheffects as pe
    fig, ax = plt.subplots(figsize=(5.6, 4.5), layout='constrained')
    ax.pcolormesh(PHI_MAP_Q / np.pi, EZ_MAP_Q, np.where(Q_MAP > 0, 1.0, np.nan), shading='nearest',
                  cmap=ListedColormap(['#e4e3df']), vmin=0, vmax=1)
    ax.pcolormesh(PHI_MAP_Q / np.pi, EZ_MAP_Q, np.where(gls, 1.0, np.nan), shading='nearest',
                  cmap=ListedColormap(['#f5c9b3']), vmin=0, vmax=1)
    im = ax.pcolormesh(PHI_MAP_Q / np.pi, EZ_MAP_Q, np.where(gpd, GAP_MAP * 1e3, np.nan), shading='nearest', cmap='Blues',
                       vmin=0, rasterized=True)
    cs = ax.contour(PHI_MAP_Q / np.pi, EZ_MAP_Q, Q_MAP, levels=[0], colors='w', linewidths=1.2, linestyles='--')
    cs.set(path_effects=[pe.withStroke(linewidth=2.6, foreground='k')])
    ax.set_xticks(np.arange(0, 2.01, 0.5))
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_ylabel(r'$E_Z$ [meV]')
    ax.set_title(r'Topological gap $\Delta_{top}$ of the infinite junction' + '\n' + r'mixed SOC, $\mu_C$ = 0.7 meV, W = 100 nm, '
                 r'a = 5 nm' + '\n' + f'gapped {gpd.mean():.1%} of the map, $Q = -1$ {np.mean(Q_MAP < 0):.1%}', fontsize=9)
    fig.colorbar(im, ax=ax, label=r'$\Delta_{top}$ [$\mu$eV]  (grey: trivial; orange: $Q = -1$ but gapless)')
    st.save_fig(fig, 'maps_mixed_a5_gap', dpi=200)
    plt.close(fig)
    print("figure: plots/maps_mixed_a5_gap.png", flush=True)


# ====================================================================================================================
# PART 'leads'  (2026-11-11)
# ====================================================================================================================
# STUDY: how the MIXED-SOC junction (thesis case: W 100, L 2 um, mu_C 0.7, mu_S 1, Delta 0.25, a 10) depends on BOTH
# lead couplings.  The pure-Rashba version is [topology] + [coupling]; the mixed SOC had only tc = 1 and tau 0.3387 /
# 0.5 / 0.7 ([tuning] cell 7, study_mixed [oscillation]).
#   SC leads    = the S/N bond tc (tc_top = tc_bot; 1 = bulk bond = Scharf; 1.25 = STRONGER than a bulk bond, model only)
#   probe leads = the probe bond tau = tc_barr_transport (enters only Gamma: the bulk cannot see it)
#   1. SC leads, infinite junction, phi = pi: Q and Delta_top (refined) along E_Z; xi_M at six fields (slowest_mode,
#      ~4 min per point on one thread); Q on a coarse (E_Z, phi) map -> topological area.
#   2. SC leads, transport (tau 0.5): kappa, G, det r on the [tuning] coarse grid (E_Z 0.05 x 31 phases) per tc.
#   3. probe leads (tc 1): Gamma(tau) from the local-Andreev Lorentzian at the eps zeros E_Z 0.285 / 0.345 (phi pi);
#      coarse maps for tau 0.2 and 1 (0.3387 / 0.5 / 0.7 exist); phi = pi lines (step 0.01) for tau 0.2 / 0.7 / 1
#      (0.3387 / 0.5 exist, cache names of study_mixed [oscillation]).
#   4. numbers; figure plots/leads_mixed.png.
#   5. checks (2026-11-12): tc > 1 vs the control tc 0.8 (bound levels, normal-state transmission, phi = pi ABS,
#      bulk lines / Q maps for tc 0.8 and 1.1); the tau charge vs Delta_top < Gamma.  Figure plots/leads_checks_mixed.png.
# Every scan is cached per point (chunksize), so an interrupted run resumes.
TC_LEADS = (0.5, 0.75, 0.9, 1.0, 1.25)
TAU_LEADS = (0.2, 0.3387, 0.5, 0.7, 1.0)
EZ_LB = np.round(np.arange(0.0, 1.6001, 0.05), 3)               # bulk line, phi = pi
EZ_LXI = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)                        # xi_M fields
EZ_LQ = np.round(np.arange(0.0, 2.0001, 0.1), 3)                # Q map (21 x 16)
PHI_LQ = np.linspace(0.0, 2 * np.pi, 16)
EZ_LT = np.round(np.arange(0.10, 1.3501, 0.005), 3)[::2]       # = study_mixed EZ_T: phi = pi lines, step 0.01
GAMMA_FIELDS = (0.285, 0.345)                                  # eps zeros at tc 1, L 2 (study_mixed [oscillation])
CLOSED_L = 1e-5     # meV: refined Delta_top below 0.01 ueV = gapless (a band crosses E = 0 at finite k; checked tc 0.5)
# GAPLESS: slowest_mode drops |lam| = 1 (propagating) modes and returns the field-independent 2nd complex-k mode
# (257 nm, k 0.0404/nm); xi_M is then undefined (infinite), not 257 nm.  Masked via the bulk line.


def junction_leads(tc=1.0, tau=0.5):
    """The thesis junction (mixed SOC, mu_C 0.7, L 2 um, a 10) with S/N bond tc and probe bond tau."""
    return st.replace(junction('mixed', 0.7, tau, 2000.0), tc=tc)


def leads_xi_point(q, E_Z, phi=np.pi):
    """[Q, xi_M (nm), k_osc (1/nm)] of the infinite junction (slowest E = 0 complex-k mode)."""
    p, n_S = q.params(E_Z=float(E_Z), phi=float(phi)), st.n_S_of(q)
    lam = slowest_mode(p, n_S)
    return np.array([st.invariant(p, n_S), -q.a / np.log(np.abs(lam)), np.abs(np.angle(lam)) / q.a])


def leads_xi_case(k, cases):
    """leads_xi_point for cases[k] = (tc, E_Z): all tc in one scan (the ~4 min points fill every worker)."""
    tc, E_Z = cases[int(k)]
    return leads_xi_point(junction_leads(tc), E_Z)


def leads_gamma_point(k, cases):
    """[Gamma (ueV), local Andreev (E = 0)/2]: half width of the local-Andreev Lorentzian at an isolated Majorana,
    cases[k] = (tau, E_Z), phi = pi, tc 1 (bisection in ln E, as study_mixed.gamma_point)."""
    t, E_Z = cases[int(k)]
    q = junction_leads(1.0, t)

    def A(E):
        c = dict(zip(st.CHANNELS, st.channel_point(q, E_Z, E)))
        return float(np.ravel((np.asarray(c['eh_local']) + np.asarray(c['he_local'])) / 2)[0])
    A0, lo, hi = A(0.0), 1e-5, 1e-1
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if A(mid) > A0 / 2 else (lo, mid)
    return np.array([np.sqrt(lo * hi) * 1e3, A0])


def leads_summary(tr, Qi):
    """Map numbers: plateau fraction (all / inside Q = -1), kappa > 0.25 area, neutral share of it, max neutral kappa,
    two-Majorana residual median on neutral kappa > 0.3 inside Q = -1."""
    k, G, d = tr[..., 0], tr[..., 1], tr[..., 3]
    P, hot, neu = plateau(tr), k > 0.25, np.abs(G) < 0.1 * k
    big = (k > 0.3) & neu & (Qi < 0)
    return dict(plat=P.mean(), plat_q=(P & (Qi < 0)).mean(), hot=hot.mean(),
                neutral=np.mean(neu[hot]) if hot.any() else np.nan,
                kmax_neu=np.nanmax(np.where(neu & (Qi < 0), k, np.nan)),
                res2m=np.median(np.abs(k - (1 - d**2) / 2)[big]) if big.any() else np.nan,
                failed=int(np.nansum(tr[..., 4])))


def q_on_grid(Qm):
    """Coarse Q map (EZ_LQ x PHI_LQ) -> the transport grid (EZ x PHI), nearest point."""
    return Qm[np.abs(EZ[:, None] - EZ_LQ[None]).argmin(1)][:, np.abs(PHI[:, None] - PHI_LQ[None]).argmin(1)]


# %% [leads] --- 1. SC LEADS: infinite junction; PROBE LEADS: Gamma(tau) ---
if run_part('leads'):
    BULK_L, XI_L, QM_L = {}, {}, {}
    for tc in TC_LEADS:
        BULK_L[tc] = st.run_scan(partial(topo_point, junction_leads(tc)), {'E_Z': EZ_LB},
                                 f'scharf_absLeads_bulk_tc{tc:g}', chunksize=1)
        print(f"leads: bulk line tc {tc:g} done", flush=True)
    GAMMA_CASES = tuple((t, e) for t in TAU_LEADS for e in GAMMA_FIELDS)
    GAM_L = st.run_scan(partial(leads_gamma_point, cases=GAMMA_CASES), {'k': np.arange(len(GAMMA_CASES))},
                        'scharf_absLeads_gamma', chunksize=1).reshape(len(TAU_LEADS), len(GAMMA_FIELDS), 2)
    print("leads: Gamma(tau) done", flush=True)
    for tc in TC_LEADS:
        QM_L[tc] = st.q_map(junction_leads(tc), EZ_LQ, PHI_LQ, f'scharf_absLeads_Q_tc{tc:g}', chunksize=len(PHI_LQ))
        print(f"leads: Q map tc {tc:g} done", flush=True)
    XI_CASES = tuple((tc, e) for tc in TC_LEADS for e in EZ_LXI)
    XI_ALL = st.run_scan(partial(leads_xi_case, cases=XI_CASES), {'k': np.arange(len(XI_CASES))}, 'scharf_absLeads_xi',
                         chunksize=1).reshape(len(TC_LEADS), len(EZ_LXI), 3)
    XI_L = {tc: XI_ALL[i] for i, tc in enumerate(TC_LEADS)}
    # 'gapless' at an xi field: Q = -1 but refined Delta_top < CLOSED_L on the bulk line
    GAPLESS_L = {tc: np.array([BULK_L[tc][np.argmin(np.abs(EZ_LB - e)), 1] < CLOSED_L for e in EZ_LXI])
                 for tc in TC_LEADS}
    print("leads: xi_M done", flush=True)

# %% [leads] --- 2. SC LEADS: transport (tau 0.5); 3. PROBE LEADS: maps and phi = pi lines (tc 1) ---
if run_part('leads'):
    TR_L, TRT_L, LN_L = {}, {}, {}
    for tc in TC_LEADS:
        name = tr_name('mixed', 0.7, 0.5, 2000.0) if tc == 1.0 else f'scharf_absLeads_tr_tc{tc:g}_tau0.5'
        TR_L[tc] = st.run_scan(partial(st.transport_phis, junction_leads(tc, 0.5), phis=PHI), {'E_Z': EZ}, name,
                               chunksize=1)
        print(f"leads: transport map tc {tc:g} done", flush=True)
    for t in TAU_LEADS:
        TRT_L[t] = st.run_scan(partial(st.transport_phis, junction('mixed', 0.7, t, 2000.0), phis=PHI), {'E_Z': EZ},
                               tr_name('mixed', 0.7, t, 2000.0), chunksize=1)
        print(f"leads: transport map tau {t:g} done", flush=True)
    for t in TAU_LEADS:
        LN_L[t] = st.run_scan(partial(st.transport_point, junction('mixed', 0.7, t, 2000.0), phi=np.pi),
                              {'E_Z': EZ_LT}, f'scharf_absOM_line_mixed_tau{t:g}', chunksize=1)
        print(f"leads: phi = pi line tau {t:g} done", flush=True)

# %% [leads] --- 4. NUMBERS ---
if run_part('leads'):
    print("\n  SC LEADS (S/N bond tc), mixed SOC, a 10: phi = pi line, coarse Q map, tau 0.5 transport map")
    print("  tc     Q=-1 area  Q=-1 at pi for E_Z   max Delta_top [ueV] (E_Z)   xi_M [nm] at E_Z 0.2/0.4/0.6/0.8/1.0/1.2"
          "    max L/xi_M")
    SUM_TC = {}
    for tc in TC_LEADS:
        b, x, gl = BULK_L[tc], XI_L[tc], GAPLESS_L[tc]
        topo = b[:, 0] < 0
        gmax = np.where(topo, b[:, 1], -1.0)
        i = int(np.argmax(gmax))
        xi_txt = ' '.join('   (+)' if q_ > 0 else ('  gpls' if g_ else f"{v:6.0f}")
                          for q_, v, g_ in zip(x[:, 0], x[:, 1], gl))
        ok = (x[:, 0] < 0) & ~gl
        lmax = np.max(2000.0 / x[ok, 1]) if ok.any() else np.nan
        rng = f"{EZ_LB[topo].min():.2f}-{EZ_LB[topo].max():.2f}" if topo.any() else "   none  "
        print(f"  {tc:4.2f}   {np.mean(QM_L[tc] < 0):5.0%}      {rng}          {gmax[i] * 1e3:6.1f} ({EZ_LB[i]:.2f})"
              f"            {xi_txt}    {lmax:5.2f}")
        SUM_TC[tc] = leads_summary(TR_L[tc], q_on_grid(QM_L[tc]))
    print("\n  tc     plateau (all / Q=-1)   kappa>0.25   neutral of those   max neutral kappa in Q=-1   2M residual   failed")
    for tc in TC_LEADS:
        s = SUM_TC[tc]
        print(f"  {tc:4.2f}   {s['plat']:5.1%} / {s['plat_q']:5.1%}       {s['hot']:5.1%}        {s['neutral']:4.2f}"
              f"               {s['kmax_neu']:.3f}                  {s['res2m']:.1e}      {s['failed']}")
    gam = GAM_L[..., 0]
    p_fit = np.polyfit(np.log(TAU_LEADS), np.log(gam.mean(1)), 1)[0]
    print(f"\n  PROBE LEADS (tc 1): Gamma = local-Andreev half width at E_Z {GAMMA_FIELDS}, phi pi;"
          f" fit Gamma ~ tau^{p_fit:.2f}")
    loc_p = np.diff(np.log(gam.mean(1))) / np.diff(np.log(TAU_LEADS))
    print("  local exponent d ln Gamma / d ln tau between neighbouring tau: "
          + ' '.join(f"{v:.2f}" for v in loc_p) + "  (not one power law)")
    print("  tau     Gamma [ueV] at the 2 fields  [mK]   A(0)/2   plateau (all / Q=-1)  kappa>0.25  neutral  | phi = pi "
          "line: kappa = 1/2 windows, widest [ueV of E_Z], neutral share of kappa > 0.25, failed")
    SUM_TAU = {}
    for j, t in enumerate(TAU_LEADS):
        s = SUM_TAU[t] = leads_summary(TRT_L[t], q_on_grid(QM_L[1.0]))
        ln = LN_L[t]
        okl = (np.abs(ln[:, 0] - 0.5) < 0.03) & (np.abs(ln[:, 1]) < 0.1 * ln[:, 0])
        runs, i0 = [], None
        for i, v in enumerate(np.r_[okl, False]):
            if v and i0 is None:
                i0 = i
            if not v and i0 is not None:
                runs.append(EZ_LT[i - 1] - EZ_LT[i0] + 0.01)
                i0 = None
        hot = ln[:, 0] > 0.25
        neu_l = np.mean(np.abs(ln[hot, 1]) < 0.1 * ln[hot, 0]) if hot.any() else np.nan
        print(f"  {t:6.4f}  {gam[j, 0]:6.2f} {gam[j, 1]:6.2f}  {gam[j].mean() * 11.6:6.0f}   {GAM_L[j, :, 1].mean():.3f}"
              f"   {s['plat']:5.1%} / {s['plat_q']:5.1%}      {s['hot']:5.1%}     {s['neutral']:4.2f}   |  {len(runs):2d},"
              f" {max(runs) * 1e3 if runs else 0:4.0f}, {neu_l:.2f}, {int(np.nansum(ln[:, 4]))}")

# %% [leads] --- FIGURE ---
if run_part('leads'):
    INKL, INK2L, GRIDL, C_CH = '#0b0b0b', '#52514e', '#d8d7d2', '#eb6834'
    cmap_l = plt.get_cmap('Blues')
    col_tc = {tc: cmap_l(0.4 + 0.6 * i / (len(TC_LEADS) - 1)) for i, tc in enumerate(TC_LEADS)}
    col_tau = {t: cmap_l(0.4 + 0.6 * i / (len(TAU_LEADS) - 1)) for i, t in enumerate(TAU_LEADS)}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.2), layout='constrained')
    for a in axs.ravel():
        a.grid(color=GRIDL, lw=0.5, alpha=0.6)
        a.tick_params(colors=INK2L, labelcolor=INKL)
    a = axs[0, 0]
    for tc in TC_LEADS:
        b = BULK_L[tc]
        a.plot(EZ_LB, np.where(b[:, 0] < 0, b[:, 1], np.nan) * 1e3, '-o', ms=3, lw=1.8, color=col_tc[tc],
               label=f'tc {tc:g}' + (' (> bulk bond)' if tc > 1 else ''))
    a.text(1.62, 20.5, 'tc 0.5: gapless at every field\n(a band crosses E = 0 at finite k)', fontsize=8,
           color=INK2L, va='bottom', ha='right')
    a.set_xlabel(r'$E_Z$ (meV)', color=INKL)
    a.set_ylabel(r'$\Delta_{top}$ ($\mu$eV), inside Q = $-1$', color=INKL)
    a.set_title(r'SC leads: protecting gap ($\phi = \pi$, infinite junction)', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a = axs[0, 1]
    for tc in TC_LEADS:
        x = XI_L[tc]
        ok = (x[:, 0] < 0) & ~GAPLESS_L[tc]
        if ok.any():
            a.plot(np.array(EZ_LXI), np.where(ok, x[:, 1], np.nan) / 1000, '-o', ms=4, lw=1.8, color=col_tc[tc],
                   label=f'tc {tc:g}')
    a.axhline(2.0 / 3, color=INK2L, lw=0.8, ls='--')
    a.text(0.2, 2.0 / 3 * 1.08, r'L/3 (L = 2 $\mu$m)', fontsize=8, color=INK2L)
    a.text(0.2, 0.2, r'tc 0.5 and tc 0.75 at $E_Z$ 1.2: gapless, $\xi_M$ undefined' + '\n'
           r'tc 1.25 below $E_Z$ 1.0: Q = +1', fontsize=8, color=INK2L, va='bottom')
    a.set_yscale('log')
    a.set_ylim(0.18, 7)
    a.set_yticks([0.2, 0.5, 1, 2, 5])
    a.set_yticklabels(['0.2', '0.5', '1', '2', '5'])
    a.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel(r'$E_Z$ (meV)', color=INKL)
    a.set_ylabel(r'$\xi_M$ ($\mu$m)', color=INKL)
    a.set_title(r'SC leads: Majorana decay length ($\phi = \pi$)', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='upper left', ncols=2)
    a = axs[0, 2]
    tcs = np.array(TC_LEADS)
    a.plot(tcs, [SUM_TC[tc]['plat_q'] * 100 for tc in TC_LEADS], '-o', color=col_tc[1.0], lw=2, ms=6, mec='w')
    for tc in TC_LEADS:
        a.annotate(f"Q=$-1$ area {np.mean(QM_L[tc] < 0):.0%}\nneutral {SUM_TC[tc]['neutral']:.2f}",
                   (tc, SUM_TC[tc]['plat_q'] * 100), textcoords='offset points',
                   xytext={0.5: (-20, 8), 0.75: (-20, 8), 0.9: (6, 6), 1.0: (-62, -24), 1.25: (-24, -28)}[tc], fontsize=7,
                   color=INKL)
    a.set_xlabel('S/N bond tc (1 = bulk bond)', color=INKL)
    a.set_ylabel(r'$\kappa = 1/2$ plateau inside Q = $-1$ (% of map)', color=INKL)
    a.set_ylim(0, 6.5)
    a.set_xlim(0.4, 1.35)
    a.set_title(r'SC leads: transport ($\tau$ 0.5, L 2 $\mu$m); neutral = share of $\kappa > 0.25$', color=INKL,
                fontsize=10)
    a = axs[1, 0]
    taus = np.array(TAU_LEADS)
    gm = gam.mean(1)
    a.plot(taus, gm, '-', color=col_tau[0.5], lw=1.4)
    for f, mk in zip(range(len(GAMMA_FIELDS)), ('o', 's')):
        a.plot(taus, gam[:, f], mk, color=col_tau[0.5], ms=6, mec='w', label=rf'$E_Z$ = {GAMMA_FIELDS[f]} meV')
    tt = np.linspace(taus.min(), taus.max(), 50)
    a.plot(tt, gm[0] * (tt / taus[0])**2, color=INK2L, lw=1, ls='--', label=r'$\tau^2$ (golden rule) from $\tau$ 0.2')
    for j, v in enumerate(loc_p):
        tm = np.sqrt(taus[j] * taus[j + 1])
        a.annotate(f"{v:.1f}", (tm, np.sqrt(gm[j] * gm[j + 1])), textcoords='offset points', xytext=(-14, 4),
                   fontsize=8, color=INKL)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xticks(taus)
    a.set_xticklabels([f'{t:g}' for t in taus])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel(r'probe bond $\tau$', color=INKL)
    a.set_ylabel(r'$\Gamma$ ($\mu$eV)', color=INKL)
    a.set_title(r'probe leads: $\Gamma$ (local Andreev width); numbers = local slope d ln$\Gamma$/d ln$\tau$',
                color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a = axs[1, 1]
    for j, t in enumerate((0.2, 0.5, 1.0)):
        ln = LN_L[t]
        ch = (np.abs(ln[:, 1]) >= 0.1 * ln[:, 0]) & (ln[:, 0] > 0.05)
        a.plot(EZ_LT, ln[:, 0] + 1.2 * j, color=col_tau[t], lw=1.6)
        a.plot(EZ_LT[ch], ln[ch, 0] + 1.2 * j, 'o', ms=2.5, color=C_CH)
        a.axhline(0.5 + 1.2 * j, color=INK2L, lw=0.6, ls=':')
        a.text(1.37, 1.2 * j + 0.05, rf'$\tau$ {t:g}', fontsize=8, color=INKL)
    a.set_xlabel(r'$E_Z$ (meV)', color=INKL)
    a.set_ylabel(r'$\kappa$ (traces offset by 1.2; dotted: $\kappa = 1/2$)', color=INKL)
    a.set_title(r'probe leads: $\kappa$ at $\phi = \pi$, tc 1 (orange: charged, $|G| \geq 0.1\kappa$)', color=INKL,
                fontsize=10)
    a = axs[1, 2]
    a.plot(taus, [SUM_TAU[t]['plat_q'] * 100 for t in TAU_LEADS], '-o', color=col_tau[0.5], lw=2, ms=6, mec='w')
    for j, t in enumerate(TAU_LEADS):
        a.annotate(rf"$\Gamma$ {gm[j]:.2g} $\mu$eV" + f"\nneutral {SUM_TAU[t]['neutral']:.2f}",
                   (t, SUM_TAU[t]['plat_q'] * 100), textcoords='offset points',
                   xytext=(8, -20) if j in (0, 1) else (5, 6), fontsize=7, color=INKL)
    a.set_xlabel(r'probe bond $\tau$', color=INKL)
    a.set_ylabel(r'$\kappa = 1/2$ plateau inside Q = $-1$ (% of map)', color=INKL)
    a.set_ylim(0, 3.2)
    a.set_xlim(0.15, 1.12)
    a.set_title(r'probe leads: transport (tc 1, L 2 $\mu$m); neutral = share of $\kappa > 0.25$', color=INKL,
                fontsize=10)
    fig.suptitle(r'Mixed SOC junction vs the lead couplings: W 100 nm, L 2 $\mu$m, $\mu_C$ 0.7 meV, a 10 nm, $E$ = 0',
                 color=INKL, fontsize=12)
    st.save_fig(fig, 'leads_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/leads_mixed.png", flush=True)


# %% [leads] --- 5. CHECKS (2026-11-12): why tc > 1 loses protection; does the tau 1 charge start at Delta_top < Gamma? ---
# (a) tc > 1.  Control tc 0.8: ONE bond x and 1/x transmit equally in the normal state (T(x) = T(1/x) at every k; checked
#     below, column 'one bond').  So 0.8 vs 1.25 separates "the bond is a barrier" from what only a STRONG bond does.
#     Cross-section at k 0 (H_cross): junction-bound normal-state levels, the phi = pi ABS at E_Z 0; normal-state
#     transmission across the whole junction (S | tc | N strip | tc | S, electron blocks, semi-infinite S); bulk lines
#     and Q maps for tc 0.8 / 1.1 (cached as the [leads] ones).
# (b) tau.  From the cached phi = pi lines: median G/kappa in E_Z windows (kappa > 0.05) for every tau, against where
#     Delta_top(E_Z) drops below Gamma(tau); at tau 1 T_ee and T_he against exp(-2L/xi_M).
TC_CHECK = (0.8, 1.0, 1.1, 1.25)
TC_DENSE = np.round(np.geomspace(0.5, 1.6, 41), 4)


def leads_band_bottom():
    """Bottom of the bulk S band at k 0 along the junction (normal state, E_Z 0), meV: min over k_y of the S chain."""
    q = junction_leads(1.0)
    p = st.replace(q, Delta=0.0).params(E_Z=0.0, phi=0.0)
    e = slice(0, 2)
    Vx = st.model_hop(p, 'x', p.t_s, p.m0, alpha=st.region_alpha(p, 's'))
    h = (st.model_onsite(p, 's') + Vx + Vx.conj().T)[e, e]
    Vy = st.model_hop(p, 'y', p.t_s, p.m0, alpha=st.region_alpha(p, 's'))[e, e]
    return min(np.linalg.eigvalsh(h + Vy * np.exp(1j * k) + Vy.conj().T * np.exp(-1j * k))[0]
               for k in np.linspace(-0.4, 0.4, 4001)) * q.t


def leads_cross_section(tc, bottom):
    """Normal state (Delta 0, E_Z 0, k 0): the junction-bound occupied levels = those below the bulk S band bottom
    `bottom` (meV; one per Kramers pair), their weight within 100 nm of the strip and their site weights; BdG at
    E_Z 0, phi pi, k 0: the lowest level (meV)."""
    q = junction_leads(tc)
    n_S = st.n_S_of(q)
    p = st.replace(q, Delta=0.0).params(E_Z=0.0, phi=np.pi)
    E, V = np.linalg.eigh(st.H_cross(p, 0.0, n_S))
    E = E * q.t
    w = np.abs(V.reshape(-1, 4, V.shape[1])) ** 2
    ele, ws = w[:, :2].sum((0, 1)), w.sum(1)
    b = np.flatnonzero((ele > 0.99) & (E < bottom - 1e-4))
    prof = (ws[:, b[0::2]] + ws[:, b[1::2]]) / 2
    near = prof[n_S - 10:n_S + p.ny + 10].sum(0)
    abs0 = np.abs(np.linalg.eigvalsh(st.H_cross(q.params(E_Z=0.0, phi=np.pi), 0.0, n_S))).min() * q.t
    return E[b[0::2]], near, prof, abs0


def leads_TN(tc, single=False):
    """Normal-state transmission per spin at E_F, normal incidence (k 0): across the junction S | tc | N strip | tc | S,
    or across one bond S | tc | S (single).  Electron blocks of model_onsite / model_hop, semi-infinite S leads."""
    from my_functions.leads import sancho_rubio
    q = junction_leads(tc)
    p = st.replace(q, Delta=0.0).params(E_Z=0.0, phi=0.0)
    e = slice(0, 2)

    def ons(r):
        Vx = st.model_hop(p, 'x', p.t_s if r == 's' else p.t_c, p.m0 if r == 's' else p.m0_c, alpha=st.region_alpha(p, r))
        return (st.model_onsite(p, r) + Vx + Vx.conj().T)[e, e]
    VyS = st.model_hop(p, 'y', p.t_s, p.m0, alpha=st.region_alpha(p, 's'))[e, e]
    VyC = st.model_hop(p, 'y', p.t_c, p.m0_c, alpha=st.region_alpha(p, 'c'))[e, e]
    Vb = st.model_hop(p, 'y', tc, st.bond_m0(p.m0, p.m0_c), bond=True)[e, e]
    hS = ons('s')
    sites, hops, c = ([hS, hS], [Vb], VyS) if single else ([ons('c')] * p.ny, [VyC] * (p.ny - 1), Vb)
    n = len(sites)
    H = np.zeros((2 * n, 2 * n), complex)
    for j, h in enumerate(sites):
        H[2*j:2*j+2, 2*j:2*j+2] = h
    for j, v in enumerate(hops):
        H[2*j:2*j+2, 2*j+2:2*j+4], H[2*j+2:2*j+4, 2*j:2*j+2] = v, v.conj().T
    z = 1e-9j * np.eye(2)
    eps_p, eps_m = sancho_rubio(hS, VyS, z[None])
    SL = c.conj().T @ np.linalg.inv(z - eps_m[0]) @ c          # lead to the left, coupled to centre site 0
    SR = c @ np.linalg.inv(z - eps_p[0]) @ c.conj().T          # lead to the right, coupled to the last site
    Sig = np.zeros_like(H)
    Sig[:2, :2], Sig[-2:, -2:] = SL, SR
    G = np.linalg.inv(1e-9j * np.eye(2 * n) - H - Sig)[:2, -2:]
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    return float(np.trace(GL @ G @ GR @ G.conj().T).real / 2)


if run_part('leads'):
    BULK_C, QM_C = {}, {}
    for tc in TC_CHECK:
        BULK_C[tc] = BULK_L.get(tc)
        if BULK_C[tc] is None:
            BULK_C[tc] = st.run_scan(partial(topo_point, junction_leads(tc)), {'E_Z': EZ_LB},
                                     f'scharf_absLeads_bulk_tc{tc:g}', chunksize=1)
        QM_C[tc] = QM_L.get(tc)
        if QM_C[tc] is None:
            QM_C[tc] = st.q_map(junction_leads(tc), EZ_LQ, PHI_LQ, f'scharf_absLeads_Q_tc{tc:g}', chunksize=len(PHI_LQ))
    BOT_C = leads_band_bottom()
    XS_C = {tc: leads_cross_section(tc, BOT_C) for tc in TC_CHECK}
    TN_D = np.array([[leads_TN(tc), leads_TN(tc, single=True)] for tc in TC_DENSE])
    XS_D = [leads_cross_section(tc, BOT_C) for tc in TC_DENSE]
    ABS_D, NB_D = np.array([x[3] for x in XS_D]), np.array([len(x[0]) for x in XS_D])
    print(f"\n  CHECK tc > 1 (normal state at E_F, k 0; bulk S band bottom {BOT_C:+.3f} meV; ABS at E_Z 0, phi pi, k 0;"
          " bulk line phi pi)")
    print("  tc     T one bond   T junction   bound levels [meV] (weight within 100 nm)   ABS(pi) [ueV]"
          "   Q=-1 at pi from   max Delta_top [ueV] (E_Z)   Q=-1 area")
    for tc in TC_CHECK:
        lv, near, _, a0 = XS_C[tc]
        b = BULK_C[tc]
        topo = b[:, 0] < 0
        g = np.where(topo, b[:, 1], -1.0)
        i = int(np.argmax(g))
        lv_txt = ' '.join(f'{v:+.3f} ({w:.2f})' for v, w in zip(lv, near)) if len(lv) else 'none'
        print(f"  {tc:4.2f}   {leads_TN(tc, True):.3f}        {leads_TN(tc):.3f}        {lv_txt:28s}"
              f"   {a0 * 1e3:6.1f}         {EZ_LB[topo].min():.2f}             {g[i] * 1e3:5.1f} ({EZ_LB[i]:.2f})"
              f"              {np.mean(QM_C[tc] < 0):4.0%}")
    print(f"  bound levels on the dense tc grid: first at tc {TC_DENSE[np.argmax(NB_D > 0)]:g} (none at "
          f"{TC_DENSE[np.argmax(NB_D > 0) - 1]:g}); any for tc <= 1: {bool(NB_D[TC_DENSE <= 1].any())}")
    # (b) tau
    WIN_C = np.round(np.arange(0.1, 1.31, 0.1), 2)
    dtop_C = lambda e: np.interp(e, EZ_LB, np.where(BULK_L[1.0][:, 0] < 0, BULK_L[1.0][:, 1], 0.0))
    xi_C = lambda e: np.exp(np.interp(e, EZ_LXI, np.log(XI_L[1.0][:, 1])))
    MED_C = np.full((len(TAU_LEADS), len(WIN_C) - 1), np.nan)
    for j, t in enumerate(TAU_LEADS):
        ln = LN_L[t]
        for w in range(len(WIN_C) - 1):
            m = (EZ_LT >= WIN_C[w]) & (EZ_LT < WIN_C[w + 1]) & (ln[:, 0] > 0.05)
            if m.any():
                MED_C[j, w] = np.median(ln[m, 1] / ln[m, 0])
    ez_f = np.linspace(0.5, 1.35, 1701)
    CROSS_C = {t: (ez_f[dtop_C(ez_f) < gm[j] * 1e-3].min() if (dtop_C(ez_f) < gm[j] * 1e-3).any() else np.nan)
               for j, t in enumerate(TAU_LEADS)}
    print("\n  CHECK tau: median G/kappa (kappa > 0.05) in E_Z windows; last column L/xi_M (tc 1)")
    print("  E_Z      " + "  ".join(f"tau {t:<6g}" for t in TAU_LEADS) + "   Delta_top [ueV]   L/xi_M")
    for w in range(len(WIN_C) - 1):
        e = WIN_C[w] + 0.05
        print(f"  {WIN_C[w]:.1f}-{WIN_C[w + 1]:.1f}  " + "  ".join(f"{v:9.2f} " if np.isfinite(v) else "        - "
                                                         for v in MED_C[:, w])
              + f"     {dtop_C(e) * 1e3:5.1f}           {2000.0 / xi_C(e):4.1f}")
    print("  Delta_top drops below Gamma(tau) at E_Z (falling side): "
          + ", ".join(f"tau {t:g}: {CROSS_C[t]:.2f}" for t in TAU_LEADS)
          + f"  (tau 1: Gamma {gm[-1]:.1f} ueV >= Delta_top everywhere except E_Z 0.50-0.51)")
    ln1 = LN_L[1.0]
    Tee1, The1 = (ln1[:, 0] + ln1[:, 1]) / 2, (ln1[:, 0] - ln1[:, 1]) / 2
    r1 = The1 / Tee1
    s1, s2 = (EZ_LT >= 0.2) & (EZ_LT <= 1.2), EZ_LT > 1.2
    print(f"  tau 1: T_he / T_ee = {r1[s1].min():.3f} .. {r1[s1].max():.3f} for E_Z 0.2-1.2, {r1[s2].min():.2f} .. "
          f"{r1[s2].max():.2f} for 1.2-1.35 (max at {EZ_LT[s2][np.argmax(r1[s2])]:.2f}, next to the gap closing at 1.4);"
          f" T_ee / exp(-2L/xi_M) at the xi fields: "
          + " ".join(f"{np.interp(e, EZ_LT, Tee1) / np.exp(-4000.0 / x):.1f}" for e, x in zip(EZ_LXI, XI_L[1.0][:, 1])))

    # --- figure ---
    col_c = {tc: cmap_l(0.4 + 0.6 * i / (len(TC_CHECK) - 1)) for i, tc in enumerate(TC_CHECK)}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.2), layout='constrained')
    for a in axs.ravel():
        a.grid(color=GRIDL, lw=0.5, alpha=0.6)
        a.tick_params(colors=INK2L, labelcolor=INKL)
    a = axs[0, 0]
    for tc in TC_CHECK:
        b = BULK_C[tc]
        a.plot(EZ_LB, np.where(b[:, 0] < 0, b[:, 1], np.nan) * 1e3, marker='o', ms=3, lw=1.8, color=col_c[tc],
               ls='--' if tc == 0.8 else '-', label=f'tc {tc:g}' + (' (control: one bond transmits as 1.25)'
                                                                     if tc == 0.8 else ''))
    a.set_xlabel(r'$E_Z$ (meV)', color=INKL)
    a.set_ylabel(r'$\Delta_{top}$ ($\mu$eV), inside Q = $-1$', color=INKL)
    a.set_title(r'protecting gap ($\phi = \pi$, infinite junction)', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a = axs[0, 1]
    a.plot(TC_DENSE, TN_D[:, 1], '--', color=INK2L, lw=1.4, label='one bond (S | tc | S)')
    a.plot(TC_DENSE, TN_D[:, 0], '-', color=col_c[1.0], lw=2, label='whole junction (S | tc | N 100 nm | tc | S)')
    a.axvspan(1.0, TC_DENSE.max(), color=GRIDL, alpha=0.35, lw=0)
    a.axvline(TC_DENSE[np.argmax(NB_D > 0)], color=INK2L, lw=0.8, ls=':')
    a.text(1.24, 0.98, f'tc >= {TC_DENSE[np.argmax(NB_D > 0)]:.2f} (dotted):\nthe junction binds\nlevels below the\n'
           'S band bottom', fontsize=8, color=INK2L, va='top')
    a.set_xscale('log')
    a.set_xticks([0.5, 0.8, 1.0, 1.25, 1.6])
    a.set_xticklabels(['0.5', '0.8', '1', '1.25', '1.6'])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_ylim(0, 1.05)
    a.set_xlabel('S/N bond tc (log axis: x and 1/x mirror)', color=INKL)
    a.set_ylabel('normal-state transmission per spin, E$_F$, k = 0', color=INKL)
    a.set_title('normal state: one bond is symmetric in tc, the junction is not', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='lower left')
    a = axs[0, 2]
    a.plot(TC_DENSE, ABS_D * 1e3, '-', color=col_c[1.0], lw=2)
    for tc in TC_CHECK:
        b = BULK_C[tc]
        a0 = XS_C[tc][3] * 1e3
        a.plot(tc, a0, 'o', color=col_c[tc], ms=7, mec='w')
        a.annotate(f"Q = $-1$ at $\\pi$ from $E_Z$ {EZ_LB[b[:, 0] < 0].min():.2f}", (tc, a0), textcoords='offset points',
                   xytext={0.8: (-6, 10), 1.0: (6, -4), 1.1: (-8, 0), 1.25: (-8, 0)}[tc],
                   ha='left' if tc == 1.0 else 'right', va='center', fontsize=8, color=INKL)
    a.set_xscale('log')
    a.set_xticks([0.5, 0.8, 1.0, 1.25, 1.6])
    a.set_xticklabels(['0.5', '0.8', '1', '1.25', '1.6'])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel('S/N bond tc', color=INKL)
    a.set_ylabel(r'lowest ABS at $E_Z$ 0, $\phi = \pi$, k = 0 ($\mu$eV)', color=INKL)
    a.set_title(r'the $\phi = \pi$ ABS gap that $E_Z$ must close', color=INKL, fontsize=10)
    a = axs[1, 0]
    wc = (WIN_C[:-1] + WIN_C[1:]) / 2
    for j, t in enumerate(TAU_LEADS):
        a.plot(wc, MED_C[j], '-o', ms=4, lw=1.8, color=col_tau[t], label=rf'$\tau$ {t:g} ($\Gamma$ {gm[j]:.2g} $\mu$eV)')
        if np.isfinite(CROSS_C[t]) and t < 1.0:
            a.plot(CROSS_C[t], 1.02, 'v', color=col_tau[t], ms=8, mec='w', clip_on=False)
    a.text(0.12, 0.93, r'$\blacktriangledown$: $\Delta_{top}$ drops below $\Gamma(\tau)$' + '\n'
           r'($\tau$ 1: $\Gamma \geq \Delta_{top}$ at every field; $\tau$ 0.2: never)', fontsize=8, color=INK2L, va='top')
    a.set_ylim(0, 1.05)
    a.set_xlabel(r'$E_Z$ (meV), windows of 0.1', color=INKL)
    a.set_ylabel(r'median $G/\kappa$ ($\kappa > 0.05$)', color=INKL)
    a.set_title(r'charge share at $\phi = \pi$, tc 1: no step where $\Delta_{top} < \Gamma$', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='center left')
    a = axs[1, 1]
    a.semilogy(EZ_LT, Tee1, color=C_CH, lw=1.8, label=r'$T_{ee}$ (electron cotunnelling)')
    a.semilogy(EZ_LT, np.maximum(The1, 1e-12), color=col_tau[1.0], lw=1.8, label=r'$T_{he}$ (crossed Andreev)')
    a.semilogy(EZ_LXI, np.exp(-4000.0 / XI_L[1.0][:, 1]), 'o--', color=INK2L, lw=1, ms=4,
               label=r'$e^{-2L/\xi_M}$ (tc 1)')
    a.set_ylim(1e-8, 1.5)
    a.set_xlabel(r'$E_Z$ (meV)', color=INKL)
    a.set_ylabel('transmission (probe to probe)', color=INKL)
    a.set_title(r'$\tau$ 1: charged at every field, both grow with the end overlap', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='lower right')
    a = axs[1, 2]
    lv, near, ws, _ = XS_C[1.25]
    n_S = st.n_S_of(junction_leads(1.25))
    y = (np.arange(ws.shape[0]) - n_S) * A
    for m, (v, ls) in enumerate(zip(lv, ('-', '--'))):
        a.plot(y, ws[:, m], ls, color=col_c[1.25], lw=1.8,
               label=f'{v:+.2f} meV, {near[m]:.0%} within 100 nm')
    a.axvspan(0, 100, color=GRIDL, alpha=0.5, lw=0)
    a.text(50, 0.075, 'N strip', ha='center', fontsize=8, color=INK2L)
    a.set_xlim(-250, 450)
    a.set_xlabel('y across the junction (nm); S outside the strip', color=INKL)
    a.set_ylabel(r'$|\psi|^2$ per site (k = 0, normal state)', color=INKL)
    a.set_title(f'tc 1.25: two levels bound in the junction (S band from {BOT_C:+.2f})', color=INKL, fontsize=10)
    a.legend(fontsize=8, frameon=False, loc='center right')
    fig.suptitle(r'Checks: why tc > 1 loses protection, and the $\tau$ 1 charge; mixed SOC, W 100 nm, L 2 $\mu$m, '
                 r'$\mu_C$ 0.7 meV, a 10 nm', color=INKL, fontsize=12)
    st.save_fig(fig, 'leads_checks_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/leads_checks_mixed.png", flush=True)


# ====================================================================================================================
# PART 'spectrum'  (2026-11-13)
# ====================================================================================================================
# STUDY: how the PROBE bond tau changes the DOS(E, phi) at one field.  The [maps] DOS(E = 0) maps (a = 5) show isolated
# bright spots inside Q = -1; here the a = 10 thesis junction (mixed SOC, mu_C 0.7, tc 1, W 100, L 2 um), probes ATTACHED
# by tau in the DOS (tools.dos_total_phis tau=; tau = 0 = probes detached).
# 1. DOS(E = 0) survey at tau 0.5 (E_Z 0.30-1.30 step 0.025, 121 phases, eta 1e-4 meV as [maps]).  The brightest point
#    inside Q = -1 (the [tuning] coarse Q map, eroded by one cell so no boundary point is picked) fixes E_Z.
# 2. Along phi at that E_Z: Q and Delta_top of the infinite cross-section (tau-independent; refined gap), and kappa, G,
#    det r at E = 0 for each tau > 0.
# 3. DOS(E, phi) for E = 0 .. SP_EMAX (step SP_DE, eta SP_ETA) and each tau.  E < 0 from particle-hole symmetry
#    (DOS(-E) = DOS(E) of the BdG system, leads included), checked at one negative energy per tau.
# Cost: ~11 s per field or energy at a = 10 in one process with all 16 BLAS threads, dominated by the 121 per-phase
# solves of size 4 nx = 800; SP_W workers x SP_THREADS BLAS threads instead (resumable, chunksize 1).
# Figure plots/spectrum_mixed_a10.png.
SP_W, SP_THREADS = 6, 2
SP_TAUS = (0.0, 0.5, 1.0)
SP_TAU_SURVEY = 0.5
SP_PH = np.linspace(0.0, 2 * np.pi, 121)
SP_EZ = np.round(np.arange(0.30, 1.3001, 0.025), 4)
SP_EMAX, SP_DE, SP_ETA = 0.040, 0.001, 5e-4                          # meV: 0 .. 40 ueV, step 1 ueV, eta 0.5 ueV


def spectrum_topo(E_Z, phi):
    """[Q, Delta_top (ueV, refined), its k (1/nm)] of the thesis cross-section at (E_Z, phi); tau does not enter."""
    q = junction('mixed', 0.7)
    p = q.params(E_Z=float(E_Z), phi=float(phi))
    _, g, k = st.gaps(p, st.n_S_of(q), st.k_grid(q.a))
    return np.array([st.q_point(q, E_Z, phi)[0], g * q.t * 1e3, k / q.a])


# %% [spectrum] --- 1. DOS(E = 0) SURVEY AND THE BRIGHT SPOT ---
if run_part('spectrum'):
    import os
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = str(SP_THREADS)        # read by the spawned workers (make_pool only sets it if unset)
    q_sp = junction('mixed', 0.7, SP_TAU_SURVEY)
    SP_S = st.run_scan(partial(st.dos_total_phis, q_sp, phis=SP_PH, eta=1e-4, E=0.0, tau=SP_TAU_SURVEY), {'E_Z': SP_EZ},
                       f'scharf_absSpec_E0_mixed_a10_tau{SP_TAU_SURVEY:g}', n_workers=SP_W, chunksize=1)
    print("spectrum: survey done", flush=True)
    _z = np.load(st.DATA / 'scharf_absTune_Q_mixed_mu0.7.npz')
    SP_QT, SP_EZQ, SP_PHQ = _z['values'][..., 0], _z['axis_E_Z'], _z['axis_phi']
    _in = SP_QT < 0
    _er = _in.copy()
    _er[1:] &= _in[:-1]
    _er[:-1] &= _in[1:]
    _er &= np.roll(_in, 1, 1) & np.roll(_in, -1, 1)
    SP_IN = _er[np.ix_(np.abs(SP_EZ[:, None] - SP_EZQ[None]).argmin(1), np.abs(SP_PH[:, None] - SP_PHQ[None]).argmin(1))]
    _c = np.where(SP_IN, SP_S, -np.inf)
    print(f"\n  [spectrum] DOS(E = 0) survey, tau {SP_TAU_SURVEY:g}, a 10: median inside Q = -1 {np.median(SP_S[SP_IN]):.0f}, "
          f"max {SP_S[SP_IN].max():.0f} /meV; brightest points inside (eroded) Q = -1:")
    for j in np.argsort(_c.ravel())[::-1][:8]:
        i, p_ = np.unravel_index(j, _c.shape)
        print(f"    E_Z {SP_EZ[i]:.3f}  phi {SP_PH[p_] / np.pi:.3f} pi  DOS {SP_S[i, p_]:.0f}")
    _i, _p = np.unravel_index(np.argmax(_c), _c.shape)
    EZ_B, PHI_B = float(SP_EZ[_i]), float(SP_PH[_p])
    print(f"  -> E_Z {EZ_B:g} meV (spot at phi {PHI_B / np.pi:.3f} pi)", flush=True)

# %% [spectrum] --- 2. ALONG phi AT E_Z: Q, Delta_top (cross-section); kappa, G, det r per tau ---
if run_part('spectrum'):
    SP_PHT = SP_PH[::2]
    SP_T = st.run_scan(spectrum_topo, {'E_Z': [EZ_B], 'phi': SP_PHT}, f'scharf_absSpec_topo_mixed_a10_EZ{EZ_B:g}',
                       n_workers=SP_W, chunksize=1)[0]
    SP_TR = {tau: st.run_scan(partial(st.transport_phis, junction('mixed', 0.7, tau), phis=SP_PH), {'E_Z': [EZ_B]},
                              f'scharf_absSpec_tr_mixed_a10_tau{tau:g}_EZ{EZ_B:g}', n_workers=1)[0]
             for tau in SP_TAUS if tau > 0}
    print("spectrum: line done", flush=True)

# %% [spectrum] --- 3. DOS(E, phi) PER PROBE BOND ---
if run_part('spectrum'):
    SP_E = np.round(np.arange(0.0, SP_EMAX + 1e-9, SP_DE), 6)
    SP_D = {}
    for tau in SP_TAUS:
        SP_D[tau] = st.run_scan(partial(st.dos_total_phis, q_sp, EZ_B, SP_PH, SP_ETA, tau=tau), {'E': SP_E},
                                f'scharf_absSpec_Ephi_mixed_a10_tau{tau:g}_EZ{EZ_B:g}', n_workers=SP_W, chunksize=1)
        print(f"spectrum: DOS(E, phi) tau {tau:g} done", flush=True)

# %% [spectrum] --- 4. NUMBERS ---
if run_part('spectrum'):
    _jb = int(np.argmin(np.abs(SP_PHT - PHI_B)))
    gl = (SP_T[:, 0] < 0) & (SP_T[:, 1] < 1e-2)
    print(f"\n  [spectrum] E_Z {EZ_B:g}: Q = -1 on {np.mean(SP_T[:, 0] < 0):.0%} of phi; gapless (Delta_top < 0.01 ueV) "
          f"inside Q = -1 at phi/pi = " + (", ".join(f"{v:.3f}" for v in SP_PHT[gl] / np.pi) or "none"))
    print(f"    at the spot (phi {SP_PHT[_jb] / np.pi:.3f} pi): Q {SP_T[_jb, 0]:+.0f}, Delta_top {SP_T[_jb, 1]:.3f} ueV "
          f"at k {SP_T[_jb, 2]:.4f} /nm; Delta_top over phi {SP_T[:, 1].min():.2f} .. {SP_T[:, 1].max():.2f} ueV")
    _pb = int(np.argmin(np.abs(SP_PH - PHI_B)))
    for tau, tr in SP_TR.items():
        print(f"    tau {tau:g} at the spot: kappa {tr[_pb, 0]:.3f}, |G| {abs(tr[_pb, 1]):.3f}, det r {tr[_pb, 3]:+.3f}, "
              f"andreev {tr[_pb, 2]:.3f}; failed solves on the line {int(np.nansum(tr[:, 4]))}")
    for tau in SP_TAUS:
        D = SP_D[tau]
        j = int(np.argmin(np.abs(SP_E - 0.010)))
        neg = st.dos_total_phis(q_sp, EZ_B, SP_PH, SP_ETA, E=-float(SP_E[j]), tau=tau)
        print(f"    tau {tau:g}: DOS(E = 0) at the spot {D[0, _pb]:.0f} /meV, median over phi {np.median(D[0]):.1f}; "
              f"PHS check at E = -{SP_E[j] * 1e3:g} ueV: max |DOS(-E) - DOS(E)| / max DOS(E) = "
              f"{np.max(np.abs(neg - D[j])) / np.max(D[j]):.1e}", flush=True)

# %% [spectrum] --- FIGURE ---
if run_part('spectrum'):
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4), layout='constrained')
    Ef = np.r_[-SP_E[:0:-1], SP_E] * 1e3
    allD = np.concatenate([SP_D[t].ravel() for t in SP_TAUS])
    norm = LogNorm(vmin=max(np.percentile(allD, 2), 1e-3), vmax=np.percentile(allD, 99.8))
    gap = np.where((SP_T[:, 0] < 0) & (SP_T[:, 1] >= 1e-2), SP_T[:, 1], np.nan)
    for a, tau in zip(axs[0], SP_TAUS):
        Df = np.r_[SP_D[tau][:0:-1], SP_D[tau]]
        im = a.pcolormesh(SP_PH / np.pi, Ef, Df, shading='nearest', cmap='magma', norm=norm, rasterized=True)
        for s in (1, -1):
            a.plot(SP_PHT / np.pi, s * gap, color='#7fd6ff', lw=1.2, ls='--')
        a.plot(SP_PHT[gl] / np.pi, np.zeros(gl.sum()), 'o', ms=4, color='#7fd6ff')
        a.axvline(PHI_B / np.pi, color='w', lw=0.8, ls=':')
        a.set_ylim(-SP_EMAX * 1e3, SP_EMAX * 1e3)
        a.set_title((r'$\tau$ = 0 (probes detached)' if tau == 0 else rf'probe bond $\tau$ = {tau:g}'), fontsize=11)
        a.set_xlabel(r'$\phi/\pi$')
        a.set_ylabel(r'$E$ ($\mu$eV)')
    fig.colorbar(im, ax=axs[0], label=r'total DOS (1/meV), $\eta$ = ' + f'{SP_ETA * 1e3:g}' + r' $\mu$eV', shrink=0.9)
    a = axs[1, 0]
    im2 = a.pcolormesh(SP_PH / np.pi, SP_EZ, SP_S, shading='nearest', cmap='magma', vmin=0,
                       vmax=np.percentile(SP_S, 99.5), rasterized=True)
    a.contour(SP_PHQ / np.pi, SP_EZQ, SP_QT, levels=[0], colors='w', linewidths=1.1, linestyles='--')
    a.axhline(EZ_B, color='#7fd6ff', lw=1.2)
    a.plot(PHI_B / np.pi, EZ_B, 'o', mfc='none', mec='#7fd6ff', ms=10, mew=1.5)
    a.set_ylim(SP_EZ[0], SP_EZ[-1])
    a.set_xlabel(r'$\phi/\pi$')
    a.set_ylabel(r'$E_Z$ (meV)')
    a.set_title(rf'DOS($E$ = 0), $\tau$ {SP_TAU_SURVEY:g}, $\eta$ 0.1 $\mu$eV; line: $E_Z$ = {EZ_B:g} meV', fontsize=11)
    fig.colorbar(im2, ax=a, label='1/meV', extend='max')
    a = axs[1, 1]
    a.plot(SP_PHT / np.pi, SP_T[:, 1], color='#2a78d6', lw=2)
    for x0 in SP_PHT[SP_T[:, 0] > 0] / np.pi:
        a.axvspan(x0 - 1 / 60, x0 + 1 / 60, color='#e4e3df', lw=0, zorder=0)
    a.plot(SP_PHT[gl] / np.pi, np.zeros(gl.sum()), 'o', ms=5, color='#eb6834', label='Q = -1 but gapless')
    a.axvline(PHI_B / np.pi, color='#52514e', lw=0.8, ls=':')
    a.set_xlabel(r'$\phi/\pi$')
    a.set_ylabel(r'$\Delta_{top}$ ($\mu$eV), infinite junction')
    a.set_title(rf'bulk gap along $\phi$ at $E_Z$ = {EZ_B:g} (grey: Q = +1)', fontsize=11)
    if gl.any():
        a.legend(fontsize=9, frameon=False)
    a = axs[1, 2]
    for (tau, tr), col in zip(SP_TR.items(), ('#2a78d6', '#1baf7a')):
        a.plot(SP_PH / np.pi, tr[:, 0], color=col, lw=2, label=rf'$\kappa/\kappa_0$, $\tau$ {tau:g}')
        a.plot(SP_PH / np.pi, np.abs(tr[:, 1]), color=col, lw=1.4, ls=':', label=rf'$|G|/G_0$, $\tau$ {tau:g}')
    a.axhline(0.5, color='#52514e', lw=0.8, ls='--')
    a.axvline(PHI_B / np.pi, color='#52514e', lw=0.8, ls=':')
    a.set_xlabel(r'$\phi/\pi$')
    a.set_title(rf'transport at $E$ = 0, $E_Z$ = {EZ_B:g}', fontsize=11)
    a.legend(fontsize=9, frameon=False, ncol=2)
    fig.suptitle(rf'Mixed SOC, $\mu_C$ 0.7 meV, W 100 nm, L 2 $\mu$m, a 10 nm: DOS($E$, $\phi$) at $E_Z$ = {EZ_B:g} meV vs the '
                 r'probe bond $\tau$ (dashed: $\pm\Delta_{top}$ of the infinite junction; dots: Q = -1 but gapless; '
                 r'$E < 0$ by PHS)', fontsize=11)
    st.save_fig(fig, 'spectrum_mixed_a10', dpi=150)
    plt.close(fig)
    print("figure: plots/spectrum_mixed_a10.png", flush=True)


# ====================================================================================================================
# PART 'ep'  (2026-11-13)
# ====================================================================================================================
# STUDY: does the open junction show the exceptional points of Li & Trauzettel (PRB 112, 184504; arXiv 2507.23022)?
# There a loss -iV delta(x) at a p-wave junction acts on ONE Majorana combination: the zero-energy crossing at phi = pi
# opens into a purely imaginary segment bounded by two EPs (Re E = 0, Im E split, eigenvectors coalesce at the EPs).
# Here the loss is the PROBE LEADS themselves: tools.nh_spectrum = poles of G with the exact ribbon and probe
# self-energies (the G of transport_point / dos_total_phis).  Two-Majorana poles
#     E = -i Gbar +- sqrt(eps^2 - delta^2 - c^2), delta = (G_L - G_R)/2, c = cross term (probe touches the far tail)
# -> EPs at |eps| = sqrt(delta^2 + c^2); equal bonds: delta = 0 exactly (mirror check), EPs ~0.001 pi wide from c alone.
# tau_R fixed at EP_TAU_R, tau_L over EP_TAUS (tau_L = tau_R: the symmetric control); the lossless reference (probes
# detached) gives +-eps(phi).  Thesis junction (mixed SOC, mu_C 0.7, tc 1, W 100, L 2 um, a 10) at E_Z = EP_EZ:
# phi = pi is a closed-box zero there (lossless pole 0.407 ueV = the cached box eps 0.40717).
# ~8 s per phase on one thread; one pool for all, so each tau_L takes ~3 min and is drawn at once
# (plots/ep_mixed_tauL<tau>.png); caches scharf_absEP_*.
EP_EZ = 0.93
EP_TAU_R = 0.1
EP_TAUS = tuple(float(t) for t in np.round(np.arange(0.1, 0.801, 0.1), 2))
EP_PH = np.linspace(0.0, 2 * np.pi, 121)
EP_K = 8
EP_W = 6


def ep_point(phi, tau_L, tau_R=EP_TAU_R, E_Z=EP_EZ, k=EP_K):
    """[Re of EP_K poles, their Im, overlap of the pair] (ueV) at one phase: the k poles closest to 0, ordered by |Re E|
    (nh_spectrum), the first EP_K kept.  k > EP_K when the broad Majorana pole lies farther from 0 than EP_K bulk poles
    (E_Z 0.70, tau_L 0.8: Im ~ -48 ueV vs bulk levels at |Re E| 25-40 -- with k = 8 it was not even computed)."""
    P, O = st.nh_spectrum(junction('mixed', 0.7), E_Z, [float(phi)], tau_L, tau_R, k=k)
    return np.r_[P[0, :EP_K].real, P[0, :EP_K].imag, O]


def ep_figure(tau_L, R, ref, tau_R=EP_TAU_R, E_Z=EP_EZ):
    """Re E, Im E of the poles and the eigenvector overlap of the two closest to 0 along phi; prints the EP phases."""
    K, x = EP_K, EP_PH / np.pi
    Re, Im, ov = R[:, :K], R[:, K:2 * K], R[:, 2 * K]
    eps = np.abs(ref[:, 0])
    btw = (np.abs(Re[:, 0]) < 1e-3) & (np.abs(Re[:, 1]) < 1e-3) & (np.abs(Im[:, 0] - Im[:, 1]) > 1e-3)
    ch = np.flatnonzero(np.diff(btw.astype(int)))
    ep_phi = (x[ch] + x[ch + 1]) / 2
    e = np.flatnonzero(np.diff(np.r_[0, btw.astype(int), 0]))
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 9.2), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.2, 1.2, 0.7)))
    for a in axs:
        for i0, i1 in zip(e[::2], e[1::2]):
            a.axvspan(x[i0] - 1 / 120, x[i1 - 1] + 1 / 120, color='#f5c9b3', lw=0, zorder=0)
        for ph in ep_phi:
            a.axvline(ph, color='#eb6834', lw=1, ls=':')
        a.grid(color='#e4e3df', lw=0.6)
    C1, C2, CO = '#2a78d6', '#1baf7a', '#b5b4ae'
    for a, Y, lab in ((axs[0], Re, r'Re $E$ ($\mu$eV)'), (axs[1], Im, r'Im $E$ ($\mu$eV)')):
        a.plot(x, Y[:, 2:], '.', color=CO, ms=2.5)
        a.plot(x, Y[:, 1], 'o', color=C2, ms=5.5, mew=0, label='Majorana pair: pole 2')       # pole 1 drawn on top,
        a.plot(x, Y[:, 0], 'o', color=C1, ms=2.6, mew=0, label='Majorana pair: pole 1')       # smaller: both visible
        a.plot([], [], '.', color=CO, ms=5, label='other poles (bulk)')
        a.set_ylabel(lab)
    a = axs[0]
    a.plot(x, eps, '--', color='#52514e', lw=1.2, label=r'$\pm\varepsilon$, probes detached')
    a.plot(x, -eps, '--', color='#52514e', lw=1.2)
    lim = min(40.0, max(5.0, 1.4 * max(np.abs(Re[:, :2]).max(), eps.max())))
    a.set_ylim(-lim, lim)
    a.legend(fontsize=8.5, frameon=False, loc='upper right', ncol=2)
    a = axs[1]
    lo = 1.3 * min(Im[:, :2].min(), -0.5)
    a.set_ylim(lo, -0.05 * lo)
    a.legend(fontsize=8.5, frameon=False, loc='lower right', ncol=3)
    a = axs[2]
    a.plot(x, ov, color='#2a78d6', lw=1.6)
    a.set_ylim(0, 1.05)
    a.set_ylabel(r'$|\langle\psi_1|\psi_2\rangle|$')
    a.set_xlabel(r'$\phi/\pi$')
    a.set_xlim(0, 2)
    fig.suptitle(rf'Poles of the open junction: probe bonds $\tau_L$ = {tau_L:g}, $\tau_R$ = {tau_R:g}, $E_Z$ = {E_Z:g} meV'
                 '\n' r'mixed SOC, L 2 $\mu$m, a 10.  Shaded: both poles at Re $E$ = 0 (between EPs); dotted: EPs',
                 fontsize=11)
    name = (f'ep_mixed_tauL{tau_L:g}' + ('' if tau_R == EP_TAU_R else f'_tauR{tau_R:g}')
            + ('' if E_Z == EP_EZ else f'_EZ{E_Z:g}'))
    st.save_fig(fig, name, dpi=150)
    plt.close(fig)
    j = int(np.argmin(np.abs(EP_PH - np.pi)))
    print(f"  tau_L {tau_L:g} / tau_R {tau_R:g}: EPs at phi/pi = " + (", ".join(f"{v:.3f}" for v in ep_phi) or "none")
          + f"; between EPs {btw.mean() * 2:.3f} pi of phase; max overlap {ov.max():.3f}; at phi = pi poles "
          f"{Re[j, 0]:+.3f}{Im[j, 0]:+.3f}i, {Re[j, 1]:+.3f}{Im[j, 1]:+.3f}i ueV", flush=True)
    print(f"figure: plots/{name}.png", flush=True)


# %% [ep] --- 1. POLES ALONG phi, ONE tau_L AT A TIME (each drawn as soon as it is done) ---
if run_part('ep'):
    EP_R = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for _tl, _tr, _nm in [(0.0, 0.0, 'ref')] + [(t, EP_TAU_R, f'tauL{t:g}') for t in EP_TAUS]:
            EP_R[_nm] = st.myf.scan(partial(ep_point, tau_L=_tl, tau_R=_tr), {'phi': EP_PH}, pool=_pool, chunksize=1,
                                    save=str(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_{_nm}_R{_tr:g}.npz'))['values']
            if _nm == 'ref':
                print(f"ep: lossless reference done; eps at phi = pi {abs(EP_R['ref'][60, 0]):.3f} ueV, "
                      f"max over phi {np.abs(EP_R['ref'][:, 0]).max():.2f} ueV", flush=True)
            else:
                ep_figure(_tl, EP_R[_nm], EP_R['ref'], _tr)
    finally:
        _pool.shutdown()


# %% [ep] --- 2. FINE phi SCANS AROUND EVERY EP, and the tau_L = tau_R control around its crossings ---
# Coarse step 0.0167 pi: EPs known to +-0.008 pi, segments a few points wide.  Fine: +-EP_FINE_HALF around every coarse EP
# at EP_FINE_STEP (windows of close EPs merge).  Control tau_L = tau_R = 0.1: the pole pair gives delta_eff = overlap x eps
# ~ 0.017 ueV (= |c|, the cross term; [ep2]: G_L = G_R exactly), i.e. EP segments ~0.001 pi wide (found: 0.0006-0.0008) ->
# +-EP_CTRL_HALF around both crossings at EP_CTRL_STEP.  Figures plots/ep_fine_mixed_tauL<tau>.png.
EP_FINE_HALF, EP_FINE_STEP = 0.012, 0.001                          # units of pi
EP_CTRL_HALF, EP_CTRL_STEP = 0.01, 0.0002


def ep_between(R):
    """Both poles of the pair on Re E = 0 with different Im (between two EPs), per phase."""
    Re, Im = R[:, :EP_K], R[:, EP_K:2 * EP_K]
    return (np.abs(Re[:, 0]) < 1e-3) & (np.abs(Re[:, 1]) < 1e-3) & (np.abs(Im[:, 0] - Im[:, 1]) > 1e-3)


def ep_windows(x, R, half, step, ctrl=False):
    """Fine phase grid (units of pi) +-half around every coarse EP (ctrl: around every crossing = local minimum of
    |Re E| of the pair below 1 ueV), overlapping windows merged; and the centres."""
    if ctrl:
        r = np.abs(R[:, 0])
        c = [float(x[i]) for i in range(1, len(x) - 1) if r[i] <= r[i - 1] and r[i] <= r[i + 1] and r[i] < 1.0]
    else:
        ch = np.flatnonzero(np.diff(ep_between(R).astype(int)))
        c = list((x[ch] + x[ch + 1]) / 2)
    if not c:
        return np.array([]), c
    pts = np.concatenate([np.arange(v - half, v + half + step / 2, step) for v in c])
    return np.unique(np.round(np.round(pts / step) * step % 2.0, 6)), c


def ep_fine_figure(tau_L, xs, R, c, tau_R=EP_TAU_R, ctrl=False):
    """One column per window: Re E, Im E of the pair (+ other poles), overlap; EPs located on the fine grid."""
    K = EP_K
    Re, Im, ov = R[:, :K], R[:, K:2 * K], R[:, 2 * K]
    b = ep_between(R)
    step = np.min(np.diff(xs))
    cont = np.diff(xs) < 1.5 * step
    groups = np.split(np.arange(len(xs)), np.flatnonzero(~cont) + 1)
    ch = np.flatnonzero((np.diff(b.astype(int)) != 0) & cont)
    epf = (xs[ch] + xs[ch + 1]) / 2
    C1, C2, CO = '#2a78d6', '#1baf7a', '#b5b4ae'
    fig, axs = plt.subplots(3, len(groups), figsize=(3.4 * len(groups) + 1.2, 8.4), sharex='col', squeeze=False,
                            layout='constrained', gridspec_kw=dict(height_ratios=(1.1, 1.1, 0.8)))
    print(f"  fine, tau_L {tau_L:g} / tau_R {tau_R:g} (step {step:g} pi): EPs at phi/pi = "
          + (", ".join(f"{v:.4f}" for v in epf) or "none") + "  (coarse: " + ", ".join(f"{v:.3f}" for v in c) + ")",
          flush=True)
    for j, g in enumerate(groups):
        xg = xs[g]
        e = np.flatnonzero(np.diff(np.r_[0, b[g].astype(int), 0]))
        for r_, Y, lab in ((0, Re, r'Re $E$ ($\mu$eV)'), (1, Im, r'Im $E$ ($\mu$eV)')):
            a = axs[r_, j]
            a.plot(xg, Y[g, 2:], '.', color=CO, ms=2)
            a.plot(xg, Y[g, 1], 'o', color=C2, ms=4.5, mew=0, label='pair: pole 2')
            a.plot(xg, Y[g, 0], 'o', color=C1, ms=2.2, mew=0, label='pair: pole 1')
            span = np.abs(Y[g, :2]).max()
            if r_ == 0:
                a.set_ylim(-1.3 * max(span, 0.02), 1.3 * max(span, 0.02))
            else:
                a.set_ylim(-1.15 * max(span, 0.02), 0.05 * max(span, 0.02))
            if j == 0:
                a.set_ylabel(lab)
        axs[2, j].plot(xg, ov[g], color=C1, lw=1.6)
        axs[2, j].set_ylim(0, 1.05)
        axs[2, j].set_xlabel(r'$\phi/\pi$')
        for a in axs[:, j]:
            for i0, i1 in zip(e[::2], e[1::2]):
                a.axvspan(xg[i0] - step / 2, xg[i1 - 1] + step / 2, color='#f5c9b3', lw=0, zorder=0)
            for v in epf[(epf > xg[0]) & (epf < xg[-1])]:
                a.axvline(v, color='#eb6834', lw=1, ls=':')
            a.grid(color='#e4e3df', lw=0.6)
        inside = b[g]
        print(f"    window {xg[0]:.3f}-{xg[-1]:.3f} pi: between-EP width {inside.sum() * step:.4f} pi; overlap max "
              f"{ov[g].max():.3f}" + (f", min inside {ov[g][inside].min():.3f}" if inside.any() else "")
              + f"; pair Im range {Im[g, :2].min():.3f}..{Im[g, :2].max():.3f} ueV", flush=True)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[2, 0].set_ylabel(r'$|\langle\psi_1|\psi_2\rangle|$')
    fig.suptitle(rf'Fine $\phi$ scans: $\tau_L$ = {tau_L:g}, $\tau_R$ = {tau_R:g}, $E_Z$ = {EP_EZ:g} meV, step {step:g}$\pi$'
                 + (' (symmetric control, around the crossings)' if ctrl else '')
                 + '\nshaded: both poles at Re E = 0 (between EPs); dotted: EPs', fontsize=11)
    name = f'ep_fine_mixed_tauL{tau_L:g}'
    st.save_fig(fig, name, dpi=150)
    plt.close(fig)
    print(f"figure: plots/{name}.png", flush=True)


if run_part('ep'):
    _x = EP_PH / np.pi
    EP_F = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for _tl in EP_TAUS:
            _nm = f'tauL{_tl:g}'
            _ctrl = _tl == EP_TAU_R
            _xs, _c = ep_windows(_x, EP_R[_nm], EP_CTRL_HALF if _ctrl else EP_FINE_HALF,
                                 EP_CTRL_STEP if _ctrl else EP_FINE_STEP, _ctrl)
            if not len(_xs):
                continue
            EP_F[_tl] = (_xs, st.myf.scan(partial(ep_point, tau_L=_tl, tau_R=EP_TAU_R), {'phi': _xs * np.pi}, pool=_pool,
                                          chunksize=1, save=str(st.DATA / f'scharf_absEP_fine_mixed_a10_EZ{EP_EZ:g}_{_nm}'
                                                                          f'_R{EP_TAU_R:g}.npz'))['values'])
            ep_fine_figure(_tl, EP_F[_tl][0], EP_F[_tl][1], _c, ctrl=_ctrl)
    finally:
        _pool.shutdown()


# %% [ep] --- 2b. tau_L 0.2: the 0.47 pi crossing, missed by cell 2 (no coarse EP there, coarse overlap 0.76) ---
if run_part('ep'):
    _xs2, _c2 = ep_windows(_x, EP_R['tauL0.2'], EP_FINE_HALF, EP_FINE_STEP, ctrl=True)    # around both crossings
    _xs2 = _xs2[_xs2 < 0.7]                                                              # the 0.47 pi one only
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        _Rx = st.myf.scan(partial(ep_point, tau_L=0.2, tau_R=EP_TAU_R), {'phi': _xs2 * np.pi}, pool=_pool, chunksize=1,
                          save=str(st.DATA / f'scharf_absEP_fine2_mixed_a10_EZ{EP_EZ:g}_tauL0.2_R{EP_TAU_R:g}.npz'))['values']
    finally:
        _pool.shutdown()
    _xa, _Ra = EP_F[0.2]
    _o = np.argsort(np.r_[_xs2, _xa])
    EP_F[0.2] = (np.r_[_xs2, _xa][_o], np.r_[_Rx, _Ra][_o])
    ep_fine_figure(0.2, EP_F[0.2][0], EP_F[0.2][1], [float(v) for v in _c2])


# %% [ep] --- 3. A SECOND FIXED tau_R = 0.5: tau_L on both sides of it ---
# EPs need G_L != G_R.  With tau_R = 0.5 the left probe is the WEAKER one for tau_L < 0.5 and the stronger one above:
# does the EP structure follow |G_L - G_R| only (EPs shrink towards tau_L = tau_R, reappear on both sides)?  And
# (tau_L, tau_R) = (0.1, 0.5) is the mirror of cell 1's (0.5, 0.1): equal poles <=> left and right equivalent (cf. the
# control's delta ~ 0.017 ueV).  tau_L = tau_R skipped (cell 1 has the control).  plots/ep_mixed_tauL<tau>_tauR0.5.png.
EP_TAU_R2 = 0.5
EP_TAUS2 = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8)

if run_part('ep'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for _tl in EP_TAUS2:
            EP_R[(_tl, EP_TAU_R2)] = st.myf.scan(
                partial(ep_point, tau_L=_tl, tau_R=EP_TAU_R2), {'phi': EP_PH}, pool=_pool, chunksize=1,
                save=str(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{_tl:g}_R{EP_TAU_R2:g}.npz'))['values']
            ep_figure(_tl, EP_R[(_tl, EP_TAU_R2)], EP_R['ref'], EP_TAU_R2)
        _pair = lambda R: np.c_[np.abs(R[:, 0]), np.sort(R[:, EP_K:EP_K + 2], axis=1)]    # |Re E|, sorted Im (order-free)
        _m, _o = _pair(EP_R[(0.1, EP_TAU_R2)]), _pair(EP_R['tauL0.5'])
        print(f"  mirror check, pair (|Re E|, Im E) of (tau_L, tau_R) = (0.1, 0.5) vs (0.5, 0.1): max difference "
              f"{np.abs(_m - _o).max():.3f} ueV at the same phi, {np.abs(_m - _o[::-1]).max():.3f} ueV at phi -> 2 pi - phi "
              f"(pair widths up to {np.abs(_o[:, 1:]).max():.2f} ueV)", flush=True)
    finally:
        _pool.shutdown()


# ====================================================================================================================
# PART 'ep2'  (2026-11-13)
# ====================================================================================================================
# The two-Majorana reduction of [ep] from the full model, no fit (tools.nh_pair_2x2): the lossless pair spans the
# subspace, gamma_L / gamma_R = its left- / right-localised combinations, the probe self-energies projected onto it
# (first order in the probes).  Gives G_L, G_R, the cross term c and eps (incl. the probe shift) along phi, and the EPs
# of the 2x2 (sign changes of D = ((lam_1 - lam_2)/2)^2, linearly interpolated) vs the exact EPs of [ep].
# Spot checks (E_Z 0.93): equal bonds 0.1 / 0.1 -> G_L = G_R = 0.100 ueV, |c| 0.011-0.014 ueV; inside the tiny segment
# (0.4695 pi) the 2x2 poles -0.086 i / -0.114 i vs exact -0.087 i / -0.116 i -> the equal-bond EPs come from c.
# (0.3, 0.1): Re to ~0.02 ueV, broad Im 10-16 % low.  (0.5, 0.1) at pi: 2x2 +-0.83 - 1.31 i (split) vs exact -1.19 i,
# -2.73 i (between EPs): first order misses ~1/3 of the total loss once the probe reshapes gamma_L.
# One eigensolve per phase (~8 s), all pairs projected from it.  plots/ep2_mixed.png; cache scharf_absEP2_*.
EP2_PAIRS = tuple((t, EP_TAU_R) for t in EP_TAUS) + tuple((t, EP_TAU_R2) for t in EP_TAUS2)


def ep2_point(phi):
    """nh_pair_2x2 for every pair in EP2_PAIRS at one phase, flattened (len(EP2_PAIRS) * 9)."""
    return st.nh_pair_2x2(junction('mixed', 0.7), EP_EZ, float(phi), EP2_PAIRS).ravel()


def ep_positions(xs, R):
    """EP phases (units of pi) on a phase grid xs (units of pi): changes of ep_between within contiguous stretches."""
    b = ep_between(R)
    step = np.min(np.diff(xs))
    cont = np.diff(xs) < 1.5 * step
    ch = np.flatnonzero((np.diff(b.astype(int)) != 0) & cont)
    return (xs[ch] + xs[ch + 1]) / 2


def zero_crossings(x, D):
    """x where D changes sign (linear interpolation)."""
    i = np.flatnonzero(np.sign(D[:-1]) * np.sign(D[1:]) < 0)
    return x[i] - D[i] * (x[i + 1] - x[i]) / (D[i + 1] - D[i])


if run_part('ep2'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP2 = st.myf.scan(ep2_point, {'phi': EP_PH}, pool=_pool, chunksize=1,
                          save=str(st.DATA / f'scharf_absEP2_mixed_a10_EZ{EP_EZ:g}.npz'))['values']
    finally:
        _pool.shutdown()
    EP2 = EP2.reshape(len(EP_PH), len(EP2_PAIRS), 9)
    _x = EP_PH / np.pi
    _ref = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_ref_R0.npz')['values']
    print(f"\n  [ep2] two-Majorana reduction, E_Z {EP_EZ:g}: EPs of the 2x2 vs exact (phi / pi)")
    _cmp = []
    for j, (tl, tr) in enumerate(EP2_PAIRS):
        pred = zero_crossings(_x, EP2[:, j, 8])
        fine = st.DATA / f'scharf_absEP_fine_mixed_a10_EZ{EP_EZ:g}_tauL{tl:g}_R{tr:g}.npz'
        if fine.exists():
            _z = np.load(fine)
            ex = ep_positions(_z['axis_phi'] / np.pi, _z['values'])
            if tl == 0.2 and tr == EP_TAU_R:
                _z2 = np.load(st.DATA / f'scharf_absEP_fine2_mixed_a10_EZ{EP_EZ:g}_tauL0.2_R{tr:g}.npz')
                ex = np.sort(np.r_[ex, ep_positions(_z2['axis_phi'] / np.pi, _z2['values'])])
        else:
            _c = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{tl:g}_R{tr:g}.npz')['values']
            ex = ep_positions(_x, _c)
        _cmp.append((tl, tr, pred, ex))
        g = EP2[:, j]
        print(f"    ({tl:g}, {tr:g}): 2x2 " + (", ".join(f"{v:.4f}" for v in pred) or "none") + "  |  exact "
              + (", ".join(f"{v:.4f}" for v in ex) or "none") + f"  |  along phi: G_L {g[:, 4].min():.3f}-{g[:, 4].max():.3f}, "
              f"G_R {g[:, 5].min():.3f}-{g[:, 5].max():.3f}, |c| {g[:, 6].min():.4f}-{g[:, 6].max():.4f} ueV", flush=True)
    C1, C2, C3, C4 = '#2a78d6', '#1baf7a', '#eb6834', '#52514e'
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for a, (tl, tr) in zip(axs[:2], ((0.1, 0.1), (0.3, 0.1))):
        g = EP2[:, EP2_PAIRS.index((tl, tr))]
        a.semilogy(_x, np.abs(_ref[:, 0]), color=C4, lw=1.2, ls='--', label=r'$\varepsilon$, probes detached')
        a.semilogy(_x, g[:, 7], color=C4, lw=1.6, label=r'$\varepsilon$ with the probes (2x2)')
        a.semilogy(_x, g[:, 4], color=C1, lw=1.8, label=r'$\Gamma_L$')
        a.semilogy(_x, g[:, 5], color=C2, lw=1.8, label=r'$\Gamma_R$')
        a.semilogy(_x, g[:, 6], color=C3, lw=1.8, label=r'$|c|$ (cross term)')
        a.semilogy(_x, np.sqrt(((g[:, 4] - g[:, 5]) / 2) ** 2 + g[:, 6] ** 2), color=C3, lw=1.0, ls=':',
                   label=r'$\sqrt{\delta^2 + |c|^2}$ (EP threshold)')
        a.set_ylim(1e-3, 30)
        a.set_xlim(0, 2)
        a.set_xlabel(r'$\phi/\pi$')
        a.set_ylabel(r'$\mu$eV')
        a.set_title(rf'2x2 from the full model: $\tau_L$ {tl:g}, $\tau_R$ {tr:g}', fontsize=11)
        a.grid(color='#e4e3df', lw=0.6, which='both')
    axs[0].legend(fontsize=8, frameon=False, loc='lower right', ncol=2)
    a = axs[2]
    for tl, tr, pred, ex in _cmp:
        if tr != EP_TAU_R:
            continue
        a.plot(np.full(len(ex), tl), ex, 'o', color=C1, ms=6, mfc='none', mew=1.5)
        a.plot(np.full(len(pred), tl) + 0.015, pred, 'x', color=C3, ms=6, mew=1.5)
    a.plot([], [], 'o', color=C1, mfc='none', mew=1.5, label='exact (poles of the full model, fine scans)')
    a.plot([], [], 'x', color=C3, mew=1.5, label='2x2 reduction (first order in the probes)')
    a.set_xlabel(r'$\tau_L$  ($\tau_R$ = 0.1)')
    a.set_ylabel(r'EP phase $\phi/\pi$')
    a.set_ylim(0, 2)
    a.set_title('Where the two-Majorana reduction holds', fontsize=11)
    a.legend(fontsize=8, frameon=False, loc='upper left')
    a.grid(color='#e4e3df', lw=0.6)
    fig.suptitle(rf'Two-Majorana reduction of the open junction from the full model (no fit), $E_Z$ = {EP_EZ:g} meV, '
                 r'mixed SOC, L 2 $\mu$m, a 10', fontsize=11)
    st.save_fig(fig, 'ep2_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep2_mixed.png", flush=True)


# ====================================================================================================================
# PART 'ep3'  (2026-11-16)
# ====================================================================================================================
# Follow-ups of [ep] / [ep2]:
# 1. kappa, G, det r (left probe) with ASYMMETRIC probes along phi (tools.transport_phis tau_R=) over the EP segments of
#    [ep]: det r = 0 <=> eps = sqrt(G_L G_R) (Nilsson, c = 0) is not the EP condition |eps| = sqrt(delta^2 + c^2).
#    First look (E_Z 0.93, (0.5, 0.1), phi = pi): det r_L = +0.775 while both poles are purely imaginary (between EPs);
#    kappa and G are identical for (0.5, 0.1) and (0.1, 0.5), det r_L differs slightly (+0.775 / +0.780).
# 2. A weak-asymmetry pair (tau_L, tau_R) = EP3_FINE over the whole phase on a fine grid (EP3_PHF): poles and transport.
# 3. The tau_R = 0.1 sweep of [ep] at a second field EP3_EZ2, the middle of the neutral kappa = 1/2 window at L 2 um
#    ([deck]: 0.60-0.92 meV), where eps is smaller.
# Figures plots/ep3_detr_mixed.png, ep3_fine_mixed_tauL0.4_tauR0.6.png, ep_mixed_tauL<t>_EZ0.7.png; caches scharf_absEP3_*.
EP3_TR_PAIRS = ((0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.6, 0.1))
EP3_FINE = (0.4, 0.6)
EP3_PHF = np.linspace(0.0, 2 * np.pi, 801)                          # step 0.0025 pi
EP3_CHUNK = 89                                                       # 9 chunks of the fine transport line
EP3_EZ2 = 0.70


def ep3_tr(i):
    """transport_phis along EP_PH for EP3_TR_PAIRS[i] (left bond in q, right bond tau_R), E_Z = EP_EZ."""
    tl, tr = EP3_TR_PAIRS[int(i)]
    return st.transport_phis(junction('mixed', 0.7, tl), EP_EZ, EP_PH, tau_R=tr)


def ep3_tr_fine(n):
    """transport_phis for EP3_FINE on chunk n of EP3_PHF (one RGF set-up per chunk, so the pool shares the line)."""
    tl, tr = EP3_FINE
    return st.transport_phis(junction('mixed', 0.7, tl), EP_EZ, EP3_PHF[int(n) * EP3_CHUNK:(int(n) + 1) * EP3_CHUNK],
                             tau_R=tr)


def ep3_shade(a, x, b, step):
    """Shade the between-EP stretches b (bool along x) on axis a."""
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    for i0, i1 in zip(e[::2], e[1::2]):
        a.axvspan(x[i0] - step / 2, x[i1 - 1] + step / 2, color='#f5c9b3', lw=0, zorder=0)


# %% [ep3] --- 1. kappa, G, det r WITH ASYMMETRIC PROBES over the EP segments (E_Z = EP_EZ) ---
if run_part('ep3'):
    _x = EP_PH / np.pi
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP3_T = st.myf.scan(ep3_tr, {'i': np.arange(len(EP3_TR_PAIRS))}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP3_tr_mixed_a10_EZ{EP_EZ:g}.npz'))['values']
    finally:
        _pool.shutdown()
    print(f"\n  [ep3] det r (left probe) with asymmetric probes vs the EP segments, E_Z {EP_EZ:g}:", flush=True)
    fig, axs = plt.subplots(len(EP3_TR_PAIRS), 1, figsize=(9.5, 2.3 * len(EP3_TR_PAIRS) + 0.8), sharex=True,
                            layout='constrained')
    for a, (tl, tr), T in zip(axs, EP3_TR_PAIRS, EP3_T):
        R = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{tl:g}_R{tr:g}.npz')['values']
        b = ep_between(R)
        ep3_shade(a, _x, b, 2 / 120)
        dz = zero_crossings(_x, T[:, 3])
        for v in dz:
            a.axvline(v, color='#1baf7a', lw=1, ls='--')
        a.plot(_x, T[:, 0], color='#2a78d6', lw=1.8, label=r'$\kappa/\kappa_0$')
        a.plot(_x, np.abs(T[:, 1]), color='#eb6834', lw=1.2, ls=':', label=r'$|G|/G_0$')
        a.plot(_x, T[:, 3], color='#1baf7a', lw=1.8, label=r'det $r_L$')
        a.axhline(0, color='#52514e', lw=0.6)
        a.set_ylim(-1.05, 1.05)
        a.set_ylabel(rf'$\tau_L$ {tl:g}, $\tau_R$ {tr:g}', fontsize=10)
        a.grid(color='#e4e3df', lw=0.6)
        ins = T[b, 3]
        print(f"    ({tl:g}, {tr:g}): EPs " + (", ".join(f"{v:.3f}" for v in ep_positions(_x, R)) or "none")
              + "  |  det r = 0 at " + (", ".join(f"{v:.3f}" for v in dz) or "none")
              + (f"  |  det r between EPs {ins.min():+.3f} .. {ins.max():+.3f}" if b.any() else "")
              + f"  |  failed solves {int(np.nansum(T[:, 4]))}", flush=True)
    axs[0].legend(fontsize=9, frameon=False, ncol=3, loc='lower right')
    axs[-1].set_xlabel(r'$\phi/\pi$')
    axs[-1].set_xlim(0, 2)
    fig.suptitle(rf'Asymmetric probes, $E_Z$ = {EP_EZ:g} meV: shaded = between EPs (poles, [ep]); dashed = det $r_L$ = 0',
                 fontsize=11)
    st.save_fig(fig, 'ep3_detr_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep3_detr_mixed.png", flush=True)

# %% [ep3] --- 2. THE WEAK-ASYMMETRY PAIR EP3_FINE OVER THE WHOLE PHASE, FINE GRID: poles and transport ---
if run_part('ep3'):
    _xf = EP3_PHF / np.pi
    _tl, _tr = EP3_FINE
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP3_TF = st.myf.scan(ep3_tr_fine, {'n': np.arange(len(EP3_PHF) // EP3_CHUNK)}, pool=_pool, chunksize=1,
                             save=str(st.DATA / f'scharf_absEP3_trfine_mixed_a10_EZ{EP_EZ:g}_tauL{_tl:g}_R{_tr:g}.npz')
                             )['values'].reshape(-1, 5)
        EP3_PF = st.myf.scan(partial(ep_point, tau_L=_tl, tau_R=_tr), {'phi': EP3_PHF}, pool=_pool, chunksize=1,
                             save=str(st.DATA / f'scharf_absEP3_fine_mixed_a10_EZ{EP_EZ:g}_tauL{_tl:g}_R{_tr:g}.npz')
                             )['values']
    finally:
        _pool.shutdown()
    K = EP_K
    Re, Im, ov = EP3_PF[:, :K], EP3_PF[:, K:2 * K], EP3_PF[:, 2 * K]
    b = ep_between(EP3_PF)
    epf = ep_positions(_xf, EP3_PF)
    dz = zero_crossings(_xf, EP3_TF[:, 3])
    _ref = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_ref_R0.npz')['values']
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.2, 1.1, 0.7, 1.0)))
    for a in axs:
        ep3_shade(a, _xf, b, 2 / 800)
        for v in epf:
            a.axvline(v, color='#eb6834', lw=0.9, ls=':')
        for v in dz:
            a.axvline(v, color='#1baf7a', lw=0.9, ls='--')
        a.grid(color='#e4e3df', lw=0.6)
    for a, Y, lab in ((axs[0], Re, r'Re $E$ ($\mu$eV)'), (axs[1], Im, r'Im $E$ ($\mu$eV)')):
        a.plot(_xf, Y[:, 2:], '.', color='#b5b4ae', ms=1.5)
        a.plot(_xf, Y[:, 1], 'o', color='#1baf7a', ms=3.2, mew=0, label='pair: pole 2')
        a.plot(_xf, Y[:, 0], 'o', color='#2a78d6', ms=1.6, mew=0, label='pair: pole 1')
        a.set_ylabel(lab)
    axs[0].plot(EP_PH / np.pi, np.abs(_ref[:, 0]), '--', color='#52514e', lw=1, label=r'$\pm\varepsilon$, probes detached')
    axs[0].plot(EP_PH / np.pi, -np.abs(_ref[:, 0]), '--', color='#52514e', lw=1)
    axs[0].set_ylim(-1.4 * np.abs(Re[:, :2]).max(), 1.4 * np.abs(Re[:, :2]).max())
    axs[0].legend(fontsize=8.5, frameon=False, ncol=3, loc='upper right')
    axs[1].set_ylim(1.15 * Im[:, :2].min(), 0.05 * abs(Im[:, :2].min()))
    axs[2].plot(_xf, ov, color='#2a78d6', lw=1.4)
    axs[2].set_ylim(0, 1.05)
    axs[2].set_ylabel(r'$|\langle\psi_1|\psi_2\rangle|$')
    axs[3].plot(_xf, EP3_TF[:, 0], color='#2a78d6', lw=1.6, label=r'$\kappa/\kappa_0$')
    axs[3].plot(_xf, np.abs(EP3_TF[:, 1]), color='#eb6834', lw=1.1, ls=':', label=r'$|G|/G_0$')
    axs[3].plot(_xf, EP3_TF[:, 3], color='#1baf7a', lw=1.6, label=r'det $r_L$')
    axs[3].axhline(0, color='#52514e', lw=0.6)
    axs[3].set_ylim(-1.05, 1.05)
    axs[3].legend(fontsize=8.5, frameon=False, ncol=3, loc='lower right')
    axs[3].set_xlabel(r'$\phi/\pi$')
    axs[3].set_xlim(0, 2)
    fig.suptitle(rf'$\tau_L$ = {_tl:g}, $\tau_R$ = {_tr:g}, $E_Z$ = {EP_EZ:g} meV, step 0.0025$\pi$: shaded = between EPs '
                 r'(dotted: EPs); dashed green = det $r_L$ = 0', fontsize=11)
    name = f'ep3_fine_mixed_tauL{_tl:g}_tauR{_tr:g}'
    st.save_fig(fig, name, dpi=150)
    plt.close(fig)
    ins = EP3_TF[b, 3]
    print(f"\n  [ep3] fine ({_tl:g}, {_tr:g}), E_Z {EP_EZ:g}: EPs at phi/pi = " + ", ".join(f"{v:.4f}" for v in epf)
          + f"; between EPs {b.mean() * 2:.3f} pi; max overlap {ov.max():.3f}; det r = 0 at "
          + ", ".join(f"{v:.4f}" for v in dz) + (f"; det r between EPs {ins.min():+.3f} .. {ins.max():+.3f}" if b.any()
                                                  else "") + f"; failed solves {int(np.nansum(EP3_TF[:, 4]))}", flush=True)
    print(f"figure: plots/{name}.png", flush=True)

# %% [ep3] --- 3. THE tau_R = 0.1 SWEEP AT A SECOND FIELD (E_Z = EP3_EZ2) ---
if run_part('ep3'):
    EP3_R2 = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for _tl, _tr, _nm in [(0.0, 0.0, 'ref')] + [(t, EP_TAU_R, f'tauL{t:g}') for t in EP_TAUS]:
            EP3_R2[_nm] = st.myf.scan(partial(ep_point, tau_L=_tl, tau_R=_tr, E_Z=EP3_EZ2, k=16), {'phi': EP_PH}, pool=_pool,
                                      chunksize=1,
                                      save=str(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP3_EZ2:g}_{_nm}_R{_tr:g}.npz'))['values']
            if _nm == 'ref':
                print(f"\n  [ep3] E_Z {EP3_EZ2:g}: lossless eps at phi = pi {abs(EP3_R2['ref'][60, 0]):.3f} ueV, max over "
                      f"phi {np.abs(EP3_R2['ref'][:, 0]).max():.2f} ueV", flush=True)
            else:
                ep_figure(_tl, EP3_R2[_nm], EP3_R2['ref'], _tr, E_Z=EP3_EZ2)
    finally:
        _pool.shutdown()


# ====================================================================================================================
# PART 'ep4'  (2026-11-16)
# ====================================================================================================================
# STUDY: can ASYMMETRIC probe bonds improve the kappa = 1/2 plateau?  [ep3] showed that strong asymmetry ((0.5, 0.1),
# E_Z 0.93) removes the parity-crossing dip (det r_L stays > 0, kappa one broad maximum 0.47 / 0.42) -- but below 1/2,
# |G| / kappa 0.13-0.17 and only ~0.03 pi wide.  Here mild asymmetry around the best symmetric point (tau 0.5, E_Z 0.93,
# study_mixed [plateau]: 1.37 pi along phi, 50 ueV along E_Z at 1.742 pi):
# 1. kappa, G, det r_L vs E_Z (0.80-1.10, step 0.005) at phi = pi and 1.742 pi for EP4_PAIRS_CUT;
# 2. the longest plateau run along phi (241 phases; |kappa - 1/2| < 0.03, |G| < 0.1 kappa, |kappa - (1 - det r^2)/2| < 0.01,
#    Q = -1 from the [plateau] Q lines) over (tau_L, tau_R) in EP4_TAUS^2 at E_Z 0.925 / 0.930 / 0.935.
#    Check: (0.5, 0.5) at 0.930 must give the [plateau] 1.37 pi.
# Figures plots/ep4_EZcut_mixed.png, ep4_plateau_mixed.png; caches scharf_absEP4_*.
EP4_PAIRS_CUT = ((0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.6, 0.5), (0.5, 0.3))
EP4_EZ = np.round(np.arange(0.80, 1.10001, 0.005), 4)
EP4_PHC = (np.pi, 1.7417 * np.pi)
EP4_TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
EP4_EZP = (0.925, 0.93, 0.935)
EP4_PHP = np.linspace(0.0, 2 * np.pi, 241)                          # = study_mixed.PHI_F (the Q lines' grid)


def ep4_cut(i, E_Z):
    """kappa, G, andreev, det r_L, n_bad at the two phases EP4_PHC for EP4_PAIRS_CUT[i] at one field: (2, 5)."""
    tl, tr = EP4_PAIRS_CUT[int(i)]
    return st.transport_phis(junction('mixed', 0.7, tl), E_Z, EP4_PHC, tau_R=tr)


def ep4_line(E_Z, i, j):
    """transport along EP4_PHP for (tau_L, tau_R) = (EP4_TAUS[i], EP4_TAUS[j]) at one field: (241, 5)."""
    return st.transport_phis(junction('mixed', 0.7, EP4_TAUS[int(i)]), E_Z, EP4_PHP, tau_R=EP4_TAUS[int(j)])


def plat_mask(T, Q=None):
    """The [plateau] criterion on transport rows T (..., 5); Q (same leading shape) adds Q = -1."""
    k, G, d = T[..., 0], T[..., 1], T[..., 3]
    m = (np.abs(k - 0.5) < 0.03) & (np.abs(G) < 0.1 * k) & (np.abs(k - (1 - d ** 2) / 2) < 0.01)
    return m & (Q < 0) if Q is not None else m


def longest_periodic(m):
    """Longest run of True on a periodic phase grid whose last point repeats the first (points)."""
    m = np.asarray(m[:-1], bool)
    if m.all():
        return len(m)
    mm = np.roll(m, -int(np.argmin(m)))
    best = cur = 0
    for v in mm:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def longest_run(m):
    """Longest run of True on an open grid (points)."""
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# %% [ep4] --- 1-2. SCANS ---
if run_part('ep4'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP4_L = st.myf.scan(ep4_line, {'E_Z': np.array(EP4_EZP), 'i': np.arange(len(EP4_TAUS)),
                                       'j': np.arange(len(EP4_TAUS))}, pool=_pool, chunksize=1,
                            save=str(st.DATA / 'scharf_absEP4_lines_mixed_a10.npz'))['values']
        print("ep4: plateau lines done", flush=True)
        EP4_C = st.myf.scan(ep4_cut, {'i': np.arange(len(EP4_PAIRS_CUT)), 'E_Z': EP4_EZ}, pool=_pool, chunksize=1,
                            save=str(st.DATA / 'scharf_absEP4_EZcut_mixed_a10.npz'))['values']
        print("ep4: E_Z cuts done", flush=True)
    finally:
        _pool.shutdown()

# %% [ep4] --- 3. PLATEAU ALONG phi OVER (tau_L, tau_R): numbers and figure ---
if run_part('ep4'):
    dphi = 2.0 / (len(EP4_PHP) - 1)
    RUN = np.zeros((len(EP4_EZP), len(EP4_TAUS), len(EP4_TAUS)))
    print("\n  [ep4] longest plateau run along phi (units of pi), Q = -1 included; rows tau_L, columns tau_R "
          + " ".join(f"{t:g}" for t in EP4_TAUS), flush=True)
    for a_, e in enumerate(EP4_EZP):
        Qe = np.load(st.DATA / f'scharf_absPlat_Q_mu0.7_EZ{e:g}.npz')['values'][0, :, 0]
        for i in range(len(EP4_TAUS)):
            for j in range(len(EP4_TAUS)):
                RUN[a_, i, j] = longest_periodic(plat_mask(EP4_L[a_, i, j], Qe)) * dphi
        print(f"    E_Z {e:g}:", flush=True)
        for i, tl in enumerate(EP4_TAUS):
            print(f"      tau_L {tl:g}: " + "  ".join(f"{RUN[a_, i, j]:.3f}" for j in range(len(EP4_TAUS))), flush=True)
    ib = np.unravel_index(np.argmax(RUN), RUN.shape)
    print(f"    best: E_Z {EP4_EZP[ib[0]]:g}, (tau_L, tau_R) = ({EP4_TAUS[ib[1]]:g}, {EP4_TAUS[ib[2]]:g}): {RUN[ib]:.3f} pi;"
          f" symmetric (0.5, 0.5) at 0.93: {RUN[1, 2, 2]:.3f} pi ([plateau]: 1.37 pi); failed solves "
          f"{int(np.nansum(EP4_L[..., 4]))}", flush=True)
    fig, axs = plt.subplots(1, len(EP4_EZP), figsize=(4.3 * len(EP4_EZP) + 1, 4.2), layout='constrained')
    for a_, (a, e) in enumerate(zip(axs, EP4_EZP)):
        im = a.imshow(RUN[a_], origin='lower', cmap='viridis', vmin=0, vmax=max(RUN.max(), 1e-3),
                      extent=(EP4_TAUS[0] - 0.05, EP4_TAUS[-1] + 0.05, EP4_TAUS[0] - 0.05, EP4_TAUS[-1] + 0.05))
        for i, tl in enumerate(EP4_TAUS):
            for j, tr in enumerate(EP4_TAUS):
                a.text(tr, tl, f'{RUN[a_, i, j]:.2f}', ha='center', va='center', fontsize=8,
                       color='w' if RUN[a_, i, j] < 0.6 * RUN.max() else 'k')
        a.plot([0.25, 0.75], [0.25, 0.75], color='w', lw=0.8, ls=':')
        a.set_xticks(EP4_TAUS)
        a.set_yticks(EP4_TAUS)
        a.set_xlabel(r'$\tau_R$')
        a.set_ylabel(r'$\tau_L$')
        a.set_title(rf'$E_Z$ = {e:g} meV', fontsize=11)
    fig.colorbar(im, ax=axs, label=r'longest plateau run along $\phi$ ($\pi$)', shrink=0.9)
    fig.suptitle(r'$\kappa = 1/2 \pm 0.03$, $|G| < 0.1\kappa$, two-Majorana relation, $Q = -1$: asymmetric probe bonds '
                 r'(dotted: $\tau_L = \tau_R$)', fontsize=11)
    st.save_fig(fig, 'ep4_plateau_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep4_plateau_mixed.png", flush=True)

# %% [ep4] --- 4. kappa, G, det r ALONG E_Z AT phi = pi, 1.742 pi: numbers and figure ---
if run_part('ep4'):
    dE = (EP4_EZ[1] - EP4_EZ[0]) * 1e3
    cols = ('#0b0b0b', '#2a78d6', '#1baf7a', '#eb6834', '#8a7ee0')
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 8.6), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.3, 0.8, 1.0)))
    print(f"\n  [ep4] plateau windows along E_Z (step {dE:g} ueV; criterion without Q, Q = -1 throughout 0.80-1.10 here):",
          flush=True)
    for c_, ph in enumerate(EP4_PHC):
        for i, ((tl, tr), col) in enumerate(zip(EP4_PAIRS_CUT, cols)):
            T = EP4_C[i, :, c_]
            lab = rf'$\tau_L$ {tl:g}, $\tau_R$ {tr:g}'
            axs[0, c_].plot(EP4_EZ, T[:, 0], color=col, lw=1.8 if i == 0 else 1.3, label=lab)
            axs[1, c_].plot(EP4_EZ, np.abs(T[:, 1]), color=col, lw=1.3)
            axs[2, c_].plot(EP4_EZ, T[:, 3], color=col, lw=1.8 if i == 0 else 1.3)
            m = plat_mask(T)
            print(f"    phi {ph / np.pi:.4f} pi, ({tl:g}, {tr:g}): longest window {longest_run(m) * dE:.0f} ueV, all plateau "
                  f"points {m.sum() * dE:.0f} ueV; kappa max {T[:, 0].max():.3f}; min det r {T[:, 3].min():+.3f}; "
                  f"failed {int(np.nansum(T[:, 4]))}", flush=True)
        axs[0, c_].axhspan(0.47, 0.53, color='#2a78d6', alpha=0.08, lw=0)
        axs[0, c_].axhline(0.5, color='#52514e', lw=0.8, ls='--')
        axs[0, c_].set_title(rf'$\phi$ = {ph / np.pi:.3f}$\pi$', fontsize=11)
        axs[2, c_].axhline(0, color='#52514e', lw=0.6)
        axs[2, c_].set_xlabel(r'$E_Z$ (meV)')
        for a in axs[:, c_]:
            a.grid(color='#e4e3df', lw=0.6)
    axs[0, 0].set_ylabel(r'$\kappa/\kappa_0$')
    axs[1, 0].set_ylabel(r'$|G|/G_0$')
    axs[2, 0].set_ylabel(r'det $r_L$')
    axs[0, 0].legend(fontsize=8.5, frameon=False, loc='upper left')
    fig.suptitle(r'Asymmetric probe bonds along $E_Z$ (mixed SOC, L 2 $\mu$m, a 10); band: $\kappa = 1/2 \pm 0.03$',
                 fontsize=11)
    st.save_fig(fig, 'ep4_EZcut_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep4_EZcut_mixed.png", flush=True)
