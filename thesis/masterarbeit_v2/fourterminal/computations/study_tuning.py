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
  ep5          (2026-11-18)  STUDY: does the EP segment grow with the junction length (end Majoranas: e^{L / xi_M})?
  ep6          (2026-11-18)  side study: (1/2) ln 2 entropy between EPs (Pino, Meir & Aguado) -- not thesis relevant
  ep7          (2026-11-18)  STUDY: thermal transport across the EP segments -- kappa(E, phi), kappa(T, phi) vs the poles
  ep8          (2026-11-18)  STUDY: EP segments and the kappa plateau ALONG E_Z vs the probe couplings (difference, strength)
  ep9          (2026-11-19)  STUDY: is the kappa = 1/2 plateau along E_Z T-robust?  kappa(E, E_Z), kappa(T, E_Z) for four [ep8] pairs
  nh           (2026-11-19)  PLOTS ONLY from caches: does the probe asymmetry (EP physics) protect the kappa plateau?  maps + T
  trans        (2026-11-19)  STUDY: long junctions at the upper transition (phi = pi): kappa vs L, tau 1 vs (1/2) sech^2(L/xi_M)
  barr         (2026-11-19)  CHECK: one-site barrier probes (lam_barr) vs bond probes: do the [nh] relations hold?
  chern        (2026-11-19)  STUDY: Chern ribbons (out-of-plane Vz in the SCs, [combine] B2): is kappa ~ 1/2 a plateau?
  clean        (2026-11-19)  PLOTS ONLY: the key EP / kappa results redrawn for a meeting (plots/clean/), one message each
  dark         (2026-11-19)  STUDY: San-Jose et al. 2016 Majorana dark states? narrow width vs coupling, helical vs standard probe
  detphi       (2026-11-23)  STUDY: does det r ~ 0 give the kappa = 1/2 plateau along phi too? (plots only, cached phase lines)
  epfield      (2026-11-23)  STUDY: the EP spectrum vs E_Z over the whole topological range, one strong probe 0.3-1.0 + a 0.1 detector
  regimes      (2026-11-24)  PLOTS ONLY: which gate (mu_C) and probe bond (tau) values give different regimes? (cached maps)
  gamma        (2026-11-24)  REDO: Gamma = pole width (Capecelatro et al.) per field along phi = pi; the constant-Gamma collapse
  classes      (2026-11-24)  PLOTS ONLY: regime criteria (topology, L/xi_M, eps/Gamma, Gamma/E_next) tested on the phi = pi line
  dis          (2026-11-25)  STUDY: uncorrelated onsite disorder (seeded) at four regime points: conductances, spectrum, LDOS

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

PARTS = ('topology', 'coupling', 'tuning', 'maps', 'leads', 'spectrum', 'ep', 'ep2', 'ep3', 'ep4', 'ep5', 'ep6', 'ep7', 'ep8',
         'ep9', 'nh', 'trans', 'barr', 'chern', 'clean', 'dark', 'detphi', 'epfield', 'regimes', 'gamma', 'classes',
         'dis')
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


def ep_point(phi, tau_L, tau_R=EP_TAU_R, E_Z=EP_EZ, k=EP_K, sigma=0.0):
    """[Re of EP_K poles, their Im, overlap of the pair] (ueV) at one phase: the k poles closest to sigma (ueV; default
    0), ordered by |Re E| (nh_spectrum), the first EP_K kept.  k > EP_K when the broad Majorana pole lies farther from 0
    than EP_K bulk poles (E_Z 0.70, tau_L 0.8: Im ~ -48 ueV vs bulk levels at |Re E| 25-40 -- with k = 8 it was not even
    computed); an off-axis sigma (e.g. -150j) when it lies beyond MANY bulk poles (tau_L 1.0: -163.5j at E_Z 0.93)."""
    P, O = st.nh_spectrum(junction('mixed', 0.7), E_Z, [float(phi)], tau_L, tau_R, k=k, sigma=sigma)
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


# ====================================================================================================================
# PART 'ep5'  (2026-11-18)
# ====================================================================================================================
# STUDY: does the EP segment grow with the junction length?  Each parity crossing opens into a segment of width
# ~ 2 delta_eff / |d eps / d phi| (DERIVATIONS 7.7).  For END Majoranas eps ~ e^{-L / xi_M}, so at fixed loss the width
# should grow ~ e^{+L / xi_M(phi_c)}; the trivial-regime EPs of Cayao & Sato (2307.15472) are L-independent.  A
# length-dependent EP segment would be a topological signature built from the EPs.
# E_Z = EP_EZ, (tau_L, tau_R) = EP5_PAIR, L in EP5_LS (L 2 um reuses [ep]).  Per L: lossless reference + the pair on
# EP_PH, fine scans (EP_FINE_STEP) around every crossing; at each crossing phase Q, Delta_top and xi_M of the infinite
# junction (slowest_mode) -- a crossing in a gapless stretch has no xi_M and should NOT scale exponentially.
# Figure plots/ep5_length_mixed.png; caches scharf_absEP5_*.
EP5_LS = (1500.0, 2000.0, 2500.0)              # along phi (cell 1); L 3 um stopped half way (L 2.5: no crossing at all)
EP5_PAIR = (0.3, 0.1)
EP5_LSZ = (1500.0, 2000.0, 2500.0, 3000.0)     # along E_Z (cell 3)
EP5_PHZ = 1.7417 * np.pi                        # the best-plateau phase: gapped, xi_M ~ 670 nm at E_Z 0.93
EP5_EZC = np.round(np.arange(0.80, 1.0001, 0.005), 4)
EP5_EZREF = 0.90


def ep5_point(phi, L, tau_L, tau_R, E_Z=EP_EZ, k=EP_K):
    """ep_point for the thesis junction of length L (nm)."""
    P, O = st.nh_spectrum(junction('mixed', 0.7, L=float(L)), E_Z, [float(phi)], tau_L, tau_R, k=k)
    return np.r_[P[0, :EP_K].real, P[0, :EP_K].imag, O]


def ep5_bulk(phi, E_Z=EP_EZ):
    """[Q, Delta_top (ueV), xi_M (nm)] of the infinite cross-section at phi (independent of L and of the probes)."""
    q = junction('mixed', 0.7)
    p = q.params(E_Z=float(E_Z), phi=float(phi))
    n_S = st.n_S_of(q)
    _, g, _ = st.gaps(p, n_S, st.k_grid(q.a))
    lam = slowest_mode(p, n_S)
    return np.array([st.q_point(q, E_Z, phi)[0], g * q.t * 1e3, -q.a / np.log(np.abs(lam))])


def ep5_name(L, nm, tr):
    return (f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_{nm}_R{tr:g}' if L == 2000.0
            else f'scharf_absEP5_mixed_a10_EZ{EP_EZ:g}_L{L / 1000:g}_{nm}_R{tr:g}')


# %% [ep5] --- 1. COARSE LINES AND FINE SCANS AROUND EVERY CROSSING, PER LENGTH ---
if run_part('ep5'):
    _x = EP_PH / np.pi
    _tl, _tr = EP5_PAIR
    EP5 = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for L in EP5_LS:
            ref = st.myf.scan(partial(ep5_point, L=L, tau_L=0.0, tau_R=0.0), {'phi': EP_PH}, pool=_pool, chunksize=1,
                              save=str(st.DATA / (ep5_name(L, 'ref', 0.0) + '.npz')))['values']
            R = st.myf.scan(partial(ep5_point, L=L, tau_L=_tl, tau_R=_tr), {'phi': EP_PH}, pool=_pool, chunksize=1,
                            save=str(st.DATA / (ep5_name(L, f'tauL{_tl:g}', _tr) + '.npz')))['values']
            xs, cs = ep_windows(_x, R, EP_FINE_HALF, EP_FINE_STEP, ctrl=True)     # windows around every crossing
            Rf = st.myf.scan(partial(ep5_point, L=L, tau_L=_tl, tau_R=_tr), {'phi': xs * np.pi}, pool=_pool,
                             chunksize=1, save=str(st.DATA / f'scharf_absEP5_fine_mixed_a10_EZ{EP_EZ:g}_L{L / 1000:g}'
                                                              f'_tauL{_tl:g}_R{_tr:g}.npz'))['values']
            EP5[L] = (ref, R, xs, Rf, cs)
            print(f"ep5: L {L / 1000:g} um done; crossings at phi/pi " + ", ".join(f"{c:.3f}" for c in cs), flush=True)
        _cs_all = sorted({round(c, 3) for L in EP5_LS for c in EP5[L][4]})
        EP5_B = st.myf.scan(ep5_bulk, {'phi': np.array(_cs_all) * np.pi}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP5_bulk_mixed_a10_EZ{EP_EZ:g}.npz'))['values']
    finally:
        _pool.shutdown()

# %% [ep5] --- 2. SEGMENTS vs L: the local law, the slope at the crossings vs L, the total phase between EPs ---
# The crossings MOVE with L (eps ~ cos(k_osc L + phi0)), so no single crossing can be followed from L to L.  Tested:
# (a) per crossing, the measured width (fine scan) vs 2 delta_eff / |d eps / d phi| (lossless slope from the coarse
#     reference; delta_eff = max Im split / 2 inside the segment) -- the local mechanism;
# (b) the lossless slope at the crossings vs L against e^{-L / xi_M} (xi_M of the infinite junction at the crossing
#     phases, gapped ones only) -- where the length enters;
# (c) the total phase between EPs (coarse) vs L.
if run_part('ep5'):
    K = EP_K
    bulk = {round(c, 3): b for c, b in zip(_cs_all, EP5_B)}
    print(f"\n  [ep5] EP segments vs L, E_Z {EP_EZ:g}, (tau_L, tau_R) = {EP5_PAIR}; bulk at the crossing phases "
          "(Q, Delta_top ueV, xi_M nm):", flush=True)
    for c, b in zip(_cs_all, EP5_B):
        print(f"    phi {c:.3f} pi: Q {b[0]:+.0f}, Delta_top {b[1]:.3f}, xi_M {b[2]:.0f}" +
              ("  (GAPLESS: xi_M not meaningful)" if b[1] < 1e-2 else ""), flush=True)
    rows = []                                  # L, phi_c, width, slope, delta_eff, Delta_top, xi_M
    for L in EP5_LS:
        ref, R, xs, Rf, cs = EP5[L]
        eps = np.abs(ref[:, 0])
        if len(xs) < 2:
            print(f"    L {L / 1000:g} um: NO parity crossing along phi (lossless |eps| >= {eps.min():.2f} ueV)", flush=True)
            continue
        step = np.min(np.diff(xs))
        cont = np.diff(xs) < 1.5 * step
        groups = np.split(np.arange(len(xs)), np.flatnonzero(~cont) + 1)
        for g in groups:
            xg = xs[g]
            c = [v for v in cs if xg[0] - 1e-9 <= v <= xg[-1] + 1e-9]
            if not c:
                continue
            b = ep_between(Rf[g])
            width = b.sum() * step
            i0 = int(np.argmin(np.abs(EP_PH / np.pi - c[0])))
            i = min(((i0 + d) % 120 for d in (-2, -1, 0, 1, 2)), key=lambda j: eps[j])   # the lossless crossing
            slope = (eps[(i + 1) % 120] + eps[(i - 1) % 120]) / (2 * (2 / 120))           # ueV per pi
            de = (np.abs(Rf[g][b, K] - Rf[g][b, K + 1]).max() / 2) if b.any() else np.nan
            bb = bulk[round(c[0], 3)]
            rows.append((L, c[0], width, slope, de, bb[1], bb[2]))
            print(f"    L {L / 1000:g} um, crossing {c[0]:.3f} pi ({'gapped' if bb[1] >= 1e-2 else 'GAPLESS'}): segment "
                  f"{width:.4f} pi; |d eps/d phi| {slope:.1f} ueV/pi; delta_eff {de:.3f} ueV; 2 delta_eff / slope "
                  f"{2 * de / slope if slope > 0 else np.nan:.4f} pi", flush=True)
    rows = np.array(rows, float)
    FR = np.array([ep_between(EP5[L][1]).mean() * 2 for L in EP5_LS])
    EM = np.array([np.abs(EP5[L][0][:, 0]).max() for L in EP5_LS])
    for L, fr, em in zip(EP5_LS, FR, EM):
        print(f"    L {L / 1000:g} um: total phase between EPs (coarse) {fr:.3f} pi; max lossless |eps| {em:.2f} ueV",
              flush=True)
    gp = rows[:, 5] >= 1e-2
    xi = np.median(rows[gp, 6]) if gp.any() else np.nan
    Ls = np.array(EP5_LS) / 1000
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.7), layout='constrained')
    a = axs[0]
    pred = 2 * rows[:, 4] / rows[:, 3]
    a.loglog(pred[gp], rows[gp, 2], 'o', color='#2a78d6', ms=7, label='crossing in a gapped stretch')
    a.loglog(pred[~gp], rows[~gp, 2], 'o', mfc='none', mec='#eb6834', ms=7, mew=1.5,
             label='crossing in a gapless stretch')
    both = np.r_[pred, rows[:, 2]]
    both = both[np.isfinite(both) & (both > 0)]
    lim = [0.5 * both.min(), 2 * both.max()]
    a.loglog(lim, lim, color='#52514e', lw=0.8, ls='--', label=r'width $= 2\delta_\mathrm{eff}/|d\varepsilon/d\phi|$')
    a.set_xlabel(r'$2\delta_\mathrm{eff}/|d\varepsilon/d\phi|$ ($\pi$)')
    a.set_ylabel(r'EP segment width, fine scan ($\pi$)')
    a.set_title('(a) the local law, every crossing and L', fontsize=11)
    a.legend(fontsize=8, frameon=False)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    a = axs[1]
    a.semilogy(rows[gp, 0] / 1000, rows[gp, 3], 'o', color='#2a78d6', ms=7, label='gapped')
    a.semilogy(rows[~gp, 0] / 1000, rows[~gp, 3], 'o', mfc='none', mec='#eb6834', ms=7, mew=1.5, label='gapless')
    if gp.any():
        s0 = np.exp(np.mean(np.log(rows[gp, 3]) + rows[gp, 0] / xi))
        a.semilogy(Ls, s0 * np.exp(-Ls * 1000 / xi), color='#52514e', lw=1, ls=':',
                   label=rf'$\propto e^{{-L/\xi_M}}$, median $\xi_M$ {xi:.0f} nm')
    a.set_xlabel(r'$L$ ($\mu$m)')
    a.set_ylabel(r'lossless $|d\varepsilon/d\phi|$ at the crossings ($\mu$eV/$\pi$)')
    a.set_title('(b) where the length enters', fontsize=11)
    a.legend(fontsize=8, frameon=False)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    a = axs[2]
    a.semilogy(Ls, FR, 'o-', color='#2a78d6', lw=1.6, label='total phase between EPs (coarse)')
    if gp.any() and FR[0] > 0:
        a.semilogy(Ls, FR[0] * np.exp((Ls - Ls[0]) * 1000 / xi), color='#52514e', lw=1, ls=':',
                   label=r'$\propto e^{L/\xi_M}$ (median $\xi_M$)')
    a.set_xlabel(r'$L$ ($\mu$m)')
    a.set_ylabel(r'phase between EPs ($\pi$)')
    a.set_title(rf'(c) $\tau_L$ {EP5_PAIR[0]:g}, $\tau_R$ {EP5_PAIR[1]:g}, $E_Z$ {EP_EZ:g} meV', fontsize=11)
    a.legend(fontsize=8, frameon=False)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    st.save_fig(fig, 'ep5_length_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep5_length_mixed.png", flush=True)

# %% [ep5] --- 3. MATCHED ZEROS: along E_Z at a fixed gapped phase, the crossing nearest EP5_EZREF for every L ---
# Along phi the crossings come and go with L (L 2.5: none at E_Z 0.93).  Along E_Z every L has zeros (spacing ~0.15 meV at
# L 2), so the zero nearest a reference field is compared across L at (nearly) the same xi_M: width_EZ ~ 2 delta_eff /
# |d eps / d E_Z| and |d eps / d E_Z| ~ e^{-L / xi_M}  ->  width ~ e^{+L / xi_M}.  Coarse pair line (EP5_EZC) at EP5_PHZ,
# then a fine scan (41 fields) around the crossing nearest EP5_EZREF, the window widened to 1.5x the coarse segment.
# Slope from the pair's own Re E just outside the segment (eps ~ sqrt(Re^2 + delta_eff^2)).  plots/ep5_lengthEZ_mixed.png.
if run_part('ep5'):
    _tl, _tr = EP5_PAIR
    EP5Z = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for L in EP5_LSZ:
            C = st.myf.scan(partial(ep5_point, EP5_PHZ, L=L, tau_L=_tl, tau_R=_tr), {'E_Z': EP5_EZC}, pool=_pool,
                            chunksize=1, save=str(st.DATA / f'scharf_absEP5_EZline_mixed_a10_phi1.7417_L{L / 1000:g}'
                                                            f'_tauL{_tl:g}_R{_tr:g}.npz'))['values']
            r = np.abs(C[:, 0])
            cz = [float(EP5_EZC[i]) for i in range(1, len(r) - 1) if r[i] <= r[i - 1] and r[i] <= r[i + 1] and r[i] < 1.0]
            if not cz:
                print(f"ep5 E_Z: L {L / 1000:g} um: no crossing on the coarse line", flush=True)
                continue
            z0 = min(cz, key=lambda e: abs(e - EP5_EZREF))
            wc = ep_between(C).sum() * 0.005
            half = round(max(0.005, 1.5 * wc), 4)
            ezf = np.round(z0 + np.linspace(-half, half, 41), 6)
            F = st.myf.scan(partial(ep5_point, EP5_PHZ, L=L, tau_L=_tl, tau_R=_tr), {'E_Z': ezf}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP5_EZfine_mixed_a10_phi1.7417_L{L / 1000:g}_z{z0:g}_h{half:g}'
                                                f'_tauL{_tl:g}_R{_tr:g}.npz'))['values']
            EP5Z[L] = (C, z0, ezf, F)
            print(f"ep5 E_Z: L {L / 1000:g} um done; crossings at E_Z " + ", ".join(f"{v:.3f}" for v in cz)
                  + f"; fine around {z0:.3f} +- {half:g} meV", flush=True)
        _zs = sorted({round(v[1], 4) for v in EP5Z.values()})
        EP5ZB = st.myf.scan(partial(ep5_bulk, EP5_PHZ), {'E_Z': np.array(_zs)}, pool=_pool, chunksize=1,
                            save=str(st.DATA / 'scharf_absEP5_EZbulk_mixed_a10_phi1.7417.npz'))['values']
    finally:
        _pool.shutdown()
    K = EP_K
    print(f"\n  [ep5] matched zeros along E_Z at phi {EP5_PHZ / np.pi:.4f} pi, (tau_L, tau_R) = {EP5_PAIR}:", flush=True)
    zrows = []                                       # L, z0, width (ueV of E_Z), slope (ueV per meV), delta_eff, xi_M
    for L, (C, z0, ezf, F) in EP5Z.items():
        ball = ep_between(F)
        step = ezf[1] - ezf[0]
        b = np.zeros_like(ball)
        idx = np.flatnonzero(ball)
        if idx.size:                                 # only the contiguous segment nearest z0 (the window can catch another)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            j0 = int(np.argmin(np.abs(ezf - z0)))
            b[min(runs, key=lambda r: np.min(np.abs(r - j0)))] = True
        width = b.sum() * step * 1e3
        trunc = bool(b[0] or b[-1])
        de = (np.abs(F[b, K] - F[b, K + 1]).max() / 2) if b.any() else np.nan
        bc = ep_between(C)
        i0 = int(np.argmin(np.abs(EP5_EZC - z0)))
        lo, hi = i0, i0
        while lo > 0 and bc[lo]:
            lo -= 1
        while hi < len(EP5_EZC) - 1 and bc[hi]:
            hi += 1
        lo, hi = max(lo - (0 if lo < i0 else 1), 0), min(hi + (0 if hi > i0 else 1), len(EP5_EZC) - 1)
        e_l = np.hypot(C[lo, 0], de if np.isfinite(de) else 0.0)
        e_h = np.hypot(C[hi, 0], de if np.isfinite(de) else 0.0)
        slope = (e_l + e_h) / (EP5_EZC[hi] - EP5_EZC[lo])
        bb = EP5ZB[_zs.index(round(z0, 4))]
        zrows.append((L, z0, width, slope, de, bb[2], bb[1]))
        print(f"    L {L / 1000:g} um: zero at {z0:.3f} meV; segment {width:.2f} ueV of E_Z" + (" (TRUNCATED by the window)"
              if trunc else "") + f"; delta_eff {de:.3f} ueV; |d eps/d E_Z| {slope:.1f} ueV/meV; 2 delta_eff / slope "
              f"{2 * de / slope * 1e3 if slope > 0 else np.nan:.2f} ueV; bulk there: Q {bb[0]:+.0f}, Delta_top {bb[1]:.1f} ueV, "
              f"xi_M {bb[2]:.0f} nm", flush=True)
    zrows = np.array(zrows, float)
    if len(zrows) >= 2:
        good = np.isfinite(zrows[:, 2]) & (zrows[:, 2] > 0)
        fit = np.polyfit(zrows[good, 0], np.log(zrows[good, 2]), 1) if good.sum() >= 2 else [np.nan, np.nan]
        print(f"    fit width ~ e^(L / xi): xi_fit {1 / fit[0]:.0f} nm vs xi_M {np.mean(zrows[:, 5]):.0f} nm (mean at the zeros)",
              flush=True)
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.7), layout='constrained')
        a = axs[0]
        Ls = zrows[:, 0] / 1000
        a.semilogy(Ls, zrows[:, 2], 'o-', color='#2a78d6', lw=1.8, ms=7, label='EP segment width (fine scan)')
        a.semilogy(Ls, 2 * zrows[:, 4] / zrows[:, 3] * 1e3, 'x--', color='#eb6834', lw=1.2, ms=7,
                   label=r'$2\delta_\mathrm{eff}/|d\varepsilon/dE_Z|$')
        xm = np.mean(zrows[:, 5])
        j = int(np.argmin(np.abs(zrows[:, 0] - 2000.0)))
        a.semilogy(Ls, zrows[j, 2] * np.exp((zrows[:, 0] - zrows[j, 0]) / xm), ':', color='#52514e', lw=1.2,
                   label=rf'$\propto e^{{L/\xi_M}}$, $\xi_M$ {xm:.0f} nm (no fit)')
        a.set_xlabel(r'$L$ ($\mu$m)')
        a.set_ylabel(r'segment width along $E_Z$ ($\mu$eV)')
        a.set_title(rf'(a) zero nearest $E_Z$ {EP5_EZREF:g} meV, $\phi$ {EP5_PHZ / np.pi:.3f}$\pi$', fontsize=11)
        a.legend(fontsize=8, frameon=False)
        a.grid(color='#e4e3df', lw=0.6, which='both')
        cols = ('#c6c0f2', '#8a7ee0', '#4a3bb8', '#1e1757')
        for (L, (C, z0, ezf, F)), col in zip(EP5Z.items(), cols):
            dz = (ezf - z0) * 1e3
            axs[1].plot(dz, F[:, 0], '.', color=col, ms=4, label=rf'$L$ {L / 1000:g} $\mu$m')
            axs[1].plot(dz, F[:, 1], '.', color=col, ms=4)
            axs[2].plot(dz, F[:, K], '.', color=col, ms=4)
            axs[2].plot(dz, F[:, K + 1], '.', color=col, ms=4)
        axs[1].set_xlabel(r'$E_Z - E_Z^0$ ($\mu$eV)')
        axs[1].set_ylabel(r'Re $E$ of the pair ($\mu$eV)')
        axs[1].set_title('(b) the pair through the zero: flat Re E = 0 between EPs', fontsize=11)
        axs[1].legend(fontsize=8, frameon=False)
        axs[2].set_xlabel(r'$E_Z - E_Z^0$ ($\mu$eV)')
        axs[2].set_ylabel(r'Im $E$ of the pair ($\mu$eV)')
        axs[2].set_title('(c) the split widths between the EPs', fontsize=11)
        for a in axs[1:]:
            a.grid(color='#e4e3df', lw=0.6)
        st.save_fig(fig, 'ep5_lengthEZ_mixed', dpi=150)
        plt.close(fig)
        print("figure: plots/ep5_lengthEZ_mixed.png", flush=True)


# ====================================================================================================================
# PART 'ep6'  (2026-11-18)
# ====================================================================================================================
# STUDY: does the open junction show the (1/2) ln 2 entropy plateau between EPs of Pino, Meir & Aguado (arXiv
# 2405.02387; one isolated Majorana, measurable through dS/dphi = -(hbar / 2e) dI/dT)?  Side study, judged NOT thesis
# relevant (2026-11-18); the planned exact supercurrent cells (Li & Trauzettel's CPR) were dropped before they finished.
#   ENTROPY from the full-model poles (tools.pole_entropy: Lloyd DOS = one Lorentzian per pole, exact for H_eff): a
#   pole -i G counts (1/2) ln 2 for G << kT and ~0 for G >> kT.  The [ep] pairs (tau_R 0.1) are added to by
#   SINGLE-PROBE lines (tau_R = 0, Pino's gamma_3 = 0).
# E_Z = EP_EZ, thesis junction (mixed SOC, mu_C 0.7, L 2 um, a 10).  Figure plots/ep6_entropy_mixed.png; caches
# scharf_absEP_*_R0 (single-probe poles).
EP6_ONE = (0.3, 0.5, 0.8)                                  # single probe: tau_L, tau_R = 0
EP6_KT = np.logspace(-2.0, np.log10(3.0), 61)              # ueV (0.12 - 35 mK); the bulk levels (>= ~20 ueV) are left
                                                           # out of S: s_F(20 / 3) ~ 0.01 per level at the top


def ep6_poles(tau_L, tau_R):
    """The pole lines of [ep] / [ep6] on EP_PH: (phi / pi, complex poles (121, EP_K), between-EPs mask)."""
    nm = 'ref' if tau_L == 0 and tau_R == 0 else f'tauL{tau_L:g}'
    z = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_{nm}_R{tau_R:g}.npz')
    R = z['values']
    return z['axis_phi'] / np.pi, R[:, :EP_K] + 1j * R[:, EP_K:2 * EP_K], ep_between(R)


def ep6_eps(btw, x):
    """EP phases (phi / pi) from a between-EPs mask: midpoints of its switches."""
    ch = np.flatnonzero(np.diff(btw.astype(int)))
    return (x[ch] + x[ch + 1]) / 2


def ep6_label(c):
    return 'probes detached' if c == (0.0, 0.0) else rf'$\tau_L$ {c[0]:g}, $\tau_R$ {c[1]:g}'


# %% [ep6] --- 1. SINGLE-PROBE POLE LINES (tau_R = 0), as [ep] cell 1 (plots/ep_mixed_tauL<t>_tauR0.png) ---
if run_part('ep6'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        _ref = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_ref_R0.npz')['values']
        for _tl in EP6_ONE:
            _R = st.myf.scan(partial(ep_point, tau_L=_tl, tau_R=0.0), {'phi': EP_PH}, pool=_pool, chunksize=1,
                             save=str(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{_tl:g}_R0.npz'))['values']
            ep_figure(_tl, _R, _ref, tau_R=0.0)
    finally:
        _pool.shutdown()


# %% [ep6] --- 2. ENTROPY S(phi, T) FROM THE POLES and the Maxwell-relation signal dI/dT = -(2 e k_B / hbar) dS/dphi ---
if run_part('ep6'):
    LN2 = np.log(2)
    cfg = [(0.0, 0.0), (0.8, 0.1)] + [(t, 0.0) for t in EP6_ONE]
    T_mK = EP6_KT / st.KB_UEV_PER_MK
    TC = (3.0, 10.0, 30.0)                                        # mK, the cuts
    CC = ('#2a78d6', '#1baf7a', '#eb6834')
    ENT = {}
    for c in cfg:
        x, P, btw = ep6_poles(*c)
        ENT[c] = (x, P, btw, st.pole_entropy(P[:, :2], EP6_KT) / LN2)
    print(f"\n  [ep6] entropy of the Majorana pair (units k_B ln 2), E_Z {EP_EZ:g}; T window where 0.45 <= S <= 0.55:",
          flush=True)
    for c in cfg:
        x, P, btw, S = ENT[c]
        for ph in (0.5, 0.7333, 1.0, 1.75):
            i = int(np.argmin(np.abs(x - ph)))
            w = T_mK[(S[i] >= 0.45) & (S[i] <= 0.55)]
            print(f"    {c}: phi {x[i]:.3f} pi ({'between EPs' if btw[i] else 'split'}), poles "
                  f"{P[i, 0].real:+.2f}{P[i, 0].imag:+.3f}i / {P[i, 1].real:+.2f}{P[i, 1].imag:+.3f}i ueV: S at "
                  + ", ".join(f"{Tc:g} mK {S[i, int(np.argmin(np.abs(T_mK - Tc)))]:.3f}" for Tc in TC)
                  + (f"; S in [0.45, 0.55] for {w.min():.2f}-{w.max():.2f} mK" if w.size else "; never near 1/2"),
                  flush=True)
    fig, axs = plt.subplots(3, len(cfg), figsize=(3.9 * len(cfg), 10.8), layout='constrained', sharex=True,
                            gridspec_kw=dict(height_ratios=(1.25, 1, 1)))
    for j, c in enumerate(cfg):
        x, P, btw, S = ENT[c]
        e = np.flatnonzero(np.diff(np.r_[0, btw.astype(int), 0]))
        a = axs[0, j]
        im = a.pcolormesh(x, T_mK, S.T, shading='nearest', cmap='viridis', vmin=0, vmax=1, rasterized=True)
        a.contour(x, T_mK, S.T, levels=[0.5], colors='w', linewidths=0.9, linestyles='--')
        for ph in ep6_eps(btw, x):
            a.axvline(ph, color='#eb6834', lw=1, ls=':')
        a.set_yscale('log')
        a.set_title(ep6_label(c), fontsize=11)
        for r in (1, 2):
            for i0, i1 in zip(e[::2], e[1::2]):
                axs[r, j].axvspan(x[i0] - 1 / 120, x[i1 - 1] + 1 / 120, color='#f5c9b3', lw=0, zorder=0)
            axs[r, j].grid(color='#e4e3df', lw=0.6)
        for Tc, col in zip(TC, CC):
            k = int(np.argmin(np.abs(T_mK - Tc)))
            axs[1, j].plot(x, S[:, k], color=col, lw=1.6, label=f'{T_mK[k]:.0f} mK')
            dS = np.gradient(S[:, k] * LN2, x * np.pi)
            axs[2, j].plot(x, -st.TWO_E_KB_HBAR_NA_MK * dS, color=col, lw=1.4, label=f'{T_mK[k]:.0f} mK')
        axs[1, j].axhline(0.5, color='#52514e', lw=0.8, ls='--')
        axs[1, j].set_ylim(0, 1.05)
        axs[2, j].set_xlabel(r'$\phi/\pi$')
    axs[0, 0].set_ylabel('T (mK)')
    axs[1, 0].set_ylabel(r'$S\,/\,k_B\ln 2$ (Majorana pair)')
    axs[2, 0].set_ylabel(r'$\partial I/\partial T = -(2ek_B/\hbar)\,\partial S/\partial\phi$ (nA/mK)')
    axs[1, 0].legend(fontsize=8.5, frameon=False, loc='upper left')
    fig.colorbar(im, ax=axs[0, :], shrink=0.9, pad=0.01, label=r'$S\,/\,k_B\ln 2$')
    fig.suptitle(rf'Entropy of the open junction from its poles (Lloyd DOS), $E_Z$ {EP_EZ:g} meV, mixed SOC, L 2 $\mu$m.'
                 '  Shaded / dotted: between EPs / EPs; dashed: S = (1/2) ln 2 (one isolated Majorana)', fontsize=11)
    st.save_fig(fig, 'ep6_entropy_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep6_entropy_mixed.png", flush=True)


# ====================================================================================================================
# PART 'ep7'  (2026-11-18)
# ====================================================================================================================
# STUDY: how do the EP segments show up in the THERMAL TRANSPORT between the probes?  At finite T the probes average the
# energy-resolved transmission over |E| ~ a few kT, i.e. over the region where the poles of the open junction sit.  All
# from the full model (tools.channel_point with separate probe bonds; poles from [ep] / [ep3] / cell 1 here):
#   Q1  kappa(E, phi) = (T_ee + T_he,cross + T_hh + T_eh,cross)(E) / 2: where is it maximal (E*), and does E* follow the
#       real parts of the pair poles -- is the heat transmission peaked at E = 0 throughout the EP segments?
#   Q2  kappa(T, phi) at 0 / 5 / 10 / 20 / 50 mK: T-robustness inside vs outside the segments.
#   Q3  the kappa ~ 1/2 plateau at 10-20 mK with EP-regime probes vs the symmetric bonds.
# Pairs (tau_L, tau_R): (0.5, 0.1) phi = pi between EPs, det r never < 0; (0.4, 0.6) weak asymmetry, segments inside
# det r < 0; (0.5, 0.5) symmetric control (EPs ~0.001 pi, unresolved); (0.8, 0.1) merged segments; (0.3, 0.1).
# E_Z = EP_EZ, thesis junction.  Energy grid 0 + 60 log points 0.01-70 ueV (16 kT at 50 mK = 69 ueV).  ~42 s per energy
# (121 phases, one thread) -> ~7.5 min per pair on EP_W workers.  Figures plots/ep7_*.png; caches scharf_absEP7_*.
EP7_CFG = ((0.5, 0.1), (0.4, 0.6), (0.5, 0.5), (0.8, 0.1), (0.3, 0.1))
EP7_E = np.r_[0.0, np.geomspace(0.01, 70.0, 60)]                    # ueV
EP7_T = (0.0, 5.0, 10.0, 20.0, 50.0)                                # mK
EP7_TCOL = ('#0b2a4a', '#1d4f8a', '#2a78d6', '#6fa8e6', '#b4d2f3')  # one hue, dark = cold


def ep7_channels(E, tau_L, tau_R):
    """The six channels at energy E (ueV) on EP_PH for the probe bonds (tau_L, tau_R): (6 * len(EP_PH),)."""
    q = st.replace(junction('mixed', 0.7), tc_barr_transport=float(tau_L))
    return st.channel_point(q, EP_EZ, float(E) * 1e-3, tuple(EP_PH), tau_R=float(tau_R))


def ep7_poles(tau_L, tau_R):
    """(phi / pi, the pair poles (n, 2) complex ueV, between-EPs mask) from the pole caches ([ep3] fine for (0.4, 0.6))."""
    if (tau_L, tau_R) == EP3_FINE:
        z = np.load(st.DATA / f'scharf_absEP3_fine_mixed_a10_EZ{EP_EZ:g}_tauL{tau_L:g}_R{tau_R:g}.npz')
    else:
        z = np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{tau_L:g}_R{tau_R:g}.npz')
    R = z['values']
    return z['axis_phi'] / np.pi, R[:, :2] + 1j * R[:, EP_K:EP_K + 2], ep_between(R)


def ep7_kappaE(V):
    """kappa(E, phi) (n_E, n_phi) from a channel scan V (n_E, 6 n_phi)."""
    C = V.reshape(len(V), -1, 6)
    return (C[..., 0] + C[..., 1] + C[..., 2] + C[..., 3]) / 2


def ep7_thermal(V, T_mK):
    """(kappa, G, local Andreev) at T (mK) for every phase of EP_PH: (n_phi, 3)."""
    return np.array([st.thermal_line(V[None], st.kT_of_mK(T_mK), EP7_E * 1e-3, phase=n)[0] for n in range(len(EP_PH))])


def ep7_figure(c, V):
    """kappa(E, phi) with the pair poles, kappa(T, phi), and the peak energy E* vs |Re| of the poles, for one pair."""
    x = EP_PH / np.pi
    K = ep7_kappaE(V)
    Es = EP7_E[np.argmax(K, axis=0)]                                 # E* per phase (0 = peaked at zero energy)
    TH = {T: ep7_thermal(V, T) for T in EP7_T}
    xp, P, btw = ep7_poles(*c)
    inside = np.interp(x, xp, btw.astype(float)) > 0.5
    eph = (xp[np.flatnonzero(np.diff(btw.astype(int)))] + xp[np.flatnonzero(np.diff(btw.astype(int))) + 1]) / 2
    e = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 11.0), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.3, 1, 0.9)))
    a = axs[0]
    m = EP7_E <= 25.0
    im = a.pcolormesh(x, EP7_E[m], K[m], shading='nearest', cmap='viridis', vmin=0, vmax=max(0.5, K[m].max()),
                      rasterized=True)
    a.plot(xp, np.abs(P[:, 0].real), '.', color='w', ms=2.2, label=r'$|\mathrm{Re}\,\lambda|$ of the pair poles')
    a.plot(xp, np.abs(P[:, 1].real), '.', color='w', ms=2.2)
    for ph in eph:
        for b in axs:
            b.axvline(ph, color='#eb6834', lw=1, ls=':')
    a.set_ylim(0, 25)
    a.set_ylabel(r'$E$ ($\mu$eV)')
    a.legend(fontsize=8.5, frameon=False, loc='upper right', labelcolor='w')
    fig.colorbar(im, ax=a, pad=0.01, label=r'$\kappa(E)/\kappa_0$')
    for b in axs[1:]:
        for i0, i1 in zip(e[::2], e[1::2]):
            b.axvspan(x[i0] - 1 / 120, x[i1 - 1] + 1 / 120, color='#f5c9b3', lw=0, zorder=0)
        b.grid(color='#e4e3df', lw=0.6)
    for T, col in zip(EP7_T, EP7_TCOL):
        axs[1].plot(x, TH[T][:, 0], color=col, lw=1.6, label=f'{T:g} mK')
    axs[1].axhline(0.5, color='#52514e', lw=0.8, ls='--')
    axs[1].set_ylabel(r'$\kappa(T)/\kappa_0$')
    axs[1].legend(fontsize=8.5, frameon=False, ncol=5, loc='upper center')
    a = axs[2]
    a.plot(x, Es, 'o', color='#2a78d6', ms=3.5, mew=0, label=r'$E^*$: maximum of $\kappa(E)$')
    a.plot(xp, np.abs(P[:, :2].real).max(axis=1), '-', color='#52514e', lw=1.0,
           label=r'$\max|\mathrm{Re}\,\lambda|$ of the pair')
    a.set_yscale('symlog', linthresh=0.05)
    a.set_ylim(-0.01, 30)
    a.set_ylabel(r'$E^*$ ($\mu$eV)')
    a.set_xlabel(r'$\phi/\pi$')
    a.legend(fontsize=8.5, frameon=False, loc='upper right')
    axs[0].set_xlim(0, 2)
    fig.suptitle(rf'Heat transport across the EP segments: $\tau_L$ {c[0]:g}, $\tau_R$ {c[1]:g}, $E_Z$ {EP_EZ:g} meV, '
                 r'mixed SOC, L 2 $\mu$m' '\n' 'dotted: EPs (full-model poles); shaded: between EPs', fontsize=11)
    name = f'ep7_kappaE_mixed_tauL{c[0]:g}_tauR{c[1]:g}'
    st.save_fig(fig, name, dpi=150)
    plt.close(fig)
    k0, k20 = TH[0.0][:, 0], TH[20.0][:, 0]
    ok = k0 > 0.05
    print(f"  ({c[0]:g}, {c[1]:g}): phases between EPs {inside.mean() * 2:.3f} pi; E* = 0 on {np.mean(Es[inside] == 0) * 100 if inside.any() else np.nan:.0f} % "
          f"of them and on {np.mean(Es[~inside] == 0) * 100:.0f} % outside; kappa(20 mK)/kappa(0) (where kappa(0) > 0.05) median "
          f"{np.median((k20 / k0)[ok & inside]) if (ok & inside).any() else np.nan:.3f} inside / "
          f"{np.median((k20 / k0)[ok & ~inside]) if (ok & ~inside).any() else np.nan:.3f} outside; max kappa at 0 / 10 / 20 mK "
          + " / ".join(f"{TH[T][:, 0].max():.3f}" for T in (0.0, 10.0, 20.0)), flush=True)
    print(f"figure: plots/{name}.png", flush=True)
    return K, Es, TH


# %% [ep7] --- 1. CHANNELS ON THE ENERGY GRID PER PROBE PAIR (pole line first where it is missing), one figure each ---
if run_part('ep7'):
    EP7 = {}
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        for c in EP7_CFG:
            if c != EP3_FINE:
                st.myf.scan(partial(ep_point, tau_L=c[0], tau_R=c[1]), {'phi': EP_PH}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP_mixed_a10_EZ{EP_EZ:g}_tauL{c[0]:g}_R{c[1]:g}.npz'))
            V = st.myf.scan(partial(ep7_channels, tau_L=c[0], tau_R=c[1]), {'E': EP7_E}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP7_ch_mixed_a10_EZ{EP_EZ:g}_tauL{c[0]:g}_R{c[1]:g}.npz'))['values']
            EP7[c] = (V,) + ep7_figure(c, V)
    finally:
        _pool.shutdown()


# ====================================================================================================================
# PART 'ep8'  (2026-11-18)
# ====================================================================================================================
# STUDY: how the probe couplings -- their DIFFERENCE and their overall STRENGTH -- set the length of the EP segments ALONG
# THE ZEEMAN FIELD, and how that relates to the broadening Gamma, to the kappa = 1/2 plateau along E_Z and to the
# transport quantities kappa is made of.  Full model only; phi fixed at EP8_PHI (the best-plateau phase of [plateau] /
# [ep4]), E_Z 0.80-1.10 meV in 5 ueV steps (the [ep4] cut).
#   EP segments: D = ((lambda_1 - lambda_2) / 2)^2 of the pair poles is real -- > 0 for a split pair (+-a - i G), < 0
#     between EPs (-i G_1, -i G_2) -- and, a symmetric function of the two poles, smooth across the EPs, where it
#     vanishes.  EPs = zeros of D (linear interpolation on the grid); a segment narrower than the grid shows as a local
#     minimum of D whose parabola through the three points dips below 0 (EPs at its roots).
#   'Unresolved' range: max |Re lambda| <= Gbar = -Im(lambda_1 + lambda_2) / 2 (the splitting hidden by the width).
#   Gamma_1(tau, E_Z) = -Im(lambda_1 + lambda_2) with ONE probe of bond tau (single-probe rows; the junction is mirror
#     symmetric, so it serves either side).  SigmaGamma / DeltaGamma of a pair = sum / difference of its two Gamma_1 at
#     the lossless zero crossings E_c of eps(E_Z) (probes detached: pair (0, 0)).
#   Transport at E = 0, T = 0: kappa, G, local Andreev, det r_L -> T_ee = (kappa + G)/2, T_he = (kappa - G)/2;
#     plateau = plat_mask (the [plateau] criterion), windows along E_Z in ueV.
# One task per field computes every pair (the ribbon set-up, ~40 s, is shared: ~6-8 s poles + ~2 s transport per pair).
# Figures plots/ep8_coupling_mixed.png, ep8_lines_mixed.png; cache scharf_absEP8_field_*.
EP8_PHI = 1.7417 * np.pi
EP8_EZ = EP4_EZ
EP8_DE = 5.0                                                     # ueV, the field step
EP8_T = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
EP8_PAIRS = ([(0.0, 0.0)] + [(t, 0.0) for t in EP8_T]
             + [(tl, tr) for tl in EP8_T for tr in EP8_T if tr <= tl])
EP8_NV = 2 * EP_K + 1 + 5                                        # ep_point values + transport row
EP8_SHOW = ((0.5, 0.5), (0.6, 0.4), (0.5, 0.1), (0.8, 0.1))


def ep8_field(E_Z):
    """At one field: ep_point (k = 16, the broad pole of strong probes is kept) and the E = 0 transport row
    (kappa, G, Andreev, det r_L, n_bad; NaN for a single probe) for every pair of EP8_PAIRS, flattened."""
    out = []
    for tl, tr in EP8_PAIRS:
        p = ep_point(EP8_PHI, tl, tr, E_Z=float(E_Z), k=16)
        t = (st.transport_phis(junction('mixed', 0.7, tl), float(E_Z), [EP8_PHI], tau_R=tr)[0] if tr > 0
             else np.full(5, np.nan))
        out.append(np.r_[p, t])
    return np.array(out).ravel()


def ep8_segments(E, D):
    """Segments with D < 0 along the open grid E: sorted (start, end, truncated) in the units of E.  Edges at the linear
    zeros of D; a local minimum of D > 0 whose parabola through the three points dips below 0 adds the segment between
    its roots."""
    n, segs, i = len(E), [], 0
    neg = D < 0
    while i < n:
        if neg[i]:
            j = i
            while j + 1 < n and neg[j + 1]:
                j += 1
            s = E[i] if i == 0 else E[i - 1] - D[i - 1] * (E[i] - E[i - 1]) / (D[i] - D[i - 1])
            e = E[j] if j == n - 1 else E[j] - D[j] * (E[j + 1] - E[j]) / (D[j + 1] - D[j])
            segs.append((s, e, i == 0 or j == n - 1))
            i = j + 1
        else:
            i += 1
    for i in range(1, n - 1):
        if D[i] > 0 and D[i] <= D[i - 1] and D[i] <= D[i + 1]:
            a, b, c = np.polyfit(E[i - 1:i + 2] - E[i], D[i - 1:i + 2], 2)
            disc = b * b - 4 * a * c
            if a > 0 and disc > 0:
                r = np.sort((-b + np.array([-1.0, 1.0]) * np.sqrt(disc)) / (2 * a)) + E[i]
                if r[0] >= E[i - 1] and r[1] <= E[i + 1]:
                    segs.append((r[0], r[1], False))
    return sorted(segs)


def ep8_crossings(E, s, tol=1.0):
    """Zero crossings of the lossless eps(E_Z) from |eps| = s on the grid: local minima below tol (ueV), placed by a
    linear sign-flipping interpolation towards the smaller neighbour."""
    out = []
    for i in range(1, len(E) - 1):
        if s[i] <= s[i - 1] and s[i] <= s[i + 1] and s[i] < tol:
            j = i + 1 if s[i + 1] < s[i - 1] else i - 1
            out.append(E[i] + (E[j] - E[i]) * s[i] / (s[i] + s[j]))
    return np.array(out)


def ep8_seg_near(segs, ec, reach=15.0):
    """The segment containing E_c, else the one whose centre is nearest within `reach` (ueV): (length, segment) or (0,
    None) -- the probes shift the level, so a segment need not contain the lossless crossing."""
    best = None
    for sg in segs:
        if sg[0] <= ec <= sg[1]:
            return sg[1] - sg[0], sg
        d = abs((sg[0] + sg[1]) / 2 - ec)
        if d <= reach and (best is None or d < best[0]):
            best = (d, sg)
    return (best[1][1] - best[1][0], best[1]) if best else (0.0, None)


# %% [ep8] --- 1. POLES AND E = 0 TRANSPORT FOR EVERY PAIR, ONE TASK PER FIELD ---
if run_part('ep8'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP8_F = st.myf.scan(ep8_field, {'E_Z': EP8_EZ}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
    finally:
        _pool.shutdown()
    EP8_F = EP8_F.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
    print("ep8: all fields done", flush=True)


# %% [ep8] --- 2. EP LENGTH ALONG E_Z vs THE COUPLINGS, Gamma, THE kappa PLATEAU AND THE kappa INGREDIENTS ---
if run_part('ep8'):
    E = EP8_EZ * 1e3                                                 # ueV
    IDX = {c: n for n, c in enumerate(EP8_PAIRS)}
    lam = lambda c: (EP8_F[:, IDX[c], 0] + 1j * EP8_F[:, IDX[c], EP_K], EP8_F[:, IDX[c], 1] + 1j * EP8_F[:, IDX[c], EP_K + 1])
    TR = lambda c: EP8_F[:, IDX[c], 2 * EP_K + 1:]
    l1, _ = lam((0.0, 0.0))
    EC = ep8_crossings(E, np.abs(l1.real))
    G1 = {0.0: np.zeros(len(E))}
    for t in EP8_T:
        a_, b_ = lam((t, 0.0))
        G1[t] = -(a_.imag + b_.imag)
    print(f"\n  [ep8] phi {EP8_PHI / np.pi:.4f} pi, E_Z {E[0]:.0f}-{E[-1]:.0f} ueV: lossless zero crossings at E_Z = "
          + ", ".join(f"{v:.1f}" for v in EC) + " ueV", flush=True)
    print("    Gamma_1(tau) at the crossings (ueV): " + "; ".join(
        f"tau {t:g}: " + "/".join(f"{np.interp(ec, E, G1[t]):.2f}" for ec in EC) for t in EP8_T), flush=True)
    SEG, ROW = {}, {}
    for c in EP8_PAIRS[1:]:
        a_, b_ = lam(c)
        D = (((a_ - b_) / 2) ** 2).real
        segs = ep8_segments(E, D)
        Gbar = -(a_.imag + b_.imag) / 2
        un = ep8_segments(E, np.abs(np.c_[a_.real, b_.real]).max(axis=1) - Gbar - 1e-9)
        SEG[c] = (segs, un, D)
        r = dict(L_EP=sum(e - s for s, e, _ in segs), n_seg=len(segs), trunc=any(t for *_, t in segs),
                 L_un=sum(e - s for s, e, _ in un))
        r['L_c'] = [ep8_seg_near(segs, ec)[0] for ec in EC]
        r['SG'] = [np.interp(ec, E, G1[c[0]]) + np.interp(ec, E, G1[c[1]]) for ec in EC]
        r['DG'] = [np.interp(ec, E, G1[c[0]]) - np.interp(ec, E, G1[c[1]]) for ec in EC]
        r['Gp'] = [np.interp(ec, E, 2 * Gbar) for ec in EC]
        if c[1] > 0:
            T = TR(c)
            k, G = T[:, 0], T[:, 1]
            m = plat_mask(T)
            r['plat'], r['plat_all'] = longest_run(m) * EP8_DE, m.sum() * EP8_DE
            r['k_c'] = [np.interp(ec, E, k) for ec in EC]
            r['Tee_c'] = [np.interp(ec, E, (k + G) / 2) for ec in EC]
            r['The_c'] = [np.interp(ec, E, (k - G) / 2) for ec in EC]
            ins = np.zeros(len(E), bool)
            for s, e, _ in segs:
                ins |= (E >= s) & (E <= e)
            r['plat_in'] = (m & ins).sum() * EP8_DE
            r['k_in'] = float(k[ins].mean()) if ins.any() else np.nan
            r['k_max'] = float(np.nanmax(k))
        ROW[c] = r
    print("    tau_L tau_R | EP length along E_Z (ueV): total [n seg] / at each crossing | unresolved | SigmaGamma, "
          "DeltaGamma at the crossings | plateau: longest window / all / inside EP segs | kappa at crossings", flush=True)
    for c in EP8_PAIRS[1:]:
        r = ROW[c]
        line = (f"    {c[0]:4.1f}  {c[1]:4.1f} | {r['L_EP']:6.1f} [{r['n_seg']}{'*' if r['trunc'] else ''}] / "
                + "/".join(f"{v:.1f}" for v in r['L_c']) + f" | {r['L_un']:6.1f} | "
                + "/".join(f"{v:.2f}" for v in r['SG']) + ", " + "/".join(f"{v:+.2f}" for v in r['DG']))
        if c[1] > 0:
            line += (f" | {r['plat']:4.0f} / {r['plat_all']:4.0f} / {r['plat_in']:4.0f} | "
                     + "/".join(f"{v:.3f}" for v in r['k_c']))
        print(line, flush=True)
    print("    (* = a segment reaches the window edge)", flush=True)
    # length law: L_EP x |d eps/dE_Z| / |DeltaGamma| at each crossing (untruncated segments > 5 ueV, tau_L != tau_R);
    # the slope from +-2 grid points around E_c on either side, averaged
    for ec in EC:
        i, s0 = int(np.searchsorted(E, ec)), np.abs(l1.real)
        sl = 0.5 * (abs(s0[i + 2] - s0[i]) / (E[i + 2] - E[i]) + abs(s0[i - 1] - s0[i - 3]) / (E[i - 1] - E[i - 3]))
        q, sh = [], []
        for c in EP8_PAIRS[8:]:
            L, sg = ep8_seg_near(SEG[c][0], ec)
            if c[0] != c[1] and sg is not None and not sg[2] and L > 5:
                q.append(L * sl / abs(np.interp(ec, E, G1[c[0]]) - np.interp(ec, E, G1[c[1]])))
                sh.append((sg[0] + sg[1]) / 2 - ec)
        print(f"    length law at E_c {ec:.1f} (|d eps/dE_Z| = {sl:.3f}): L_EP |d eps/dE_Z| / |DeltaGamma| = "
              f"{np.mean(q):.2f} +- {np.std(q):.2f} (n {len(q)}); segment centre shift {min(sh):+.1f} .. {max(sh):+.1f} ueV",
              flush=True)
    # longest kappa-only windows |kappa - 1/2| < tol, and the two pole widths / splitting in the tol 0.02 window
    print("    kappa-only windows (ueV) for tol 0.01/0.02/0.03, and the poles in the tol-0.02 window:", flush=True)
    for c in sorted(EP8_PAIRS[8:], key=lambda c: -longest_run(np.abs(TR(c)[:, 0] - 0.5) < 0.02))[:8]:
        k = TR(c)[:, 0]
        a_, b_ = lam(c)
        m = np.abs(k - 0.5) < 0.02
        n = longest_run(m)
        j = next(j for j in range(len(m) - n + 1) if m[j:j + n].all())
        w = np.sort(np.c_[-a_.imag, -b_.imag][j:j + n], axis=1)
        re = np.abs(np.c_[a_.real, b_.real][j:j + n]).max(axis=1)
        print(f"     {c}: " + "/".join(f"{longest_run(np.abs(k - 0.5) < t) * EP8_DE:.0f}" for t in (0.01, 0.02, 0.03))
              + f" | E_Z {E[j]:.0f}-{E[j + n - 1]:.0f}: widths {w[:, 0].min():.1f}-{w[:, 0].max():.1f} and "
              f"{w[:, 1].min():.1f}-{w[:, 1].max():.1f}, max|Re lambda| {re.min():.1f}-{re.max():.1f} ueV", flush=True)
    vals = (0.0,) + EP8_T

    def square(key):
        M = np.full((len(vals), len(vals)), np.nan)
        for c, r in ROW.items():
            if key in r:
                i, j = vals.index(c[0]), vals.index(c[1])
                M[i, j] = M[j, i] = r[key]
        return M

    fig, axs = plt.subplots(2, 3, figsize=(17.5, 10.5), layout='constrained')
    a = axs[0, 0]
    for n, ec in enumerate(EC):
        a.semilogy(EP8_T, [np.interp(ec, E, G1[t]) for t in EP8_T], 'o-', ms=5, lw=1.4,
                   color=('#2a78d6', '#1baf7a', '#eb6834', '#9b59d0', '#52514e')[n % 5], label=f'crossing at {ec:.0f} ueV')
    a.set_xlabel(r'probe bond $\tau$')
    a.set_ylabel(r'$\Gamma_1(\tau)$: pair width with ONE probe ($\mu$eV)')
    a.set_title('(a) broadening by one probe, at the lossless crossings', fontsize=11)
    a.legend(fontsize=8.5, frameon=False)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    for a, key, title in ((axs[0, 1], 'L_EP', r'(b) EP length along $E_Z$ ($\mu$eV), strict'),
                          (axs[0, 2], 'plat', r'(c) longest $\kappa = 1/2$ plateau window along $E_Z$ ($\mu$eV)')):
        M = square(key)
        im = a.imshow(M, origin='lower', cmap='viridis', vmin=0, vmax=max(np.nanmax(M), 1.0))
        for i in range(len(vals)):
            for j in range(len(vals)):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=7,
                           color='w' if M[i, j] < 0.6 * np.nanmax(M) else 'k')
        a.set_xticks(range(len(vals)), [f'{v:g}' for v in vals])
        a.set_yticks(range(len(vals)), [f'{v:g}' for v in vals])
        a.set_xlabel(r'$\tau_R$')
        a.set_ylabel(r'$\tau_L$')
        a.set_title(title, fontsize=11)
        fig.colorbar(im, ax=a, shrink=0.85)
    a = axs[1, 0]
    xs, ys, cs = [], [], []
    for c, r in ROW.items():
        for n in range(len(EC)):
            if r['L_c'][n] > 0 and abs(r['DG'][n]) > 0:
                xs.append(abs(r['DG'][n]))
                ys.append(r['L_c'][n])
                cs.append(r['SG'][n])
    sc = a.scatter(xs, ys, c=cs, s=34, cmap='viridis', norm=mpl.colors.LogNorm())
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'$|\Delta\Gamma| = |\Gamma_1(\tau_L) - \Gamma_1(\tau_R)|$ at the crossing ($\mu$eV)')
    a.set_ylabel(r'EP segment at the crossing ($\mu$eV of $E_Z$)')
    a.set_title(r'(d) segment length vs the width difference (colour: $\Sigma\Gamma$)', fontsize=11)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    fig.colorbar(sc, ax=a, label=r'$\Sigma\Gamma$ ($\mu$eV)')
    a = axs[1, 1]
    for key, col, lab, ms in (('k_c', '#2a78d6', r'$\kappa$', 34), ('Tee_c', '#1baf7a', r'$T_{ee}$', 20),
                              ('The_c', '#eb6834', r'$T_{he}$', 20)):
        xs, ys = [], []
        for c, r in ROW.items():
            if key in r:
                for n in range(len(EC)):
                    xs.append(abs(r['DG'][n]) / r['SG'][n])
                    ys.append(r[key][n])
        a.scatter(xs, ys, color=col, s=ms, label=lab + ' at the crossings')
    a.set_xlabel(r'$|\Delta\Gamma|\,/\,\Sigma\Gamma$ at the crossing')
    a.set_ylabel(r'$E = 0$, $T = 0$')
    a.set_title(r'(e) the $\kappa$ ingredients at the crossings', fontsize=11)
    a.legend(fontsize=8.5, frameon=False)
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 2]
    cc = [c for c in ROW if 'plat' in ROW[c]]
    sc = a.scatter([np.mean(ROW[c]['SG']) for c in cc], [ROW[c]['plat'] for c in cc],
                   c=[np.mean(np.abs(ROW[c]['DG']) / np.array(ROW[c]['SG'])) for c in cc], s=40, cmap='viridis',
                   vmin=0, vmax=1)
    for c in cc:
        a.annotate(f'{c[0]:g}/{c[1]:g}', (np.mean(ROW[c]['SG']), ROW[c]['plat']), fontsize=6.5,
                   xytext=(3, 2), textcoords='offset points', color='#52514e')
    a.set_xscale('log')
    a.set_xlabel(r'$\Sigma\Gamma$, mean over the crossings ($\mu$eV)')
    a.set_ylabel(r'longest plateau window along $E_Z$ ($\mu$eV)')
    a.set_title(r'(f) plateau vs overall strength (colour: $|\Delta\Gamma|/\Sigma\Gamma$)', fontsize=11)
    a.grid(color='#e4e3df', lw=0.6, which='both')
    fig.colorbar(sc, ax=a, label=r'$|\Delta\Gamma|/\Sigma\Gamma$')
    fig.suptitle(rf'EP segments and the $\kappa$ plateau ALONG $E_Z$ vs the probe couplings, $\phi$ = '
                 rf'{EP8_PHI / np.pi:.4f} $\pi$, mixed SOC, L 2 $\mu$m (full model; $\tau_L\geq\tau_R$ computed, mirrored)',
                 fontsize=12)
    st.save_fig(fig, 'ep8_coupling_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep8_coupling_mixed.png", flush=True)
    fig, axs = plt.subplots(3, len(EP8_SHOW), figsize=(4.6 * len(EP8_SHOW), 10.0), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.1, 1.1, 0.8)))
    for j, c in enumerate(EP8_SHOW):
        a_, b_ = lam(c)
        segs, un, D = SEG[c]
        T = TR(c)
        m = plat_mask(T)
        for a in axs[:, j]:
            for s, e, _ in segs:
                a.axvspan(s, e, color='#f5c9b3', lw=0, zorder=0)
            for i in np.flatnonzero(m):
                a.axvspan(E[i] - EP8_DE / 2, E[i] + EP8_DE / 2, ymin=0.0, ymax=0.04, color='#1baf7a', lw=0)
            for ec in EC:
                a.axvline(ec, color='#52514e', lw=0.8, ls=':')
            a.grid(color='#e4e3df', lw=0.6)
        a = axs[0, j]
        a.plot(E, T[:, 0], color='#2a78d6', lw=1.6, label=r'$\kappa$')
        a.plot(E, T[:, 1], color='#eb6834', lw=1.2, label=r'$G$')
        a.plot(E, (T[:, 0] + T[:, 1]) / 2, color='#1baf7a', lw=1.0, ls='--', label=r'$T_{ee}$')
        a.plot(E, (T[:, 0] - T[:, 1]) / 2, color='#9b59d0', lw=1.0, ls='--', label=r'$T_{he}$')
        a.axhline(0.5, color='#52514e', lw=0.8, ls='--')
        a.set_title(rf'$\tau_L$ {c[0]:g}, $\tau_R$ {c[1]:g}', fontsize=11)
        a = axs[1, j]
        a.plot(E, a_.real, '.', color='#2a78d6', ms=3)
        a.plot(E, b_.real, '.', color='#2a78d6', ms=3, label=r'Re $\lambda$')
        a.plot(E, a_.imag, '.', color='#eb6834', ms=3)
        a.plot(E, b_.imag, '.', color='#eb6834', ms=3, label=r'Im $\lambda$')
        a.plot(E, np.abs(l1.real), color='#52514e', lw=1.0, ls='--', label=r'$|\varepsilon|$, probes detached')
        a = axs[2, j]
        a.plot(E, D, color='#2a78d6', lw=1.4)
        a.axhline(0, color='#52514e', lw=0.8)
        a.set_yscale('symlog', linthresh=0.05)
        a.set_xlabel(r'$E_Z$ ($\mu$eV)')
    axs[0, 0].set_ylabel(r'$E = 0$, $T = 0$')
    axs[1, 0].set_ylabel(r'pair poles ($\mu$eV)')
    axs[2, 0].set_ylabel(r'$D = ((\lambda_1-\lambda_2)/2)^2$ ($\mu$eV$^2$)')
    axs[0, 0].legend(fontsize=8, frameon=False, ncol=2)
    axs[1, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(rf'Along $E_Z$ at $\phi$ = {EP8_PHI / np.pi:.4f} $\pi$: orange = between EPs, green ticks = $\kappa = 1/2$ '
                 'plateau points, dotted = lossless zero crossings', fontsize=11)
    st.save_fig(fig, 'ep8_lines_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep8_lines_mixed.png", flush=True)


# ====================================================================================================================
# PART 'ep9'  (2026-11-19)
# ====================================================================================================================
# STUDY: is the kappa = 1/2 plateau ALONG E_Z robust in temperature, and does that depend on the route [ep8] found it by?
#   (A) balanced probes: the pair split, both poles ~6-14 ueV broad, the plateau OUTSIDE the EP segments;
#   (B) strong + weak bond: the plateau INSIDE one segment, carried by a ~2 ueV pole pinned at E = 0 (by [ep7] the
#       T-fragile kind).
# Full model, phi = EP8_PHI, E_Z = EP8_EZ (0.80-1.10 meV, 5 ueV), the six channels on the [ep7] energy grid (0 + 60 log
# points 0.01-70 ueV) -> kappa(E, E_Z) and kappa, G at 0 / 5 / 10 / 20 / 50 mK (tools.thermal; 16 kT at 50 mK = 69 ueV).
# Pairs: (0.5, 0.4), (0.5, 0.5) route A; (0.8, 0.2) route B; (0.6, 0.3) the threshold-sensitive one.
# One task per ENERGY: tools.channel_fields decimates the SC ribbons once (they see neither E_Z nor the bonds; checked
# on every solve, bit-identical to channel_point), then ~2.1 s per field and pair -> 244 solves ~ 9 min per task, 61
# tasks on EP_W workers ~1.5 h.  Poles and EP segments from the [ep8] cache; check: kappa, G at T = 0 = the [ep8] rows.
# Figures plots/ep9_kappaT_mixed.png, ep9_summary_mixed.png; cache scharf_absEP9_ch_*.
EP9_PAIRS = ((0.5, 0.4), (0.5, 0.5), (0.8, 0.2), (0.6, 0.3))
EP9_TOL = (0.02, 0.03)
EP9_PCOL = ('#2a78d6', '#1baf7a', '#eb6834', '#9b59d0')


def ep9_energy(E):
    """The six channels at energy E (ueV), phi = EP8_PHI, for every field of EP8_EZ and pair of EP9_PAIRS, flattened."""
    return st.channel_fields(junction('mixed', 0.7), float(E) * 1e-3, EP8_EZ, EP9_PAIRS, EP8_PHI).ravel()


def ep9_window(k, G, tol):
    """Longest run along E_Z with |kappa - 1/2| < tol: (length in ueV, max |G| / kappa in it, its slice or None)."""
    m = np.abs(k - 0.5) < tol
    n = longest_run(m)
    if n == 0:
        return 0.0, np.nan, None
    j = next(j for j in range(len(m) - n + 1) if m[j:j + n].all())
    return n * EP8_DE, float(np.max(np.abs(G[j:j + n]) / k[j:j + n])), slice(j, j + n)


# %% [ep9] --- 1. CHANNELS ON THE ENERGY GRID, ONE TASK PER ENERGY ---
if run_part('ep9'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EP9_V = st.myf.scan(ep9_energy, {'E': EP7_E}, pool=_pool, chunksize=1,
                            save=str(st.DATA / f'scharf_absEP9_ch_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
    finally:
        _pool.shutdown()
    EP9_V = EP9_V.reshape(len(EP7_E), len(EP8_EZ), len(EP9_PAIRS), 6).transpose(1, 0, 2, 3)   # (E_Z, E, pair, 6)
    print("ep9: all energies done", flush=True)


# %% [ep9] --- 2. kappa(T) ALONG E_Z: plateau windows vs T, inside / outside the EP segments, figures ---
if run_part('ep9'):
    E = EP8_EZ * 1e3                                                 # ueV
    F8 = np.load(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz')['values']
    F8 = F8.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
    IDX = {c: n for n, c in enumerate(EP8_PAIRS)}
    EC = ep8_crossings(E, np.abs(F8[:, IDX[(0.0, 0.0)], 0]))
    RES = {}
    for p, c in enumerate(EP9_PAIRS):
        a_ = F8[:, IDX[c], 0] + 1j * F8[:, IDX[c], EP_K]
        b_ = F8[:, IDX[c], 1] + 1j * F8[:, IDX[c], EP_K + 1]
        segs = ep8_segments(E, (((a_ - b_) / 2) ** 2).real)
        ins = np.zeros(len(E), bool)
        for s, e, _ in segs:
            ins |= (E >= s) & (E <= e)
        V = EP9_V[:, :, p, :]
        TH = {T: st.thermal_line(V, st.kT_of_mK(T), EP7_E * 1e-3) for T in EP7_T}
        T8 = F8[:, IDX[c], 2 * EP_K + 1:]
        K = (V[..., 0] + V[..., 1] + V[..., 2] + V[..., 3]) / 2          # kappa(E_Z, E)
        RES[c] = dict(a=a_, b=b_, segs=segs, ins=ins, TH=TH, K=K, Es=EP7_E[np.argmax(K, axis=1)],
                      chk=(np.abs(TH[0.0][:, 0] - T8[:, 0]).max(), np.abs(TH[0.0][:, 1] - T8[:, 1]).max()))
    print(f"\n  [ep9] phi {EP8_PHI / np.pi:.4f} pi, E_Z {E[0]:.0f}-{E[-1]:.0f} ueV; lossless crossings at "
          + ", ".join(f"{v:.1f}" for v in EC) + " ueV", flush=True)
    for c, r in RES.items():
        k0 = r['TH'][0.0][:, 0]
        w0 = ep9_window(k0, r['TH'][0.0][:, 1], EP9_TOL[0])
        wd = np.sort(np.c_[-r['a'].imag, -r['b'].imag], axis=1)
        print(f"    ({c[0]:g}, {c[1]:g}): check vs [ep8] at T = 0: max |d kappa| {r['chk'][0]:.1e}, |d G| {r['chk'][1]:.1e}; "
              f"EP segments: " + (", ".join(f"{s:.0f}-{e:.0f}" for s, e, _ in r['segs']) or "none") + " ueV; "
              f"E* = 0 on {np.mean(r['Es'][r['ins']] == 0) * 100 if r['ins'].any() else np.nan:.0f} % of the fields "
              f"inside / {np.mean(r['Es'][~r['ins']] == 0) * 100 if (~r['ins']).any() else np.nan:.0f} % outside",
              flush=True)
        if w0[2] is not None:
            s = w0[2]
            print(f"      T = 0 window (tol 0.02) E_Z {E[s][0]:.0f}-{E[s][-1]:.0f} ueV: {r['ins'][s].mean() * 100:.0f} % inside "
                  f"segments; pole widths {wd[s, 0].min():.1f}-{wd[s, 0].max():.1f} / {wd[s, 1].min():.1f}-"
                  f"{wd[s, 1].max():.1f} ueV (k_B T = 0.43 / 0.86 / 1.72 / 4.31 ueV at 5 / 10 / 20 / 50 mK)", flush=True)
        for T in EP7_T:
            k, G = r['TH'][T][:, 0], r['TH'][T][:, 1]
            w = [ep9_window(k, G, tol) for tol in EP9_TOL]
            ok = k0 > 0.05
            med = lambda mk: np.median((k / k0)[ok & mk]) if (ok & mk).any() else np.nan
            on = f"{k[w0[2]].mean():.3f} / {k[w0[2]].min():.3f}" if w0[2] is not None else "--"
            print(f"      {T:4.0f} mK: kappa max {np.nanmax(k):.3f}, max |G| {np.nanmax(np.abs(G)):.3f}; window <0.02: "
                  f"{w[0][0]:3.0f} ueV (max |G|/kappa {w[0][1]:.3f}), <0.03: {w[1][0]:3.0f} ueV; kappa on the T = 0 window "
                  f"mean / min {on}; kappa(T)/kappa(0) median in / out of segments {med(r['ins']):.3f} / {med(~r['ins']):.3f}",
                  flush=True)
    fig, axs = plt.subplots(3, len(EP9_PAIRS), figsize=(4.7 * len(EP9_PAIRS), 10.5), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.2, 1.1, 0.7)))
    me = EP7_E <= 25.0
    for j, c in enumerate(EP9_PAIRS):
        r = RES[c]
        a = axs[0, j]
        im = a.pcolormesh(E, EP7_E[me], r['K'][:, me].T, shading='nearest', cmap='viridis', vmin=0,
                          vmax=max(0.5, r['K'][:, me].max()), rasterized=True)
        a.plot(E, np.abs(r['a'].real), '.', color='w', ms=2.5)
        a.plot(E, np.abs(r['b'].real), '.', color='w', ms=2.5, label=r'$|\mathrm{Re}\,\lambda|$ of the pair poles')
        for s, e, _ in r['segs']:
            for x in (s, e):
                a.axvline(x, color='#eb6834', lw=1.0, ls=':')
        a.set_ylim(0, 25)
        a.set_title(rf'$\tau_L$ {c[0]:g}, $\tau_R$ {c[1]:g}', fontsize=11)
        fig.colorbar(im, ax=a, pad=0.01, label=r'$\kappa(E)/\kappa_0$' if j == len(EP9_PAIRS) - 1 else None)
        for b in axs[1:, j]:
            for s, e, _ in r['segs']:
                b.axvspan(s, e, color='#f5c9b3', lw=0, zorder=0)
            for ec in EC:
                b.axvline(ec, color='#52514e', lw=0.8, ls=':')
            b.grid(color='#e4e3df', lw=0.6)
        for T, col in zip(EP7_T, EP7_TCOL):
            axs[1, j].plot(E, r['TH'][T][:, 0], color=col, lw=1.6, label=f'{T:g} mK')
            axs[2, j].plot(E, r['TH'][T][:, 1], color=col, lw=1.3)
        axs[1, j].axhline(0.5, color='#52514e', lw=0.8, ls='--')
        axs[1, j].set_ylim(0, 0.56)
        axs[2, j].axhline(0, color='#52514e', lw=0.8)
        axs[2, j].set_xlabel(r'$E_Z$ ($\mu$eV)')
    axs[0, 0].set_ylabel(r'$E$ ($\mu$eV)')
    axs[1, 0].set_ylabel(r'$\kappa(T)/\kappa_0$')
    axs[2, 0].set_ylabel(r'$G(T)/G_0$')
    axs[0, 0].legend(fontsize=8, frameon=False, loc='upper right', labelcolor='w')
    axs[1, 0].legend(fontsize=8, frameon=False, ncol=2, loc='lower center')
    fig.suptitle(rf'Heat transport along $E_Z$ at $\phi$ = {EP8_PHI / np.pi:.4f} $\pi$ vs temperature, mixed SOC, L 2 $\mu$m '
                 '(full model)\nshaded / orange dotted: between EPs (full-model poles); grey dotted: lossless zero '
                 'crossings', fontsize=11)
    st.save_fig(fig, 'ep9_kappaT_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep9_kappaT_mixed.png", flush=True)
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.6), layout='constrained')
    for c, col in zip(EP9_PAIRS, EP9_PCOL):
        r = RES[c]
        for a, tol in zip(axs[:2], EP9_TOL):
            a.plot(EP7_T, [ep9_window(r['TH'][T][:, 0], r['TH'][T][:, 1], tol)[0] for T in EP7_T], 'o-', color=col,
                   ms=6, lw=1.6, label=f'({c[0]:g}, {c[1]:g})')
        axs[2].plot(EP7_T, [np.nanmax(r['TH'][T][:, 0]) for T in EP7_T], 'o-', color=col, ms=6, lw=1.6,
                    label=f'({c[0]:g}, {c[1]:g})')
    for a, tol in zip(axs[:2], EP9_TOL):
        a.set_ylabel(rf'longest $|\kappa - 1/2| < {tol:g}$ window along $E_Z$ ($\mu$eV)')
    axs[2].axhline(0.5, color='#52514e', lw=0.8, ls='--')
    axs[2].set_ylabel(r'max $\kappa(T)/\kappa_0$ over $E_Z$')
    for a, t in zip(axs, ('(a)', '(b)', '(c)')):
        a.set_xlabel('T (mK)')
        a.set_title(t, fontsize=11, loc='left')
        a.grid(color='#e4e3df', lw=0.6)
    axs[0].legend(title=r'($\tau_L$, $\tau_R$)', fontsize=9, frameon=False)
    fig.suptitle(rf'The $\kappa = 1/2$ plateau along $E_Z$ (0.80-1.10 meV) vs temperature, $\phi$ = {EP8_PHI / np.pi:.4f} '
                 r'$\pi$, mixed SOC, L 2 $\mu$m', fontsize=11)
    st.save_fig(fig, 'ep9_summary_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/ep9_summary_mixed.png", flush=True)


# =====================================================================================================================
# PART [nh] (2026-11-19) -- PLOTS ONLY, from existing caches (no new transport).  QUESTION: does the probe asymmetry,
# i.e. the non-Hermitian EP physics, protect the kappa plateau?  Sources:
#   [ep7] kappa(E, phi) at E_Z 0.93 for four bond pairs (scharf_absEP7_ch_*, E = 0 + 0.01-70 ueV log);
#   [ep8] kappa(E_Z) at E = 0, phi 1.7417 pi, 36 bond pairs, with the pair poles (scharf_absEP8_field_*);
#   study_mixed [deck] 3: kappa(E, E_Z) for EQUAL bonds 0.5 / 0.5 at 1.7417 pi (scharf_absDeck_T_mixed_tau0.5_L2_EZ0.85-1.05,
#   E 0-35 ueV, E_Z step 2.5 ueV).
# NOT cached: kappa(E_Z, phi) and finite T along E_Z for unequal bonds ([ep9] stopped at 12 / 61 energies, all < 0.05 ueV);
# kappa vs mu_C only for equal bonds (plots/tuning_kappa_tau0.5.png, mu_C 0.3 / 0.7 / 1.2).
# Figures plots/nh_phi_mixed.png, nh_field_mixed.png.
NH_PH = ((0.5, 0.5), (0.4, 0.6), (0.5, 0.1), (0.8, 0.1))          # [ep7] pairs, equal -> strongly unequal
NH_PH_LBL = ('equal bonds', 'weakly unequal', 'unequal', 'strongly unequal')
NH_TL = (0.5, 0.8)                                                # [ep8] maps: tau_L fixed, tau_R varied
NH_T = (0.0, 10.0, 20.0, 50.0)                                    # mK
NH_TCOL = ('#0b2a4a', '#2a78d6', '#6fa8e6', '#b4d2f3')            # one hue, dark = cold
NH_SEG, NH_SEG_LT = '#eb6834', '#f5c9b3'                          # EP segments (between EPs: both poles on Re = 0)
NH_PCOL = {(0.8, 0.2): '#9b59d0', (0.5, 0.4): '#1baf7a'}          # the two [ep8] plateau routes B / A
NH_DECK = 'scharf_absDeck_T_mixed_tau0.5_L2_EZ0.85-1.05.npz'      # phase 0 of study_mixed.PHI_TB = 1.7417 pi


def nh_spans(x, m, pad=0.0):
    """(start - pad, end + pad) of the True runs of m on the grid x."""
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(m).astype(int), 0]))
    return [(x[i0] - pad, x[i1 - 1] + pad) for i0, i1 in zip(e[::2], e[1::2])]


def nh_kT_lines(a, kT, Ts=(10.0, 20.0)):
    """Horizontal k_B T lines (ueV) with a label at the right edge."""
    for T in Ts:
        a.axhline(kT[T], color='#52514e', lw=0.9, ls='--')
        a.text(0.99, kT[T] * 1.08, rf'$k_BT$, {T:g} mK', transform=a.get_yaxis_transform(), ha='right', va='bottom',
               fontsize=7.5, color='#52514e')


# %% [nh] --- 1. ALONG phi at E_Z 0.93: kappa(E, phi) and kappa(T, phi), equal -> strongly unequal bonds ---
if run_part('nh'):
    x = EP_PH / np.pi
    kT = {T: st.kT_of_mK(T) * 1e3 for T in NH_T}                     # ueV
    fig, axs = plt.subplots(2, len(NH_PH), figsize=(4.3 * len(NH_PH), 7.6), sharex=True, sharey='row',
                            layout='constrained', gridspec_kw=dict(height_ratios=(1.25, 1)))
    print(f"\n  [nh] ALONG phi, E_Z {EP_EZ:g} meV ([ep7] caches): share of phi between EPs; kappa max between / outside "
          f"the EPs at {' / '.join(f'{T:g}' for T in NH_T)} mK; median kappa(20 mK) / kappa(0) between / outside; "
          "narrower pole width between / outside (median, ueV)", flush=True)
    for col, c in enumerate(NH_PH):
        V = np.load(st.DATA / f'scharf_absEP7_ch_mixed_a10_EZ{EP_EZ:g}_tauL{c[0]:g}_R{c[1]:g}.npz')['values']
        K = ep7_kappaE(V)
        k = {T: ep7_thermal(V, T)[:, 0] for T in NH_T}
        xp, P, btw = ep7_poles(*c)
        ins = np.interp(x, xp, btw.astype(float)) > 0.5
        wn = np.interp(x, xp, np.min(-P.imag, axis=1))               # the narrower pole's width (ueV)
        mx = lambda T, m: np.max(k[T][m]) if m.any() else np.nan
        ok = k[0.0] > 0.05
        med = lambda m: np.median((k[20.0] / k[0.0])[ok & m]) if (ok & m).any() else np.nan
        mw = lambda m: np.median(wn[m]) if m.any() else np.nan
        print(f"    ({c[0]:g}, {c[1]:g}): {ins.mean() * 100:4.1f} % of phi; kappa max "
              + "  ".join(f"{mx(T, ins):.3f}/{mx(T, ~ins):.3f}" for T in NH_T)
              + f"; ratio {med(ins):.3f} / {med(~ins):.3f}; width {mw(ins):.2f} / {mw(~ins):.2f}", flush=True)
        a = axs[0, col]
        im = a.pcolormesh(x, EP7_E[1:], K[1:], shading='nearest', cmap='Blues', vmin=0, vmax=0.6, rasterized=True)
        a.plot(x, wn, color='#1b1b1b', lw=1.1, label='narrower pole: width')
        nh_kT_lines(a, kT)
        for s, e in nh_spans(x, ins, pad=1 / 120):
            a.axvspan(s, e, ymin=0.965, ymax=1.0, color=NH_SEG, lw=0)
        a.set_yscale('log')
        a.set_ylim(0.01, 70.0)
        a.set_title(f'({c[0]:g}, {c[1]:g}): {NH_PH_LBL[col]}', fontsize=10.5)
        b = axs[1, col]
        for s, e in nh_spans(x, ins, pad=1 / 120):
            b.axvspan(s, e, color=NH_SEG_LT, lw=0, zorder=0)
        for T, tc in zip(NH_T, NH_TCOL):
            b.plot(x, k[T], color=tc, lw=1.6, label=f'{T:g} mK')
        b.axhline(0.5, color='#52514e', lw=0.8, ls='--')
        b.text(0.02, 0.95, f'between EPs: {ins.mean() * 100:.0f} % of $\\phi$', transform=b.transAxes, fontsize=8.5,
               va='top')
        b.set_ylim(0, 0.62)
        b.set_xlabel(r'$\phi/\pi$')
        b.grid(color='#e4e3df', lw=0.6)
    axs[0, 0].set_ylabel(r'$E$ ($\mu$eV)')
    axs[1, 0].set_ylabel(r'$\kappa(T)/\kappa_0$')
    axs[0, 0].legend(fontsize=8, frameon=False, loc='lower left')
    axs[1, 0].legend(title='T', fontsize=8.5, frameon=False, loc='upper right')
    fig.colorbar(im, ax=axs[0, :], pad=0.01, label=r'$\kappa(E)/\kappa_0$, T = 0')
    fig.suptitle(rf'Along $\phi$ at $E_Z$ = {EP_EZ:g} meV, bonds ($\tau_L$, $\tau_R$): $\kappa(E, \phi)$ (top) and '
                 r'$\kappa(T, \phi)$ (bottom).  Orange: between EPs (both pair poles on Re $\lambda$ = 0)', fontsize=11)
    st.save_fig(fig, 'nh_phi_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/nh_phi_mixed.png", flush=True)


# %% [nh] --- 2. ALONG E_Z at phi 1.7417 pi: kappa(E_Z, tau_R), plateau vs the widths, equal bonds at finite T ---
if run_part('nh'):
    E = EP8_EZ * 1e3                                                 # ueV
    kT = {T: st.kT_of_mK(T) * 1e3 for T in NH_T}
    F8 = np.load(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz')['values']
    F8 = F8.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
    IDX = {c: n for n, c in enumerate(EP8_PAIRS)}
    EC = ep8_crossings(E, np.abs(F8[:, IDX[(0.0, 0.0)], 0]))

    def nh_pair(c):
        """kappa, G, the between-EPs mask and the pole widths (narrow, broad; ueV) along E_Z for the bond pair c."""
        a_ = F8[:, IDX[c], 0] + 1j * F8[:, IDX[c], EP_K]
        b_ = F8[:, IDX[c], 1] + 1j * F8[:, IDX[c], EP_K + 1]
        ins = np.zeros(len(E), bool)
        for s, e, _ in ep8_segments(E, (((a_ - b_) / 2) ** 2).real):
            ins |= (E >= s) & (E <= e)
        t = F8[:, IDX[c], 2 * EP_K + 1:]
        return t[:, 0], t[:, 1], ins, np.sort(np.c_[-a_.imag, -b_.imag], axis=1)

    G1 = {t: -(F8[:, IDX[(t, 0.0)], EP_K] + F8[:, IDX[(t, 0.0)], EP_K + 1]) for t in EP8_T}   # one-probe pair width
    PR = {c: nh_pair(c) for c in EP8_PAIRS if c[1] > 0}
    print(f"\n  [nh] ALONG E_Z, phi {EP8_PHI / np.pi:.4f} pi, E = 0, T = 0 ([ep8]): plateau windows (|kappa - 1/2| < "
          "0.01 / 0.02 / 0.03, ueV), share of the 0.02 window between EPs, sqrt(Gamma_L Gamma_R) and the narrower pole "
          "width on it (ueV); sorted by sqrt(Gamma_L Gamma_R)", flush=True)
    SC = []
    for c, (k_, G_, ins, wd) in PR.items():
        w = [ep9_window(k_, G_, tol) for tol in (0.01, 0.02, 0.03)]
        s = w[1][2] if w[1][2] is not None else slice(None)
        gm = float(np.median(np.sqrt(G1[c[0]][s] * G1[c[1]][s])))
        SC.append((gm, c, w, float(ins[s].mean()) if w[1][2] is not None else np.nan,
                   (wd[s, 0].min(), wd[s, 0].max()) if w[1][2] is not None else (np.nan, np.nan)))
    for gm, c, w, sh, wn in sorted(SC):
        if w[2][0] > 0:
            print(f"    ({c[0]:g}, {c[1]:g}): sqrt(G_L G_R) {gm:5.2f}; windows {w[0][0]:3.0f} / {w[1][0]:3.0f} / "
                  f"{w[2][0]:3.0f}; between EPs {sh * 100 if sh == sh else np.nan:3.0f} %; narrow pole "
                  f"{wn[0]:.1f}-{wn[1]:.1f}", flush=True)
    D = np.load(st.DATA / NH_DECK)
    VD, ED, EZD = D['values'], D['axis_E'], D['axis_E_Z']             # (n_EZ, n_E, 6 x 3 phases), meV, meV
    KD = (VD[..., 0] + VD[..., 1] + VD[..., 2] + VD[..., 3]) / 2      # phase 0 = 1.7417 pi: kappa(E_Z, E)
    THD = {T: st.thermal_line(VD, st.kT_of_mK(T), ED, phase=0) for T in NH_T}
    com = np.isin(np.round(EZD, 4), np.round(EP8_EZ, 4))
    k55 = PR[(0.5, 0.5)][0][np.isin(np.round(EP8_EZ, 4), np.round(EZD, 4))]
    print(f"    CHECK: deck 0.5 / 0.5 at T = 0 vs [ep8] (0.5, 0.5) on {com.sum()} common fields: max |d kappa| "
          f"{np.abs(THD[0.0][com, 0] - k55).max():.1e}", flush=True)
    dE = (EZD[1] - EZD[0]) * 1e3
    print(f"    EQUAL bonds 0.5 / 0.5 along E_Z {EZD[0]:g}-{EZD[-1]:g} meV (step {dE:g} ueV), finite T (deck cache):",
          flush=True)
    for T in NH_T:
        kk, GG = THD[T][:, 0], THD[T][:, 1]
        print(f"      {T:4.0f} mK: kappa max {kk.max():.3f}; window <0.02 / <0.03: "
              f"{longest_run(np.abs(kk - 0.5) < 0.02) * dE:3.0f} / {longest_run(np.abs(kk - 0.5) < 0.03) * dE:3.0f} ueV; "
              f"longest kappa > 0.45 run {longest_run(kk > 0.45) * dE:3.0f} ueV, kappa > 0.4 {longest_run(kk > 0.4) * dE:3.0f} "
              f"ueV; max |G| {np.abs(GG).max():.3f}", flush=True)

    fig, axs = plt.subplots(2, 3, figsize=(16.5, 9.4), layout='constrained')
    for a, tl in zip(axs[0, :2], NH_TL):
        rows = [tr for tr in EP8_T if tr <= tl]
        im = a.pcolormesh(EP8_EZ, np.arange(len(rows)), np.array([PR[(tl, tr)][0] for tr in rows]), shading='nearest',
                          cmap='Blues', vmin=0, vmax=0.6, rasterized=True)
        for j, tr in enumerate(rows):
            k_, _, ins, _ = PR[(tl, tr)]
            for s, e in nh_spans(EP8_EZ, ins, pad=0.0025):
                a.plot([s, e], [j - 0.2] * 2, color=NH_SEG, lw=2.8, solid_capstyle='butt')
            for s, e in nh_spans(EP8_EZ, np.abs(k_ - 0.5) < 0.02, pad=0.0025):
                a.plot([s, e], [j + 0.2] * 2, color='#1b1b1b', lw=2.8, solid_capstyle='butt')
        for ec in EC:
            a.axvline(ec * 1e-3, color='#52514e', lw=0.9, ls=':')
        a.set_yticks(range(len(rows)), [f'{tr:g}' for tr in rows])
        a.set_ylabel(r'$\tau_R$')
        a.set_xlabel(r'$E_Z$ (meV)')
        a.set_title(rf'$\kappa(E_Z)$ at $E$ = 0, T = 0, $\tau_L$ = {tl:g}', fontsize=10.5)
    axs[0, 0].legend(handles=[mpl.lines.Line2D([], [], color=NH_SEG, lw=3.2, label='between EPs'),
                              mpl.lines.Line2D([], [], color='#1b1b1b', lw=3.2, label=r'$|\kappa - 1/2| < 0.02$'),
                              mpl.lines.Line2D([], [], color='#52514e', lw=0.9, ls=':', label='lossless crossing')],
                     fontsize=8, frameon=True, loc='upper left', framealpha=0.9)
    fig.colorbar(im, ax=axs[0, 1], pad=0.01, label=r'$\kappa/\kappa_0$')
    a = axs[0, 2]
    lbl = {(0.5, 0.4): (-8, 0, 'right'), (0.8, 0.2): (8, 0, 'left'), (0.6, 0.4): (-8, 5, 'right'),
           (0.5, 0.5): (8, 5, 'left'), (0.6, 0.3): (-8, -10, 'right'), (0.8, 0.4): (8, -10, 'left'),
           (0.8, 0.3): (-8, -10, 'right'), (0.6, 0.5): (8, -10, 'left')}
    for gm, c, w, sh, _ in SC:
        inside = sh == sh and sh >= 0.5
        a.plot(gm, w[1][0], 'o', ms=7, mfc=NH_SEG if inside else '#2a78d6', mec='w', mew=1.0)
        if c in lbl:
            dx, dy, ha = lbl[c]
            a.annotate(f'({c[0]:g}, {c[1]:g})', (gm, w[1][0]), xytext=(dx, dy), textcoords='offset points', fontsize=8,
                       ha=ha, va='center')
    a.set_xscale('log')
    a.set_xlabel(r'$\sqrt{\Gamma_L \Gamma_R}$ on the window ($\mu$eV; $\Gamma$ = one-probe pair width)')
    a.set_ylabel(r'longest $|\kappa - 1/2| < 0.02$ window along $E_Z$ ($\mu$eV)')
    a.set_title('plateau length vs the probe widths, T = 0', fontsize=10.5)
    a.legend(handles=[mpl.lines.Line2D([], [], ls='', marker='o', ms=7, mfc=NH_SEG, mec='w', label='window between EPs'),
                      mpl.lines.Line2D([], [], ls='', marker='o', ms=7, mfc='#2a78d6', mec='w',
                                       label='window outside EP segments')], fontsize=8, frameon=False, loc='upper left')
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 0]
    m = ED > 0
    im = a.pcolormesh(EZD, ED[m] * 1e3, KD[:, m].T, shading='nearest', cmap='Blues', vmin=0, vmax=0.6, rasterized=True)
    nh_kT_lines(a, kT, (10.0, 20.0, 50.0))
    a.set_yscale('log')
    a.set_ylim(ED[m][0] * 1e3, ED[-1] * 1e3)
    a.set_ylabel(r'$E$ ($\mu$eV)')
    a.set_title(r'$\kappa(E, E_Z)$, equal bonds 0.5 / 0.5, T = 0', fontsize=10.5)
    fig.colorbar(im, ax=a, pad=0.01, label=r'$\kappa(E)/\kappa_0$')
    a = axs[1, 1]
    for T, tc in zip(NH_T, NH_TCOL):
        a.plot(EZD, THD[T][:, 0], color=tc, lw=1.7, label=f'(0.5, 0.5), {T:g} mK')
    for c, col in NH_PCOL.items():
        a.plot(EP8_EZ, PR[c][0], color=col, lw=1.5, ls='--', label=f'({c[0]:g}, {c[1]:g}), T = 0 only')
    a.axhline(0.5, color='#52514e', lw=0.8, ls='--')
    a.set_ylim(0, 0.8)
    a.set_ylabel(r'$\kappa/\kappa_0$')
    a.set_title(r'$\kappa(E_Z)$: equal bonds vs T; unequal T = 0 only', fontsize=10.5)
    a.legend(fontsize=7.5, frameon=False, loc='upper center', ncol=2)
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 2]
    for c, col in NH_PCOL.items():
        wd = PR[c][3]
        a.plot(EP8_EZ, wd[:, 0], color=col, lw=1.7, label=f'({c[0]:g}, {c[1]:g}) narrower pole')
        a.plot(EP8_EZ, wd[:, 1], color=col, lw=1.2, ls='--', label=f'({c[0]:g}, {c[1]:g}) broader pole')
    nh_kT_lines(a, kT, (10.0, 20.0, 50.0))
    a.set_yscale('log')
    a.set_ylabel(r'pole width $-\mathrm{Im}\,\lambda$ ($\mu$eV)')
    a.set_title('the two plateau routes: pole widths vs $k_BT$', fontsize=10.5)
    a.legend(fontsize=7.5, frameon=False, loc='upper right')
    a.grid(color='#e4e3df', lw=0.6, which='both')
    for a in axs[1]:
        a.set_xlim(EP8_EZ[0], EP8_EZ[-1])
        a.set_xlabel(r'$E_Z$ (meV)')
        for ec in EC:
            a.axvline(ec * 1e-3, color='#52514e', lw=0.9, ls=':')
    for a, t in zip(axs.ravel(), '(a) (b) (c) (d) (e) (f)'.split()):
        a.set_title(t, fontsize=11, loc='left')
    fig.suptitle(rf'Along $E_Z$ at $\phi$ = {EP8_PHI / np.pi:.4f} $\pi$, mixed SOC, L 2 $\mu$m.  (d), (e) equal bonds: '
                 r'$E_Z$ 0.85-1.05 meV only (study_mixed [deck]); finite T for unequal bonds not computed', fontsize=11)
    st.save_fig(fig, 'nh_field_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/nh_field_mixed.png", flush=True)


# %% [nh] --- 3. WHY inside / outside the EPs differ: E = 0 sees the PRODUCT of the pair poles, finite T the NEAREST pole ---
# Tested here on the full model (no model assumed): (1) det r_L(E = 0) = 1 - 2 q, q = Gamma_L Gamma_R / |lambda_1 lambda_2|
# (Gamma = one-probe pair width [ep8], lambda = two-probe pair poles); (2) kappa(E) / kappa(0) = |l1 l2|^2 / |(E - l1)(E - l2)|^2;
# (3) k_B T_half (kappa(T) = kappa(0) / 2) vs the nearest pole's distance from E = 0.  Particle-hole symmetry puts the pair at
# +-E0 - i gamma (outside the EPs: |l1| = |l2|) or at -i w+, -i w- (between them): same product, different nearest pole.
NH_EX_EZ = 945.0                                                  # ueV: inside both [ep8] plateau windows (A and B)
NH_TS = np.geomspace(0.2, 200.0, 60)                              # mK, for T_half


def nh_thalf(V, n, k0):
    """T (mK) where kappa(T) first drops to k0 / 2 at phase index n of an [ep7] channel scan (log-interpolated)."""
    k = np.array([st.thermal_line(V[None], st.kT_of_mK(T), EP7_E * 1e-3, phase=n)[0][0] for T in NH_TS])
    j = np.flatnonzero(k < k0 / 2)
    if j.size == 0 or j[0] == 0:
        return np.nan
    j = j[0]
    f = (k[j - 1] - k0 / 2) / (k[j - 1] - k[j])
    return float(np.exp(np.log(NH_TS[j - 1]) + f * np.log(NH_TS[j] / NH_TS[j - 1])))


if run_part('nh'):
    DR = {c: F8[:, IDX[c], 2 * EP_K + 4] for c in PR}                 # det r_L at E = 0
    PP = {c: (F8[:, IDX[c], 0] + 1j * F8[:, IDX[c], EP_K]) * (F8[:, IDX[c], 1] + 1j * F8[:, IDX[c], EP_K + 1]) for c in PR}
    QQ = {c: G1[c[0]] * G1[c[1]] / np.abs(PP[c]) for c in PR}
    print("\n  [nh] WHY (1): det r_L = 1 - 2 Gamma_L Gamma_R / |l1 l2| along E_Z, every field: max |deviation| per pair "
          "(max |Im(l1 l2)| / |l1 l2| = " + f"{max(np.max(np.abs(PP[c].imag) / np.abs(PP[c])) for c in PR):.1e})", flush=True)
    print("    " + "  ".join(f"({c[0]:g},{c[1]:g}) {np.abs(DR[c] - (1 - 2 * QQ[c])).max():.3f}" for c in PR), flush=True)
    print("    kappa = (1 - det r^2) / 2: max |deviation| over all pairs and fields "
          f"{max(np.abs(PR[c][0] - (1 - DR[c] ** 2) / 2).max() for c in PR):.3f}", flush=True)
    # ep7: kappa(E) two-pole test and T_half vs the nearest pole
    EX7 = {}
    TH7 = []
    print("  [nh] WHY (2), (3) along phi at E_Z 0.93 ([ep7]): kappa(E <= 20 ueV) vs the two-pole form; k_B T_half / min|lambda|",
          flush=True)
    for c in EP7_CFG:
        V = np.load(st.DATA / f'scharf_absEP7_ch_mixed_a10_EZ{EP_EZ:g}_tauL{c[0]:g}_R{c[1]:g}.npz')['values']
        K = ep7_kappaE(V)
        xp, P, btw = ep7_poles(*c)
        j = np.array([np.argmin(np.abs(xp - v)) for v in EP_PH / np.pi])
        P, btw = P[j], btw[j]
        l1, l2 = P[:, 0], P[:, 1]
        pred = K[0][None] * np.abs(l1 * l2)[None] ** 2 / np.abs((EP7_E[:, None] - l1[None]) * (EP7_E[:, None] - l2[None])) ** 2
        ok, me = K[0] > 0.05, EP7_E <= 20.0
        err = np.abs(K - pred)[me][:, ok]
        lmin = np.minimum(np.abs(l1), np.abs(l2))
        th = np.array([nh_thalf(V, n, K[0, n]) if ok[n] else np.nan for n in range(len(EP_PH))])
        r = st.kT_of_mK(th) * 1e3 / lmin
        TH7.append((c, st.kT_of_mK(th) * 1e3, lmin, btw))
        EX7[c] = (K, pred, l1, l2, btw)
        g = np.isfinite(r)
        print(f"    ({c[0]:g}, {c[1]:g}): kappa(E) max |dk| {err.max():.3f} (median {np.median(err):.1e}); "
              f"k_B T_half / min|lambda| median {np.median(r[g]):.2f} (p10-p90 {np.percentile(r[g], 10):.2f}-"
              f"{np.percentile(r[g], 90):.2f}, {g.sum()} phases)", flush=True)
    RALL = np.concatenate([t[1] / t[2] for t in TH7])
    FT = float(np.nanmedian(RALL))
    print(f"    all pairs: k_B T_half = {FT:.2f} x min|lambda| (median; p10-p90 {np.nanpercentile(RALL, 10):.2f}-"
          f"{np.nanpercentile(RALL, 90):.2f})", flush=True)
    i0 = int(np.argmin(np.abs(E - NH_EX_EZ)))
    print(f"  [nh] the two plateau routes at E_Z {E[i0]:.0f} ueV (phi 1.7417 pi):", flush=True)
    for c in NH_PCOL:
        a_, b_ = PP[c][i0], None
        l = np.array([F8[i0, IDX[c], 0] + 1j * F8[i0, IDX[c], EP_K], F8[i0, IDX[c], 1] + 1j * F8[i0, IDX[c], EP_K + 1]])
        lm = np.abs(l).min()
        print(f"    ({c[0]:g}, {c[1]:g}): poles {l[0]:.2f}, {l[1]:.2f} ueV; |l1 l2| {np.abs(PP[c][i0]):.1f} vs 2 G_L G_R "
              f"{2 * G1[c[0]][i0] * G1[c[1]][i0]:.1f} ueV^2; kappa {PR[c][0][i0]:.3f}; nearest pole {lm:.2f} ueV -> "
              f"kappa halved near {lm * FT / (st.kT_of_mK(1.0) * 1e3):.0f} mK (rule above; not computed)", flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(13.0, 10.6), layout='constrained')
    a = axs[0, 0]
    th_ = np.linspace(0, np.pi, 200)
    for c, col in NH_PCOL.items():
        l = np.array([F8[i0, IDX[c], 0] + 1j * F8[i0, IDX[c], EP_K], F8[i0, IDX[c], 1] + 1j * F8[i0, IDX[c], EP_K + 1]])
        R = np.sqrt(np.abs(l[0] * l[1]))
        a.plot(R * np.cos(th_), R * np.sin(th_), color=col, lw=1.0, ls='--')
        a.plot(l.real, -l.imag, 'o', ms=9, mfc=col, mec='w', mew=1.2,
               label=rf'({c[0]:g}, {c[1]:g}): $\sqrt{{|\lambda_1\lambda_2|}}$ = {R:.1f} $\mu$eV, nearest {np.abs(l).min():.1f}')
    for T in (10.0, 20.0):
        R = st.kT_of_mK(T) * 1e3 / FT
        a.plot(R * np.cos(th_), R * np.sin(th_), color='#52514e', lw=0.9, ls=':')
        a.text(R + 0.3, 0.3, rf'$\kappa/2$ at {T:g} mK', ha='left', va='bottom', fontsize=7.5, color='#52514e',
               rotation=90)
    a.set_aspect('equal')
    a.set_xlim(-16, 16)
    a.set_ylim(0, 48)
    a.set_xlabel(r'Re $\lambda$ ($\mu$eV)')
    a.set_ylabel(r'$-$Im $\lambda$ = width ($\mu$eV)')
    a.set_title(rf'pair poles, plateau routes, $E_Z$ = {E[i0]:.0f} $\mu$eV:' + '\nsame product, different nearest pole',
                fontsize=9.5)
    a.legend(fontsize=7.5, frameon=False, loc='upper left')
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[0, 1]
    for c in PR:
        if c == (0.8, 0.8):
            continue
        ins = PR[c][2]
        a.plot(QQ[c][~ins], DR[c][~ins], '.', ms=3, color='#2a78d6')
        a.plot(QQ[c][ins], DR[c][ins], '.', ms=3, color=NH_SEG)
    qq = np.linspace(0, 1, 50)
    a.plot(qq, 1 - 2 * qq, color='#1b1b1b', lw=1.0, label=r'$1 - 2q$')
    a.plot([], [], 'o', color='#2a78d6', ms=5, label='outside EP segments')
    a.plot([], [], 'o', color=NH_SEG, ms=5, label='between EPs')
    a.axvline(0.5, color='#52514e', lw=0.8, ls='--')
    a.axhline(0.0, color='#52514e', lw=0.8, ls='--')
    a.text(0.51, 0.75, r'$q = 1/2$: det $r = 0$, $\kappa = 1/2$', fontsize=8, color='#52514e')
    a.set_xlabel(r'$q = \Gamma_L\Gamma_R\, /\, |\lambda_1 \lambda_2|$  (one-probe widths / two-probe pole product)')
    a.set_ylabel(r'det $r_L$ at $E$ = 0 (full model)')
    a.set_title(r'$E$ = 0: only the pole PRODUCT enters' + '\n' + r'(25 pairs x 61 fields along $E_Z$; '
                r'$\kappa = (1 - \det r^2)/2$)', fontsize=9.5)
    a.legend(fontsize=8, frameon=False, loc='lower left')
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 0]
    cin, cout = (0.5, 0.1), (0.4, 0.6)
    for c, ok_in, col in ((cin, True, NH_SEG), (cout, False, '#2a78d6')):
        K, pred, l1, l2, btw = EX7[c]
        m = btw if ok_in else ~btw
        n = int(np.flatnonzero(m)[np.argmax(K[0][m])]) if ok_in else int(np.argmin(np.abs(EP_PH / np.pi - 1.5)))
        lm = min(abs(l1[n]), abs(l2[n]))
        a.plot(EP7_E[1:], K[1:, n], color=col, lw=2.0,
               label=rf'({c[0]:g}, {c[1]:g}), $\phi$ = {EP_PH[n] / np.pi:.2f}$\pi$ ' + ('between EPs' if ok_in else 'outside'))
        a.plot(EP7_E[1:], pred[1:, n], color='#1b1b1b', lw=1.0, ls='--')
        a.axvline(lm, color=col, lw=1.0, ls=':')
        a.text(lm * 1.08, 0.56, f'nearest pole\n{lm:.2f} $\\mu$eV', color=col, fontsize=7.5, va='top')
    a.plot([], [], color='#1b1b1b', lw=1.0, ls='--', label='two nearest poles only')
    a.set_xscale('log')
    a.set_xlim(0.01, 70)
    a.set_ylim(0, 0.6)
    a.set_xlabel(r'$E$ ($\mu$eV)')
    a.set_ylabel(r'$\kappa(E)/\kappa_0$, T = 0')
    a.set_title(r'finite $E$: $\kappa(E)$ falls off at the NEAREST pole ($E_Z$ 0.93 meV)', fontsize=9.5)
    a.legend(fontsize=7.5, frameon=False, loc='lower left')
    a.grid(color='#e4e3df', lw=0.6, which='both')
    a = axs[1, 1]
    for c, kth, lmin, btw in TH7:
        a.plot(lmin[~btw], kth[~btw], '.', ms=4, color='#2a78d6')
        a.plot(lmin[btw], kth[btw], '.', ms=4, color=NH_SEG)
    ll = np.geomspace(0.1, 30, 20)
    a.plot(ll, FT * ll, color='#1b1b1b', lw=1.0, label=rf'$k_BT_{{1/2}} = {FT:.2f}\,|\lambda|_{{min}}$')
    a.plot([], [], 'o', color='#2a78d6', ms=5, label='outside EP segments')
    a.plot([], [], 'o', color=NH_SEG, ms=5, label='between EPs')
    for T in (10.0, 20.0, 50.0):
        a.axhline(st.kT_of_mK(T) * 1e3, color='#52514e', lw=0.8, ls=':')
        a.text(0.12, st.kT_of_mK(T) * 1e3 * 1.06, f'{T:g} mK', fontsize=7.5, color='#52514e')
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'distance of the nearest pair pole from $E$ = 0, min $|\lambda|$ ($\mu$eV)')
    a.set_ylabel(r'$k_B T_{1/2}$: $\kappa(T) = \kappa(0)/2$ ($\mu$eV)')
    a.set_title('finite T: the nearest pole sets the temperature scale (5 pairs x 121 phases)', fontsize=9.5)
    a.legend(fontsize=8, frameon=False, loc='upper left')
    a.grid(color='#e4e3df', lw=0.6, which='both')
    for a, t in zip(axs.ravel(), '(a) (b) (c) (d)'.split()):
        a.set_title(f'{t}  {a.get_title()}', fontsize=9.5, loc='left')
        a.set_title('')
    fig.suptitle('Why the plateau does not care about the EPs at T = 0 but does at finite T (full model, mixed SOC)',
                 fontsize=11.5)
    st.save_fig(fig, 'nh_why_poles_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/nh_why_poles_mixed.png", flush=True)


# %% [nh] --- 4. THE TWO kappa = 1/2 TYPES: det r crossing 0 (lobe flank) vs det r peaking just below 0 (lobe top) ---
# With det r = 1 - 2q = tanh(ln x), x = eps_eff / sqrt(Gamma_L Gamma_R), eps_eff^2 = |l1 l2| - Gamma_L Gamma_R (the
# probe-shifted splitting): |kappa - 1/2| < 0.02 <=> |det r| < 0.2 <=> x within 0.82-1.22.  det r peaks where eps_eff
# peaks (the top of the splitting lobe between the two lossless crossings).
NH_TYPES = {(0.5, 0.5): '#2a78d6', (0.5, 0.4): '#1baf7a', (0.3, 0.3): '#d94f8a'}
NH_DOME = (850.0, 1020.0)                                          # ueV: the lobe between the lossless crossings


if run_part('nh'):
    XX = {c: np.sqrt(np.where(QQ[c] < 1, 1 / QQ[c] - 1, np.nan)) for c in PR}
    dome = (E >= NH_DOME[0]) & (E <= NH_DOME[1])
    print("\n  [nh] TWO kappa = 1/2 TYPES along E_Z (|kappa - 1/2| < 0.02 windows; tanh check: det r vs tanh(ln x) max "
          f"{max(np.nanmax(np.abs(DR[c] - np.tanh(np.log(XX[c])))) for c in PR if c != (0.8, 0.8)):.3f} without 0.8/0.8)",
          flush=True)
    CL = {'flank': [], 'top below': [], 'top just above': [], 'det r > 0': []}
    for c in PR:
        k_ = PR[c][0]
        m = np.abs(k_ - 0.5) < 0.02
        e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        for j0, j1 in zip(e[::2], e[1::2]):
            sg = DR[c][max(j0 - 1, 0):min(j1 + 1, len(E))]
            nc = int(np.sum(np.diff(np.sign(sg)) != 0))
            key = ('flank' if nc == 1 else 'top just above' if nc >= 2 else 'top below' if sg.max() < 0 else 'det r > 0')
            CL[key].append(((j1 - j0) * EP8_DE, c))
    for key, v in CL.items():
        L = [w for w, _ in v]
        print(f"    {key:15s}: {len(v):2d} windows, length median {np.median(L) if L else 0:3.0f}, max {max(L) if L else 0:3.0f} "
              f"ueV; longest: " + ", ".join(f"({c[0]:g},{c[1]:g}) {w:.0f}" for w, c in sorted(v, key=lambda r: -r[0])[:4]),
              flush=True)
    TOP = {c: (np.nanmax(DR[c][dome]), longest_run(np.abs(PR[c][0][dome] - 0.5) < 0.02) * EP8_DE) for c in PR}
    print("    det r at the lobe top (max over E_Z 850-1020 ueV) -> longest window there: "
          + "  ".join(f"({c[0]:g},{c[1]:g}) {t:+.2f}->{w:.0f}" for c, (t, w) in sorted(TOP.items(), key=lambda r: r[1][0])),
          flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.6), layout='constrained')
    lbl = {(0.5, 0.5): 'lobe top just below', (0.5, 0.4): 'lobe top on the line', (0.3, 0.3): 'lobe top far above'}
    a = axs[0, 0]
    a.axhspan(np.exp(-np.arctanh(0.2)), np.exp(np.arctanh(0.2)), color='#e4e3df', lw=0)
    a.axhline(1.0, color='#52514e', lw=0.8, ls='--')
    for c, col in NH_TYPES.items():
        a.plot(EP8_EZ, XX[c], color=col, lw=1.8, label=f'({c[0]:g}, {c[1]:g}): {lbl[c]}')
    a.set_yscale('log')
    a.set_ylim(0.05, 5)
    a.set_ylabel(r'$x = \varepsilon_{eff} / \sqrt{\Gamma_L\Gamma_R}$')
    a.set_title(r'splitting vs probe coupling (grey: $|\kappa - 1/2| < 0.02$)', fontsize=9.5)
    a.legend(fontsize=8, frameon=False, loc='lower center')
    a = axs[0, 1]
    a.axhspan(-0.2, 0.2, color='#e4e3df', lw=0)
    a.axhline(0.0, color='#52514e', lw=0.8, ls='--')
    for c, col in NH_TYPES.items():
        a.plot(EP8_EZ, DR[c], color=col, lw=1.8)
    a.set_ylim(-1.05, 1.05)
    a.set_ylabel(r'det $r_L$ at $E$ = 0')
    a.set_title(r'det $r$ = tanh(ln $x$): crosses 0 where $x$ = 1, peaks where the splitting peaks', fontsize=9.5)
    a = axs[1, 0]
    a.axhspan(0.48, 0.52, color='#e4e3df', lw=0)
    for c, col in NH_TYPES.items():
        a.plot(EP8_EZ, PR[c][0], color=col, lw=1.8)
    a.set_ylim(0, 0.56)
    a.set_ylabel(r'$\kappa/\kappa_0$ at $E$ = 0, T = 0')
    a.set_title(r'$\kappa = (1 - \det r^2)/2$: a crossing gives a short window, a flat top a long one', fontsize=9.5)
    for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
        a.set_xlabel(r'$E_Z$ (meV)')
        for ec in EC:
            a.axvline(ec * 1e-3, color='#52514e', lw=0.9, ls=':')
        a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 1]
    a.axvspan(-0.2, 0.2, color='#e4e3df', lw=0)
    for c, (t, w) in TOP.items():
        col = NH_TYPES.get(c, '#8a8986')
        a.plot(t, w, 'o', ms=8 if c in NH_TYPES else 6, mfc=col, mec='w', mew=1.0)
        if c in ((0.8, 0.2), (0.6, 0.4), (0.6, 0.3), (0.8, 0.4)):
            left = c == (0.8, 0.2)
            a.annotate(f'({c[0]:g}, {c[1]:g})', (t, w), xytext=(-7 if left else 6, 2), textcoords='offset points',
                       fontsize=7.5, ha='right' if left else 'left')
    a.set_xlabel(r'det $r$ at the top of the splitting lobe (max over $E_Z$ 0.85-1.02 meV)')
    a.set_ylabel(r'longest $|\kappa - 1/2| < 0.02$ window on that lobe ($\mu$eV)')
    a.set_title(r'all 26 pairs: the plateau is longest when the lobe top sits at det $r \approx 0$', fontsize=9.5)
    a.grid(color='#e4e3df', lw=0.6)
    for a, t in zip(axs.ravel(), '(a) (b) (c) (d)'.split()):
        a.set_title(f'{t}  {a.get_title()}', fontsize=9.5, loc='left')
        a.set_title('')
    fig.suptitle(rf'The two $\kappa = 1/2$ types along $E_Z$ ($\phi$ = {EP8_PHI / np.pi:.4f}$\pi$, full model): det $r$ crossing 0 '
                 r'vs det $r$ peaking near 0', fontsize=11.5)
    st.save_fig(fig, 'nh_detr_types_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/nh_detr_types_mixed.png", flush=True)


# =====================================================================================================================
# PART [trans] (2026-11-19) -- STUDY: long junctions AT the upper topological transition (phi = pi, mixed SOC).  Akhmerov
# et al. (PRL 106, 057001, Eq. (3)): in a long class-D wire kappa = (1/2) sum_n T_n, T_n = 1 / cosh^2(lambda_n), the
# slowest lambda ~ L / xi_M -> a universal kappa_0 / 2 at the transition (xi_M -> inf), exponentially small inside the
# phase.  Does the finite junction show it, and does the kappa = 1/2 feature converge onto E_c as L grows?
# Bulk at phi = pi (scratch run 2026-11-19, osc_point): Q -1 -> +1 between E_Z 1.36 and 1.38 meV; gap 2.3 / 1.1 / 0.27 ueV
# at 1.30 / 1.34 / 1.36, xi_M 7.2 / 10.8 / 15.5 um; the trivial side is GAPLESS (gap 0 from 1.40).  So L >> xi_M is out of
# reach next to E_c (L 8 um costs 163 s / point, ~ L^3 from the ribbon decimation); the test is the trend with L.
# tau 1 (bulk bond: the probe continues the strip) is the ideal-lead case of Akhmerov; tau 0.3387 the thesis tunnel probe.
# Caches scharf_absTR_*; figure plots/trans_mixed.png.
TR_L = (2000.0, 4000.0)                     # L 8 um (TR_EZ / TR_W keep its grid): postponed by the user 2026-11-19 -- add 8000.0 back
TR_TAU = (0.3387, 1.0)
TR_EZ = {2000.0: np.round(np.arange(1.20, 1.50001, 0.002), 4), 4000.0: np.round(np.arange(1.20, 1.50001, 0.002), 4),
         8000.0: np.round(np.arange(1.24, 1.46001, 0.004), 4)}
TR_EZB = np.round(np.arange(1.3450, 1.38251, 0.0025), 4)          # fine bulk grid around E_c
TR_W = {2000.0: 6, 4000.0: 6, 8000.0: 4}                          # workers (L 8: ~4x the memory per worker)


def trans_name(L, tau):
    return f'scharf_absTR_mixed_tau{tau:g}_L{L / 1000:g}'


# %% [trans] --- 1. SCANS: bulk modes on a fine grid around E_c; E = 0 transport for L 2 / 4 / 8 um x tau 0.3387 / 1 ---
if run_part('trans'):
    from study_mixed import osc_point, QM
    TRB = st.run_scan(partial(osc_point, QM), {'E_Z': TR_EZB}, 'scharf_absTR_bulk_mixed_a10', n_workers=6, chunksize=1)
    print("trans: bulk done", flush=True)
    TRV = {}
    for L in TR_L:
        for tau in TR_TAU:
            TRV[(L, tau)] = st.run_scan(partial(st.transport_point, junction('mixed', 0.7, tau, L=L)), {'E_Z': TR_EZ[L]},
                                        trans_name(L, tau), n_workers=TR_W[L], chunksize=1)
            print(f"trans: L {L / 1000:g} um, tau {tau:g} done", flush=True)


# %% [trans] --- 2. kappa, G near E_c vs L; tau 1 against (1/2) sech^2(L / xi_M) (bulk xi_M, no fit) ---
if run_part('trans'):
    OMc = np.load(st.DATA / 'scharf_absOM_modes_mixed_a10.npz')
    ezb = np.r_[OMc['axis_E_Z'], TR_EZB]
    xib, gpb, Qb = np.r_[OMc['values'][:, 0], TRB[:, 0]], np.r_[OMc['values'][:, 6], TRB[:, 6]], np.r_[OMc['values'][:, 7], TRB[:, 7]]
    o = np.argsort(ezb)
    ezb, xib, gpb, Qb = ezb[o], xib[o], gpb[o] * 1e3, Qb[o]
    j = np.flatnonzero(np.diff(Qb) != 0)
    EC = float((ezb[j[-1]] + ezb[j[-1] + 1]) / 2) if j.size else np.nan
    print(f"\n  [trans] bulk at phi = pi: E_c = {EC:.4f} meV (Q flips between {ezb[j[-1]]:.4f} and {ezb[j[-1] + 1]:.4f}); "
          "gap / xi_M just below: " + ", ".join(f"{e:.4f}: {g:.2f} ueV / {x / 1000:.1f} um" for e, g, x in
                                           zip(ezb[(ezb > 1.33) & (Qb < 0)], gpb[(ezb > 1.33) & (Qb < 0)],
                                               xib[(ezb > 1.33) & (Qb < 0)])), flush=True)
    xi_at = lambda e: np.exp(np.interp(e, ezb[Qb < 0], np.log(xib[Qb < 0])))
    fig, axs = plt.subplots(len(TR_L), len(TR_TAU), figsize=(14.0, 10.0), sharex=True, sharey=True, layout='constrained')
    print("  per (L, tau): kappa max within 0.1 meV below E_c (where, G / kappa); kappa at the last field below E_c; "
          "neutral (|G| < 0.1 kappa) kappa > 0.45 nearest E_c; kappa above E_c (trivial, gapless) median", flush=True)
    for r, L in enumerate(TR_L):
        for c_, tau in enumerate(TR_TAU):
            V, e = TRV[(L, tau)], TR_EZ[L]
            k, G = V[:, 0], V[:, 1]
            near = (e > EC - 0.1) & (e < EC)
            i = np.flatnonzero(near)[np.argmax(k[near])]
            neu = (np.abs(G) < 0.1 * np.maximum(k, 1e-12)) & (k > 0.45) & (e < EC)
            last = np.flatnonzero(e < EC)[-1]
            print(f"    L {L / 1000:g} um, tau {tau:g}: max {k[i]:.3f} at {e[i]:.3f} (E_c - {(EC - e[i]) * 1e3:.0f} ueV, "
                  f"G/kappa {G[i] / k[i]:+.2f}); at {e[last]:.3f}: {k[last]:.3f} (G {G[last]:+.3f}); neutral > 0.45 nearest "
                  f"E_c: {e[neu][-1] if neu.any() else np.nan:.3f}; above E_c median {np.median(k[e > EC]):.3f} "
                  f"(|G| {np.median(np.abs(G[e > EC])):.3f})", flush=True)
            a = axs[r, c_]
            a.axvspan(EC, e[-1] + 0.01, color='#e4e3df', lw=0, zorder=0)
            a.axvline(EC, color='#52514e', lw=1.0, ls='--')
            a.plot(e, k, color='#1b1b1b', lw=1.5, label=r'$\kappa/\kappa_0$')
            a.plot(e, np.abs(G), color='#d03b3b', lw=1.1, label=r'$|G|/G_0$')
            if tau == 1.0:
                eb = e[e < EC]
                a.plot(eb, 0.5 / np.cosh(L / xi_at(eb)) ** 2, color='#2a78d6', lw=1.3, ls='--',
                       label=r'$\frac{1}{2}\,\mathrm{sech}^2(L/\xi_M)$, bulk $\xi_M$')
            a.axhline(0.5, color='#52514e', lw=0.8, ls=':')
            a.set_title(rf'L = {L / 1000:g} $\mu$m, $\tau$ = {tau:g}', fontsize=10.5, loc='left')
            a.set_xlim(TR_EZ[2000.0][0], TR_EZ[2000.0][-1])
            a.set_ylim(0, 1.05)
            a.grid(color='#e4e3df', lw=0.6)
            if r == len(TR_L) - 1:
                a.set_xlabel(r'$E_Z$ (meV), $\phi = \pi$')
    axs[0, 1].legend(fontsize=8.5, frameon=False, loc='upper left')
    axs[0, 0].text(EC + 0.005, 0.95, 'trivial,\ngapless', fontsize=8.5, va='top', color='#52514e')
    fig.suptitle(rf'Long junctions at the upper transition (mixed SOC, $E = 0$, T = 0): bulk $E_c$ = {EC:.3f} meV (dashed)',
                 fontsize=11.5)
    st.save_fig(fig, 'trans_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/trans_mixed.png", flush=True)


# =====================================================================================================================
# PART [barr] (2026-11-19) -- CHECK: does a PHYSICAL barrier contact change the [nh] relations?  Equal probes with a
# one-site barrier lam_barr (meV nm; tc_barr_transport = 1, every bond a bulk bond; folded exactly into BOTH probe
# self-energies, junction.py) instead of one weak bond tau.  Same E_Z grid and phase as [ep8] (the bond reference).
# Tested: (1) det r_L = 1 - 2 Gamma_L Gamma_R / |l1 l2|; (2) plateau window vs sqrt(Gamma_L Gamma_R); (3) where things
# sit in E_Z (the probe's level shift).  Unequal barriers are not possible (Params has ONE barrier for both probes).
# Caches scharf_absBARR_*; figure plots/barr_mixed.png.
BARR_LAM = (260.0, 220.0, 180.0, 150.0, 120.0)                    # meV nm; [a]: lam 240 ~ tau 0.3387 + 20 % per channel
BARR_COL = ('#0b2a4a', '#1d4f8a', '#2a78d6', '#6fa8e6', '#9b59d0')


def barr_q(lam):
    """The thesis junction (mixed, mu_C 0.7, L 2 um, a 10) with a one-site barrier lam (meV nm) in both probes."""
    return st.replace(junction('mixed', 0.7, 1.0), lam_barr=float(lam))


def barr_field(E_Z, i):
    """At one field, barrier BARR_LAM[i]: pair poles with both probes and with the left probe only (EP_K Re, EP_K Im each,
    k = 16 as [ep8]), and the E = 0 transport row (kappa, G, Andreev, det r_L, n_bad)."""
    q = barr_q(BARR_LAM[int(i)])
    P2, _ = st.nh_spectrum(q, float(E_Z), [EP8_PHI], 1.0, 1.0, k=16)
    P1, _ = st.nh_spectrum(q, float(E_Z), [EP8_PHI], 1.0, 0.0, k=16)
    t = st.transport_phis(q, float(E_Z), [EP8_PHI])[0]
    return np.r_[P2[0, :EP_K].real, P2[0, :EP_K].imag, P1[0, :EP_K].real, P1[0, :EP_K].imag, t]


# %% [barr] --- 1. SCAN: E_Z 0.80-1.10 (step 5 ueV) x five barriers ---
if run_part('barr'):
    BAV = st.run_scan(barr_field, {'E_Z': EP8_EZ, 'i': np.arange(len(BARR_LAM))}, 'scharf_absBARR_mixed_a10_phi1.7417',
                      n_workers=6, chunksize=1)                    # (n_EZ, n_lam, 4 EP_K + 5)
    print("barr: scan done", flush=True)


# %% [barr] --- 2. the [nh] relations with barrier contacts vs the bond reference ([ep8]) ---
if run_part('barr'):
    E = EP8_EZ * 1e3
    F8 = np.load(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz')['values']
    F8 = F8.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
    IDX = {c: n for n, c in enumerate(EP8_PAIRS)}
    EC = ep8_crossings(E, np.abs(F8[:, IDX[(0.0, 0.0)], 0]))
    K = EP_K
    G1b = {t: -(F8[:, IDX[(t, 0.0)], K] + F8[:, IDX[(t, 0.0)], K + 1]) for t in EP8_T}
    BOND = {}
    for c in EP8_PAIRS:
        if c[1] == 0:
            continue
        pp = (F8[:, IDX[c], 0] + 1j * F8[:, IDX[c], K]) * (F8[:, IDX[c], 1] + 1j * F8[:, IDX[c], K + 1])
        t = F8[:, IDX[c], 2 * K + 1:]
        BOND[c] = dict(q=G1b[c[0]] * G1b[c[1]] / np.abs(pp), dr=t[:, 3], k=t[:, 0], G=t[:, 1],
                       s=np.sqrt(G1b[c[0]] * G1b[c[1]]))
    BAR = {}
    for i, lam in enumerate(BARR_LAM):
        V = BAV[:, i]
        l2a, l2b = V[:, 0] + 1j * V[:, K], V[:, 1] + 1j * V[:, K + 1]
        g1 = -(V[:, 3 * K] + V[:, 3 * K + 1])                         # one-probe pair width (left barrier only)
        t = V[:, 4 * K:]
        BAR[lam] = dict(q=g1 * g1 / np.abs(l2a * l2b), dr=t[:, 3], k=t[:, 0], G=t[:, 1], s=g1)
    print(f"\n  [barr] equal probes, one-site barrier lam_barr (tc_barr 1), phi {EP8_PHI / np.pi:.4f} pi, E_Z 800-1100 ueV; "
          f"failed lead solves {int(np.nansum(BAV[:, :, -1]))}", flush=True)
    print("    (1) det r = 1 - 2q: max |dev| -- bond pairs (all, w/o 0.8/0.8) "
          f"{max(np.abs(b['dr'] - (1 - 2 * b['q'])).max() for c, b in BOND.items() if c != (0.8, 0.8)):.3f}", flush=True)
    dmin_of = lambda b: [E[np.argmin(np.where(np.abs(E - ec) < 25, b['k'], np.inf))] for ec in EC]
    for lam, b in BAR.items():
        w = [ep9_window(b['k'], b['G'], tol)[0] for tol in (0.01, 0.02, 0.03)]
        print(f"    lam {lam:5.0f}: Gamma_1 {b['s'].min():.2f}-{b['s'].max():.2f} ueV (at the crossings "
              + "/".join(f"{np.interp(ec, E, b['s']):.2f}" for ec in EC) + "); det r vs 1-2q max |dev| "
              f"{np.abs(b['dr'] - (1 - 2 * b['q'])).max():.3f}; windows {w[0]:.0f}/{w[1]:.0f}/{w[2]:.0f} ueV; kappa max "
              f"{b['k'].max():.3f}, max |G| {np.abs(b['G']).max():.3f}; kappa minima near the crossings at "
              + "/".join(f"{d:.0f}" for d in dmin_of(b)) + " ueV", flush=True)
    for c in ((0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6)):
        b = BOND[c]
        print(f"    bond {c[0]:g}/{c[1]:g}: Gamma_1 at the crossings " + "/".join(f"{np.interp(ec, E, b['s']):.2f}" for ec in EC)
              + "; windows " + "/".join(f"{ep9_window(b['k'], b['G'], tol)[0]:.0f}" for tol in (0.01, 0.02, 0.03))
              + " ueV; kappa minima near the crossings at " + "/".join(f"{d:.0f}" for d in dmin_of(b)) + " ueV", flush=True)
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 10.0), layout='constrained')
    a = axs[0, 0]
    for c, b in BOND.items():
        if c != (0.8, 0.8):
            a.plot(b['q'], b['dr'], '.', ms=2, color='#c9c8c3')
    for (lam, b), col in zip(BAR.items(), BARR_COL):
        a.plot(b['q'], b['dr'], 'o', ms=3, color=col, label=rf'barrier $\lambda$ = {lam:g} meV nm')
    a.plot([0, 1], [1, -1], color='#1b1b1b', lw=1.0)
    a.plot([], [], '.', color='#c9c8c3', ms=6, label='bond contacts ([ep8])')
    a.set_xlabel(r'$q = \Gamma_L\Gamma_R / |\lambda_1\lambda_2|$')
    a.set_ylabel(r'det $r_L$ at $E$ = 0')
    a.set_title(r'(a)  det $r = 1 - 2q$ with barrier contacts', fontsize=10, loc='left')
    a.legend(fontsize=8, frameon=False)
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[0, 1]
    for c, b in BOND.items():
        a.plot(np.median(b['s']), ep9_window(b['k'], b['G'], 0.02)[0], 'o', ms=6, color='#c9c8c3', mec='w')
    for (lam, b), col in zip(BAR.items(), BARR_COL):
        a.plot(np.median(b['s']), ep9_window(b['k'], b['G'], 0.02)[0], '*', ms=14, color=col, mec='w')
    a.set_xscale('log')
    a.set_xlabel(r'$\sqrt{\Gamma_L\Gamma_R}$ (median over $E_Z$, $\mu$eV)')
    a.set_ylabel(r'longest $|\kappa - 1/2| < 0.02$ window ($\mu$eV)')
    a.set_title('(b)  plateau length vs the widths: barriers (stars) vs bonds (grey)', fontsize=10, loc='left')
    a.grid(color='#e4e3df', lw=0.6, which='both')
    a = axs[1, 0]
    for (lam, b), col in zip(BAR.items(), BARR_COL):
        a.plot(EP8_EZ, b['k'], color=col, lw=1.6, label=rf'barrier $\lambda$ {lam:g}')
    for c, ls in (((0.4, 0.4), '--'), ((0.5, 0.5), ':')):
        a.plot(EP8_EZ, BOND[c]['k'], color='#52514e', lw=1.4, ls=ls, label=f'bond {c[0]:g}/{c[1]:g}')
    for ec in EC:
        a.axvline(ec * 1e-3, color='#52514e', lw=0.8, ls=':')
    a.axhline(0.5, color='#52514e', lw=0.8)
    a.set_xlabel(r'$E_Z$ (meV)')
    a.set_ylabel(r'$\kappa/\kappa_0$, $E$ = 0, T = 0')
    a.set_title(r'(c)  $\kappa(E_Z)$: barrier vs bond contacts', fontsize=10, loc='left')
    a.legend(fontsize=7.5, frameon=False, ncol=2, loc='lower center')
    a.grid(color='#e4e3df', lw=0.6)
    a = axs[1, 1]
    for (lam, b), col in zip(BAR.items(), BARR_COL):
        a.plot(EP8_EZ, b['s'], color=col, lw=1.6)
    for t in (0.3, 0.4, 0.5, 0.6):
        a.plot(EP8_EZ, G1b[t], color='#52514e', lw=1.0, ls='--')
        a.text(EP8_EZ[-1], G1b[t][-1], f' bond {t:g}', fontsize=7.5, va='center', color='#52514e')
    a.set_yscale('log')
    a.set_xlabel(r'$E_Z$ (meV)')
    a.set_ylabel(r'one-probe width $\Gamma_1$ ($\mu$eV)')
    a.set_title(r'(d)  $\Gamma_1(E_Z)$: barriers (colour) vs bonds (dashed)', fontsize=10, loc='left')
    a.grid(color='#e4e3df', lw=0.6, which='both')
    fig.suptitle(rf'Barrier vs bond probe contacts (equal probes, mixed SOC, $\phi$ = {EP8_PHI / np.pi:.4f}$\pi$, full model)',
                 fontsize=11.5)
    st.save_fig(fig, 'barr_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/barr_mixed.png", flush=True)


# =====================================================================================================================
# PART [chern] (2026-11-19) -- STUDY: is the Chern-ribbon kappa ~ 1/2 of study_mixed [combine] B2 a PLATEAU?  B2 (single
# points, a 20): an OUT-OF-PLANE Zeeman Vz >= 1.0 meV in the SC ribbons only makes them Chern superconductors (B1: C = +1
# from Vz ~ 0.969, bulk gap 31 / 68 / 43 / 33 ueV at Vz 1.0 / 1.05 / 2 / 3); with a DEPLETED centre (mu_C -0.2) and a
# transparent probe (tau 1) kappa = 0.45-0.50, |G| <= 0.002 at phi = pi AND 0, in-plane E_Z 0 / 0.3 / 0.93.  A propagating
# (chiral) Majorana along the junction would make that plateau protected by the ribbons' bulk gap -- the regime the
# advisor expects.  Here: (Vz, mu_C) maps at phi = pi / 0 for tau 1 and 0.5 (mixed SOC, a 20, L 2 um, in-plane E_Z 0).
# Caches scharf_absCH_*; figure plots/chern_maps_mixed.png.
CH_VZ = np.array([0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00])   # meV
CH_MU = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7])                                       # meV
CH_TAU = (1.0, 0.5)
CH_A = 20.0
CH_VZC = 0.969                                                    # meV, Chern transition of the ribbon bulk (B1)


def chern_q(mu_C, tau, L=2000.0, a=CH_A):
    """The thesis junction (mixed SOC, W 100, t_SN 1, mu_S 1) at lattice constant a, centre mu_C, probe bond tau."""
    al, be = st.mixed_soc(16.0, 0.15 * np.pi)
    return st.replace(st.case(100.0, L=L, alpha=al, beta=be, a=a), tc=1.0, mu_C=float(mu_C), tc_barr_transport=float(tau))


def chern_point(Vz, mu_C, tau, E_Z=0.0, L=2000.0, a=CH_A, E=0.0):
    """[combine] B2 generalised: out-of-plane Zeeman Vz (meV) in the ribbons only, in-plane E_Z in the strip, energy E
    (meV).  -> [kappa, G, local Andreev] at phi = pi, the same at phi = 0, det r(pi) (E = 0 only, else nan), failed lead
    solves.  Uses the user's RGFFourTerminal / FourTerminalJunction unchanged."""
    q = chern_q(mu_C, tau, L, a)
    n0 = myf_leads_count()
    p = st.replace(q.params_transport(E_Z=float(E_Z)), Bz_s=float(Vz) / q.t)
    rgf = st.myf.RGFFourTerminal(st.myf.FourTerminalJunction(p), np.array([float(E) / q.t]))
    out = []
    for ph in (np.pi, 0.0):
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(ph, 'left').items()}
        out += [c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local']]
    return np.array(out + [st.det_r(rgf, np.pi) if E == 0 else np.nan, myf_leads_count() - n0])


def myf_leads_count():
    return st.myf.leads.N_UNRELIABLE


def chern_grid_point(i, j, t):
    """chern_point on the (CH_VZ[i], CH_MU[j], CH_TAU[t]) grid, in-plane E_Z 0, L 2 um."""
    return chern_point(CH_VZ[int(i)], CH_MU[int(j)], CH_TAU[int(t)])


# %% [chern] --- 1. (Vz, mu_C) MAPS, tau 1 and 0.5, phi = pi / 0 ---
if run_part('chern'):
    CHV = st.run_scan(chern_grid_point, {'i': np.arange(len(CH_VZ)), 'j': np.arange(len(CH_MU)), 't': np.arange(len(CH_TAU))},
                      'scharf_absCH_maps_mixed_a20_L2', n_workers=6, chunksize=1)     # (n_Vz, n_mu, n_tau, 8)
    print("chern: maps done", flush=True)


# %% [chern] --- 2. WHERE IS kappa ~ 1/2 NEUTRAL, and is it a plateau along Vz / mu_C? ---
if run_part('chern'):
    plat = lambda k, G: (np.abs(k - 0.5) < 0.03) & (np.abs(G) < 0.1 * np.maximum(k, 1e-12))
    print(f"\n  [chern] mixed SOC, a 20, L 2 um, in-plane E_Z 0; ribbons Chern for Vz > {CH_VZC} meV.  Plateau = |kappa - 1/2| "
          "< 0.03 and |G| < 0.1 kappa.  Per (tau, mu_C): Vz with a plateau at phi = pi / at phi = 0; failed lead solves "
          f"{int(np.nansum(CHV[..., 7]))}", flush=True)
    for t, tau in enumerate(CH_TAU):
        for j, mu in enumerate(CH_MU):
            pp, p0 = plat(CHV[:, j, t, 0], CHV[:, j, t, 1]), plat(CHV[:, j, t, 3], CHV[:, j, t, 4])
            print(f"    tau {tau:g}, mu_C {mu:+.1f}: pi {''.join('#' if b else '.' for b in pp)}  0 {''.join('#' if b else '.' for b in p0)}"
                  f"   kappa(pi) {CHV[:, j, t, 0].min():.2f}-{CHV[:, j, t, 0].max():.2f}, max |G| {np.abs(CHV[:, j, t, 1]).max():.3f}",
                  flush=True)
    print("    (columns = Vz " + " ".join(f"{v:g}" for v in CH_VZ) + " meV)", flush=True)
    fig, axs = plt.subplots(len(CH_TAU), 3, figsize=(15.0, 8.6), layout='constrained', sharex=True, sharey=True)
    X, Y = np.arange(len(CH_VZ)), np.arange(len(CH_MU))
    for t, tau in enumerate(CH_TAU):
        for c_, (col, lab) in enumerate(((0, r'$\kappa/\kappa_0$, $\phi = \pi$'), (3, r'$\kappa/\kappa_0$, $\phi = 0$'),
                                         (1, r'$|G|/G_0$, $\phi = \pi$'))):
            a = axs[t, c_]
            Z = np.abs(CHV[:, :, t, col]).T
            im = a.pcolormesh(X, Y, Z, shading='nearest', cmap='Blues', vmin=0, vmax=1.0 if col != 1 else 0.5)
            if col != 1:
                m = plat(CHV[:, :, t, col], CHV[:, :, t, col + 1]).T
                yy, xx = np.nonzero(m)
                a.plot(xx, yy, 's', ms=7, mfc='none', mec='#eb6834', mew=1.6)
            a.axvline(np.interp(CH_VZC, CH_VZ, X), color='#52514e', lw=1.0, ls='--')
            a.set_title(rf'$\tau$ = {tau:g}: ' + lab, fontsize=10, loc='left')
            fig.colorbar(im, ax=a, pad=0.01)
    for a in axs[-1]:
        a.set_xticks(X, [f'{v:g}' for v in CH_VZ], rotation=60, fontsize=8)
        a.set_xlabel(r'$V_z$ in the SC ribbons (meV, out of plane)')
    for a in axs[:, 0]:
        a.set_yticks(Y, [f'{v:g}' for v in CH_MU])
        a.set_ylabel(r'$\mu_C$ (meV)')
    fig.suptitle(r'Chern ribbons (out-of-plane $V_z$ in the SCs only), mixed SOC, a 20, L 2 $\mu$m, $E_Z^{\parallel}$ = 0.  '
                 r'Orange squares: $|\kappa - 1/2| < 0.03$, $|G| < 0.1\kappa$; dashed: ribbon Chern transition', fontsize=11)
    st.save_fig(fig, 'chern_maps_mixed', dpi=150)
    plt.close(fig)
    print("figure: plots/chern_maps_mixed.png", flush=True)


# =====================================================================================================================
# PART [clean] (2026-11-19) -- PLOTS ONLY: the key EP / kappa results redrawn for a meeting, ONE message per figure,
# one or two panels each, large fonts.  From the [ep7], [ep8] and study_mixed [deck] caches (same data as [nh]).
# Figures plots/clean/c1_ep_poles.png ... c8_equal_bonds_T.png.
CL_OUT, CL_IN, CL_INK, CL_MUT, CL_GRID = '#2a78d6', '#eb6834', '#1b1b1b', '#6b6a66', '#e4e3df'
CL_RC = {'font.size': 12, 'axes.titlesize': 11, 'axes.labelsize': 12.5, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
         'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8986'}


def cl_head(fig, ax, message, sub):
    """The figure's message as a left-aligned headline, the setting as a muted line under it."""
    fig.suptitle(message, x=0.01, ha='left', fontsize=14, fontweight='semibold')
    ax.set_title(sub, loc='left', fontsize=10.5, color=CL_MUT)


def cl_save(fig, name):
    (st.PLOTS / 'clean').mkdir(exist_ok=True)
    st.save_fig(fig, f'clean/{name}', dpi=170)
    plt.close(fig)
    print(f"figure: plots/clean/{name}.png", flush=True)


if run_part('clean'):
    with plt.rc_context(CL_RC):
        E = EP8_EZ * 1e3
        F8 = np.load(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz')['values']
        F8 = F8.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
        IDX = {c: n for n, c in enumerate(EP8_PAIRS)}
        K = EP_K
        s0 = np.abs(F8[:, IDX[(0.0, 0.0)], 0])                         # lossless |eps|
        EC = ep8_crossings(E, s0)
        G1 = {t: -(F8[:, IDX[(t, 0.0)], K] + F8[:, IDX[(t, 0.0)], K + 1]) for t in EP8_T}
        PL = lambda c: (F8[:, IDX[c], 0] + 1j * F8[:, IDX[c], K], F8[:, IDX[c], 1] + 1j * F8[:, IDX[c], K + 1])
        TR = lambda c: F8[:, IDX[c], 2 * K + 1:]
        SEGS = {c: ep8_segments(E, (((PL(c)[0] - PL(c)[1]) / 2) ** 2).real) for c in EP8_PAIRS if c != (0.0, 0.0)}

        def ins_mask(c):
            m = np.zeros(len(E), bool)
            for s, e, _ in SEGS[c]:
                m |= (E >= s) & (E <= e)
            return m

        def shade(ax, c):
            for s, e, _ in SEGS[c]:
                ax.axvspan(max(s, E[0]) * 1e-3, min(e, E[-1]) * 1e-3, color='#f7dccd', lw=0, zorder=0)

        # --- C1: what an EP segment is ---
        c = (0.6, 0.4)
        a_, b_ = PL(c)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True, layout='constrained')
        for ax in (ax1, ax2):
            shade(ax, c)
            ax.grid(color=CL_GRID, lw=0.6)
        ax1.plot(EP8_EZ, a_.real, 'o', ms=3.6, color=CL_OUT)
        ax1.plot(EP8_EZ, b_.real, 'o', ms=3.6, color=CL_OUT)
        ax1.plot(EP8_EZ, s0, color=CL_MUT, lw=1.0, ls='--')
        ax1.plot(EP8_EZ, -s0, color=CL_MUT, lw=1.0, ls='--')
        ax1.text(0.957, 7.6, r'$\pm\varepsilon$ without probes', fontsize=10, color=CL_MUT, ha='center')
        ax1.set_ylabel(r'pole position Re $\lambda$ ($\mu$eV)')
        ax2.plot(EP8_EZ, -a_.imag, 'o', ms=3.6, color=CL_IN)
        ax2.plot(EP8_EZ, -b_.imag, 'o', ms=3.6, color=CL_IN)
        ax2.set_ylabel(r'pole width $-$Im $\lambda$ ($\mu$eV)')
        ax2.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        ax2.set_ylim(0, None)
        seg0 = SEGS[c][0]
        xm = 0.5 * (seg0[0] + seg0[1]) * 1e-3
        ax2.text(xm, 2.0, 'between EPs:\nboth at E = 0,\none narrow + one broad', ha='center', fontsize=10.5)
        ax2.text(0.943, 2.0, 'split pair:\n' + r'$\pm E_0$, equal widths', ha='center', fontsize=10.5)
        cl_head(fig, ax1, 'Between the EPs the Majorana pair is pinned at zero energy',
                rf'probes ($\tau_L$, $\tau_R$) = (0.6, 0.4), $\phi$ = {EP8_PHI / np.pi:.2f}$\pi$; shaded: between EPs')
        cl_save(fig, 'c1_ep_poles')

        # --- C2: segment length = width difference / slope of the splitting ---
        fig, ax = plt.subplots(figsize=(7.2, 5.4), layout='constrained')
        X, Y = [], []
        for ec, mk, lab in zip(EC, ('o', 's'), (rf'crossing at {EC[0]:.0f} $\mu$eV', rf'crossing at {EC[1]:.0f} $\mu$eV')):
            i = int(np.searchsorted(E, ec))
            sl = 0.5 * (abs(s0[i + 2] - s0[i]) / (E[i + 2] - E[i]) + abs(s0[i - 1] - s0[i - 3]) / (E[i - 1] - E[i - 3]))
            xs, ys = [], []
            for c in EP8_PAIRS[8:]:
                L_, sg = ep8_seg_near(SEGS[c], ec)
                if c[0] != c[1] and sg is not None and not sg[2] and L_ > 5:
                    xs.append(abs(np.interp(ec, E, G1[c[0]]) - np.interp(ec, E, G1[c[1]])) / sl)
                    ys.append(L_)
            ax.plot(xs, ys, mk, ms=8, mfc=CL_OUT, mec='w', mew=1.0, ls='', label=lab)
            X += xs
            Y += ys
        xx = np.geomspace(min(X) * 0.7, max(X) * 1.4, 20)
        fit = float(np.median(np.array(Y) / np.array(X)))
        ax.plot(xx, fit * xx, color=CL_INK, lw=1.2)
        ax.text(xx[13], fit * xx[13] * 1.25, rf'length = {fit:.2f} $\times$ prediction', fontsize=10.5, rotation=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\Gamma_L - \Gamma_R|\, /\, |d\varepsilon/dE_Z|$ at the crossing ($\mu$eV)')
        ax.set_ylabel(r'EP segment length along $E_Z$ ($\mu$eV)')
        ax.legend(frameon=False, loc='upper left')
        ax.grid(color=CL_GRID, lw=0.6, which='both')
        cl_head(fig, ax, 'An EP segment is as long as the probe-width difference allows',
                'every unequal probe pair, both zero crossings of the splitting; no fit')
        cl_save(fig, 'c2_ep_length')

        # --- C3: T = 0 plateau length vs sqrt(Gamma_L Gamma_R), in / out of segments ---
        fig, ax = plt.subplots(figsize=(7.2, 5.4), layout='constrained')
        for c in EP8_PAIRS:
            if c[1] == 0:
                continue
            k_, G_ = TR(c)[:, 0], TR(c)[:, 1]
            w, _, s = ep9_window(k_, G_, 0.02)
            ss = s if s is not None else slice(None)
            gm = float(np.median(np.sqrt(G1[c[0]][ss] * G1[c[1]][ss])))
            inside = s is not None and ins_mask(c)[s].mean() >= 0.5
            ax.plot(gm, w, 'o', ms=9, mfc=CL_IN if inside else CL_OUT, mec='w', mew=1.0)
            if c in ((0.5, 0.4), (0.8, 0.2)):
                ax.annotate(f'({c[0]:g}, {c[1]:g})', (gm, w), xytext=(-10 if c == (0.5, 0.4) else 10, 0),
                            textcoords='offset points', ha='right' if c == (0.5, 0.4) else 'left', va='center', fontsize=10.5)
        ax.plot([], [], 'o', ms=9, mfc=CL_OUT, mec='w', ls='', label='plateau outside EP segments')
        ax.plot([], [], 'o', ms=9, mfc=CL_IN, mec='w', ls='', label='plateau between EPs')
        ax.set_xscale('log')
        ax.set_xlabel(r'geometric mean of the probe widths $\sqrt{\Gamma_L\Gamma_R}$ ($\mu$eV)')
        ax.set_ylabel(r'longest $|\kappa - \frac{1}{2}| < 0.02$ window along $E_Z$ ($\mu$eV)')
        ax.set_ylim(-4, 96)
        ax.legend(frameon=False, loc='center left')
        ax.grid(color=CL_GRID, lw=0.6, which='both')
        cl_head(fig, ax, r'At T = 0 the $\kappa \approx \frac{1}{2}$ plateau only cares about $\sqrt{\Gamma_L\Gamma_R}$',
                rf'26 probe pairs along $E_Z$ (0.80-1.10 meV), $\phi$ = {EP8_PHI / np.pi:.2f}$\pi$, E = 0')
        cl_save(fig, 'c3_plateau_vs_widths')

        # --- C4 + C5: finite energy / temperature ([ep7]) ---
        EX = {}
        THS = []
        for c in EP7_CFG:
            V = np.load(st.DATA / f'scharf_absEP7_ch_mixed_a10_EZ{EP_EZ:g}_tauL{c[0]:g}_R{c[1]:g}.npz')['values']
            Kc = ep7_kappaE(V)
            xp, P, btw = ep7_poles(*c)
            j = np.array([np.argmin(np.abs(xp - v)) for v in EP_PH / np.pi])
            P, btw = P[j], btw[j]
            lmin = np.min(np.abs(P[:, :2]), axis=1)
            th = np.array([nh_thalf(V, n, Kc[0, n]) if Kc[0, n] > 0.05 else np.nan for n in range(len(EP_PH))])
            THS.append((lmin, th, btw))
            EX[c] = (Kc, lmin, btw)
        fig, ax = plt.subplots(figsize=(7.6, 5.2), layout='constrained')
        for c, inside, col, txt in (((0.4, 0.6), False, CL_OUT, 'outside the EP segments'),
                                    ((0.5, 0.1), True, CL_IN, 'between EPs')):
            Kc, lmin, btw = EX[c]
            n = (int(np.flatnonzero(btw)[np.argmax(Kc[0][btw])]) if inside else int(np.argmin(np.abs(EP_PH / np.pi - 1.5))))
            ax.plot(EP7_E[1:], Kc[1:, n], color=col, lw=2.6)
            ax.text(0.012, Kc[1, n] + 0.015, rf'{txt}: ($\tau_L$, $\tau_R$) = ({c[0]:g}, {c[1]:g}), $\phi$ = {EP_PH[n] / np.pi:.2f}$\pi$',
                    color=col, fontsize=10.5, va='bottom')
        for T, ls in ((10.0, '--'), (20.0, ':')):
            e_ = st.kT_of_mK(T) * 1e3
            ax.axvline(e_, color=CL_MUT, lw=1.1, ls=ls)
            ax.text(e_ * 1.07, 0.03, rf'$k_BT$ at {T:g} mK', rotation=90, color=CL_MUT, fontsize=10, va='bottom')
        ax.set_xscale('log')
        ax.set_xlim(0.01, 70)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel(r'quasiparticle energy $E$ ($\mu$eV)')
        ax.set_ylabel(r'$\kappa(E)/\kappa_0$')
        ax.grid(color=CL_GRID, lw=0.6, which='both')
        cl_head(fig, ax, r'Between the EPs $\kappa(E)$ collapses below 1 $\mu$eV',
                rf'$E_Z$ = {EP_EZ:g} meV, T = 0; a temperature averages $\kappa(E)$ over a few $k_BT$')
        cl_save(fig, 'c4_kappa_vs_energy')
        fig, ax = plt.subplots(figsize=(7.2, 5.4), layout='constrained')
        rat = []
        for lmin, th, btw in THS:
            ax.plot(lmin[~btw], th[~btw], 'o', ms=4.5, color=CL_OUT, alpha=0.8)
            ax.plot(lmin[btw], th[btw], 'o', ms=4.5, color=CL_IN, alpha=0.9)
            rat.append(st.kT_of_mK(th) * 1e3 / lmin)
        ft = float(np.nanmedian(np.concatenate(rat)))
        ll = np.geomspace(0.1, 30, 20)
        ax.plot(ll, ft * ll / (st.kT_of_mK(1.0) * 1e3), color=CL_INK, lw=1.2)
        ax.text(3.0, ft * 3.0 / (st.kT_of_mK(1.0) * 1e3) * 0.42, rf'$k_BT_{{1/2}} = {ft:.2f}\times$ distance', fontsize=10.5)
        ax.plot([], [], 'o', ms=7, color=CL_OUT, ls='', label='outside the EP segments')
        ax.plot([], [], 'o', ms=7, color=CL_IN, ls='', label='between EPs')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'distance of the nearest pole from E = 0 ($\mu$eV)')
        ax.set_ylabel(r'temperature where $\kappa$ halves, $T_{1/2}$ (mK)')
        ax.legend(frameon=False, loc='upper left')
        ax.grid(color=CL_GRID, lw=0.6, which='both')
        cl_head(fig, ax, 'The nearest pole sets the temperature where $\\kappa$ halves',
                rf'5 probe pairs x 121 phases, $E_Z$ = {EP_EZ:g} meV, full NEGF $\kappa(T)$')
        cl_save(fig, 'c5_temperature_rule')

        # --- C6 + C7: the two kappa = 1/2 types ---
        PRS = {(0.5, 0.5): ('#2a78d6', 'lobe top just below'), (0.5, 0.4): ('#1baf7a', 'lobe top on the line'),
               (0.3, 0.3): ('#d94f8a', 'lobe top far above')}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True, layout='constrained')
        ax1.axhspan(-0.2, 0.2, color=CL_GRID, lw=0)
        ax2.axhspan(0.48, 0.52, color=CL_GRID, lw=0)
        for c, (col, lab) in PRS.items():
            ax1.plot(EP8_EZ, TR(c)[:, 3], color=col, lw=2.2, label=f'({c[0]:g}, {c[1]:g}): {lab}')
            ax2.plot(EP8_EZ, TR(c)[:, 0], color=col, lw=2.2)
        ax1.axhline(0, color=CL_MUT, lw=0.8)
        ax1.set_ylabel(r'det $r$ (E = 0)')
        ax1.set_ylim(-1.05, 1.05)
        ax1.legend(frameon=False, loc='lower center', fontsize=10)
        ax2.set_ylabel(r'$\kappa/\kappa_0$ (E = 0, T = 0)')
        ax2.set_ylim(0, 0.56)
        ax2.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        for ax in (ax1, ax2):
            ax.grid(color=CL_GRID, lw=0.6)
            for ec in EC:
                ax.axvline(ec * 1e-3, color=CL_MUT, lw=0.8, ls=':')
        cl_head(fig, ax1, r'$\kappa \approx \frac{1}{2}$ lasts longest when det $r$ peaks close to 0',
                r'grey bands: $|\det r| < 0.2$ $\Leftrightarrow$ $|\kappa - \frac{1}{2}| < 0.02$; dotted: splitting zeros')
        cl_save(fig, 'c6_two_types')
        fig, ax = plt.subplots(figsize=(7.2, 5.0), layout='constrained')
        dome = (E >= 850.0) & (E <= 1020.0)
        ax.axvspan(-0.2, 0.2, color=CL_GRID, lw=0)
        for c in EP8_PAIRS:
            if c[1] == 0:
                continue
            t_ = np.nanmax(TR(c)[dome, 3])
            w_ = longest_run(np.abs(TR(c)[dome, 0] - 0.5) < 0.02) * EP8_DE
            col = PRS.get(c, (CL_MUT, ''))[0]
            ax.plot(t_, w_, 'o', ms=9 if c in PRS else 7, mfc=col, mec='w', mew=1.0)
            if c == (0.8, 0.2):
                ax.annotate('(0.8, 0.2)', (t_, w_), xytext=(-9, 0), textcoords='offset points', ha='right', va='center',
                            fontsize=10)
        ax.set_xlabel(r'det $r$ at the top of the splitting lobe')
        ax.set_ylabel(r'longest $|\kappa - \frac{1}{2}| < 0.02$ window ($\mu$eV)')
        ax.grid(color=CL_GRID, lw=0.6)
        cl_head(fig, ax, r'The best plateaus have the lobe top at det $r \approx 0$',
                'every probe pair; coloured: the three pairs of the previous figure')
        cl_save(fig, 'c7_window_vs_lobe_top')

        # --- C8: equal probes at finite T along E_Z ([deck] cache) ---
        D = np.load(st.DATA / NH_DECK)
        VD, ED, EZD = D['values'], D['axis_E'], D['axis_E_Z']
        fig, ax = plt.subplots(figsize=(7.6, 5.0), layout='constrained')
        for T, col in zip(NH_T, NH_TCOL):
            kk = st.thermal_line(VD, st.kT_of_mK(T), ED, phase=0)[:, 0]
            ax.plot(EZD, kk, color=col, lw=2.4, label=f'{T:g} mK')
        ax.legend(frameon=False, loc='upper left', title='temperature')
        ax.axhline(0.5, color=CL_MUT, lw=0.8, ls='--')
        ax.set_ylim(0, 0.56)
        ax.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        ax.set_ylabel(r'$\kappa/\kappa_0$')
        ax.grid(color=CL_GRID, lw=0.6)
        cl_head(fig, ax, r'Equal probes: the $\kappa \approx \frac{1}{2}$ dome fades between 10 and 20 mK',
                rf'probe bonds 0.5 / 0.5, $\phi$ = 1.74$\pi$, full NEGF $\kappa(T)$')
        cl_save(fig, 'c8_equal_probes_T')


# =====================================================================================================================
# PART [dark] (2026-11-19) -- STUDY: does the junction follow San-Jose, Cayao, Prada & Aguado, Sci. Rep. 6, 21427 (2016)
# [arXiv:1409.7306]?  There, past an EP one pole's decay rate Gamma_0 -> 0 (a 'Majorana dark state') when (1) SOC at the
# contact, (2) the reservoir is a SINGLE-MODE, helical lead (B > mu_N), (3) the contact is transparent, (4) the
# superconductor is in the Andreev limit; poles = eigenvalues of H_S + Sigma_N(omega = 0) (as our nh_spectrum does for
# the probes).  In a 2x2 pair picture a rank-one (single-channel) loss gives l1 l2 = -eps^2 exactly, so the narrow width
# ~ eps^2 / broad width -> 0 as the coupling grows; a second channel or a second probe adds a floor.
# Our thesis probes: mu_N 3 meV, no field -> 4 open BdG channels (2 spins x e/h).  'helical' probe here = the normal
# continuation of the strip: mu = mu_C 0.7 meV + the strip's in-plane field E_Z at the same angle -> 2 open BdG channels
# (one spin-polarised mode; the field-free mu 0.7 lead is closed).  One probe decoupled (tau_R = 0) = the paper's single
# reservoir; tau_R 0.1 shows the floor from a second probe.  phi 1.7417 pi at a lossless crossing (840 ueV, eps 0.10),
# the lobe top (955, eps 7.5) and the other crossing (1025, eps 0.34).  Cache scharf_absDARK_*; plots/dark_mixed.png.
DK_TAU = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DK_EZ = (0.840, 0.955, 1.025)
DK_TR = (0.0, 0.1)
DK_LEADS = ('standard', 'helical')


def helical_lead(p):
    """Probe leads = the normal continuation of the strip: mu_C and the strip's in-plane field (same angle)."""
    return st.replace(p, mu_n=p.mu_c, Bxy_n=p.Bxy_c, theta_z_n=p.theta_z)


def dark_point(l, e, t, r):
    """Pair poles (EP_K Re, EP_K Im, overlap; k = 16) and the E = 0 transport row (kappa, G, local Andreev, det r_L,
    n_bad) for lead type DK_LEADS[l], field DK_EZ[e], left bond DK_TAU[t], right bond DK_TR[r] (0 = decoupled)."""
    lead = helical_lead if DK_LEADS[int(l)] == 'helical' else None
    tl, tr, EZ = DK_TAU[int(t)], DK_TR[int(r)], DK_EZ[int(e)]
    q = junction('mixed', 0.7, tl)
    P, O = st.nh_spectrum(q, EZ, [EP8_PHI], tl, tr, k=16, lead=lead)
    row = st.transport_phis(q, EZ, [EP8_PHI], tau_R=tr, lead=lead)[0]
    return np.r_[P[0, :EP_K].real, P[0, :EP_K].imag, O, row]


# %% [dark] --- 1. SCAN ---
if run_part('dark'):
    DKV = st.run_scan(dark_point, {'l': np.arange(len(DK_LEADS)), 'e': np.arange(len(DK_EZ)), 't': np.arange(len(DK_TAU)),
                                   'r': np.arange(len(DK_TR))}, 'scharf_absDARK_mixed_a10_phi1.7417', n_workers=6, chunksize=1)
    print("dark: scan done", flush=True)


# %% [dark] --- 2. narrow width vs coupling, standard vs helical, one vs two probes; |l1 l2| vs eps^2 ---
if run_part('dark'):
    F8 = np.load(st.DATA / f'scharf_absEP8_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz')['values']
    F8 = F8.reshape(len(EP8_EZ), len(EP8_PAIRS), EP8_NV)
    eps0 = {EZ: abs(F8[int(np.argmin(np.abs(EP8_EZ - EZ))), 0, 0]) for EZ in DK_EZ}      # lossless pair (0, 0)
    K = EP_K
    W = {}
    print(f"\n  [dark] phi {EP8_PHI / np.pi:.4f} pi; widths in ueV (narrow / broad), 'p' = pinned (both poles on Re = 0); "
          "|l1 l2| / eps_lossless^2 (= 1 for a rank-one, single-channel loss in the 2x2 picture); local Andreev (e^2/h); "
          f"failed lead solves {int(np.nansum(DKV[..., -1]))}", flush=True)
    for l, ld in enumerate(DK_LEADS):
        for r, tr in enumerate(DK_TR):
            for e, EZ in enumerate(DK_EZ):
                V = DKV[l, e, :, r]
                re, im = V[:, :2], V[:, K:K + 2]
                w = np.sort(-im, axis=1)
                pin = (np.abs(re) < 1e-3).all(axis=1) & (np.abs(im[:, 0] - im[:, 1]) > 1e-3)
                prod = np.abs((re[:, 0] + 1j * im[:, 0]) * (re[:, 1] + 1j * im[:, 1]))
                W[(ld, tr, EZ)] = (w, pin, prod)
                print(f"    {ld:8s} tau_R {tr:g}, E_Z {EZ * 1e3:4.0f} (eps {eps0[EZ]:.2f}):", flush=True)
                print("      tau_L      " + " ".join(f"{t:7.1f}" for t in DK_TAU), flush=True)
                print("      narrow     " + " ".join(f"{x:7.3f}" for x in w[:, 0]), flush=True)
                print("      broad      " + " ".join(f"{x:7.2f}" for x in w[:, 1]), flush=True)
                print("      pinned     " + " ".join(f"{'p' if b else '-':>7s}" for b in pin), flush=True)
                print("      |l1l2|/e^2 " + " ".join(f"{x / eps0[EZ] ** 2:7.2f}" for x in prod), flush=True)
                print("      Andreev    " + " ".join(f"{x:7.3f}" for x in V[:, 2 * K + 1 + 2]), flush=True)
    print("    NOTE: tau_L >= 0.9 is a PAIR-SELECTION failure (the broad pole leaves the k = 16 window, a bulk pole is taken:"
          " 'broad' collapses to < 1 ueV, the helical 'narrow' is 0.002 even with tau_R = 0.1) -- excluded from the figure.",
          flush=True)
    ok = np.array(DK_TAU) <= 0.8
    tt = np.array(DK_TAU)[ok]
    with plt.rc_context(CL_RC):
        fig, axs = plt.subplots(1, len(DK_EZ), figsize=(15.0, 5.0), sharey=True, layout='constrained')
        for a, EZ in zip(axs, DK_EZ):
            for ld, col in (('standard', CL_OUT), ('helical', CL_IN)):
                w, pin, _ = W[(ld, 0.0, EZ)]
                a.plot(tt, w[ok, 0], '-o', ms=5, color=col, lw=2.0, label=f'{ld} probe: narrow width')
                a.plot(tt, eps0[EZ] ** 2 / w[ok, 1], ':', color=col, lw=1.6,
                       label=rf'{ld}: $\varepsilon^2$ / broad width (single channel, no leak)')
            a.set_yscale('log')
            a.set_xlabel(r'probe coupling $\tau_L$ (other probe decoupled)')
            a.set_title(rf'$E_Z$ = {EZ * 1e3:.0f} $\mu$eV, splitting $\varepsilon$ = {eps0[EZ]:.2f} $\mu$eV', fontsize=11,
                        loc='left')
            a.grid(color=CL_GRID, lw=0.6, which='both')
        axs[0].set_ylabel(r'narrow pole width $-$Im $\lambda$ ($\mu$eV)')
        axs[1].legend(frameon=False, fontsize=9, loc='lower left')
        fig.suptitle(r'Dark-state law $\varepsilon^2$/broad holds only where $\varepsilon$ is large; at the splitting zeros '
                     r'a leak growing with the coupling takes over, helical probe or not', x=0.01, ha='left', fontsize=12.5,
                     fontweight='semibold')
        st.save_fig(fig, 'dark_mixed', dpi=160)
        plt.close(fig)
    print("figure: plots/dark_mixed.png", flush=True)


# %% [dark] --- 3. THE FLOOR vs JUNCTION LENGTH: is it the far Majorana's tail, ~ e^{-2L/xi_M}? ---
# One probe (tau_R 0); for each L a window of fields around the splitting zero nearest 0.9 meV ([ep5]); the narrow width's
# MINIMUM over the window (where the probe-shifted splitting vanishes, so eps^2 / broad -> 0) is the floor; floor / broad
# = the far Majorana's weight at the probe, to compare with e^{-2L/xi_M} (xi_M at that zero, [ep5] bulk cache).
DKL_L = (1500.0, 2000.0, 2500.0, 3000.0)
DKL_ZERO = (0.905, 0.850, 0.945, 0.895)                            # meV, [ep5] zeros nearest 0.9 meV
DKL_XI = (628.6, 547.0, 694.9, 613.0)                              # nm, xi_M there (scharf_absEP5_EZbulk_*)
DKL_DE = np.round(np.arange(-0.035, 0.01001, 0.0015), 4)           # meV, offsets from the zero (probes shift it down)
DKL_SER = (('standard', 0.5), ('standard', 0.8), ('helical', 0.8))


def dark_L_point(s, j, d):
    """Pair poles (EP_K Re, EP_K Im, overlap; k = 16) with ONE probe, series DKL_SER[s], L = DKL_L[j], field
    DKL_ZERO[j] + DKL_DE[d]."""
    ld, tl = DKL_SER[int(s)]
    q = junction('mixed', 0.7, tl, L=DKL_L[int(j)])
    P, O = st.nh_spectrum(q, DKL_ZERO[int(j)] + DKL_DE[int(d)], [EP8_PHI], tl, 0.0, k=16,
                          lead=helical_lead if ld == 'helical' else None)
    return np.r_[P[0, :EP_K].real, P[0, :EP_K].imag, O]


# %% [dark] --- 4. THE ZERO-BIAS DIP: energy-resolved local Andreev with one probe (the paper's Fig. 4h) ---
DKE_E = np.r_[0.0, np.geomspace(1e-4, 30.0, 45)]                   # ueV
DKE_TAU = (0.3, 0.5, 0.8)
DKE_ETA = 1e-7                                                     # eta_rel: eta = 2.5e-5 ueV (checked: no failed solves)


def dark_E_point(l, t, e):
    """Local Andreev (eh_local + he_local, e^2/h) at energy DKE_E[e], one probe of bond DKE_TAU[t], lead DK_LEADS[l],
    E_Z 0.840 meV, phi EP8_PHI, L 2 um."""
    q = st.replace(junction('mixed', 0.7, DKE_TAU[int(t)]), eta_rel_transport=DKE_ETA)
    c = st.channel_point(q, 0.840, DKE_E[int(e)] * 1e-3, phis=(EP8_PHI,), tau_R=0.0,
                         lead=helical_lead if DK_LEADS[int(l)] == 'helical' else None)
    return np.array([c[4] + c[5]])


if run_part('dark'):
    DLV = st.run_scan(dark_L_point, {'s': np.arange(len(DKL_SER)), 'j': np.arange(len(DKL_L)), 'd': np.arange(len(DKL_DE))},
                      'scharf_absDARK_L_mixed_a10_phi1.7417', n_workers=6, chunksize=1)        # (3, 4, n_d, 2 EP_K + 1)
    DEV = st.run_scan(dark_E_point, {'l': np.arange(len(DK_LEADS)), 't': np.arange(len(DKE_TAU)), 'e': np.arange(len(DKE_E))},
                      'scharf_absDARK_E_mixed_a10_phi1.7417', n_workers=6, chunksize=1)[..., 0]   # (2, 3, n_E)
    K = EP_K
    print("\n  [dark] FLOOR vs L (one probe): minimum narrow width over the window around the zero, the broad width there, "
          "floor / broad vs e^{-2L/xi_M}", flush=True)
    FL = {}
    for s, (ld, tl) in enumerate(DKL_SER):
        for j, L in enumerate(DKL_L):
            V = DLV[s, j]
            w = np.sort(-V[:, K:K + 2], axis=1)
            pin = (np.abs(V[:, :2]) < 1e-3).all(axis=1)
            i = int(np.flatnonzero(pin)[np.argmin(w[pin, 0])]) if pin.any() else int(np.argmin(w[:, 0]))
            FL[(s, j)] = (w[i, 0], w[i, 1], w[i, 0] / w[i, 1])
            print(f"    {ld:8s} tau {tl:g}, L {L / 1000:g} um: min narrow {w[i, 0]:.4f} at E_Z {(DKL_ZERO[j] + DKL_DE[i]) * 1e3:.1f} "
                  f"ueV ({'pinned' if pin[i] else 'SPLIT'}; {pin.sum()} pinned of {len(pin)}), broad {w[i, 1]:.2f}; floor/broad "
                  f"{w[i, 0] / w[i, 1]:.2e} vs e^(-2L/xi_M) {np.exp(-2 * L / DKL_XI[j]):.2e}"
                  + ("  (minimum at the window edge)" if i in (0, len(DKL_DE) - 1) else ""), flush=True)
    print("  [dark] ZERO-BIAS DIP (one probe, E_Z 840 ueV, eta 2.5e-5 ueV): local Andreev at E = 0, its maximum, and the "
          "energy where it first reaches half the maximum", flush=True)
    for l, ld in enumerate(DK_LEADS):
        for t, tl in enumerate(DKE_TAU):
            a_ = DEV[l, t]
            jh = np.flatnonzero(a_ >= 0.5 * a_.max())
            print(f"    {ld:8s} tau {tl:g}: A(0) {a_[0]:.3f}, max {a_.max():.3f} at {DKE_E[np.argmax(a_)]:.3g} ueV; half max first at "
                  f"{DKE_E[jh[0]] if jh.size else np.nan:.3g} ueV", flush=True)
    with plt.rc_context(CL_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.0, 5.2), layout='constrained')
        Ls = np.array(DKL_L) / 1000
        a1.plot(Ls, [np.exp(-2 * L / x) for L, x in zip(DKL_L, DKL_XI)], 'D', ms=9, mfc='none', mec=CL_INK, mew=1.5,
                label=r'$e^{-2L/\xi_M}$ (bulk $\xi_M$, no fit)')
        for s, ((ld, tl), col, mk) in enumerate(zip(DKL_SER, (CL_OUT, '#1d4f8a', CL_IN), ('o', 's', '^'))):
            a1.plot(Ls, [FL[(s, j)][2] for j in range(len(DKL_L))], '-' + mk, ms=8, color=col, lw=1.6,
                    label=f'{ld} probe, ' + rf'$\tau$ = {tl:g}')
        a1.set_yscale('log')
        a1.set_xlabel(r'junction length $L$ ($\mu$m)')
        a1.set_ylabel('floor of the narrow width / broad width')
        a1.set_title('(a) the floor follows the far Majorana\'s tail weight', fontsize=11, loc='left')
        a1.legend(frameon=False, fontsize=9.5)
        a1.grid(color=CL_GRID, lw=0.6, which='both')
        for l, ld in enumerate(DK_LEADS):
            for t, (tl, ls) in enumerate(zip(DKE_TAU, (':', '--', '-'))):
                a2.plot(DKE_E[1:], DEV[l, t, 1:], ls=ls, color=CL_OUT if ld == 'standard' else CL_IN, lw=2.0,
                        label=f'{ld}, ' + rf'$\tau$ = {tl:g}')
        a2.set_xscale('log')
        a2.set_ylim(0, 2.15)
        a2.axhline(2.0, color=CL_MUT, lw=0.8, ls='--')
        a2.set_xlabel(r'energy $E$ ($\mu$eV)')
        a2.set_ylabel(r'local Andreev conductance ($e^2/h$)')
        a2.set_title(r'(b) zero-bias dip on a $2e^2/h$ background ($E_Z$ = 840 $\mu$eV, one probe)', fontsize=11, loc='left')
        hl, lb = a2.get_legend_handles_labels()
        order = [0, 3, 1, 4, 2, 5]                          # columns = tau, rows = probe type
        a2.legend([hl[i] for i in order], [lb[i] for i in order], frameon=False, fontsize=9.5, ncol=3,
                  loc='upper center', bbox_to_anchor=(0.5, -0.13))
        a2.grid(color=CL_GRID, lw=0.6, which='both')
        fig.suptitle('Our junction reproduces the dark-state transport signature, but the dark pole keeps a floor set by '
                     'the far Majorana\'s tail', x=0.01, ha='left', fontsize=12.5, fontweight='semibold')
        st.save_fig(fig, 'dark_L_dip_mixed', dpi=160)
        plt.close(fig)
    print("figure: plots/dark_L_dip_mixed.png", flush=True)



# ====================================================================================================================
# PART [detphi] (2026-11-23) -- STUDY: does det r ~ 0 give the kappa = 1/2 plateau ALONG phi too?  Along E_Z ([nh] cell 4)
# the longest |kappa - 1/2| < 0.02 windows sit where the top of a splitting lobe reaches det r ~ 0 to +0.17.  PLOTS ONLY,
# from caches: study_mixed [plateau] lines scharf_absPlat_tr_mu0.7_* (equal bonds 0.4-0.7, L 1.5-3 um, 21 fields x 241
# phases each) and the [ep4] lines (25 bond pairs x 3 fields, L 2 um); duplicates dropped.  Decomposition at E_Z 0.93 from
# the [ep] poles (lossless pair, tau 0.5 one probe, 0.5 / 0.5): det r = tanh(ln x), x = eps_eff / sqrt(Gamma_L Gamma_R),
# eps_eff^2 = |l1 l2| - Gamma_L Gamma_R, Gamma = one-probe pair width (Gamma_L = Gamma_R at equal bonds: mirror check of
# [ep]).  Window as [nh] cell 4: |kappa - 1/2| < 0.02 and |G| < 0.1 kappa.  Lobe: a periodic run of det r >= -0.5 (the
# zeros of the splitting are the det r dips toward -1); its top = max det r on it.
# %% [detphi] --- 1. windows along phi, their det r type, the lobe-top rule, the E_Z 0.93 decomposition; figure ---
DPH_TOL, DPH_DIP, DPH_EZ = 0.02, -0.5, 0.93


def periodic_runs(m):
    """Maximal runs of True on a periodic grid whose last point repeats the first -> list of index arrays (in order)."""
    m = np.asarray(m, bool)[:-1]
    if m.all():
        return [np.arange(len(m))]
    s0, out, cur = int(np.argmin(m)), [], []
    for k in range(len(m)):
        i = (s0 + k) % len(m)
        if m[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def detphi_lines():
    """{(tau_L, tau_R, L um, E_Z): (kappa, G, det r)} on the 241-phase grid; the first cache with a key wins."""
    import re
    out = {}
    for f in sorted(st.DATA.glob('scharf_absPlat_tr_mu0.7_tau*_L*_EZ*.npz')):
        m = re.search(r'tau([\d.]+)_L([\d.]+)_EZ', f.name)
        z = np.load(f)
        for e, v in zip(z['axis_E_Z'], z['values']):
            out.setdefault((float(m.group(1)), float(m.group(1)), float(m.group(2)), round(float(e), 4)),
                           (v[:, 0], v[:, 1], v[:, 3]))
    z = np.load(st.DATA / 'scharf_absEP4_lines_mixed_a10.npz')
    for a, e in enumerate(z['axis_E_Z']):
        for i, tl in enumerate(EP4_TAUS):
            for j, tr in enumerate(EP4_TAUS):
                v = z['values'][a, i, j]
                out.setdefault((tl, tr, 2.0, round(float(e), 4)), (v[:, 0], v[:, 1], v[:, 3]))
    return out


if run_part('detphi'):
    LN = detphi_lines()
    n_ph = 240
    dph = 2.0 / n_ph
    WIN, LOBE, rel = [], [], 0.0
    for key, (k, G, d) in LN.items():
        m = (np.abs(k - 0.5) < DPH_TOL) & (np.abs(G) < 0.1 * k)
        for w in periodic_runs(m):
            rel = max(rel, float(np.max(np.abs(k[w] - (1 - d[w] ** 2) / 2))))
            ext = d[np.r_[(w[0] - 1) % n_ph, w, (w[-1] + 1) % n_ph]]
            nc = int(np.sum(np.diff(np.sign(ext)) != 0))
            typ = ('flank (crosses 0 once)' if nc == 1 else 'lobe top just above 0' if nc == 2 else 'wanders around 0'
                   if nc >= 3 else 'lobe top below 0' if ext.max() < 0 else 'above 0')
            WIN.append((len(w) * dph, typ, key, d[w].min(), d[w].max(), w[0] * dph, w[-1] * dph))
        for lb in periodic_runs(d >= DPH_DIP):
            win = longest_periodic(m) if len(lb) == n_ph else longest_run(m[lb])
            LOBE.append((float(d[lb].max()), win * dph, len(lb) * dph, key))
    print(f"\n  [detphi] {len(LN)} phase lines (241 phases), {len(WIN)} windows |kappa - 1/2| < {DPH_TOL}, |G| < 0.1 kappa; "
          f"kappa = (1 - det r^2)/2 inside every window to {rel:.3f}", flush=True)
    for t in ('flank (crosses 0 once)', 'lobe top below 0', 'lobe top just above 0', 'above 0', 'wanders around 0'):
        L_ = [r[0] for r in WIN if r[1] == t]
        if L_:
            print(f"    {t:24s}: {len(L_):3d} windows, median {np.median(L_):.3f} pi, p90 {np.percentile(L_, 90):.3f}, "
                  f"max {max(L_):.3f} pi", flush=True)
    print("    longest windows:", flush=True)
    for r in sorted(WIN, key=lambda r: -r[0])[:8]:
        print(f"      {r[0]:.3f} pi  ({r[5]:.3f} -> {r[6]:.3f} pi)  {r[1]:24s} det r {r[3]:+.2f}..{r[4]:+.2f}  "
              f"bonds ({r[2][0]:g}, {r[2][1]:g}) L {r[2][2]:g} um E_Z {r[2][3]:g}", flush=True)
    TOPS = np.array([(t, w, l) for t, w, l, _ in LOBE])
    print(f"    LOBE-TOP RULE ({len(LOBE)} lobes): longest window on a lobe vs det r at its top", flush=True)
    for lo, hi in ((-1.0, -0.2), (-0.2, 0.0), (0.0, 0.2), (0.2, 0.45), (0.45, 1.01)):
        s = (TOPS[:, 0] >= lo) & (TOPS[:, 0] < hi)
        if s.any():
            print(f"      top in [{lo:+.2f}, {hi:+.2f}): {s.sum():3d} lobes, window median {np.median(TOPS[s, 1]):.3f}, max "
                  f"{TOPS[s, 1].max():.3f} pi; lobe length median {np.median(TOPS[s, 2]):.3f} pi", flush=True)
    # decomposition at E_Z 0.93, bonds 0.5 / 0.5 (121 phases of the [ep] poles; transport on every second phase)
    K = EP_K
    ld = lambda n: np.load(st.DATA / f'scharf_absEP_mixed_a10_EZ{DPH_EZ:g}_{n}.npz')['values']
    P0, P1, P2 = ld('ref_R0'), ld('tauL0.5_R0'), ld('tauL0.5_R0.5')
    x = EP_PH / np.pi
    eps0 = np.abs(P0[:, 0])
    G1 = -(P1[:, K] + P1[:, K + 1])
    PP = np.abs((P2[:, 0] + 1j * P2[:, K]) * (P2[:, 1] + 1j * P2[:, K + 1]))
    epe = np.sqrt(np.clip(PP - G1 ** 2, 0, None))
    drp = 1 - 2 * G1 ** 2 / PP
    k93, G93, d93 = LN[(0.5, 0.5, 2.0, DPH_EZ)]
    xf = np.linspace(0, 2, 241)
    w93 = (np.abs(k93 - 0.5) < DPH_TOL) & (np.abs(G93) < 0.1 * k93)
    inw = w93[::2]
    dd = np.abs(drp - d93[::2])
    print(f"    E_Z {DPH_EZ} meV, bonds 0.5 / 0.5: det r from the poles vs transport: median {np.median(dd):.3f}, max "
          f"{dd[inw].max():.3f} inside the window, {dd.max():.2f} overall (at the splitting zeros)", flush=True)
    print(f"      inside the window: lossless eps {eps0[inw].min():.2f}-{eps0[inw].max():.2f}, eps_eff {epe[inw].min():.2f}-"
          f"{epe[inw].max():.2f}, Gamma_1 {G1[inw].min():.2f}-{G1[inw].max():.2f} ueV, x {(epe / G1)[inw].min():.2f}-"
          f"{(epe / G1)[inw].max():.2f}", flush=True)
    oth = (x > 0.6) & (x < 0.85)
    print(f"      other lobe (0.6-0.85 pi): lossless eps up to {eps0[oth].max():.2f}, Gamma_1 {G1[oth].min():.2f}-"
          f"{G1[oth].max():.2f}, x up to {(epe / G1)[oth].max():.2f}, kappa down to {k93[(xf > 0.6) & (xf < 0.85)].min():.3f}",
          flush=True)

    with plt.rc_context(CL_RC):
        fig = plt.figure(figsize=(15.0, 8.6), layout='constrained')
        gs = fig.add_gridspec(3, 2, width_ratios=(1.45, 1.0), height_ratios=(1.15, 1.0, 1.0))
        a1 = fig.add_subplot(gs[0, 0])
        a2 = fig.add_subplot(gs[1, 0], sharex=a1)
        a3 = fig.add_subplot(gs[2, 0], sharex=a1)
        a4 = fig.add_subplot(gs[:, 1])
        for a in (a1, a2, a3):
            for r in periodic_runs(w93):
                for piece in np.split(r, np.flatnonzero(np.diff(r) != 1) + 1):     # a run through phi = 0 wraps
                    a.axvspan(xf[piece[0]] - dph / 2, xf[piece[-1]] + dph / 2, color='#fbe3d6', lw=0, zorder=0)
            a.grid(color=CL_GRID, lw=0.6)
        a1.plot(x, eps0, color=CL_MUT, lw=1.2, ls='--', label='splitting without probes')
        a1.plot(x, epe, color=CL_OUT, lw=2.2, label=r'splitting with probes, $\varepsilon_{eff}$')
        a1.plot(x, G1, color=CL_IN, lw=2.2, label=r'probe width $\Gamma$ (one probe)')
        a1.set_ylabel(r'energy ($\mu$eV)')
        a1.set_ylim(0, 10.5)
        a1.legend(frameon=False, loc='upper center', ncol=3, fontsize=10)
        a1.set_title(rf'(a) $E_Z$ = {DPH_EZ} meV, bonds 0.5 / 0.5, $Q = -1$: the lobe through $\phi$ = 0 is flat and as '
                     r'high as $\Gamma$', loc='left', fontsize=11)
        a2.axhspan(-0.2, 0.2, color=CL_GRID, lw=0)
        a2.axhline(0, color=CL_MUT, lw=0.8, ls='--')
        a2.plot(xf, d93, color=CL_INK, lw=2.0, label='transport')
        a2.plot(x, drp, 'o', ms=3.2, color=CL_OUT, label=r'poles: $1 - 2\Gamma^2/|\lambda_1\lambda_2|$')
        a2.set_ylim(-1.05, 1.05)
        a2.set_ylabel(r'det $r$')
        a2.legend(frameon=False, loc='lower left', fontsize=10)
        a2.set_title(r'(b) so det $r$ = tanh(ln $\varepsilon_{eff}/\Gamma$) stays inside $\pm$0.2 (grey) across it',
                     loc='left', fontsize=11)
        a3.axhspan(0.48, 0.52, color=CL_GRID, lw=0)
        a3.plot(xf, k93, color=CL_INK, lw=2.0)
        a3.set_ylim(0, 0.56)
        a3.set_ylabel(r'$\kappa/\kappa_0$')
        a3.set_xlabel(r'phase $\phi$ ($\pi$)')
        a3.set_xlim(0, 2)
        a3.set_title(r'(c) and $\kappa$ = (1 $-$ det $r^2$)/2 sits at 1/2 (shaded: $|\kappa - 1/2| < 0.02$, longest run '
                     r'1.18$\pi$)', loc='left', fontsize=11)
        for a in (a1, a2):
            plt.setp(a.get_xticklabels(), visible=False)
        TY = (('wanders around 0', 'wanders around 0\n(crosses 3+ times)'),
              ('lobe top just above 0', 'turns just above 0\n(crosses twice)'),
              ('lobe top below 0', 'turns just below 0'),
              ('above 0', 'stays above 0'),
              ('flank (crosses 0 once)', 'crosses 0 once'))
        rng = np.random.default_rng(1)
        ylab = []
        for yi, (t, lab) in enumerate(TY):
            L_ = np.array([r[0] for r in WIN if r[1] == t])
            jit = rng.uniform(-0.22, 0.22, len(L_))
            a4.plot(L_, yi + jit, 'o', ms=5, mfc=CL_OUT, mec='w', mew=0.5, alpha=0.75)
            a4.plot([np.median(L_)] * 2, [yi - 0.32, yi + 0.32], color=CL_INK, lw=2.2)
            ylab.append(f'{lab}\nn = {len(L_)}')
            if t == 'wanders around 0':
                i0 = int(np.argmax(L_))
                a4.plot(L_[i0], yi + jit[i0], 'o', ms=12, mfc='none', mec=CL_INK, mew=1.6)
                a4.annotate('(a)-(c)', (L_[i0], yi + jit[i0]), xytext=(-26, 22), textcoords='offset points', fontsize=10,
                            ha='right')
        a4.set_yticks(range(len(TY)))
        a4.set_yticklabels(ylab, fontsize=10.5)
        a4.set_ylim(len(TY) - 0.5, -0.5)
        a4.set_xlim(0, 1.25)
        a4.set_xlabel(r'length of the $|\kappa - 1/2| < 0.02$ window ($\pi$)')
        a4.grid(color=CL_GRID, lw=0.6, axis='x')
        a4.set_title(f'(d) all {len(WIN)} windows on {len(LN)} phase lines, by what det $r$ does there\n'
                     '(black bar: median): crossing 0 once gives a short window;\n'
                     'long ones need det $r$ to turn or wander near 0', loc='left', fontsize=11)
        fig.suptitle(r'Along $\phi$ too, $\kappa$ = 1/2 exactly where $|$det $r| < 0.2$; the long $\phi$ plateau is a flat '
                     r'splitting lobe matched to the probe width', x=0.01, ha='left', fontsize=13, fontweight='semibold')
        st.save_fig(fig, 'detphi_mixed', dpi=160)
        plt.close(fig)
    print("figure: plots/detphi_mixed.png", flush=True)



# ====================================================================================================================
# PART [epfield] (2026-11-23) -- STUDY: the EP spectrum vs the Zeeman field over the WHOLE topological range at the
# working phase phi = EP8_PHI = 1.7417 pi (Q = -1 for E_Z ~0.53-1.235 meV; tuning Q map + scharf_absPlat_Qcut).  Pair
# poles (ep_point) for one strong probe tau_L = 0.3 / 0.5 / 0.7 / 1.0 plus a weak detector tau_R = 0.1 (the Li &
# Trauzettel set-up: one lossy contact; equal bonds have EP segments of <= ~0.05 pi only) and the lossless pair (0, 0);
# the E = 0 transport row (kappa, G, Andreev, det r_L, n_bad) for the two-probe pairs on the way.  [ep8] covered
# 0.80-1.10 meV only, without 0.7 / 1.0.  The broad pole of strong probes lies beyond many bulk poles (tau_L 1.0 at
# E_Z 0.93: -163.5j ueV vs bulk |Re| 47-51; not among the 32 poles closest to 0), so tau_L 0.7 / 1.0 search around
# -40j / -150j ueV (k 24; checked at 0.93: same pair as sigma 0 for 0.7).  One task per field.
# Cache scharf_absEPF_field_*; plots/epfield_mixed.png.
# %% [epfield] --- 1. SCAN: poles + E = 0 transport per field ---
EPF_EZ = np.round(np.arange(0.45, 1.30001, 0.005), 4)
EPF_PAIRS = ((0.0, 0.0), (0.3, 0.1), (0.5, 0.1), (0.7, 0.1), (1.0, 0.1))
EPF_K = {0.7: 24, 1.0: 24}                                         # pole window per tau_L (default 16)
EPF_SIG = {0.7: -40j, 1.0: -150j}                                  # search centre per tau_L (ueV; default 0)
EPF_NV = 2 * EP_K + 1 + 5                                          # ep_point values + transport row


def epf_field(E_Z):
    """At one field, for every pair of EPF_PAIRS: ep_point at EP8_PHI and the E = 0 transport row (NaN for (0, 0))."""
    out = []
    for tl, tr in EPF_PAIRS:
        p = ep_point(EP8_PHI, tl, tr, E_Z=float(E_Z), k=EPF_K.get(tl, 16), sigma=EPF_SIG.get(tl, 0.0))
        t = (st.transport_phis(junction('mixed', 0.7, tl), float(E_Z), [EP8_PHI], tau_R=tr)[0] if tl > 0
             else np.full(5, np.nan))
        out.append(np.r_[p, t])
    return np.array(out).ravel()


if run_part('epfield'):
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EPF = st.myf.scan(epf_field, {'E_Z': EPF_EZ}, pool=_pool, chunksize=1,
                          save=str(st.DATA / f'scharf_absEPF_field_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
    finally:
        _pool.shutdown()
    EPF = EPF.reshape(len(EPF_EZ), len(EPF_PAIRS), EPF_NV)
    print("epfield: all fields done", flush=True)


# %% [epfield] --- 2. EP segments over the whole field range, narrow / broad widths, the segment-length law; figure ---
# D = ((l1 - l2)/2)^2 (real): > 0 split pair, < 0 between EPs ([ep8] ep8_segments / ep8_crossings).  Topological range
# from the Q cut at this phase (upper edge) and the tuning Q map (lower edge, 0.50-0.55 meV, coarse).
EPF_LOW = (0.50, 0.55)                                             # meV: Q flips between these (tuning Q map, phi 1.733 pi)
EPF_COL = {0.3: '#2a78d6', 0.5: '#1baf7a', 0.7: '#eb6834', 1.0: '#8a7ee0'}

if run_part('epfield'):
    E = EPF_EZ * 1e3
    K = EP_K
    lam = EPF[:, :, :2] + 1j * EPF[:, :, K:K + 2]
    eps0 = np.abs(EPF[:, 0, 0])
    EC = ep8_crossings(E, eps0)
    _qz = np.load(st.DATA / f'scharf_absPlat_Qcut_mu0.7_phi{EP8_PHI / np.pi:.4f}pi.npz')
    _qe, _qv = _qz['axis_E_Z'] * 1e3, _qz['values'][:, 0]
    E_UP = _qe[np.flatnonzero(_qv > 0)[0]] - 2.5                    # ueV: midpoint of the last -1 / first +1 step
    topo = (E > EPF_LOW[1] * 1e3) & (E < E_UP)
    # the pair = the two poles with the smallest |Re E| fails next to the transitions: a gap-closing bulk state (width
    # 0.005-0.07 ueV, decoupled from the probes) takes its place (e.g. tau 0.3 at 1220: 4.41 - 0.009i, while the Majorana
    # pair sits at +-6.9 - 1.02i).  Masked: within 10 / 25 ueV of the lower / upper transition, or where the two-pole
    # kappa(E <= 20 ueV) exceeds 1 (impossible for a pair).
    _Echk = np.r_[0.0, np.geomspace(1e-3, 20.0, 120)]
    VALID = {}
    for n, (tl, tr) in enumerate(EPF_PAIRS):
        if tl > 0:
            l1, l2, k0 = lam[:, n, 0], lam[:, n, 1], EPF[:, n, 2 * K + 1]
            mx = np.array([np.max(k0[i] * np.abs(l1[i] * l2[i]) ** 2 / np.abs((_Echk - l1[i]) * (_Echk - l2[i])) ** 2)
                           for i in range(len(E))])
            VALID[tl] = (E > EPF_LOW[1] * 1e3 + 10) & (E < E_UP - 25) & (mx <= 1.0)
            print(f'    ({tl:g}, {tr:g}): pair valid on {VALID[tl][topo].sum()} of {topo.sum()} topological fields; masked '
                  + ', '.join(f'{e:.0f}' for e in E[topo & ~VALID[tl]]), flush=True)
    print(f"\n  [epfield] phi {EP8_PHI / np.pi:.4f} pi, E_Z {E[0]:.0f}-{E[-1]:.0f} ueV (step {E[1] - E[0]:.0f}); topological "
          f"for E_Z {EPF_LOW[0] * 1e3:.0f}-{EPF_LOW[1] * 1e3:.0f} ... {E_UP:.1f} ueV; failed solves "
          f"{int(np.nansum(EPF[:, 1:, 2 * K + 5]))}", flush=True)
    print(f"    lossless crossings of eps(E_Z) (ueV): " + ", ".join(f"{e:.1f}" for e in EC), flush=True)
    # |d eps / dE_Z| at each crossing from the signed lossless splitting: |eps| has a V there -> mean of the two side slopes
    sl = []
    for ec in EC:
        i = int(np.argmin(np.abs(E - ec)))
        lo, hi = max(i - 3, 0), min(i + 4, len(E))
        a_ = np.polyfit(E[lo:i], eps0[lo:i], 1)[0] if i - lo >= 2 else np.nan
        b_ = np.polyfit(E[i + 1:hi], eps0[i + 1:hi], 1)[0] if hi - i - 1 >= 2 else np.nan
        sl.append(np.nanmean([abs(a_), abs(b_)]))
    print("    |d eps/dE_Z| at the crossings: " + ", ".join(f"{s:.3f}" for s in sl), flush=True)
    SEG = {}
    for n, (tl, tr) in enumerate(EPF_PAIRS):
        if tl == 0:
            continue
        l1, l2 = lam[:, n, 0], lam[:, n, 1]
        D = (((l1 - l2) / 2) ** 2).real
        segs = ep8_segments(E, D)
        SEG[tl] = segs
        w = -np.c_[l1.imag, l2.imag]
        nar, bro = w.min(axis=1), w.max(axis=1)
        pin = D < 0
        kap = EPF[:, n, 2 * K + 1]
        tot = sum(min(s[1], E_UP) - max(s[0], EPF_LOW[1] * 1e3) for s in segs if s[1] > EPF_LOW[1] * 1e3 and s[0] < E_UP)
        print(f"    ({tl:g}, {tr:g}): {len(segs)} segments, between EPs {tot:.0f} of {E_UP - EPF_LOW[1] * 1e3:.0f} ueV of the "
              f"topological range ({tot / (E_UP - EPF_LOW[1] * 1e3) * 100:.0f} %); narrow width between EPs median "
              f"{np.median(nar[pin & topo]) if (pin & topo).any() else np.nan:.3f}, broad median "
              f"{np.median(bro[pin & topo]) if (pin & topo).any() else np.nan:.2f} ueV; split-pair width median "
              f"{np.median(nar[~pin & topo]):.2f}; kappa(E = 0) max {np.nanmax(kap[topo]):.3f}", flush=True)
        for s in segs:
            inn = [j for j, ec in enumerate(EC) if s[0] - 15 <= ec <= s[1] + 15]
            tag = (f"crossing {EC[inn[0]]:.1f} (slope {sl[inn[0]]:.3f}) -> 2 delta_eff ~ L x slope = "
                   f"{(s[1] - s[0]) * sl[inn[0]]:.2f} ueV" if len(inn) == 1 else 'no crossing inside (loss-driven)'
                   if not inn else 'merged over the crossings ' + ', '.join(f'{EC[j]:.0f}' for j in inn))
            print(f"      {s[0]:7.1f} - {s[1]:7.1f}  ({s[1] - s[0]:6.1f} ueV){'  [grid edge]' if s[2] else ''}  {tag}",
                  flush=True)

    with plt.rc_context(CL_RC):
        taus = [tl for tl, _ in EPF_PAIRS if tl > 0]
        fig, axs = plt.subplots(2, len(taus), figsize=(4.1 * len(taus) + 1.0, 7.6), sharex=True, sharey='row',
                                layout='constrained', gridspec_kw=dict(height_ratios=(1.0, 1.0)))
        for c, tl in enumerate(taus):
            n = [i for i, (a, _) in enumerate(EPF_PAIRS) if a == tl][0]
            l1, l2 = lam[:, n, 0], lam[:, n, 1]
            w = -np.c_[l1.imag, l2.imag]
            ar, ai = axs[0, c], axs[1, c]
            for a in (ar, ai):
                for s in SEG[tl]:
                    a.axvspan(s[0], s[1], color='#fbe3d6', lw=0, zorder=0)
                a.axvspan(E[0], EPF_LOW[1] * 1e3, color=CL_GRID, lw=0, zorder=0)
                a.axvspan(E_UP, E[-1], color=CL_GRID, lw=0, zorder=0)
                a.axvspan(E_UP - 25, E_UP, color=CL_GRID, lw=0, zorder=0, alpha=0.5)
                for ec in EC:
                    a.axvline(ec, color=CL_MUT, lw=0.7, ls=':')
                a.grid(color=CL_GRID, lw=0.5)
            ar.plot(E, eps0, color=CL_MUT, lw=1.0, ls='--', label='no probes')
            ar.plot(E, -eps0, color=CL_MUT, lw=1.0, ls='--')
            ar.plot(E, np.abs(l1.real), color=EPF_COL[tl], lw=1.8, label='with probes')
            ar.plot(E, -np.abs(l1.real), color=EPF_COL[tl], lw=1.8)
            ai.plot(E, w.max(axis=1), color=EPF_COL[tl], lw=1.8, label='broad')
            ai.plot(E, w.min(axis=1), color=CL_INK, lw=1.4, label='narrow')
            ai.set_yscale('log')
            ar.set_title(rf'$\tau_L$ = {tl:g}, $\tau_R$ = 0.1', fontsize=12)
        axs[0, 0].set_ylabel(r'Re $\lambda$ ($\mu$eV)')
        axs[1, 0].set_ylabel(r'pole width $-$Im $\lambda$ ($\mu$eV)')
        axs[0, 0].legend(frameon=False, fontsize=9.5, loc='upper right')
        axs[1, 0].legend(frameon=False, fontsize=9.5, loc='lower right')
        fig.suptitle(rf'EP spectrum along the field ($\phi$ = {EP8_PHI / np.pi:.4f}$\pi$, full model): each zero of the splitting '
                     r'opens into an EP segment (shaded)' '\n' r'that widens with the strong probe until the segments merge; at '
                     r'$\tau_L$ = 1 the whole topological range lies between EPs',
                     x=0.01, ha='left', fontsize=12.5, fontweight='semibold')
        fig.supxlabel(r'Zeeman energy $E_Z$ ($\mu$eV)      (grey: trivial, $Q = +1$;  dotted: zeros of the splitting '
                      'without probes)', fontsize=12)
        st.save_fig(fig, 'epfield_mixed', dpi=160)
        plt.close(fig)
    print("figure: plots/epfield_mixed.png", flush=True)


# %% [epfield] --- 3. kappa(E) and kappa(T) ALONG E_Z from the pair poles (the tested two-pole form) ---
# kappa(E) = kappa(0) |l1 l2|^2 / |(E - l1)(E - l2)|^2 ([nh] WHY (2): full model to median 1e-4 for E <= 20 ueV along
# phi), kappa(0) from the scan's E = 0 transport row; kappa(T) with the tools.thermal kernel.  Two Majorana poles only:
# valid while they are the nearest poles to E = 0 and k_B T << the bulk levels (~47 ueV at 0.93 meV) -- tested in cell 4.
EPF_E = np.r_[0.0, np.geomspace(1e-3, 60.0, 240)]                 # ueV
EPF_T = (0.0, 5.0, 10.0, 20.0, 50.0)                              # mK


def epf_pole_kappa(k0, l1, l2, E):
    """Two-pole kappa(E) (E in ueV, array) at one field."""
    return k0 * np.abs(l1 * l2) ** 2 / np.abs((E - l1) * (E - l2)) ** 2


def epf_kT(kE, T, E=None):
    """kappa(T) (T in mK) from kappa(E) on the grid E (ueV, default EPF_E), via tools.thermal (S_th = 2 kappa(E))."""
    E = EPF_E if E is None else E
    z = np.zeros_like(kE)
    ch = {'ee': kE, 'hh': kE, 'he_cross': z, 'eh_cross': z, 'eh_local': z, 'he_local': z}
    return st.thermal(ch, st.kT_of_mK(T), E * 1e-3)[0]


if run_part('epfield'):
    KE, KT = {}, {}
    for n, (tl, tr) in enumerate(EPF_PAIRS):
        if tl == 0:
            continue
        k0 = EPF[:, n, 2 * K + 1]
        KE[tl] = np.array([epf_pole_kappa(k0[i], lam[i, n, 0], lam[i, n, 1], EPF_E) for i in range(len(E))])
        KT[tl] = {T: np.array([epf_kT(KE[tl][i], T) for i in range(len(E))]) for T in EPF_T}
        KE[tl][~VALID[tl]] = np.nan
        for T in EPF_T:
            KT[tl][T][~VALID[tl]] = np.nan
    print(f"\n  [epfield] kappa(T) along E_Z from the poles (inside = between EPs; topological range only):", flush=True)
    for tl in KE:
        n = [i for i, (a, _) in enumerate(EPF_PAIRS) if a == tl][0]
        D = (((lam[:, n, 0] - lam[:, n, 1]) / 2) ** 2).real
        ins = (D < 0) & topo & VALID[tl]
        ok = topo & VALID[tl] & (KT[tl][0.0] > 0.05)
        r20 = KT[tl][20.0] / KT[tl][0.0]
        win = {T: longest_run(np.abs(KT[tl][T][topo & VALID[tl]] - 0.5) < 0.02) * EP8_DE for T in EPF_T}
        print(f"    tau_L {tl:g}: kappa max " + " / ".join(f"{np.nanmax(KT[tl][T][topo & VALID[tl]]):.3f}" for T in EPF_T)
              + f" at {' / '.join(f'{T:g}' for T in EPF_T)} mK; |kappa - 1/2| < 0.02 window " + " / ".join(f"{win[T]:.0f}" for T in EPF_T)
              + f" ueV; kappa(20 mK)/kappa(0) median {np.median(r20[ok & ins]) if (ok & ins).any() else np.nan:.3f} between EPs / "
              f"{np.median(r20[ok & ~ins]) if (ok & ~ins).any() else np.nan:.3f} outside", flush=True)
    with plt.rc_context(CL_RC):
        taus = list(KE)
        fig, axs = plt.subplots(2, len(taus), figsize=(4.1 * len(taus) + 1.0, 7.4), sharex=True, layout='constrained',
                                gridspec_kw=dict(height_ratios=(1.0, 1.0)))
        for c, tl in enumerate(taus):
            am, al = axs[0, c], axs[1, c]
            pm = am.pcolormesh(E, EPF_E[1:], KE[tl][:, 1:].T, cmap='viridis', vmin=0, vmax=0.5, shading='auto')
            am.set_yscale('log')
            am.set_ylim(1e-3, 60)
            for s in SEG[tl]:
                am.plot([s[0], s[1]], [1.3e-3, 1.3e-3], color='#eb6834', lw=5, solid_capstyle='butt')
            am.set_title(rf'$\tau_L$ = {tl:g}, $\tau_R$ = 0.1', fontsize=12)
            for s in SEG[tl]:
                al.axvspan(s[0], s[1], color='#fbe3d6', lw=0, zorder=0)
            al.axvspan(E[0], EPF_LOW[1] * 1e3, color=CL_GRID, lw=0, zorder=0)
            al.axvspan(E_UP, E[-1], color=CL_GRID, lw=0, zorder=0)
            al.axhspan(0.48, 0.52, color=CL_GRID, lw=0)
            for T, col in zip(EPF_T, EP7_TCOL):
                al.plot(E, KT[tl][T], color=col, lw=1.6, label=f'{T:g} mK')
            al.set_ylim(0, 0.56)
            al.grid(color=CL_GRID, lw=0.5)
        axs[0, 0].set_ylabel(r'energy $E$ ($\mu$eV)')
        axs[1, 0].set_ylabel(r'$\kappa/\kappa_0$')
        axs[1, -1].legend(frameon=False, fontsize=9.5, loc='upper left', title='temperature', title_fontsize=9.5)
        fig.colorbar(pm, ax=axs[0, :], label=r'$\kappa(E)/\kappa_0$', shrink=0.9)
        fig.supxlabel(r'Zeeman energy $E_Z$ ($\mu$eV)      (orange bars / shading: between EPs;  grey: trivial)', fontsize=12)
        fig.suptitle(r'Heat transport along the field from the pair poles: between EPs the zero-energy peak is as narrow as the '
                     r'narrow pole, so $\kappa$ there is the first to go with temperature', x=0.01, ha='left', fontsize=12.5,
                     fontweight='semibold')
        st.save_fig(fig, 'epfield_kappaT_mixed', dpi=160)
        plt.close(fig)
    print("figure: plots/epfield_kappaT_mixed.png", flush=True)


# %% [epfield] --- 4. FULL-TRANSPORT CHECKS at three fields per coupling: outside, at an EP, inside a segment ---
# Per tau_L: the segment nearest the slow crossing (~842 ueV); its EP edge refined by bisection on D (7 steps from the
# 5 ueV grid -> ~0.04 ueV); 'inside' = the grid field of that segment with the narrowest pole; 'outside' = a split-pair
# grid field ~max(10 ueV, half the segment) beyond the edge.  Poles there (ep_point) + the six channels on EPF_EC
# (eta_rel 1e-7 as [dark]; unreliable lead solves counted) -> kappa(E), nonlocal G(E) = T_ECT - T_CAR vs the two-pole
# kappa form; at the EP the double pole predicts a squared Lorentzian w^4/(E^2 + w^2)^2 (half width 0.644 w).
EPF_EC = np.r_[0.0, np.geomspace(3e-3, 100.0, 48)]                # ueV
EPF_ETA = 1e-7
EPF_SLOW = 841.7                                                  # ueV: the slow lossless crossing ([ep8])


def epf_D(tl, ez_ueV):
    """D = ((l1 - l2)/2)^2 of the pair poles at one field (ueV) for (tl, 0.1), and the pair."""
    v = ep_point(EP8_PHI, tl, 0.1, E_Z=ez_ueV * 1e-3, k=EPF_K.get(tl, 16), sigma=EPF_SIG.get(tl, 0.0))
    l1, l2 = v[0] + 1j * v[EP_K], v[1] + 1j * v[EP_K + 1]
    return (((l1 - l2) / 2) ** 2).real, l1, l2


def epf_bisect(i, br):
    """EP edge for br[i] = (tau_L, a, b) with D(a), D(b) of opposite sign: [E_EP, a, b, D(a), D(b), Re/Im of the pair]."""
    tl, a, b = br[int(i)]
    da = epf_D(tl, a)[0]
    for _ in range(7):
        m = (a + b) / 2
        dm = epf_D(tl, m)[0]
        if np.sign(dm) == np.sign(da):
            a, da = m, dm
        else:
            b = m
    e = (a + b) / 2
    d, l1, l2 = epf_D(tl, e)
    return np.array([e, a, b, da, d, l1.real, l1.imag, l2.real, l2.imag])


def epf_check_poles(i, combos):
    """Pair poles [Re l1, Im l1, Re l2, Im l2] (ueV) at the check combo i = (E_Z meV, tau_L, tau_R)."""
    ez, tl, tr = combos[int(i)]
    v = ep_point(EP8_PHI, tl, tr, E_Z=float(ez), k=EPF_K.get(tl, 16), sigma=EPF_SIG.get(tl, 0.0))
    return np.array([v[0], v[EP_K], v[1], v[EP_K + 1]])


def epf_check_point(E, combos):
    """Six channels at E (ueV), phi EP8_PHI, for every (E_Z meV, tau_L, tau_R) of combos, flattened + n_unreliable."""
    q = st.replace(junction('mixed', 0.7), eta_rel_transport=EPF_ETA)
    ch, nbad = st.channel_combos(q, float(E) * 1e-3, combos, EP8_PHI)
    return np.r_[ch.ravel(), nbad]


if run_part('epfield'):
    # 1. the check fields per coupling from the field scan
    PICK, BR = {}, []
    for tl in [t for t, _ in EPF_PAIRS if t > 0]:
        n = [i for i, (a, _) in enumerate(EPF_PAIRS) if a == tl][0]
        D = (((lam[:, n, 0] - lam[:, n, 1]) / 2) ** 2).real
        nar = np.min(-np.c_[lam[:, n, 0].imag, lam[:, n, 1].imag], axis=1)
        cand = [s for s in SEG[tl] if not s[2] and s[0] > EPF_LOW[1] * 1e3 + 10 and s[1] < E_UP - 25]
        if not cand:                                                  # no EP in the valid range (tau 1.0): inside only
            ok_ = np.flatnonzero(VALID[tl] & (D < 0))
            PICK[tl] = dict(seg=None, side=None, ins=float(E[ok_[np.argmin(nar[ok_])]]),
                            top=float(E[ok_[np.argmin(np.abs(E[ok_] - 945.0))]]))
            continue
        seg = min(cand, key=lambda s: max(s[0] - EPF_SLOW, EPF_SLOW - s[1], 0.0))
        out_lo = np.flatnonzero((E < seg[0]) & (E > EPF_LOW[1] * 1e3) & (D > 0))
        side = 'lower' if out_lo.size else 'upper'
        edge = seg[0] if side == 'lower' else seg[1]
        j = int(np.searchsorted(E, edge))                             # E[j-1] < edge <= E[j]
        BR.append((tl, float(E[j - 1]), float(E[j])))
        gap = max(10.0, 0.5 * (seg[1] - seg[0]))
        if side == 'lower':
            io = out_lo[np.argmin(np.abs(E[out_lo] - (seg[0] - gap)))]
        else:
            o_ = np.flatnonzero((E > seg[1]) & (E < E_UP) & (D > 0))
            io = o_[np.argmin(np.abs(E[o_] - (seg[1] + gap)))]
        ii = np.flatnonzero((E > seg[0]) & (E < seg[1]))
        iin = ii[np.argmin(nar[ii])] if ii.size else int(np.argmin(np.abs(E - (seg[0] + seg[1]) / 2)))
        PICK[tl] = dict(seg=seg, side=side, out=float(E[io]), ins=float(E[iin]))
    _pool = st.myf.scans.make_pool(EP_W)
    try:
        EPB = st.myf.scan(partial(epf_bisect, br=tuple(BR)), {'i': np.arange(len(BR))}, pool=_pool, chunksize=1,
                          save=str(st.DATA / f'scharf_absEPF_epedge_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
        for (tl, a, b), r in zip(BR, EPB):
            assert abs(r[1] - a) <= b - a and abs(r[2] - b) <= b - a, 'stale EP-edge cache: brackets changed'
            PICK[tl]['ep'] = float(r[0])
        WHERE = tuple((tl, w) for tl in PICK for w in (('out', 'ep', 'ins') if PICK[tl]['seg'] else ('top', 'ins')))
        COMBOS = tuple((PICK[tl][w] * 1e-3, tl, 0.1) for tl, w in WHERE)
        cf = st.DATA / f'scharf_absEPF_chk_combos_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npy'
        if cf.exists():
            assert np.allclose(np.load(cf), np.array(COMBOS)), 'stale check cache: the check fields changed'
        else:
            np.save(cf, np.array(COMBOS))
        CP = st.myf.scan(partial(epf_check_point, combos=COMBOS), {'E': EPF_EC}, pool=_pool, chunksize=1,
                         save=str(st.DATA / f'scharf_absEPF_chk_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
        CPP = st.myf.scan(partial(epf_check_poles, combos=COMBOS), {'i': np.arange(len(COMBOS))}, pool=_pool, chunksize=1,
                          save=str(st.DATA / f'scharf_absEPF_chkpoles_mixed_a10_phi{EP8_PHI / np.pi:.4f}.npz'))['values']
    finally:
        _pool.shutdown()
    print(f"\n  [epfield] FULL-TRANSPORT CHECKS (eta_rel {EPF_ETA:g}; unreliable lead solves {int(CP[:, -1].sum())}):", flush=True)
    CH = CP[:, :-1].reshape(len(EPF_EC), len(COMBOS), 6)
    RES = {}
    LAB = {'out': 'outside', 'ep': 'at the EP', 'ins': 'inside (narrowest)', 'top': 'inside (lobe top)'}
    for m, (ez, tl, tr) in enumerate(COMBOS):
        w = LAB[WHERE[m][1]]
        ch = {k: CH[:, m, n] for n, k in enumerate(st.CHANNELS)}
        k_full = (ch['ee'] + ch['he_cross'] + ch['hh'] + ch['eh_cross']) / 2
        g_full = (ch['ee'] - ch['he_cross'] + ch['hh'] - ch['eh_cross']) / 2
        l1, l2 = CPP[m, 0] + 1j * CPP[m, 1], CPP[m, 2] + 1j * CPP[m, 3]
        k_pole = epf_pole_kappa(k_full[0], l1, l2, EPF_EC)
        sel = (EPF_EC <= 20.0) & (k_full[0] > 1e-3)
        # (kappa(0) <= 1e-3: nothing to compare -- the no-EP 'narrowest' picks sit next to the lower transition)
        dev = np.max(np.abs(k_pole[sel] - k_full[sel])) / max(k_full[0], 1e-12) if sel.any() else np.nan
        half = EPF_EC[np.flatnonzero(k_full < 0.5 * k_full[0])[0]] if (k_full < 0.5 * k_full[0]).any() else np.nan
        kts = [st.thermal(ch, st.kT_of_mK(T), EPF_EC * 1e-3)[0] for T in EPF_T]
        ktp = [epf_kT(k_pole, T, EPF_EC) for T in EPF_T]
        RES[m] = dict(k=k_full, g=g_full, kp=k_pole, l1=l1, l2=l2, half=half)
        print(f"    tau_L {tl:g}, E_Z {ez * 1e3:.2f} ueV ({w}): poles {l1:.3f}, {l2:.3f}; kappa(0) {k_full[0]:.3f}, half at "
              f"{half:.3g} ueV; two-pole form max dev (E <= 20) {dev:.3f} of kappa(0); max |G_nl(E)| {np.max(np.abs(g_full)):.3f}"
              f"; kappa(T) full / poles at " + ", ".join(f"{T:g}: {a:.3f}/{b:.3f}" for T, a, b in zip(EPF_T, kts, ktp)),
              flush=True)
        if w == 'at the EP':
            wv = -0.5 * (l1.imag + l2.imag)
            sq = k_full[0] * wv ** 4 / (EPF_EC ** 2 + wv ** 2) ** 2
            print(f"      squared Lorentzian, w = {wv:.3f} ueV (half width 0.644 w = {0.644 * wv:.3g}): max dev (E <= 20) "
                  f"{np.max(np.abs(sq[sel] - k_full[sel])) / max(k_full[0], 1e-12):.3f} of kappa(0); |l1 - l2| = "
                  f"{abs(l1 - l2):.3f} ueV", flush=True)
    with plt.rc_context(CL_RC):
        taus = list(PICK)
        fig, axs = plt.subplots(len(taus), 3, figsize=(14.0, 3.0 * len(taus) + 1.0), sharex=True, layout='constrained')
        for r_, tl in enumerate(taus):
            cols_ = [m for m, (t_, _) in enumerate(WHERE) if t_ == tl]
            for c_ in range(3):
                a = axs[r_, c_]
                if c_ >= len(cols_):
                    a.set_axis_off()
                    continue
                m = cols_[c_]
                w = LAB[WHERE[m][1]]
                R = RES[m]
                a.plot(EPF_EC[1:], R['k'][1:], color=CL_INK, lw=2.0, label=r'$\kappa$ full model')
                a.plot(EPF_EC[1:], R['kp'][1:], color=EPF_COL[tl], lw=1.6, ls='--', label='two-pole form')
                a.plot(EPF_EC[1:], R['g'][1:], color='#c0392b', lw=1.2, ls=':',
                       label=r'nonlocal $G$ (ECT $-$ CAR)')
                a.set_xscale('log')
                a.axhline(0, color=CL_MUT, lw=0.6)
                a.grid(color=CL_GRID, lw=0.5)
                a.set_title(rf'$\tau_L$ {tl:g}, {w}: $E_Z$ {COMBOS[m][0] * 1e3:.1f} $\mu$eV' '\n'
                            rf'poles {R["l1"].real:+.2f}{R["l1"].imag:+.2f}i, {R["l2"].real:+.2f}{R["l2"].imag:+.2f}i $\mu$eV',
                            fontsize=9.5, loc='left')
            axs[r_, 0].set_ylabel(r'$\kappa/\kappa_0$, $G/G_0$')
        for a in axs[-1]:
            a.set_xlabel(r'energy $E$ ($\mu$eV)')
        axs[0, 0].legend(frameon=False, fontsize=8.5, loc='upper right')
        fig.suptitle(r'Full transport vs the pair poles ($\tau_R$ = 0.1, $\phi$ = 1.7417$\pi$): the zero-energy peak follows the '
                     'two poles exactly' '\n' 'outside, at and between the EPs; nonlocal G stays ~0', x=0.01, ha='left',
                     fontsize=12.5, fontweight='semibold')
        st.save_fig(fig, 'epfield_checks_mixed', dpi=150)
        plt.close(fig)
    print("figure: plots/epfield_checks_mixed.png", flush=True)



# ====================================================================================================================
# PART 'regimes'  (2026-11-24)  PLOTS ONLY from caches: which gate (mu_C) and probe bond (tau) values show different regimes?
# ====================================================================================================================
# Thesis junction (mixed SOC, W 100, L 2 um, tc 1, a 10).  Sources, all full-model caches:
#   phi = pi bulk Q and refined Delta_top on (mu_C 0-2.4, E_Z 0-2)   study_mixed [deck] 4   scharf_absDeck_topo_mixed_tc1
#   Gamma(tau) = local-Andreev half width at the eps zeros E_Z 0.285 / 0.345, phi pi, mu_C 0.7   [leads] 3   scharf_absLeads_gamma
#   coarse (E_Z, phi) transport maps per (mu_C, tau)   [tuning]   tr_name('mixed', mu, tau, 2000)
# Gapless = Q = -1 with refined Delta_top < CLOSED_L ([leads]); values are either < 1e-3 or > 0.08 ueV, nothing between.
REG_MU = {0.3: 'R1', 0.7: 'R2', 1.0: 'R3', 1.6: 'R4'}           # proposed gate picks (meV)
REG_MU_TXT = {0.3: 'ends decoupled', 0.7: 'ends overlap (working point)', 1.0: 'gapped at low field only',
              1.6: 'Q = -1 but gapless'}
REG_TAU = {0.2: 'C1', 0.5: 'C2', 0.7: 'C3', 1.0: 'C4'}            # proposed probe-bond picks
REG_TAU_TXT = {0.2: 'tunnel', 0.5: 'matched', 0.7: 'crossover', 1.0: 'transparent'}
REG_COL = {0.3: '#2a78d6', 0.7: '#1baf7a', 1.0: '#eb6834', 1.6: '#8a7ee0', 1.2: '#6b6a66'}
REG_TR = ((0.3, (0.3387, 0.5)), (0.7, (0.2, 0.3387, 0.5, 0.7, 1.0)), (1.2, (0.3387, 0.5)))   # cached coarse maps

if run_part('regimes'):
    from matplotlib.patches import Rectangle
    _z = np.load(st.DATA / 'scharf_absDeck_topo_mixed_tc1.npz')
    RMU, REZ, RV = _z['axis_mu_C'], _z['axis_E_Z'], _z['values']
    RQ, RGAP = RV[..., 0], RV[..., 1] * 1e3                                    # Q, Delta_top (ueV; NaN where Q = +1)
    topo_ = RQ < 0
    gapless_ = topo_ & ~(RGAP >= CLOSED_L * 1e3)
    gapped_ = topo_ & ~gapless_
    _g = np.load(st.DATA / 'scharf_absLeads_gamma.npz')['values'][:, 0].reshape(len(TAU_LEADS), len(GAMMA_FIELDS))
    GAM_R = dict(zip(TAU_LEADS, _g))                                           # ueV, at the two eps zeros
    print(f"\n  [regimes] phi = pi bulk (tc 1, mixed SOC): Q = -1 on {topo_.mean() * 100:.0f} % of (mu_C 0-2.4, E_Z 0-2), "
          f"gapless share of it {gapless_.sum() / topo_.sum() * 100:.0f} %", flush=True)
    print("    gate pick   Q = -1 at E_Z (meV)   gapped at E_Z      max Delta_top (ueV, at E_Z)", flush=True)
    for mu, lab in REG_MU.items():
        i = int(np.argmin(np.abs(RMU - mu)))
        t_, g_ = REZ[topo_[i]], REZ[gapped_[i]]
        j = int(np.nanargmax(np.where(gapped_[i], RGAP[i], -1))) if gapped_[i].any() else None
        print(f"    {lab} mu_C {mu:3.1f}   {t_.min():.1f}-{t_.max():.1f}" + (f"             {g_.min():.1f}-{g_.max():.1f}"
              if len(g_) else "             none        ") + (f"         {RGAP[i, j]:.0f} ({REZ[j]:.1f})" if j is not None
                                                             else "         0 (gapless)") + f"   {REG_MU_TXT[mu]}", flush=True)
    print("    bond pick   Gamma (ueV) at the two zeros   Gamma / k_B (mK)   Gamma / Delta_top,max(mu_C 0.7)", flush=True)
    d07 = np.nanmax(np.where(gapped_[int(np.argmin(np.abs(RMU - 0.7)))], RGAP[int(np.argmin(np.abs(RMU - 0.7)))], np.nan))
    for t in TAU_LEADS:
        g = GAM_R[t]
        print(f"    {REG_TAU.get(t, '  ')} tau {t:<6g}  {g[0]:6.2f} / {g[1]:6.2f}                {g.mean() / st.K_B:6.0f}"
              f"             {g.mean() / d07:.2f}   {REG_TAU_TXT.get(t, '(reference)')}", flush=True)
    TRS = {}
    print("    cached coarse maps (E_Z 0-2 x 31 phases, L 2 um): kappa > 0.25 share, neutral share of those "
          "(|G| < 0.1 kappa)", flush=True)
    for mu, taus in REG_TR:
        for t in taus:
            v = np.load(st.DATA / f'{tr_name("mixed", mu, t, 2000.0)}.npz')['values']
            k_, G_ = v[..., 0], v[..., 1]
            big = k_ > 0.25
            TRS[mu, t] = (big.mean(), (big & (np.abs(G_) < 0.1 * k_)).sum() / max(big.sum(), 1))
            print(f"      mu_C {mu:3.1f}  tau {t:<6g}  {TRS[mu, t][0] * 100:5.1f} %   neutral {TRS[mu, t][1]:.2f}", flush=True)

    with plt.rc_context(CL_RC):
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), layout='constrained', gridspec_kw=dict(width_ratios=(1.25, 1, 1)))
        a = axs[0]
        dz, dm = REZ[1] - REZ[0], RMU[1] - RMU[0]
        ez_e = np.r_[REZ - dz / 2, REZ[-1] + dz / 2]
        mu_e = np.r_[RMU - dm / 2, RMU[-1] + dm / 2]
        pm = a.pcolormesh(ez_e, mu_e, np.where(gapped_, RGAP, np.nan), cmap='Greys', vmin=0, vmax=80, shading='flat')
        for i, j in zip(*np.nonzero(gapless_)):
            a.add_patch(Rectangle((REZ[j] - dz / 2, RMU[i] - dm / 2), dz, dm, fill=False, hatch='////', edgecolor='#c0392b',
                                  lw=0))
        for mu, lab in REG_MU.items():
            a.axhline(mu, color=REG_COL[mu], lw=2.2)
            a.text(2.02, mu + 0.03, f'{lab}  $\\mu_C$ {mu:g}', color=CL_INK, va='bottom', ha='right', fontsize=10.5)
        a.set_xlim(ez_e[0], ez_e[-1])
        a.set_ylim(mu_e[0], mu_e[-1])
        a.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        a.set_ylabel(r'strip chemical potential $\mu_C$ (meV)')
        a.set_title(r'(a) bulk at $\phi = \pi$: grey = gapped $Q = -1$ (shade $\Delta_{top}$),' '\n'
                    r'red hatch = $Q = -1$ but gapless, white = trivial', loc='left', fontsize=9.5, color=CL_MUT)
        cb = fig.colorbar(pm, ax=a, location='bottom', shrink=0.6, pad=0.02, aspect=30)
        cb.set_label(r'$\Delta_{top}$ ($\mu$eV), gapped $Q = -1$ only', fontsize=10.5)
        cb.set_ticks([0, 20, 40, 60, 80])

        a = axs[1]
        tt = np.array(TAU_LEADS)
        gm = np.array([GAM_R[t].mean() for t in tt])
        a.errorbar(tt, gm, yerr=[gm - [GAM_R[t].min() for t in tt], [GAM_R[t].max() for t in tt] - gm], fmt='o', ms=8,
                   color=CL_INK, capsize=3, lw=1.2, zorder=3)
        a.plot(tt, gm, color=CL_INK, lw=1.0, zorder=2)
        for t in tt:
            if t in REG_TAU:
                a.annotate(f'{REG_TAU[t]} {REG_TAU_TXT[t]}', (t, GAM_R[t].mean()), xytext=(-10, -16) if t == 1.0 else (10, -14),
                           textcoords='offset points', ha='right' if t == 1.0 else 'left', fontsize=10, color=CL_INK)
        for T in (10, 20, 50):
            a.axhline(st.K_B * T * 1e-3 * 1e3, color=CL_MUT, lw=0.9, ls='--')
            a.text(0.19, st.K_B * T * 1e-3 * 1e3 * 1.08, f'$k_BT$ at {T} mK', fontsize=9, color=CL_MUT, ha='left')
        for mu in REG_MU:
            i = int(np.argmin(np.abs(RMU - mu)))
            if gapped_[i].any():
                d = np.nanmax(np.where(gapped_[i], RGAP[i], np.nan))
                a.axhline(d, color=REG_COL[mu], lw=1.6)
                a.text(1.02, d, rf'$\Delta_{{top}}$ max, {REG_MU[mu]}', fontsize=9, color=CL_INK, va='center', ha='left',
                       transform=a.get_yaxis_transform(), clip_on=False)
        a.set_yscale('log')
        a.set_xlim(0.15, 1.05)
        a.set_xlabel(r'probe bond $\tau$')
        a.set_ylabel(r'probe width $\Gamma$ ($\mu$eV)')
        a.grid(color=CL_GRID, lw=0.5, which='both')
        a.set_title(r'(b) $\Gamma(\tau)$ (local Andreev half width, $\mu_C$ 0.7,' '\n'
                    r'two $\epsilon$ zeros) vs the gap and $k_BT$', loc='left', fontsize=9.5, color=CL_MUT)

        a = axs[2]
        for mu, taus in (REG_TR[1], REG_TR[0], REG_TR[2]):                    # mu_C 0.3 drawn over 0.7 (nearly equal)
            x = np.array(taus) + {0.3: -0.015, 0.7: 0.0, 1.2: 0.015}[mu]
            ns = np.array([TRS[mu, t][1] for t in taus])
            fr = np.array([TRS[mu, t][0] for t in taus])
            lab = (rf'$\mu_C$ {mu:g} ({REG_MU[mu]})' if mu in REG_MU else rf'$\mu_C$ {mu:g} (cached, nearest to R3)')
            a.plot(x, ns, color=REG_COL[mu], lw=2.0, zorder=2, label=lab)
            a.scatter(x, ns, s=40 + 900 * fr, color=REG_COL[mu], edgecolor='white', lw=1.5, zorder=3)
        a.set_xlim(0.15, 1.05)
        a.set_ylim(-0.03, 1.05)
        a.set_xlabel(r'probe bond $\tau$')
        a.set_ylabel(r'neutral share of $\kappa > 0.25$ ($|G| < 0.1\,\kappa$)')
        a.grid(color=CL_GRID, lw=0.5)
        a.legend(frameon=False, fontsize=9.5, loc='lower left')
        a.text(0.98, 0.98, r'marker area $\propto$ share of the map with $\kappa > 0.25$' '\n' r'R4 ($\mu_C$ 1.6): no transport yet',
               transform=a.transAxes, ha='right', va='top', fontsize=9, color=CL_MUT)
        a.set_title(r'(c) cached coarse maps ($E_Z$ 0-2 meV $\times$ 31 phases, L 2 $\mu$m;' '\n'
                    r'$\mu_C$ 0.3 / 1.2 shifted by $\mp$0.015 in $\tau$ for visibility)', loc='left', fontsize=9.5, color=CL_MUT)
        fig.suptitle(r'Choosing regimes: the gate sets protection (gap, gapless strip), the probe bond sets $\Gamma$ against '
                     r'$\Delta_{top}$ and $k_BT$', x=0.01, ha='left', fontsize=13.5, fontweight='semibold')
        st.save_fig(fig, 'regimes_mixed', dpi=150)
        plt.close(fig)
    print("figure: plots/regimes_mixed.png", flush=True)



# ====================================================================================================================
# PART 'gamma'  (2026-11-24)  Gamma = the POLE WIDTH along the phi = pi line; redo of the constant-Gamma advisor figures
# ====================================================================================================================
# Definition (Capecelatro, Marciani, Campagnano & Lucignano, arXiv:2412.01702, Eqs. 16-18): quasi-bound states are the
# poles z of the junction's retarded Green's function (sub-gap sheet), Gamma = -Im z.  Computed with tools.nh_spectrum
# (pencil (H_C + Sigma(0)) psi = lam M psi; M^-1 = their weight matrix Z, linearised; error ~ (lam / Delta)^2).
# Per field of the study_mixed [oscillation] transport line (E_Z 0.10-1.35 step 0.01, phi pi, mu_C 0.7, L 2 um, a 10,
# tc 1) seven probe configurations (tau_L, tau_R): (0, 0) closed; (t, 0), (0, t), (t, t) for t = 0.3387, 0.5.
#   Gamma_L = -Im(l1 + l2) of the left-probe-only pair (the end-mode width at an eps zero, the summed pair width
#   elsewhere), Gamma_R likewise; two-probe pair l1, l2 -> eps_eff^2 = |l1 l2| - Gamma_L Gamma_R ([nh]);
#   x = eps_eff / sqrt(Gamma_L Gamma_R); det r_pole = 1 - 2 Gamma_L Gamma_R / |l1 l2|.
# Compared with the cached NEGF line (scharf_absOM_line_mixed_tau*) and with the old collapse (ONE constant Gamma per
# tau = local-Andreev HWHM at two eps zeros, scharf_absOM_gamma_mixed; slides_nov.collapse_mixed / mixed_overlap /
# mixed_oscillation).  Cache scharf_absGM_poles_mixed_a10_pi; figure plots/gamma_line_mixed.png.
GM_EZ = np.round(np.arange(0.10, 1.3501, 0.005), 3)[::2]          # = study_mixed EZ_T (the transport line)
GM_TAUS = (0.3387, 0.5)
GM_CFG = ((0.0, 0.0),) + tuple(c for t in GM_TAUS for c in ((t, 0.0), (0.0, t), (t, t)))
GM_K = 8
GM_PAIR = 0.94                                                    # meV: two-Majorana points end here (L / xi_M small)


def gm_field(E_Z):
    """Poles (GM_K Re, then GM_K Im; ueV) for every GM_CFG at one field, phi = pi, flattened.  One q for all configurations,
    so the ribbon self-energies are computed once per field (tools._RIB_CACHE)."""
    q = junction('mixed', 0.7, 0.5)                               # tau of q unused: nh_spectrum takes tau_L, tau_R
    out = []
    for tl, tr in GM_CFG:
        P, _ = st.nh_spectrum(q, float(E_Z), [np.pi], tl, tr, k=GM_K)
        out.append(np.r_[P[0].real, P[0].imag])
    return np.array(out).ravel()


# %% [gamma] --- 1. SCAN: the poles per field and probe configuration ---
if run_part('gamma'):
    GMV = st.run_scan(gm_field, {'E_Z': GM_EZ}, 'scharf_absGM_poles_mixed_a10_pi', n_workers=6, chunksize=1)
    GMV = GMV.reshape(len(GM_EZ), len(GM_CFG), 2, GM_K)
    print("gamma: scan done", flush=True)


# %% [gamma] --- 2. Gamma(E_Z) from the poles; kappa vs x with the per-field pole Gamma vs the old constant Gamma ---
if run_part('gamma'):
    E = GM_EZ
    lam = GMV[:, :, 0, :] + 1j * GMV[:, :, 1, :]                  # (n_EZ, n_cfg, K) ueV
    CI = {c: i for i, c in enumerate(GM_CFG)}
    gm_pair = lambda c: lam[:, CI[c], :2]
    eps_box = np.load(st.DATA / 'scharf_absOM_box_mixed_a10_L2.npz')['values'][::2, 0]       # ueV, same grid
    eps_cl = np.abs(gm_pair((0.0, 0.0)).real).mean(axis=1)
    ok_cl = np.abs(eps_cl - eps_box) <= np.maximum(0.05 * eps_box, 0.01)
    _gd = np.load(st.DATA / 'scharf_absOM_gamma_mixed.npz')['values'][:, 0]
    G_OLD = {0.3387: _gd[:2].mean(), 0.5: _gd[2:].mean()}
    inpair = E <= GM_PAIR + 1e-9
    print(f"\n  [gamma] phi = pi, E_Z {E[0]:.2f}-{E[-1]:.2f} (step {E[1] - E[0]:.2f}); closed pair |Re| = eps_box (5 % or 0.01 ueV)"
          f" on {ok_cl.sum()} / {len(E)} fields (fails at E_Z {np.round(E[~ok_cl], 2)})", flush=True)
    k2m = lambda x: 2 * x ** 2 / (1 + x ** 2) ** 2
    GR = {}
    # eta: the ribbons' absorption gives each mode eta0 (closed pair: 2 eta0 in total, ~0.0025 ueV).  One-probe pair sum =
    # Gamma_L + 2 eta0, two-probe product = eps^2 + (Gamma_L + eta0)(Gamma_R + eta0) -> subtract ONE eta0 so that eps_eff
    # is exact (subtracting both left a spurious eps_eff ~ sqrt(2 Gamma eta0) ~ 0.05 ueV: |kappa - model| ~ 0.005 at low
    # field; tested 0 / 0.5 / 1 x the closed loss: medians 0.0033 / 0.0003 / 0.0042 at tau 0.3387).
    eta0 = -gm_pair((0.0, 0.0)).imag.sum(axis=1) / 2
    print(f"    eta loss per closed-pair mode (subtracted once from Gamma_L, Gamma_R): {np.median(eta0):.5f} ueV", flush=True)
    for t in GM_TAUS:
        gL = -gm_pair((t, 0.0)).imag.sum(axis=1) - eta0
        gR = -gm_pair((0.0, t)).imag.sum(axis=1) - eta0
        l2 = gm_pair((t, t))
        prod = np.abs(l2[:, 0] * l2[:, 1])
        s = np.sqrt(gL * gR)
        ee = np.sqrt(np.clip(prod - gL * gR, 0.0, None))
        V = np.load(st.DATA / f'scharf_absOM_line_mixed_tau{t:g}.npz')['values']
        k, G, d = V[:, 0], V[:, 1], V[:, 3]
        neu = np.abs(G) < 0.1 * np.maximum(k, 1e-12)
        sel = inpair & ok_cl & neu
        x = dict(old=eps_box / G_OLD[t], box=eps_box / s, eff=ee / s)
        res = {n: np.abs(k - k2m(v)) for n, v in x.items()}
        dpole = 1 - 2 * gL * gR / prod
        GR[t] = dict(gL=gL, gR=gR, s=s, ee=ee, k=k, G=G, neu=neu, sel=sel, x=x, res=res, dpole=dpole, d=d, l2=l2)
        zi = [int(np.argmin(np.abs(E - z))) for z in (0.285, 0.345)]
        big = sel & (k > 0.3)
        print(f"    tau {t}: Gamma_L(E_Z) on the pair range {gL[inpair & ok_cl].min():.2f}-{gL[inpair & ok_cl].max():.2f} ueV "
              f"(old constant {G_OLD[t]:.2f}); |Gamma_R / Gamma_L - 1| max {np.max(np.abs(gR / gL - 1)[inpair & ok_cl]):.1e}; "
              f"at the eps zeros 0.285 / 0.345: pole {gL[zi[0]]:.3f} / {gL[zi[1]]:.3f} vs Lorentzian "
              f"{_gd[0 if t < 0.4 else 2]:.3f} / {_gd[1 if t < 0.4 else 3]:.3f}", flush=True)
        print(f"      |kappa - 2x^2/(1+x^2)^2| on {sel.sum()} neutral pair-range points (median / 90 % / max; kappa > 0.3: "
              f"{big.sum()} pts):", flush=True)
        for n, lab in (('old', 'x = eps_box / Gamma_const (old slides)'), ('box', 'x = eps_box / Gamma_pole(E_Z)'),
                       ('eff', 'x = eps_eff / Gamma_pole(E_Z) (poles only)')):
            r = res[n][sel]
            print(f"        {lab:44s} {np.median(r):.4f} / {np.percentile(r, 90):.4f} / {r.max():.4f};  kappa > 0.3: "
                  f"median {np.median(res[n][big]) if big.any() else np.nan:.4f}", flush=True)
        print(f"      det r: NEGF vs 1 - 2 G_L G_R / |l1 l2| on the same points: median {np.median(np.abs(d - dpole)[sel]):.4f},"
              f" max {np.max(np.abs(d - dpole)[sel]):.4f}", flush=True)

    with plt.rc_context(CL_RC):
        GCOL = {0.3387: CL_OUT, 0.5: CL_IN}
        fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.0), layout='constrained')
        a = axs[0, 0]
        a.semilogy(E, np.where(eps_box > 0, eps_box, np.nan), 'o', ms=3.2, color=CL_MUT, label=r'splitting $\varepsilon$ (closed box)')
        for t in GM_TAUS:
            g = GR[t]
            a.semilogy(E, np.where(ok_cl, g['gL'], np.nan), color=GCOL[t], lw=2.0,
                       label=rf'$\Gamma$ = pole width, $\tau$ = {t:.2f}')
            a.axhline(G_OLD[t], color=GCOL[t], lw=1.2, ls='--')
        a.plot([], [], color=CL_INK, lw=1.2, ls='--', label=r'old: one constant $\Gamma$ (Andreev half width at 2 zeros)')
        a.axvspan(GM_PAIR, E[-1], color=CL_GRID, lw=0, zorder=0)
        a.set_ylim(1e-3, 3e2)
        a.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        a.set_ylabel(r'energy ($\mu$eV)')
        a.legend(frameon=False, fontsize=9.5, loc='upper left')
        a.set_title(r'(a) the pole width drifts with the field; grey: $E_Z$ > 0.94 (no separated end pair)', loc='left',
                    fontsize=10, color=CL_MUT)
        a = axs[0, 1]
        for t in GM_TAUS:
            g = GR[t]
            m = g['sel']
            a.semilogy(E[m], np.maximum(g['res']['old'][m], 1e-5), 'o', ms=5, mfc='none', mec=GCOL[t], mew=1.2,
                       label=rf'old constant $\Gamma$, $\tau$ = {t:.2f}')
            a.semilogy(E[m], np.maximum(g['res']['eff'][m], 1e-5), 'o', ms=5, color=GCOL[t], mec='white', mew=0.5,
                       label=rf'pole $\Gamma(E_Z)$ and $\varepsilon_{{eff}}$, $\tau$ = {t:.2f}')
        a.set_xlabel(r'Zeeman energy $E_Z$ (meV)')
        a.set_ylabel(r'$|\kappa_{NEGF} - 2x^2/(1+x^2)^2|$')
        a.legend(frameon=False, fontsize=9.5, loc='upper left', ncol=2)
        a.set_title('(b) deviation from the two-Majorana form, neutral points of the pair range', loc='left', fontsize=10,
                    color=CL_MUT)
        xx = np.geomspace(1e-3, 30, 400)
        for a, n, ttl in ((axs[1, 0], 'old', r'(c) old: $x = \varepsilon_{box}/\Gamma_{const}$'),
                          (axs[1, 1], 'eff', r'(d) new: $x = \varepsilon_{eff}/\Gamma_{pole}(E_Z)$, all from the poles')):
            a.semilogx(xx, k2m(xx), color=CL_INK, lw=2.2, label=r'$2x^2/(1+x^2)^2$')
            for t in GM_TAUS:
                g = GR[t]
                xv = np.maximum(g['x'][n], 1e-3)
                m, c = g['sel'], inpair & ok_cl & ~g['neu']
                a.semilogx(xv[m], g['k'][m], 'o', ms=6, color=GCOL[t], mec='white', mew=0.6, label=rf'$\tau$ = {t:.2f}')
                a.semilogx(xv[c], g['k'][c], 'o', ms=6, mfc='none', mec=GCOL[t], mew=1.2)
            a.plot([], [], 'o', ms=6, mfc='none', mec=CL_INK, mew=1.2, label=r'open: charged ($|G| > 0.1\,\kappa$)')
            a.axhline(0.5, color=CL_MUT, lw=0.8, ls=':')
            a.set_xlim(1e-3, 30)
            a.set_ylim(0, 0.62)
            a.set_xlabel('x')
            a.set_ylabel(r'$\kappa/\kappa_0$ (NEGF)')
            a.legend(frameon=False, fontsize=9.5, loc='upper left')
            a.set_title(ttl, loc='left', fontsize=10, color=CL_MUT)
        fig.suptitle(r'$\Gamma$ as the pole width (Capecelatro et al.) along $\phi = \pi$: the per-field $\Gamma$ '
                     'replaces the constant one in the collapse', x=0.01, ha='left', fontsize=13, fontweight='semibold')
        st.save_fig(fig, 'gamma_line_mixed', dpi=150)
        plt.close(fig)
    print("figure: plots/gamma_line_mixed.png", flush=True)



# ====================================================================================================================
# PART 'classes'  (2026-11-24)  PLOTS ONLY: criteria that sort a working point into regimes, tested on the phi = pi line
# ====================================================================================================================
# Each point (mu_C, E_Z, phi, L, tau) gets four labels, from quantities that need NO transport; transport then tests them.
#   A  bulk topology (infinite junction; independent of L, tau):  Q (Pfaffian) and the refined gap Delta_top
#        trivial (Q = +1) | gapped (Q = -1, Delta_top > CLS_GAP) | gapless (Q = -1, Delta_top <= CLS_GAP)
#        CLS_GAP 0.01 ueV: Delta_top is bimodal (< 1e-3 or > 0.08 ueV, [deck] 4) -- the same cut as [leads] CLOSED_L.
#   B  end separation (finite junction, independent of tau):  L / xi_M (bulk slowest E = 0 mode)
#        separated (>= 4) | overlapping (1.5-4) | merged (< 1.5)
#        4: neutral kappa > 0.45 switches on at L / xi_M 4.0-4.2 for L 2 / 3 / 4 um ([deck] LENGTH); 1.5: first charged
#        peaks at L / xi_M 1.5-1.8, and the two-Majorana points end at 0.94 meV = L / xi_M 1.4 ([oscillation]).
#   C  splitting vs probe width (needs tau):  x = eps_eff / sqrt(Gamma_L Gamma_R), all from the poles ([gamma] / [nh])
#        decoupled (x < sqrt2 - 1) | resonant (sqrt2 - 1 .. sqrt2 + 1, core 0.82-1.22) | split (x > sqrt2 + 1)
#        sqrt2 -+ 1 = the half-maximum points of kappa = 2x^2/(1+x^2)^2 (kappa >= 1/4); core = |kappa - 1/2| < 0.02
#        <=> |det r| < 0.2 ([nh]).
#   D  isolation of the pair (needs tau):  Gamma / E_next, E_next = the lowest NON-pair level of the closed junction
#        (third pole of the (0, 0) pencil)  -- Capecelatro et al. Eq. 25: width << distance to other spectral weight
#        isolated (< 1/3) | marginal (1/3-1) | overlapping (>= 1)   [PROVISIONAL cuts: HWHM = pole to 1.6 % at
#        Gamma / E_next 0.36 (tau 0.7), fails at 2.0 (tau 1); to be tested with the pole-dominance check]
# Tests (transport, scharf_absOM_line_mixed_tau*): kappa, neutrality |G| < 0.1 kappa, two-Majorana residual
# |kappa - (1 - det r^2)/2|.  Data: [oscillation] bulk modes (scharf_absOM_modes_mixed_a10), [gamma] poles.
# Figure plots/classes_line_mixed.png.
CLS_GAP = 0.01                                                     # ueV
CLS_SEP = (1.5, 4.0)                                               # L / xi_M
CLS_X = (np.sqrt(2) - 1, np.sqrt(2) + 1)                           # half maximum of kappa(x)
CLS_CORE = (0.82, 1.22)                                            # |det r| < 0.2
CLS_ISO = (1 / 3, 1.0)                                             # Gamma / E_next (provisional)
CLS_ROWS = {'topology': (('trivial', '#d8d6d0'), ('gapped', '#2a78d6'), ('gapless', '#c0392b')),
            'separation': (('merged', '#a9cdf5'), ('overlapping', '#3987e5'), ('separated', '#0d366b')),
            'x': (('decoupled', '#8a7ee0'), ('resonant', '#1baf7a'), ('split', '#eb6834')),
            'isolation': (('isolated', '#b9e4d0'), ('marginal', '#f2c14e'), ('overlapping', '#c0392b'))}


def cls_topology(Q, gap_ueV):
    return np.where(Q > 0, 0, np.where(gap_ueV > CLS_GAP, 1, 2))


def cls_separation(L_over_xi):
    return np.digitize(L_over_xi, CLS_SEP)                        # 0 merged, 1 overlapping, 2 separated


def cls_x(x):
    return np.digitize(x, CLS_X)                                   # 0 decoupled, 1 resonant, 2 split


def cls_isolation(ratio):
    return np.digitize(ratio, CLS_ISO)                             # 0 isolated, 1 marginal, 2 overlapping


def cls_runs(E, labels):
    """Contiguous runs of equal labels along the grid E: [(start, end, label)] with edges half a step out."""
    h = (E[1] - E[0]) / 2
    cut = np.flatnonzero(np.diff(labels)) + 1
    return [(E[a] - h, E[b - 1] + h, labels[a]) for a, b in zip(np.r_[0, cut], np.r_[cut, len(E)])]


if run_part('classes'):
    E = GM_EZ
    _m = np.load(st.DATA / 'scharf_absOM_modes_mixed_a10.npz')
    assert np.allclose(_m['axis_E_Z'][::2], E)
    MV = _m['values'][::2]
    xi_M, gap, Qb = MV[:, 0], MV[:, 6] * 1e3, MV[:, 7]
    Ls = 2000.0 / xi_M
    lam = np.load(st.DATA / 'scharf_absGM_poles_mixed_a10_pi.npz')['values'].reshape(len(E), len(GM_CFG), 2, GM_K)
    lam = lam[:, :, 0, :] + 1j * lam[:, :, 1, :]
    CI = {c: i for i, c in enumerate(GM_CFG)}
    eps_cl = np.abs(lam[:, CI[(0.0, 0.0)], :2].real).mean(axis=1)
    E_next = np.abs(lam[:, CI[(0.0, 0.0)], 2].real)
    eta0 = -lam[:, CI[(0.0, 0.0)], :2].imag.sum(axis=1) / 2
    LAB = {'topology': cls_topology(Qb, gap), 'separation': cls_separation(Ls)}
    TRL, XS = {}, {}
    for t in GM_TAUS:
        gL = -lam[:, CI[(t, 0.0)], :2].imag.sum(axis=1) - eta0
        gR = -lam[:, CI[(0.0, t)], :2].imag.sum(axis=1) - eta0
        l2 = lam[:, CI[(t, t)], :2]
        s = np.sqrt(gL * gR)
        x = np.sqrt(np.clip(np.abs(l2[:, 0] * l2[:, 1]) - gL * gR, 0.0, None)) / s
        XS[t] = dict(G=s, x=x, iso=s / E_next)
        LAB[f'x{t}'] = cls_x(x)
        LAB[f'iso{t}'] = cls_isolation(s / E_next)
        TRL[t] = np.load(st.DATA / f'scharf_absOM_line_mixed_tau{t:g}.npz')['values']
    print(f"\n  [classes] phi = pi, mu_C 0.7, L 2 um, E_Z {E[0]:.2f}-{E[-1]:.2f}: Q = -1 on all fields, gap "
          f"{gap.min():.2f}-{gap.max():.2f} ueV; E_next {E_next.min():.1f}-{E_next.max():.1f} ueV", flush=True)
    for name in ('topology', 'separation'):
        names = [n for n, _ in CLS_ROWS[name]]
        print(f"    {name:10s}: " + ', '.join(f"{names[l_]} {s_:.3f}-{e_:.3f}" for s_, e_, l_ in cls_runs(E, LAB[name])),
              flush=True)
    print("    TEST against transport (all fields; neutral = |G| < 0.1 kappa among kappa > 0.02; 2M = |kappa - (1 - det r^2)/2|):",
          flush=True)
    print("      tau     separation   x class      n   kappa median  kappa > 1/4  neutral  2M residual median", flush=True)
    for t in GM_TAUS:
        k, G, d = TRL[t][:, 0], TRL[t][:, 1], TRL[t][:, 3]
        res = np.abs(k - (1 - d ** 2) / 2)
        for sp in (2, 1, 0):
            for xc in (0, 1, 2):
                m = (LAB['separation'] == sp) & (LAB[f'x{t}'] == xc)
                if not m.any():
                    continue
                hot = m & (k > 0.02)
                neu = np.mean(np.abs(G[hot]) < 0.1 * k[hot]) if hot.any() else np.nan
                print(f"      {t:<7g} {CLS_ROWS['separation'][sp][0]:12s} {CLS_ROWS['x'][xc][0]:10s} {m.sum():4d}   "
                      f"{np.median(k[m]):10.3f}   {np.mean(k[m] > 0.25):9.2f}   {neu:6.2f}   {np.median(res[m]):.4f}", flush=True)
        core = (XS[t]['x'] > CLS_CORE[0]) & (XS[t]['x'] < CLS_CORE[1])
        print(f"      {t:<7g} x in the core 0.82-1.22: {core.sum()} fields, kappa {k[core].min() if core.any() else np.nan:.3f}-"
              f"{k[core].max() if core.any() else np.nan:.3f}; isolation Gamma / E_next {XS[t]['iso'].min():.3f}-"
              f"{XS[t]['iso'].max():.3f} (classes present: {sorted(set(LAB[f'iso{t}']))})", flush=True)

    with plt.rc_context(CL_RC):
        fig, axs = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=True, layout='constrained',
                                gridspec_kw=dict(height_ratios=(1.25, 1.0, 1.0)))
        a = axs[0]
        a.semilogy(E, np.maximum(eps_cl, 1e-3), 'o', ms=3, color=CL_MUT, label=r'splitting $\varepsilon$ (closed junction)')
        a.semilogy(E, gap, color=CL_INK, lw=1.6, ls='--', label=r'bulk gap $\Delta_{top}$')
        a.semilogy(E, E_next, color=CL_INK, lw=1.6, label=r'next level $E_{next}$ (closed junction)')
        for t, c in zip(GM_TAUS, (CL_OUT, CL_IN)):
            a.semilogy(E, XS[t]['G'], color=c, lw=2.0, label=rf'probe width $\Gamma$ (pole), $\tau$ = {t:.2f}')
        a.set_ylim(1e-3, 2e2)
        a.set_ylabel(r'energy ($\mu$eV)')
        a.legend(frameon=False, fontsize=9.5, ncol=3, loc='lower right')
        a.set_title(r'(a) the four energies that set the regime', loc='left', fontsize=10.5, color=CL_MUT)
        ax2 = a.twiny()
        ax2.set_xlim(E[0], E[-1])
        tk = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        ax2.set_xticks(tk)
        ax2.set_xticklabels([f'{np.interp(v, E, Ls):.1f}' for v in tk])
        ax2.set_xlabel(r'$L/\xi_M$', fontsize=11)
        a = axs[1]
        rows = [('topology', 'topology', 'A  bulk topology'), ('separation', 'separation', r'B  separation $L/\xi_M$')]
        for t in GM_TAUS:
            rows += [(f'x{t}', 'x', rf'C  $x = \varepsilon/\Gamma$, $\tau$ {t:.2f}'),
                     (f'iso{t}', 'isolation', rf'D  $\Gamma/E_{{next}}$, $\tau$ {t:.2f}')]
        for r, (key, kind, lab) in enumerate(rows):
            y = len(rows) - 1 - r
            for s_, e_, l_ in cls_runs(E, LAB[key]):
                a.broken_barh([(s_, e_ - s_)], (y + 0.12, 0.76), color=CLS_ROWS[kind][l_][1], lw=0)
            a.text(E[0] - 0.012, y + 0.5, lab, ha='right', va='center', fontsize=10)
        a.set_ylim(0, len(rows))
        a.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for kind in ('separation', 'x', 'isolation') for _, c in CLS_ROWS[kind]]
        names = [n for kind in ('separation', 'x', 'isolation') for n, _ in CLS_ROWS[kind]]
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=CLS_ROWS['topology'][1][1]))
        names.insert(0, 'gapped Q = -1')
        a.legend(handles, names, frameon=False, fontsize=9, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.02))
        a.set_title('(b) the labels (no transport used)', loc='left', fontsize=10.5, color=CL_MUT)
        a = axs[2]
        for t, c in zip(GM_TAUS, (CL_OUT, CL_IN)):
            k, G = TRL[t][:, 0], TRL[t][:, 1]
            a.plot(E, k, color=c, lw=1.8, label=rf'$\kappa/\kappa_0$, $\tau$ = {t:.2f}')
            a.plot(E, np.abs(G), color=c, lw=1.0, ls=':', label=rf'$|G|/G_0$, $\tau$ = {t:.2f}')
        a.axhline(0.5, color=CL_MUT, lw=0.8, ls='--')
        a.axhline(0.25, color=CL_MUT, lw=0.6, ls=':')
        a.set_ylim(0, 0.65)
        a.set_ylabel('NEGF')
        a.set_xlabel(r'Zeeman energy $E_Z$ (meV)   ($\phi = \pi$, $\mu_C$ 0.7, L 2 $\mu$m)')
        a.legend(frameon=False, fontsize=9.5, ncol=4, loc='upper left')
        a.set_title('(c) the test: transport', loc='left', fontsize=10.5, color=CL_MUT)
        for a in axs:
            a.set_xlim(E[0], E[-1])
        fig.suptitle('Regime criteria on one line: bulk topology, end separation, splitting vs probe width, isolation of the pair',
                     x=0.01, ha='left', fontsize=13, fontweight='semibold')
        st.save_fig(fig, 'classes_line_mixed', dpi=150)
        plt.close(fig)
    print("figure: plots/classes_line_mixed.png", flush=True)



# ====================================================================================================================
# PART 'dis'  (2026-11-25)  ONSITE DISORDER, uncorrelated (seeded), in the thesis junction: conductances, spectrum, LDOS
# ====================================================================================================================
# Model (my_functions.junction.column_onsite, unchanged): every site of the NORMAL strip (nx x ny = 200 x 10 at a 10)
# gets V_i uniform in [-W/2, W/2] meV (rms W / sqrt 12), entering as V_i tau_z; default_rng(seed) -> the same seed gives
# the same pattern at every W (scaled).  Ribbons (semi-infinite leads) and probes stay clean.  tools._nh_setup now
# includes the disorder (fixed 2026-11-25; checked: pole det r = 1 - 2 G_L G_R / |l1 l2| vs RGF det r 0.550 / 0.577,
# 0.405 / 0.422, -1.02 / -0.97 at W 2 / 2 / 6 meV).  Old disorder results (CLAUDE.md "Disorder") are OLD bonds / a 20.
# Thesis junction: mixed SOC, mu_C 0.7, L 2 um, a 10, tc 1; probe tau 0.5 (+ tau 1 for the strong-probe Andreev / det r).
# Working points = one per regime class of [classes] (tau 0.5 clean line) + a trivial one:
#   sep    (E_Z 0.30, pi)  separated + decoupled: kappa 0.0004, local Andreev 1.996, det r -0.998
#   res    (0.72, pi)      overlapping + resonant: kappa 0.499 neutral
#   merged (1.06, pi)      merged: kappa 0.523, G 0.214 (charged false positive)
#   triv   (0.30, 0)       trivial (Q = +1 below the V of the Q map)
# Cell 1: per (point, W, seed): transport at tau 0.5 and tau 1 + poles (0,0), (t,0), (0,t), (t,t) (k 8).  ~25 s each.
# Cell 2: LDOS (probes attached at tau 0.5, transport eta): E = 0 map on all 200 x 10 sites, and LDOS(E) -40..40 ueV on
#         the end / middle columns; seed 0 (~3.5 min per task).
# Cell 3: numbers + figures plots/dis_points_mixed.png, dis_ldos_mixed.png.
DIS_TAU = 0.5
DIS_W = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)                            # meV (box width)
DIS_SEEDS = np.arange(16)
DIS_PTS = (('sep', 0.30, np.pi), ('res', 0.72, np.pi), ('merged', 1.06, np.pi), ('triv', 0.30, 0.0))
DIS_CFG = ((0.0, 0.0), (DIS_TAU, 0.0), (0.0, DIS_TAU), (DIS_TAU, DIS_TAU))
DIS_K = 8
DIS_LW = (0.0, 1.0, 2.0, 4.0, 8.0)                                # LDOS strengths
DIS_LSEEDS = (0,)                                                 # one LDOS task ~3.5 min on one thread (21 energies)
DIS_E = np.round(np.arange(-40.0, 40.001, 4.0), 3)                # ueV, energy-resolved LDOS (21 energies: ~3.5 min/task)
DIS_LCOLS = tuple(range(0, 10)) + tuple(range(95, 105)) + tuple(range(190, 200))   # left end, middle, right end


def dis_q(W, seed, tau=DIS_TAU):
    """The thesis junction with onsite disorder W (meV, box width) of realisation seed."""
    return st.replace(junction('mixed', 0.7, tau), disorder=float(W), disorder_seed=int(seed))


def dis_point(i, W, seed):
    """[transport tau 0.5 (kappa, G, andreev, det r, n_bad), transport tau 1 (same), poles for DIS_CFG (DIS_K Re then
    DIS_K Im each, ueV)] at DIS_PTS[i]."""
    _, E_Z, phi = DIS_PTS[int(i)]
    q = dis_q(W, seed)
    tr = st.transport_point(q, E_Z, phi)
    trs = st.transport_point(st.replace(q, tc_barr_transport=1.0), E_Z, phi)
    P = [st.nh_spectrum(q, E_Z, [phi], tl, tr_, k=DIS_K)[0][0] for tl, tr_ in DIS_CFG]
    return np.r_[tr, trs, np.concatenate([np.r_[p.real, p.imag] for p in P])]


def dis_ldos(i, W, seed):
    """LDOS (e + h, summed over spin; 1/meV per site) with the probes attached: E = 0 on all nx x ny sites (flattened),
    then LDOS(E) on DIS_E for DIS_LCOLS summed over y (len(DIS_E) x len(DIS_LCOLS), flattened)."""
    _, E_Z, phi = DIS_PTS[int(i)]
    q = dis_q(W, seed)
    J = st.myf.FourTerminalJunction(q.params_transport(E_Z=E_Z))
    e0, h0 = st.myf.LocalGreen(J, np.array([0.0]), cols=list(range(J.p.nx))).ldos(phi)
    eE, hE = st.myf.LocalGreen(J, DIS_E / (q.t * 1e3), cols=list(DIS_LCOLS)).ldos(phi)
    return np.r_[((e0 + h0)[0] / q.t).ravel(), ((eE + hE).sum(axis=2) / q.t).ravel()]


# %% [dis] --- 1. SCAN: transport + poles per (point, W, seed) ---
if run_part('dis'):
    DSV = st.run_scan(dis_point, {'i': np.arange(len(DIS_PTS)), 'W': np.array(DIS_W), 'seed': DIS_SEEDS},
                      'scharf_absDIS_points_mixed_tau0.5', n_workers=6, chunksize=1)
    print("dis: points done", flush=True)


# %% [dis] --- 2. SCAN: LDOS for two seeds ---
if run_part('dis'):
    DSL = st.run_scan(dis_ldos, {'i': np.arange(len(DIS_PTS)), 'W': np.array(DIS_LW), 'seed': np.array(DIS_LSEEDS)},
                      'scharf_absDIS_ldos_mixed_tau0.5', n_workers=6, chunksize=1)
    print("dis: LDOS done", flush=True)


# %% [dis] --- 3. numbers and figures ---
if run_part('dis'):
    nP, nW, nS = len(DIS_PTS), len(DIS_W), len(DIS_SEEDS)
    tr, trs = DSV[..., :5], DSV[..., 5:10]
    lam = DSV[..., 10:].reshape(nP, nW, nS, len(DIS_CFG), 2, DIS_K)
    lam = lam[..., 0, :] + 1j * lam[..., 1, :]                     # (P, W, S, cfg, K) ueV
    eps = np.abs(lam[..., 0, :2].real).mean(axis=-1)
    E_next = np.abs(lam[..., 0, 2].real)
    eta0 = -lam[..., 0, :2].imag.sum(axis=-1) / 2
    gL = -lam[..., 1, :2].imag.sum(axis=-1) - eta0
    gR = -lam[..., 2, :2].imag.sum(axis=-1) - eta0
    prod = np.abs(lam[..., 3, 0] * lam[..., 3, 1])
    x_eff = np.sqrt(np.clip(prod - gL * gR, 0.0, None) / np.maximum(gL * gR, 1e-12))
    k, G, AND, det_s, AND_s = tr[..., 0], tr[..., 1], tr[..., 2], trs[..., 3], trs[..., 2]
    nbad = int(np.nansum(tr[..., 4]) + np.nansum(trs[..., 4]))
    print(f"\n  [dis] uncorrelated box disorder in the strip, tau {DIS_TAU} (+ tau 1), {nS} seeds; failed lead solves {nbad}",
          flush=True)
    for p_, (name, E_Z, phi) in enumerate(DIS_PTS):
        print(f"    {name} (E_Z {E_Z}, phi {phi / np.pi:g} pi):", flush=True)
        print("      W [meV]  rms   kappa med [min-max]       neutral  Q=-1 (det r_tau1<0)  Andreev_tau1 ~2  eps med  "
              "E_next med  Gamma_L med  x in resonant", flush=True)
        for w_, W in enumerate(DIS_W):
            kk, gg = k[p_, w_], G[p_, w_]
            hot = kk > 0.02
            neu = np.mean(np.abs(gg[hot]) < 0.1 * kk[hot]) if hot.any() else np.nan
            xr = np.mean((x_eff[p_, w_] > CLS_X[0]) & (x_eff[p_, w_] < CLS_X[1]))
            print(f"      {W:5.2f}   {W / np.sqrt(12):5.2f}   {np.median(kk):.3f} [{kk.min():.3f}-{kk.max():.3f}]   "
                  f"{neu:5.2f}    {np.mean(det_s[p_, w_] < 0):5.2f}               {np.mean(np.abs(AND_s[p_, w_] - 2) < 0.05):5.2f}"
                  f"          {np.median(eps[p_, w_]):7.3f}  {np.median(E_next[p_, w_]):7.2f}     "
                  f"{np.median(gL[p_, w_]):6.2f}       {xr:5.2f}", flush=True)
    nE, nC = len(DIS_E), len(DIS_LCOLS)
    LW = np.array(DIS_LW)
    p0 = junction('mixed', 0.7).params_transport(E_Z=0.3)
    nx, ny = p0.nx, p0.ny
    Lmap = DSL[..., :nx * ny].reshape(nP, len(LW), len(DIS_LSEEDS), nx, ny)
    LE = DSL[..., nx * ny:].reshape(nP, len(LW), len(DIS_LSEEDS), nE, nC)
    with plt.rc_context(CL_RC):
        WCOL = ('#1b1b1b', '#2a78d6', '#1baf7a', '#eb6834', '#c0392b', '#8a7ee0')
        xpos = np.arange(nW)
        fig, axs = plt.subplots(4, nP, figsize=(4.0 * nP + 1.2, 12.5), sharex=True, layout='constrained')
        for p_, (name, E_Z, phi) in enumerate(DIS_PTS):
            jit = (DIS_SEEDS - DIS_SEEDS.mean()) / len(DIS_SEEDS) * 0.5
            a = axs[0, p_]
            for w_ in range(nW):
                a.plot(xpos[w_] + jit, k[p_, w_], 'o', ms=3.5, color=CL_MUT, alpha=0.7)
            a.plot(xpos, np.median(k[p_], axis=1), 'o-', color=CL_INK, lw=1.8, ms=6)
            a.axhline(0.5, color=CL_MUT, lw=0.8, ls='--')
            a.set_ylim(-0.02, 1.0)
            a.set_title(f'{name}: $E_Z$ {E_Z} meV, $\\phi$ = {phi / np.pi:g}$\\pi$', loc='left', fontsize=11)
            a = axs[1, p_]
            for w_ in range(nW):
                hot = k[p_, w_] > 0.02
                a.semilogy(xpos[w_] + jit[hot], np.maximum(np.abs(G[p_, w_, hot]) / k[p_, w_, hot], 1e-5), 'o', ms=3.5,
                           color=CL_MUT, alpha=0.7)
            a.axhline(0.1, color=CL_IN, lw=1.0, ls='--')
            a.set_ylim(1e-5, 3)
            a = axs[2, p_]
            for w_ in range(nW):
                a.plot(xpos[w_] + jit, AND_s[p_, w_], 'o', ms=3.5, color=CL_MUT, alpha=0.7)
                a.plot(xpos[w_] + jit, np.where(det_s[p_, w_] < 0, AND_s[p_, w_], np.nan), 'o', ms=3.5, mfc='none',
                       mec=CL_OUT, mew=1.0)
            a.axhline(2.0, color=CL_MUT, lw=0.8, ls='--')
            a.set_ylim(-0.05, 2.6)
            a = axs[3, p_]
            for w_ in range(nW):
                a.semilogy(xpos[w_] + jit - 0.12, np.maximum(eps[p_, w_], 1e-4), 'o', ms=3.5, color=CL_INK, alpha=0.7)
                a.semilogy(xpos[w_] + jit + 0.12, E_next[p_, w_], 'D', ms=3.5, color=CL_OUT, alpha=0.7)
                a.semilogy(xpos[w_] + jit, np.sqrt(gL[p_, w_] * gR[p_, w_]), '_', ms=7, color=CL_IN, mew=1.6)
            a.set_ylim(1e-4, 3e2)
            a.set_xticks(xpos)
            a.set_xticklabels([f'{w:g}' for w in DIS_W])
            a.set_xlabel('W (meV, box width)')
        axs[0, 0].set_ylabel(r'$\kappa/\kappa_0$ ($\tau$ 0.5)')
        axs[1, 0].set_ylabel(r'$|G|/\kappa$ ($\kappa$ > 0.02)')
        axs[2, 0].set_ylabel(r'local Andreev ($\tau$ 1), $e^2/h$')
        axs[3, 0].set_ylabel(r'energy ($\mu$eV)')
        axs[0, -1].plot([], [], 'o', color=CL_MUT, ms=4, label='one seed')
        axs[0, -1].plot([], [], 'o-', color=CL_INK, label='median')
        axs[0, -1].legend(frameon=False, fontsize=9, loc='upper right')
        axs[2, -1].plot([], [], 'o', ms=4, mfc='none', mec=CL_OUT, label=r'det $r$ < 0 at $\tau$ 1')
        axs[2, -1].legend(frameon=False, fontsize=9, loc='upper right')
        axs[3, -1].plot([], [], 'o', color=CL_INK, ms=4, label=r'splitting $\varepsilon$')
        axs[3, -1].plot([], [], 'D', color=CL_OUT, ms=4, label=r'next level $E_{next}$')
        axs[3, -1].plot([], [], '_', color=CL_IN, ms=8, mew=1.6, label=r'$\Gamma$ (pole, $\tau$ 0.5)')
        axs[3, -1].legend(frameon=False, fontsize=9, loc='lower left')
        fig.suptitle(f'Uncorrelated onsite disorder in the strip ({nS} seeds per W): conductances and the closed-junction '
                     'spectrum at four working points', x=0.01, ha='left', fontsize=13, fontweight='semibold')
        st.save_fig(fig, 'dis_points_mixed', dpi=140)
        plt.close(fig)

        xs = (np.arange(nx) + 0.5) * 2000.0 / nx / 1000.0          # um
        fig, axs = plt.subplots(nP, 2, figsize=(14.5, 3.0 * nP + 1.0), layout='constrained',
                                gridspec_kw=dict(width_ratios=(1.6, 1)))
        for p_, (name, E_Z, phi) in enumerate(DIS_PTS):
            a, b = axs[p_]
            for w_, W in enumerate(LW):
                a.semilogy(xs, np.maximum(Lmap[p_, w_, 0].sum(axis=1), 1e-6), color=WCOL[w_], lw=1.3, label=f'W {W:g}')
                b.semilogy(DIS_E, np.maximum(LE[p_, w_, 0, :, :10].sum(axis=1), 1e-6), color=WCOL[w_], lw=1.6)
                b.semilogy(DIS_E, np.maximum(LE[p_, w_, 0, :, 10:20].sum(axis=1), 1e-6), color=WCOL[w_], lw=1.0, ls=':')
            a.set_title(f'{name}: $E_Z$ {E_Z} meV, $\\phi$ = {phi / np.pi:g}$\\pi$ -- LDOS at E = 0 along the junction '
                        '(summed across, seed 0)', loc='left', fontsize=10)
            b.set_title('LDOS(E): first 10 columns (solid), middle 10 (dotted)', loc='left', fontsize=10)
            a.set_ylabel('LDOS (1/meV per column)')
            a.set_xlim(0, 2)
        axs[0, 0].legend(frameon=False, fontsize=9, ncol=5, loc='upper center', title='disorder (meV)')
        axs[-1, 0].set_xlabel(r'position along the junction ($\mu$m; probes at 0 and 2)')
        axs[-1, 1].set_xlabel(r'energy $E$ ($\mu$eV)')
        fig.suptitle('LDOS with the probes attached: where the zero-energy weight sits, and how disorder fills the gap',
                     x=0.01, ha='left', fontsize=13, fontweight='semibold')
        st.save_fig(fig, 'dis_ldos_mixed', dpi=140)
        plt.close(fig)
    print("figures: plots/dis_points_mixed.png, plots/dis_ldos_mixed.png", flush=True)
