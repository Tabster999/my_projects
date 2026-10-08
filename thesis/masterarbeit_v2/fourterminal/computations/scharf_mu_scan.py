"""Scan mu_N: does kappa oscillate in mu_N, and does the Andreev plateau extend across it?

Two runs, both at phi = pi:
  (1) a fine 1D line in mu_N at E_Z = 0.3 meV
  (2) a 2D (E_Z, mu_N) map

Each point gives BOTH couplings: weak (tc_barr=0.2) for the nonlocal kappa/G, strong
(tc_barr=1.0) for the local Andreev conductance.  E_T = (pi/2) 2 sqrt(mu_N t)/n_across grows
like sqrt(mu_N), so if the plateau edge really is the Thouless energy the 2D boundary must be
a sqrt curve -- that is the falsifiable part.
"""
import sys, os, time
for _d in ("C:/coding/my_projects/thesis/masterarbeit_v2/fourterminal/computations",
           "C:/coding/my_projects/thesis/masterarbeit_v2/fourterminal"):
    sys.path.insert(0, _d)
import matplotlib; matplotlib.use("Agg")
import numpy as np
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor

OUT = os.path.dirname(os.path.abspath(__file__))     # caches land beside this script
PHI = np.pi
N_WORKERS = 8


def _one(args):
    """(mu_N, E_Z) -> (kappa, G, andreev_weak, andreev_strong, n_bad)."""
    mu_N, E_Z = args
    import scharf_ldos as S
    import my_functions as myf
    n0 = myf.leads.N_UNRELIABLE
    try:
        weak = replace(S.PP, mu_N=float(mu_N), tc_barr_transport=0.2)
        pw = weak.params_transport(E_Z=float(E_Z))
        k, G = myf.linear_response(pw, PHI, kT=0.0, side='left')
        chw = myf.RGFFourTerminal(myf.FourTerminalJunction(pw),
                                  np.array([0.0])).channels_at_phi(PHI, 'left')
        aw = float(chw['eh_local'][0] + chw['he_local'][0])

        strong = replace(S.PP, mu_N=float(mu_N), tc_barr_transport=1.0)
        ps = strong.params_transport(E_Z=float(E_Z))
        chs = myf.RGFFourTerminal(myf.FourTerminalJunction(ps),
                                  np.array([0.0])).channels_at_phi(PHI, 'left')
        a_s = float(chs['eh_local'][0] + chs['he_local'][0])
        ks, Gs = myf.linear_response(ps, PHI, kT=0.0, side='left')
        return (float(k), float(G), aw, a_s, float(ks), float(Gs),
                myf.leads.N_UNRELIABLE - n0)
    except Exception as e:
        return (np.nan,) * 6 + (repr(e),)


def grid(mus, EZs, tag):
    jobs = [(m, e) for m in mus for e in EZs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        res = list(ex.map(_one, jobs, chunksize=4))
    print(f"  {tag}: {len(jobs)} points in {time.time()-t0:.0f}s")
    sh = (len(mus), len(EZs))
    out = {}
    for i, name in enumerate(('kappa', 'G', 'andreev_weak', 'andreev_strong',
                              'kappa_strong', 'G_strong')):
        out[name] = np.array([r[i] for r in res], float).reshape(sh)
    bad = sum(1 for r in res if isinstance(r[6], str))
    if bad:
        print(f"    {bad} point(s) raised; first: "
              f"{next(r[6] for r in res if isinstance(r[6], str))}")
    out['mus'], out['E_Z'] = np.asarray(mus), np.asarray(EZs)
    return out


if __name__ == "__main__":
    import scharf_ldos as S
    t = S.PP.t
    print(f"t = {t:.4f} meV,  nx = {S.PP.n_along}, ny = {S.PP.n_across},  phi = pi")
    print(f"E_T(mu_N) = {(np.pi/2)*2*np.sqrt(t)/S.PP.n_across:.4f} * sqrt(mu_N)  meV\n")

    # ---- (1) fine line in mu_N at E_Z = 0.3 meV --------------------------------
    mus1 = np.round(np.arange(0.10, 1.501, 0.01), 3)
    print(f"(1) line: {len(mus1)} mu_N values, E_Z = 0.3 meV")
    R1 = grid(mus1, [0.3], "line")
    np.savez(f"{OUT}/scharf_mu_line.npz", **R1)

    print("\n  mu_N   kF*L/pi   kappa   |G|/k   And(weak)  And(strong)  kappa_s")
    for i, m in enumerate(mus1):
        k = R1['kappa'][i, 0]
        r = abs(R1['G'][i, 0]) / k if k > 1e-6 else np.nan
        print(f"  {m:5.2f}  {np.sqrt(m/t)*S.PP.n_along/np.pi:7.2f}  {k:7.4f} "
              f"{r:7.3f}  {R1['andreev_weak'][i,0]:9.4f}  "
              f"{R1['andreev_strong'][i,0]:10.4f}  {R1['kappa_strong'][i,0]:8.4f}")

    # ---- (2) 2D map (E_Z, mu_N) ------------------------------------------------
    mus2 = np.round(np.linspace(0.10, 1.50, 29), 4)
    EZs2 = np.round(np.linspace(0.0, 1.20, 31), 4)
    print(f"\n(2) map: {len(mus2)} x {len(EZs2)} = {len(mus2)*len(EZs2)} points")
    R2 = grid(mus2, EZs2, "map")
    np.savez(f"{OUT}/scharf_mu_map.npz", **R2)
    print("saved scharf_mu_line.npz and scharf_mu_map.npz")
