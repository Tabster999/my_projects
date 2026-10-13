"""
OLD vs SCALED interface bonds in the Scharf junction -- a copy of scharf_disorder.py with a switch.

THE ISSUE.  A bond across an interface (probe: tc_barr, ribbon: tc) is built as
    model_hop(p, dir, tc, alpha=bond_alpha)  ->  -tc tau_z  +  SOC(alpha, beta)
so tc scales only the spin-conserving hop; the SOC part stays at full strength.  At Scharf's
alpha_t = 0.29, beta_t = 0.15 that is a 0.16 t spin-flip hop that survives tc_barr = 0
(kappa = 0.21 at point A with "decoupled" probes).

THE VARIANT.  scaled=True makes every INTERFACE bond tc x one lattice bond (alpha, beta scaled by tc):
    bulk hops (t = 1)  and  tc = 1 bonds:  unchanged
    tc_barr = 0.2:                         SOC x 0.2   (the whole bond scales, as for a barrier)
    tc_barr = 0:                           truly decoupled
Since 2026-10-13 the package itself uses the scaled bonds (model_hop(..., bond=True)).  This script
keeps the OLD variant reproducible: the switch wraps model_hop and is carried by each Params
instance (attribute scale_soc), so worker processes always build the right variant.

HOW TO USE -- exactly as scharf_disorder.py, plus `scaled`:
    m = measure(W=0, seed=0, E_Z=0.505, phi=np.pi, scaled=True)
    R = sweep(W=0, seed=0, E_Z=EZ, phi=np.pi, scaled=True, pp=replace(PP, tc_barr_transport=0.2))
    R = load_or_run('name', ...)      saved as scharf_soc_<name>.npz
"""
# %% --- IMPORTS ---
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                  # interactive window: run from computations/
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
DATA = HERE / 'plot_npz'                           # where the .npz results live
import scharf_ldos as S
from scharf_ldos import myf
import my_functions.junction as _junction
import my_functions.spectra as _spectra


# %% --- THE SWITCH ---



def _model_hop_switch(p, direction, t, m0, alpha=None, bond=False, _orig=_junction.model_hop):
    """
    The package's model_hop now builds interface bonds itself (bond=True).  scale_soc=False
    rebuilds the OLD unscaled bond: -t tau_z + the full SOC of the global alpha, beta.  Untagged Params keep the package's
    behaviour, so importing this script never changes other scripts in the same session.
    """
    if bond and not getattr(p, 'scale_soc', True):
        return _orig(p, direction, t, m0, p.alpha)
    return _orig(p, direction, t, m0, alpha, bond=bond)


_model_hop_switch.is_switch = True
for _mod in (_junction, _spectra):
    if not getattr(_mod.model_hop, 'is_switch', False):        # install once per process
        _mod.model_hop = _model_hop_switch


def tagged(p, scaled):
    """Params p carrying the variant (replace() drops the tag, so tag the final object)."""
    p.scale_soc = bool(scaled)
    return p


# %% --- SETUP ---
PP = replace(S.PP, tc_barr_transport=0.2)          # weak probe for kappa and G
TC_STRONG = 1.0                                    # strong probe for the Andreev peak and det r
A = (0.505, np.pi)      # (E_Z in meV, phi): topological, on a kappa resonance
B = (0.505, 0.0)        # trivial: same field, phi = 0
C = (0.0, np.pi)        # trivial: no field

KEYS = ('kappa', 'G', 'andreev', 'kappa_s', 'G_s', 'andreev_s',
        'ldos_lo', 'ldos_hi', 'ldos_mid', 'curv_lo', 'curv_hi', 'curv_mid',
        'candidate', 'det', 'n_bad')


# %% --- ONE REALISATION (as scharf_disorder.py, plus `scaled`) ---
def _ldos(pp, phi, scaled):
    parts = {'lo': S.cols_near_end(pp, side='low'), 'hi': S.cols_near_end(pp, side='high'),
             'mid': S.cols_near_middle(pp)}
    cols = sorted(set().union(*parts.values()))
    dE = S._dE(pp)
    lg = myf.LocalGreen(myf.FourTerminalJunction(tagged(pp.params(), scaled)),
                        np.array([-dE, 0.0, dE]) / pp.t, cols=cols)
    e, h = lg.ldos(phi)
    D = (e + h).sum(axis=2)
    out = {}
    for name, c in parts.items():
        d = D[:, [cols.index(k) for k in c]].sum(axis=1)
        out[f'ldos_{name}'] = d[1]
        out[f'curv_{name}'] = (d[0] - 2 * d[1] + d[2]) / dE**2
    return out


def _transport(pp, phi, tc, scaled):
    J = myf.FourTerminalJunction(tagged(replace(pp.params_transport(), tc_barr=tc), scaled))
    rgf = myf.RGFFourTerminal(J, np.array([0.0]))
    c = {k: float(v[0]) for k, v in rgf.channels_at_phi(phi, 'left').items()}
    return c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'], rgf


def _det_r(rgf, phi):
    G_LL = rgf.green_blocks(phi)[0][0]
    w, U = np.linalg.eigh(rgf.Gamma_L[0])
    keep = w > 1e-6 * w.max()
    s, U = np.sqrt(w[keep]), U[:, keep]
    r = -np.eye(len(s)) + 1j * s[:, None] * (U.conj().T @ G_LL @ U) * s[None, :]
    return np.linalg.det(r).real


def measure(W, seed, E_Z, phi, scaled, invariant=False, pp=None):
    """Everything for ONE realisation; scaled=True uses the tc-scaled interface bonds."""
    pp = replace(PP if pp is None else pp, disorder=float(W), disorder_seed=int(seed))
    pp.B = pp.B_of_E_Z(E_Z)
    phi = float(phi)
    n_bad0 = myf.leads.N_UNRELIABLE
    m = _ldos(pp, phi, scaled)
    m['candidate'] = float(m['curv_lo'] < 0 and m['ldos_lo'] / m['ldos_mid'] > 3)
    m['kappa'], m['G'], m['andreev'], _ = _transport(pp, phi, pp.tc_barr_transport, scaled)
    m['kappa_s'], m['G_s'], m['andreev_s'], rgf = _transport(pp, phi, TC_STRONG, scaled)
    m['det'] = _det_r(rgf, phi) if invariant else np.nan
    m['n_bad'] = myf.leads.N_UNRELIABLE - n_bad0
    return m


# %% --- MANY REALISATIONS ---
def _job(args):
    return measure(*args)


def sweep(W, seed, E_Z, phi, scaled, invariant=False, n_workers=1, pp=None):
    """measure() on every combination of W, seed, E_Z, phi; arrays [iW, iseed, iE_Z, iphi]."""
    axes = {'W': np.atleast_1d(np.asarray(W, float)), 'seed': np.atleast_1d(np.asarray(seed, int)),
            'E_Z': np.atleast_1d(np.asarray(E_Z, float)), 'phi': np.atleast_1d(np.asarray(phi, float))}
    shape = tuple(len(a) for a in axes.values())
    pp = PP if pp is None else pp
    jobs = [(w, s, e, p, scaled, invariant, pp) for w, s, e, p in itertools.product(*axes.values())]
    print(f"sweep ({'scaled' if scaled else 'old'}): {len(jobs)} points on {n_workers} worker(s)")
    if n_workers == 1:
        out = list(map(_job, jobs))
    else:
        import scharf_soc_bond_compare as mod         # workers import _job from the module
        pool = myf.make_pool(n_workers)
        try:
            out = list(pool.map(mod._job, jobs, chunksize=4))
        finally:
            pool.shutdown()
    R = {k: np.array([m[k] for m in out]).reshape(shape) for k in KEYS}
    R.update(axes)
    if np.nansum(R['n_bad']):
        print(f"WARNING: {int(np.nansum(R['n_bad']))} failed lead solves (NaN in the result)")
    return R


def save(name, R):
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / f"scharf_soc_{name}.npz", **R)


def load(name):
    with np.load(DATA / f"scharf_soc_{name}.npz") as d:
        return {k: d[k] for k in d.files}


def load_or_run(name, W, seed, E_Z, phi, scaled, invariant=False, n_workers=1, pp=None):
    """load(name) if that file holds this grid, else sweep and save (a changed PP is NOT noticed)."""
    if (DATA / f"scharf_soc_{name}.npz").exists():
        R = load(name)
        want = {'W': W, 'seed': seed, 'E_Z': E_Z, 'phi': phi}
        if all(R[k].shape == np.atleast_1d(np.asarray(v, float)).shape
               and np.allclose(R[k], np.atleast_1d(np.asarray(v, float))) for k, v in want.items()):
            return R
        print(f"scharf_soc_{name}.npz holds a different grid: recomputing")
    R = sweep(W, seed, E_Z, phi, scaled, invariant, n_workers, pp)
    save(name, R)
    return R


def peaks(k, k_min=0.3):
    """Indices of the local maxima of a 1D kappa curve that are above k_min."""
    return np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > k_min)) + 1


def variants():
    return (('old', False), ('scaled', True))


NW = 14

# %% --- TEST 1: does tc_barr = 0 decouple?  kappa at clean point A against tc_barr ---
if __name__ == "__main__":
    TCS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    T1 = {v: [load_or_run(f't1_{v}_tc{tc:g}', 0, 0, A[0], A[1], sc,
                          pp=replace(PP, tc_barr_transport=tc), n_workers=1) for tc in TCS]
          for v, sc in variants()}
    print("\nTEST 1  clean A: kappa / G / local Andreev against tc_barr")
    print("tc_barr    old: kappa      G   Andreev   scaled: kappa      G   Andreev")
    for i, tc in enumerate(TCS):
        o, s = T1['old'][i], T1['scaled'][i]
        print(f"{tc:7.2f}   {o['kappa'].item():11.4f} {o['G'].item():8.4f} {o['andreev'].item():8.4f}"
              f"   {s['kappa'].item():12.4f} {s['G'].item():8.4f} {s['andreev'].item():8.4f}")

# %% --- TEST 2: clean E_Z sweep at phi = pi -- where are the resonances, how high? ---
if __name__ == "__main__":
    EZ = np.round(np.arange(0.0, 1.0001, 0.005), 4)
    T2 = {v: load_or_run(f't2_{v}', 0, 0, EZ, np.pi, sc, n_workers=NW) for v, sc in variants()}
    T2['scaled_tc0.26'] = load_or_run('t2_scaled_tc0.26', 0, 0, EZ, np.pi, True, n_workers=NW,
                                      pp=replace(PP, tc_barr_transport=0.26))
    print("\nTEST 2  clean E_Z sweep, phi = pi")
    for v, R in T2.items():
        k, G, a = R['kappa'][0, 0, :, 0], R['G'][0, 0, :, 0], R['andreev_s'][0, 0, :, 0]
        p = peaks(k)
        print(f"  {v:14s} peaks at {np.array2string(EZ[p], precision=3, max_line_width=150)}")
        print(f"  {'':14s} heights  {np.array2string(k[p], precision=3, max_line_width=150)}"
              f"  max |G|/k at peaks {np.max(np.abs(G[p]) / k[p]) if len(p) else np.nan:.1e}")
        print(f"  {'':14s} strong-probe Andreev identical to old: "
              f"{np.allclose(a, T2['old']['andreev_s'][0, 0, :, 0], atol=1e-8)}")
    plt.figure(figsize=(9, 3.5))
    for v, R in T2.items():
        plt.plot(EZ, R['kappa'][0, 0, :, 0], lw=1, label=v)
    plt.xlabel(r'$E_Z$ [meV]')
    plt.ylabel(r'$\kappa/\kappa_0$')
    plt.legend()
    plt.title(r'clean, $\phi=\pi$, $t_\mathrm{barr}=0.2$ unless stated')
    plt.tight_layout()
    plt.show()

# %% --- TEST 3: the (E_Z, phi) map -- does |G|/kappa still separate the false positives? ---
if __name__ == "__main__":
    EZm, PHm = np.linspace(0.0, 1.0, 61), np.linspace(0.0, 2 * np.pi, 61)
    T3 = {v: load_or_run(f't3_{v}', 0, 0, EZm, PHm, sc, n_workers=NW) for v, sc in variants()}
    print("\nTEST 3  61 x 61 (E_Z, phi) map, clean")
    for v, R in T3.items():
        k, G = R['kappa'][0, 0], R['G'][0, 0]
        cand = R['candidate'][0, 0] > 0
        on = k > 0.05
        r = np.abs(G) / np.where(on, k, np.nan)
        contrast = R['ldos_lo'][0, 0] / R['ldos_mid'][0, 0]
        print(f"  {v:7s} kappa>0.05: {on.sum():4d}  neutral (|G|/k<0.1): {(r < 0.1).sum():4d}"
              f"  charged (>0.5): {(r > 0.5).sum():3d}  max kappa {k.max():.4f}"
              f"  LDOS candidates: {cand.sum():4d}  charged AND candidate: {((r > 0.5) & cand).sum():3d}"
              f"  contrast of those [{np.nanmin(np.where((r > 0.5) & cand, contrast, np.nan)):.1f}"
              f" .. {np.nanmax(np.where((r > 0.5) & cand, contrast, np.nan)):.1f}]"
              f"  of neutral candidates [{np.nanmin(np.where((r < 0.1) & cand, contrast, np.nan)):.1f}"
              f" .. {np.nanmax(np.where((r < 0.1) & cand, contrast, np.nan)):.1f}]")
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for j, (v, R) in enumerate(T3.items()):
        for i, key in enumerate(('kappa', 'G')):
            im = axs[i, j].pcolormesh(EZm, PHm / np.pi, R[key][0, 0].T, shading='nearest',
                                      vmin=0 if key == 'kappa' else None)
            fig.colorbar(im, ax=axs[i, j])
            axs[i, j].set_title(f'{v}: {key}', fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel(r'$E_Z$ [meV]')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$\phi/\pi$')
    plt.tight_layout()
    plt.show()

# %% --- TEST 4: weak disorder (W <= Delta) with scaled bonds, against the saved old results ---
if __name__ == "__main__":
    import scharf_disorder as SD
    W_WEAK = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
    old = SD.load('weak_strength')
    new = load_or_run('t4_weak_strength', W_WEAK, range(64), A[0], A[1], True, n_workers=NW)
    print("\nTEST 4  point A, 64 seeds: mean / worst kappa and max |G|/kappa")
    print("W [meV]     old: mean   worst   |G|/k     scaled: mean   worst   |G|/k")
    for i, W in enumerate(W_WEAK):
        ko, kn = old['kappa'][i, :, 0, 0], new['kappa'][i, :, 0, 0]
        go, gn = old['G'][i, :, 0, 0], new['G'][i, :, 0, 0]
        print(f"{W:7.4f}   {ko.mean():10.4f} {ko.min():7.4f} {np.max(np.abs(go) / ko):7.1e}"
              f"   {kn.mean():13.4f} {kn.min():7.4f} {np.max(np.abs(gn) / kn):7.1e}")
    for name, (E_Z, phi) in (('B', B), ('C', C)):
        R = load_or_run(f't4_weak_trivial_{name}', [0.1, 0.25], range(128), E_Z, phi, True,
                        n_workers=NW)
        Ro = SD.load(f'weak_trivial_{name}')
        for i, W in enumerate(R['W']):
            print(f"  {name} W={W}: candidates old {int(Ro['candidate'][i].sum())} scaled"
                  f" {int(R['candidate'][i].sum())};  max kappa old {Ro['kappa'][i].max():.1e}"
                  f" scaled {R['kappa'][i].max():.1e}")
