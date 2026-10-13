"""
Onsite disorder in the Scharf junction  (findings: "Disorder" in ../CLAUDE.md).

MODEL.  Every site of the NORMAL strip gets a random potential V_i, uniform in [-W/2, W/2] (meV),
entering as V_i tau_z.  Ribbons and probes stay clean.  (W, seed) fixes the realisation: the same
seed always gives the same potential.

HOW TO USE -- run the cells top to bottom in the interactive window.

    m = measure(W=0.1, seed=3, E_Z=0.505, phi=np.pi)        one realisation -> dict of numbers
    m['kappa'], m['G'], ...

    R = sweep(W=[0, 0.1], seed=range(16), E_Z=0.505, phi=np.pi)    a grid -> dict of arrays
    R['kappa'][iW, iseed, iE_Z, iphi]     ALWAYS these four axes, in this order
    R['W'], R['seed'], R['E_Z'], R['phi'] the values along them

    R = load_or_run('my_name', W=..., seed=..., E_Z=..., phi=...)
        same as sweep, but saved to scharf_disorder_my_name.npz; next time it just loads.
        (It notices a changed grid, but NOT a changed PP -- delete the file then.)

    Add n_workers=14 to sweep / load_or_run to use 14 processes (~1-2 s per realisation).

WHAT measure() RETURNS (everything at E = 0):
    kappa, G, andreev        weak probe   (tc_barr = 0.2): the nonlocal Majorana test
    kappa_s, G_s, andreev_s  strong probe (tc_barr = 1.0): the 2e^2/h zero-bias peak
    ldos_lo, ldos_hi, ldos_mid   Scharf's LDOS (his eta, tc_barr = 0) near y = -L/2, y = +L/2,
    curv_lo, curv_hi, curv_mid   and the middle, and its energy curvature (< 0: a peak at E = 0)
    candidate                Scharf's criterion at the low end: curv_lo < 0 and lo/mid > 3
    det                      det r of the left lead, strong probe: < 0 topological, > 0 trivial
                             (only with invariant=True, NaN otherwise; |det| << 1: unreliable)
    n_bad                    failed lead solves -- should be 0
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


# %% --- SETUP ---
PP = replace(S.PP, tc_barr_transport=0.2)          # weak probe for kappa and G
TC_STRONG = 1.0                                    # strong probe for the Andreev peak and det r

A = (0.98, np.pi)      # (E_Z in meV, phi): topological, on a kappa resonance
B = (0.505, 0.0)        # trivial: same field, phi = 0
C = (0.0, np.pi)        # trivial: no field

KEYS = ('kappa', 'G', 'andreev', 'kappa_s', 'G_s', 'andreev_s',
        'ldos_lo', 'ldos_hi', 'ldos_mid', 'curv_lo', 'curv_hi', 'curv_mid',
        'candidate', 'det', 'n_bad')


# %% --- ONE REALISATION ---
def _ldos(pp, phi):
    """Scharf's LDOS at E = 0 and its energy curvature, near both ends and in the middle."""
    parts = {'lo': S.cols_near_end(pp, side='low'), 'hi': S.cols_near_end(pp, side='high'),
             'mid': S.cols_near_middle(pp)}
    cols = sorted(set().union(*parts.values()))
    dE = S._dE(pp)
    lg = myf.LocalGreen(myf.FourTerminalJunction(pp.params()),
                        np.array([-dE, 0.0, dE]) / pp.t, cols=cols)
    e, h = lg.ldos(phi)
    D = (e + h).sum(axis=2)                                     # (3 energies, len(cols))
    out = {}
    for name, c in parts.items():
        d = D[:, [cols.index(k) for k in c]].sum(axis=1)
        out[f'ldos_{name}'] = d[1]
        out[f'curv_{name}'] = (d[0] - 2 * d[1] + d[2]) / dE**2
    return out


def _transport(pp, phi, tc):
    """kappa, G and the local Andreev conductance at E = 0 for probe coupling tc, plus the solver."""
    J = myf.FourTerminalJunction(replace(pp.params_transport(), tc_barr=tc))
    rgf = myf.RGFFourTerminal(J, np.array([0.0]))
    c = {k: float(v[0]) for k, v in rgf.channels_at_phi(phi, 'left').items()}
    return c['ee'] + c['he_cross'], c['ee'] - c['he_cross'], c['eh_local'] + c['he_local'], rgf


def _det_r(rgf, phi):
    """
    det r at E = 0, r = reflection matrix of the left lead (Fulga et al., PRB 83, 155429):
        r = -1 + i Gamma^1/2 G_LL Gamma^1/2   on the propagating channels.
    Real by particle-hole symmetry; its sign is the topological invariant.
    """
    G_LL = rgf.green_blocks(phi)[0][0]
    w, U = np.linalg.eigh(rgf.Gamma_L[0])
    keep = w > 1e-6 * w.max()                                   # propagating channels only
    s, U = np.sqrt(w[keep]), U[:, keep]
    r = -np.eye(len(s)) + 1j * s[:, None] * (U.conj().T @ G_LL @ U) * s[None, :]
    return np.linalg.det(r).real


def measure(W, seed, E_Z, phi, invariant=False, pp=None):
    """Everything for ONE disorder realisation (W in meV, seed) at (E_Z in meV, phi)."""
    pp = replace(PP if pp is None else pp, disorder=float(W), disorder_seed=int(seed))
    pp.B = pp.B_of_E_Z(E_Z)
    phi = float(phi)
    n_bad0 = myf.leads.N_UNRELIABLE

    m = _ldos(pp, phi)
    m['candidate'] = float(m['curv_lo'] < 0 and m['ldos_lo'] / m['ldos_mid'] > 3)
    m['kappa'], m['G'], m['andreev'], _ = _transport(pp, phi, pp.tc_barr_transport)
    m['kappa_s'], m['G_s'], m['andreev_s'], rgf = _transport(pp, phi, TC_STRONG)
    m['det'] = _det_r(rgf, phi) if invariant else np.nan
    m['n_bad'] = myf.leads.N_UNRELIABLE - n_bad0
    return m


# %% --- MANY REALISATIONS ---
def _job(args):
    return measure(*args)


def sweep(W, seed, E_Z, phi, invariant=False, n_workers=1, pp=None):
    """measure() on every combination of W, seed, E_Z, phi (each a number or a list)."""
    axes = {'W': np.atleast_1d(np.asarray(W, float)), 'seed': np.atleast_1d(np.asarray(seed, int)),
            'E_Z': np.atleast_1d(np.asarray(E_Z, float)), 'phi': np.atleast_1d(np.asarray(phi, float))}
    shape = tuple(len(a) for a in axes.values())
    pp = PP if pp is None else pp
    jobs = [(w, s, e, p, invariant, pp) for w, s, e, p in itertools.product(*axes.values())]
    print(f"sweep: {len(jobs)} realisations on {n_workers} worker(s)")

    if n_workers == 1:
        results = map(_job, jobs)
    else:
        # The workers must import _job from the MODULE: a function defined in the interactive
        # window lives in __main__, which they cannot see.
        import scharf_disorder
        pool = myf.make_pool(n_workers)
        results = pool.map(scharf_disorder._job, jobs, chunksize=4)
    out = []
    try:
        for i, m in enumerate(results):
            out.append(m)
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    finally:
        if n_workers != 1:
            pool.shutdown()

    R = {k: np.array([m[k] for m in out]).reshape(shape) for k in KEYS}
    R.update(axes)
    if np.nansum(R['n_bad']):
        print(f"WARNING: {int(np.nansum(R['n_bad']))} failed lead solves (NaN in the result)")
    return R


def save(name, R):
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / f"scharf_disorder_{name}.npz", **R)


def load(name):
    with np.load(DATA / f"scharf_disorder_{name}.npz") as d:
        return {k: d[k] for k in d.files}


def load_or_run(name, W, seed, E_Z, phi, invariant=False, n_workers=1, pp=None):
    """load(name) if that file holds exactly this grid, otherwise sweep(...) and save it."""
    if (DATA / f"scharf_disorder_{name}.npz").exists():
        R = load(name)
        want = {'W': W, 'seed': seed, 'E_Z': E_Z, 'phi': phi}
        same = all(R[k].shape == np.atleast_1d(np.asarray(v, float)).shape
                   and np.allclose(R[k], np.atleast_1d(np.asarray(v, float))) for k, v in want.items())
        if same and not (invariant and np.isnan(R['det']).any()):
            return R
        print(f"scharf_disorder_{name}.npz holds a different grid: recomputing")
    R = sweep(W, seed, E_Z, phi, invariant, n_workers, pp)
    save(name, R)
    return R


def peaks(k, k_min=0.3):
    """Indices of the local maxima of a 1D kappa curve that are above k_min."""
    return np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:]) & (k[1:-1] > k_min)) + 1


# The cells below are guarded with `if __name__ == "__main__":` so that the worker processes,
# which import this file, do not run them.  In the interactive window they run normally.

# %% --- STUDY 1: point A (topological), kappa against weak disorder W <= Delta ---
if __name__ == "__main__":
    W_WEAK = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
    RA = load_or_run('weak_strength', W=W_WEAK, seed=range(64), E_Z=A[0], phi=A[1],
                     invariant=True, n_workers=14)
    k, G, det = RA['kappa'][:, :, 0, 0], RA['G'][:, :, 0, 0], RA['det'][:, :, 0, 0]   # (W, seed)

    print("W [meV]  W/Delta  mean kappa  min kappa  max kappa  max |G|/kappa  topological")
    for i, W in enumerate(RA['W']):
        print(f"{W:7.4f}  {W / PP.Delta:7.2f}  {k[i].mean():10.4f}  {k[i].min():9.4f}"
              f"  {k[i].max():9.4f}  {np.max(np.abs(G[i]) / k[i]):13.1e}  {(det[i] < 0).sum():5d}/{k.shape[1]}")

    x = RA['W'] / PP.Delta
    plt.figure(figsize=(5, 3.5))
    plt.plot(np.repeat(x, k.shape[1]), k.ravel(), '.', color='0.7', label='one seed')
    plt.plot(x, k.mean(axis=1), 'ko-', label='mean')
    plt.xlabel(r'$W/\Delta$')
    plt.ylabel(r'$\kappa/\kappa_0$')
    plt.title(f'point A: $E_Z$ = {A[0]} meV, $\\phi = \\pi$')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% --- STUDY 2: trivial points B, C -- does weak disorder fake a Majorana? ---
if __name__ == "__main__":
    for name, (E_Z, phi) in (('B', B), ('C', C)):
        R = load_or_run(f'weak_trivial_{name}', W=[0.1, 0.25], seed=range(128), E_Z=E_Z, phi=phi,
                        invariant=True, n_workers=14)
        print(f"\npoint {name}: E_Z = {E_Z} meV, phi = {phi / np.pi:.0f} pi")
        for i, W in enumerate(R['W']):
            k, cand, det = R['kappa'][i, :, 0, 0], R['candidate'][i, :, 0, 0], R['det'][i, :, 0, 0]
            print(f"  W = {W} meV: LDOS candidates {int(cand.sum())}/{len(k)},"
                  f"  max kappa {k.max():.1e},  topological {(det < 0).sum()}/{len(k)}")

# %% --- STUDY 3: sweep the field at phi = pi -- do the kappa resonances move? ---
if __name__ == "__main__":
    EZ = np.round(np.arange(0.0, 1.0001, 0.005), 4)        # resonances are ~0.03 meV wide
    R0 = load_or_run('weak_field_clean', W=0.0, seed=0, E_Z=EZ, phi=np.pi, n_workers=14)
    RF = load_or_run('weak_field', W=[0.05, 0.1, 0.25], seed=range(16), E_Z=EZ, phi=np.pi,
                     n_workers=14)
    k0 = R0['kappa'][0, 0, :, 0]
    p0 = peaks(k0)
    print(f"clean peaks at E_Z = {EZ[p0]}")

    fig, axs = plt.subplots(len(RF['W']), 1, figsize=(7, 2.2 * len(RF['W'])), sharex=True)
    for i, W in enumerate(RF['W']):
        k = RF['kappa'][i, :, :, 0]                          # (seed, E_Z)
        # for each clean peak: the highest kappa of each seed within +-0.025 meV, and where it is
        best, moved = [], []
        for j in p0:
            near = np.abs(EZ - EZ[j]) <= 0.025
            best.append(k[:, near].max(axis=1))
            moved.append(np.abs(EZ[near][k[:, near].argmax(axis=1)] - EZ[j]))
        best, moved = np.array(best), np.array(moved)            # (clean peak, seed)
        print(f"W = {W} meV: peak height kept to {np.median(best / k0[p0, None]):.3f} (median),"
              f" {(best / k0[p0, None]).min():.3f} (worst);  peak moved by median"
              f" {np.median(moved):.4f} meV, worst {moved.max():.4f} meV;"
              f"  kappa AT the clean position: median {np.median(k[:, p0]):.3f}, worst {k[:, p0].min():.3f}")
        for ks in k[:4]:
            axs[i].plot(EZ, ks, lw=0.8)
        axs[i].plot(EZ, k0, 'k--', lw=1, label='clean')
        axs[i].set_ylabel(r'$\kappa/\kappa_0$')
        axs[i].set_title(f'W = {W} meV  (first 4 seeds)', fontsize=9)
    axs[0].legend()
    axs[-1].set_xlabel(r'$E_Z$ [meV]')
    plt.tight_layout()
    plt.show()

# %%
