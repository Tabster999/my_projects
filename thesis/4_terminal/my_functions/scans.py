"""
Parameter scans.

scan(func, grid, n_workers, save) is the general tool: it evaluates func(**point) on a parameter grid,
in parallel worker processes (one BLAS thread each), resumable via `save`, and returns arrays.
ThermalPoint is the ready-made point function for (kappa, G): it reuses each worker's central RGF solve
when only ribbon parameters change, so put ribbon parameters (mu_s, Bz_s, delta, t_s, m0) LAST in the grid.

    f = myf.ThermalPoint(p0, kT=1e-3, phis=(np.pi, 0.0))
    res = myf.scan(f, {'mu_s': [0.15, 0.4], 'Bz_s': np.linspace(0.4, 1.6, 25)}, n_workers=8, save='scan.npz')
    res['values'][..., 0]   # kappa(pi)   (columns: kappa, G per phase, then the number of failed lead energies)

The lower-level thermal_point / thermal_scan below are what ThermalPoint and scan are built on:

    E, w_th, w_el = myf.linear_response_nodes_phs(kT)          # trapezoid in E/kT, 33 energies
    rows = myf.thermal_scan(param_list, E, w_th, w_el, phis=(np.pi, 0.0), n_workers=8)

Each row is [kappa(phi_0), G(phi_0), kappa(phi_1), G(phi_1), ...].  Consecutive
parameter sets that differ only in ribbon parameters (mu_s, Bz_s, delta, t_s, ...)
share one central solve per worker, so order scans with ribbon parameters innermost.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import leads
from .junction import FourTerminalJunction
from .rgf import RGFFourTerminal, central_fingerprint
from dataclasses import replace
from .transport import thermal_from_channels_phs, linear_response_nodes_phs

_CACHE = {}          # per process: {'fp': fingerprint, 'rgf': RGFFourTerminal}


def thermal_point(p, E, w_th, w_el, phis=(np.pi,), side='right'):
    """
    (kappa/kappa0, G/G0) at each phase in `phis` for one parameter set, plus the number
    of lead energies that failed the Sancho-Rubio check (leads.N_UNRELIABLE increment).
    Reuses the last central RGF solve in this process when only ribbon parameters changed.
    """
    n_bad0 = leads.N_UNRELIABLE
    J = FourTerminalJunction(p)
    fp = central_fingerprint(J, E)
    if _CACHE.get('fp') == fp:
        rgf = _CACHE['rgf'].with_ribbons(J)
    else:
        rgf = RGFFourTerminal(J, E)
        _CACHE.update(fp=fp, rgf=rgf)
    out = []
    for ph in phis:
        out += thermal_from_channels_phs(rgf.channels_at_phi(ph, side), w_th, w_el)
    return np.array(out), leads.N_UNRELIABLE - n_bad0


def _init_worker():
    try:                                    # one BLAS thread per worker: points run in parallel instead
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _worker(args):
    return thermal_point(*args)


def thermal_scan(param_list, E, w_th, w_el, phis=(np.pi,), side='right', n_workers=None, pool=None):
    """
    thermal_point for every Params in param_list, in order.  Returns (rows, n_unreliable),
    rows of shape (len(param_list), 2 * len(phis)).

    n_workers : processes (None -> os.cpu_count(); 1 -> serial in this process).  Each worker
                is pinned to one BLAS thread.  Points are handed out in contiguous chunks
                so a worker sees runs of neighbouring parameters and can reuse its central solve.
    pool      : an existing ProcessPoolExecutor to reuse across calls (avoids restart cost).
    """
    param_list = list(param_list)
    n_workers = n_workers or os.cpu_count() or 1
    if n_workers == 1 and pool is None:
        res = [thermal_point(p, E, w_th, w_el, phis, side) for p in param_list]
    else:
        own = pool is None
        if own:
            pool = make_pool(n_workers)
        n = getattr(pool, '_max_workers', n_workers)
        chunk = max(1, -(-len(param_list) // n))
        try:
            res = list(pool.map(_worker, [(p, E, w_th, w_el, phis, side) for p in param_list],
                                chunksize=chunk))
        finally:
            if own:
                pool.shutdown()
    rows = np.array([r[0] for r in res]).reshape(len(param_list), 2 * len(phis))
    return rows, int(sum(r[1] for r in res))


def make_pool(n_workers=None):
    """Spawn-based process pool with one BLAS thread per worker (works on Windows and in notebooks)."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")     # read by freshly spawned workers before numpy loads
    return ProcessPoolExecutor(max_workers=n_workers or os.cpu_count(),
                               mp_context=mp.get_context('spawn'), initializer=_init_worker)


# =============================================================================
#  general scans
# =============================================================================
class ThermalPoint:
    """
    Picklable point function for scan(): for parameters **kw it evaluates replace(p0, **kw) and returns
    [kappa(phi_0), G(phi_0), kappa(phi_1), G(phi_1), ..., n_failed_lead_energies] at temperature kT
    (kT = 0: T -> 0), reusing the worker's central solve across ribbon-only changes.
    """
    def __init__(self, p0, kT=0.0, phis=(np.pi,), side='right', h=0.5, x_max=16.0):
        self.p0, self.phis, self.side = p0, tuple(np.atleast_1d(phis)), side
        self.E, self.w_th, self.w_el = linear_response_nodes_phs(kT, h, x_max)

    def __call__(self, **kw):
        vals, n_bad = thermal_point(replace(self.p0, **kw), self.E, self.w_th, self.w_el, self.phis, self.side)
        return np.append(vals, n_bad)


def _call(args):
    func, kw = args
    return func(**kw)


def scan(func, grid, n_workers=1, save=None, pool=None):
    """
    Evaluate func(**point) on every point of the grid {name: values, ...} (outer product; the LAST
    parameter varies fastest) and return {'values': array (*grid shape, *result shape), 'done', 'axes'}.

    n_workers > 1 (or a pool from make_pool) runs points in parallel; neighbouring points go to the same
    worker, so a ThermalPoint reuses its central solve along the last axis.  func must then be picklable
    (module level, or a ThermalPoint), and scripts need the `if __name__ == "__main__":` guard.
    save='file.npz' stores the results after every point; rerunning with the same grid skips finished points.
    """
    import itertools
    names = list(grid)
    axes = [np.atleast_1d(np.asarray(grid[n])) for n in names]
    shape = tuple(len(a) for a in axes)
    values, done = None, np.zeros(shape, bool)
    if save and os.path.exists(save):
        d = np.load(save, allow_pickle=False)
        if list(d['names']) != names or any(not np.array_equal(d[f'axis_{n}'], a) for n, a in zip(names, axes)):
            raise ValueError(f"{save} belongs to a different grid; delete it or choose another file")
        values, done = d['values'].copy(), d['done'].copy()

    def store(ix, r):
        nonlocal values
        r = np.asarray(r, float)
        if values is None:
            values = np.full(shape + r.shape, np.nan)
        values[ix], done[ix] = r, True
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
            np.savez(save, values=values, done=done, names=np.array(names), **{f'axis_{n}': a for n, a in zip(names, axes)})

    todo = [ix for ix in itertools.product(*[range(n) for n in shape]) if not done[ix]]
    jobs = [(func, {n: axes[k][ix[k]].item() for k, n in enumerate(names)}) for ix in todo]
    if (n_workers or 1) == 1 and pool is None:
        for ix, job in zip(todo, jobs):
            store(ix, _call(job))
    else:
        own = pool is None
        pool = pool or make_pool(n_workers)
        chunk = max(1, -(-len(jobs) // getattr(pool, '_max_workers', n_workers or 1)))
        try:
            for ix, r in zip(todo, pool.map(_call, jobs, chunksize=chunk)):
                store(ix, r)
        finally:
            if own:
                pool.shutdown()
    return {'values': values, 'done': done, 'axes': dict(zip(names, axes))}
