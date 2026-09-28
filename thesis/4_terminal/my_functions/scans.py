"""
Fast (kappa, G) parameter scans: particle-hole-halved energies, reuse of the central
RGF solve across ribbon-only parameter changes, and parallel worker processes.

    E, w_th, w_el = myf.linear_response_nodes_phs(kT, n=16)
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
from .transport import thermal_from_channels_phs

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
