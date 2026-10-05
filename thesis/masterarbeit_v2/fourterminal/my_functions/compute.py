"""
One-call functions that return arrays (no plotting).  Each builds the junction and the RGF solver for
the given Params, evaluates, and returns numpy arrays / dicts; plot them however you like.

    T = myf.transmissions(p, E, phi)               # dict: 'ee', 'he_cross', 'hh', 'eh_cross', 'eh_local', 'he_local'
    kappa, G = myf.linear_response(p, phi, kT)     # thermal / electrical linear response (kT = 0: T -> 0)
    A = myf.ldos(p, E, phi, cols)                  # local density of states, shape (n_E, n_cols, ny)
    I = myf.currents(p, E, V_out, V_in, kT, phi)   # finite-bias current [e/h]
    G_loc, G_nl = myf.conductances(p, E, V_out, V_in, kT, phi)   # differential conductances [e^2/h]

`phi` may be a number or an array (all phases share one solver, so extra phases are cheap); with an
array the results gain a leading phase axis.  For many parameter points use scans.scan.
"""
import numpy as np
from .junction import FourTerminalJunction
from .rgf import RGFFourTerminal
from .spectra import LocalGreen
from .transport import (current, differential_conductance, linear_response_nodes_phs,
                        thermal_from_channels_phs, thermal_error_phs)


def _phases(phi):
    return np.atleast_1d(np.asarray(phi, float)), np.ndim(phi) == 0


def transmissions(p, E, phi=np.pi, side='right'):
    """Channel transmissions of lead `side` at energies E: dict of arrays (n_E) or (n_phi, n_E)."""
    phis, scalar = _phases(phi)
    r = RGFFourTerminal(FourTerminalJunction(p), np.atleast_1d(np.asarray(E, float)))
    rows = [r.channels_at_phi(ph, side) for ph in phis]
    out = {k: np.array([np.asarray(c[k]) for c in rows]) for k in rows[0]}
    return {k: v[0] for k, v in out.items()} if scalar else out


def linear_response(p, phi=np.pi, kT=0.0, side='right', h=0.5, x_max=16.0, error=False):
    """
    (kappa/kappa0, G/G0) at temperature kT (kT = 0: T -> 0) for one phase or an array of phases.
    error=True returns (kappa, G, err_kappa, err_G) with the free error estimate of thermal_error_phs.
    """
    phis, scalar = _phases(phi)
    E, w_th, w_el = linear_response_nodes_phs(kT, h, x_max)
    r = RGFFourTerminal(FourTerminalJunction(p), E)
    res = []
    for ph in phis:
        ch = r.channels_at_phi(ph, side)
        k, g = thermal_from_channels_phs(ch, w_th, w_el)
        res.append((k, g, *thermal_error_phs(ch, kT, h, x_max)) if error else (k, g))
    res = np.array(res)
    return tuple(res[0]) if scalar else tuple(res.T)


def ldos(p, E, phi=np.pi, cols=None):
    """LDOS (electron + hole) at energies E on columns `cols` (default: all), shape (n_E, n_cols, ny)."""
    cols = list(range(p.nx)) if cols is None else list(cols)
    e, h = LocalGreen(FourTerminalJunction(p), np.atleast_1d(np.asarray(E, float)), cols=cols).ldos(phi)
    return e + h


def currents(p, E, V_out, V_in, kT, phi=np.pi, side='left', parts=False):
    """Finite-bias current into lead `side` [e/h]; the grid E must cover the bias window plus a few kT."""
    return current(transmissions(p, E, phi, side), np.asarray(E, float), V_out, V_in, kT, parts=parts)


def conductances(p, E, V_out, V_in, kT, phi=np.pi, side='left', dV=1e-5):
    """(dI/dV_out, dI/dV_in) of lead `side` [e^2/h] at the bias point(s)."""
    return differential_conductance(transmissions(p, E, phi, side), np.asarray(E, float), V_out, V_in, kT, dV)
