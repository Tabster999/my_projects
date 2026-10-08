"""
Fermi occupation functions and DC current / conductance post-processing.

Current formula (Nambu-symmetric, BdG; units e/h for currents, e^2/h for conductances)
-----------------------------------------------------------------------------------
For lead i ("out") and the other normal lead j ("in"), summing over injected
electrons AND holes:

  I_i = (1/2) Int dE [  T_ee (f_ie - f_je) - T_hh (f_ih - f_jh)                  EC
                      + T_eh (f_ie - f_jh) - T_he (f_ih - f_je)                  CAR
                      + (R_eh + R_he) (f_ie - f_ih)                   ]          LAR

The prefactor 1/2 removes the Nambu double counting: by particle-hole symmetry
T_hh(E) = T_ee(-E), R_he(E) = R_eh(-E), T_he(E) = T_eh(-E), so the hole terms
equal the electron terms on a symmetric energy grid.  (Check: a clean normal
ribbon with N open spinful channels gives G_ii = +N, G_ij = -N.)

Known limitation: quasiparticle transfer from lead i into the grounded SC
ribbons is not included.  The NON-LOCAL conductance dI_i/dV_j is exact (those
terms don't depend on V_j); the LOCAL conductance dI_i/dV_i misses them above
the gap, and below it whenever the ribbons carry propagating edge modes.

Linear response in temperature: kappa/kappa0 = Int dx (3/pi^2) x^2/(4cosh^2(x/2)) T_th(x kT),
G/G0 = Int dx 1/(4cosh^2(x/2)) T_el(x kT), x = E/kT, both kernels integrate to 1 (kT -> 0 gives
T_th(0), T_el(0)).  Evaluated with the trapezoid rule on a uniform grid (step h, |x| <= x_max),
negative energies from particle-hole symmetry.  h = 0.5 is accurate to <~2e-3 where T(E) is smooth
on the kT scale; sharp resonances (e.g. at phi = 0 above the minigap, or open normal channels) can
need h = 0.25 -- check with thermal_error_phs, which is free.  (The previous Gauss-Legendre rule was
off by up to 0.14 at phi = 0 for exactly that reason.)
"""

import warnings
import numpy as np

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz     # NumPy 1.x name


# ============================ Fermi functions ============================
def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -1000, 1000)))


def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E + mu) / kT, -1000, 1000)))


# ============================ finite bias: one core ============================
def current(ch, E, V_out, V_in, kT, parts=False):
    """
    DC current into lead 'out' [e/h] from its channels `ch` on the energy grid E, for bias V_out on
    that lead and V_in on the other normal lead (scalars or equal-shape arrays; result has that shape).
    parts=True returns {'EC', 'CAR', 'LAR', 'total'}.
    """
    Vo, Vi = np.broadcast_arrays(np.asarray(V_out, float), np.asarray(V_in, float))
    Vo, Vi = Vo[..., None], Vi[..., None]
    fo_e, fo_h, fi_e, fi_h = f_electron(E, Vo, kT), f_hole(E, Vo, kT), f_electron(E, Vi, kT), f_hole(E, Vi, kT)
    EC = 0.5 * (ch['ee'] * (fo_e - fi_e) - ch['hh'] * (fo_h - fi_h))
    CAR = 0.5 * (ch['eh_cross'] * (fo_e - fi_h) - ch['he_cross'] * (fo_h - fi_e))
    LAR = 0.5 * (ch['eh_local'] + ch['he_local']) * (fo_e - fo_h)
    I = {k: _trapezoid(v, E, axis=-1) for k, v in (('EC', EC), ('CAR', CAR), ('LAR', LAR))}
    I['total'] = I['EC'] + I['CAR'] + I['LAR']
    return I if parts else I['total']


def differential_conductance(ch, E, V_out, V_in, kT, dV=1e-5):
    """(dI_out/dV_out, dI_out/dV_in) [e^2/h] at the bias point(s), central differences."""
    G_loc = (current(ch, E, np.add(V_out, dV), V_in, kT) - current(ch, E, np.subtract(V_out, dV), V_in, kT)) / (2 * dV)
    G_nl = (current(ch, E, V_out, np.add(V_in, dV), kT) - current(ch, E, V_out, np.subtract(V_in, dV), kT)) / (2 * dV)
    return G_loc, G_nl


# ---- the previous finite-bias functions, kept as thin wrappers ----
def other_name(name):
    return 'right' if name == 'left' else 'left'


def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    """Current into `out_name` for a bias dict {'left': V_L, 'right': V_R}, split into EC/CAR/LAR [e/h]."""
    return current(ch, E_sweep, bias[out_name], bias[in_name], kT, parts=True)


def eval_I_total(ch, E, VL, VR, kT):
    """Total current into the lead of `ch` for bias arrays VL (that lead) and VR (other lead) [e/h]."""
    return current(ch, E, np.squeeze(VL), np.squeeze(VR), kT)


def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    """G_ij = dI_i/dV_j [e^2/h] around the bias dict bias0; channel_sweeps[i] = channels of lead i."""
    G = {}
    for i in leads:
        loc, nl = differential_conductance(channel_sweeps[i], E_sweep, bias0[i], bias0[other_name(i)], kT, dV)
        for j in leads:
            G[(i, j)] = loc if j == i else nl
    return G


def partial_G_vectorized(ch, E, V, dV, kT, scheme="sym"):
    """dI_L/dV_L and dI_L/dV_R [e^2/h] along V_R = +V ('sym') or V_R = -V ('anti')."""
    if scheme not in ("sym", "anti"):
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")
    V = np.asarray(V, float)
    return differential_conductance(ch, E, V, V if scheme == "sym" else -V, kT, dV)


def total_dIdV_map(ch, E, V, dV, kT, scheme):
    """Total derivative dI_L/dV along V_R = +V ('sym') or V_R = -V ('anti') [e^2/h]."""
    if scheme not in ("sym", "anti"):
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")
    s, V = (1 if scheme == "sym" else -1), np.asarray(V, float)
    return (current(ch, E, V + dV, s * (V + dV), kT) - current(ch, E, V - dV, s * (V - dV), kT)) / (2 * dV)


# ============================ linear response in temperature ============================
def _kernels(x):
    w = 1.0 / (4.0 * np.cosh(x / 2) ** 2)
    return (3 / np.pi**2) * x**2 * w, w


def _ignored_n(n):
    if n is not None:
        warnings.warn("`n` is ignored: the thermal integral now uses the trapezoid rule (set its step with h).",
                      stacklevel=3)


def linear_response_nodes(kT, h=0.5, x_max=16.0, n=None):
    """
    Symmetric energy grid and trapezoid weights: kappa/kappa0 = sum(w_th * (T_ee + T_he)),
    G/G0 = sum(w_el * (T_ee - T_he)) with channels computed at E (both signs).
    Prefer linear_response_nodes_phs (half the energies).  kT = 0: E = [0], weights 1.
    """
    _ignored_n(n)
    if kT == 0:
        return np.zeros(1), np.ones(1), np.ones(1)
    x = np.arange(-x_max, x_max + h / 2, h)
    W_th, W_el = _kernels(x)
    w = np.full(len(x), h); w[0] = w[-1] = h / 2
    return x * kT, W_th * w, W_el * w


def linear_response_nodes_phs(kT, h=0.5, x_max=16.0, n=None):
    """
    Energies E >= 0 and weights for thermal_from_channels_phs: the trapezoid rule of
    linear_response_nodes folded onto E >= 0 with particle-hole symmetry
    (T_ee(-E) = T_hh(E), T_he(-E) = T_eh(E), exact).  33 energies with the defaults.
    kT = 0 returns E = [0] with weights 1/2 (the same sums then give T_th(0), T_el(0)).
    """
    _ignored_n(n)
    if kT == 0:
        return np.zeros(1), np.full(1, 0.5), np.full(1, 0.5)
    x = np.arange(0.0, x_max + h / 2, h)
    W_th, W_el = _kernels(x)
    fold = np.full(len(x), h); fold[0] = fold[-1] = h / 2
    return x * kT, W_th * fold, W_el * fold


def thermal_from_channels_phs(ch, w_th, w_el):
    """(kappa/kappa0, G/G0) from channels at the energies of linear_response_nodes_phs."""
    ee, hh = np.asarray(ch['ee']), np.asarray(ch['hh'])
    he, eh = np.asarray(ch['he_cross']), np.asarray(ch['eh_cross'])
    return float(np.dot(w_th, ee + he + hh + eh)), float(np.dot(w_el, ee - he + hh - eh))


def phs_residual(ch, E, tol=None):
    """
    Particle-hole identity at E = 0, where PHS forces T_ee = T_hh and T_eh = T_he EXACTLY.
    Returns the largest violation over the E = 0 entries of the grid (0.0 if it has none), and
    with `tol` also warns when that exceeds it.  Free: it reuses channels you already have.

    Scope, measured: this is a REGRESSION check, not a breakdown detector.  It stays at ~1e-12
    even where the channels are badly wrong -- at the Gresta point with eta = 1e-7 it reads
    5e-13 while kappa(pi) = 0.64 instead of 0.50.  Sancho-Rubio can converge to the wrong
    (growing) branch, which satisfies both this identity and the Dyson residual, so neither test
    sees it.  Do NOT use it to validate a small-eta run; keep eta >~ 1e-6 instead, and note that
    T_ee = T_he (ee vs he_cross) is the MAJORANA condition, not a symmetry, so it cannot be
    checked either.
    """
    E = np.atleast_1d(np.asarray(E, float))
    at0 = np.flatnonzero(E == 0.0)
    if at0.size == 0:
        return 0.0
    g = lambda k: np.atleast_1d(np.asarray(ch[k], float))[at0]
    r = float(max(np.abs(g('ee') - g('hh')).max(), np.abs(g('eh_cross') - g('he_cross')).max()))
    if tol is not None and r > tol:
        warnings.warn(f"particle-hole symmetry violated at E = 0 by {r:.2e} (> {tol:.1e}); the "
                      "channels are unreliable even though the lead residual check passed",
                      stacklevel=2)
    return r


def thermal_error_phs(ch, kT, h=0.5, x_max=16.0):
    """
    Conservative error estimates (err_kappa, err_G) of thermal_from_channels_phs, from the same
    channels on the step-2h sub-grid (free: no extra energies).  If too large, halve h.
    """
    if kT == 0:
        return 0.0, 0.0
    _, wt, we = linear_response_nodes_phs(kT, h, x_max)
    _, wt2, we2 = linear_response_nodes_phs(kT, 2 * h, x_max)
    sub = {k: np.asarray(v)[::2] for k, v in ch.items()}
    k1, g1 = thermal_from_channels_phs(ch, wt, we)
    k2, g2 = thermal_from_channels_phs(sub, wt2, we2)
    return abs(k1 - k2), abs(g1 - g2)
