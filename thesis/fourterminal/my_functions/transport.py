"""Fermi occupation functions and DC current / conductance post-processing."""

import numpy as np


def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -1000, 1000)))


def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E + mu) / kT, -1000, 1000)))


def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    """Integrate the EC/CAR/LAR channel contributions into a DC current for one lead."""
    mu_out, mu_in = bias[out_name], bias[in_name]
    f_out_e, f_out_h = f_electron(E_sweep, mu_out, kT), f_hole(E_sweep, mu_out, kT)
    f_in_e, f_in_h = f_electron(E_sweep, mu_in, kT), f_hole(E_sweep, mu_in, kT)

    I_EC = np.trapezoid(ch['ee'] * (f_out_e - f_in_e) - ch['hh'] * (f_out_h - f_in_h), E_sweep)
    I_CAR = np.trapezoid(ch['eh_cross'] * (f_out_e - f_in_h) - ch['he_cross'] * (f_out_h - f_in_e), E_sweep)
    I_LAR = np.trapezoid(ch['eh_local'] * (f_out_e - f_out_h), E_sweep)
    return {'EC': I_EC, 'CAR': I_CAR, 'LAR': I_LAR, 'total': I_EC + I_CAR + I_LAR}


def other_name(name):
    return 'right' if name == 'left' else 'left'


def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    """Full 2x2 (or NxN) differential conductance matrix G_ij = dI_i/dV_j via central differences."""
    G = {}
    for i in leads:
        ch_i, in_name = channel_sweeps[i], other_name(i)
        for j in leads:
            bp, bm = bias0.copy(), bias0.copy()
            bp[j] += dV
            bm[j] -= dV
            rp = dc_current_channels(ch_i, E_sweep, bp, kT, i, in_name)
            rm = dc_current_channels(ch_i, E_sweep, bm, kT, i, in_name)
            G[(i, j)] = (rp['total'] - rm['total']) / (2 * dV)
    return G


def eval_I_total(ch, E, VL, VR, kT):
    """Vectorized total left-lead current I_L(V_L, V_R) for arrays of bias points."""
    fLe, fLh = f_electron(E[None, :], VL, kT), f_hole(E[None, :], VL, kT)
    fRe, fRh = f_electron(E[None, :], VR, kT), f_hole(E[None, :], VR, kT)
    I_curr = (
        ch["ee"][None, :] * (fLe - fRe)
        - ch["hh"][None, :] * (fLh - fRh)
        + ch["eh_cross"][None, :] * (fLe - fRh)
        - ch["he_cross"][None, :] * (fLh - fRe)
        + (ch["eh_local"] + ch["he_local"])[None, :] * (fLe - fLh)
    )
    return np.trapezoid(I_curr, E, axis=1)


def partial_G_vectorized(ch, E, V, dV, kT, scheme="sym"):
    """
    Partial conductances G_LL = dI_L/dV_L and G_LR = dI_L/dV_R, evaluated
    around a bias baseline set by `scheme` ('sym': V_R = +V, 'anti': V_R = -V).
    """
    V = V[:, None]
    if scheme == "sym":
        VR_base = V
    elif scheme == "anti":
        VR_base = -V
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")

    def I_eval(VL, VR):
        return eval_I_total(ch, E, VL, VR, kT)

    G_LL = (I_eval(V + dV, VR_base) - I_eval(V - dV, VR_base)) / (2 * dV)
    G_LR = (I_eval(V, VR_base + dV) - I_eval(V, VR_base - dV)) / (2 * dV)
    return G_LL, G_LR


def total_dIdV_map(ch, E, V, dV, kT, scheme):
    """
    Direct derivative of I_L along the actual bias line — 'sym': V_R=+V,
    'anti': V_R=-V — evaluated at the correct point for each scheme.
    """
    V = V[:, None]
    if scheme == "sym":
        sign = 1
    elif scheme == "anti":
        sign = -1
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be 'sym' or 'anti'")
    Ip = eval_I_total(ch, E, V + dV, sign * (V + dV), kT)
    Im = eval_I_total(ch, E, V - dV, sign * (V - dV), kT)
    return (Ip - Im) / (2 * dV)


# --- THERMAL TRANSPORT (all voltages zero, SC leads grounded at mu=0) ---
#
# Every quasiparticle at energy E (electron or hole) carries heat E, so unlike
# the charge current there is no sgn(e/h) weight, and local Andreev reflection
# carries no heat. Normalization matches dc_current_channels (sum over both
# BdG sectors, no 1/2): a single perfect channel gives G/G0 = kappa/kappa0 = 2,
# so the Lorenz ratio (kappa/kappa0)/(G/G0) is 1 for energy-independent transmission.


def minus_dfdE(E, kT):
    """-df/dE of the equilibrium Fermi function at mu=0 (overflow-safe)."""
    x = np.clip(E / (2 * kT), -350, 350)
    return 1.0 / (4 * kT * np.cosh(x) ** 2)


def nonlocal_T(ch):
    """Total lead-to-lead transmission (EC + CAR, both BdG sectors)."""
    return ch['ee'] + ch['hh'] + ch['eh_cross'] + ch['he_cross']


def linear_response_coeffs(ch, E, kT):
    """
    Zero-bias linear-response charge and thermal conductances for the lead
    `ch` belongs to (L, with R the other normal lead), from one channel dict.

    Returns a dict (G in e^2/h, kappa in kappa0 = pi^2 k_B^2 T / 3h):
        G_LL, G_LR           : dI_L/dV_L, dI_L/dV_R
        kappa_LL, kappa_LR   : dJ_L/dT_L, dJ_L/dT_R
        kappa_LS             : kappa_LL + kappa_LR, heat into the SC ribbons
                               (quasiparticles above the gap; ~0 for kT << Delta)
        lorenz_LL, lorenz_LR : (kappa/kappa0)/(G/G0), = 1 under Wiedemann-Franz

    G_LL uses the out_e/out_h escape channels, so unlike partial_G_vectorized
    it also counts quasiparticle current into the SC ribbons above the gap.
    E must resolve kT (dE << kT) and extend to ~ +-15 kT.
    """
    w = minus_dfdE(E, kT)
    w2 = (E / kT) ** 2 * w * 3 / np.pi**2
    out = ch['out_e'] + ch['out_h']
    lar = ch['eh_local'] + ch['he_local']
    cross = ch['eh_cross'] + ch['he_cross'] - ch['ee'] - ch['hh']

    G_LL = np.trapezoid(w * (out + 2 * lar), E, axis=-1)
    G_LR = np.trapezoid(w * cross, E, axis=-1)
    kappa_LL = np.trapezoid(w2 * out, E, axis=-1)
    kappa_LR = -np.trapezoid(w2 * nonlocal_T(ch), E, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lorenz_LL = kappa_LL / G_LL
        lorenz_LR = kappa_LR / G_LR
    return {'G_LL': G_LL, 'G_LR': G_LR,
            'kappa_LL': kappa_LL, 'kappa_LR': kappa_LR, 'kappa_LS': kappa_LL + kappa_LR,
            'lorenz_LL': lorenz_LL, 'lorenz_LR': lorenz_LR}


def heat_current(ch, E, kT_L, kT_R, kT_S):
    """
    Heat current out of lead L for arbitrary (not linearized) lead temperatures,
    all voltages zero, in units of 1/h (energy^2). Split into the part going
    to lead R and the part absorbed by the SC ribbons.
    """
    f = lambda kT: f_electron(E, 0.0, kT)
    T_LR = nonlocal_T(ch)
    T_LS = ch['out_e'] + ch['out_h'] - T_LR
    J_R = np.trapezoid(E * T_LR * (f(kT_L) - f(kT_R)), E)
    J_S = np.trapezoid(E * T_LS * (f(kT_L) - f(kT_S)), E)
    return {'to_R': J_R, 'to_S': J_S, 'total': J_R + J_S}
