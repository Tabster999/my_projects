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
