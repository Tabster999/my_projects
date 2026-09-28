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
"""

import numpy as np

# np.trapezoid is NumPy >= 2.0; np.trapz is its NumPy 1.x name
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def f_electron(E, mu, kT):
    if kT == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -1000, 1000)))


def f_hole(E, mu, kT):
    if kT == 0:
        return np.where(E < -mu, 1.0, np.where(E == -mu, 0.5, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip((E + mu) / kT, -1000, 1000)))


def _current_terms(ch, f_oe, f_oh, f_ie, f_ih):
    """EC, CAR and LAR integrands for lead 'out' (channels ch of that side), incl. the 1/2."""
    EC = 0.5 * (ch['ee'] * (f_oe - f_ie) - ch['hh'] * (f_oh - f_ih))
    CAR = 0.5 * (ch['eh_cross'] * (f_oe - f_ih) - ch['he_cross'] * (f_oh - f_ie))
    LAR = 0.5 * (ch['eh_local'] + ch['he_local']) * (f_oe - f_oh)
    return EC, CAR, LAR


def dc_current_channels(ch, E_sweep, bias, kT, out_name, in_name):
    """Integrate the EC/CAR/LAR channel contributions into a DC current for one lead [e/h]."""
    mu_out, mu_in = bias[out_name], bias[in_name]
    EC, CAR, LAR = _current_terms(ch,
                                  f_electron(E_sweep, mu_out, kT), f_hole(E_sweep, mu_out, kT),
                                  f_electron(E_sweep, mu_in, kT), f_hole(E_sweep, mu_in, kT))
    I_EC, I_CAR, I_LAR = (_trapezoid(x, E_sweep) for x in (EC, CAR, LAR))
    return {'EC': I_EC, 'CAR': I_CAR, 'LAR': I_LAR, 'total': I_EC + I_CAR + I_LAR}


def other_name(name):
    return 'right' if name == 'left' else 'left'


def conductance_matrix(bias0, leads, channel_sweeps, E_sweep, kT, dV=1e-5):
    """Differential conductance matrix G_ij = dI_i/dV_j [e^2/h] via central differences."""
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
    """Vectorized total left-lead current I_L(V_L, V_R) [e/h] for arrays of bias points."""
    EC, CAR, LAR = _current_terms({k: v[None, :] for k, v in ch.items()},
                                  f_electron(E[None, :], VL, kT), f_hole(E[None, :], VL, kT),
                                  f_electron(E[None, :], VR, kT), f_hole(E[None, :], VR, kT))
    return _trapezoid(EC + CAR + LAR, E, axis=1)


def partial_G_vectorized(ch, E, V, dV, kT, scheme="sym"):
    """
    Partial conductances G_LL = dI_L/dV_L and G_LR = dI_L/dV_R [e^2/h], evaluated
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


# =============================================================================
# Thermal conductance: kappa/kappa0 = (3/pi^2) Int de (e/kT)^2 (-df/de) T_th(e)
# with T_th = T_ee + T_he.  The weight integrates to 1, so as T -> 0 it collapses
# onto e = 0 and kappa/kappa0 -> T_th(0)  (the T -> 0 limit used in the maps).
# At finite T the weight peaks at |e| ~ 2.4 kT, so E = 0 itself is NOT sampled.
# =============================================================================
def linear_response_nodes(kT, n=16, x_max=10.0):
    """
    Energies and weights of the linear-response integrals at temperature kT:
 
        kappa/kappa0 = sum_i w_th[i] * T_th(E_i),     T_th = T_ee + T_he
        G/G0         = sum_i w_el[i] * T_el(E_i),     T_el = T_ee - T_he
 
    (Gauss-Legendre on x = E/kT; thermal weight (3/pi^2) x^2/(4cosh^2(x/2)), electrical
    weight 1/(4cosh^2(x/2)); both integrate to 1.)  The thermal weight peaks at
    |E| ~ 2.4 kT, so E = 0 itself is not sampled at finite temperature.
    kT = 0 returns the T -> 0 rule: one node at E = 0 with both weights 1, i.e.
    kappa/kappa0 = T_th(0) and G/G0 = T_el(0).
 
    Convergence (checked against a dense-grid integration of a junction with structure
    in T_th on the scale 0.007):
      n      : n = 6 / 8 / 12 are off by 0.2 / 0.09 / 0.01; from n = 16 the result is
               converged to <~ 2e-3.  n = 16 is a good default, n = 24 is safe.
      x_max  : the kernel holds 71% / 92% / 98.3% / 99.7% of its weight inside
               |x| < 4 / 6 / 8 / 10, so small x_max biases kappa low.  x_max = 10-12 is
               optimal; going beyond ~15 at FIXED n makes it worse again, because the
               nodes then resolve the peak at |x| ~ 2.4 less well.
    Two things the quadrature cannot fix:
      * features of T_th narrower than the node spacing ~ 2 x_max kT / n are missed
        (e.g. the SC-ribbon edge-overlap gap: keep kT >> that gap);
      * the transmission must be available up to |E| ~ x_max * kT, which for large kT
        reaches well beyond the superconducting gap.
    """
    if kT == 0:
        return np.zeros(1), np.ones(1), np.ones(1)
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = x * x_max, w * x_max
    kernel = w / (4 * np.cosh(x / 2)**2)
    return x * kT, (3 / np.pi**2) * kernel * x**2, kernel
 
 
def kappa_nodes(kT, n=16, x_max=10.0):
    """Energies and thermal weights only (see linear_response_nodes)."""
    E, w_th, _ = linear_response_nodes(kT, n, x_max)
    return E, w_th
 
 
def kappa_from_channels(ch, weights):
    """kappa/kappa0 from channels evaluated at the energies of linear_response_nodes."""
    return float(np.dot(weights, np.asarray(ch['ee']) + np.asarray(ch['he_cross'])))

