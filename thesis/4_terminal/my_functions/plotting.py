"""Shared plotting helpers used across the run scripts."""

import numpy as np


def param_title(p, extra=""):
    """
    Parameter string for figure titles: everything needed to reproduce a plot.
    Usage:  fig.suptitle("Thermal conductance\n" + myf.param_title(p_base), fontsize=9)
    """
    ts = (f"t={p.t_c:g}" if p.t_n == p.t_c == p.t_s else
          f"$t_n$={p.t_n:g}, $t_c$={p.t_c:g}, $t_s$={p.t_s:g}")
    soc = (f"$\\alpha$={p.alpha:g}, $\\beta$={p.beta:g}" if p.model == 'rashba' else
           f"$m_0$={p.m0:g}, $m_0^c$={p.m0_c:g}, $m_0^n$={p.m0_n:g}")
    return (f"{p.model}: $n_x$={p.nx}, $n_y$={p.ny}, {ts}, {soc}, $\\Delta$={p.delta:g}, "
            f"$\\phi$={p.phi / np.pi:.2f}$\\pi$\n"
            f"$\\mu_n$={p.mu_n:g}, $\\mu_c$={p.mu_c:g}, $\\mu_s$={p.mu_s:g}, "
            f"$t_c^{{top}}$={p.tc_top:g}, $t_c^{{bot}}$={p.tc_bot:g}, $t_c^{{barr}}$={p.tc_barr:g}, "
            f"$B_z$={p.Bz:g}, $B_{{xy}}$={p.Bxy:g}, $B_{{z,s}}$={p.Bz_s:g}, $\\eta$={p.eta:g}"
            + (f"\n{extra}" if extra else ""))




def style_axis(ax, title):
    """Apply the shared axis style: no top/right spine, inward ticks."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.set_title(title)


def print_params(p):
    """Render a Params instance as a LaTeX parameter table (Jupyter/IPython display)."""
    from IPython.display import display, Math   # optional dependency, only needed here
    latex_str = (
        r"\begin{array}{ll | ll | ll}\hline"
        r"\text{Chemical Potentials} & \text{Value} & \text{Hoppings} & \text{Value} & \text{Other} & \text{Value} \\\hline"
        rf"\mu_n & {p.mu_n} & t_n & {p.t_n} & n_x & {p.nx} \\"
        rf"\mu_c & {p.mu_c} & t_c & {p.t_c} & \Delta & {p.delta} \\"
        rf"\mu_s & {p.mu_s} & t_s & {p.t_s} & \phi & {p.phi:.3g} \\"
        rf"& & t_{{\text{{top}}}} & {p.tc_top} & \eta & {p.eta} \\"
        rf"& & t_{{\text{{bot}}}} & {p.tc_bot} & kT & {p.kT} \\"
        rf"& & t_{{\text{{barr}}}} & {p.tc_barr} & & \\\hline\end{{array}}"
    )
    display(Math(latex_str))
