"""Shared plotting helpers used across the run scripts."""

from IPython.display import display, Math


def style_axis(ax, title):
    """Apply the shared axis style: no top/right spine, inward ticks."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which='both', direction='in')
    ax.set_title(title)


def print_params(p):
    """Render a Params instance as a LaTeX parameter table (Jupyter/IPython display)."""
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
