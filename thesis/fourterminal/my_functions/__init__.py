"""
my_functions
============

NEGF simulation package for the 4-terminal Josephson junction
(SC leads top/bottom, normal leads left/right), in the Scharf-Pientka
Nambu-spin basis Psi = (c_up, c_down, c_down^dagger, -c_up^dagger).

Modules
-------
params        : Params dataclass holding all physical/numerical parameters
hamiltonians  : onsite/hopping block builders, lattice assembly, index helpers
leads         : Lead class (Sancho-Rubio surface Green's function + self-energy)
junction      : FourTerminalJunction class (assembles H_C, leads, channels())
transport     : Fermi functions, dc currents, conductance matrix, bias maps,
                linear-response thermal conductance and heat currents
plotting      : shared plot styling helpers
analysis      : phase/bias maps, thermal (phi, kT) sweeps, parameter-grid scans

Typical usage
-------------
    from fourterminal import Params, FourTerminalJunction
    from fourterminal.transport import dc_current_channels, conductance_matrix

    p = Params(nx=11, ny=2, delta=0.35, phi=np.pi/4)
    junction = FourTerminalJunction(p)
    channels = junction.channels(E_sweep)
"""

from .params import Params
from .hamiltonians import (
    onsite_block,
    Vx,
    Vy,
    make_row_hamiltonian,
    get_2d_hamiltonian,
    flat_site_idx,
)
from .leads import Lead
from .junction import FourTerminalJunction, lead_escape
from .transport import (
    f_electron,
    f_hole,
    dc_current_channels,
    other_name,
    conductance_matrix,
    partial_G_vectorized,
    eval_I_total,
    total_dIdV_map,
    minus_dfdE,
    thermal_grid,
    nonlocal_T,
    linear_response_coeffs,
    heat_current,
)
from .plotting import style_axis, print_params
from .fast_phase_sweep import FastPhaseSweep
from .analysis import (
    compute_phase_bias_maps,
    compute_thermal_vs_phi,
    summarize_map,
    scan_grid,
    plot_grid_thumbnails,
    plot_summary_trends,
)

__all__ = [
    "Params",
    "onsite_block",
    "Vx",
    "Vy",
    "make_row_hamiltonian",
    "get_2d_hamiltonian",
    "flat_site_idx",
    "Lead",
    "FourTerminalJunction",
    "f_electron",
    "f_hole",
    "dc_current_channels",
    "other_name",
    "conductance_matrix",
    "partial_G_vectorized",
    "eval_I_total",
    "total_dIdV_map",
    "minus_dfdE",
    "thermal_grid",
    "nonlocal_T",
    "linear_response_coeffs",
    "heat_current",
    "lead_escape",
    "style_axis",
    "print_params",
    "FastPhaseSweep",
    "compute_phase_bias_maps",
    "compute_thermal_vs_phi",
    "summarize_map",
    "scan_grid",
    "plot_grid_thumbnails",
    "plot_summary_trends",
]