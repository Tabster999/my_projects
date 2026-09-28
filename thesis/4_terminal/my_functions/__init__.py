"""
my_functions
============

NEGF simulation package for the 4-terminal Josephson junction
(SC ribbons top/bottom, normal leads left/right), in the Nambu-spin basis
Psi = (c_up, c_down, c_down^dagger, -c_up^dagger).

Two Hamiltonian models share all geometry/lead/solver code (Params.model):
  'rashba' : 2DEG with Rashba/Dresselhaus SOC, Zeeman and s-wave pairing
  'dirac'  : lattice Dirac-BdG model with Wilson mass (Gresta et al., PRB 114, 125405, Eq. (1))

Modules
-------
params           : Params dataclass
hamiltonians     : onsite/hopping blocks of both models, lattice assembly
leads            : Lead (Sancho-Rubio surface GF, self-consistency check + Newton refinement)
junction         : FourTerminalJunction (geometry, leads, dense channels() reference)
rgf              : RGFFourTerminal (fast, robust; use this for large systems and maps)
fast_phase_sweep : FastPhaseSweep (dense, cheap phase updates; small systems)
transport        : Fermi functions, currents, conductances, linear-response nodes
scans            : fast parallel (kappa, G) parameter scans (PHS-halved energies, central-solve reuse)
spectra          : bulk Chern number, SC-ribbon gap, junction spectrum with PBC in x
plotting, analysis : helpers
"""

from .params import Params
from .hamiltonians import (
    onsite_block, Vx, Vy, dirac_onsite_block, dirac_Vx, dirac_Vy,
    model_onsite, model_hop, make_row_hamiltonian, get_2d_hamiltonian, flat_site_idx,
)
from .leads import Lead
from .junction import FourTerminalJunction
from .rgf import RGFFourTerminal, central_fingerprint
from .fast_phase_sweep import FastPhaseSweep
from .transport import (
    f_electron, f_hole, dc_current_channels, other_name, conductance_matrix,
    partial_G_vectorized, eval_I_total, total_dIdV_map, linear_response_nodes, kappa_nodes, kappa_from_channels,
    linear_response_nodes_phs, thermal_from_channels_phs,
)
from .plotting import style_axis, print_params, param_title
from .scans import thermal_point, thermal_scan, make_pool
from .spectra import chern_number, junction_spectrum_kx, ribbon_gap
from .analysis import (
    compute_phase_bias_maps, summarize_map, scan_grid, plot_grid_thumbnails, plot_summary_trends,
)
