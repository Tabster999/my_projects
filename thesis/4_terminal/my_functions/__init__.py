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
fast_phase_sweep : FastPhaseSweep (dense reference solver for small systems / checks)
transport        : finite-bias current/conductances; thermal linear response (trapezoid, PHS-halved)
scans            : scan (general parallel, resumable scans), ThermalPoint, thermal_point/thermal_scan
compute          : one-call functions returning arrays: transmissions, linear_response, ldos, currents, conductances
spectra          : Chern number, SC-ribbon edge gap, Majorana decay length, junction spectrum, LDOS
plotting         : param_title (parameter string for figure titles)
"""

from .params import Params
from .hamiltonians import (
    onsite_block, Vx, Vy, dirac_onsite_block, dirac_Vx, dirac_Vy,
    model_onsite, model_hop, region_alpha, bond_alpha, make_row_hamiltonian, get_2d_hamiltonian, flat_site_idx,
)
from .leads import Lead
from .junction import FourTerminalJunction
from .rgf import RGFFourTerminal, central_fingerprint
from .fast_phase_sweep import FastPhaseSweep
from .transport import (
    f_electron, f_hole, current, differential_conductance,
    dc_current_channels, other_name, conductance_matrix, partial_G_vectorized, eval_I_total, total_dIdV_map,
    linear_response_nodes, linear_response_nodes_phs, thermal_from_channels_phs, thermal_error_phs,
)
from .spectra import (chern_number, junction_spectrum_kx, ribbon_gap, LocalGreen,
                      edge_state_profile, coherence_length)
from .scans import scan, ThermalPoint, thermal_point, thermal_scan, make_pool
from .compute import transmissions, linear_response, ldos, currents, conductances
from .plotting import style_axis, print_params, param_title
