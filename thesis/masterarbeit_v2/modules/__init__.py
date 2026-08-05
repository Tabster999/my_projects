"""
modules - Small local library for SNS/Josephson-junction NEGF simulations
==========================================================================

Submodules
----------
helpers   : generic utilities (phase_matrix, edge indices, LDOS/pairing
            traces, embed_block, ...) - no physics-heavy logic, just
            building blocks used everywhere else.
matrices  : Hamiltonian builders. Two basis conventions live side by side:
              - plain Nambu (c up, c down, c down dagger, c up dagger):
                onsite_matrix, t_matrix_x, t_matrix_y, build_sns_* functions
              - Scharf-Pientka (c up, c down, c down dagger, minus c up dagger),
                suffixed _sp: onsite_matrix_sc_sp, onsite_matrix_normal_sp,
                hopping_x_sp, hopping_y_sp, make_slice_*_sp, make_x_hopping_sp
            These two conventions are NOT interchangeable - see the basis
            note at the top of matrices.py before mixing them.
solvers   : Green's function solvers - Sancho-Rubio surface GFs
            (get_surface_gf), phase gauge (apply_phase_gauge), and RGF
            sweeps (get_rgf_sns, get_rgf_finite_system, get_rgf_phi_sweep).
transport : Multi-terminal self-energies, broadening matrices, edge-lead
            embedding (attach_edge_lead, build_dressed_slices), and
            Landauer-Buttiker / current-phase current formulas.

Typical usage
-------------
    from modules import matrices, solvers, transport, helpers

    H_slice = matrices.make_slice_normal_sp(...)
    g_surf, _ = solvers.get_surface_gf(energy, H_lead, V_lead, eta=eta)
    Sigma = transport.self_energy(V_couple, g_surf)
    ldos = helpers.ldos_trace(G, idx)
"""

from . import helpers
from . import matrices
from . import solvers
from . import transport

__all__ = ["helpers", "matrices", "solvers", "transport"]