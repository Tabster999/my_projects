"""
modules/__init__.py
===================

Flat re-export of all public symbols so existing call-sites like

    import modules
    modules.get_rgf_sns(...)
    modules.onsite_matrix(...)

continue to work unchanged, while new code can also do

    from modules import SNSJunction
    from modules.systems import SNSJunction
    from modules.matrices import build_sns_slice_2d
"""

# ── matrix / Hamiltonian builders ────────────────────────────────────────────
from .matrices import (
    # elementary blocks
    onsite_matrix,
    t_matrix_x,
    t_matrix_y,
    # 1D builders
    get_tb_hamiltonian,
    build_sns_junction,
    build_sns_junction_sliced,
    build_middle_region,
    # 2D builders
    build_sns_slice_2d,
    build_sns_junction_sliced_2d,
    build_sns_normal_only_2d,
    get_lead_slice_2d,
)

# ── solvers ───────────────────────────────────────────────────────────────────
from .solvers import (
    get_surface_gf,
    calc_G,
    get_G_energy,
    get_G_lehmann,
    get_G_k_energy,
    get_rgf_sns,
    get_rgf_finite_system,
    get_surface_gfs_phased,
    get_surface_gfs_2d_phased,
)

# ── helpers ───────────────────────────────────────────────────────────────────
from .helpers import (
    phase_matrix,
    get_z,
    get_ldos_site,
    get_pairing_amplitude,
    get_pairing_phase,
    check_particle_hole_symmetry,
    check_ldos_positivity,
    get_colorbar_label,
)

# ── high-level OOP interface ──────────────────────────────────────────────────
from .systems import SNSJunction

__all__ = [
    # matrices
    "onsite_matrix", "t_matrix_x", "t_matrix_y",
    "get_tb_hamiltonian",
    "build_sns_junction", "build_sns_junction_sliced", "build_middle_region",
    "build_sns_slice_2d", "build_sns_junction_sliced_2d",
    "build_sns_normal_only_2d", "get_lead_slice_2d",
    # solvers
    "get_surface_gf", "calc_G", "get_G_energy", "get_G_lehmann", "get_G_k_energy",
    "get_rgf_sns", "get_rgf_finite_system",
    "get_surface_gfs_phased", "get_surface_gfs_2d_phased",
    # helpers
    "phase_matrix", "get_z", "get_ldos_site",
    "get_pairing_amplitude", "get_pairing_phase",
    "check_particle_hole_symmetry", "check_ldos_positivity",
    "get_colorbar_label",
    # OOP
    "SNSJunction",
]
