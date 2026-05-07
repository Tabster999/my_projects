"""
config.py - Centralized Parameters & Constants
===============================================

This is Step 2 of your workflow: Define Parameters

All default values are here. You can import and override them as needed:

    from config import DEFAULT_PARAMS
    params = DEFAULT_PARAMS.copy()
    params['Delta'] = 0.2  # override a value
"""

# ============================================================================
# DEFAULT PARAMETERS FOR SNS JUNCTION CALCULATIONS
# ============================================================================

DEFAULT_PARAMS = {
    # ── Material Parameters ────────────────────────────────────────────────
    "t": 1.0,                  # Hopping amplitude (energy scale)
    "mu_sc": 0.0025,           # Chemical potential in superconductor
    "mu_n": 0.01,              # Chemical potential in normal region
    "alpha": 0.30,             # Rashba spin-orbit coupling strength
    "B": 0.30,                 # Zeeman field (magnetic field)
    "Delta": 0.10,             # Superconducting gap
    
    # ── System Geometry ────────────────────────────────────────────────────
    "SL": 100,                 # Sites in left superconductor
    "SM": 50,                  # Sites in middle (normal region)
    "SR": 100,                 # Sites in right superconductor
    "DOF": 4,                  # Degrees of freedom (spin-up, spin-down, their conjugates)
    
    # ── Numerical Parameters ───────────────────────────────────────────────
    "eta": 1e-4,               # Broadening parameter for Green's functions
    "symmetric": False,        # Gauge: True → ±φ/2 on left/right, False → φ on right only
    
    # ── Energy/Phase Sweep ─────────────────────────────────────────────────
    "N_E": 81,                # Number of energy points in sweep
    "N_PHI": 81,              # Number of phase points in sweep
    "E_min": None,             # Minimum energy (set to -3*Delta if None)
    "E_max": None,             # Maximum energy (set to +3*Delta if None)
    
    # ── Matsubara Frequency (for Josephson Current) ────────────────────────
    "T": 1e-4,                 # Temperature
    "N_MATS": 100,             # Number of Matsubara frequencies
    
    # ── Plotting ───────────────────────────────────────────────────────────
    "figsize": (12, 8),        # Figure size
    "dpi": 150,                # Resolution
    "cmap": "RdBu_r",          # Colormap for heatmaps
}

# ============================================================================
# COLOUR PALETTE (colour-blind friendly)
# ============================================================================

COLORS = {
    "inversion": "#2166ac",    # blue - direct matrix inversion
    "finite_rgf": "#d6604d",   # red - finite RGF
    "infinite_leads": "#4dac26", # green - Sancho-López
    "alpha_default": 0.85,
}

# ============================================================================
# PLOTTING STYLES
# ============================================================================

PLOT_STYLE = {
    "font_size": 14,
    "title_size": 18,
    "label_size": 16,
    "tick_size": 12,
    "legend_size": 10,
    "figure_title_size": 20,
}


def get_energy_range(params: dict) -> tuple:
    """
    Get energy range from parameters.
    If E_min/E_max are None, use ±3*Delta
    
    Returns:
        tuple: (E_min, E_max)
    """
    E_min = params.get("E_min")
    E_max = params.get("E_max")
    Delta = params["Delta"]
    
    if E_min is None:
        E_min = -3 * Delta
    if E_max is None:
        E_max = 3 * Delta
    
    return E_min, E_max
