"""Central parameter container for the 4-terminal junction simulation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    # --- model ---
    # 'rashba' : 2DEG, quadratic band (onsite 4t - mu) + Rashba/Dresselhaus SOC
    # 'dirac'  : lattice Dirac-BdG model with Wilson mass, Gresta et al., PRB 114, 125405, Eq. (1):
    #            H = sum_a t sin(k_a) s_a tz + [Z + m(k)] sz t0 - mu tz + Delta t_phi,
    #            m(k) = m0 (2 - cos kx - cos ky); alpha/beta are ignored.
    model: str = 'rashba'

    # --- geometry ---
    nx: int = 10
    ny: int = 1

    # --- normal leads (left/right) ---
    t_n: float = 1.0
    mu_n: float = 0.0

    # --- central 2D region ---
    t_c: float = 1.0
    mu_c: float = 0.0

    # --- SC ribbons (top/bottom) ---
    t_s: float = 1.0
    mu_s: float = 0.0
    delta: float = 0.1
    phi: float = np.pi

    # --- couplings (tc replaces t on the bond; SOC / Wilson term stay) ---
    tc_top: float = 0.6
    tc_bot: float = 0.6
    tc_barr: float = 0.5

    # --- spin-orbit / Zeeman ---
    alpha: float = 0.0      # Rashba          (model='rashba' only)
    beta: float = 0.0       # Dresselhaus     (model='rashba' only)
    Bz: float = 0.0         # out-of-plane Zeeman, central region + normal leads
    Bxy: float = 0.0        # in-plane Zeeman magnitude (all regions)
    theta_z: float = 0.0    # in-plane Zeeman angle, central region + SC ribbons
    theta_z_n: float = 0.0  # in-plane Zeeman angle, normal leads
    Bz_s: float = 0.0       # out-of-plane Zeeman in SC ribbons (= Z in Gresta et al.)
    # optional per-region overrides (None = use the global value above)
    alpha_c: float = None   # Rashba in the central region
    alpha_n: float = None   # Rashba in the normal leads
    alpha_s: float = None   # Rashba in the SC ribbons
    Bz_n: float = None      # out-of-plane Zeeman in the normal leads (default: Bz)

    # --- Wilson masses (model='dirac' only); paper: all 0.8, appendix: m0_c = m0_n = 0 ---
    m0: float = 0.8         # SC ribbons
    m0_c: float = 0.8       # central region
    m0_n: float = 0.8       # normal leads

    # --- numerics ---
    eta: float = 1e-5
    max_iter: int = 100     # Sancho-Rubio steps (step n couples 2^n cells)
    tol: float = 1e-14
    lead_refine: bool = True  # Newton-refine Sancho-Rubio results that fail the self-consistency check
    kT: float = 1e-3

    def __post_init__(self):
        if self.model not in ('rashba', 'dirac'):
            raise ValueError(f"model must be 'rashba' or 'dirac', got {self.model!r}")
