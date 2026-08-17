"""Central parameter container for the 4-terminal junction simulation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    # --- geometry ---
    nx: int = 10
    ny: int = 1

    # --- normal region (central, x-hop) ---
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

    # --- couplings ---
    tc_top: float = 0.6
    tc_bot: float = 0.6
    tc_barr: float = 0.5

    # --- spin-orbit / Zeeman ---
    alpha: float = 0.0    # Rashba
    beta: float = 0.0     # Dresselhaus
    Bz: float = 0.0       # out-of-plane Zeeman
    Bxy: float = 0.0      # in-plane Zeeman magnitude
    theta_z: float = 0.0  # in-plane Zeeman angle

    # --- numerics ---
    eta: float = 1e-5
    max_iter: int = 450
    tol: float = 1e-14
    kT: float = 1e-3
