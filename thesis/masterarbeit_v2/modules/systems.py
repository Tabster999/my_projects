"""
systems.py - Object-oriented interface for SNS junction calculations
====================================================================

Provides the ``SNSJunction`` class: a high-level object that holds all
physical parameters and geometry, builds Hamiltonians on demand, and exposes
named methods for computing observables.

Typical usage
-------------
::

    from modules.systems import SNSJunction

    junc = SNSJunction(
        t=1.0, mu_sc=0.1e-3, mu_n=0.7e-3,
        alpha=16e-3, h=0.5e-3, delta=250e-6,
        SL=200, SM=100, SR=200,
        eta=1e-5,
    )

    # LDOS at several probe sites vs energy
    energies = np.linspace(-0.2, 0.2, 151)
    ldos = junc.ldos_vs_energy(energies, phi=np.pi, probe_x=[0, 50, 99], probe_y=[0, 5, 9])

    # Full LDOS map (spatial) at zero energy, infinite leads
    ldos_map = junc.ldos_spatial_map(energy=0.0, phi=np.pi)

    # LDOS vs phase at fixed energy and a single probe site
    phases = np.linspace(0, 2*np.pi, 151)
    ldos_phi = junc.ldos_vs_phase(phases, energy=0.0, probe_x=[50], probe_y=[5])

    # 2D (energy × phase) sweep at a single probe
    ldos_2d = junc.ldos_2d_sweep(energies, phases, probe_x=0, probe_y=0)

Public API
----------
Methods that return observables
    ldos_vs_energy      – LDOS(E) at probe sites for fixed φ
    ldos_spatial_map    – LDOS(x, y) at fixed E and φ
    ldos_vs_phase       – LDOS(φ) at probe sites for fixed E
    ldos_2d_sweep       – LDOS(E, φ) at a single probe site
    green_function      – raw Green's function (diagonal blocks) at (E, φ)
    pairing_vs_energy   – anomalous amplitude |G[0,3]| vs E at probe sites
    pairing_vs_phase    – anomalous amplitude |G[0,3]| vs φ at probe sites

Helper / geometry
    get_local_block     – extract (dof × dof) block for site y from a slice GF
    build_lead_slice    – return the unphased SC lead slice for this junction

All heavy lifting is parallelised with ``joblib`` where beneficial.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Union
from joblib import Parallel, delayed

from . import matrices as mat
from . import solvers  as sol
from .helpers import get_pairing_amplitude


# ============================================================================
# UTILITY
# ============================================================================

def _get_local_block(G_slice: np.ndarray, y: int, dof: int = 4) -> np.ndarray:
    """Extract the (dof × dof) on-site block for transverse site y."""
    s = y * dof
    return G_slice[s:s + dof, s:s + dof]


def _ldos_from_block(block: np.ndarray) -> float:
    return -np.imag(np.trace(block)) / np.pi


# ============================================================================
# MAIN CLASS
# ============================================================================

class SNSJunction:
    """
    Superconductor–Normal–Superconductor junction.

    All lengths are in units of the lattice constant.
    All energies are in the same units as ``t``.

    Parameters
    ----------
    t : float
        Hopping amplitude (sets the energy scale).
    mu_sc : float
        Chemical potential in the SC leads.
    mu_n : float
        Chemical potential in the normal region.
    alpha : float
        Rashba spin-orbit coupling strength.
    h : float
        Zeeman energy (exchange field / applied B).
    delta : float
        Superconducting pairing amplitude (real; phase is applied separately).
    SL : int
        Number of slices in the left SC lead (1D: sites; 2D: columns).
    SM : int
        Number of slices in the normal region.
    SR : int
        Number of slices in the right SC lead.
    N_y : int
        Number of sites in the transverse (y) direction.
        Use ``N_y=1`` for a strictly 1D junction.
    eta : float
        Imaginary broadening for Green's function denominators.
    symmetric : bool
        Gauge choice.  ``True`` → ±φ/2 (symmetric);
        ``False`` → 0/φ (asymmetric, default).
    n_jobs : int
        Number of parallel workers for sweeps (default -1 = all cores).
    """

    def __init__(
        self,
        t: float,
        mu_sc: float,
        mu_n: float,
        alpha: float,
        h: float,
        delta: float,
        SL: int,
        SM: int,
        SR: int,
        N_y: int = 1,
        eta: float = 1e-4,
        symmetric: bool = False,
        n_jobs: int = -1,
    ) -> None:
        self.t         = t
        self.mu_sc     = mu_sc
        self.mu_n      = mu_n
        self.alpha     = alpha
        self.h         = h
        self.delta     = delta
        self.SL        = SL
        self.SM        = SM
        self.SR        = SR
        self.N_y       = N_y
        self.eta       = eta
        self.symmetric = symmetric
        self.n_jobs    = n_jobs

        self._is_2d = N_y > 1
        self.dof    = 4 * N_y  # degrees of freedom per slice

    # ------------------------------------------------------------------
    # GEOMETRY / HAMILTONIAN BUILDERS
    # ------------------------------------------------------------------

    def build_lead_slice(self) -> np.ndarray:
        """
        Return the unphased SC lead slice Hamiltonian.

        For 1D (N_y=1) this is a (4, 4) matrix; for 2D it is (4*N_y, 4*N_y).
        The phase is applied by the surface-GF routines internally.
        """
        if self._is_2d:
            return mat.get_lead_slice_2d(
                self.N_y, self.t, self.mu_sc, self.h, self.alpha, self.delta
            )
        else:
            return mat.onsite_matrix(self.t, self.mu_sc, self.h, self.delta)

    def build_normal_slices(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return (H_slices, V) for the normal region only.

        Used when the SC leads are replaced by surface Green's functions.
        """
        if self._is_2d:
            return mat.build_sns_normal_only_2d(
                self.N_y, self.t, self.mu_n, self.h, self.alpha, self.SM
            )
        else:
            return mat.build_middle_region(
                self.t, self.mu_n, self.alpha, self.h, self.SM
            )

    def build_full_slices(self, phi: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return (H_slices, V) for the entire SNS junction at phase ``phi``.

        Includes SC lead slices — suitable for the finite-system RGF.
        """
        if self._is_2d:
            return mat.build_sns_junction_sliced_2d(
                self.N_y, self.t, self.mu_sc, self.mu_n, self.h, self.alpha,
                self.delta, phi, self.SL, self.SR, self.SM,
                symmetric=self.symmetric
            )
        else:
            return mat.build_sns_junction_sliced(
                self.t, self.mu_sc, self.mu_n, self.alpha, self.h,
                self.delta, phi, self.SL, self.SR, self.SM,
                symmetric=self.symmetric
            )

    def get_local_block(self, G_slice: np.ndarray, y: int) -> np.ndarray:
        """Extract the (4 × 4) on-site block for transverse index ``y``."""
        return _get_local_block(G_slice, y, dof=4)

    # ------------------------------------------------------------------
    # SURFACE GREEN'S FUNCTIONS
    # ------------------------------------------------------------------

    def _get_surface_gfs(self, energy: float, phi: float):
        """Return (g_L, g_R) surface GFs with phase rotation applied."""
        H_lead = self.build_lead_slice()
        _, V   = self.build_normal_slices()   # V is the same for lead and normal

        if self._is_2d:
            return sol.get_surface_gfs_2d_phased(
                energy, H_lead, V, self.N_y, phi,
                symmetric=self.symmetric, eta=self.eta
            )
        else:
            return sol.get_surface_gfs_phased(
                energy, H_lead, V, self.N_y, phi,
                symmetric=self.symmetric, eta=self.eta
            )

    # ------------------------------------------------------------------
    # RAW GREEN'S FUNCTION ACCESS
    # ------------------------------------------------------------------

    def green_function(
        self,
        energy: float,
        phi: float,
        mode: str = "infinite",
        return_full: bool = False,
    ) -> np.ndarray:
        """
        Compute the Green's function diagonal blocks at a single (E, φ).

        Parameters
        ----------
        energy      : energy at which to evaluate G
        phi         : SC phase difference
        mode        : ``'infinite'`` (surface-GF leads) or ``'finite'``
                      (explicit SC slices, open boundaries)
        return_full : if True, also compute and return off-diagonal blocks

        Returns
        -------
        G_diag : ndarray, shape (N_slices, dof, dof)
            For ``mode='infinite'`` N_slices = SM;
            for ``mode='finite'``  N_slices = SL + SM + SR.
        """
        if mode == "infinite":
            H_slices, V = self.build_normal_slices()
            g_L, g_R   = self._get_surface_gfs(energy, phi)
            G_diag, *_ = sol.get_rgf_sns(
                H_slices, V, g_L, g_R, energy,
                eta=self.eta, return_full=return_full
            )
        elif mode == "finite":
            H_slices, V = self.build_full_slices(phi)
            G_diag, *_ = sol.get_rgf_finite_system(
                H_slices, V, energy,
                eta=self.eta, return_full=return_full
            )
        else:
            raise ValueError(f"mode must be 'infinite' or 'finite', got {mode!r}")

        return G_diag

    # ------------------------------------------------------------------
    # OBSERVABLES
    # ------------------------------------------------------------------

    def ldos_vs_energy(
        self,
        energies: np.ndarray,
        phi: float,
        probe_x: List[int],
        probe_y: List[int],
        modes: Union[str, List[str]] = "infinite",
    ) -> np.ndarray:
        """
        LDOS as a function of energy at probe sites for a fixed phase.

        Parameters
        ----------
        energies : 1D array of energies
        phi      : SC phase difference
        probe_x  : slice indices to probe (0-based, within the normal region)
        probe_y  : transverse site indices to probe
        modes    : ``'infinite'``, ``'finite'``, or ``['infinite', 'finite']``

        Returns
        -------
        ldos : ndarray, shape (n_probe_x, n_probe_y, N_E, n_modes)
        """
        if isinstance(modes, str):
            modes = [modes]

        def _one_energy(E):
            result = []
            for mode in modes:
                G = self.green_function(E, phi, mode=mode)
                offset = self.SL if mode == "finite" else 0
                out = np.zeros((len(probe_x), len(probe_y)))
                for ix, x in enumerate(probe_x):
                    for iy, y in enumerate(probe_y):
                        blk = _get_local_block(G[offset + x], y)
                        out[ix, iy] = _ldos_from_block(blk)
                result.append(out)
            return np.stack(result, axis=-1)  # (n_px, n_py, n_modes)

        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_one_energy)(E) for E in energies
        )
        return np.stack(out_list, axis=2)   # (n_px, n_py, N_E, n_modes)

    def ldos_spatial_map(
        self,
        energy: float,
        phi: float,
        mode: str = "infinite",
    ) -> np.ndarray:
        """
        Spatial LDOS map across the full normal region at a single (E, φ).

        Parameters
        ----------
        energy : single energy value
        phi    : SC phase difference
        mode   : ``'infinite'`` or ``'finite'``

        Returns
        -------
        ldos_map : ndarray, shape (SM, N_y)
        """
        G      = self.green_function(energy, phi, mode=mode)
        offset = self.SL if mode == "finite" else 0
        ldos   = np.zeros((self.SM, self.N_y))
        for x in range(self.SM):
            for y in range(self.N_y):
                blk = _get_local_block(G[offset + x], y)
                ldos[x, y] = _ldos_from_block(blk)
        return ldos

    def ldos_vs_phase(
        self,
        phases: np.ndarray,
        energy: float,
        probe_x: List[int],
        probe_y: List[int],
        modes: Union[str, List[str]] = "infinite",
    ) -> np.ndarray:
        """
        LDOS as a function of SC phase difference at probe sites.

        Parameters
        ----------
        phases   : 1D array of phase values
        energy   : fixed energy
        probe_x  : slice indices (within normal region)
        probe_y  : transverse site indices
        modes    : ``'infinite'``, ``'finite'``, or list of both

        Returns
        -------
        ldos : ndarray, shape (n_probe_x, n_probe_y, N_PHI, n_modes)
        """
        if isinstance(modes, str):
            modes = [modes]

        def _one_phi(phi):
            result = []
            for mode in modes:
                G = self.green_function(energy, phi, mode=mode)
                offset = self.SL if mode == "finite" else 0
                out = np.zeros((len(probe_x), len(probe_y)))
                for ix, x in enumerate(probe_x):
                    for iy, y in enumerate(probe_y):
                        blk = _get_local_block(G[offset + x], y)
                        out[ix, iy] = _ldos_from_block(blk)
                result.append(out)
            return np.stack(result, axis=-1)

        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_one_phi)(phi) for phi in phases
        )
        return np.stack(out_list, axis=2)   # (n_px, n_py, N_PHI, n_modes)

    def ldos_2d_sweep(
        self,
        energies: np.ndarray,
        phases: np.ndarray,
        probe_x: int,
        probe_y: int,
        modes: Union[str, List[str]] = "infinite",
    ) -> np.ndarray:
        """
        LDOS over a 2D grid of (energy, phase) at a single probe site.

        Parameters
        ----------
        energies : 1D array of energies, length N_E
        phases   : 1D array of phases,   length N_PHI
        probe_x  : single slice index (within normal region)
        probe_y  : single transverse site index
        modes    : ``'infinite'``, ``'finite'``, or list of both

        Returns
        -------
        ldos : ndarray, shape (N_E, N_PHI, n_modes)
        """
        if isinstance(modes, str):
            modes = [modes]

        def _one_energy_row(E):
            row = np.zeros((len(phases), len(modes)))
            for p_idx, phi in enumerate(phases):
                for m_idx, mode in enumerate(modes):
                    G      = self.green_function(E, phi, mode=mode)
                    offset = self.SL if mode == "finite" else 0
                    blk    = _get_local_block(G[offset + probe_x], probe_y)
                    row[p_idx, m_idx] = _ldos_from_block(blk)
            return row   # (N_PHI, n_modes)

        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(_one_energy_row)(E) for E in energies
        )
        return np.stack(rows, axis=0)   # (N_E, N_PHI, n_modes)

    def pairing_vs_energy(
        self,
        energies: np.ndarray,
        phi: float,
        probe_x: List[int],
        probe_y: List[int],
        modes: Union[str, List[str]] = "infinite",
    ) -> np.ndarray:
        """
        Anomalous pairing amplitude |G[0,3]| vs energy at probe sites.

        Returns
        -------
        pairing : ndarray, shape (n_probe_x, n_probe_y, N_E, n_modes)
        """
        if isinstance(modes, str):
            modes = [modes]

        def _one_energy(E):
            result = []
            for mode in modes:
                G = self.green_function(E, phi, mode=mode)
                offset = self.SL if mode == "finite" else 0
                out = np.zeros((len(probe_x), len(probe_y)))
                for ix, x in enumerate(probe_x):
                    for iy, y in enumerate(probe_y):
                        blk = _get_local_block(G[offset + x], y)
                        out[ix, iy] = get_pairing_amplitude(blk)
                result.append(out)
            return np.stack(result, axis=-1)

        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_one_energy)(E) for E in energies
        )
        return np.stack(out_list, axis=2)

    def pairing_vs_phase(
        self,
        phases: np.ndarray,
        energy: float,
        probe_x: List[int],
        probe_y: List[int],
        modes: Union[str, List[str]] = "infinite",
    ) -> np.ndarray:
        """
        Anomalous pairing amplitude |G[0,3]| vs phase at probe sites.

        Returns
        -------
        pairing : ndarray, shape (n_probe_x, n_probe_y, N_PHI, n_modes)
        """
        if isinstance(modes, str):
            modes = [modes]

        def _one_phi(phi):
            result = []
            for mode in modes:
                G = self.green_function(energy, phi, mode=mode)
                offset = self.SL if mode == "finite" else 0
                out = np.zeros((len(probe_x), len(probe_y)))
                for ix, x in enumerate(probe_x):
                    for iy, y in enumerate(probe_y):
                        blk = _get_local_block(G[offset + x], y)
                        out[ix, iy] = get_pairing_amplitude(blk)
                result.append(out)
            return np.stack(result, axis=-1)

        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_one_phi)(phi) for phi in phases
        )
        return np.stack(out_list, axis=2)

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dim = f"2D (N_y={self.N_y})" if self._is_2d else "1D"
        return (
            f"SNSJunction({dim}, "
            f"SL={self.SL}, SM={self.SM}, SR={self.SR}, "
            f"t={self.t}, mu_sc={self.mu_sc}, mu_n={self.mu_n}, "
            f"alpha={self.alpha}, h={self.h}, delta={self.delta}, "
            f"eta={self.eta}, symmetric={self.symmetric})"
        )
