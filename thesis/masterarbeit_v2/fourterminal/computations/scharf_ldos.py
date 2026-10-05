"""
Phase-controlled planar Josephson junction with an in-plane Zeeman field, in PHYSICAL UNITS
(Scharf et al., PRB 99, 214503): edge/bulk LDOS, real-space Majorana bound states, conductances.
Uses the 4_terminal package.

GEOMETRY.  Axis letters follow SCHARF throughout this file:
    x = ACROSS the junction,  W = 100 nm  ->  5 sites at a = 20 nm
    y = ALONG the interfaces, L = 2000 nm -> 100 sites
Internally the package uses the opposite letters (its nx counts sites ALONG the interfaces =
Scharf's y, its ny counts sites ACROSS = Scharf's x); PhysParams does the translation, so you
never need the internal names.
    SC ribbons above/below the centre (semi-infinite), phase difference phi, trivial (Bz_s = 0)
    in-plane Zeeman ONLY in the normal region (Bxy_c), as in Scharf's (V0 tau_z - E_Z.s) h(x)
    normal leads at the two ends = weakly coupled tunneling probes (tc_barr small; 0 = hard walls)
    Majorana bound states appear at the ENDS of the normal region, y = +-L/2.

SPIN-ORBIT CONVENTION.  Params.soc_axis selects the crystallographic setup, in this package's
geometry (x along the interfaces, y across them):
    '100'  (Scharf Eq. 1, used for most of the paper):  H_soc = -(alpha s_x + beta s_y) sin kx
                                                                + (beta s_x + alpha s_y) sin ky
    '110'  (Scharf Eq. 17, Secs. IV / Figs. 6, 10):     H_soc = (alpha - beta) sin kx s_y
                                                                - (alpha + beta) sin ky s_x
The field must be perpendicular to the spin-orbit field n_soc of propagation ALONG the
interfaces (the sin kx term).  In this package's Zeeman convention the in-plane field points
along (cos theta_z, -sin theta_z) in (s_x, s_y), so the optimum is
    '100':  theta_z = arctan2(alpha, beta)      (rotates with beta/alpha; = pi/2 for pure Rashba)
    '110':  theta_z = 0                          (fixed, for any alpha, beta)
PhysParams.theta_z_optimal() returns this.  Note the sign: in the paper's own convention the
[100] optimum reads arctan2(alpha, -beta); the sign of beta flips here because of how theta_z
enters our Bxy term.  Both were checked by scanning theta_z numerically.
The topological gap closes where the relevant n_soc vanishes: alpha = beta for '110'.
Adding pi to theta_z reverses the field; this MIRRORS every map about phi = pi, because the
symmetry of the Hamiltonian is (phi -> -phi together with E_Z -> -E_Z), not either alone.

UNITS.  PhysParams holds meV / nm / T and converts to the lattice units the package uses:
    t = hbar^2 / (2 m a^2),  mu_t = mu/t,  Delta_t = Delta/t,  alpha_t = alpha/(a t),
    E_Z = g mu_B B / 2  (note the 1/2),  eta = eta_rel * Delta_t.
Validated against the paper: t = 2.49 meV and E_T = 829 / 276 / 166 ueV for W = 100 / 300 /
500 nm, against the paper's 832 / 277 / 166 ueV.

TRANSPORT vs LDOS SETTINGS.  The two cannot share one parameter set.  Scharf's LDOS uses
decoupled probes (tc_barr = 0) and a broad eta = 0.05 Delta; both make every transport quantity
vanish identically -- no probe coupling, and a broadening far wider than the Majorana resonance.
kappa/G therefore use their own fields (tc_barr_transport, eta_rel_transport) through
params_transport(), while the LDOS keeps pp.tc_barr / pp.eta_rel.  At the paper's junction and
E_Z = 0.5 meV, phi = pi this gives kappa/kappa0 = 0.491 with G/G0 = 0 (T_ee = T_he to four
digits: the Majorana condition), against 0.000 for both at eta = 0.05 Delta.
The coupling must stay WEAK: at tc_barr ~ 1 the end states hybridise with the leads and kappa
collapses to 0.001.  Note this is a resonance, not a plateau -- its width is the hybridisation
splitting of the two end Majoranas across L, not a topological gap (Scharf's ribbons are trivial,
Bz_s = 0), so it is gone above kT ~ 0.1 ueV (~1 mK).
"""
# %% --- IMPORTS ---
import os
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

try:
    import my_functions as myf
except ModuleNotFoundError:                  # not pip-installed: find the package by walking up
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(next(p for p in (_here, *_here.parents) if (p / "my_functions").is_dir())))
    import my_functions as myf

_hbar, _e, _m0, _muB = 1.05e-34, 1.602e-19, 9.1e-31, 5.78e-2      # J s, C, kg, meV/T


# %% --- UNITS ---
@dataclass
class PhysParams:
    """Physical parameters (meV, nm, T) and their conversion to the lattice model."""
    a: float = 20.0            # nm, lattice constant
    m_eff: float = 0.038       # in units of m0
    Delta: float = 0.25        # meV, induced gap in the SC regions
    mu_S: float = 1.0          # meV, chemical potential in the SC regions
    mu_N: float = 0.7          # meV, chemical potential in the normal region
    alpha: float = 16.0        # meV nm, Rashba SOC
    beta: float = 0.0          # meV nm, Dresselhaus SOC
    soc_axis: str = '100'      # '100' (Scharf Eq. 1, most of the paper) or '110' (Eq. 17)
    g: float = 10.0            # g factor
    B: float = 2.0             # T, in-plane magnetic field
    theta_z: float = None      #type: ignore    # rad, in-plane field angle; None = theta_z_optimal() 
    W: float = 100.0           # nm, junction width  (across)
    L: float = 2000.0          # nm, junction length (along the interfaces)
    tc: float = 1.0            # SC <-> normal coupling, in units of t
    tc_barr: float = 0.00      # probe coupling, in units of t (0 = decoupled, hard walls)
    eta_rel: float = 0.05     # LDOS broadening eta = eta_rel * Delta (Scharf: 0.05)
    # --- transport only; the LDOS fields above stay at Scharf's values (see docstring) ---
    tc_barr_transport: float = 0.20   # probe coupling for kappa/G (0 gives kappa = 0 exactly)
    eta_rel_transport: float = 1e-5   # broadening for kappa/G; must be << the resonance width
    kT_transport: float = 0.0         # meV; temperature for the map's kappa_kT/G_kT (0 = skip)

    # --- derived scales ---
    @property
    def t(self):
        """Hopping energy t = hbar^2 / (2 m a^2), in meV."""
        return (_hbar**2 / (2 * self.m_eff * _m0 * (self.a * 1e-9) ** 2)) * (1e3 / _e)

    @property
    def E_Z(self):
        """Zeeman energy g mu_B B / 2, in meV."""
        return self.g * _muB * self.B / 2

    def B_of_E_Z(self, E_Z_meV):
        """The field (T) that gives a Zeeman energy E_Z (meV): the inverse of E_Z above."""
        return 2 * E_Z_meV / (self.g * _muB)

    @property
    def E_T(self):
        """Thouless energy (pi/2) hbar v_F / W, in meV (v_F from mu_N)."""
        return (np.pi / 2) * 2 * np.sqrt(self.mu_N * self.t) / round(self.W / self.a)

    def theta_z_optimal(self):
        """In-plane field angle with E_Z perpendicular to n_soc (see the module docstring)."""
        return np.arctan2(self.alpha, self.beta) if self.soc_axis == '100' else 0.0

    @property
    def ny(self):
        return max(1, round(self.W / self.a))

    @property
    def nx(self):
        return max(1, round(self.L / self.a))

    def params(self, E_Z=None, phi=np.pi):
        """The corresponding my_functions.Params (lattice units). E_Z in meV overrides g mu_B B/2."""
        t = self.t
        EZ = self.E_Z if E_Z is None else E_Z
        return myf.Params(
            model='rashba', nx=self.nx, ny=self.ny, phi=phi,
            t_n=1.0, t_c=1.0, t_s=1.0,
            mu_n=self.mu_N / t, mu_c=self.mu_N / t, mu_s=self.mu_S / t,
            delta=self.Delta / t,
            alpha=self.alpha / (self.a * t), beta=self.beta / (self.a * t),
            tc_top=self.tc, tc_bot=self.tc, tc_barr=self.tc_barr,
            soc_axis=self.soc_axis,
            Bz_s=0.0,                                   
            theta_z=self.theta_z_optimal() if self.theta_z is None else self.theta_z,
            Bxy=0.0, Bxy_c=EZ / t, Bxy_n=0.0, Bxy_s=0.0,
            eta=self.eta_rel * self.Delta / t)

    def params_transport(self, E_Z=None, phi=np.pi):
        """
        Params for kappa/G: params() with the transport probe coupling and broadening swapped in.
        The LDOS settings (tc_barr = 0, eta = 0.05 Delta) give kappa = 0 identically; see the
        module docstring.
        """
        return replace(self.params(E_Z, phi),
                       tc_barr=self.tc_barr_transport,
                       eta=self.eta_rel_transport * self.Delta / self.t)

    def summary(self):
        t = self.t
        print("-" * 64)
        print(f"  t            = {t:8.3f} meV       (a = {self.a} nm, m = {self.m_eff} m0)")
        for name, val, unit, val_t in (("Delta", self.Delta, "meV", self.Delta / t),
                                       ("mu_S", self.mu_S, "meV", self.mu_S / t),
                                       ("mu_N", self.mu_N, "meV", self.mu_N / t),
                                       ("alpha", self.alpha, "meV nm", self.alpha / (self.a * t)),
                                       ("beta", self.beta, "meV nm", self.beta / (self.a * t)),
                                       ("E_Z", self.E_Z, "meV", self.E_Z / t),
                                       ("E_T", self.E_T, "meV", self.E_T / t)):
            print(f"  {name:<12} = {val:8.3f} {unit:<7} = {val_t:8.4f} t")
        th = self.theta_z_optimal() if self.theta_z is None else self.theta_z
        print(f"  E_Z / E_T    = {self.E_Z / self.E_T:8.2f}        (topological diamond is near 1)")
        print(f"  soc_axis     = {self.soc_axis:>8s}        theta_z = {th/np.pi:.3f} pi"
              f"  (optimum {self.theta_z_optimal()/np.pi:.3f} pi)")
        print(f"  alpha k_F    = {self.alpha / (self.a * t) * np.sqrt(self.mu_N / t) * t:8.3f} meV"
              f"     (should exceed E_Z)")
        print(f"  geometry     = {self.nx} x {self.ny} sites   (L = {self.L} nm, W = {self.W} nm)")
        print(f"  eta          = {self.eta_rel * self.Delta:8.4f} meV")
        print("-" * 64)


# %% --- PARAMETERS ---
alpha   =   14.3
beta    =   7.3
E_z     =   0.46                     # meV; only sets PP.B, the maps sweep B themselves

PP = PhysParams(a=20.0, m_eff=0.038, Delta=0.25, mu_S=1.0, mu_N=0.7, alpha=alpha, beta=beta,
                soc_axis='100', g=10.0, B=0.0, theta_z=None, W=100.0, L=2000.0,  # type: ignore
                tc=1.0, tc_barr=0.0, eta_rel=0.05)                           

PP.B = PP.B_of_E_Z(E_z)           
PP.theta_z = PP.theta_z_optimal() + np.pi      
print(f"Zeeman energy = {E_z:.3f} meV  (B = {PP.B:.2f} T, g = {PP.g:g})")

step_sizes = 101
B_VALS = np.linspace(0.0, 3.5, step_sizes)
PHIS = np.linspace(0.0, 2 * np.pi, step_sizes)
ENERGY_VALS = np.linspace(-1.01 * PP.Delta, 1.01 * PP.Delta, step_sizes)
SAVE = None
N_WORKERS = 1                   
COL_END = [0, 1, 2, 3, 4]
COL_MID_FRAC = 0.5


# %% --- 1. THE MAP (Scharf Figs. 8(e), 8(f)) ---
def _cols(pp):
    mid = int(COL_MID_FRAC * pp.nx)
    return sorted(set(COL_END) | {mid - 1, mid, mid + 1}), [mid - 1, mid, mid + 1]


def _dE(pp):
    """Energy step for the second difference: half the broadening (meV)."""
    return 0.5 * pp.eta_rel * pp.Delta


def _row(pp, phis, cols, i_end, i_mid):
    """
    One field value: returns an array (len(phis), len(KEYS)) with the columns of KEYS.
    The LDOS columns (ldos_*, curv_*) use pp's own eta / tc_barr -- Scharf's values; the transport
    columns (kappa, G, dIdV, andreev, *_kT) use params_transport(), which they must: see the
    module docstring.  kappa_kT / G_kT are NaN unless pp.kT_transport > 0 (they cost 33 energies
    per phase instead of one, so they are off by default).
    The two curvatures are d2(LDOS)/dE2 at E = 0 -- the ENERGY curvature, which is Scharf's
    Fig. 8(e),(f) quantity: negative = zero-energy peak, positive = dip.
    """
    dE = _dE(pp)
    lg = myf.LocalGreen(myf.FourTerminalJunction(pp.params()),
                        np.array([-dE, 0.0, dE]) / pp.t, cols=cols)
    J_t = myf.FourTerminalJunction(pp.params_transport())
    rgf = myf.RGFFourTerminal(J_t, np.array([0.0]))                      # channels at E = 0
    if pp.kT_transport > 0:
        E_th, w_th, w_el = myf.linear_response_nodes_phs(pp.kT_transport / pp.t)
        rgf_T = myf.RGFFourTerminal(J_t, E_th)
    out = np.empty((len(phis), len(KEYS)))
    for j, ph in enumerate(phis):
        e, h = lg.ldos(float(ph))
        D = (e + h).sum(axis=2)
        A = D[1]
        d_end, d_mid = D[:, i_end].sum(axis=1), D[:, i_mid].sum(axis=1)
        c = {k: float(np.asarray(v)[0]) for k, v in rgf.channels_at_phi(float(ph), 'left').items()}
        if pp.kT_transport > 0:
            k_T, g_T = myf.thermal_from_channels_phs(
                rgf_T.channels_at_phi(float(ph), 'left'), w_th, w_el)
        else:
            k_T = g_T = np.nan
        out[j] = [A[i_end].sum(), A[i_mid].sum(),
                  c['ee'] + c['he_cross'],                       # kappa, T -> 0
                  c['ee'] - c['he_cross'],                       # G
                  0.5 * (c['ee'] + c['hh'] + c['eh_cross'] + c['he_cross'])
                  + c['eh_local'] + c['he_local'],               # zero-bias dI_L/dV_L
                  (d_end[0] - 2 * d_end[1] + d_end[2]) / dE**2,  # curv_end
                  (d_mid[0] - 2 * d_mid[1] + d_mid[2]) / dE**2,  # curv_mid
                  c['eh_local'] + c['he_local'],                 # local Andreev conductance
                  k_T, g_T]                                      # kappa, G at pp.kT_transport
    return out


KEYS = ('ldos_end', 'ldos_mid', 'kappa', 'G', 'dIdV', 'curv_end', 'curv_mid',
        'andreev', 'kappa_kT', 'G_kT')


def run_map(pp=PP, phis=PHIS, b_vals=B_VALS, save=SAVE):
    """
    LDOS(E=0) at the junction end and middle, its ENERGY curvature at E = 0, kappa/kappa0,
    G/G0, the zero-bias dI/dV and the local Andreev conductance, on the (phi, B) grid.  Saves
    after every field value and resumes from `save`.  Returns a dict of arrays of shape
    (len(b_vals), len(phis)), plus 'n_unreliable' (failed lead solves; see CLAUDE.md).
    """
    cols, mid_cols = _cols(pp)
    i_end = [cols.index(c) for c in COL_END]
    i_mid = [cols.index(c) for c in mid_cols]
    n_bad0 = myf.leads.N_UNRELIABLE
    if save and os.path.exists(save):
        d = np.load(save)
        missing = [k for k in KEYS if k not in d.files]
        if missing:
            raise ValueError(f"{save} was written before the transport columns existed "
                             f"(missing {', '.join(missing)}); delete it and re-run")
        R = {k: d[k].copy() for k in KEYS}
        done = d['done'].copy()
    else:
        R = {k: np.full((len(b_vals), len(phis)), np.nan) for k in KEYS}
        done = np.zeros(len(b_vals), bool)

    for i, B in enumerate(b_vals):
        if done[i]:
            continue
        out = _row(replace(pp, B=float(B)), phis, cols, i_end, i_mid)
        for n, k in enumerate(KEYS):
            R[k][i] = out[:, n]
        done[i] = True
        if save:
            np.savez(save, done=done, phis=phis, b_vals=b_vals, **R)
        print(f"B = {B:.2f} T  (E_Z = {pp.g * _muB * B / 2:.3f} meV) done   ({i + 1}/{len(b_vals)})",
              flush=True)
    n_bad = myf.leads.N_UNRELIABLE - n_bad0
    if n_bad:
        print(f"WARNING: {n_bad} lead energies failed the Sancho-Rubio check (NaN in the result)")
    R.update(phis=phis, b_vals=b_vals, E_Z=pp.g * _muB * b_vals / 2, n_unreliable=n_bad)
    return R


class _MapRow:
    """Picklable callable for myf.scan: one field value B (tesla) -> array (len(phis), 7)."""
    def __init__(self, pp, phis, cols, i_end, i_mid):
        self.pp, self.phis = pp, np.asarray(phis)
        self.cols, self.i_end, self.i_mid = cols, i_end, i_mid

    def __call__(self, B):
        return _row(replace(self.pp, B=float(B)), self.phis, self.cols, self.i_end, self.i_mid)


def run_map_parallel(pp=PP, phis=PHIS, b_vals=B_VALS, n_workers=8, save="scharf_map_par.npz"):
    """
    Same result as run_map, but the field values run in `n_workers` processes (one BLAS thread
    each), resumable through `save`.
    MUST be called from inside `if __name__ == "__main__":` -- the workers re-import this module.
    """
    cols, mid_cols = _cols(pp)
    f = _MapRow(pp, phis, cols, [cols.index(c) for c in COL_END],
                [cols.index(c) for c in mid_cols])
    res = myf.scan(f, {'B': np.asarray(b_vals)}, n_workers=n_workers, save=save)
    V = res['values']                                       # (n_B, n_phi, len(KEYS))
    R = {k: V[..., n] for n, k in enumerate(KEYS)}
    n_bad = int(np.isnan(V[..., KEYS.index('kappa')]).sum())
    if n_bad:
        print(f"WARNING: {n_bad} (B, phi) points have NaN kappa -- failed lead solves or unfinished")
    R.update(phis=np.asarray(phis), b_vals=np.asarray(b_vals),
             E_Z=pp.g * _muB * np.asarray(b_vals) / 2, n_unreliable=n_bad)
    return R


def plot_map(R, pp=PP):
    """Edge and bulk LDOS at E = 0, normalised, vs (E_Z, phi)."""
    phis, EZ = R['phis'], R['E_Z']

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in ((axs[0], 'ldos_end', "Edge LDOS"), (axs[1], 'ldos_mid', "Bulk LDOS")):
        Z = R[key]
        im = ax.pcolormesh(EZ, phis / np.pi, Z.T / np.nanmax(Z), shading='nearest', cmap='bwr')
        fig.colorbar(im, ax=ax, label=r'$\mathrm{LDOS}$ (norm.)')
        ax.set_xlabel(r"$E_Z$ [meV]", fontsize=10)
        ax.set_ylabel(r"$\phi/\pi$", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.5, 1, 1.5, 2])

    fig.suptitle(f"[{pp.soc_axis}]  L={pp.L:.0f} nm, W={pp.W:.0f} nm, "
                 f"Delta={pp.Delta} meV, alpha={pp.alpha} meV nm, "
                 f"beta={pp.beta} meV nm, mu_N={pp.mu_N} meV, "
                 f"tc_barr={pp.tc_barr};  E_T = {pp.E_T:.2f} meV", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_curvature(R, pp=PP, n_levels=101):
    """Scharf Figs. 8(e),(f): d2(LDOS)/dE2 at E = 0.  Negative (blue) = zero-energy peak."""
    phis, EZ = R['phis'], R['E_Z']
    eta = pp.eta_rel * pp.Delta
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in ((axs[0], 'curv_end', "Edge"), (axs[1], 'curv_mid', "Bulk")):
        Z = R[key].T * eta**2
        v = np.nanpercentile(np.abs(Z), 99.9)
        levels = np.linspace(-v, v, n_levels)       
        im = ax.contourf(EZ, phis / np.pi, Z, levels=levels, cmap='bwr', extend='both')
        fig.colorbar(im, ax=ax, label=r"$\eta^2\,\partial^2\mathrm{D}/\partial E^2$")
        ax.set_xlabel(r"$E_Z$ [meV]", fontsize=10)
        ax.set_ylabel(r"$\phi/\pi$", fontsize=10)
        ax.set_title(title + r":  $\frac{\partial^2\mathrm{D}}{\partial E^2}$ at $E=0$",
                     fontsize=10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.5, 1, 1.5, 2])

    fig.suptitle("negative (blue) = zero-energy peak, positive (red) = dip"
                 f"      [{pp.soc_axis}]  L={pp.L:.0f} nm, W={pp.W:.0f} nm, "
                 f"alpha={pp.alpha}, beta={pp.beta} meV nm;  E_T = {pp.E_T:.2f} meV", fontsize=8)
    plt.tight_layout()
    plt.show()


# --- 2. REAL-SPACE LDOS AT ONE POINT (Scharf Fig. 7) ---
def real_space(phi, B, pp=PP, E=0.0, every=1, show=True, equal_aspect=False):
    """
    LDOS at energy E (meV) for one (phi, B), in Scharf's axis convention:
        x = ACROSS the junction  (W, few sites),  y = ALONG the interfaces (L, many sites)
    Returns (x_nm, y_nm, A) with A of shape (len(y_nm), len(x_nm)), i.e. A[j, i] is the LDOS at
    y = y_nm[j], x = x_nm[i] -- exactly what pcolormesh(x_nm, y_nm, A) expects.  Coordinates are
    site centres measured from the middle of the junction, so x runs over [-W/2, W/2] and y over
    [-L/2, L/2] (Scharf's Fig. 7 convention).
    """
    p = pp.params(E_Z=pp.g * _muB * B / 2)
    rows = list(range(0, p.nx, every))
    A = myf.ldos(p, [E / pp.t], float(phi), rows)[0]   # (n_y, n_x)
    # x in [-W/2, W/2] and y in [-L/2, L/2]
    y_nm = (np.array(rows) + 0.5) * pp.a - pp.L / 2    # along the interfaces
    x_nm = (np.arange(p.ny) + 0.5) * pp.a - pp.W / 2   # across the junction
    if show:
        prof = A.sum(axis=1)                           # sum over x -> one value per y
        fig, axs = plt.subplots(1, 2, figsize=(9, 5.5))
        im = axs[0].pcolormesh(x_nm, y_nm, A, shading='nearest', cmap='cividis')
        fig.colorbar(im, ax=axs[0]).set_label("LDOS")
        axs[0].set_xlabel("x  [nm]")
        axs[0].set_ylabel("y  [nm]")
        axs[0].set_title(rf"LDOS at $E$={E:.2f} meV, $\phi$={phi/np.pi:.2f}$\pi$, "
                         rf"($E_Z$={pp.g*_muB*B/2:.2f} meV)", fontsize=10)
        axs[0].set_xticks([-pp.W / 2, -pp.W / 4, 0, pp.W / 4, pp.W / 2])
        axs[0].set_yticks([-pp.L / 2, -pp.L / 4, 0, pp.L / 4, pp.L / 2])
        if equal_aspect:
            axs[0].set_aspect('equal')
        axs[1].plot(y_nm, prof, 'o-', ms=3)
        axs[1].set_xlabel("y  [nm]")
        axs[1].set_ylabel(r"$\sum_{x} \mathrm{LDOS}$")
        axs[1].set_title(f"end / middle = {prof[0] / prof[len(prof) // 2]:.1f}", fontsize=10)
        plt.tight_layout()
        plt.show()
    return x_nm, y_nm, A


# --- 3. LDOS SPECTRUM AT ONE FIELD (Scharf Figs. 8(a)-(d)) ---
def spectrum(B, pp=PP, E_meV=None, phis=None, show=True):
    """LDOS(E, phi) at the end and in the middle.  Returns (E_meV, phis, end, mid)."""
    E_meV = np.linspace(-1.01*pp.Delta, 1.01* pp.Delta, 61) if E_meV is None else np.asarray(E_meV, float)
    phis = np.linspace(0, 2 * np.pi, 41) if phis is None else np.asarray(phis, float)
    p = pp.params(E_Z=pp.g * _muB * B / 2)
    cols, mid_cols = _cols(pp)
    lg = myf.LocalGreen(myf.FourTerminalJunction(p), E_meV / pp.t, cols=cols)
    end = np.empty((len(phis), len(E_meV)))
    mid = np.empty((len(phis), len(E_meV)))
    for j, ph in enumerate(phis):
        e, h = lg.ldos(float(ph))
        A = (e + h).sum(axis=2)                                   # (n_E, n_cols)
        end[j] = A[:, [cols.index(c) for c in COL_END]].sum(axis=1)
        mid[j] = A[:, [cols.index(c) for c in mid_cols]].sum(axis=1)
    if show:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, Z, ttl in ((axs[0], end, "end"), (axs[1], mid, "middle")):
            im = ax.pcolormesh(E_meV / pp.Delta, phis / np.pi, Z,
                               shading='auto', cmap='cividis')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(r"$E/\Delta$")
            ax.set_ylabel(r"$\phi/\pi$")
            ax.set_title(f"LDOS at {ttl},  $E_z$ = {E_z:.2f} meV", fontsize=10)
        plt.tight_layout()
        plt.show()
    return E_meV, phis, end, mid


# --- 4. TRANSPORT AND REAL-SPACE LDOS AT CHOSEN POINTS ---
def transport_point(pp, phi, E_Z=None, kTs=(0.0,), side='left'):
    """
    kappa/kappa0 and G/G0 at one (E_Z, phi) for every temperature in kTs (meV), plus the channel
    decomposition at E = 0.  Returns (kappas, Gs, channels, n_unreliable); the two arrays have
    shape (len(kTs),).  Uses params_transport(): with the LDOS settings all of this is zero.
    """
    kTs = np.atleast_1d(np.asarray(kTs, float))
    n_bad0 = myf.leads.N_UNRELIABLE
    p = pp.params_transport(E_Z=E_Z)
    kappas, Gs = np.empty(len(kTs)), np.empty(len(kTs))
    for i, kT in enumerate(kTs):
        kappas[i], Gs[i] = myf.linear_response(p, float(phi), kT=float(kT) / pp.t, side=side)
    ch = myf.RGFFourTerminal(myf.FourTerminalJunction(p),
                             np.array([0.0])).channels_at_phi(float(phi), side)
    return (kappas, Gs, {k: float(np.asarray(v)[0]) for k, v in ch.items()},
            myf.leads.N_UNRELIABLE - n_bad0)


def pick_points(R, n=2, target=0.5, require_phi=None):
    """
    The n points of a run_map result whose kappa is closest to `target`, best first, each a dict
    with E_Z, phi, kappa, G, ldos_end, ldos_mid and contrast = ldos_end / ldos_mid.
    require_phi (rad) restricts the search to the nearest phase column.  NaN points are skipped.
    """
    EZ, phis, K = R['E_Z'], R['phis'], R['kappa']
    mask = np.isfinite(K)
    if require_phi is not None:
        keep = np.zeros_like(mask)
        keep[:, int(np.argmin(np.abs(phis - require_phi)))] = True
        mask &= keep
    cost = np.where(mask, np.abs(K - target), np.inf)
    out = []
    for f in np.argsort(cost, axis=None)[:n]:
        i, j = np.unravel_index(f, K.shape)
        if not np.isfinite(cost[i, j]):
            continue
        out.append(dict(E_Z=float(EZ[i]), phi=float(phis[j]), kappa=float(K[i, j]),
                        G=float(R['G'][i, j]), ldos_end=float(R['ldos_end'][i, j]),
                        ldos_mid=float(R['ldos_mid'][i, j]),
                        contrast=float(R['ldos_end'][i, j] / R['ldos_mid'][i, j])))
    return out


def point_report(pp, points, kTs=(0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3), show=True):
    """
    For each point (a dict from pick_points, or a plain (E_Z, phi) pair): print the E = 0 channels
    and kappa/G against temperature, and plot the real-space LDOS there (LDOS settings, so the
    bound states look as they do in Scharf's Fig. 7).  Returns a list of (kappas, Gs, channels).
    """
    out = []
    for pt in points:
        E_Z, phi = (pt['E_Z'], pt['phi']) if isinstance(pt, dict) else (float(pt[0]), float(pt[1]))
        kap, G, ch, n_bad = transport_point(pp, phi, E_Z=E_Z, kTs=kTs)
        print("=" * 70)
        print(f"E_Z = {E_Z:.3f} meV ({E_Z / pp.E_T:.2f} E_T),  phi = {phi / np.pi:.3f} pi")
        print("  channels at E=0:  " + "  ".join(f"{k}={v:.4f}" for k, v in ch.items()))
        print(f"  local Andreev = {ch['eh_local'] + ch['he_local']:.4f} e^2/h"
              f"      failed lead solves: {n_bad}")
        print("     kT [ueV]      kappa/kappa0        G/G0")
        for kT, k, g in zip(np.atleast_1d(np.asarray(kTs, float)), kap, G):
            print(f"   {kT * 1e3:9.3f}    {k:10.4f}   {g:10.4f}")
        if show:
            real_space(phi, pp.B_of_E_Z(E_Z), pp=pp, show=True)
        out.append((kap, G, ch))
    return out


def plot_transport(R, pp=PP):
    """kappa/kappa0 and G/G0 over the (E_Z, phi) map; kappa = 0.5 with G = 0 is the Majorana point."""
    phis, EZ = R['phis'], R['E_Z']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, label in ((axs[0], 'kappa', r"$\kappa/\kappa_0$"), (axs[1], 'G', r"$G/G_0$")):
        Z = R[key]
        im = ax.pcolormesh(EZ, phis / np.pi, Z.T, shading='nearest', cmap='viridis', vmin=0.0,
                           vmax=max(0.5, float(np.nanmax(Z))) if key == 'kappa' else None)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(r"$E_Z$ [meV]", fontsize=10)
        ax.set_ylabel(r"$\phi/\pi$", fontsize=10)
        ax.set_title(label, fontsize=10)
    fig.suptitle(f"transport: tc_barr={pp.tc_barr_transport:g}, eta={pp.eta_rel_transport:g} Delta"
                 f"   (the LDOS panels use tc_barr={pp.tc_barr:g}, eta={pp.eta_rel:g} Delta)",
                 fontsize=8)
    plt.tight_layout()
    plt.show()


# %% --- RUN ---
if __name__ == "__main__":
    PP.summary()
    R = run_map() if N_WORKERS == 1 else run_map_parallel(n_workers=N_WORKERS)
    plot_map(R)
    plot_curvature(R)
    plot_transport(R)
    pts = pick_points(R, n=2, require_phi=np.pi)
    print("\npicked points (kappa closest to 0.5 at phi = pi):")
    for pt in pts:
        print(f"   E_Z={pt['E_Z']:.3f} meV  phi={pt['phi'] / np.pi:.2f} pi  "
              f"kappa={pt['kappa']:.4f}  G={pt['G']:.4f}  end/mid={pt['contrast']:.1f}")
    point_report(PP, pts)
    spectrum(PP.B, E_meV=ENERGY_VALS, phis=PHIS)

# %%
